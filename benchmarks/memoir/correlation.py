"""
TRACE-FB Correlation Gate.

Validates that the ROUGE1-recall attribution signal is a meaningful proxy for
retrieval quality *before* running the full benchmark suite.

Protocol (GT-proxy mode — no LLM required):
  For each LoCoMo sample:
    1. Feed all conversation turns into MemoirStore.
    2. For each QA pair:
       a. Retrieve k=5 memories.
       b. Compute attribution(m, GT_answer) = ROUGE1-recall(m.text, gt_answer)
          for each retrieved memory.
       c. Compute context_recall = 1 if any retrieved memory contains the
          evidence turn's content, else 0.
       d. Record: max_attribution (X) and context_recall (Y) per QA pair.
    3. Compute Pearson r(X, Y) across all QA pairs.

If r < 0.2: attribution signal is too noisy — diagnose before continuing.
If r >= 0.2: signal is valid, proceed with eval_locomo.py.

Note: This is the GT-proxy variant. The full experiment (correlation between
ROUGE1-recall(m, LLM_answer) and QA_F1(LLM_answer, GT)) requires a working
Anthropic API key. Run with --llm-mode when a valid direct API key is available.

Usage:
  python benchmarks/memoir/correlation.py \
    --dataset /path/to/locomo10.json \
    --n-samples 10 \
    --output results/correlation_gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent.parent
_LOCAL_DATA = _REPO / "data"
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # vendored load_dataset.py

from harpo.memoir.store import MemoirStore
from harpo.memoir.trace_fb import rouge1_recall


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", str(text).lower()))


def context_recall(retrieved_texts: list[str], gt_answer: str) -> float:
    """1.0 if any retrieved memory contains at least one GT answer token."""
    gt_tokens = _tokenize(gt_answer)
    if not gt_tokens:
        return 0.0
    for text in retrieved_texts:
        overlap = _tokenize(text) & gt_tokens
        if overlap:
            return 1.0
    return 0.0


def is_numeric_answer(gt_answer: str) -> bool:
    """Pure numeric/count answers — the documented ROUGE1 blind spot."""
    return re.fullmatch(r"\d+(\.\d+)?", str(gt_answer).strip()) is not None


def run_correlation(
    dataset_path: str,
    n_samples: int = 10,
    k_retrieve: int = 5,
    verbose: bool = True,
    attr_mode: str = "both",   # "rouge1" | "cf" | "both"
    cf_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    from load_dataset import load_locomo_dataset

    samples = load_locomo_dataset(dataset_path)
    if n_samples > 0:
        samples = samples[:n_samples]

    use_rouge = attr_mode in ("rouge1", "both")
    use_cf = attr_mode in ("cf", "both")
    cf_attr = None
    if use_cf:
        from harpo.memoir.cf_attributor import CFAttributor
        cf_attr = CFAttributor.get(cf_model)

    all_max_attr: list[float] = []       # rouge1
    all_max_attr_cf: list[float] = []
    all_ctx_recall: list[float] = []
    all_is_numeric: list[bool] = []
    per_sample_stats: list[dict] = []

    for s_idx, sample in enumerate(samples):
        if verbose:
            print(f"\n[{s_idx+1}/{len(samples)}] sample {sample.sample_id} "
                  f"({len(sample.qa)} QAs, {len(sample.conversation.sessions)} sessions)")

        store = MemoirStore()

        # Feed all turns into the store
        n_turns = 0
        for sess_id in sorted(sample.conversation.sessions.keys()):
            sess = sample.conversation.sessions[sess_id]
            for turn in sess.turns:
                text = f"Speaker {turn.speaker} says: {turn.text}"
                store.add(text, metadata={"session": sess_id, "time": sess.date_time,
                                          "dia_id": turn.dia_id})
                n_turns += 1

        if verbose:
            print(f"  Loaded {n_turns} turns into store ({len(store)} memories)")

        # Evaluate QA pairs
        sample_max_attr: list[float] = []
        sample_max_attr_cf: list[float] = []
        sample_ctx_recall: list[float] = []

        for qa in sample.qa:
            gt = str(qa.final_answer)
            retrieved = store.search_with_scores(qa.question, k=k_retrieve)
            retrieved_texts = [m.text for m, _ in retrieved]

            # X: max attribution across retrieved memories
            if use_rouge:
                attributions = [rouge1_recall(text, gt) for text in retrieved_texts]
                sample_max_attr.append(max(attributions) if attributions else 0.0)
            if use_cf:
                cf_scores = cf_attr.score_loo(qa.question, gt, retrieved_texts)
                sample_max_attr_cf.append(max(cf_scores) if cf_scores else 0.0)

            # Y: evidence-based recall when evidence dia_ids exist (exact —
            # and crucially not lexical, so it stays valid on numeric answers
            # where GT-token overlap is structurally 0). Fallback: token overlap.
            evidence = set(getattr(qa, "evidence", None) or [])
            if evidence:
                hit = any((m.metadata or {}).get("dia_id") in evidence
                          for m, _ in retrieved)
                sample_ctx_recall.append(1.0 if hit else 0.0)
            else:
                sample_ctx_recall.append(context_recall(retrieved_texts, gt))
            all_is_numeric.append(is_numeric_answer(gt))

        all_max_attr.extend(sample_max_attr)
        all_max_attr_cf.extend(sample_max_attr_cf)
        all_ctx_recall.extend(sample_ctx_recall)

        stat = {
            "sample_id": sample.sample_id,
            "n_qa": len(sample.qa),
            "mean_ctx_recall": float(np.mean(sample_ctx_recall)),
        }
        if use_rouge:
            stat["mean_max_attr"] = float(np.mean(sample_max_attr))
            stat["pearson_r"] = _pearson(sample_max_attr, sample_ctx_recall)
        if use_cf:
            stat["mean_max_attr_cf"] = float(np.mean(sample_max_attr_cf))
            stat["pearson_r_cf"] = _pearson(sample_max_attr_cf, sample_ctx_recall)
        per_sample_stats.append(stat)

        if verbose:
            parts = [f"ctx_recall={np.mean(sample_ctx_recall):.3f}"]
            if use_rouge:
                parts.append(f"r_rouge={stat['pearson_r']:.3f}")
            if use_cf:
                parts.append(f"r_cf={stat['pearson_r_cf']:.3f}")
            print("  " + "  ".join(parts))

    n_qa_total = len(all_ctx_recall)
    numeric_idx = [i for i, b in enumerate(all_is_numeric) if b]

    def _subset(xs: list[float], idx: list[int]) -> list[float]:
        return [xs[i] for i in idx]

    result = {
        "mode": "gt_proxy",
        "attr_mode": attr_mode,
        "n_samples": len(samples),
        "n_qa_total": n_qa_total,
        "n_qa_numeric": len(numeric_idx),
        "k_retrieve": k_retrieve,
        "mean_context_recall": float(np.mean(all_ctx_recall)),
        "per_sample": per_sample_stats,
    }
    if use_rouge:
        result["pearson_r"] = _pearson(all_max_attr, all_ctx_recall)
        result["mean_max_attribution"] = float(np.mean(all_max_attr))
        result["pearson_r_numeric"] = _pearson(
            _subset(all_max_attr, numeric_idx), _subset(all_ctx_recall, numeric_idx))
        result["gate_passed"] = result["pearson_r"] >= 0.2
    if use_cf:
        result["cf_model"] = cf_model
        result["pearson_r_cf"] = _pearson(all_max_attr_cf, all_ctx_recall)
        result["mean_max_attribution_cf"] = float(np.mean(all_max_attr_cf))
        result["pearson_r_cf_numeric"] = _pearson(
            _subset(all_max_attr_cf, numeric_idx), _subset(all_ctx_recall, numeric_idx))
        result["gate_passed_cf"] = result["pearson_r_cf"] >= 0.2

    print(f"\n{'='*60}")
    print("CORRELATION GATE RESULT")
    print(f"  N QA pairs evaluated : {n_qa_total} (numeric-answer subset: {len(numeric_idx)})")
    print(f"  Mean context recall  : {result['mean_context_recall']:.4f}")
    if use_rouge:
        print(f"  ROUGE1: r={result['pearson_r']:.4f}  "
              f"r_numeric={result['pearson_r_numeric']:.4f}  "
              f"gate {'PASSED ✓' if result['gate_passed'] else 'FAILED ✗'}")
    if use_cf:
        print(f"  CF    : r={result['pearson_r_cf']:.4f}  "
              f"r_numeric={result['pearson_r_cf_numeric']:.4f}  "
              f"gate {'PASSED ✓' if result['gate_passed_cf'] else 'FAILED ✗'}")
    print(f"{'='*60}")

    return result


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    xv = np.array(x, dtype=float)
    yv = np.array(y, dtype=float)
    if xv.std() < 1e-9 or yv.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="TRACE-FB Correlation Gate")
    parser.add_argument(
        "--dataset",
        default=str(_LOCAL_DATA / "locomo10.json"),
        help="Path to locomo10.json",
    )
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--k", type=int, default=5, help="Retrieval k")
    parser.add_argument("--output", default="results/correlation_gate.json",
                        help="Path to write JSON result")
    parser.add_argument("--attr", default="both", choices=["rouge1", "cf", "both"],
                        help="Attribution signal(s) to correlate")
    parser.add_argument("--cf-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Scorer LM for CF attribution")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_correlation(
        dataset_path=args.dataset,
        n_samples=args.n_samples,
        k_retrieve=args.k,
        verbose=not args.quiet,
        attr_mode=args.attr,
        cf_model=args.cf_model,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
