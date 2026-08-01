"""
LongMemEval-S evaluation using MemoryAgentBench (ai-hyz/MemoryAgentBench).

This is the CROSS-BENCHMARK generalizability test for MEMOIR.
Data source: ai-hyz/MemoryAgentBench, split=Accurate_Retrieval, source=longmemeval_s*
  - 5 samples, 60 Q/A pairs each = 300 QA pairs total
  - NOT used in any MEMOIR training (LoCoMo only was used for encoder fine-tuning)

Structure: metadata.haystack_sessions = 60 sessions × 2 turns per session.
  Each session is a list of 2 items (turns). Each turn is itself a list
  (typically [speaker/date, text] or similar). We convert each turn to a
  text chunk and add it as a MemoryItem.

Evaluation protocol (two-retrieval):
  For each QA pair (Q, GT):
    1. Build store from all 120 haystack turns (60 sessions × 2 turns)
    2. retrieve_before  = store.search(Q, k=5)  → ctx_recall_before
    3. store.feedback_answer(Q, GT, k=20)        → TRACE-FB updates w_imp
    4. retrieve_after   = store.search(Q, k=5)  → ctx_recall_after
    5. Δctx_recall = ctx_recall_after - ctx_recall_before
    6. Reset w_imp=0 for next QA pair (independent pairs)

Systems:
  memoir_no_trace  — skip step 3, report ctx_recall from step 2
  oracle_trace     — use GT answer as TRACE-FB feedback (upper bound)

Why two-retrieval not multi-session:
  MemoryAgentBench batches 60 Qs per sample sharing the same sessions.
  The two-retrieval protocol directly measures TRACE-FB's per-question
  retrieval shift — it is the cleanest demonstration that does not need
  cross-session ordering.

Usage:
  python benchmarks/memoir/eval_longmemeval_hf.py \\
    --systems memoir_no_trace,oracle_trace \\
    --n-samples 5 \\
    --output results/longmemeval_hf.json
"""

from __future__ import annotations

import argparse
import json
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from harpo.memoir.store import MemoirStore


# ── text utils ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", str(text).lower()))


def ctx_recall(retrieved_texts: list[str], gt: str) -> float:
    gt_toks = _tokenize(gt)
    if not gt_toks:
        return 0.0
    for t in retrieved_texts:
        if _tokenize(t) & gt_toks:
            return 1.0
    return 0.0


# ── session-to-text converter ─────────────────────────────────────────────────

def _parse_turn_str(s: str) -> list[dict]:
    """
    Parse a Python-repr or JSON string like:
      '[{"role": "user", "content": "...", "has_answer": false}, ...]'
    Returns list of turn dicts.
    """
    s = s.strip()
    if not s or s in ("[]", ""):
        return []
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        # Fall back to ast.literal_eval for Python repr strings
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except Exception:
        pass
    return []


def _turn_to_text(turn: dict) -> str:
    """Convert a turn dict {role, content, has_answer} to a text string."""
    role    = str(turn.get("role", "")).capitalize()
    content = str(turn.get("content", "")).strip()
    if role and content:
        return f"{role}: {content}"
    return content


def sessions_to_chunks(sessions: list, max_chars: int = 3000) -> list[str]:
    """
    Convert haystack_sessions to flat text chunks for MEMOIR ingestion.

    Observed LongMemEval-S structure (MemoryAgentBench):
      sessions[i]    = list of 2 sub-session strings
      sessions[i][j] = Python repr string of list[{role, content, has_answer}]

    We parse all sub-session strings, extract 'content' per turn,
    and produce one chunk per turn. Long turns are sentence-split.

    All 60 sessions go into ONE shared store (full haystack per sample).
    """
    chunks: list[str] = []
    for session in sessions:
        if isinstance(session, (list, tuple)):
            for sub in session:
                if isinstance(sub, str):
                    turns = _parse_turn_str(sub)
                    for t in turns:
                        if isinstance(t, dict):
                            text = _turn_to_text(t)
                        else:
                            text = str(t).strip()
                        if not text:
                            continue
                        if len(text) <= max_chars:
                            chunks.append(text)
                        else:
                            sents = re.split(r"(?<=[.!?])\s+", text)
                            buf = ""
                            for sent in sents:
                                if len(buf) + len(sent) + 1 > max_chars and buf:
                                    chunks.append(buf)
                                    buf = sent
                                else:
                                    buf = (buf + " " + sent).strip() if buf else sent
                            if buf:
                                chunks.append(buf)
                elif isinstance(sub, (list, tuple)):
                    for t in sub:
                        if isinstance(t, dict):
                            text = _turn_to_text(t)
                            if text:
                                chunks.append(text[:max_chars])
                        else:
                            text = str(t).strip()
                            if text:
                                chunks.append(text[:max_chars])
                elif isinstance(sub, dict):
                    text = _turn_to_text(sub)
                    if text:
                        chunks.append(text[:max_chars])
        elif isinstance(session, str):
            turns = _parse_turn_str(session)
            for t in turns:
                text = _turn_to_text(t) if isinstance(t, dict) else str(t).strip()
                if text:
                    chunks.append(text[:max_chars])
    return chunks


# ── system factory ────────────────────────────────────────────────────────────

def make_store(sys_name: str, model_name: str = "all-MiniLM-L6-v2") -> MemoirStore:
    store = MemoirStore(model_name=model_name)
    if sys_name == "cosine":
        store._ablation = {"flat_w": True, "no_bm25": True, "no_mmr": True}
    else:
        store._ablation = {}
    return store


# ── per-sample evaluation ─────────────────────────────────────────────────────

def _reset_w_imp(store: MemoirStore) -> None:
    """Zero out w_imp on all memory items — restores store to pre-TRACE-FB state."""
    for item in store._items.values():
        item.w.w_imp = 0.0


def evaluate_sample(
    sessions: list,
    questions: list[str],
    answers: list[list[str]],
    question_types: list[str],
    sys_name: str,
    k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """
    Evaluate all QA pairs from one sample (two-retrieval protocol).

    Build the store ONCE from haystack_sessions, then for each QA pair:
      memoir_no_trace : retrieve once, no feedback — cr_after = cr_before
      oracle_trace    : retrieve before, apply TRACE-FB (oracle GT), retrieve after,
                        then reset w_imp=0 so each pair is independent

    Using haystack_sessions (60 sess × 2 turns = 120 chunks) instead of the
    raw 1.6M-char context field — cleaner boundaries, faster embedding.
    """
    chunks = sessions_to_chunks(sessions)
    results: list[dict] = []

    # One store shared across all QA pairs — w_imp reset between pairs
    store    = make_store("memoir_no_trace", model_name)
    ablation = getattr(store, "_ablation", {})
    for chunk in chunks:
        if chunk.strip():
            store.add(chunk)

    for q_idx, (question, ans_list, q_type) in enumerate(
        zip(questions, answers, question_types)
    ):
        gt = ans_list[0] if isinstance(ans_list, list) and ans_list else str(ans_list)
        gt = gt.strip()
        if not gt:
            continue

        # Retrieval BEFORE any TRACE-FB
        ret_before   = store.search(question, k=k, **ablation)
        texts_before = [m.text for m in ret_before]
        cr_before    = ctx_recall(texts_before, gt)

        if sys_name == "oracle_trace":
            # Apply TRACE-FB with oracle answer
            store.feedback_answer(question, gt, k=k * 4, **ablation)

            # Retrieval AFTER TRACE-FB
            ret_after   = store.search(question, k=k, **ablation)
            texts_after = [m.text for m in ret_after]
            cr_after    = ctx_recall(texts_after, gt)

            # Reset w_imp so next QA pair starts fresh
            _reset_w_imp(store)
        else:
            cr_after = cr_before  # no TRACE-FB for memoir_no_trace

        results.append({
            "q_type":    q_type,
            "cr_before": cr_before,
            "cr_after":  cr_after,
            "delta_cr":  cr_after - cr_before,
        })

    return results


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(all_results: list[dict]) -> dict:
    if not all_results:
        return {}
    n = len(all_results)

    def mean(key: str) -> float:
        vals = [r[key] for r in all_results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    by_type: dict[str, list] = defaultdict(list)
    for r in all_results:
        by_type[r["q_type"]].append(r)

    type_agg = {}
    for qt, items in sorted(by_type.items()):
        type_agg[qt] = {
            "n":         len(items),
            "cr_before": float(np.mean([x["cr_before"] for x in items])),
            "cr_after":  float(np.mean([x["cr_after"]  for x in items])),
            "delta_cr":  float(np.mean([x["delta_cr"]  for x in items])),
        }

    return {
        "n":         n,
        "cr_before": mean("cr_before"),
        "cr_after":  mean("cr_after"),
        "delta_cr":  mean("delta_cr"),
        "by_type":   type_agg,
    }


# ── data loader ───────────────────────────────────────────────────────────────

def load_longmemeval_hf(n_samples: int = 0) -> list[dict]:
    """
    Stream LongMemEval-S samples from ai-hyz/MemoryAgentBench.

    Returns list of dicts with keys: sessions, questions, answers, question_types.
    'sessions' = metadata.haystack_sessions (60 session-lists per sample).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "ai-hyz/MemoryAgentBench",
        split="Accurate_Retrieval",
        streaming=True,
    )

    samples = []
    for sample in ds:
        meta = sample.get("metadata", {})
        if not isinstance(meta, dict):
            continue
        src = meta.get("source", "")
        if "longmemeval" not in src:
            continue

        sessions = meta.get("haystack_sessions", [])
        qtypes   = meta.get("question_types", [])

        samples.append({
            "sessions":       sessions,
            "questions":      sample.get("questions", []),
            "answers":        sample.get("answers",   []),
            "question_types": qtypes if isinstance(qtypes, list) else [],
        })

        if n_samples > 0 and len(samples) >= n_samples:
            break

    print(f"Loaded {len(samples)} LongMemEval-S samples from MemoryAgentBench")
    return samples


# ── main eval ─────────────────────────────────────────────────────────────────

_ALL_SYSTEMS = ["memoir_no_trace", "oracle_trace"]


def run_eval(
    systems: list[str],
    n_samples: int = 5,
    k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    verbose: bool = True,
) -> dict[str, dict]:
    samples = load_longmemeval_hf(n_samples)
    if not samples:
        raise RuntimeError("No LongMemEval-S samples found in MemoryAgentBench")

    total_qs = sum(len(s["questions"]) for s in samples)
    print(f"  {len(samples)} samples, {total_qs} total QA pairs")

    all_results: dict[str, dict] = {}

    for sys_name in systems:
        print(f"\n{'─'*60}\nSystem: {sys_name}\n{'─'*60}")
        per_qa: list[dict] = []

        for s_idx, sample in enumerate(samples):
            n_qs = len(sample["questions"])
            if verbose:
                n_sess = len(sample["sessions"])
                print(f"  [sample {s_idx+1}/{len(samples)}] {n_qs} Qs, "
                      f"{n_sess} sessions, chunking...")

            q_types = sample["question_types"]
            if len(q_types) < len(sample["questions"]):
                q_types = q_types + ["unknown"] * (len(sample["questions"]) - len(q_types))

            sample_results = evaluate_sample(
                sessions=sample["sessions"],
                questions=sample["questions"],
                answers=sample["answers"],
                question_types=q_types,
                sys_name=sys_name,
                k=k,
                model_name=model_name,
            )
            per_qa.extend(sample_results)
            if verbose:
                cr = np.mean([r["cr_before"] for r in sample_results])
                dc = np.mean([r["delta_cr"] for r in sample_results])
                print(f"    done: cr_before={cr:.4f}, mean_Δ={dc:+.4f}")

        agg = aggregate(per_qa)
        all_results[sys_name] = agg

        print(f"\n  {sys_name}: ctx_recall {agg['cr_before']:.4f} → {agg['cr_after']:.4f} "
              f"(Δ={agg['delta_cr']:+.4f})  n={agg['n']}")
        for qt, m in agg.get("by_type", {}).items():
            print(f"    {qt:<35}: {m['cr_before']:.4f}→{m['cr_after']:.4f} "
                  f"Δ={m['delta_cr']:+.4f} n={m['n']}")

    return all_results


def print_table(results: dict) -> None:
    print("\n\n## LongMemEval-S (MemoryAgentBench) Results — Cross-Benchmark\n")
    print(f"{'System':<22} {'cr_before':>10} {'cr_after':>10} {'Δcr':>8} {'n':>6}")
    print("─" * 60)
    for sys_name, r in results.items():
        print(f"{sys_name:<22} {r['cr_before']:>10.4f} {r['cr_after']:>10.4f} "
              f"{r['delta_cr']:>+8.4f} {r['n']:>6}")

    print("\n## Δctx_recall by question_type (oracle_trace)\n")
    if "oracle_trace" in results:
        for qt, m in results["oracle_trace"].get("by_type", {}).items():
            bar = "▓" * int(abs(m["delta_cr"]) * 50)
            neg = "-" if m["delta_cr"] < 0 else ""
            print(f"  {qt:<35}: {m['delta_cr']:>+.4f}  {neg}{bar}  (n={m['n']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval-S cross-benchmark eval (MemoryAgentBench)")
    parser.add_argument("--systems",   default="memoir_no_trace,oracle_trace")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of MemoryAgentBench samples (max 5)")
    parser.add_argument("--k",         type=int, default=5)
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2",
                        help="Embedding model: 'all-MiniLM-L6-v2' or 'models/memoir-encoder-v1'")
    parser.add_argument("--output",    default="results/longmemeval_hf.json")
    parser.add_argument("--quiet",     action="store_true")
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    results = run_eval(
        systems=systems,
        n_samples=args.n_samples,
        k=args.k,
        model_name=args.model_name,
        verbose=not args.quiet,
    )

    print_table(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
