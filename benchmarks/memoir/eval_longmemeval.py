"""
LongMemEval-S Evaluation Harness (PRIMARY benchmark).

Dataset: xiaowu0162/longmemeval-cleaned (500 samples)
  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
  (Already downloaded to data/longmemeval_s.json)

Five question types:
  single-session-user / single-session-preference / single-session-assistant
                              → information_extraction
  multi-session               → multi_session_reasoning
  temporal-reasoning          → temporal_reasoning
  knowledge-update            → knowledge_update

Protocol (two-retrieval TRACE-FB evaluation):
  For each sample:
    1. Add all haystack sessions to memory store.
    2. Retrieve k memories for question → record ctx_recall_before.
    3. Run TRACE-FB with feedback answer (GT, LLM, or none).
    4. Retrieve k memories again → record ctx_recall_after.

  memoir_no_trace  : skip step 3 (ctx_recall_before only)
  memoir_trace     : step 3 with LLM-generated answer (needs API key)
  oracle_trace     : step 3 with GT answer (upper bound)
  cf_trace         : step 3 with GT answer + TRACE-CF counterfactual
                     attribution (local Qwen2.5-0.5B LOO likelihood shift)
                     instead of ROUGE1 — isolates the attribution signal

Why two-retrieval: shows explicitly how much TRACE-FB shifts retrieval quality.
Why LongMemEval not LoCoMo: answers appear verbatim in turns → ROUGE1 attribution > 0.

Usage:
  # Retrieval-only (no LLM, oracle TRACE-FB)
  python benchmarks/memoir/eval_longmemeval.py \\
    --data data/longmemeval_s.json \\
    --systems memoir_no_trace,oracle_trace \\
    --output results/longmemeval_s.json

  # Full QA eval (needs direct API key)
  ANTHROPIC_API_KEY=sk-ant-api03-... \\
  python benchmarks/memoir/eval_longmemeval.py \\
    --data data/longmemeval_s.json \\
    --systems cosine,memoir_no_trace,memoir_trace,oracle_trace
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from harpo.memoir.store import MemoirStore
from harpo.memoir.trace_fb import rouge1_recall
from eval_locomo import token_f1, context_recall_at_k, best_token_f1
from metrics import calculate_metrics


# ── question type → ability ───────────────────────────────────────────────────

_TYPE_TO_ABILITY = {
    "single-session-user":       "information_extraction",
    "single-session-preference": "information_extraction",
    "single-session-assistant":  "information_extraction",
    "multi-session":             "multi_session_reasoning",
    "temporal-reasoning":        "temporal_reasoning",
    "knowledge-update":          "knowledge_update",
}


# ── data loader ───────────────────────────────────────────────────────────────

def _parse_longmemeval_datetime(date_str: Optional[str]) -> Optional[float]:
    """
    Parse LongMemEval's 'YYYY/MM/DD (Day) HH:MM' strings (e.g.
    '2023/05/20 (Sat) 02:21') to an epoch float. Haystack sessions in a sample
    span real months/years -- without this, every session gets the same
    wall-clock time.time() at ingestion (all added within the same fast eval
    process), making w_rec structurally degenerate for every question, exactly
    the failure mode schema.py's live_w_rec docstring and eval_locomo.py's
    _parse_locomo_datetime already guard against for LoCoMo. Returns None
    (-> callers fall back to wall-clock) on parse failure.
    """
    if not date_str:
        return None
    try:
        import datetime as _dt
        # strip the "(Sat)" weekday parenthetical -- not needed for parsing
        cleaned = re.sub(r"\s*\([A-Za-z]+\)", "", date_str).strip()
        return _dt.datetime.strptime(cleaned, "%Y/%m/%d %H:%M").timestamp()
    except ValueError:
        return None


def load_longmemeval_s(path: str) -> list[dict]:
    """
    Load LongMemEval-S. Prefers .jsonl (one sample per line, memory-efficient).
    Falls back to .json (loads entire file — needs ~2GB RAM for 500 samples).

    Convert to JSONL first if needed:
      jq -c '.[]' data/longmemeval_s.json > data/longmemeval_s.jsonl
    """
    path = Path(path)
    # Auto-try .jsonl sibling if .json given and .jsonl exists
    if path.suffix == ".json" and path.with_suffix(".jsonl").exists():
        path = path.with_suffix(".jsonl")

    if not path.exists():
        raise FileNotFoundError(
            f"LongMemEval-S not found at {path}\n"
            "Download:\n"
            "  wget https://huggingface.co/datasets/xiaowu0162/"
            "longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json\n"
            "  jq -c '.[]' longmemeval_s_cleaned.json > data/longmemeval_s.jsonl"
        )

    if path.suffix == ".jsonl":
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"Loaded {len(samples)} LongMemEval-S samples (JSONL)")
        return samples

    # Large JSON — warn and load (requires ~2GB RAM)
    print(f"Warning: loading {path.stat().st_size//1024//1024}MB JSON into memory. "
          f"Convert to JSONL to avoid OOM:\n  jq -c '.[]' {path} > {path.with_suffix('.jsonl')}")
    with open(path) as f:
        data = json.load(f)
    samples = data if isinstance(data, list) else data.get("data", [])
    print(f"Loaded {len(samples)} LongMemEval-S samples")
    return samples


def sample_ability(sample: dict) -> str:
    return _TYPE_TO_ABILITY.get(sample.get("question_type", ""), "other")


# ── metrics ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", str(text).lower()))


def ctx_recall(texts: list[str], gt: str) -> float:
    gt_toks = _tokenize(gt)
    if not gt_toks:
        return 0.0
    for t in texts:
        if _tokenize(t) & gt_toks:
            return 1.0
    return 0.0


def best_rouge1(texts: list[str], gt: str) -> float:
    if not texts:
        return 0.0
    return max(rouge1_recall(t, gt) for t in texts)


# ── system factory ────────────────────────────────────────────────────────────

def make_store(sys_name: str) -> MemoirStore:
    if sys_name == "cf_trace":
        store = MemoirStore(use_cf_attr=True)
    elif sys_name == "memoir_ucb":
        # memoir_trace + UCB exploration bonus on w_imp (SimUtil-UCB adapted,
        # RetroAgent arXiv:2603.08561) -- same component tested on LoCoMo
        # (docs/COMBINED_PAPER_PLAN.md, 2026-07-22), now cross-benchmarked here.
        store = MemoirStore(use_ucb=True)
    elif sys_name == "memoir_reliability":
        # memoir_trace + RELIABILITY: Laplace-smoothed eta() soft bonus
        # (MSCE-inspired, arXiv:2607.16621 sec 4.3) -- oracle-feedback-only,
        # decoupled from attribution path via reliability_correct.
        store = MemoirStore(use_reliability=True)
    else:
        store = MemoirStore()
    if sys_name == "cosine":
        store._ablation = {"flat_w": True, "no_bm25": True, "no_mmr": True}
    elif sys_name == "no_trace_no_rec":
        # Diagnostic: isolate w_rec's contribution now that it computes real
        # timestamps (docs/COMBINED_PAPER_PLAN.md, 2026-07-21 date-fix entry) --
        # lambda_rec=0.1/hour (~7h half-life) is LoCoMo-scale; LongMemEval
        # sessions span weeks/months, where w_rec collapses to ~0 for
        # everything except same-day sessions, likely biasing ranking toward
        # "happens to be recent" over "actually relevant but older".
        store._ablation = {"no_w_rec": True}
    elif sys_name == "no_trace_flatw":
        # Isolate BM25+MMR's net effect: kills w_rel/w_rec/w_imp together
        # (like cosine) but leaves BM25 and MMR active (unlike cosine, which
        # disables those too) -- no_w_rec alone (0.8500) didn't close the gap
        # to cosine (0.8640), so testing whether BM25/MMR is the remaining
        # culprit rather than w_rel.
        store._ablation = {"flat_w": True}
    elif sys_name == "no_trace_no_bm25_mmr":
        # Complementary isolation: keep w_rel/w_rec active, kill BM25+MMR.
        store._ablation = {"no_bm25": True, "no_mmr": True}
    elif sys_name == "no_trace_no_bm25":
        store._ablation = {"no_bm25": True}
    elif sys_name == "no_trace_no_mmr":
        store._ablation = {"no_mmr": True}
    elif sys_name == "oracle_trace_no_bm25":
        # Combine the fix (BM25 was the isolated culprit, confirmed via
        # ablation) with TRACE-FB's actual uplift mechanism (oracle feedback)
        # to test for a genuine win, not just parity with cosine.
        store._ablation = {"no_bm25": True}
    else:
        store._ablation = {}
    return store


# ── per-sample evaluation ─────────────────────────────────────────────────────

def evaluate_sample(
    sample: dict,
    sys_name: str,
    k: int = 5,
    llm_client=None,
) -> dict:
    """
    Online TRACE-FB evaluation for one sample.

    Sessions are ingested one by one. After each answer session, TRACE-FB
    runs with the GT answer to update w_imp. Final retrieval is measured
    after all sessions are ingested.

    This matches real agent use: importance accumulates across sessions.
    memoir_no_trace and cosine skip the TRACE-FB step entirely.

    Returns dict with: ability, ctx_recall_before, ctx_recall_after,
      delta_ctx_recall, best_rouge1_before, best_rouge1_after
    """
    gt       = str(sample.get("answer", ""))
    question = sample.get("question", "")
    ability  = sample_ability(sample)
    hs       = sample.get("haystack_sessions", [])
    hids     = sample.get("haystack_session_ids", [])
    hdates   = sample.get("haystack_dates", [])
    ans_ids  = set(sample.get("answer_session_ids", []))

    # Dataset-native timestamps (see _parse_longmemeval_datetime) -- q_epoch is
    # used as `now` for every search()/feedback_answer() call below so w_rec
    # sees real relative ages instead of collapsing to ~0 (or, if only
    # created_at were fixed without also fixing `now`, underflowing to ~0 from
    # the other direction: age_hours computed against wall-clock "now" in a
    # different year is astronomically large for every item alike).
    q_epoch = _parse_longmemeval_datetime(sample.get("question_date"))

    store   = make_store(sys_name)
    ablation = getattr(store, "_ablation", {})

    def _ev_recall(items) -> float:
        """Evidence recall: any retrieved memory from an answer session."""
        if not ans_ids:
            return 0.0
        return 1.0 if any((m.metadata or {}).get("sid") in ans_ids
                          for m in items) else 0.0

    # ── Online ingestion: add sessions one by one, TRACE-FB after each answer session ──
    for i, (sess_id, session) in enumerate(zip(hids, hs)):
        is_ans_sess = sess_id in ans_ids

        # Batch-embed all turns in this session at once (10-50× faster than per-turn add())
        turn_texts = [
            f"{turn.get('role', 'user').capitalize()}: {turn.get('content', '')}"
            for turn in session
        ]
        sess_epoch = (_parse_longmemeval_datetime(hdates[i])
                     if i < len(hdates) else None)
        created_ats = [sess_epoch] * len(turn_texts) if sess_epoch is not None else None
        # sid metadata → evidence-based recall (retrieved memory from an answer
        # session). The token-overlap ctx_recall is structurally blind on
        # numeric/count answers whose token appears in no memory text.
        store.add_many(turn_texts, metadata={"sid": sess_id}, created_ats=created_ats)

        # Run TRACE-FB after each answer session (oracle: GT, LLM: generated, else skip)
        if is_ans_sess and sys_name not in (
                "memoir_no_trace", "cosine", "no_trace_no_rec",
                "no_trace_flatw", "no_trace_no_bm25_mmr",
                "no_trace_no_bm25", "no_trace_no_mmr"):
            fb_answer = ""
            if sys_name in ("oracle_trace", "cf_trace", "oracle_trace_no_bm25",
                            "memoir_ucb", "memoir_reliability"):
                fb_answer = gt
            elif sys_name == "memoir_trace" and llm_client is not None:
                retrieved_now = store.search(question, k=k, now=q_epoch, **ablation)
                from llm_client import format_context
                fb_answer = llm_client.answer(question, format_context(retrieved_now))

            if fb_answer:
                # RELIABILITY (n_pass/n_trial, schema.py MemoryItem.eta()): a
                # per-query oracle-style correctness probe, decoupled from
                # attribution (reliability_correct param, store.py) so it
                # doesn't switch memoir_ucb off ROUGE1/GT attribution -- same
                # oracle_correct=rec5>0.5 pattern as eval_locomo.py.
                reliability_correct = None
                if sys_name in ("oracle_trace", "cf_trace", "oracle_trace_no_bm25",
                                "memoir_ucb", "memoir_reliability"):
                    probe = store.search(question, k=k, now=q_epoch, **ablation)
                    reliability_correct = _ev_recall(probe) > 0.5
                # Wider scan for feedback so memories just below top-k can accumulate w_imp
                store.feedback_answer(question, fb_answer, k=k * 4, now=q_epoch,
                                      reliability_correct=reliability_correct, **ablation)

    # ── Measure retrieval BEFORE TRACE-FB (cosine/no_trace equivalent) ─────────────
    # We compare memoir_no_trace (no TRACE-FB ever) vs oracle_trace (accumulated w_imp)
    # The "before" measurement is the no_trace retrieval quality (using cosine only).
    # For oracle_trace we report final retrieval quality.
    retrieved_final = store.search(question, k=k, now=q_epoch, **ablation)
    texts_final     = [m.text for m in retrieved_final]
    cr_final  = ctx_recall(texts_final, gt)
    r1_final  = best_rouge1(texts_final, gt)
    ev_final  = _ev_recall(retrieved_final)

    # For the "before" comparison: rebuild a fresh cosine-only store (no TRACE-FB)
    # Only needed for oracle_trace to compute Δ
    cr_before = cr_final  # default: same (for no-feedback systems)
    r1_before = r1_final
    ev_before = ev_final

    if sys_name in ("oracle_trace", "memoir_trace", "cf_trace", "oracle_trace_no_bm25",
                    "memoir_ucb", "memoir_reliability"):
        fresh = make_store(
            "no_trace_no_bm25" if sys_name == "oracle_trace_no_bm25" else "memoir_no_trace")
        fresh_ablation = getattr(fresh, "_ablation", {})
        for f_i, (f_sid, f_session) in enumerate(zip(hids, hs)):
            f_epoch = (_parse_longmemeval_datetime(hdates[f_i])
                      if f_i < len(hdates) else None)
            f_texts = [f"{turn.get('role', 'user').capitalize()}: {turn.get('content', '')}"
                      for turn in f_session]
            fresh.add_many(
                f_texts, metadata={"sid": f_sid},
                created_ats=[f_epoch] * len(f_texts) if f_epoch is not None else None,
            )
        baseline = fresh.search(question, k=k, now=q_epoch, **fresh_ablation)
        base_texts = [m.text for m in baseline]
        cr_before = ctx_recall(base_texts, gt)
        r1_before = best_rouge1(base_texts, gt)
        ev_before = _ev_recall(baseline)

    f1_score = None

    return {
        "sample_id":          sample.get("question_id", sample.get("question", question)),
        "question":           question,
        "ability":            ability,
        "ctx_recall_before":  cr_before,
        "ctx_recall_after":   cr_final,
        "delta_ctx_recall":   cr_final - cr_before,
        "ev_recall_before":   ev_before,
        "ev_recall_after":    ev_final,
        "delta_ev_recall":    ev_final - ev_before,
        "best_rouge1_before": r1_before,
        "best_rouge1_after":  r1_final,
        "delta_rouge1":       r1_final - r1_before,
        "f1":                 f1_score,
    }


# ── aggregate ─────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    n = len(results)

    def _mean(key: str) -> float:
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    by_ability: dict[str, list] = defaultdict(list)
    for r in results:
        by_ability[r["ability"]].append(r)

    ability_agg = {}
    for ab, items in sorted(by_ability.items()):
        ability_agg[ab] = {
            "n": len(items),
            "ctx_recall_before": float(np.mean([x["ctx_recall_before"] for x in items])),
            "ctx_recall_after":  float(np.mean([x["ctx_recall_after"]  for x in items])),
            "delta_ctx_recall":  float(np.mean([x["delta_ctx_recall"]  for x in items])),
            "ev_recall_before":  float(np.mean([x.get("ev_recall_before", 0.0) for x in items])),
            "ev_recall_after":   float(np.mean([x.get("ev_recall_after", 0.0)  for x in items])),
            "delta_ev_recall":   float(np.mean([x.get("delta_ev_recall", 0.0)  for x in items])),
            "best_rouge1_before": float(np.mean([x["best_rouge1_before"] for x in items])),
            "best_rouge1_after":  float(np.mean([x["best_rouge1_after"]  for x in items])),
        }

    return {
        "n":                 n,
        "ctx_recall_before": _mean("ctx_recall_before"),
        "ctx_recall_after":  _mean("ctx_recall_after"),
        "delta_ctx_recall":  _mean("delta_ctx_recall"),
        "ev_recall_before":  _mean("ev_recall_before"),
        "ev_recall_after":   _mean("ev_recall_after"),
        "delta_ev_recall":   _mean("delta_ev_recall"),
        "best_rouge1_before": _mean("best_rouge1_before"),
        "best_rouge1_after":  _mean("best_rouge1_after"),
        "delta_rouge1":      _mean("delta_rouge1"),
        "f1":                _mean("f1"),
        "by_ability":        ability_agg,
    }


# ── main ──────────────────────────────────────────────────────────────────────

_ALL_SYSTEMS = ["cosine", "memoir_no_trace", "memoir_trace", "oracle_trace", "cf_trace",
               "memoir_ucb", "memoir_reliability"]


def _stream_samples(data_path: str, n_samples: int, offset: int = 0) -> list[dict]:
    """
    Memory-efficient loader: reads JSONL line by line.
    offset: skip this many samples from the start (useful for targeting specific question types).
    """
    path = Path(data_path)
    if path.suffix == ".json" and path.with_suffix(".jsonl").exists():
        path = path.with_suffix(".jsonl")

    samples = []
    if path.suffix == ".jsonl":
        with open(path) as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                if n_samples > 0 and len(samples) >= n_samples:
                    break
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    full = load_longmemeval_s(str(data_path))
    full = full[offset:]
    return full[:n_samples] if n_samples > 0 else full


def run_eval(
    data_path: str,
    systems: list[str],
    n_samples: int = 0,
    k: int = 5,
    verbose: bool = True,
    llm_client=None,
    offset: int = 0,
    dump_perqa: bool = False,
    output: str = "results/longmemeval_s.json",
) -> dict[str, dict]:
    samples = _stream_samples(data_path, n_samples, offset=offset)
    total   = len(samples)
    print(f"Evaluating {total} samples on {len(systems)} systems")

    all_results: dict[str, dict] = {}

    for sys_name in systems:
        print(f"\n{'─'*60}\nSystem: {sys_name}  ({total} samples)\n{'─'*60}")
        per_sample = []

        for s_idx, sample in enumerate(samples):
            if verbose and (s_idx % 50 == 0 or s_idx < 3):
                print(f"  [{s_idx+1}/{total}] {sample.get('question_type')} "
                      f"Q: {sample.get('question','')[:60]}")
            try:
                r = evaluate_sample(sample, sys_name, k=k, llm_client=llm_client)
                per_sample.append(r)
            except Exception as e:
                print(f"  [ERROR] sample {s_idx}: {e}")
                continue

        agg = aggregate(per_sample)
        all_results[sys_name] = agg

        if dump_perqa:
            perqa_dir = Path(output).parent / "perqa"
            perqa_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(output).stem
            with open(perqa_dir / f"{stem}_{sys_name}.jsonl", "w") as f:
                for row in per_sample:
                    f.write(json.dumps(row, default=str) + "\n")

        print(f"\n  {sys_name}: "
              f"ctx_recall {agg['ctx_recall_before']:.4f} → {agg['ctx_recall_after']:.4f} "
              f"(Δ={agg['delta_ctx_recall']:+.4f}) "
              f"ev_recall {agg['ev_recall_before']:.4f} → {agg['ev_recall_after']:.4f} "
              f"(Δ={agg['delta_ev_recall']:+.4f}) "
              f"rouge1 {agg['best_rouge1_before']:.4f} → {agg['best_rouge1_after']:.4f}")
        for ab, m in agg.get("by_ability", {}).items():
            print(f"    {ab:<30}: ctx {m['ctx_recall_before']:.4f}→{m['ctx_recall_after']:.4f} "
                  f"Δ={m['delta_ctx_recall']:+.4f} "
                  f"ev {m['ev_recall_before']:.4f}→{m['ev_recall_after']:.4f} "
                  f"Δev={m['delta_ev_recall']:+.4f} n={m['n']}")

    return all_results


def print_table(results: dict) -> None:
    print("\n\n## LongMemEval-S Results\n")
    print(f"{'System':<22} {'cr_before':>10} {'cr_after':>10} {'Δcr':>8} "
          f"{'r1_before':>10} {'r1_after':>10} {'Δr1':>8}")
    print("─" * 82)
    for sys_name, r in results.items():
        print(f"{sys_name:<22} "
              f"{r['ctx_recall_before']:>10.4f} {r['ctx_recall_after']:>10.4f} "
              f"{r['delta_ctx_recall']:>+8.4f} "
              f"{r['best_rouge1_before']:>10.4f} {r['best_rouge1_after']:>10.4f} "
              f"{r['delta_rouge1']:>+8.4f}")

    # Δ by ability for oracle_trace
    if "oracle_trace" in results:
        print("\n## Oracle TRACE-FB Δctx_recall by ability\n")
        for ab, m in results["oracle_trace"].get("by_ability", {}).items():
            bar = "▓" * int(abs(m["delta_ctx_recall"]) * 40)
            print(f"  {ab:<30}: {m['delta_ctx_recall']:>+.4f} {bar} (n={m['n']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval-S eval for MEMOIR")
    parser.add_argument("--data", default="data/longmemeval_s.json")
    parser.add_argument("--systems",
                        default="cosine,memoir_no_trace,oracle_trace",
                        help="Comma-separated system names")
    parser.add_argument("--n-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N samples. Use to target specific question types: "
                             "0=single-session-user(0-69), 70=multi-session(70-232), "
                             "233=temporal-reasoning(233-365), 366=knowledge-update(366-443)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="results/longmemeval_s.json")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dump-perqa", action="store_true",
                        help="Write per-sample rows to results/perqa/<stem>_<system>.jsonl "
                             "for paired significance testing (significance_longmemeval.py)")
    args = parser.parse_args()

    api_key  = os.getenv("ANTHROPIC_API_KEY", "")
    is_oauth = "oat" in api_key
    if "memoir_trace" in args.systems and (not api_key or is_oauth):
        print("Note: memoir_trace needs a direct API key (sk-ant-api03-...). "
              "Skipping memoir_trace.\n")
        systems = [s for s in args.systems.split(",") if s.strip() != "memoir_trace"]
    else:
        systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    llm_client = None
    if "memoir_trace" in systems and api_key and not is_oauth:
        from llm_client import LLMClient
        llm_client = LLMClient(api_key=api_key)

    results = run_eval(
        data_path=args.data,
        systems=systems,
        n_samples=args.n_samples,
        k=args.k,
        verbose=not args.quiet,
        llm_client=llm_client,
        offset=args.offset,
        dump_perqa=args.dump_perqa,
        output=args.output,
    )

    print_table(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
