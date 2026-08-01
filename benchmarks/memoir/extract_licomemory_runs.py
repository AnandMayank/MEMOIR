"""
Extract LiCoMemory's 5-run LoCoMo group_1 metrics from the EXISTING run
artifacts (no rerun — their runs are expensive), producing the mean ± std row
for the paper's main same-model comparison table (T1 in docs/EXPERIMENT_TABLES.md).

Sources per run dir (/home/user/LiCoMemory/results/llama8b_group1_run{1..5}):
  dynamic_memory.log   — their OWN LLM-judge verdict line
                         ("LLM Evaluation completed: X/N correct (accuracy: Y)"),
                         their Session Matching score, per-question Time/Token
                         Statistics blocks, and log timestamps (T_R via wall-clock
                         gaps between consecutive "Question i/N:" entries — the
                         protocol documented in results/memoir_vs_licomemory_llama8b.md)
  results/results.json — per-question gold/output/top_session_ids (post-patch),
                         used for session-level Rec@5 / Rec@15 against LoCoMo's
                         evidence annotations (D<sess>:<turn>).

Honesty notes carried into the output JSON:
  - K_R is an UNDERCOUNT (their repo logs Retrieval Tokens: 0 — documented bug #2).
  - Rec@k here is SESSION-level (their protocol), not memory-level like MEMOIR's
    rec@5 — comparable within this table only because both sides get the
    session-level metric in T1's LiCoMemory rows (MEMOIR rows use its own
    harness's session-mapped metric from the same-model runs).

Usage:
  python benchmarks/memoir/extract_licomemory_runs.py \
    --runs-dir /home/user/LiCoMemory/results \
    --dataset data/locomo10.json \
    --output results/licomemory_llama8b_5run.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_QUESTION = re.compile(r"Question (\d+)/(\d+):")
_LLM_EVAL = re.compile(r"LLM Evaluation completed: (\d+)/(\d+) correct \(accuracy: ([\d.]+)\)")
_MATCHING = re.compile(r"Session Matching Evaluation: average score = ([\d.]+)")
_TOTAL_TOKENS = re.compile(r"Total Tokens: (\d+)")


def parse_log(log_path: Path) -> dict:
    """Pull judged accuracy, matching score, T_R (timestamp gaps), K_R."""
    text = log_path.read_text(errors="replace")
    out: dict = {}

    m = _LLM_EVAL.search(text)
    if m:
        out["llm_correct"] = int(m.group(1))
        out["llm_total"] = int(m.group(2))
        out["llm_accuracy"] = float(m.group(3))
    m = _MATCHING.search(text)
    if m:
        out["session_matching"] = float(m.group(1))

    # T_R: wall-clock gaps between consecutive "Question i/N:" entries —
    # includes entity extraction, retrieval, reranking, generation, judge
    # (the same all-inclusive definition used in the earlier single-run doc).
    stamps = []
    for line in text.splitlines():
        if _QUESTION.search(line):
            ts = _TS.match(line)
            if ts:
                stamps.append(datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S"))
    if len(stamps) >= 2:
        gaps = [(b - a).total_seconds() for a, b in zip(stamps[:-1], stamps[1:])]
        # guard against idle gaps (log includes post-run evaluation phases)
        gaps = [g for g in gaps if 0 < g < 600]
        if gaps:
            out["t_r_mean_s"] = statistics.mean(gaps)
            out["t_r_median_s"] = statistics.median(gaps)
            out["n_timing_gaps"] = len(gaps)

    tokens = [int(t) for t in _TOTAL_TOKENS.findall(text)]
    if tokens:
        out["k_r_mean_tokens"] = statistics.mean(tokens)
        out["k_r_note"] = "UNDERCOUNT: repo logs Retrieval Tokens: 0 (their bug #2)"
    return out


def gold_sessions_by_question(dataset_path: str, sample_idx: int = 0) -> dict:
    """question text -> set of gold session ids (from D<sess>:<turn> evidence)."""
    data = json.loads(Path(dataset_path).read_text())
    sample = data[sample_idx]
    gold = {}
    for qa in sample.get("qa", []):
        evs = qa.get("evidence") or []
        sess = set()
        for ev in evs:
            m = re.match(r"D(\d+):", str(ev))
            if m:
                sess.add(int(m.group(1)))
        if sess:
            gold[str(qa.get("question", "")).strip()] = sess
    return gold


def session_recall(results_json: Path, gold: dict, k: int) -> tuple[float, int]:
    rows = json.loads(results_json.read_text())
    hits = n = 0
    for r in rows:
        g = gold.get(str(r.get("question", "")).strip())
        if not g:
            continue
        n += 1
        # their session ids are strings like 'D4' — strip the D prefix
        top = []
        for s in (r.get("top_session_ids") or [])[:k]:
            m = re.match(r"D?(\d+)$", str(s))
            if m:
                top.append(int(m.group(1)))
        if g & set(top):
            hits += 1
    return (hits / n if n else 0.0), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--runs-dir", default="/home/user/LiCoMemory/results")
    ap.add_argument("--runs-glob", default="llama8b_group1_run*")
    ap.add_argument("--dataset", default=str(_REPO / "data" / "locomo10.json"))
    ap.add_argument("--sample-idx", type=int, default=0,
                    help="LoCoMo sample for group_1 (sample 0 = 199 QAs)")
    ap.add_argument("--output", default="results/licomemory_llama8b_5run.json")
    args = ap.parse_args()

    run_dirs = sorted(Path(args.runs_dir).glob(args.runs_glob))
    if not run_dirs:
        raise SystemExit(
            f"No run dirs matching {args.runs_glob!r} under {args.runs_dir} — "
            f"the LiCoMemory artifacts live on the dev machine, not on compute "
            f"instances. NOT writing {args.output} (a previously-extracted "
            f"summary may already be committed there; do not clobber it).")

    gold = gold_sessions_by_question(args.dataset, args.sample_idx)
    print(f"Gold session sets for {len(gold)} questions")

    per_run = []
    for run_dir in run_dirs:
        row: dict = {"run": run_dir.name}
        log = run_dir / "dynamic_memory.log"
        if log.exists():
            row.update(parse_log(log))
        rj = run_dir / "results" / "results.json"
        if rj.exists():
            rec5, n5 = session_recall(rj, gold, k=5)
            rec15, _ = session_recall(rj, gold, k=15)
            row["rec5_session"] = rec5
            row["rec15_session"] = rec15
            row["n_matched_questions"] = n5
        complete = "llm_accuracy" in row and "rec5_session" in row
        row["complete"] = complete
        per_run.append(row)
        print(f"  {run_dir.name}: "
              + (f"acc={row.get('llm_accuracy', float('nan')):.4f} " if 'llm_accuracy' in row else "acc=MISSING ")
              + (f"rec5={row.get('rec5_session', float('nan')):.4f} " if 'rec5_session' in row else "rec5=MISSING ")
              + (f"T_R={row.get('t_r_mean_s', float('nan')):.2f}s " if 't_r_mean_s' in row else "")
              + (f"K_R={row.get('k_r_mean_tokens', float('nan')):.0f}" if 'k_r_mean_tokens' in row else ""))

    def agg(key: str) -> dict:
        vals = [r[key] for r in per_run if key in r and r.get("complete")]
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals)}

    n_complete = sum(1 for r in per_run if r.get("complete"))
    if n_complete == 0:
        raise SystemExit(
            f"Found {len(per_run)} run dir(s) but none had complete "
            f"accuracy+recall data — NOT overwriting {args.output}.")

    summary = {
        "protocol": "LiCoMemory public repo, llama3.1:8b via Ollama, LoCoMo "
                    "group_1 (sample 0, 199 QAs), patched top_session_ids "
                    "(their bug #1), MEMOIR fine-tuned encoder substitute for "
                    "BGE-M3 (documented confound)",
        "accuracy": agg("llm_accuracy"),
        "rec5_session": agg("rec5_session"),
        "rec15_session": agg("rec15_session"),
        "t_r_mean_s": agg("t_r_mean_s"),
        "k_r_mean_tokens": agg("k_r_mean_tokens"),
        "k_r_note": "UNDERCOUNT: their repo logs Retrieval Tokens: 0 (bug #2)",
        "per_run": per_run,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    a, r5 = summary["accuracy"], summary["rec5_session"]
    print(f"\n{'='*60}")
    print(f"LiCoMemory llama3.1:8b group_1, {a['n']} complete runs:")
    if a["mean"] is not None:
        print(f"  Acc  = {a['mean']*100:.2f}% ± {a['std']*100:.2f}")
        print(f"  Rec@5(session) = {r5['mean']*100:.2f}% ± {r5['std']*100:.2f}")
        print(f"  T_R  = {summary['t_r_mean_s']['mean']:.2f}s ± {summary['t_r_mean_s']['std']:.2f}")
        print(f"  K_R  = {summary['k_r_mean_tokens']['mean']:.0f} ± {summary['k_r_mean_tokens']['std']:.0f} (undercount)")
    print(f"{'='*60}\nSaved to {out}")


if __name__ == "__main__":
    main()
