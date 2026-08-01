"""
Paired bootstrap significance test for LongMemEval-S per-sample dumps
(eval_longmemeval.py --dump-perqa). Mirrors significance.py's LoCoMo test but
pairs on (sample_id, question) and groups by "ability" instead of "category" --
LongMemEval-S has no repeated-question artifact (one question per sample, one
row per evaluate_sample() call), so no dedup step is needed here (contrast
eval_locomo.py's _aggregate() fix, docs/COMBINED_PAPER_PLAN.md 2026-07-22).

Usage:
  python benchmarks/memoir/significance_longmemeval.py \\
      --a results/perqa/longmemeval_s_memoir_ucb.jsonl \\
      --b results/perqa/longmemeval_s_memoir_no_trace.jsonl \\
      --metric ev_recall_after --n-boot 10000
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from significance import paired_bootstrap


def load_rows(path: str) -> dict[tuple, dict]:
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r.get("sample_id"), r.get("question"))
            rows[key] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired bootstrap for MEMOIR LongMemEval-S systems")
    ap.add_argument("--a", required=True, help="per-sample JSONL for system A")
    ap.add_argument("--b", required=True, help="per-sample JSONL for system B")
    ap.add_argument("--metric", default="ev_recall_after",
                    choices=["ev_recall_after", "ev_recall_before", "delta_ev_recall",
                             "ctx_recall_after", "ctx_recall_before", "delta_ctx_recall",
                             "best_rouge1_after", "delta_rouge1"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rows_a = load_rows(args.a)
    rows_b = load_rows(args.b)
    keys = sorted(set(rows_a) & set(rows_b), key=str)
    dropped = len(set(rows_a) ^ set(rows_b))
    if dropped:
        print(f"warning: {dropped} unpaired rows dropped")
    if not keys:
        raise SystemExit("no paired rows -- did both runs use the same samples?")

    a = np.array([rows_a[k][args.metric] for k in keys], dtype=float)
    b = np.array([rows_b[k][args.metric] for k in keys], dtype=float)

    name_a = Path(args.a).stem
    name_b = Path(args.b).stem
    result = {"metric": args.metric, "system_a": name_a, "system_b": name_b,
              "overall": paired_bootstrap(a, b, args.n_boot)}

    by_ability: dict[str, list] = defaultdict(list)
    for i, k in enumerate(keys):
        by_ability[rows_a[k].get("ability", "unknown")].append(i)
    result["by_ability"] = {}
    for ab, idxs in sorted(by_ability.items()):
        ii = np.array(idxs)
        result["by_ability"][ab] = paired_bootstrap(a[ii], b[ii], args.n_boot)

    o = result["overall"]
    print(f"\nPaired bootstrap ({args.metric}, {args.n_boot} resamples, n={o['n']} samples)")
    print(f"  A={name_a}: {o['mean_a']:.4f}   B={name_b}: {o['mean_b']:.4f}")
    print(f"  Δ={o['delta']:+.4f}  95% CI [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}]"
          f"  p={o['p_two_sided']:.4f}")
    for ab, r in result["by_ability"].items():
        star = " *" if r["p_two_sided"] < 0.05 else ""
        print(f"    {ab:<28} Δ={r['delta']:+.4f}  p={r['p_two_sided']:.4f}"
              f"  (n={r['n']}){star}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
