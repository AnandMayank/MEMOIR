"""
Paired bootstrap significance test between two systems' per-QA results.

Answers the "n=10 LoCoMo samples" objection: the paired unit is the QA pair
(n≈2433), resampled with replacement 10k times; systems are compared on the
SAME QA set (pairing key: sample_id + question + category).

Usage:
  # First produce per-QA dumps:
  python benchmarks/memoir/eval_locomo.py --systems memoir_trace,memoir_v3 \
      --dump-perqa --output results/locomo_sig.json ...
  # Then:
  python benchmarks/memoir/significance.py \
      --a results/perqa/locomo_sig_memoir_v3.jsonl \
      --b results/perqa/locomo_sig_memoir_trace.jsonl \
      --metric rec5 --n-boot 10000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_CATEGORY_NAMES = {
    1: "single_hop", 2: "temporal", 3: "multi_hop",
    4: "open_ended", 5: "adversarial",
}


def load_rows(path: str) -> dict[tuple, dict]:
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("count", True):
                continue
            key = (r.get("sample_id"), r.get("question"), r.get("category"))
            rows[key] = r
    return rows


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int,
                     seed: int = 13) -> dict:
    """Two-sided paired bootstrap on mean difference (a − b)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diff = a - b
    observed = float(diff.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    # two-sided p: proportion of bootstrap means on the other side of 0,
    # doubled (with the +1 correction)
    if observed >= 0:
        p = 2.0 * (np.sum(boot_means <= 0) + 1) / (n_boot + 1)
    else:
        p = 2.0 * (np.sum(boot_means >= 0) + 1) / (n_boot + 1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": observed,
        "ci95": [float(lo), float(hi)],
        "p_two_sided": min(1.0, float(p)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired bootstrap for MEMOIR systems")
    ap.add_argument("--a", required=True, help="per-QA JSONL for system A")
    ap.add_argument("--b", required=True, help="per-QA JSONL for system B")
    ap.add_argument("--metric", default="rec5",
                    choices=["rec5", "rec15", "full_ev5", "full_ev15",
                             "ctx_recall", "best_tf1", "rec", "acc"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--output", default=None, help="optional JSON output path")
    args = ap.parse_args()

    rows_a = load_rows(args.a)
    rows_b = load_rows(args.b)
    keys = sorted(set(rows_a) & set(rows_b), key=str)
    dropped = len(set(rows_a) ^ set(rows_b))
    if dropped:
        print(f"warning: {dropped} unpaired rows dropped")
    if not keys:
        raise SystemExit("no paired rows — did both runs use the same QA set?")

    a = np.array([rows_a[k][args.metric] for k in keys], dtype=float)
    b = np.array([rows_b[k][args.metric] for k in keys], dtype=float)

    name_a = Path(args.a).stem
    name_b = Path(args.b).stem
    result = {"metric": args.metric, "system_a": name_a, "system_b": name_b,
              "overall": paired_bootstrap(a, b, args.n_boot)}

    by_cat: dict[int, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_cat[int(k[2])].append(i)
    result["by_category"] = {}
    for cat, idxs in sorted(by_cat.items()):
        ii = np.array(idxs)
        result["by_category"][_CATEGORY_NAMES.get(cat, str(cat))] = \
            paired_bootstrap(a[ii], b[ii], args.n_boot)

    o = result["overall"]
    print(f"\nPaired bootstrap ({args.metric}, {args.n_boot} resamples, "
          f"n={o['n']} QAs)")
    print(f"  A={name_a}: {o['mean_a']:.4f}   B={name_b}: {o['mean_b']:.4f}")
    print(f"  Δ={o['delta']:+.4f}  95% CI [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}]"
          f"  p={o['p_two_sided']:.4f}")
    for cat, r in result["by_category"].items():
        star = " *" if r["p_two_sided"] < 0.05 else ""
        print(f"    {cat:<12} Δ={r['delta']:+.4f}  p={r['p_two_sided']:.4f}"
              f"  (n={r['n']}){star}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
