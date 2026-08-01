"""
Paper figures: MEMOIR retrieval-quality improvements on LoCoMo (evidence-based
rec@5, v7 protocol, n=1905 deduped QA pairs -- see docs/COMBINED_PAPER_PLAN.md).

Reads the paired-bootstrap significance JSONs already produced by
significance.py (results/sig_*.json) plus any newly produced ones, and
renders two figures:
  1. Overall rec@5 bar chart: cosine baseline -> memoir_no_trace -> memoir_trace
     -> memoir_ucb -> memoir_reliability, with significance stars vs no_trace.
  2. Per-category delta grouped bars: memoir_ucb and memoir_reliability vs
     memoir_no_trace, showing where each component's gain concentrates.

Usage:
  python benchmarks/memoir/plot_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_RESULTS = Path(__file__).parent.parent.parent / "results"
_FIGDIR = _RESULTS / "figures"
_FIGDIR.mkdir(parents=True, exist_ok=True)

_CATS = ["single_hop", "temporal", "multi_hop", "open_ended", "adversarial"]


def _load(name: str) -> dict | None:
    p = _RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fig_overall_bar() -> None:
    """Overall rec@5: baseline ladder with significance stars vs no_trace."""
    trace_vs_cosine = _load("sig_trace_vs_cosine.json")
    trace_vs_notrace = _load("sig_trace_vs_notrace.json")
    ucb_vs_notrace = _load("sig_ucb_vs_notrace.json")
    rel_vs_notrace = _load("sig_reliability_vs_notrace.json")

    if trace_vs_cosine is None or ucb_vs_notrace is None:
        print("  [fig_overall_bar] missing required significance JSONs, skipping")
        return

    labels = ["cosine", "memoir_no_trace", "memoir_trace", "memoir_ucb"]
    means = [
        trace_vs_cosine["overall"]["mean_b"],
        trace_vs_notrace["overall"]["mean_b"] if trace_vs_notrace else None,
        trace_vs_cosine["overall"]["mean_a"],
        ucb_vs_notrace["overall"]["mean_a"],
    ]
    star_vs_notrace = [None, None,
                       _stars(trace_vs_notrace["overall"]["p_two_sided"]) if trace_vs_notrace else None,
                       _stars(ucb_vs_notrace["overall"]["p_two_sided"])]

    if rel_vs_notrace is not None:
        labels.append("memoir_reliability")
        means.append(rel_vs_notrace["overall"]["mean_a"])
        star_vs_notrace.append(_stars(rel_vs_notrace["overall"]["p_two_sided"]))

    # drop any None (missing) entries
    keep = [i for i, m in enumerate(means) if m is not None]
    labels = [labels[i] for i in keep]
    means = [means[i] for i in keep]
    star_vs_notrace = [star_vs_notrace[i] for i in keep]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#9aa5b1", "#6c7a89", "#2f7dd1", "#1a9e6f", "#c9622a"][: len(labels)]
    bars = ax.bar(labels, means, color=colors, width=0.6)
    for bar, m, star in zip(bars, means, star_vs_notrace):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10)
        if star:
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.022, star,
                    ha="center", va="bottom", fontsize=11, color="#333")

    ax.set_ylabel("rec@5 (evidence recall @ k=5)")
    ax.set_title("LoCoMo retrieval quality (n=1905 QA pairs, evidence-based, v7 protocol)")
    ax.set_ylim(0, max(means) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, -0.22, "stars: significance vs memoir_no_trace (paired bootstrap, 10k resamples)",
           transform=ax.transAxes, ha="center", fontsize=8, color="#666")
    fig.tight_layout()
    out = _FIGDIR / "locomo_overall_rec5.png"
    fig.savefig(out, dpi=200)
    print(f"  saved {out}")
    plt.close(fig)


def fig_per_category_delta() -> None:
    """Grouped bars: per-category delta (vs memoir_no_trace) for each component."""
    series = {}
    ucb = _load("sig_ucb_vs_notrace.json")
    if ucb:
        series["memoir_ucb"] = ucb
    rel = _load("sig_reliability_vs_notrace.json")
    if rel:
        series["memoir_reliability"] = rel
    trace = _load("sig_trace_vs_notrace.json")
    if trace:
        series["memoir_trace (ROUGE1)"] = trace

    if not series:
        print("  [fig_per_category_delta] no significance JSONs found, skipping")
        return

    x = np.arange(len(_CATS))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1a9e6f", "#c9622a", "#2f7dd1"]

    for i, (name, data) in enumerate(series.items()):
        by_cat = data.get("by_category", {})
        deltas = [by_cat.get(c, {}).get("delta", 0.0) for c in _CATS]
        pvals = [by_cat.get(c, {}).get("p_two_sided", 1.0) for c in _CATS]
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, deltas, width, label=name, color=colors[i % len(colors)])
        for b, d, p in zip(bars, deltas, pvals):
            star = _stars(p)
            y = d + (0.003 if d >= 0 else -0.012)
            ax.text(b.get_x() + b.get_width() / 2, y, star, ha="center",
                   fontsize=8, color="#333")

    ax.axhline(0, color="#999", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(_CATS, rotation=20, ha="right")
    ax.set_ylabel("Δ rec@5 vs memoir_no_trace")
    ax.set_title("Per-category retrieval gains (paired bootstrap significance stars)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = _FIGDIR / "locomo_per_category_delta.png"
    fig.savefig(out, dpi=200)
    print(f"  saved {out}")
    plt.close(fig)


_LME_SLICES = ["single_session_user", "multi_session", "temporal", "knowledge_update"]
_LME_LABELS = ["single-session-user", "multi-session", "temporal", "knowledge-update"]


def fig_longmemeval_ceiling() -> None:
    """LongMemEval-S cross-benchmark check: ev_recall_after per slice/system,
    confirming the same ceiling effect TABLE 2 documented for TRACE-FB also
    holds for UCB and RELIABILITY (both Δ=0.0000 in every slice)."""
    rows = []
    for slice_name in _LME_SLICES:
        p = _RESULTS / f"longmemeval_{slice_name}_reliability.json"
        if not p.exists():
            print(f"  [fig_longmemeval_ceiling] missing {p}, skipping")
            return
        rows.append(json.loads(p.read_text()))

    systems = ["cosine", "memoir_no_trace", "memoir_ucb", "memoir_reliability"]
    colors = ["#9aa5b1", "#6c7a89", "#1a9e6f", "#c9622a"]

    x = np.arange(len(_LME_SLICES))
    width = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    for i, sys_name in enumerate(systems):
        vals = [r[sys_name]["ev_recall_after"] for r in rows]
        offset = (i - (len(systems) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=sys_name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(_LME_LABELS, rotation=15, ha="right")
    ax.set_ylabel("ev_recall @ k=5 (evidence recall)")
    ax.set_ylim(0, 1.08)
    ax.set_title("LongMemEval-S (n=20/slice): retrieval already at ceiling pre-feedback")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, -0.30, "memoir_ucb and memoir_reliability: Δ=0.0000, p=1.0000 vs "
           "memoir_no_trace in all 4 slices (no headroom to move)",
           transform=ax.transAxes, ha="center", fontsize=8, color="#666")
    fig.tight_layout()
    out = _FIGDIR / "longmemeval_ceiling.png"
    fig.savefig(out, dpi=200)
    print(f"  saved {out}")
    plt.close(fig)


def main() -> None:
    print("Generating paper figures...")
    fig_overall_bar()
    fig_per_category_delta()
    fig_longmemeval_ceiling()
    print(f"Done. Figures in {_FIGDIR}")


if __name__ == "__main__":
    main()
