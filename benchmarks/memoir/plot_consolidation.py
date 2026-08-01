"""
Paper figures: prospective-consolidation results (docs/CONSOLIDATION_APPENDIX.md).
Companion to plot_results.py (which covers the retrospective TRACE-family figures).
Renders the 3 main-body figures for the consolidation half:
  1. Rate-distortion frontier: Deltatokens% vs eps_hat, one point per variant.
  2. Per-category Delta rec@5: certified / certified+supersede / uncertified vs no_sleep.
  3. RQ6 ledger preservation: mean r_pre -> r_post per variant (certified vs uncertified).

Usage:
  python benchmarks/memoir/plot_consolidation.py
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
_CAT_ID = {1: "single_hop", 2: "temporal", 3: "multi_hop", 4: "open_ended", 5: "adversarial"}


def _load(name: str) -> dict | None:
    p = _RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_dedup_perqa(name: str) -> list[dict]:
    p = _RESULTS / name
    if not p.exists():
        return []
    rows: dict[tuple, dict] = {}
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            if not r.get("count", True):
                continue
            key = (r["sample_id"], r["question"], r["category"])
            rows[key] = r  # last-occurrence wins (matches significance.py convention)
    return list(rows.values())


def fig_rate_distortion_frontier() -> None:
    """Scatter: Deltatokens% (x) vs eps_hat (y), one point per consolidation variant."""
    rq5 = _load("consolidation_rq5.json")
    summarize = _load("ablation_A3_summarize_all.json")
    if rq5 is None:
        print("  [fig_rate_distortion_frontier] missing consolidation_rq5.json, skipping")
        return

    anchor_tokens = rq5["no_sleep"]["final_tokens_total"]
    points = []  # (label, delta_tokens_pct, eps_hat, color)
    variant_colors = {
        "no_sleep": "#9aa5b1",
        "certified": "#1a9e6f",
        "uncertified": "#c9622a",
        "recency_prune": "#6c7a89",
        "random_prune": "#8a6c89",
    }
    for variant, color in variant_colors.items():
        v = rq5.get(variant)
        if v is None:
            continue
        delta_pct = 100.0 * (v["final_tokens_total"] - anchor_tokens) / anchor_tokens
        points.append((variant, delta_pct, v["eps_hat"], color))

    if summarize is not None:
        v = summarize["summarize_all"]
        anchor2 = summarize["no_sleep"]["final_tokens_total"]
        delta_pct = 100.0 * (v["final_tokens_total"] - anchor2) / anchor2
        points.append(("summarize_all", delta_pct, v["eps_hat"], "#d1345b"))

    # Small manual label offsets to de-clutter the near-zero-compression cluster
    # (no_sleep/certified/recency_prune/random_prune all sit within ~0.3% of x=0).
    label_offsets = {
        "no_sleep": (10, -14),
        "certified": (10, 10),
        "recency_prune": (10, 26),
        "random_prune": (10, 42),
        "uncertified": (8, 6),
        "summarize_all": (8, 6),
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, x, y, color in points:
        ax.scatter(x, y, s=140, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        dx, dy = label_offsets.get(label, (8, 6))
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=9)

    ax.set_xlabel("Δ tokens (%) vs no_sleep anchor  (negative = compression)")
    ax.set_ylabel("ε̂ (estimated sufficiency distortion)")
    ax.set_title("Rate–distortion frontier: certified consolidation vs uncertified baselines")
    ax.axvline(0, color="#ccc", linewidth=0.8, zorder=1)
    ax.set_ylim(-0.006, max(p[2] for p in points) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    fig.text(0.5, 0.10,
             "certified: low compression, low ε̂. uncertified: high compression, high ε̂\n"
             "(the expected tradeoff). summarize_all: ~ -80% compression but ε̂=0.000 —\n"
             "flagged as a likely metric artifact (dia_id-union credit transfer lets one\n"
             "mega-item spuriously satisfy many probes), not genuine free compression —\n"
             "see docs/CONSOLIDATION_APPENDIX.md §4.9",
             ha="center", fontsize=8, color="#666")
    out = _FIGDIR / "consolidation_rate_distortion.png"
    fig.savefig(out, dpi=200)
    print(f"  saved {out}")
    plt.close(fig)


def fig_per_category_delta() -> None:
    """Grouped bars: per-category Delta rec@5 vs no_sleep for certified / +supersede / uncertified."""
    no_sleep = _load_dedup_perqa("consolidation_rq5_no_sleep_perqa.jsonl")
    certified = _load_dedup_perqa("consolidation_rq5_certified_perqa.jsonl")
    uncertified = _load_dedup_perqa("consolidation_rq5_uncertified_perqa.jsonl")
    supersede = _load_dedup_perqa("consolidation_supersede_certified_supersede_perqa.jsonl")

    if not no_sleep or not certified:
        print("  [fig_per_category_delta] missing required perqa dumps, skipping")
        return

    def per_cat_mean(rows: list[dict]) -> dict[str, float]:
        cat: dict[str, list[float]] = {}
        for r in rows:
            name = _CAT_ID.get(r["category"], str(r["category"]))
            cat.setdefault(name, []).append(r["rec5"])
        return {c: sum(v) / len(v) for c, v in cat.items()}

    base = per_cat_mean(no_sleep)
    series = {"certified": per_cat_mean(certified)}
    if supersede:
        series["certified+supersede"] = per_cat_mean(supersede)
    if uncertified:
        series["uncertified"] = per_cat_mean(uncertified)

    x = np.arange(len(_CATS))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1a9e6f", "#2f7dd1", "#c9622a"]

    for i, (name, vals) in enumerate(series.items()):
        deltas = [100.0 * (vals.get(c, 0.0) - base.get(c, 0.0)) for c in _CATS]
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, deltas, width, label=name, color=colors[i % len(colors)])
        for b, d in zip(bars, deltas):
            y = d + (0.15 if d >= 0 else -0.5)
            ax.text(b.get_x() + b.get_width() / 2, y, f"{d:+.1f}", ha="center",
                    fontsize=7, color="#333")

    ax.axhline(0, color="#999", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(_CATS, rotation=20, ha="right")
    ax.set_ylabel("Δ rec@5 (pp) vs no_sleep")
    ax.set_title("Consolidation is category-safe: per-category Δ rec@5 (n=1905, deduped)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, -0.30,
            "no significant per-category regression (paired bootstrap, results/sig_*.json);\n"
            "largest single move (multi_hop, certified) is not significant at n=87",
            transform=ax.transAxes, ha="center", fontsize=8, color="#666")
    fig.tight_layout()
    out = _FIGDIR / "consolidation_per_category_delta.png"
    fig.savefig(out, dpi=200)
    print(f"  saved {out}")
    plt.close(fig)


def fig_rq6_ledger() -> None:
    """Bar: mean r_pre -> r_post per variant (certified vs uncertified), RQ6 headline."""
    b9 = _load("ablation_B9_correlation.json")
    if b9 is None:
        print("  [fig_rq6_ledger] missing ablation_B9_correlation.json, skipping")
        return

    def mean_pre_post(variant: str) -> tuple[float, float, int]:
        pre, post = [], []
        for samp in b9.get(variant, {}).get("per_sample", []):
            for c in samp.get("correlation", []):
                pre.append(c["r_pre"])
                post.append(c["r_post"])
        if not pre:
            return (0.0, 0.0, 0)
        return (sum(pre) / len(pre), sum(post) / len(post), len(pre))

    variants = ["certified", "uncertified"]
    colors = ["#1a9e6f", "#c9622a"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(variants))
    width = 0.32

    pre_vals, post_vals, ns = [], [], []
    for v in variants:
        pre, post, n = mean_pre_post(v)
        pre_vals.append(pre)
        post_vals.append(post)
        ns.append(n)

    bars_pre = ax.bar(x - width / 2, pre_vals, width, label="r_pre (before sleep)",
                       color="#9aa5b1")
    bars_post = ax.bar(x + width / 2, post_vals, width, label="r_post (after sleep)",
                        color=colors)

    for b, v in zip(bars_pre, pre_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for b, v in zip(bars_post, post_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n(n={n} sleep-events)" for v, n in zip(variants, ns)])
    ax.set_ylabel("Pearson r (attribution vs evidence recall)")
    ax.set_title("RQ6: ledger preservation across sleep (full 10-sample LoCoMo run)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.text(0.5, 0.03,
             "certified degrades less than uncertified (Δr −0.0055 vs −0.0170), but the\n"
             "paired-bootstrap test on that gap is not significant at n=28 (p=0.20) — see\n"
             "docs/CONSOLIDATION_APPENDIX.md §4.1 / EXPERIMENT_TABLES.md T4",
             ha="center", fontsize=8, color="#666")
    out = _FIGDIR / "consolidation_rq6_ledger.png"
    fig.savefig(out, dpi=200)
    print(f"  saved {out}")
    plt.close(fig)


def fig_readiness() -> None:
    """Two-panel readiness ledger (docs/CONSOLIDATION_APPENDIX.md Part 8): which
    results clear the bar to cite as a paper claim, and how far the rest are from
    their target. Values are hardcoded from Part 8's table -- this figure
    visualizes already-verified numbers, it does not compute anything new."""
    # (label, p_value or None if not a significance test, status)
    # status: "meets" (#1a9e6f green), "below" (#c9622a orange), "notrun" (#9aa5b1 gray)
    # NOTE: B7 (credit-transfer max vs zero) is a real, correctly-signed, citable
    # descriptive result (0.5898 vs 0.5878 vs no_sleep's 0.5882) but has never had
    # a paired-bootstrap significance test run against it -- deliberately excluded
    # from this p-value panel rather than showing a fabricated number.
    sig_rows = [
        ("certified vs uncertified\n(overall rec@5)", 0.30, "below"),
        ("SUPERSEDE vs certified\n(overall rec@5)", 0.7381, "below"),
        ("RQ6 Δr\n(certified vs uncertified)", 0.2024, "below"),
        ("B6 credit-protection\n(on vs off)", None, "notrun"),
        ("B8 co-cite clusters\n(on vs off)", None, "notrun"),
        ("B7 credit-transfer\n(max vs zero)", None, "notrun"),
    ]

    # (label, current_n, target_n)
    n_rows = [
        ("tau sweep", 2, 10),
        ("sleep_every sweep", 2, 10),
        ("DIGEST accuracy", 1, 10),
        ("RQ6 sleep-events", 28, 110),
    ]

    colors = {"meets": "#1a9e6f", "below": "#c9622a", "notrun": "#9aa5b1"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # ── Panel A: significance readiness (log-scale p vs alpha=0.05) ──────────
    y = np.arange(len(sig_rows))
    for i, (label, p, status) in enumerate(sig_rows):
        color = colors[status]
        if p is None:
            ax1.barh(i, 1.0, color=color, alpha=0.35, height=0.55)
            ax1.text(1.05, i, "not yet run", va="center", fontsize=8, color="#666")
        else:
            ax1.barh(i, p, color=color, height=0.55)
            ax1.text(min(p, 1.0) * 1.05 if p < 0.9 else p - 0.15, i, f"p={p:.3f}",
                     va="center", ha="left" if p < 0.9 else "right", fontsize=8,
                     color="#333" if p < 0.9 else "white")
    ax1.axvline(0.05, color="#333", linewidth=1.2, linestyle="--", zorder=3)
    ax1.text(0.05, len(sig_rows) - 0.3, " α=0.05", fontsize=8, color="#333",
             rotation=90, va="top", ha="left")
    ax1.set_xscale("log")
    ax1.set_xlim(0.01, 1.3)
    ax1.set_yticks(y)
    ax1.set_yticklabels([l for l, _, _ in sig_rows], fontsize=8)
    ax1.set_xlabel("p-value (log scale) — left of the dashed line clears significance")
    ax1.set_title("Significance readiness")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.invert_yaxis()

    # ── Panel B: sample-size readiness (current n vs target n) ───────────────
    y2 = np.arange(len(n_rows))
    for i, (label, cur, tgt) in enumerate(n_rows):
        ax2.barh(i, cur, color="#c9622a", height=0.5, label="current n" if i == 0 else None)
        ax2.plot([tgt, tgt], [i - 0.3, i + 0.3], color="#1a9e6f", linewidth=3,
                 solid_capstyle="butt", label="target n" if i == 0 else None)
        ax2.text(max(cur, 1) + tgt * 0.03, i, f"{cur} → {tgt}", va="center", fontsize=8,
                 color="#333")
    ax2.set_yticks(y2)
    ax2.set_yticklabels([l for l, _, _ in n_rows], fontsize=8)
    ax2.set_xlabel("n (samples or sleep-events)")
    ax2.set_title("Sample-size readiness")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.invert_yaxis()

    fig.suptitle("Consolidation-appendix readiness for the next full run", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(0.5, 0.01,
             "green = meets bar, orange = below target, gray = not yet run — see "
             "docs/CONSOLIDATION_APPENDIX.md Part 8 for the full table",
             ha="center", fontsize=8, color="#666")
    out = _FIGDIR / "consolidation_readiness.png"
    fig.savefig(out, dpi=200)
    print(f"  saved {out}")
    plt.close(fig)


def main() -> None:
    print("Generating consolidation-appendix figures...")
    fig_rate_distortion_frontier()
    fig_per_category_delta()
    fig_rq6_ledger()
    fig_readiness()
    print("Done.")


if __name__ == "__main__":
    main()
