"""
Certified memory consolidation for MemoirStore (the PROSPECTIVE half of the
combined lifecycle paper — see docs/COMBINED_PAPER_PLAN.md).

Ported from /home/user/certified-memory-consolidation (overlap_judge.py,
certificate_calibration.py, merger.py, pruner.py) and adapted to MemoirStore's
MemoryItem substrate. The operator performs only certificate-gated moves:

  - certified merge:  {x_i} -> z applied iff a faithfulness-judge ensemble
    certifies z entails every x_i (score >= tau; RCPS-calibratable).
  - dominated prune:  remove x only if a retained neighbor entails x
    (judge-gated; the cited entailer is protected for the pass).

MEMOIR-specific additions (the algorithmic bridges):
  - Credit protection (Bridge 1): items with w_imp > protect_w_imp are never
    pruned and merge only under a UNANIMOUS certificate — TRACE-observed
    answer-carrying memories are where sufficiency distortion actually lands.
  - Co-cite-proposed clusters (Bridge 1): the co-citation graph seeds merge
    candidates at a lower cosine threshold (credit earned together = extra
    redundancy evidence). Co-cited but embedding-dissimilar pairs (multi-hop
    supports covering different hops) are deliberately NOT clustered.
  - Credit transfer (Bridge 2): merged item inherits w_imp = max(constituents)
    and the union of evidence dia_ids; co-cite edges are re-keyed onto the
    merged id. Sound under a truly lossless merge (Thm 1 corollary).

Execution is REVERSIBLE: merged-away constituents are marked pruned=True
(MEMOIR's PRUNE-EFP flag), never deleted — `unprune_all()` restores the store.

Every applied op is recorded as a ConsolidationOp (audit trail) so delta can be
measured by an independent auditor and rho by intersecting op constituents with
TRACE-observed supports.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .schema import MemoryItem

logger = logging.getLogger("harpo.memoir.consolidate")


# ── Judge prompts (3 diverse framings; diversity from the prompt, not sampling) ─

FAITHFULNESS_PROMPT = """You verify whether a MERGED memory entry preserves ALL factual \
content of the original entries it replaces. This is a strict entailment check.

A merge is LOSSLESS only if EVERY fact, name, number, date, and relation in EACH original \
entry can still be recovered from the merged entry. If even one detail is dropped, changed, \
or made ambiguous, the merge is LOSSY.

Original entries:
{originals}

Merged entry:
{merged}

Answer with exactly one word on the first line: LOSSLESS or LOSSY.
Answer:"""

FAITHFULNESS_PROMPT_CHECKLIST = """Compare the ORIGINAL memory entries with their MERGED \
replacement. Work as a checklist: for each original entry, list mentally every fact, \
entity, number, date, and relation, and check whether the merged entry still states or \
implies it.

Original entries:
{originals}

Merged entry:
{merged}

If every checklist item is preserved, the verdict is LOSSLESS; if ANY item fails, the \
verdict is LOSSY. Answer with exactly one word on the first line: LOSSLESS or LOSSY.
Answer:"""

FAITHFULNESS_PROMPT_ADVERSARIAL = """Your job is to FIND A LOST FACT. Below are original \
memory entries and the merged entry that replaced them. Search hard for ANY detail — a \
fact, name, number, date, qualifier, or relation — that appears in an original but cannot \
be recovered from the merged entry.

Original entries:
{originals}

Merged entry:
{merged}

If you found even one lost detail, the merge is LOSSY. Only if you genuinely cannot find \
any lost detail is it LOSSLESS. Answer with exactly one word on the first line: LOSSLESS \
or LOSSY.
Answer:"""

_PROMPT_VARIANTS = (
    FAITHFULNESS_PROMPT,
    FAITHFULNESS_PROMPT_CHECKLIST,
    FAITHFULNESS_PROMPT_ADVERSARIAL,
)

MERGE_PROMPT = """\
You are a memory consolidation agent. Combine the following related memory entries \
into a SINGLE, more concise entry that preserves ALL unique information.

Rules:
- Preserve every unique fact, detail, or data point
- Remove redundant phrasing and repetition
- Maintain temporal order for events if applicable
- Keep the most specific details when there are conflicts
- The merged entry should be shorter than the sum of inputs

Entries to merge:
{entries}

Write the merged entry (a single concise paragraph):"""


# ── Retrieval-safety token coverage (BM25 findability guard) ────────────────────

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "and", "or",
    "of", "in", "on", "at", "to", "for", "by", "with", "as", "it", "its", "this",
    "that", "he", "she", "they", "his", "her", "their",
}


def retrieval_token_coverage(merged: str, originals: list[str]) -> float:
    """
    Fraction of the originals' content tokens surviving in the merged text.
    BM25 retrieves on lexical tokens: an entailment-lossless merge can still be
    retrieval-lossy ("resides" -> "lives" preserves meaning, drops the token a
    query may use). The certificate must cover FINDABILITY, not just content.
    """
    orig_toks: set[str] = set()
    for o in originals:
        orig_toks |= set(re.findall(r"\w+", o.lower()))
    orig_toks -= _STOPWORDS
    if not orig_toks:
        return 1.0
    merged_toks = set(re.findall(r"\w+", merged.lower()))
    return len(orig_toks & merged_toks) / len(orig_toks)


# ── Faithfulness judge ──────────────────────────────────────────────────────────

class FaithfulnessJudge:
    """
    LLM entailment ensemble certifying a consolidation op as lossless.

    CONSERVATIVE BY DESIGN: any ambiguity, parse failure, or exception is a
    LOSSY vote — a false certification silently breaks sufficiency; a false
    rejection only costs a missed compression opportunity.

    score() returns the LOSSLESS vote fraction over n_votes prompt framings;
    is_lossless() certifies iff score >= certify_threshold (default 1.0 =
    unanimous; calibrate_certify_threshold() selects tau with a finite-sample
    Pr[lossy | certified] <= alpha guarantee).
    """

    def __init__(self, generate_fn: Callable[[str], str],
                 n_votes: int = 3, certify_threshold: float = 1.0):
        self.generate_fn = generate_fn
        self.n_votes = max(1, int(n_votes))
        self.certify_threshold = float(certify_threshold)
        self.call_count = 0

    def score(self, merged: str, originals: list[str]) -> float:
        """LOSSLESS vote fraction across prompt framings (exception -> LOSSY)."""
        originals_text = "\n".join(f"- {o}" for o in originals)
        votes = []
        for i in range(self.n_votes):
            prompt = _PROMPT_VARIANTS[i % len(_PROMPT_VARIANTS)].format(
                originals=originals_text, merged=merged,
            )
            try:
                self.call_count += 1
                response = self.generate_fn(prompt)
            except Exception as exc:  # a judge error must never hang consolidation
                logger.warning("Faithfulness judge call failed (%s); LOSSY vote", exc)
                votes.append(False)
                continue
            votes.append(self.parse_verdict(response or ""))
        return sum(votes) / len(votes)

    def is_lossless(self, merged: str, originals: list[str]) -> bool:
        return self.score(merged, originals) >= self.certify_threshold

    @staticmethod
    def parse_verdict(response: str) -> bool:
        """
        Verdict anchored to the FIRST alphabetic token of the first non-empty
        line; any negation word on that line rejects. (A substring test would
        certify "NOT LOSSLESS" — a certificate-breaking bug shipped once in the
        source repo.) Anything else -> LOSSY.
        """
        if not response or not response.strip():
            return False
        first = response.strip().splitlines()[0].upper()
        words = re.findall(r"[A-Z]+", first)
        if not words:
            return False
        negations = {"NOT", "NO", "NOT_LOSSLESS", "LOSSY", "ISN", "NEVER", "CANNOT"}
        return words[0] == "LOSSLESS" and not any(w in negations for w in words)


# ── RCPS-style threshold calibration ────────────────────────────────────────────

def hoeffding_ucb(n_lossy: int, n_total: int, confidence: float = 0.95) -> float:
    """One-sided Hoeffding upper confidence bound on a Bernoulli mean."""
    if n_total <= 0:
        return 1.0
    p_hat = n_lossy / n_total
    slack = math.sqrt(math.log(1.0 / (1.0 - confidence)) / (2.0 * n_total))
    return min(1.0, p_hat + slack)


@dataclass
class CalibrationResult:
    tau: float
    alpha: float
    confidence: float
    n_calibration: int
    n_at_tau: int
    lossy_rate_at_tau: float
    lossy_ucb_at_tau: float
    certified_fraction: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def calibrate_certify_threshold(
    scores: list[float],
    lossy_labels: list[bool],
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> Optional[CalibrationResult]:
    """
    Fixed-sequence threshold selection (Learn-then-Test / RCPS): walk candidate
    thresholds strictest-first; return the most permissive tau whose one-sided
    Hoeffding UCB on Pr[lossy | score >= tau] is <= alpha. None means "delta not
    certifiable at alpha with this judge" — report that honestly.
    """
    if len(scores) != len(lossy_labels) or not scores:
        raise ValueError("scores and lossy_labels must be equal-length and non-empty")

    candidates = sorted(set(scores), reverse=True)
    best: Optional[CalibrationResult] = None
    for tau in candidates:
        sel = [(s, y) for s, y in zip(scores, lossy_labels) if s >= tau]
        n = len(sel)
        k = sum(1 for _, y in sel if y)
        ucb = hoeffding_ucb(k, n, confidence)
        if ucb <= alpha:
            best = CalibrationResult(
                tau=tau, alpha=alpha, confidence=confidence,
                n_calibration=len(scores), n_at_tau=n,
                lossy_rate_at_tau=k / n if n else 0.0,
                lossy_ucb_at_tau=ucb,
                certified_fraction=n / len(scores),
            )
        else:
            break  # fixed-sequence: first failure ends the walk
    return best


# ── Commit calibration (CCOMMIT's N_COMMIT/COMMIT_DELTA, done properly) ─────────
#
# Diagnostic finding (docs/PAPER_COMPONENTS.md, "Sparse-evidence guard"): CCOMMIT's
# fixed COMMIT_DELTA=0.15 / N_COMMIT=2 were hand-picked for an NLI noise floor that
# no longer applies. Under ROUGE1 attribution they're badly mismatched with the
# actual score distribution (mean attribution ~0.06 on LoCoMo) — 90% of credited
# items never reach commit_count>=2 and stay permanently gated to w_imp=0.
#
# Fix: treat attr(m) as a probabilistic forecast that m is truly load-bearing, and
# calibrate the commit decision the same way a forecaster's calibration is
# assessed (Murphy 1973 Brier decomposition) rather than picking a threshold by
# hand. Two pieces:
#   1. Isotonic (monotonic, PAV) calibration: attr -> P(load-bearing | attr).
#   2. A memory commits once its ACCUMULATED log-likelihood-ratio (not a raw
#      event count) crosses a threshold selected the same RCPS way as
#      calibrate_certify_threshold — most permissive LLR threshold whose
#      Hoeffding UCB on the false-commit rate is <= alpha.
# This makes the commit decision self-adjusting to whichever attribution signal
# feeds it (ROUGE1 vs TRACE-CF) instead of a magic number tuned for neither.

def _pool_adjacent_violators(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    """
    Isotonic regression via PAV: xs sorted ascending, ys the (already
    x-sorted) targets. Returns (x, calibrated_y) breakpoints of the resulting
    monotonically non-decreasing step function. No external deps.
    """
    # Each block: [sum_y, count, x_max] — merge adjacent blocks that violate
    # monotonicity (later block's mean < earlier block's mean).
    blocks: list[list[float]] = []
    for x, y in zip(xs, ys):
        blocks.append([y, 1.0, x])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (blocks[-1][0] / blocks[-1][1]):
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] = b[2]  # extend to the merged block's right edge
    return [(x_max, s / n) for s, n, x_max in blocks]


def _isotonic_lookup(breakpoints: list[tuple[float, float]], x: float) -> float:
    """Step-function lookup: calibrated_p for the first breakpoint >= x."""
    for x_max, p in breakpoints:
        if x <= x_max:
            return p
    return breakpoints[-1][1] if breakpoints else 0.5


def brier_decomposition(scores: list[float], labels: list[bool], n_bins: int = 10) -> dict:
    """
    Murphy (1973) three-term Brier decomposition: BS = reliability - resolution
    + uncertainty. Lower reliability is better (well-calibrated); higher
    resolution is better (informative, not just calibrated-to-the-base-rate).
    This is the diagnostic a single Pearson r can't give: r=0.45 doesn't tell
    you whether a signal is calibrated-but-weak or miscalibrated-but-sharp.
    """
    n = len(scores)
    if n == 0:
        return {"brier_score": None, "reliability": None, "resolution": None,
                "uncertainty": None, "n": 0}
    base_rate = sum(labels) / n
    brier = sum((s - float(y)) ** 2 for s, y in zip(scores, labels)) / n

    edges = [i / n_bins for i in range(n_bins + 1)]
    reliability = 0.0
    resolution = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        bin_pairs = [(s, y) for s, y in zip(scores, labels)
                     if (lo <= s < hi) or (hi == 1.0 and s == 1.0)]
        if not bin_pairs:
            continue
        nk = len(bin_pairs)
        bin_mean_score = sum(s for s, _ in bin_pairs) / nk
        bin_obs_rate = sum(float(y) for _, y in bin_pairs) / nk
        reliability += nk * (bin_mean_score - bin_obs_rate) ** 2
        resolution += nk * (bin_obs_rate - base_rate) ** 2
    reliability /= n
    resolution /= n
    uncertainty = base_rate * (1 - base_rate)
    return {"brier_score": brier, "reliability": reliability,
            "resolution": resolution, "uncertainty": uncertainty,
            "base_rate": base_rate, "n": n}


@dataclass
class CommitCalibration:
    breakpoints: list  # list[(attr_threshold, calibrated_p)], isotonic step function
    llr_threshold: float
    base_rate: float
    alpha: float
    confidence: float
    n_calibration: int
    n_at_threshold: int
    false_commit_rate_at_threshold: float
    false_commit_ucb_at_threshold: float
    brier: dict

    def calibrated_p(self, attr: float) -> float:
        return _isotonic_lookup(self.breakpoints, attr)

    def llr_increment(self, attr: float) -> float:
        """Log-likelihood-ratio of this attribution event vs the base rate,
        clipped to keep any single event from single-handedly crossing the
        commit threshold (mirrors CCOMMIT's original 'N events, not 1' intent)."""
        p = min(max(self.calibrated_p(attr), 1e-4), 1 - 1e-4)
        p0 = min(max(self.base_rate, 1e-4), 1 - 1e-4)
        llr = math.log(p / (1 - p)) - math.log(p0 / (1 - p0))
        return max(-2.0, min(2.0, llr))

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["breakpoints"] = [list(bp) for bp in self.breakpoints]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CommitCalibration":
        d = dict(d)
        d["breakpoints"] = [tuple(bp) for bp in d["breakpoints"]]
        return cls(**d)


def calibrate_commit_threshold(
    attr_label_pairs: list[tuple[float, bool]],
    alpha: float = 0.1,
    confidence: float = 0.9,
    n_bins: int = 10,
) -> Optional[CommitCalibration]:
    """
    Fit the isotonic attr->P(load-bearing) curve, then RCPS-select the most
    permissive cumulative-LLR commit threshold whose Hoeffding UCB on the
    false-commit rate (committing a memory that was NOT truly load-bearing)
    is <= alpha. Mirrors calibrate_certify_threshold's fixed-sequence walk.
    Returns None if not certifiable at this alpha/confidence with this data
    (report honestly, same convention as the merge calibration).
    """
    if not attr_label_pairs:
        raise ValueError("attr_label_pairs must be non-empty")
    scores = [a for a, _ in attr_label_pairs]
    labels = [bool(y) for _, y in attr_label_pairs]
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    xs = [scores[i] for i in order]
    ys = [float(labels[i]) for i in order]
    breakpoints = _pool_adjacent_violators(xs, ys)
    brier = brier_decomposition(scores, labels, n_bins=n_bins)
    base_rate = brier["base_rate"]

    calib = CommitCalibration(
        breakpoints=breakpoints, llr_threshold=0.0, base_rate=base_rate,
        alpha=alpha, confidence=confidence, n_calibration=len(scores),
        n_at_threshold=0, false_commit_rate_at_threshold=0.0,
        false_commit_ucb_at_threshold=0.0, brier=brier,
    )
    llrs = sorted({calib.llr_increment(a) for a in scores}, reverse=True)
    best: Optional[CommitCalibration] = None
    for thr in llrs:
        sel = [(a, y) for a, y in zip(scores, labels) if calib.llr_increment(a) >= thr]
        n = len(sel)
        k = sum(1 for _, y in sel if not y)  # false commits: high LLR but NOT load-bearing
        ucb = hoeffding_ucb(k, n, confidence)
        if ucb <= alpha:
            best = CommitCalibration(
                breakpoints=breakpoints, llr_threshold=thr, base_rate=base_rate,
                alpha=alpha, confidence=confidence, n_calibration=len(scores),
                n_at_threshold=n, false_commit_rate_at_threshold=k / n if n else 0.0,
                false_commit_ucb_at_threshold=ucb, brier=brier,
            )
        else:
            break
    return best


# ── Cluster proposal (embedding + co-cite, zero LLM calls) ──────────────────────

def propose_clusters(
    items: list["MemoryItem"],
    sim_threshold: float = 0.6,
    max_clusters: int = 16,
    max_cluster_size: int = 4,
    co_cite_graph: Optional[dict] = None,
    co_cite_sim_floor: float = 0.45,
    protect_w_imp: float = 1.1,  # >1 disables protection by default at this layer
) -> list[list["MemoryItem"]]:
    """
    Greedy candidate clusters from unit-norm embeddings (dot == cosine), with
    two MEMOIR-specific rules:

    - Co-cite seeding (Bridge 1): a pair with a co-citation edge clusters at the
      lower `co_cite_sim_floor` instead of `sim_threshold` — credit earned
      together is extra redundancy evidence. Below the floor the pair is left
      alone even if co-cited: multi-hop supports (different hops, low cosine)
      must NOT be merged.
    - Credit protection: items with w_imp > protect_w_imp are excluded here;
      the caller separately allows them only under a unanimous certificate.

    Judge calls are bounded by the number of returned clusters (<= max_clusters).
    """
    live = [m for m in items
            if not m.hallucinated and not getattr(m, "pruned", False)
            and m.embedding is not None and m.w.w_imp <= protect_w_imp]
    if len(live) < 2:
        return []

    vecs = np.stack([m.embedding for m in live])  # unit-norm rows

    def _edge(a: "MemoryItem", b: "MemoryItem") -> float:
        if not co_cite_graph:
            return 0.0
        return co_cite_graph.get(tuple(sorted([a.id, b.id])), 0.0)

    used: set[int] = set()
    clusters: list[list["MemoryItem"]] = []
    for i in range(len(live)):
        if i in used or len(clusters) >= max_clusters:
            continue
        group = [i]
        for j in range(i + 1, len(live)):
            if j in used or len(group) >= max_cluster_size:
                continue
            cos = float(vecs[i] @ vecs[j])
            threshold = co_cite_sim_floor if _edge(live[i], live[j]) > 0.0 else sim_threshold
            if cos >= threshold:
                group.append(j)
        if len(group) >= 2:
            used.update(group)
            clusters.append([live[k] for k in group])

    return clusters


# ── Merge text generation ───────────────────────────────────────────────────────

def generate_merge_text(generate_fn: Callable[[str], str],
                        items: list["MemoryItem"]) -> str:
    entries_text = "\n".join(
        f"Entry {i+1}: {m.text}" for i, m in enumerate(items)
    )
    return generate_fn(MERGE_PROMPT.format(entries=entries_text)).strip()


# ── Audit record ────────────────────────────────────────────────────────────────

@dataclass
class ConsolidationOp:
    """One applied (or rejected) consolidation operation — the delta/rho unit."""
    op_type: str                      # "merge" | "prune"
    applied: bool
    constituent_ids: list[str]        # merged-away items / pruned item
    result_id: Optional[str]          # merged item id / kept entailer id
    judge_score: float
    retrieval_coverage: float
    constituent_texts: list[str] = field(default_factory=list)
    result_text: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "op_type": self.op_type, "applied": self.applied,
            "constituent_ids": self.constituent_ids, "result_id": self.result_id,
            "judge_score": self.judge_score,
            "retrieval_coverage": self.retrieval_coverage,
            "constituent_texts": self.constituent_texts,
            "result_text": self.result_text, "timestamp": self.timestamp,
        })


# ── Credit transfer (Bridge 2) ──────────────────────────────────────────────────

def transfer_credit(merged: "MemoryItem", constituents: list["MemoryItem"],
                    co_cite_graph: Optional[dict], mode: str = "max") -> None:
    """
    Move retrospective state onto the merged item.

    w_imp(z) = max_i w_imp(x_i) (mode="max", conservative — sum overcounts;
    mode="zero" is the Bridge-2 ablation arm). Evidence dia_ids of ALL
    constituents are preserved as a list so evidence-based rec@k stays
    well-defined after merging. Co-cite edges are re-keyed onto the merged id
    (per-neighbor max over constituent edges).

    Sound only under a truly lossless merge (Thm 1 corollary — the soundness
    lemma in the paper); under a lossy merge the transferred credit is
    corrupted, which is exactly what the Bridge-2 experiment measures.
    """
    if mode == "max":
        merged.w.w_imp = max((m.w.w_imp for m in constituents), default=0.0)
    elif mode == "zero":
        merged.w.w_imp = 0.0
    else:
        raise ValueError(f"unknown credit transfer mode: {mode!r}")

    dia_ids: list[str] = []
    for m in constituents:
        meta = m.metadata or {}
        own = meta.get("dia_id")
        inherited = meta.get("dia_ids", [])
        for d in ([own] if own else []) + list(inherited):
            if d not in dia_ids:
                dia_ids.append(d)
    if dia_ids:
        merged.metadata["dia_ids"] = dia_ids
        # keep singular key pointing at the first for code paths reading dia_id
        merged.metadata.setdefault("dia_id", dia_ids[0])
    merged.metadata["merged_from"] = [m.id for m in constituents]

    if co_cite_graph:
        cids = {m.id for m in constituents}
        new_edges: dict = {}
        for (a, b), w in list(co_cite_graph.items()):
            ends = {a, b}
            hit = ends & cids
            if not hit:
                continue
            outside = (ends - cids)
            del co_cite_graph[(a, b)]
            if not outside:
                continue  # edge internal to the merged cluster — absorbed
            other = outside.pop()
            key = tuple(sorted([merged.id, other]))
            new_edges[key] = max(new_edges.get(key, 0.0), w)
        for key, w in new_edges.items():
            co_cite_graph[key] = max(co_cite_graph.get(key, 0.0), w)


# ── SUPERSEDE: certified retirement of contradicted facts ──────────────────────
# The missing lifecycle operator (2026 curation critique: "temporal versioning,
# contradiction detection" — Mem0/Zep carry temporal validity; our operator set
# was {add, merge} only). Reuses TRIAGE-α's negation-overlap conflict signal
# (previously computed into w_conf and never read anywhere downstream) as the
# candidate proposal, and the same judge-ensemble certification as merge.

SUPERSEDE_PROMPT = """You decide whether a NEWER memory entry supersedes an OLDER one. \
"Supersedes" means the newer entry updates, corrects, or contradicts the older entry \
about the SAME fact, so the older entry is now outdated and should no longer be used.

Older entry (recorded earlier):
{older}

Newer entry (recorded later):
{newer}

The verdict is SUPERSEDES only if they are about the same fact AND the newer entry \
makes the older one outdated. If they describe different facts, or both can be true \
at once (e.g. an additional detail, a different event), the verdict is INDEPENDENT.
Answer with exactly one word on the first line: SUPERSEDES or INDEPENDENT.
Answer:"""

SUPERSEDE_PROMPT_TIMELINE = """Two memory entries from the same ongoing conversation are \
shown with their order. Determine if the LATER one replaces the EARLIER one.

Earlier: {older}
Later: {newer}

Think of a timeline of facts: does the later entry change the truth of the earlier one \
(a correction, a reversal, an update to a value/status/plan)? If yes answer SUPERSEDES. \
If the earlier entry remains true alongside the later one, answer INDEPENDENT.
Answer with exactly one word on the first line: SUPERSEDES or INDEPENDENT.
Answer:"""

SUPERSEDE_PROMPT_ADVERSARIAL = """Your job is to protect a memory store from losing valid \
information. Someone proposes deleting the OLD entry because the NEW entry allegedly \
replaces it.

OLD: {older}
NEW: {newer}

Argue against the deletion: is there ANY reading where the old entry is still true and \
useful alongside the new one? If you find one, answer INDEPENDENT. Only if the old entry \
is genuinely contradicted or made obsolete by the new one, answer SUPERSEDES.
Answer with exactly one word on the first line: SUPERSEDES or INDEPENDENT.
Answer:"""

_SUPERSEDE_VARIANTS = (
    SUPERSEDE_PROMPT,
    SUPERSEDE_PROMPT_TIMELINE,
    SUPERSEDE_PROMPT_ADVERSARIAL,
)

# TRIAGE-α's negation lexicon (triage.py) — the same syntactic conflict signal,
# now applied pairwise at sleep time instead of only scalar-at-write-time.
_NEGATION = frozenset({
    "not", "no", "never", "isn't", "aren't", "wasn't", "weren't",
    "doesn't", "didn't", "won't", "can't", "cannot", "incorrect",
    "false", "wrong", "actually", "correction", "mistaken", "untrue",
})

SUPERSEDE_SIM_THRESHOLD: float = 0.85     # same as TRIAGE's conflict cosine
SUPERSEDE_CREDIT_DISCOUNT: float = 0.5    # partial transfer: content CHANGED
                                          # (unlike lossless merge, where max
                                          # transfer is provably sound)


def _word_set(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\b\w+\b", text.lower()))


def is_conflict_pair(a_text: str, b_text: str) -> bool:
    """Same content tokens AND at least one side carries a negation marker —
    TRIAGE-α's pair test, factored out for supersede candidate proposal.
    Content overlap excludes stopwords (unlike TRIAGE's scalar version) —
    sharing only "is"/"the" is not evidence two memories discuss one fact."""
    a_tok, b_tok = _word_set(a_text), _word_set(b_text)
    skip = _NEGATION | _STOPWORDS
    content_overlap = (a_tok - skip) & (b_tok - skip)
    return bool(content_overlap and ((a_tok & _NEGATION) or (b_tok & _NEGATION)))


def propose_supersede_pairs(
    items: list["MemoryItem"],
    sim_threshold: float = SUPERSEDE_SIM_THRESHOLD,
    max_pairs: int = 8,
) -> list[tuple["MemoryItem", "MemoryItem"]]:
    """
    (older, newer) candidate pairs: embedding-close (cosine >= threshold),
    negation-conflict-flagged, ordered oldest-first by created_at. Bounded by
    max_pairs (hard cap on judge calls, same discipline as merge clusters).
    """
    with_emb = [m for m in items if m.embedding is not None]
    if len(with_emb) < 2:
        return []
    vecs = np.stack([m.embedding for m in with_emb])
    sims = vecs @ vecs.T
    pairs: list[tuple[float, "MemoryItem", "MemoryItem"]] = []
    for i in range(len(with_emb)):
        for j in range(i + 1, len(with_emb)):
            if sims[i, j] < sim_threshold:
                continue
            a, b = with_emb[i], with_emb[j]
            if not is_conflict_pair(a.text, b.text):
                continue
            older, newer = (a, b) if a.created_at <= b.created_at else (b, a)
            if older.created_at == newer.created_at:
                continue  # no temporal direction — nothing to supersede
            pairs.append((float(sims[i, j]), older, newer))
    pairs.sort(key=lambda t: -t[0])
    return [(o, n) for _, o, n in pairs[:max_pairs]]


def judge_supersedes(judge: "FaithfulnessJudge", older: str, newer: str) -> float:
    """SUPERSEDES vote fraction over the 3 prompt framings. Conservative like
    the merge judge: any parse failure/exception counts as INDEPENDENT (the
    non-destructive verdict)."""
    votes = 0
    n = min(judge.n_votes, len(_SUPERSEDE_VARIANTS))
    for prompt_tpl in _SUPERSEDE_VARIANTS[:n]:
        try:
            response = judge.generate_fn(
                prompt_tpl.format(older=older, newer=newer))
        except Exception:
            continue
        first = ""
        for line in (response or "").strip().splitlines():
            if line.strip():
                first = line.strip().upper()
                break
        token = re.split(r"[^A-Z]", first, 1)[0] if first else ""
        if token == "SUPERSEDES":
            votes += 1
    return votes / max(1, n)


def transfer_supersede_credit(newer: "MemoryItem", older: "MemoryItem",
                              co_cite_graph: Optional[dict]) -> None:
    """
    Partial credit migration onto the superseding item. Unlike merge's max
    rule (provably sound under a lossless rewrite), supersession CHANGES the
    fact — the old item's earned credit is evidence its *topic* is load-
    bearing, not that its (now outdated) content is. Discounted transfer keeps
    the topic retrievable without letting stale credit dominate.
    """
    newer.w.w_imp = max(newer.w.w_imp,
                        SUPERSEDE_CREDIT_DISCOUNT * older.w.w_imp)
    meta = newer.metadata
    meta.setdefault("supersedes", []).append(older.id)

    if co_cite_graph:
        moved: dict = {}
        for (a, b), w in list(co_cite_graph.items()):
            if older.id not in (a, b):
                continue
            other = b if a == older.id else a
            del co_cite_graph[(a, b)]
            if other == newer.id:
                continue
            key = tuple(sorted([newer.id, other]))
            moved[key] = max(moved.get(key, 0.0),
                             SUPERSEDE_CREDIT_DISCOUNT * w)
        for key, w in moved.items():
            co_cite_graph[key] = max(co_cite_graph.get(key, 0.0), w)


# ── DIGEST: certified session-level synthesis (indirect-answerability) ─────────
# Targets a diagnosed generation-time weakness (docs/COMBINED_PAPER_PLAN.md,
# 2026-07-20 entry): raw dialogue-turn context makes it hard for the answering
# LLM to extract facts that require connecting information across turns, even
# when retrieval finds the right turns (ctx@5 was already strong on the
# affected category — this is a generation problem, not a retrieval one).
# LiCoMemory and RMM (arXiv:2503.08026) both address this with session
# summaries / topic-granularity memory, but neither certifies the synthesis
# step, and LiCoMemory pays for it with several sequential LLM calls PER
# QUERY at retrieval time. DIGEST stays training-free/certified (a digest
# unsupported by its sources is discarded, never applied — unlike RMM's
# uncertified heuristic summarization) and additive-only: unlike merge,
# constituents are NEVER pruned, so raw turns stay retrievable for precision
# and TRACE-CF's per-memory credit signal doesn't collapse the way it did
# under summarize_all_to_core's whole-store destructive collapse (r_post→0.0,
# docs/PAPER_ARCHITECTURE.md). All LLM calls happen only in this occasional
# sleep-phase pass, never in the hot per-query retrieval path.

DIGEST_PROMPT = """\
You are extracting discrete facts and relationships from a session's memory entries, so \
each fact is independently answerable rather than buried in a paragraph. This is structured \
extraction, not prose summarization -- similar to writing out (subject, relation, object) \
facts in plain language, one per line.

Rules:
- Write one short factual sentence per line for each fact or relationship you can support,
  e.g. "Grandma gave Caroline a necklace." or "Caroline is from Sweden."
- Include facts that require connecting information across multiple entries below (e.g. who
  did what for whom, who gave/received what, who is related to whom), not just facts
  restating a single entry
- Do not invent any fact, name, number, date, or relation that isn't supported by the
  entries
- Prefer named entities and concrete relations over vague restatements

Entries:
{entries}

Write the facts (one short sentence per line):"""

DIGEST_FAITHFUL_PROMPT = """You verify whether a DIGEST is faithful to its SOURCE memory \
entries. This is a hallucination check, not a completeness check — the digest does not \
need to restate every source detail, but everything IT states must be supported by \
(stated in or directly inferable from) the sources.

Source entries:
{originals}

Digest:
{merged}

Answer with exactly one word on the first line: FAITHFUL or UNFAITHFUL.
Answer:"""

DIGEST_FAITHFUL_PROMPT_ADVERSARIAL = """Your job is to FIND A FABRICATION. Below are \
source memory entries and a digest that summarizes them. Search hard for ANY claim in the \
digest — a fact, name, number, date, or relation — that is NOT supported by the sources, \
even indirectly.

Source entries:
{originals}

Digest:
{merged}

If you found even one unsupported claim, the digest is UNFAITHFUL. Only if every claim in \
the digest is genuinely supported is it FAITHFUL. Answer with exactly one word on the \
first line: FAITHFUL or UNFAITHFUL.
Answer:"""

_DIGEST_FAITHFUL_VARIANTS = (
    DIGEST_FAITHFUL_PROMPT,
    DIGEST_FAITHFUL_PROMPT_ADVERSARIAL,
)


def propose_session_groups(
    items: list["MemoryItem"],
    min_group_size: int = 2,
    max_groups: int = 8,
) -> list[list["MemoryItem"]]:
    """
    Group live, non-digest items by metadata["session"] (already set at write
    time by every eval harness — see eval_locomo.py/eval_qa_together.py
    system_add call sites). Zero LLM calls. Groups smaller than
    min_group_size are dropped (nothing to synthesize). Bounded by max_groups
    (judge-call budget, same discipline as propose_clusters/propose_supersede_pairs).
    """
    live = [m for m in items
            if not m.hallucinated and not getattr(m, "pruned", False)
            and (m.metadata or {}).get("kind") != "digest"
            and not (m.metadata or {}).get("digested")]
    by_session: dict = {}
    for m in live:
        sess = (m.metadata or {}).get("session")
        if sess is None:
            continue
        by_session.setdefault(sess, []).append(m)
    groups = [g for g in by_session.values() if len(g) >= min_group_size]
    return groups[:max_groups]


def generate_digest_text(generate_fn: Callable[[str], str],
                         items: list["MemoryItem"]) -> str:
    entries_text = "\n".join(
        f"Entry {i+1}: {m.text}" for i, m in enumerate(items)
    )
    return generate_fn(DIGEST_PROMPT.format(entries=entries_text)).strip()


def judge_digest_faithful(judge: "FaithfulnessJudge", digest_text: str,
                          originals: list[str]) -> float:
    """FAITHFUL vote fraction over the digest-specific prompt framings.
    Conservative like the merge/supersede judges: any parse failure/exception
    counts as UNFAITHFUL (the non-applying verdict)."""
    originals_text = "\n".join(f"- {o}" for o in originals)
    votes = 0
    n = min(judge.n_votes, len(_DIGEST_FAITHFUL_VARIANTS))
    for prompt_tpl in _DIGEST_FAITHFUL_VARIANTS[:n]:
        try:
            response = judge.generate_fn(
                prompt_tpl.format(originals=originals_text, merged=digest_text))
        except Exception:
            continue
        first = ""
        for line in (response or "").strip().splitlines():
            if line.strip():
                first = line.strip().upper()
                break
        token = re.split(r"[^A-Z]", first, 1)[0] if first else ""
        if token == "FAITHFUL":
            votes += 1
    return votes / max(1, n)


def transfer_digest_credit(digest: "MemoryItem", constituents: list["MemoryItem"],
                           mode: str = "max") -> None:
    """
    Additive credit assignment for a digest item — unlike transfer_credit
    (merge), constituents are NOT pruned; this only sets the digest's own
    w_imp/provenance so it participates in retrieval and audit like any other
    item. metadata["digest_of"] records provenance (parallel to merge's
    metadata["merged_from"]) without removing the sources it was built from.
    """
    if mode == "max":
        digest.w.w_imp = max((m.w.w_imp for m in constituents), default=0.0)
    elif mode == "zero":
        digest.w.w_imp = 0.0
    else:
        raise ValueError(f"unknown credit transfer mode: {mode!r}")

    dia_ids: list[str] = []
    for m in constituents:
        meta = m.metadata or {}
        own = meta.get("dia_id")
        inherited = meta.get("dia_ids", [])
        for d in ([own] if own else []) + list(inherited):
            if d not in dia_ids:
                dia_ids.append(d)
    if dia_ids:
        digest.metadata["dia_ids"] = dia_ids
        digest.metadata.setdefault("dia_id", dia_ids[0])
    digest.metadata["kind"] = "digest"
    digest.metadata["digest_of"] = [m.id for m in constituents]
    # Mark constituents covered so a later consolidate() call doesn't
    # re-propose the same session group again (propose_session_groups skips
    # metadata["digested"]) -- items stay live/retrievable, only excluded
    # from future digest candidacy.
    for m in constituents:
        m.metadata["digested"] = True


# ── Token-matched baseline: summarize-all (MemGPT/Letta-style) ─────────────────

SUMMARIZE_ALL_PROMPT = """\
Summarize the following memory entries into a compact set of notes that preserves the \
important facts, names, numbers, dates, and relations. Be concise but do not invent \
information that is not present below.

Entries:
{entries}

Write the summary:"""

# Rough char budget per summarize_all call — un-gated (no judge, no certificate),
# so this stays cheap: a single LLM call per sleep regardless of store size.
_SUMMARIZE_CHAR_BUDGET = 6000


def summarize_all_to_core(store, generate_fn: Callable[[str], str]) -> dict:
    """
    MemGPT/Letta-style baseline: replace ALL live items with a single LLM
    summary, unconditionally (no certificate, no candidate selection — the
    uncertified extreme of the compression axis). Uses MemoirStore's existing
    reversible pruned=True mechanism, so `unprune_all()` still recovers the
    original store. Returns the same stats shape as MemoirStore.consolidate()
    for drop-in use in eval_consolidation.py's sleep hook.
    """
    from .schema import MemoryItem, WVector

    live = [store._items[i] for i in store._insertion_order
            if i in store._items
            and not store._items[i].pruned and not store._items[i].hallucinated]
    tokens_before = store.total_tokens()
    if not live:
        return {"applied_merges": 0, "applied_prunes": 0,
                "tokens_before": tokens_before, "tokens_after": tokens_before,
                "judge_calls": 0}

    entries_text = "\n".join(f"Entry {i+1}: {m.text}" for i, m in enumerate(live))
    if len(entries_text) > _SUMMARIZE_CHAR_BUDGET:
        entries_text = entries_text[:_SUMMARIZE_CHAR_BUDGET]
    try:
        summary = generate_fn(SUMMARIZE_ALL_PROMPT.format(entries=entries_text)).strip()
    except Exception:
        return {"applied_merges": 0, "applied_prunes": 0,
                "tokens_before": tokens_before, "tokens_after": tokens_before,
                "judge_calls": 0}
    if not summary:
        return {"applied_merges": 0, "applied_prunes": 0,
                "tokens_before": tokens_before, "tokens_after": tokens_before,
                "judge_calls": 0}

    embedding = store._embedder.encode(summary)
    core = MemoryItem(
        text=summary, embedding=embedding,
        w=WVector(w_rel=max((m.w.w_rel for m in live), default=0.0),
                  w_imp=max((m.w.w_imp for m in live), default=0.0)),
        created_at=max(m.created_at for m in live),
        metadata={"dia_ids": [d for m in live
                              for d in ([  # union of all evidence ids, dedup
                                  (m.metadata or {}).get("dia_id"),
                                  *((m.metadata or {}).get("dia_ids") or []),
                              ]) if d]},
    )
    for m in live:
        m.pruned = True
    store._items[core.id] = core
    store._insertion_order.append(core.id)
    store._texts_for_bm25.append(summary)
    store._matrix_dirty = True
    store._bm25_dirty = True
    store.consolidation_ops.append(ConsolidationOp(
        op_type="merge", applied=True,
        constituent_ids=[m.id for m in live], result_id=core.id,
        judge_score=1.0, retrieval_coverage=1.0,
        constituent_texts=[m.text for m in live], result_text=summary,
    ))

    return {"applied_merges": 1, "applied_prunes": 0,
            "tokens_before": tokens_before, "tokens_after": store.total_tokens(),
            "judge_calls": 1}


# ── CRPC: Counterfactual Retrieval-Preservation Certificate ────────────────────
# Retriever-in-the-loop alternative to `retrieval_token_coverage`'s lexical
# proxy: instead of asking "does the merged text still contain the source's
# tokens", CRPC asks "does the merged text still WIN the retrievals the
# sources used to win" — replayed through the exact deployed RECALL-Q stack
# against a temporary, throwaway counterfactual store (the real store is never
# touched). Merge-only scope (same as the rest of this module); no LLM calls.

# ── 1. Deterministic anchor extraction (no LLM, no spaCy/NLTK download — the
#      repo has neither a cached NER extractor nor a downloaded NLTK corpus,
#      so this stays regex/lexicon-based, same spirit as retrieval_token_coverage
#      and triage.py's negation lexicon, which it reuses directly) ────────────

from .triage import _NEGATION as _ANCHOR_NEGATIONS  # noqa: E402  (reuse, don't duplicate)

_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december",
           "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
           "oct", "nov", "dec")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")
_RELATIVE_DATES = ("yesterday", "today", "tomorrow", "tonight", "week", "month",
                    "year", "ago", "last", "next")
# Capitalized function words that should NOT count as entities even though
# English capitalizes them at a sentence start (a positional heuristic like
# "exclude every sentence-initial word" would also wrongly drop genuine
# single-mention proper nouns in short merged sentences -- e.g. "Caroline
# attended..." -- so exclude by word identity instead).
_ENTITY_STOPWORDS = {
    "i", "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "we", "they", "she", "he", "his", "her", "their", "our", "your", "you",
    "there", "here", "what", "when", "where", "who", "why", "how", "speaker",
}


def _extract_entities(text: str) -> set[str]:
    """Capitalized-word-span proxy for named entities, excluding common
    capitalized function words (see _ENTITY_STOPWORDS) rather than filtering
    by sentence position -- position-based filtering drops genuine
    single-mention proper nouns at a sentence's start."""
    caps = set(re.findall(r"\b[A-Z][a-zA-Z]+\b", text))
    return {w.lower() for w in caps if w.lower() not in _ENTITY_STOPWORDS}


def _extract_dates(text: str) -> set[str]:
    out: set[str] = set()
    low = text.lower()
    for tok in re.findall(r"\b\w+\b", low):
        if tok in _MONTHS or tok in _WEEKDAYS or tok in _RELATIVE_DATES:
            out.add(tok)
    # explicit date-like patterns: "May 7", "7 May", "2023", "5/7/2023"
    out.update(m.group(0).lower() for m in re.finditer(
        r"\b\d{4}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", text))
    return out


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def _extract_negations(text: str) -> set[str]:
    tokens = set(re.findall(r"\b\w+'?\w*\b", text.lower()))
    return tokens & _ANCHOR_NEGATIONS


def _extract_speakers(text: str) -> set[str]:
    """LoCoMo turns are stored as 'Speaker <name> says: ...' (eval_locomo.py's
    system_add) — a cheap, exact, format-specific signal for who-said-what
    that's worth more than the generic entity heuristic above when present."""
    return {m.lower() for m in re.findall(r"[Ss]peaker\s+(\w+)", text)}


def extract_anchors(text: str) -> dict[str, set[str]]:
    """All anchor categories for one piece of text, keyed by category name
    (entities/dates/numbers/negations/speakers) so callers (and the C1-C4
    component ablations) can include/exclude categories independently."""
    return {
        "entities": _extract_entities(text),
        "dates": _extract_dates(text),
        "numbers": _extract_numbers(text),
        "negations": _extract_negations(text),
        "speakers": _extract_speakers(text),
    }


def anchor_recall(originals: list[str], merged: str,
                  categories: Optional[list[str]] = None) -> float:
    """Fraction of the union of source anchors (across `categories`, default
    all five) still present in the merged text. categories=None means every
    category; pass a subset for the C1-C4 component ablations (e.g. drop
    "entities" for C1, drop "negations" for C3)."""
    cats = categories or ["entities", "dates", "numbers", "negations", "speakers"]
    source_anchors: set[str] = set()
    for o in originals:
        cats_o = extract_anchors(o)
        for c in cats:
            source_anchors |= cats_o.get(c, set())
    if not source_anchors:
        return 1.0
    merged_cats = extract_anchors(merged)
    merged_anchors: set[str] = set()
    for c in cats:
        merged_anchors |= merged_cats.get(c, set())
    return len(source_anchors & merged_anchors) / len(source_anchors)


# ── 2. Historical probe-query collection ────────────────────────────────────

def collect_probe_queries(
    query_log: list[dict],
    cluster_ids: set[str],
    min_attr: float = 0.15,
    max_probes: int = 10,
) -> list[dict]:
    """Historical queries where any cluster member was retrieved, deduplicated
    by query text (last occurrence wins, matching this project's dedup
    convention elsewhere — significance.py, eval_locomo.py's _aggregate).
    Prefers (sorts first) queries where some cluster member's TRACE
    attribution exceeded `min_attr` ("credited" probes), then fills remaining
    slots with uncredited touches up to max_probes — per spec: "prefer",
    not "filter to only"."""
    by_query: dict[str, dict] = {}
    for entry in query_log:
        touched = cluster_ids & set(entry.get("retrieved_ids", []))
        if not touched:
            continue
        attrs = entry.get("attributions", {})
        max_attr = max((attrs.get(mid, 0.0) for mid in touched), default=0.0)
        by_query[entry["query"]] = {
            "query": entry["query"],
            "touched_ids": touched,
            "ranks": entry.get("ranks", {}),
            "max_attr": max_attr,
            "credited": max_attr >= min_attr,
            "now": entry.get("now"),
        }
    probes = list(by_query.values())
    probes.sort(key=lambda p: (-int(p["credited"]), -p["max_attr"]))
    return probes[:max_probes]


# ── 3. Counterfactual replay ─────────────────────────────────────────────────

def _replay(query: str, items: list["MemoryItem"], store, k: int,
           now: Optional[float],
           co_cite_graph: Optional[dict] = None) -> list[tuple["MemoryItem", float]]:
    """One RECALL-Q pass against an arbitrary (possibly temporary) item list,
    using the store's live scoring hyperparameters. `peek=True` so replay
    never bumps retrieval_count on the real (shared-by-reference) items --
    this function is read-only regardless of which item list it's given."""
    from .bm25 import BM25Okapi
    from .recall import recall_q_with_bm25
    valid = [it for it in items if it.embedding is not None]
    if not valid:
        return []
    matrix = np.stack([it.embedding for it in valid])
    tokenized = [re.findall(r"\w+", it.text.lower()) for it in valid]
    bm25 = BM25Okapi(tokenized)
    q_vec = store._embedder.encode(query)
    return recall_q_with_bm25(
        query=query, query_vec=q_vec, store_items=valid,
        embeddings_matrix=matrix, bm25=bm25, k=k,
        alpha=store._alpha_bm25, lam=store._lambda_mmr,
        beta_floor=store._beta_floor, gamma_imp=store._gamma_imp,
        delta_rec=store._delta_rec, now=now, peek=True,
        co_cite_graph=co_cite_graph,
    )


def crpc_replay(
    store, cluster: list["MemoryItem"], candidate: "MemoryItem",
    probes: list[dict], k: int = 5,
    post_co_cite_graph: Optional[dict] = None,
) -> list[dict]:
    """Pre- and post-merge replay for every probe, against the CURRENT live
    store (Def. "replay against the current store" -- not the stale ranks
    logged at original query time, since the store may have evolved since).
    Never mutates `store` -- both replays run on freshly assembled temporary
    item lists; `candidate` is only ever spliced into the temporary post list.

    post_co_cite_graph: a SCRATCH co-citation graph (already redirected onto
    `candidate` by the caller via transfer_credit on a copy of the real
    graph -- spec sec. 3 "redirect co-citation edges exactly as the real
    merge would... do not modify the persistent store"). The pre-merge
    replay uses the store's real, unmodified graph.
    """
    cluster_ids = {m.id for m in cluster}
    live = store.all_items()
    base = [it for it in live if it.id not in cluster_ids]  # store minus cluster
    pre_items = base + cluster       # current store, unmodified
    post_items = base + [candidate]  # counterfactual: cluster -> candidate
    pre_graph = store._co_cite_graph if getattr(store, "_use_co_cite", False) else None

    rows: list[dict] = []
    for probe in probes:
        pre = _replay(probe["query"], pre_items, store, k, probe.get("now"),
                      co_cite_graph=pre_graph)
        post = _replay(probe["query"], post_items, store, k, probe.get("now"),
                       co_cite_graph=post_co_cite_graph)

        pre_ranks = {it.id: r for r, (it, _s) in enumerate(pre)}
        pre_scores = {it.id: s for it, s in pre}
        post_ranks = {it.id: r for r, (it, _s) in enumerate(post)}
        post_scores = {it.id: s for it, s in post}

        # "missing top-k items assigned k+1" (spec sec. 5)
        pre_rank = min((pre_ranks.get(mid, k + 1) for mid in probe["touched_ids"]),
                       default=k + 1)
        post_rank = post_ranks.get(candidate.id, k + 1)
        max_pre_cluster_score = max(
            (pre_scores.get(mid, 0.0) for mid in probe["touched_ids"]), default=0.0)
        post_candidate_score = post_scores.get(candidate.id, 0.0)

        rows.append({
            "query": probe["query"],
            "pre_rank": pre_rank,
            "post_rank": post_rank,
            "retained_top_k": post_rank < k,
            "score_ratio": (post_candidate_score / max_pre_cluster_score
                           if max_pre_cluster_score > 0 else 0.0),
            "credited": probe["credited"],
            "n_touched": len(probe["touched_ids"]),
        })
    return rows


# ── 4. Metrics from a replay ─────────────────────────────────────────────────

def crpc_metrics(replay_rows: list[dict]) -> dict:
    """query_retention, full_evidence_retention, mean_rank_shift, score_ratio
    -- spec sec. 5. full_evidence_retention here means "the candidate stayed
    top-k for every probe that had touched MULTIPLE cluster members at once"
    (the multi-hop case: transfer_credit's dia_id union means one surviving
    top-k hit already carries every hop's evidence id, so retention on those
    probes specifically is the honest multi-hop-safety signal)."""
    if not replay_rows:
        return {"query_retention": 1.0, "full_evidence_retention": 1.0,
               "mean_rank_shift": 0.0, "mean_score_ratio": 1.0, "n_probes": 0}
    n = len(replay_rows)
    retained = sum(1 for r in replay_rows if r["retained_top_k"])
    multi_hop_rows = [r for r in replay_rows if r.get("n_touched", 1) > 1]
    full_ev = (sum(1 for r in multi_hop_rows if r["retained_top_k"]) / len(multi_hop_rows)
              if multi_hop_rows else 1.0)
    return {
        "query_retention": retained / n,
        "full_evidence_retention": full_ev,
        "mean_rank_shift": sum(r["post_rank"] - r["pre_rank"] for r in replay_rows) / n,
        "mean_score_ratio": sum(r["score_ratio"] for r in replay_rows) / n,
        "n_probes": n,
    }


# ── 5. Full certificate: gate decision + diagnostics for every mode ─────────

CRPC_MODES = ("none", "lexical", "anchor", "crpc", "hybrid")


def crpc_certificate(
    store, cluster: list["MemoryItem"], candidate: "MemoryItem",
    merged_text: str, originals: list[str],
    mode: str = "lexical",
    tau_lex: float = 0.45,
    tau_anchor: float = 0.9,
    tau_query: float = 0.8,
    min_probes: int = 3,
    max_probes: int = 10,
    min_attr: float = 0.15,
    k: int = 5,
    credit_transfer: str = "max",
    anchor_categories: Optional[list[str]] = None,
) -> dict:
    """Evaluate the retrieval-preservation gate under one of CRPC_MODES.
    Returns a dict with the pass/fail decision plus every measured diagnostic
    (so a rejected candidate's audit log always carries the numbers, not just
    the verdict -- spec sec. 8). Read-only: never mutates `store` or its real
    `_co_cite_graph` -- CRPC modes preview the candidate's post-merge credit
    (w_imp, dia_ids, co-cite redirect) on `candidate` itself and a SCRATCH
    copy of the graph; the caller (store.consolidate()) re-applies
    transfer_credit against the real graph only after the gate accepts.

      none    -- always passes (semantic certificate + length are the only gates)
      lexical -- current baseline: kappa_lex(merged, originals) >= tau_lex
      anchor  -- anchor_recall(originals, merged) >= tau_anchor
      crpc    -- query_retention >= tau_query AND anchor_recall >= tau_anchor,
                 when |Q_C| >= min_probes; sparse clusters (< min_probes) are
                 REJECTED outright (the strict/A4 reading of the spec's rule)
      hybrid  -- same as crpc, but sparse clusters fall back to the lexical
                 gate instead of being rejected (spec's literal "Full CRPC
                 application rule", sec. 7 -- the A5 reading)

    anchor_categories: subset of {"entities","dates","numbers","negations",
    "speakers"} to check (None = all five) -- the C1-C4 component ablations
    remove one category at a time.
    """
    kappa_lex = retrieval_token_coverage(merged_text, originals)
    a_recall = anchor_recall(originals, merged_text, categories=anchor_categories)

    needs_replay = mode in ("crpc", "hybrid")
    cluster_ids = {m.id for m in cluster}
    probes: list[dict] = []
    sparse = True
    replay_rows: list[dict] = []
    metrics = {"query_retention": None, "full_evidence_retention": None,
              "mean_rank_shift": None, "mean_score_ratio": None, "n_probes": 0}
    if needs_replay:
        probes = collect_probe_queries(store._query_log, cluster_ids,
                                       min_attr=min_attr, max_probes=max_probes)
        sparse = len(probes) < min_probes
        if probes:
            use_co_cite = getattr(store, "_use_co_cite", False)
            scratch_graph = dict(store._co_cite_graph) if use_co_cite else None
            # Preview the candidate's post-merge credit exactly as the real
            # merge would apply it (w_imp, dia_ids, co-cite redirect) -- on
            # `candidate` and the SCRATCH graph copy only.
            transfer_credit(candidate, cluster, co_cite_graph=scratch_graph,
                            mode=credit_transfer)
            replay_rows = crpc_replay(store, cluster, candidate, probes, k=k,
                                      post_co_cite_graph=scratch_graph)
            metrics = crpc_metrics(replay_rows)
        metrics["n_probes"] = len(probes)

    if mode == "none":
        passed, reason = True, "no_retrieval_guard"
    elif mode == "lexical":
        passed = kappa_lex >= tau_lex
        reason = "lexical" if passed else "lexical_below_tau"
    elif mode == "anchor":
        passed = a_recall >= tau_anchor
        reason = "anchor" if passed else "anchor_below_tau"
    elif mode == "crpc":
        if sparse:
            passed, reason = False, "sparse_history_reject"
        else:
            passed = (metrics["query_retention"] >= tau_query
                      and a_recall >= tau_anchor)
            reason = "crpc" if passed else "crpc_below_tau"
    elif mode == "hybrid":
        if sparse:
            passed = kappa_lex >= tau_lex
            reason = "lexical_fallback" if passed else "lexical_fallback_below_tau"
        else:
            passed = (metrics["query_retention"] >= tau_query
                      and a_recall >= tau_anchor)
            reason = "crpc" if passed else "crpc_below_tau"
    else:
        raise ValueError(f"unknown retrieval guard mode: {mode!r}")

    return {
        "passed": passed, "reason": reason, "mode": mode,
        "kappa_lex": kappa_lex, "anchor_recall": a_recall,
        "sparse_history": sparse,
        **metrics,  # includes n_probes
        "probes": [p["query"] for p in probes],
        "replay_rows": replay_rows,
    }
