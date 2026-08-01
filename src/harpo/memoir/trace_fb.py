"""
TRACE-FB: Outcome-conditioned memory importance update.

Core contribution of MEMOIR. Updates w_imp for memories based on how much
of their content appeared in generated answers — a retrospective attribution
signal requiring no ground-truth labels.

Attribution (two-path, fast-first):
  Path 1 — ROUGE1-recall (default, always runs):
    attribution(m, answer) = |tokens(m.text) ∩ tokens(answer)| / |tokens(m.text)|

  Path 2 — LLM fallback (opt-in, use_llm_attr=True):
    Fires when ROUGE1 < ROUGE1_THRESHOLD AND w_imp < LLM_ATTR_W_IMP_GATE.
    Asks claude-haiku-4-5: "did this memory help answer the question?"
    Correct for numeric/count/temporal answers where ROUGE1 is structurally 0.
    Batched: all blind memories in one asyncio.gather() call per feedback invocation.

Why ROUGE1-recall (not Jaccard):
  Asymmetric — bounded by memory size, not answer length.
  A long answer credits a short memory without score dilution.

w_imp update rule:
  delta = alpha × attribution(m, answer) × (1 / (rank + 1))
  w_imp ← clip(w_imp + delta, 0, 1)

Hallucination reset:
  Memory contributed to a hallucination step → w_imp reset to 0.0.

Oracle TRACE (upper bound, requires GT):
  Replace attribution with 1.0 if answer_f1 > 0.5 else 0.0.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoirStore


# ── Token utilities ────────────────────────────────────────────────────────────

_STOP = frozenset({
    # Articles, prepositions, conjunctions
    "a", "an", "the", "of", "in", "on", "at", "to", "by", "for",
    "with", "as", "and", "or", "but", "if", "so", "not",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their",
    # Aux verbs (all common forms)
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "can", "let",
    # Demonstratives and wh-words
    "that", "this", "what", "how", "when", "where", "who", "why",
    # Common high-frequency words that cause false-positive attribution
    "just", "also", "then", "now", "here", "there", "no", "all",
    "said", "than", "more", "new", "like", "one", "two", "use",
    "very", "some", "any", "each", "only", "such", "over", "about",
})


def _tokens(text: str) -> frozenset[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return frozenset(w for w in words if w not in _STOP and len(w) >= 2)


# ── Attribution constants ──────────────────────────────────────────────────────

# ROUGE1 below this → LLM fallback fires (if use_llm_attr=True)
ROUGE1_THRESHOLD: float = 0.05
# Skip LLM call if w_imp already high — memory will rank well regardless
LLM_ATTR_W_IMP_GATE: float = 0.5

# CCOMMIT (CIFS analog, ReSurgSAM2 arXiv:2505.08581 Eq. 1): a memory's credit
# becomes retrieval-effective only after clearing the attribution noise floor
# in N_COMMIT distinct feedback events — one spurious attribution can no longer
# permanently boost retrieval. Threshold matches the co-cite noise floor.
COMMIT_DELTA: float = 0.15
N_COMMIT: int = 2

# SPACING (Item 2, Landauer & Bjork 1978 spacing effect): saturating
# session-gap multiplier on the w_imp delta. f(gap) = 1 + kappa*(1-exp(-gap/G)):
# a re-credit within the same session (gap=0) gets f=1 (no change from
# today's behavior); a re-credit after a genuine cross-session gap gets up to
# 1+kappa. Saturating so a memory re-surfacing after a very long gap by
# chance doesn't get an unbounded reward — mirrors the same "no unbounded
# credit" caution as UCB_KAPPA/RELIABILITY_KAPPA in recall.py.
SPACING_KAPPA: float = 0.5
SPACING_G: float = 2.0


def spacing_multiplier(gap: int) -> float:
    """f(gap) for SPACING (Item 2): monotonic increasing, saturating at 1+kappa."""
    if gap <= 0:
        return 1.0
    return 1.0 + SPACING_KAPPA * (1.0 - math.exp(-gap / SPACING_G))


# Sparse-evidence guard: a MUCH lower bar than COMMIT_DELTA, deliberately.
# Diagnostic finding (2026-07-10, results/locomo_v3_isolated_sparseguard.json):
# gating the guard's own trigger on COMMIT_DELTA made it inherit CCOMMIT's
# exact blind spot — under ROUGE1 attribution, multi-hop scores almost never
# clear 0.15 (measured range 0.0-0.14 across sampled multi-hop QAs), so the
# guard never fired and multi_hop deltas were exactly 0.0000. The guard only
# needs to distinguish "some real signal, above pure lexical noise" from
# "nothing" — not "commit-worthy" — so it uses its own, separate, low bar.
SPARSE_CITE_DELTA: float = 0.02


def rouge1_recall(memory_text: str, answer_text: str) -> float:
    """ROUGE-1 recall: fraction of memory tokens appearing in the answer."""
    mem_tokens = _tokens(memory_text)
    if not mem_tokens:
        return 0.0
    ans_tokens = _tokens(answer_text)
    return len(mem_tokens & ans_tokens) / len(mem_tokens)


def position_weight(rank: int) -> float:
    return 1.0 / (rank + 1)


# ── FORESIGHT-NORM: path-baseline-normalized attribution ──────────────────────
# Inspired by Wan, Peng, Gangwani "Hindsight Foresight Relabeling" (ICLR 2022,
# arXiv:2109.09031): raw utility isn't comparable across regimes with different
# achievable ceilings — their log Z(psi) partition function normalizes per-task.
# Here, "task" = attribution path ("rouge1" | "llm"), since ROUGE1 has a
# structurally different achievable ceiling than LLM-fallback scores (numeric/
# temporal answers are near-zero under ROUGE1 by construction). Z_running is a
# closed-form EMA — no gradient updates, no critic network, stays training-free.
Z_EMA_DECAY: float = 0.98
Z_FLOOR: float = 0.05  # avoid divide-by-zero / over-crediting on first observations


def update_z_baseline(path: str, attr: float, z_running: dict[str, float]) -> None:
    prev = z_running.get(path, Z_FLOOR)
    z_running[path] = Z_EMA_DECAY * prev + (1 - Z_EMA_DECAY) * attr


# ── Verification gate: stop crediting memories for implausible answers ────────
# The self-reinforcing-error failure mode (the central named risk of reflective
# memory in the 2026 surveys, and admitted-without-mechanism in our own draft's
# Ethics Statement): TRACE credits memories from UNVERIFIED answers, so a wrong
# answer strengthens exactly the memories that produced it, which get retrieved
# more, and reproduce the same wrong answer. Two training-free signals gate the
# credit instead:
#   1. Abstention: "I don't know"-style answers carry no evidence any memory
#      helped — hard gate to 0 (any attribution path).
#   2. Scorer confidence (CF path only): exp(lp_full) — the geometric-mean
#      per-token answer probability under the full context, already computed
#      as the LOO reference — normalized against a running EMA baseline (same
#      closed-form pattern as FORESIGHT-NORM's z_running: no new models, no
#      thresholds to hand-tune). Answers the scorer finds unusually unlikely
#      relative to this store's typical answers get proportionally less credit.
V_EMA_DECAY: float = 0.98

_ABSTENTION_PATTERNS = re.compile(
    r"(?:i\s+(?:do\s*n[o']t|don't|cannot|can\s*not|can't)\s+"
    r"(?:know|recall|remember|say|tell|find|answer)"
    r"|not\s+(?:sure|mentioned|specified|stated|provided|available)"
    r"|no\s+(?:information|memory|record|mention)"
    r"|unable\s+to\s+(?:find|answer|determine|recall)"
    r"|unknown|unanswerable)",
    re.IGNORECASE,
)


def is_abstention(answer: str) -> bool:
    """True for empty or 'I don't know'-style answers — no credit evidence."""
    text = (answer or "").strip()
    if not text:
        return True
    return bool(_ABSTENTION_PATTERNS.search(text))


def verification_gate(lp_full: Optional[float],
                      v_running: Optional[dict[str, float]]) -> float:
    """
    Soft gate in (0, 1] from scorer answer-confidence. lp_full is nats/token;
    the gate is exp(lp_full - baseline) clipped to 1 — i.e. full credit for
    answers at least as plausible as this store's running typical answer,
    proportionally less for less-plausible ones. Baseline is a closed-form
    EMA seeded with the first observation (gate=1.0 on the very first call —
    no prior to compare against). Returns 1.0 (no-op) when lp_full is
    unavailable (non-CF attribution paths).
    """
    if lp_full is None or v_running is None:
        return 1.0
    if "lp" not in v_running:
        v_running["lp"] = lp_full
        return 1.0
    baseline = v_running["lp"]
    gate = min(1.0, math.exp(lp_full - baseline))
    v_running["lp"] = V_EMA_DECAY * baseline + (1 - V_EMA_DECAY) * lp_full
    return gate


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class TraceFBResult:
    """Summary of one TRACE-FB feedback call."""
    updated:           dict[str, float] = field(default_factory=dict)
    reset:             list[str]        = field(default_factory=list)
    attributions:      dict[str, float] = field(default_factory=dict)
    attribution_paths: dict[str, str]   = field(default_factory=dict)  # "rouge1"|"llm"|"oracle"|"cf"
    verification_gate: float            = 1.0  # credit multiplier applied this call


# ── Main feedback function ────────────────────────────────────────────────────

def apply_trace_feedback(
    retrieved: list[tuple["MemoryItem", float]],
    answer: str,
    store: "MemoirStore",
    alpha: float = 0.1,
    hallucinated_ids: Optional[set[str]] = None,
    oracle_correct: Optional[bool] = None,
    query: str = "",           # required for LLM attribution prompt
    use_llm_attr: bool = False, # opt-in: LLM fallback for ROUGE1-blind items
    use_nli: bool = False,     # deprecated no-op; kept for call-site compat
    alpha_neg: float = 0.0,    # asymmetric decay rate for retrieved-not-cited items (0=disabled)
    use_foresight_norm: bool = False,  # HFR-inspired path-baseline normalization
    z_running: Optional[dict[str, float]] = None,  # per-store running baselines by path
    use_cf_attr: bool = False,  # TRACE-CF: counterfactual LOO attribution replaces ROUGE1
    cf_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    use_verification_gate: bool = False,  # gate credit on answer plausibility (anti self-reinforcing-error)
    v_running: Optional[dict[str, float]] = None,  # per-store lp EMA; required when gating with CF
    commit_calibration=None,  # Optional[CommitCalibration] (consolidate.py) — calibrated
                               # CCOMMIT mode; None falls back to fixed COMMIT_DELTA/commit_count
    query_vec=None,            # MULTI-ANCHOR (Item 1): query embedding to cache as an anchor
                                # on credited items; None disables anchoring regardless of flag
    use_multi_anchor: bool = False,  # MULTI-ANCHOR (Item 1): cache query_vec as an auxiliary
                                       # retrieval anchor on items earning real credit this call
    anchor_min_attr: float = None,  # MULTI-ANCHOR (Item 1): minimum attribution to record an
                                      # anchor; defaults to SPARSE_CITE_DELTA (see below). Measured
                                      # temporal regression (docs/PAPER_COMPONENTS.md #11) with the
                                      # low default motivated a stricter-threshold experiment —
                                      # exposed as a real parameter rather than a hand-edited constant.
    session_idx: Optional[int] = None,  # SPACING (Item 2): current session index
    use_spacing: bool = False,  # SPACING (Item 2): gate the session-gap multiplier on delta
) -> TraceFBResult:
    """
    Update w_imp for each retrieved memory via ROUGE1-recall attribution.
    Optionally fires LLM fallback (claude-haiku-4-5) for ROUGE1-blind items.

    Args:
        retrieved:      (item, score) pairs from MemoirStore.search()
        answer:         LLM-generated answer text for the query
        store:          MemoirStore (unused directly; retained for interface compat)
        alpha:          learning rate for w_imp update
        hallucinated_ids: memory ids to hard-reset to 0
        oracle_correct: if not None, bypass attribution (True=1.0, False=0.0)
        query:          original query — required when use_llm_attr=True
        use_llm_attr:   if True, fire LLM for items where ROUGE1 < threshold
                        and w_imp < LLM_ATTR_W_IMP_GATE
        use_nli:        deprecated no-op
        alpha_neg:      if > 0, retrieved-but-uncited items (attr==0) with
                        w_imp > 0 get a soft negative update — addresses
                        anchoring (memories cited early dominate forever).
                        5x softer than alpha by convention (caller sets it).
        use_foresight_norm: if True, positive deltas are computed relative to
                        a per-attribution-path running baseline instead of
                        raw attribution — see FORESIGHT-NORM docstring above.
        z_running:      mutable dict shared across feedback calls on the same
                        store; required (non-None) when use_foresight_norm=True.
    """
    result = TraceFBResult()
    if not retrieved:
        return result

    # Verification gate, part 1 — abstention: an "I don't know" answer is not
    # evidence that any retrieved memory helped. Oracle path is exempt (its
    # binary signal already encodes correctness). Hallucination resets still
    # run below — the gate only blocks POSITIVE credit, never safety resets.
    gate = 1.0
    if use_verification_gate and oracle_correct is None and is_abstention(answer):
        gate = 0.0
    result.verification_gate = gate

    attr_scores: dict[str, float] = {}
    skip_ids:    set[str]         = set()
    llm_needed:  list[tuple[int, "MemoryItem"]] = []  # (rank, item)
    cf_needed:   list["MemoryItem"]             = []

    # ── Phase 1: hallucinations, oracle, ROUGE1 / CF routing ──────────────────
    for rank, (item, _score) in enumerate(retrieved):
        if hallucinated_ids and item.id in hallucinated_ids:
            item.hallucinated = True
            item.w.w_imp = 0.0
            result.reset.append(item.id)
            skip_ids.add(item.id)
            continue

        if item.hallucinated:
            result.reset.append(item.id)
            skip_ids.add(item.id)
            continue

        if oracle_correct is not None:
            attr = 1.0 if oracle_correct else 0.0
            attr_scores[item.id] = attr
            result.attributions[item.id] = attr
            result.attribution_paths[item.id] = "oracle"
            continue

        if use_cf_attr:
            # TRACE-CF replaces both ROUGE1 and the LLM fallback: attribution
            # is computed behaviorally in one batched pass after this loop.
            cf_needed.append(item)
            continue

        r1 = rouge1_recall(item.text, answer)
        attr_scores[item.id] = r1
        result.attribution_paths[item.id] = "rouge1"

        if (use_llm_attr
                and r1 < ROUGE1_THRESHOLD
                and item.w.w_imp < LLM_ATTR_W_IMP_GATE
                and query):
            llm_needed.append((rank, item))

    # ── Phase 1.5: TRACE-CF leave-one-out attribution (single batched pass) ───
    if cf_needed:
        from .cf_attributor import CFAttributor
        cf = CFAttributor.get(cf_model_name)
        scores, lp_full = cf.score_loo_with_confidence(
            question=query,
            answer=answer,
            memories=[item.text for item in cf_needed],
        )
        for item, score in zip(cf_needed, scores):
            attr_scores[item.id] = score
            result.attribution_paths[item.id] = "cf"
            # CF-SHIFT GATE (Item 4): cache the LOO shift on the item itself —
            # previously discarded after folding into attr_scores. Zero extra
            # computation; recall.py's frictionless gate reads this to exempt
            # load-bearing memories from Weibull/CCOMMIT/CORE-DLM friction.
            item.cf_shift = score
        # Verification gate, part 2 — scorer confidence (CF only; other paths
        # have no likelihood signal and keep gate as set by the abstention
        # check). Skipped entirely when the abstention gate already zeroed
        # credit — don't pollute the lp baseline with an abstention's lp.
        if use_verification_gate and gate > 0.0 and oracle_correct is None:
            gate = verification_gate(lp_full, v_running)
            result.verification_gate = gate

    # ── Phase 2: batch LLM calls for ROUGE1-blind items ───────────────────────
    if llm_needed:
        from .llm_attributor import LLMAttributor
        attributor = LLMAttributor.get()
        items_to_score = [
            (item.text, query, answer)
            for _, item in llm_needed
        ]
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context (e.g. Jupyter) — use run_coroutine_threadsafe
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    attributor.score_batch_async(items_to_score), loop
                )
                scores = future.result(timeout=60)
            else:
                scores = loop.run_until_complete(
                    attributor.score_batch_async(items_to_score)
                )
        except RuntimeError:
            scores = asyncio.run(attributor.score_batch_async(items_to_score))

        for (_, item), score in zip(llm_needed, scores):
            attr_scores[item.id] = score
            result.attribution_paths[item.id] = "llm"

    # ── Record final attributions ──────────────────────────────────────────────
    for item, _ in retrieved:
        if item.id in skip_ids:
            continue
        if item.id in attr_scores:
            result.attributions[item.id] = attr_scores[item.id]

    # Sparse-evidence flag: this answer drew on MULTIPLE memories (a multi-hop
    # support set), and this memory's citation here is its first real one.
    # Such memories are exactly what credit-friction mechanisms (decay/commit-
    # gates/core-discounts) starve — measured as the friction-family multi_hop
    # regression. Flag them frictionless until a second real citation arrives
    # (then normal friction applies — they're no longer sparse).
    # Uses SPARSE_CITE_DELTA, NOT COMMIT_DELTA — gating on COMMIT_DELTA made
    # an earlier version of this guard inherit CCOMMIT's own blind spot (see
    # SPARSE_CITE_DELTA's docstring).
    n_credited = sum(1 for a in attr_scores.values() if a >= COMMIT_DELTA)
    n_cited = sum(1 for a in attr_scores.values() if a >= SPARSE_CITE_DELTA)

    # ── Phase 3: apply w_imp updates ──────────────────────────────────────────
    for rank, (item, _score) in enumerate(retrieved):
        if item.id in skip_ids:
            continue
        attr = attr_scores.get(item.id, 0.0)
        if attr >= COMMIT_DELTA:
            item.commit_count += 1  # CCOMMIT fallback: always tracked; gating is retrieval-side
        if commit_calibration is not None and attr >= SPARSE_CITE_DELTA:
            # Calibrated CCOMMIT: accumulate log-likelihood-ratio instead of a
            # raw event count, so two strong-signal citations commit faster
            # than two barely-above-floor ones (fixes the count-based gate's
            # mismatch with weak/dense attribution signals — see
            # consolidate.py:CommitCalibration docstring).
            item.commit_llr += commit_calibration.llr_increment(attr)
        if attr >= SPARSE_CITE_DELTA:
            item.cite_count += 1
            if item.cite_count == 1 and n_cited >= 2:
                item.sparse_evidence = True
            elif item.cite_count >= 2:
                item.sparse_evidence = False
        path = result.attribution_paths.get(item.id, "rouge1")

        if attr == 0.0:
            # Asymmetric TRACE-FB decay: retrieved but did not contribute.
            # Softer than the positive path by convention (caller sets alpha_neg,
            # e.g. alpha*0.2) — addresses anchoring: without this, memories
            # cited early keep dominating retrieval forever (Zhou et al. 2026,
            # arXiv:2606.24775: "catastrophic degradation as evidence becomes
            # more distant").
            if alpha_neg > 0.0 and item.w.w_imp > 0.0:
                decay = alpha_neg * position_weight(rank)
                old_imp = item.w.w_imp
                item.w.w_imp = max(0.0, old_imp - decay)
                result.updated[item.id] = item.w.w_imp - old_imp
            continue

        # FORESIGHT-NORM: credit attribution relative to what's typically
        # achievable on this path, not its raw magnitude (HFR, arXiv:2109.09031).
        # Oracle attribution bypasses this — it's already a clean binary signal,
        # not subject to a path-specific achievable ceiling.
        if use_foresight_norm and path != "oracle" and z_running is not None:
            baseline = z_running.get(path, Z_FLOOR)
            effective_attr = max(0.0, attr - baseline)
            update_z_baseline(path, attr, z_running)
        else:
            effective_attr = attr

        # Verification gate scales POSITIVE credit only — the alpha_neg decay
        # path above runs ungated (a low-confidence answer is weak evidence a
        # memory helped, but "retrieved and uncited" decay evidence is
        # independent of answer quality), and commit_count tracking above is
        # likewise ungated (CCOMMIT gates retrieval-effectiveness separately).
        # SPACING (Item 2): reward re-credit after a genuine cross-session gap
        # more than same-session re-credit — independently dampens the
        # rich-get-richer lock-in UCB otherwise patches. gap=0 (first credit,
        # or same-session recredit) leaves delta unchanged (spacing_mult=1.0).
        spacing_mult = 1.0
        if use_spacing and session_idx is not None:
            gap = session_idx - (item.last_credit_session
                                  if item.last_credit_session is not None
                                  else session_idx)
            spacing_mult = spacing_multiplier(gap)

        delta = alpha * effective_attr * position_weight(rank) * gate * spacing_mult
        old_imp = item.w.w_imp
        item.w.w_imp = min(1.0, old_imp + delta)
        result.updated[item.id] = item.w.w_imp - old_imp

        if use_spacing and session_idx is not None:
            item.last_credit_session = session_idx

        # MULTI-ANCHOR (Item 1): this query earned the item real credit —
        # cache it as an auxiliary retrieval path (Stern 2002 cognitive
        # reserve). Reuses SPARSE_CITE_DELTA as the "real signal, not noise"
        # bar (same rationale as the sparse-evidence guard above).
        _anchor_bar = SPARSE_CITE_DELTA if anchor_min_attr is None else anchor_min_attr
        if use_multi_anchor and query_vec is not None and attr >= _anchor_bar:
            item.add_anchor(query_vec, attr, max_anchors=3)

    return result


def hallucinated_ids_from_trajectory(trajectory) -> set[str]:
    """Extract memory ids that contributed to a hallucination step."""
    try:
        from harpo.trajectory.schema import FailureMode
    except ImportError:
        return set()

    ids: set[str] = set()
    if trajectory is None:
        return ids
    report = getattr(trajectory, "failure_report", None)
    if report is None:
        return ids
    if FailureMode.HALLUCINATION not in getattr(report, "failure_modes", []):
        return ids
    contributing_step_ids = set(getattr(report, "contributing_steps", []))
    for step in trajectory.steps:
        if step.step_id not in contributing_step_ids:
            continue
        if step.memory_access and step.memory_access.key:
            ids.add(step.memory_access.key)
    return ids
