"""
RECALL-Q: Composite retrieval scoring with ablation flags.

Formula (v5 — additive bonuses on cosine):
  score(m) = cos(q,m) · (1 + beta·w_rel(m) + gamma_imp·w_imp(m))
           + α·BM25(q,m)  −  λ·MMR(m, selected)

  Design: cosine is the base signal; w_rel and TRACE-FB are proportional bonuses.
    - Fresh memories (w_imp=0, w_rel=0): score = cos  — same as cosine baseline
    - Session-relevant memories (w_rel>0): score = cos*(1 + beta*w_rel) — small boost
    - TRACE-FB validated memories (w_imp>0): score = cos*(1 + gamma_imp*w_imp) — proportional boost

  This formula works with both weak (MiniLM) and fine-tuned encoders:
    - With weak encoder: cosine differences small, bonuses help rerank
    - With fine-tuned encoder: cosine is already discriminative, bonuses add precision

  Previous v4 formula: cos*w_rel*beta_floor + gamma_imp*cos*w_imp
    Problem with fine-tuned: cos*w_rel*0.1 suppressed cosine by 10× for non-TRACE memories,
    making memoir_no_trace score lower than pure cosine baseline.

Hyperparameters:
  alpha      = 0.3   BM25 contribution weight
  lam        = 0.1   MMR diversity penalty
  beta_floor = 0.1   w_rel bonus weight (keeps session-relevance signal small but present)
  gamma_imp  = 1.0   TRACE-FB bonus weight

Ablation flags:
  no_w_imp  — zero the gamma_imp*w_imp term
  no_bm25   — force alpha = 0.0
  no_mmr    — force lam = 0.0
  flat_w    — force w_rel=1.0, w_imp=0 (pure cosine+BM25 baseline)
  use_ucb   — exploration bonus (SimUtil-UCB adapted)
  use_reliability — Laplace-smoothed eta() soft bonus (oracle-feedback only)
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from .schema import MemoryItem

# Co-citation graph bonus scale: max contribution per neighbor = 0.05 × 1.0 × 1.0 = 0.05
CO_CITE_BONUS_SCALE: float = 0.05

# Weibull temporal decay on w_imp (retrieval-time only, zero write-path cost).
# S(t) = exp(-(t/lambda)^k) — decays ACCUMULATED IMPORTANCE specifically, not
# raw relevance (w_rec already handles general recency). lambda=168h (1 week),
# k=0.5 (heavy-tailed: penalizes recent staleness but retains long-tail credit
# for distant-but-still-cited memories). Matches LiCoMemory's own ablation-
# validated contribution (arXiv:2511.01448, their #1 ablation after structured
# retrieval), adapted here to decay w_imp rather than raw triple relevance.
WEIBULL_LAMBDA_HOURS: float = 168.0
WEIBULL_K: float = 0.5

# RELIABILITY: Laplace-smoothed success-rate soft bonus (MSCE-inspired,
# arXiv:2607.16621 §4.3 eta), oracle-feedback-only signal (MemoryItem.eta()).
# Proportional, ablation-gated, default OFF: an untried item (eta=0.5) is
# lifted by RELIABILITY_KAPPA*0.5 (near-neutral vs the item's own baseline);
# a proven-useful item (eta->1.0) gets up to RELIABILITY_KAPPA additional
# w_imp-scale credit. Small relative to UCB_KAPPA since eta is directly
# outcome-conditioned (less exploratory, more confirmatory) than the UCB term.
RELIABILITY_KAPPA: float = 0.1

# UCB exploration bonus on w_imp (adapted from RetroAgent's SimUtil-UCB,
# arXiv:2603.08561 Eq. 9): w_imp_eff = w_imp + κ·sqrt(ln N / (n_m + 1)) where
# n_m is the item's retrieval count and N the store-wide total. Counters
# rich-get-richer lock-in: without it, memories that accumulate w_imp early
# keep being retrieved and keep earning credit while never-retrieved memories
# can never receive TRACE feedback at all. κ is small because the bonus
# multiplies cosine through gamma_imp (same scale as w_imp itself, [0,1]).
UCB_KAPPA: float = 0.05

# CCOMMIT retrieval gate (see trace_fb.COMMIT_DELTA/N_COMMIT for the write side)
COMMIT_N: int = 2

# CORE-DLM: non-core memories receive a discounted w_imp bonus. Core membership
# is decided by diversity-driven promotion in the store (ReSurgSAM2 Eqs. 2-3).
CORE_DISCOUNT: float = 0.25

# MULTI-ANCHOR (Item 1, Stern 2002 cognitive reserve + Craik & Lockhart 1972
# deferred depth-of-encoding): a memory's cosine score is the MAX over its
# base embedding and any auxiliary query-embedding anchors it has earned real
# TRACE credit from (MemoryItem.anchors, capped at MAX_ANCHORS in schema.py's
# add_anchor()). Read-only here — anchors are written in trace_fb.py.
MAX_ANCHORS: int = 3

# TYPE-EMERGENT TAXONOMY (Item 3, Tulving 1972): certified-merge output is
# tagged stratum="semantic" (store.py consolidate()) and gets a slower w_rec
# decay-rate than the "episodic" default — a generalized/consolidated fact
# should outlast the raw single-session evidence it was built from. Gated by
# use_stratum_decay; off by default (falls back to the original flat 0.1 for
# every item regardless of stratum).
STRATUM_LAMBDA_REC: dict = {"episodic": 0.1, "semantic": 0.03}

# CF-SHIFT GATE (Item 4, Ebbinghaus 1885 controlled forgetting refined by
# Anderson/Bjork/Bjork 1994 selective retrieval-induced forgetting): a memory
# whose cached TRACE-CF leave-one-out shift (MemoryItem.cf_shift, written in
# trace_fb.py Phase 1.5) clears this bar is exempt from ALL friction
# mechanisms below (Weibull, CCOMMIT, CORE-DLM) — the same credit-protection
# logic "Bridge 1" already applies at consolidation time (store.py
# consolidate()'s has_protected check), reused here for decay instead of
# merge/prune. Gated by use_cf_shift_gate; requires use_cf_attr=True to
# populate cf_shift at all — degrades to a safe no-op otherwise (cf_shift
# stays 0.0, never clears the bar).
CF_SHIFT_EXEMPT_THRESHOLD: float = 0.3


def _weibull_decay(age_hours: float) -> float:
    if age_hours <= 0:
        return 1.0
    return math.exp(-((age_hours / WEIBULL_LAMBDA_HOURS) ** WEIBULL_K))


def _mmr_penalty(
    item: MemoryItem,
    selected: list[MemoryItem],
    embeddings_matrix: np.ndarray,
    item_idx: int,
) -> float:
    """Max cosine similarity to any already-selected item."""
    if not selected or item.embedding is None:
        return 0.0
    selected_vecs = np.stack([s.embedding for s in selected if s.embedding is not None])
    sims = selected_vecs @ item.embedding  # embeddings are unit-norm
    return float(np.max(sims))


def recall_q(
    query_vec: np.ndarray,
    store_items: list[MemoryItem],
    embeddings_matrix: np.ndarray,  # shape (n, dim), unit-norm rows
    bm25,                            # rank_bm25.BM25Okapi instance
    k: int = 10,
    alpha: float = 0.3,
    lam: float = 0.1,
    beta_floor: float = 0.1,
    oversample: int = 3,
    # Ablation flags
    no_w_imp: bool = False,
    no_bm25:  bool = False,
    no_mmr:   bool = False,
    flat_w:   bool = False,
) -> list[tuple[MemoryItem, float]]:
    """
    Returns up to k (item, score) pairs ordered by RECALL-Q score.
    Items with hallucinated=True are excluded.
    """
    if not store_items:
        return []

    valid_items = [m for m in store_items if not m.hallucinated and m.embedding is not None]
    if not valid_items:
        return []

    n = len(valid_items)
    k_fetch = min(k * oversample, n)

    # ── Step 1: cosine scores (vectorized) ──────────────────────────────────
    valid_embs = embeddings_matrix[:n]   # caller ensures matrix is current
    cos_scores = valid_embs @ query_vec   # shape (n,)

    # ── Step 2: BM25 scores ──────────────────────────────────────────────────
    if not no_bm25 and alpha > 0:
        query_tokens = query_vec  # caller passes actual query text via bm25_query
        # bm25_query is passed separately; retrieve via closure isn't clean here.
        # Scores computed by caller and passed — see store.py for integration.
        # Fallback: zero BM25 if not provided (handled in store.py)
        bm25_scores_raw = np.zeros(n, dtype=np.float32)
    else:
        bm25_scores_raw = np.zeros(n, dtype=np.float32)

    # ── Step 3: oversampled candidates ───────────────────────────────────────
    top_indices = np.argpartition(cos_scores, -k_fetch)[-k_fetch:]
    top_indices = top_indices[np.argsort(cos_scores[top_indices])[::-1]]

    # ── Step 4: full score per candidate ─────────────────────────────────────
    selected: list[MemoryItem] = []
    results:  list[tuple[MemoryItem, float]] = []

    for idx in top_indices:
        item = valid_items[idx]
        cos  = float(cos_scores[idx])
        bm25 = float(bm25_scores_raw[idx])

        # Live recency (not cached)
        item.w.w_rec = item.live_w_rec()

        if flat_w:
            w_rel   = 1.0
            w_imp_v = 0.0
        else:
            w_rel   = item.w.w_rel
            w_imp_v = 0.0 if no_w_imp else item.w.w_imp

        bm25_term = 0.0 if no_bm25 else alpha * bm25
        mmr_pen   = 0.0 if no_mmr  else lam * _mmr_penalty(item, selected, valid_embs, idx)

        score = cos * (1.0 + beta_floor * w_rel + 1.0 * w_imp_v) + bm25_term - mmr_pen
        results.append((item, score))

    # ── Step 5: greedy MMR selection ─────────────────────────────────────────
    results.sort(key=lambda x: x[1], reverse=True)
    final: list[tuple[MemoryItem, float]] = []
    for item, score in results:
        if len(final) >= k:
            break
        final.append((item, score))
        selected.append(item)

    return final


def recall_q_with_bm25(
    query: str,
    query_vec: np.ndarray,
    store_items: list[MemoryItem],
    embeddings_matrix: np.ndarray,
    bm25,
    k: int = 10,
    alpha: float = 0.3,
    lam: float = 0.1,
    beta_floor: float = 0.1,
    gamma_imp: float = 1.0,
    delta_rec: float = 0.1,
    co_cite_graph: "dict[tuple[str, str], float] | None" = None,
    now: "float | None" = None,
    core_ids: "set[str] | None" = None,
    **ablation_flags,
) -> list[tuple[MemoryItem, float]]:
    """
    Full RECALL-Q with BM25 scores pre-computed here.
    This is the function called by MemoirStore.search().

    `now`: reference timestamp for w_rec / Weibull-decay age computation.
    Defaults to wall-clock time.time() (correct in production). Eval harnesses
    that add() a whole multi-session dataset within seconds of wall-clock time
    should pass the dataset's simulated "current session time" here instead —
    otherwise every item's age collapses to ~0 and both w_rec and Weibull
    decay become no-ops regardless of the real (dataset-native) time gaps.
    """
    if now is None:
        now = time.time()
    if not store_items:
        return []

    valid_pos = [i for i, m in enumerate(store_items)
                 if not m.hallucinated and not m.pruned and not m.superseded
                 and m.embedding is not None]
    valid_items = [store_items[i] for i in valid_pos]
    if not valid_items:
        return []

    n = len(valid_items)
    k_fetch = min(k * 3, n)

    # embeddings_matrix and BM25 corpus are parallel to store_items (insertion
    # order) — index by valid positions so filtering (hallucinated/pruned)
    # cannot misalign scores with items.
    valid_embs = embeddings_matrix[valid_pos]
    cos_scores = valid_embs @ query_vec

    # BM25 scores
    no_bm25 = ablation_flags.get("no_bm25", False)
    if not no_bm25 and alpha > 0 and bm25 is not None:
        query_tokens = query.lower().split()
        raw_bm25 = np.array(bm25.get_scores(query_tokens), dtype=np.float32)[valid_pos]
        # Normalize to [0, 1]
        b_min, b_max = raw_bm25.min(), raw_bm25.max()
        bm25_scores = (raw_bm25 - b_min) / (b_max - b_min + 1e-9)
    else:
        bm25_scores = np.zeros(n, dtype=np.float32)

    flat_w   = ablation_flags.get("flat_w", False)
    no_w_imp = ablation_flags.get("no_w_imp", False)
    no_mmr   = ablation_flags.get("no_mmr", False)
    no_w_rec = ablation_flags.get("no_w_rec", False)
    use_weibull = ablation_flags.get("use_weibull", False)
    # peek: read-only search (no retrieval_count bump) — used by measurement
    # probes (e.g. eval_consolidation.py's epsilon_delta) so the act of
    # probing doesn't itself perturb UCB/recency state between two calls.
    peek     = ablation_flags.get("peek", False)
    use_ucb  = ablation_flags.get("use_ucb", False)
    if use_ucb:
        total_retrievals = sum(m.retrieval_count for m in valid_items)
        ucb_log_n = math.log(max(total_retrievals, 2))
    use_reliability = ablation_flags.get("use_reliability", False)
    use_commit = ablation_flags.get("use_commit", False)
    use_multi_anchor = ablation_flags.get("use_multi_anchor", False)
    use_stratum_decay = ablation_flags.get("use_stratum_decay", False)
    use_cf_shift_gate = ablation_flags.get("use_cf_shift_gate", False)
    # Calibrated CCOMMIT: if a threshold was provided (store.py sets this from
    # a loaded CommitCalibration), gate on accumulated LLR instead of the
    # fixed COMMIT_DELTA/COMMIT_N event counter — see consolidate.py's
    # CommitCalibration for why the fixed threshold was mismatched with weak
    # attribution signals. commit_llr_threshold=None means "no calibration
    # loaded", falling back to the original fixed-threshold behavior.
    commit_llr_threshold = ablation_flags.get("commit_llr_threshold", None)

    # Oversample
    top_indices = np.argpartition(cos_scores, -k_fetch)[-k_fetch:]
    top_indices = top_indices[np.argsort(cos_scores[top_indices])[::-1]]

    selected: list[MemoryItem] = []
    scored:   list[tuple[MemoryItem, float]] = []

    for idx in top_indices:
        item = valid_items[idx]
        cos  = float(cos_scores[idx])
        bm   = float(bm25_scores[idx])

        # MULTI-ANCHOR (Item 1): score by the best of the base embedding or
        # any auxiliary query-embedding anchors this item earned real TRACE
        # credit from — independent retrieval paths to the same memory
        # (Stern 2002 cognitive reserve). Anchors are NOT part of
        # embeddings_matrix, so this is a small per-item Python max over at
        # most MAX_ANCHORS extra dot products, only for items that have any.
        if use_multi_anchor and item.anchors:
            anchor_sims = [float(np.dot(a, query_vec)) for a in item.anchors]
            cos = max(cos, max(anchor_sims))

        # TYPE-EMERGENT TAXONOMY (Item 3): semantic-stratum items (certified-
        # merge output) decay slower than episodic ones when the ablation is
        # on; off (default) uses the original flat lambda_rec=0.1 regardless
        # of stratum.
        if use_stratum_decay:
            lambda_rec = STRATUM_LAMBDA_REC.get(item.stratum, STRATUM_LAMBDA_REC["episodic"])
            item.w.w_rec = item.live_w_rec(lambda_rec=lambda_rec, now=now)
        else:
            item.w.w_rec = item.live_w_rec(now=now)

        if flat_w:
            w_rel   = 1.0
            w_imp_v = 0.0
            w_rec_v = 0.0
        else:
            w_rel   = item.w.w_rel
            w_imp_v = 0.0 if no_w_imp else item.w.w_imp
            w_rec_v = 0.0 if no_w_rec else item.w.w_rec
            # Sparse-evidence guard: multi-hop supports are typically cited
            # ONCE across a long cross-session gap — every credit-friction
            # mechanism (decay, commit-gates, core discounts) structurally
            # starves exactly those citations (measured: Weibull/CCOMMIT/
            # CORE-DLM all regress multi_hop 4.6-5.3pp while helping
            # elsewhere — the friction-family finding). Items flagged
            # sparse_evidence by TRACE-FB keep their credit friction-free.
            # CF-SHIFT GATE (Item 4) extends the same exemption continuously:
            # any item whose cached TRACE-CF LOO shift clears
            # CF_SHIFT_EXEMPT_THRESHOLD is load-bearing regardless of the
            # ROUGE1-derived sparse_evidence heuristic — mirrors Bridge 1's
            # credit-protection pattern from consolidate(), applied to decay.
            frictionless = item.sparse_evidence or (
                use_cf_shift_gate and item.cf_shift >= CF_SHIFT_EXEMPT_THRESHOLD)
            if use_weibull and w_imp_v > 0.0 and not frictionless:
                age_hours = (now - item.created_at) / 3600.0
                w_imp_v = w_imp_v * _weibull_decay(age_hours)
            # CCOMMIT: credit invisible to retrieval until commit criterion is
            # met (applied before the UCB exploration bonus, which is
            # independent of accumulated credit). Calibrated mode (LLR
            # threshold present) supersedes the fixed event-count fallback.
            if use_commit and not frictionless:
                if commit_llr_threshold is not None:
                    if item.commit_llr < commit_llr_threshold:
                        w_imp_v = 0.0
                elif item.commit_count < COMMIT_N:
                    w_imp_v = 0.0
            # CORE-DLM: full bonus only for diversity-promoted core members
            if (core_ids is not None and item.id not in core_ids
                    and not frictionless):
                w_imp_v = w_imp_v * CORE_DISCOUNT
            if use_ucb and not no_w_imp:
                w_imp_v = w_imp_v + UCB_KAPPA * math.sqrt(
                    ucb_log_n / (item.retrieval_count + 1))
            if use_reliability and not no_w_imp:
                w_imp_v = w_imp_v + RELIABILITY_KAPPA * item.eta()

        bm25_term = 0.0 if no_bm25 else alpha * bm
        mmr_pen   = 0.0 if no_mmr  else lam * _mmr_penalty(item, selected, valid_embs, idx)

        # v5 formula: cosine is the base, w_rel/w_rec/TRACE-FB are additive bonuses
        # score = cos * (1 + beta*w_rel + delta*w_rec + gamma*w_imp) + BM25 - MMR
        # This means: fresh memories rank by cosine (no suppression), TRACE-FB
        # and session relevance add proportional boosts on top.
        # Previous v4: cos*w_rel*beta + gamma*cos*w_imp  — suppressed cosine by w_rel*beta
        score = (cos * (1.0 + beta_floor * w_rel + delta_rec * w_rec_v + gamma_imp * w_imp_v)
                 + bm25_term - mmr_pen)
        scored.append((item, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    final: list[tuple[MemoryItem, float]] = []
    for item, base_score in scored:
        if len(final) >= k:
            break

        # Co-cite bonus: propagate w_imp from already-selected neighbors
        co_cite_bonus = 0.0
        if co_cite_graph and selected:
            for sel in selected:
                edge_key = tuple(sorted([sel.id, item.id]))
                ew = co_cite_graph.get(edge_key, 0.0)
                if ew > 0.0:
                    co_cite_bonus += ew * sel.w.w_imp

        effective_score = base_score + CO_CITE_BONUS_SCALE * co_cite_bonus
        final.append((item, effective_score))
        selected.append(item)
        if not peek:
            item.bump_retrieval()

    return final
