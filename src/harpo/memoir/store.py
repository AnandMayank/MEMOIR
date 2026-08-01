"""
MemoirStore: unified public API.

Three-method interface:
  add(text)               → memory_id (sync; TRIAGE-α runs async in background)
  search(query, k)        → list[MemoryItem] (sync; full RECALL-Q)
  feedback(query, answer) → TraceFBResult  (sync; updates w_imp via TRACE-FB)

Internal state:
  _items            dict[str, MemoryItem]
  _embeddings       np.ndarray shape (n, dim), unit-norm rows
  _centroid         np.ndarray (dim,)  — session relevance centroid
  _n_seen           int  — items processed for centroid rolling average
  _bm25             BM25Okapi  — rebuilt on each add
  _texts_for_bm25   list[str]  — parallel to _items insertion order
  _co_cite_graph    dict[tuple[str,str], float] — undirected edge weights
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional, TYPE_CHECKING

import numpy as np

from .embedder import Embedder
from .recall import recall_q_with_bm25
from .schema import MemoryItem, WVector
from .triage import TriageConfig, triage_alpha
from .trace_fb import (
    TraceFBResult,
    apply_trace_feedback,
    hallucinated_ids_from_trajectory,
)

if TYPE_CHECKING:
    pass


# CORE-DLM constants (ReSurgSAM2 arXiv:2505.08581 Eqs. 2-3, adapted:
# their IoU confidence → our TRACE attribution; their frame diversity → memory
# embedding diversity). Bounded, diversity-promoted set of credit-boosted
# memories — structural fix for rich-get-richer credit concentration.
CORE_GAMMA_CAND: float = 0.15   # attribution floor to enter candidate pool
CORE_N_POOL: int = 8            # pool size that triggers a promotion
CORE_N_CAP: int = 64            # core bank capacity (FIFO, permanent pinned)

# Co-citation graph constants
CO_CITE_RATE: float = 0.05              # edge weight growth per co-citation event
# Recalibrated from 0.15: mean_max_attribution across 1986 LoCoMo QA pairs is 0.0625
# (results/correlation_nli_gate.json) — the old 0.15 threshold (chosen for a since-removed
# NLI noise floor) sat above where ROUGE1 attribution actually lives, so almost no pair of
# co-cited memories ever cleared it and the graph stayed structurally empty (see
# results/locomo_co_cite.json: memoir_co_cite == memoir_trace to 4 decimals). Set just below
# the observed mean so edges can actually form from real, if modest, attribution signal.
MIN_ATTRIBUTION_FOR_EDGE: float = 0.05

# DEDUP-α: write-time near-duplicate suppression constant
DEDUP_COSINE_THRESHOLD: float = 0.97    # near-duplicate, not just similar


class MemoirStore:
    """
    In-memory MEMOIR store for a single agent session or evaluation run.

    Thread-safety: NOT thread-safe (same as harpo SharedMemoryStore).
    For multi-threaded eval, instantiate one store per thread.
    """

    def __init__(
        self,
        model_name:   str = "all-MiniLM-L6-v2",
        triage_cfg:   Optional[TriageConfig] = None,
        alpha_bm25:   float = 0.3,
        lambda_mmr:   float = 0.1,
        beta_floor:   float = 0.1,
        alpha_trace:  float = 0.3,   # increased from 0.1: faster w_imp accumulation
        gamma_imp:    float = 1.0,   # v4: TRACE-FB bonus weight, decoupled from w_rel
        delta_rec:    float = 0.1,   # w_rec bonus weight (was computed but unused in v5 score)
        use_nli:      bool  = False,  # deprecated no-op; NLI removed after ablation
        use_co_cite:  bool  = True,   # enable co-citation graph
        use_llm_attr: bool  = False,  # LLM fallback attribution (claude-haiku-4-5)
        use_dedup:    bool  = True,   # DEDUP-α: write-time near-duplicate suppression
        alpha_neg:    float = 0.0,    # asymmetric TRACE-FB decay rate (0 = disabled)
        use_foresight_norm: bool = False,  # FORESIGHT-NORM: path-baseline-normalized TRACE-FB
        use_weibull:  bool  = False,  # Weibull temporal decay on w_imp at retrieval time
        use_cf_attr:  bool  = False,  # TRACE-CF: counterfactual LOO attribution (local small LM)
        cf_model_name: str  = "Qwen/Qwen2.5-0.5B-Instruct",
        use_ucb:      bool  = False,  # UCB exploration bonus on w_imp (SimUtil-UCB adapted)
        use_reliability: bool = False,  # RELIABILITY: Laplace-smoothed eta() soft bonus
                                        # (oracle-feedback-only signal; MSCE-inspired, arXiv:2607.16621)
        use_learned_weights: bool = False,  # LinUCB gate: per-query retrieval weights
        use_commit:   bool  = False,  # CCOMMIT: credit effective after N credible events
        use_core_bank: bool = False,  # CORE-DLM: diversity-driven bounded core memory bank
        use_verification_gate: bool = False,  # gate TRACE credit on answer plausibility
        commit_calibration=None,  # Optional[CommitCalibration] (consolidate.py) — calibrated
                                   # CCOMMIT threshold; None = fixed COMMIT_DELTA/N_COMMIT fallback
        use_multi_anchor: bool = False,  # MULTI-ANCHOR (Item 1, Stern 2002 cognitive reserve):
                                          # cache crediting queries as auxiliary retrieval anchors
        anchor_min_attr: Optional[float] = None,  # minimum attribution to record an anchor;
                                                    # None = SPARSE_CITE_DELTA (trace_fb.py default)
        use_spacing:  bool  = False,  # SPACING (Item 2, Landauer & Bjork 1978 spacing effect):
                                       # session-gap multiplier on the TRACE-FB w_imp delta
        use_stratum_decay: bool = False,  # TYPE-EMERGENT TAXONOMY (Item 3, Tulving 1972):
                                           # semantic-stratum items (certified-merge output)
                                           # decay slower than episodic ones
        use_cf_shift_gate: bool = False,  # CF-SHIFT GATE (Item 4, controlled forgetting refined
                                           # by Anderson/Bjork/Bjork 1994): exempt high-TRACE-CF-
                                           # shift memories from Weibull/CCOMMIT/CORE-DLM friction.
                                           # Requires use_cf_attr=True to have any effect (degrades
                                           # to a safe no-op otherwise — see recall.py).
    ) -> None:
        if use_nli:
            import warnings
            warnings.warn(
                "use_nli=True is deprecated and has no effect. "
                "The NLI attribution path was removed after ablation showed regression. "
                "TRACE-FB uses ROUGE1-recall exclusively.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._embedder      = Embedder.get(model_name)
        self._triage_cfg    = triage_cfg or TriageConfig()
        self._alpha_bm25    = alpha_bm25
        self._lambda_mmr    = lambda_mmr
        self._beta_floor    = beta_floor
        self._alpha_trace   = alpha_trace
        self._gamma_imp     = gamma_imp
        self._delta_rec     = delta_rec
        self._use_co_cite   = use_co_cite
        self._use_llm_attr  = use_llm_attr
        self._use_dedup     = use_dedup
        self._alpha_neg     = alpha_neg
        self._use_foresight_norm = use_foresight_norm
        self._use_weibull   = use_weibull
        self._use_cf_attr   = use_cf_attr
        self._cf_model_name = cf_model_name
        self._use_ucb       = use_ucb
        self._use_reliability = use_reliability
        self._use_commit    = use_commit
        self._use_core_bank = use_core_bank
        self._commit_calibration = commit_calibration
        self._use_multi_anchor = use_multi_anchor
        self._anchor_min_attr = anchor_min_attr
        self._use_spacing   = use_spacing
        self._use_stratum_decay = use_stratum_decay
        self._use_cf_shift_gate = use_cf_shift_gate
        # CORE-DLM state (ReSurgSAM2 Eqs. 2-3 analog)
        self._core_pool:  list[str] = []            # candidate pool P
        self._core_bank:  list[str] = []            # bounded core bank L (FIFO)
        self._core_permanent: Optional[str] = None  # highest-attr member, never evicted
        self._core_best_attr: float = 0.0
        self._core_last_promoted: Optional[str] = None
        self._gate = None
        self._last_gate_arm: Optional[int] = None
        self._last_q_vec = None
        if use_learned_weights:
            from .gating import LinUCBGate
            self._gate = LinUCBGate(dim=self._embedder.dim)
        self.llm_attribution_count: int = 0
        self.cf_attribution_count: int = 0
        self.dedup_hit_count: int = 0
        self._z_running: dict[str, float] = {}  # FORESIGHT-NORM: per-path running baselines
        self._use_verification_gate = use_verification_gate
        self._v_running: dict[str, float] = {}  # verification gate: lp EMA baseline
        # Certified consolidation state (docs/COMBINED_PAPER_PLAN.md)
        self.consolidation_ops: list = []           # ConsolidationOp audit trail
        self._rho_incidence_counts: list[int] = []  # k_q per feedback event (ρ̂)
        # CRPC (Counterfactual Retrieval-Preservation Certificate, consolidate.py):
        # one entry per feedback_answer() call — the historical-query record CRPC
        # replays at merge-candidate time. Each entry: query text, retrieved item
        # ids in rank order, per-item score, per-item TRACE attribution, and the
        # `now` epoch used for that retrieval (so a probe replay reproduces the
        # same w_rec/Weibull conditions). Unbounded for now — CRPC only reads it
        # at sleep time, not on the online path.
        self._query_log: list[dict] = []

        self._items:               dict[str, MemoryItem]        = {}
        self._insertion_order:     list[str]                    = []
        self._embeddings:          Optional[np.ndarray]         = None
        self._matrix_dirty:        bool                         = False
        self._co_cite_graph:       dict[tuple[str, str], float] = {}
        self.co_cite_edge_count:   int                          = 0
        self._centroid:            np.ndarray                   = np.zeros(self._embedder.dim, np.float32)
        self._n_seen:              int                          = 0
        self._bm25                                              = None
        self._bm25_dirty:          bool                         = False
        self._texts_for_bm25:      list[str]                    = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def _find_duplicate(self, embedding: np.ndarray) -> Optional[MemoryItem]:
        """
        DEDUP-α: check the last session_window items for a near-duplicate
        (cosine >= DEDUP_COSINE_THRESHOLD). O(window) — same cost class as
        triage's _compute_w_conf, which scans the same window.
        """
        if not self._use_dedup:
            return None
        window = self._triage_cfg.session_window
        recent_ids = self._insertion_order[-window:]
        for rid in recent_ids:
            existing = self._items.get(rid)
            if existing is None or existing.embedding is None:
                continue
            sim = float(np.dot(embedding, existing.embedding))  # unit-norm → dot == cosine
            if sim >= DEDUP_COSINE_THRESHOLD:
                return existing
        return None

    def add(self, text: str, metadata: Optional[dict] = None,
            created_at: Optional[float] = None) -> str:
        """
        Store text immediately with a default w-vector (w_rel=0, w_imp=0).
        Fires TRIAGE-α asynchronously to compute w_rel and w_conf.
        Returns memory_id before TRIAGE-α completes.

        DEDUP-α: if a near-duplicate (cosine >= 0.97) exists in the recent
        window, the duplicate is collapsed into the existing item instead of
        creating a new one — addresses write-time noise accumulation
        (documented empirically in Mem0 GitHub issue #4573).

        created_at: defaults to wall-clock time.time(). Eval harnesses replaying
        a dataset with real timestamps (e.g. LoCoMo session_N_date_time) should
        pass the dataset-native epoch here — otherwise every item's age is
        measured from the fast wall-clock add() loop, not the simulated
        conversation timeline, and w_rec / Weibull decay become no-ops.
        """
        embedding = self._embedder.encode(text)

        dup = self._find_duplicate(embedding)
        if dup is not None:
            dup.duplicate_count += 1
            dup.bump_retrieval()
            self.dedup_hit_count += 1
            return dup.id

        item = MemoryItem(
            text=text,
            embedding=embedding,
            w=WVector(),  # defaults: w_rel=0, w_rec=1, w_conf=0, w_imp=0
            created_at=created_at if created_at is not None else time.time(),
            metadata=metadata or {},
        )
        self._items[item.id] = item
        self._insertion_order.append(item.id)
        self._matrix_dirty = True
        self._texts_for_bm25.append(text)
        self._bm25_dirty = True

        # Fire TRIAGE-α — async if event loop running, sync otherwise
        # Use only the last session_window items for conflict detection snapshot —
        # w_conf only needs local context, and limiting to 20 items avoids O(n²)
        # list allocation in eval loops where add() is called thousands of times.
        n_prev = len(self._insertion_order) - 1  # exclude the item we just added
        window = self._triage_cfg.session_window
        prev_ids = self._insertion_order[max(0, n_prev - window):n_prev]
        store_snapshot = [self._items[i] for i in prev_ids if i in self._items]
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._run_triage(item.id, store_snapshot,
                                  self._centroid.copy(), self._n_seen)
            )
        except RuntimeError:
            # No running loop (script / test context): run synchronously
            asyncio.run(
                self._run_triage(item.id, store_snapshot,
                                  self._centroid.copy(), self._n_seen)
            )
        return item.id

    def add_many(self, texts: list[str], metadata: Optional[dict] = None,
                 batch_size: int = 64,
                 created_ats: Optional[list[float]] = None) -> list[str]:
        """
        Batch-add multiple texts — 10-50× faster than sequential add() calls
        because embeddings are computed in a single GPU batch.
        TRIAGE-α is simplified: w_rel computed against rolling centroid,
        w_conf skipped (fine for eval; use add() for full triage in production).

        created_ats: optional per-item epoch timestamps (dataset-native, e.g.
        LoCoMo session_N_date_time), same rationale as add()'s created_at.
        """
        if not texts:
            return []
        embeddings = self._embedder.encode_batch(texts, batch_size=batch_size)
        ids: list[str] = []
        now = time.time()
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            dup = self._find_duplicate(embedding)
            if dup is not None:
                dup.duplicate_count += 1
                dup.bump_retrieval()
                self.dedup_hit_count += 1
                ids.append(dup.id)
                continue

            item = MemoryItem(
                text=text,
                embedding=embedding,
                w=WVector(),
                created_at=created_ats[i] if created_ats is not None else now + i * 1e-6,
                metadata=metadata or {},
            )
            self._items[item.id] = item
            self._insertion_order.append(item.id)
            self._texts_for_bm25.append(text)
            # Compute w_rel analytically against current centroid (O(1) per item)
            from .triage import _compute_w_rel, _update_centroid
            w_rel = _compute_w_rel(embedding, self._centroid, self._n_seen)
            item.w.w_rel = w_rel
            self._centroid, self._n_seen = _update_centroid(
                self._centroid, self._n_seen, embedding,
                self._triage_cfg.session_window,
            )
            ids.append(item.id)
        self._matrix_dirty = True
        self._bm25_dirty = True
        return ids

    def search(
        self,
        query: str,
        k: int = 10,
        now: Optional[float] = None,
        **ablation_flags,
    ) -> list[MemoryItem]:
        """
        Retrieve top-k memories via RECALL-Q.

        Ablation flags (all False by default):
          no_w_imp, no_bm25, no_mmr, no_w_rec, flat_w, use_weibull

        now: reference epoch for w_rec / Weibull-decay age (defaults to
        wall-clock time.time()). Pass the dataset-native "current time" when
        replaying a timestamped corpus in eval — see add(created_at=...).
        """
        scored = self.search_with_scores(query, k=k, now=now, **ablation_flags)
        return [item for item, _score in scored]

    def search_with_scores(
        self,
        query: str,
        k: int = 10,
        now: Optional[float] = None,
        **ablation_flags,
    ) -> list[tuple[MemoryItem, float]]:
        """Like search() but returns (item, score) pairs. Used by eval harnesses."""
        if not self._items:
            return []

        self._ensure_embeddings()
        self._ensure_bm25()
        if self._embeddings is None:
            return []

        q_vec = self._embedder.encode(query)
        items_list = [self._items[i] for i in self._insertion_order
                      if i in self._items]

        # LinUCB gate: per-query weight profile (deterministic given gate state,
        # so feedback_answer()'s internal search re-selects the same arm).
        # q_vec is cached unconditionally (not just when the gate is active) —
        # MULTI-ANCHOR (Item 1) needs it in feedback_answer() to record the
        # crediting query as an auxiliary anchor, independent of use_learned_weights.
        self._last_q_vec = q_vec
        alpha, lam = self._alpha_bm25, self._lambda_mmr
        beta, gamma = self._beta_floor, self._gamma_imp
        if self._gate is not None:
            arm = self._gate.select(q_vec)
            self._last_gate_arm = arm
            profile = self._gate.profiles[arm]
            alpha, lam = profile.alpha_bm25, profile.lambda_mmr
            beta, gamma = profile.beta_floor, profile.gamma_imp

        ablation_flags.setdefault("use_weibull", self._use_weibull)
        ablation_flags.setdefault("use_ucb", self._use_ucb)
        ablation_flags.setdefault("use_reliability", self._use_reliability)
        ablation_flags.setdefault("use_commit", self._use_commit)
        ablation_flags.setdefault("use_multi_anchor", self._use_multi_anchor)
        ablation_flags.setdefault("use_stratum_decay", self._use_stratum_decay)
        ablation_flags.setdefault("use_cf_shift_gate", self._use_cf_shift_gate)
        if self._commit_calibration is not None:
            ablation_flags.setdefault("commit_llr_threshold", self._commit_calibration.llr_threshold)
        co_cite_graph = self._co_cite_graph if (self._use_co_cite and self._co_cite_graph) else None
        core_ids = set(self._core_bank) if self._use_core_bank else None
        return recall_q_with_bm25(
            query=query,
            query_vec=q_vec,
            store_items=items_list,
            embeddings_matrix=self._embeddings,
            bm25=self._bm25,
            k=k,
            alpha=alpha,
            lam=lam,
            beta_floor=beta,
            gamma_imp=gamma,
            delta_rec=self._delta_rec,
            co_cite_graph=co_cite_graph,
            now=now,
            core_ids=core_ids,
            **ablation_flags,
        )

    def feedback_answer(
        self,
        query: str,
        answer: str,
        k: int = 10,
        trajectory=None,
        oracle_correct: Optional[bool] = None,
        reliability_correct: Optional[bool] = None,
        now: Optional[float] = None,
        session_idx: Optional[int] = None,
        **ablation_flags,
    ) -> TraceFBResult:
        """
        Run TRACE-FB: retrieve memories for query, then update w_imp based on
        how much of each memory's content appeared in the generated answer.

        Args:
            query:          the retrieval query used to generate the answer
            answer:         the LLM-generated answer text
            trajectory:     optional AgentTrajectory for hallucination detection
            oracle_correct: if not None, use binary oracle signal (True/False)
                            AS THE ATTRIBUTION MECHANISM (replaces ROUGE1/CF)
            reliability_correct: if not None, use this correctness signal ONLY
                            to update n_pass/n_trial (RELIABILITY, schema.py
                            MemoryItem.eta()) — decoupled from oracle_correct so
                            reliability tracking can run alongside ROUGE1/CF
                            attribution without switching the attribution path.
                            Falls back to oracle_correct when None (oracle_trace
                            naturally reuses its existing binary signal).
            now:            reference epoch for w_rec / Weibull age (see search())
            session_idx:    SPACING (Item 2) — current session index, used to
                            compute sessions-since-last-credit per memory. Only
                            has an effect when use_spacing=True; callers not
                            running the spacing ablation may omit it.
        """
        retrieved = self.search_with_scores(query, k=k, now=now, **ablation_flags)
        # CRPC query log: record BEFORE attribution runs so the log reflects the
        # actual retrieval this query saw, independent of whether feedback below
        # ends up gated/abstained.
        self._query_log.append({
            "query": query,
            "retrieved_ids": [item.id for item, _score in retrieved],
            "ranks": {item.id: rank for rank, (item, _score) in enumerate(retrieved)},
            "scores": {item.id: score for item, score in retrieved},
            "now": now,
        })
        hallucinated = hallucinated_ids_from_trajectory(trajectory)
        result = apply_trace_feedback(
            retrieved=retrieved,
            answer=answer,
            store=self,
            alpha=self._alpha_trace,
            hallucinated_ids=hallucinated if hallucinated else None,
            oracle_correct=oracle_correct,
            query=query,
            use_llm_attr=self._use_llm_attr,
            alpha_neg=self._alpha_neg,
            use_foresight_norm=self._use_foresight_norm,
            z_running=self._z_running,
            use_cf_attr=self._use_cf_attr,
            cf_model_name=self._cf_model_name,
            use_verification_gate=self._use_verification_gate,
            v_running=self._v_running,
            commit_calibration=self._commit_calibration,
            query_vec=self._last_q_vec,
            use_multi_anchor=self._use_multi_anchor,
            anchor_min_attr=self._anchor_min_attr,
            session_idx=session_idx,
            use_spacing=self._use_spacing,
        )
        self._query_log[-1]["attributions"] = dict(result.attributions)
        # RELIABILITY: Laplace-smoothed success tracking (MemoryItem.eta()).
        # oracle_correct is a per-query signal (did the retrieved context lead
        # to a correct answer), applied to every item retrieved for this
        # query — only accrues under oracle/GT feedback (oracle_correct is
        # None in production/ROUGE1/CF modes, where n_trial stays 0 and
        # eta() stays neutral at 0.5).
        reliability_signal = (reliability_correct if reliability_correct is not None
                             else oracle_correct)
        if reliability_signal is not None:
            for item, _score in retrieved:
                item.n_trial += 1
                if reliability_signal:
                    item.n_pass += 1
        self.llm_attribution_count += sum(
            1 for v in result.attribution_paths.values() if v == "llm"
        )
        self.cf_attribution_count += sum(
            1 for v in result.attribution_paths.values() if v == "cf"
        )
        # A fully-gated (abstention) answer carries no co-citation, core-bank,
        # rho, or bandit-reward evidence either — the whole downstream credit
        # cascade is one gated unit, not just the w_imp update.
        gated_out = self._use_verification_gate and result.verification_gate <= 0.0
        if self._use_co_cite and not gated_out:
            self._update_co_cite_graph(
                retrieved=retrieved,
                attributions=result.attributions,
            )
        if self._use_core_bank and not gated_out:
            self._update_core_bank(result.attributions)
        if self.consolidation_ops and not gated_out:
            self._record_rho_event(result.attributions)
        # LinUCB gate update: reward = max attribution over the retrieved set —
        # the same outcome signal TRACE-FB already produces, no extra labels.
        if self._gate is not None and self._last_gate_arm is not None and not gated_out:
            reward = max(result.attributions.values(), default=0.0)
            self._gate.update(self._last_q_vec, self._last_gate_arm,
                              min(1.0, max(0.0, reward)))
        return result

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        return self._items.get(memory_id)

    def __len__(self) -> int:
        return len(self._items)

    def all_items(self) -> list[MemoryItem]:
        return [self._items[i] for i in self._insertion_order if i in self._items]

    def get_co_cite_neighbors(self, item_id: str) -> dict[str, float]:
        """Return {neighbor_id: edge_weight} for all co-cite edges involving item_id."""
        result: dict[str, float] = {}
        for (a, b), w in self._co_cite_graph.items():
            if a == item_id:
                result[b] = w
            elif b == item_id:
                result[a] = w
        return result

    # ── Internal ───────────────────────────────────────────────────────────────

    def _update_co_cite_graph(
        self,
        retrieved: list[tuple[MemoryItem, float]],
        attributions: dict[str, float],
    ) -> None:
        """
        Form or strengthen co-citation edges between memories with attribution >= threshold.
        Called at end of feedback_answer(); never called from add() or search().
        Edge key = tuple(sorted([id_a, id_b])) for undirected storage.
        """
        cited_items = [
            (item, attributions[item.id])
            for item, _ in retrieved
            if item.id in attributions
            and attributions[item.id] >= MIN_ATTRIBUTION_FOR_EDGE
        ]
        if self._use_commit:
            # CCOMMIT: only committed memories may form co-cite edges —
            # stops a single spurious attribution from seeding false edges.
            # Calibrated mode (if loaded) supersedes the fixed event count.
            if self._commit_calibration is not None:
                thr = self._commit_calibration.llr_threshold
                cited_items = [(it, a) for it, a in cited_items if it.commit_llr >= thr]
            else:
                from .trace_fb import N_COMMIT
                cited_items = [(it, a) for it, a in cited_items
                               if it.commit_count >= N_COMMIT]
        if len(cited_items) < 2:
            return
        for i, (item_a, attr_a) in enumerate(cited_items):
            for item_b, attr_b in cited_items[i + 1:]:
                key = tuple(sorted([item_a.id, item_b.id]))
                delta = min(attr_a, attr_b) * CO_CITE_RATE
                self._co_cite_graph[key] = min(1.0, self._co_cite_graph.get(key, 0.0) + delta)
        self.co_cite_edge_count = len(self._co_cite_graph)

    def _update_core_bank(self, attributions: dict[str, float]) -> None:
        """
        CORE-DLM promotion (ReSurgSAM2 Eqs. 2-3 analog). High-attribution
        memories enter the candidate pool; when the pool fills, promote the
        candidate most dissimilar to the last-promoted core member. The
        highest-attribution memory ever seen is pinned (their "initial frame
        permanent" analog).
        """
        in_bank = set(self._core_bank)
        for mid, attr in attributions.items():
            if attr < CORE_GAMMA_CAND or mid in in_bank or mid in self._core_pool:
                continue
            if mid not in self._items:
                continue
            self._core_pool.append(mid)
            if attr > self._core_best_attr:
                self._core_best_attr = attr
                self._core_permanent = mid

        if len(self._core_pool) < CORE_N_POOL:
            return

        # Promote: first promotion = permanent (highest-attr) candidate if
        # pooled, else most-diverse vs last promoted member
        if self._core_last_promoted is None:
            chosen = (self._core_permanent
                      if self._core_permanent in self._core_pool
                      else self._core_pool[0])
        else:
            ref = self._items.get(self._core_last_promoted)
            if ref is None or ref.embedding is None:
                chosen = self._core_pool[0]
            else:
                sims = []
                for mid in self._core_pool:
                    item = self._items.get(mid)
                    if item is None or item.embedding is None:
                        sims.append((1.0, mid))
                        continue
                    sims.append((float(np.dot(item.embedding, ref.embedding)), mid))
                chosen = min(sims)[1]  # unit-norm → dot == cosine

        self._core_pool = []
        self._core_bank.append(chosen)
        self._core_last_promoted = chosen
        if len(self._core_bank) > CORE_N_CAP:
            # FIFO eviction, permanent member pinned
            for i, mid in enumerate(self._core_bank):
                if mid != self._core_permanent:
                    self._core_bank.pop(i)
                    break

    def prune(self, target_ratio: float,
              lambda_red: float = 0.3, lambda_rec: float = 0.1) -> int:
        """
        PRUNE-EFP (SurgSAM2 arXiv:2408.07931 analog): reversibly exclude
        redundant low-value memories from retrieval. Greedy keep-selection by
        keep_score = w_imp + lambda_red·(1 − max_cos_to_kept) + lambda_rec·w_rec,
        computed incrementally (one matvec per kept item, like MMR).
        Returns the number of pruned items. target_ratio=1.0 → no-op.
        """
        items = [self._items[i] for i in self._insertion_order if i in self._items]
        items = [m for m in items if m.embedding is not None]
        n = len(items)
        n_keep = int(np.ceil(target_ratio * n))
        if n == 0 or n_keep >= n:
            return 0

        embs = np.stack([m.embedding for m in items])
        base = np.array([m.w.w_imp + lambda_rec * m.w.w_rec for m in items],
                        dtype=np.float32)
        max_cos = np.full(n, -1.0, dtype=np.float32)
        kept_mask = np.zeros(n, dtype=bool)

        for _ in range(n_keep):
            scores = base + lambda_red * (1.0 - np.clip(max_cos, 0.0, 1.0))
            scores[kept_mask] = -np.inf
            pick = int(np.argmax(scores))
            kept_mask[pick] = True
            max_cos = np.maximum(max_cos, embs @ embs[pick])

        pruned = 0
        for i, m in enumerate(items):
            m.pruned = not kept_mask[i]
            pruned += int(not kept_mask[i])
        return pruned

    def unprune_all(self) -> None:
        for m in self._items.values():
            m.pruned = False

    # ── Certified consolidation (prospective half — docs/COMBINED_PAPER_PLAN.md) ──

    def total_tokens(self) -> int:
        """Whitespace-token rate r(Z) over live (non-pruned, non-superseded,
        non-hallucinated) items."""
        return sum(len(m.text.split()) for m in self._items.values()
                   if not m.pruned and not m.superseded and not m.hallucinated)

    @property
    def rho_hat(self) -> float:
        """
        Running support–operation incidence estimate: E_q[# applied ops touching
        the TRACE-observed support S_q], over feedback events with non-empty S_q
        (Bridge 1 — replaces the certified repo's offline probe-based ρ)."""
        if not self._rho_incidence_counts:
            return 0.0
        return float(np.mean(self._rho_incidence_counts))

    def consolidate(
        self,
        judge,
        merge_generate_fn=None,
        sim_threshold: float = 0.6,
        co_cite_sim_floor: float = 0.45,
        max_clusters: int = 16,
        max_cluster_size: int = 4,
        protect_w_imp: float = 0.3,
        credit_transfer: str = "max",
        retrieval_coverage_min: float = 0.6,
        max_prune_checks: int = 32,
        prune_neighbor_sim: float = 0.6,
        use_co_cite_clusters: bool = True,
        certified: bool = True,
        audit_path: Optional[str] = None,
        tau_prune: Optional[float] = None,
        max_merge_ratio: float = 0.9,
        max_supersede_checks: int = 0,
        tau_supersede: Optional[float] = None,
        max_digest_checks: int = 0,
        digest_generate_fn=None,
        tau_digest: Optional[float] = None,
        retrieval_guard: str = "lexical",
        tau_lex: Optional[float] = None,
        tau_anchor: float = 0.9,
        tau_query: float = 0.8,
        min_probes: int = 3,
        max_probes: int = 10,
        crpc_min_attr: float = 0.15,
        merge_regeneration_attempts: int = 0,
        log_counterfactual_probes: Optional[str] = None,
    ) -> dict:
        """
        One sleep pass: certified merges + dominated prunes over the live store.

        judge: FaithfulnessJudge (consolidate.py). With `certified=False` the
        judge gate is bypassed (MemRefine-style uncertified merge — the ablation
        isolating the certificate) but scores are still recorded for the audit.
        merge_generate_fn: callable(prompt)->str for merge-text generation;
        defaults to judge.generate_fn.

        max_merge_ratio: length gate — a merge is only certified if
        len(merged) <= max_merge_ratio * sum(len(originals)). A judge-certified,
        coverage-passing merge can still fail to actually compress (measured:
        llama3.1:8b inflated tokens +5.3% net across 44 applied merges,
        results/consolidation_full10.json) — content-safe but rate-negative
        merges defeat the point of consolidation. Uncertified bypasses this too
        (certified=False), matching the other two gates.

        Credit protection: items with w_imp > protect_w_imp never enter prune
        candidacy, and clusters containing them require a UNANIMOUS certificate
        (score == 1.0) regardless of the judge's calibrated threshold.

        Execution is reversible: merged-away constituents and pruned items get
        pruned=True (excluded from retrieval); nothing is deleted. Every op is
        recorded in self.consolidation_ops (and appended to audit_path JSONL).

        tau_prune: separate certify threshold for dominated-prune, defaulting
        to judge.certify_threshold if None. The single-sentence entailment
        check ("does this neighbor entail this candidate") empirically scores
        much lower than the merge ensemble on the same judge/prompt family —
        sharing one tau starves prunes; give it its own operating point.

        max_supersede_checks: SUPERSEDE operator — certified retirement of
        contradicted facts (the missing UPDATE lifecycle op). Candidate pairs
        come from TRIAGE-α's negation-conflict signal applied pairwise;
        the newer item supersedes the older only if the judge's 3-framing
        ensemble certifies it (tau_supersede, defaults to tau). Default 0
        (off) until validated — same conservative rollout as prune.

        retrieval_guard: which retrieval-preservation gate a merge candidate
        must also clear (in addition to the judge certificate and length
        gate) — one of consolidate.CRPC_MODES:
          "lexical" (default) — current baseline, kappa_lex >= tau_lex
          "none"    — no retrieval-preservation gate at all
          "anchor"  — exact-anchor recall (entities/dates/numbers/negations/
                      speakers) >= tau_anchor, no retriever replay
          "crpc"    — counterfactual retrieval replay: query_retention >=
                      tau_query AND anchor_recall >= tau_anchor, using up to
                      max_probes historical queries per cluster (TRACE-credited
                      ones preferred); clusters with fewer than min_probes
                      historical queries are REJECTED outright (conservative)
          "hybrid"  — same as "crpc" but sparse-history clusters fall back to
                      the lexical gate instead of being rejected
        merge_regeneration_attempts: on a retrieval_guard rejection, retry
        merge-text generation up to this many times with the missing anchors
        appended to the prompt as a hint, then re-run the SAME gate once per
        attempt. 0 = no regeneration (default).
        log_counterfactual_probes: optional JSONL path — one line per
        evaluated candidate with every CRPC diagnostic (kappa_lex,
        anchor_recall, query_retention, mean_rank_shift, per-probe
        pre/post ranks), regardless of whether it was applied.

        Returns summary stats dict.
        """
        from .consolidate import (
            ConsolidationOp, generate_merge_text, judge_supersedes,
            propose_clusters, propose_supersede_pairs,
            retrieval_token_coverage, transfer_credit,
            transfer_supersede_credit,
            generate_digest_text, judge_digest_faithful,
            propose_session_groups, transfer_digest_credit,
            crpc_certificate, anchor_recall, extract_anchors,
        )
        merge_generate_fn = merge_generate_fn or judge.generate_fn
        digest_generate_fn = digest_generate_fn or merge_generate_fn
        tau_digest = judge.certify_threshold if tau_digest is None else tau_digest
        tau_prune = judge.certify_threshold if tau_prune is None else tau_prune
        tau_lex = retrieval_coverage_min if tau_lex is None else tau_lex
        _crpc_log_f = (open(log_counterfactual_probes, "a")
                      if log_counterfactual_probes else None)
        tau_supersede = judge.certify_threshold if tau_supersede is None \
            else tau_supersede
        tokens_before = self.total_tokens()
        ops_start = len(self.consolidation_ops)

        live = [self._items[i] for i in self._insertion_order
                if i in self._items
                and not self._items[i].pruned and not self._items[i].superseded
                and not self._items[i].hallucinated]

        # ── Phase 1: certified merges ──────────────────────────────────────────
        # protect_w_imp is enforced here (not in propose_clusters) so protected
        # items can still merge under a unanimous certificate.
        clusters = propose_clusters(
            live,
            sim_threshold=sim_threshold,
            max_clusters=max_clusters,
            max_cluster_size=max_cluster_size,
            co_cite_graph=self._co_cite_graph if use_co_cite_clusters else None,
            co_cite_sim_floor=co_cite_sim_floor,
            protect_w_imp=2.0,  # no exclusion at proposal; gated below
        )
        applied_merges = 0
        for cluster in clusters:
            originals = [m.text for m in cluster]
            cluster_ids_set = {m.id for m in cluster}
            regen_hint = ""
            ok = False
            merged_text = ""
            score = coverage = 0.0
            guard_result: Optional[dict] = None
            for attempt in range(max(0, merge_regeneration_attempts) + 1):
                try:
                    prompt_hint = regen_hint
                    merged_text = generate_merge_text(
                        merge_generate_fn, cluster) if not prompt_hint else \
                        generate_merge_text(
                            lambda p: merge_generate_fn(p + "\n\n" + prompt_hint),
                            cluster)
                except Exception:
                    break
                if not merged_text:
                    break
                score = judge.score(merged_text, originals)
                coverage = retrieval_token_coverage(merged_text, originals)
                has_protected = any(m.w.w_imp > protect_w_imp or m.sparse_evidence
                                    for m in cluster)
                threshold = 1.0 if has_protected else judge.certify_threshold
                # Length gate: a judge-certified, coverage-passing merge can still
                # fail to compress (measured empirically — llama3.1:8b inflated
                # tokens +5.3% net on 44 applied merges, results/consolidation_full10.json
                # — the "shorter than the sum of inputs" instruction in MERGE_PROMPT
                # isn't reliably followed on short conversational-turn clusters).
                # Consolidation's entire point is the rate axis; a content-safe
                # merge that grows the store is not a merge worth keeping.
                orig_len = sum(len(t.split()) for t in originals)
                merged_len = len(merged_text.split())
                compresses = merged_len <= max_merge_ratio * orig_len if orig_len else True
                semantic_ok = (score >= threshold and compresses) if certified else True

                # Retrieval-preservation guard (CRPC or its ablation modes) —
                # runs a temporary candidate through the same gate; on
                # rejection nothing is written to the store yet.
                guard_ok = True
                if certified and semantic_ok and retrieval_guard != "none":
                    candidate_preview = MemoryItem(
                        text=merged_text,
                        embedding=self._embedder.encode(merged_text),
                        w=WVector(w_rel=max(m.w.w_rel for m in cluster)),
                        created_at=max(m.created_at for m in cluster),
                        metadata={}, stratum="semantic",
                    )
                    guard_result = crpc_certificate(
                        self, cluster, candidate_preview, merged_text, originals,
                        mode=retrieval_guard, tau_lex=tau_lex, tau_anchor=tau_anchor,
                        tau_query=tau_query, min_probes=min_probes,
                        max_probes=max_probes, min_attr=crpc_min_attr,
                        k=5,  # matches the paper's k=5 evaluation default;
                              # not the same "k" as consolidate()'s other params
                        credit_transfer=credit_transfer,
                    )
                    guard_ok = guard_result["passed"]
                    if _crpc_log_f is not None:
                        _crpc_log_f.write(json.dumps({
                            "constituent_ids": list(cluster_ids_set),
                            "attempt": attempt, "semantic_score": score,
                            **{k2: v for k2, v in guard_result.items()
                              if k2 != "replay_rows"},
                        }, default=str) + "\n")
                elif retrieval_guard == "none":
                    guard_result = {"mode": "none", "passed": True, "reason": "no_retrieval_guard"}

                ok = semantic_ok and guard_ok
                if ok or attempt >= merge_regeneration_attempts:
                    break
                # Regeneration hint (spec sec. 8): tell the generator which
                # anchors/tokens it dropped, then retry once (or up to
                # merge_regeneration_attempts times).
                missing_anchor_terms: list[str] = []
                if guard_result is not None:
                    src_anchor_sets = [extract_anchors(o) for o in originals]
                    merged_anchor_sets = extract_anchors(merged_text)
                    for cat in ("entities", "dates", "numbers", "negations", "speakers"):
                        src = set().union(*(s.get(cat, set()) for s in src_anchor_sets)) \
                            if src_anchor_sets else set()
                        missing_anchor_terms.extend(sorted(src - merged_anchor_sets.get(cat, set())))
                regen_hint = (
                    "The previous merge attempt dropped required details. "
                    "Ensure the merged entry includes: " + ", ".join(missing_anchor_terms)
                ) if missing_anchor_terms else ""

            op = ConsolidationOp(
                op_type="merge", applied=ok,
                constituent_ids=[m.id for m in cluster],
                result_id=None, judge_score=score,
                retrieval_coverage=coverage,
                constituent_texts=originals, result_text=merged_text,
            )
            if ok:
                embedding = self._embedder.encode(merged_text)
                merged = MemoryItem(
                    text=merged_text,
                    embedding=embedding,
                    w=WVector(w_rel=max(m.w.w_rel for m in cluster)),
                    created_at=max(m.created_at for m in cluster),
                    metadata={},
                    # TYPE-EMERGENT TAXONOMY (Item 3, Tulving 1972): a certified
                    # merge is exactly the "same fact observed enough times to
                    # generalize" event — tag the merged node semantic so it can
                    # get a slower decay rate (recall.py STRATUM_LAMBDA_REC,
                    # gated by use_stratum_decay). Un-merged singletons stay
                    # "episodic" (MemoryItem default).
                    stratum="semantic",
                )
                transfer_credit(merged, cluster,
                                co_cite_graph=self._co_cite_graph,
                                mode=credit_transfer)
                self._items[merged.id] = merged
                self._insertion_order.append(merged.id)
                self._texts_for_bm25.append(merged_text)
                for m in cluster:
                    m.pruned = True
                op.result_id = merged.id
                applied_merges += 1
            # Rejected candidates: every source memory stays active and
            # unchanged (nothing above ever set .pruned on a rejected
            # cluster's members) -- logged via `op` regardless of outcome.
            self.consolidation_ops.append(op)
        if _crpc_log_f is not None:
            _crpc_log_f.close()

        # ── Phase 1.5: certified supersede (contradiction retirement) ──────────
        applied_supersedes = 0
        if max_supersede_checks > 0:
            live = [self._items[i] for i in self._insertion_order
                    if i in self._items
                    and not self._items[i].pruned
                    and not self._items[i].superseded
                    and not self._items[i].hallucinated]
            for older, newer in propose_supersede_pairs(
                    live, max_pairs=max_supersede_checks):
                # Credit protection applies to the OLDER item — retiring a
                # load-bearing memory needs a unanimous certificate, exactly
                # like merging one.
                threshold = 1.0 if (older.w.w_imp > protect_w_imp
                                    or older.sparse_evidence) else tau_supersede
                score = judge_supersedes(judge, older.text, newer.text)
                ok = score >= threshold if certified else True
                op = ConsolidationOp(
                    op_type="supersede", applied=ok,
                    constituent_ids=[older.id], result_id=newer.id,
                    judge_score=score, retrieval_coverage=1.0,
                    constituent_texts=[older.text], result_text=newer.text,
                )
                if ok:
                    older.superseded = True
                    transfer_supersede_credit(newer, older, self._co_cite_graph)
                    applied_supersedes += 1
                self.consolidation_ops.append(op)

        # ── Phase 2: dominated prunes (certified redundancy) ───────────────────
        live = [self._items[i] for i in self._insertion_order
                if i in self._items
                and not self._items[i].pruned and not self._items[i].superseded
                and not self._items[i].hallucinated]
        applied_prunes = 0
        if len(live) >= 2 and max_prune_checks > 0:
            vecs = np.stack([m.embedding for m in live if m.embedding is not None])
            by_pos = [m for m in live if m.embedding is not None]
            # lowest-credit first; credit-protected items are never prune candidates
            candidates = sorted(
                (m for m in by_pos
                 if m.w.w_imp <= protect_w_imp and not m.sparse_evidence),
                key=lambda m: m.w.w_imp,
            )[:max_prune_checks]
            pos = {m.id: i for i, m in enumerate(by_pos)}
            removed: set[str] = set()
            protected: set[str] = set()  # cited entailers survive the pass
            for cand in candidates:
                if cand.id in protected or cand.id in removed:
                    continue
                sims = vecs @ vecs[pos[cand.id]]
                order = np.argsort(sims)[::-1]
                neighbor = None
                for j in order:
                    other = by_pos[int(j)]
                    if other.id == cand.id or other.id in removed:
                        continue
                    if float(sims[int(j)]) < prune_neighbor_sim:
                        break
                    neighbor = other
                    break
                if neighbor is None:
                    continue
                score = judge.score(neighbor.text, [cand.text])
                ok = score >= tau_prune if certified else True
                op = ConsolidationOp(
                    op_type="prune", applied=ok,
                    constituent_ids=[cand.id], result_id=neighbor.id,
                    judge_score=score, retrieval_coverage=1.0,
                    constituent_texts=[cand.text], result_text=neighbor.text,
                )
                if ok:
                    cand.pruned = True
                    removed.add(cand.id)
                    protected.add(neighbor.id)
                    applied_prunes += 1
                self.consolidation_ops.append(op)

        # ── Phase 3: certified session digests (additive synthesis) ────────────
        # Targets a diagnosed generation-time weakness (docs/COMBINED_PAPER_PLAN.md,
        # 2026-07-20/21 entries): raw dialogue-turn context makes it hard for the
        # answering LLM to extract facts that require connecting information
        # across turns, which good-retrieval/bad-accuracy comparisons pinned on
        # generation, not retrieval. Unlike merge, constituents are NEVER pruned
        # — additive only, so raw turns stay retrievable and TRACE-CF's
        # per-memory credit signal doesn't collapse the way summarize_all_to_core's
        # destructive collapse did (r_post→0.0).
        applied_digests = 0
        if max_digest_checks > 0:
            live = [self._items[i] for i in self._insertion_order
                    if i in self._items
                    and not self._items[i].pruned and not self._items[i].superseded
                    and not self._items[i].hallucinated]
            groups = propose_session_groups(live, max_groups=max_digest_checks)
            for group in groups:
                originals = [m.text for m in group]
                try:
                    digest_text = generate_digest_text(digest_generate_fn, group)
                except Exception:
                    continue
                if not digest_text:
                    continue
                score = judge_digest_faithful(judge, digest_text, originals)
                ok = score >= tau_digest if certified else True
                op = ConsolidationOp(
                    op_type="digest", applied=ok,
                    constituent_ids=[m.id for m in group],
                    result_id=None, judge_score=score,
                    retrieval_coverage=1.0,
                    constituent_texts=originals, result_text=digest_text,
                )
                if ok:
                    embedding = self._embedder.encode(digest_text)
                    digest = MemoryItem(
                        text=digest_text,
                        embedding=embedding,
                        w=WVector(w_rel=max(m.w.w_rel for m in group)),
                        created_at=max(m.created_at for m in group),
                        metadata={},
                        # TYPE-EMERGENT TAXONOMY (Item 3): additive session
                        # digests are also a generalization event (they
                        # synthesize across multiple raw turns), so tag
                        # semantic — same rationale as merge acceptance above.
                        stratum="semantic",
                    )
                    transfer_digest_credit(digest, group, mode=credit_transfer)
                    self._items[digest.id] = digest
                    self._insertion_order.append(digest.id)
                    self._texts_for_bm25.append(digest_text)
                    op.result_id = digest.id
                    applied_digests += 1
                self.consolidation_ops.append(op)

        if applied_merges or applied_prunes or applied_supersedes or applied_digests:
            # recall.py excludes pruned items BY POSITION, keeping matrix/BM25
            # rows aligned with insertion order — no index compaction needed
            # (and none wanted: unprune_all() must stay able to restore).
            self._matrix_dirty = True
            self._bm25_dirty = True

        if audit_path:
            with open(audit_path, "a") as f:
                for op in self.consolidation_ops[ops_start:]:
                    f.write(op.to_json() + "\n")

        return {
            "applied_merges": applied_merges,
            "applied_prunes": applied_prunes,
            "applied_supersedes": applied_supersedes,
            "applied_digests": applied_digests,
            "proposed_clusters": len(clusters),
            "tokens_before": tokens_before,
            "tokens_after": self.total_tokens(),
            "judge_calls": judge.call_count,
        }

    def consolidate_to_budget(
        self,
        judge,
        target_ratio: float = 0.8,
        max_passes: int = 4,
        **consolidate_kwargs,
    ) -> dict:
        """
        Compression-targeted consolidation: repeated certified passes until
        r(Z) <= target_ratio * r(X) or a pass applies no operations.

        Why a loop, not one bigger pass: a single pass at the safety-first
        defaults (max_clusters=8) touches only ~2% of store tokens — measured
        on the full10 run, capping store-level compression at ~0.5% even with
        perfectly-compressing merge text. Repeated passes re-cluster the
        SHRINKING store, so pass-2 merges can combine pass-1 merge results —
        hierarchical consolidation under the same per-op certificate. Each
        pass is individually certified; there is no compression-rate
        guarantee (the certificate can legitimately refuse everything),
        which is exactly the tension the paper's RQ5 measures: how much
        compression is reachable WITHOUT giving up the certificate.

        Every kwarg accepted by consolidate() passes through; per-pass caps
        (max_clusters etc.) apply per pass, not in total. Returns combined
        stats with per-pass breakdown.
        """
        tokens_start = self.total_tokens()
        target_tokens = target_ratio * tokens_start
        passes: list[dict] = []
        totals = {"applied_merges": 0, "applied_prunes": 0,
                  "applied_supersedes": 0, "judge_calls": 0}

        for _ in range(max_passes):
            if self.total_tokens() <= target_tokens:
                break
            stats = self.consolidate(judge, **consolidate_kwargs)
            passes.append(stats)
            for k in totals:
                totals[k] += stats.get(k, 0)
            applied = (stats["applied_merges"] + stats["applied_prunes"]
                       + stats["applied_supersedes"])
            if applied == 0:
                break  # certificate refused everything left — honest stop

        return {
            **totals,
            "tokens_before": tokens_start,
            "tokens_after": self.total_tokens(),
            "target_ratio": target_ratio,
            "achieved_ratio": self.total_tokens() / max(1, tokens_start),
            "target_met": self.total_tokens() <= target_tokens,
            "n_passes": len(passes),
            "passes": passes,
        }

    def _record_rho_event(self, attributions: dict[str, float]) -> None:
        """
        Bridge 1: after each feedback event, S_q = credited memory ids; count
        applied ops whose constituents or result intersect S_q. Appends k_q.
        """
        s_q = {mid for mid, a in attributions.items()
               if a >= MIN_ATTRIBUTION_FOR_EDGE}
        if not s_q:
            return
        # expand credited merged items back to their constituents
        expanded = set(s_q)
        for mid in list(s_q):
            item = self._items.get(mid)
            if item is not None:
                expanded.update(item.metadata.get("merged_from", []))
        k_q = sum(
            1 for op in self.consolidation_ops
            if op.applied and (set(op.constituent_ids) & expanded
                               or (op.result_id and op.result_id in expanded))
        )
        self._rho_incidence_counts.append(k_q)

    async def _run_triage(
        self,
        item_id: str,
        store_snapshot: list[MemoryItem],
        centroid: np.ndarray,
        n_seen: int,
    ) -> None:
        """Background TRIAGE-α task. Updates item's w-vector and session centroid."""
        item = self._items.get(item_id)
        if item is None:
            return
        updated_item, new_centroid, new_n_seen = await triage_alpha(
            item=item,
            store_snapshot=store_snapshot,
            centroid=centroid,
            n_seen=n_seen,
            config=self._triage_cfg,
        )
        # Write back (item is mutated in-place by triage_alpha)
        self._centroid = new_centroid
        self._n_seen   = new_n_seen

    def _rebuild_matrix(self) -> None:
        """Rebuild the (n, dim) embedding matrix after add()."""
        items = [self._items[i] for i in self._insertion_order if i in self._items]
        if not items:
            self._embeddings = None
            return
        vecs = [item.embedding for item in items if item.embedding is not None]
        if vecs:
            self._embeddings = np.stack(vecs, axis=0)

    def _ensure_embeddings(self) -> None:
        """Build embedding matrix if items have been added since last build (lazy)."""
        if self._matrix_dirty:
            self._rebuild_matrix()
            self._matrix_dirty = False

    def _ensure_bm25(self) -> None:
        """Rebuild BM25 index if texts have been added since last rebuild (lazy)."""
        if self._bm25_dirty and self._texts_for_bm25:
            self._rebuild_bm25()
            self._bm25_dirty = False

    def _rebuild_bm25(self) -> None:
        """Build BM25 index from all current texts. O(n) once vs O(n²) incremental."""
        from .bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in self._texts_for_bm25]
        self._bm25 = BM25Okapi(tokenized)

    def _update_item_wvector(self, item_id: str, new_w: WVector) -> None:
        """Called by triage if needed to explicitly set w-vector from outside."""
        item = self._items.get(item_id)
        if item:
            item.w = new_w
