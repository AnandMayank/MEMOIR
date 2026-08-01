"""
MEMOIR data models.

WVector: 4-dimensional analytical quality score computed at write time (async).
MemoryItem: one stored memory with embedding and quality scores.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class WVector:
    """
    4-dimensional analytical quality decomposition.

    Dimensions:
      w_rel  — cosine(embedding, session_centroid): relevance to current session
      w_rec  — exp(-λ·age_hours) × (1 + log1p(retrieval_count)): recency × usage
      w_conf — fraction of close memories (cos > 0.85) with negation overlap: conflict signal
      w_imp  — ROUGE1-recall attribution from TRACE-FB; 0.0 until outcome feedback runs
    """
    w_rel:  float = 0.0
    w_rec:  float = 1.0
    w_conf: float = 0.0
    w_imp:  float = 0.0

    def effective_imp(self, beta_floor: float = 0.1) -> float:
        """
        Additive floor so fresh memories (w_imp=0) score via cosine at beta_floor,
        while TRACE-FB updates add proportionally on top.
        Replaces max(w_imp, beta_floor) which masked all updates below the floor.
        """
        return beta_floor + self.w_imp

    def as_dict(self) -> dict[str, float]:
        return {"w_rel": self.w_rel, "w_rec": self.w_rec,
                "w_conf": self.w_conf, "w_imp": self.w_imp}


@dataclass
class MemoryItem:
    """One stored memory unit."""

    id:               str             = field(default_factory=lambda: str(uuid.uuid4()))
    text:             str             = ""
    embedding:        Optional[Any]   = field(default=None, repr=False)  # np.ndarray (384,)
    w:                WVector         = field(default_factory=WVector)
    created_at:       float           = field(default_factory=time.time)
    retrieval_count:  int             = 0
    hallucinated:     bool            = False
    metadata:         dict            = field(default_factory=dict)
    duplicate_count:  int             = 0  # DEDUP-α: near-duplicate writes collapsed into this item
    commit_count:     int             = 0  # CCOMMIT: feedback events with attribution ≥ noise floor
                                           # (fixed-threshold fallback when no calibration is loaded)
    commit_llr:       float           = 0.0  # CCOMMIT calibrated mode: accumulated log-likelihood-
                                           # ratio from CommitCalibration.llr_increment (consolidate.py)
    cite_count:       int             = 0  # sparse-evidence guard: events with attribution ≥
                                           # SPARSE_CITE_DELTA (a lower, independent bar than
                                           # COMMIT_DELTA — see trace_fb.py for why they must differ)
    pruned:           bool            = False  # PRUNE-EFP: excluded from retrieval (reversible)
    superseded:       bool            = False  # SUPERSEDE: retired by a certified newer contradicting fact
                                               # (reversible; deliberately NOT reset by unprune_all)
    sparse_evidence:  bool            = False  # single citation on a multi-evidence answer — exempt from
                                               # credit-friction (decay/commit/core-discount); cleared once
                                               # the memory earns a second credible citation
    n_pass:           int             = 0  # RELIABILITY: retrievals under oracle feedback that led to
                                           # a correct answer (oracle_correct=True)
    n_trial:          int             = 0  # RELIABILITY: retrievals under oracle feedback total
                                           # (oracle_correct is not None) — denominator for eta()

    # ── Cognition.md extensions (all default-off via store-level use_* flags;
    #    fields themselves are inert until their flag reads them) ──────────────
    anchors:          list            = field(default_factory=list)  # MULTI-ANCHOR (Item 1,
                                           # Stern 2002 cognitive reserve): auxiliary query
                                           # embeddings (np.ndarray, unit-norm) that earned this
                                           # item real TRACE credit — independent retrieval paths
                                           # to the same memory. Capped at MAX_ANCHORS (recall.py).
    anchor_attrs:     list            = field(default_factory=list)  # parallel attribution scores
                                           # for anchors, used to evict the weakest on overflow.
    last_credit_session: Optional[int] = None  # SPACING (Item 2, Landauer & Bjork 1978 spacing
                                           # effect): session index of this item's most recent
                                           # positive TRACE credit; None until first credit.
    stratum:          str             = "episodic"  # TYPE-EMERGENT TAXONOMY (Item 3, Tulving 1972):
                                           # "episodic" | "semantic". Sole population/decay
                                           # policy live outside this dataclass — see consolidate()
                                           # (sets "semantic" on certified-merge acceptance) and
                                           # recall.py's STRATUM_LAMBDA_REC (decay-rate lookup).
    cf_shift:         float           = 0.0  # CF-SHIFT GATE (Item 4, Ebbinghaus 1885 controlled
                                           # forgetting, refined by Anderson/Bjork/Bjork 1994
                                           # selective retrieval-induced forgetting): cached
                                           # TRACE-CF leave-one-out attribution shift, last
                                           # observed value (trace_fb.py Phase 1.5). Used to
                                           # exempt load-bearing sparse memories from friction
                                           # (Weibull/CCOMMIT/CORE-DLM) in recall.py.

    def add_anchor(self, query_vec: "np.ndarray", attr: float, max_anchors: int = 3) -> None:
        """
        MULTI-ANCHOR (Item 1): record a query embedding that earned this item
        real TRACE credit as an auxiliary retrieval path. Evicts the anchor
        with the lowest recorded attribution when at capacity, so the anchor
        set converges on the item's strongest, most distinct retrieval paths
        rather than growing unbounded. query_vec is assumed unit-norm (same
        embedder as the base `embedding`) — required for the max-cosine read
        path in recall.py to remain a valid similarity.
        """
        if len(self.anchors) < max_anchors:
            self.anchors.append(query_vec)
            self.anchor_attrs.append(attr)
            return
        min_idx = min(range(len(self.anchor_attrs)), key=lambda i: self.anchor_attrs[i])
        if attr > self.anchor_attrs[min_idx]:
            self.anchors[min_idx] = query_vec
            self.anchor_attrs[min_idx] = attr

    def eta(self) -> float:
        """
        Laplace-smoothed success rate (adapted from MSCE's η, arXiv:2607.16621
        §4.3): (n_pass+1)/(n_trial+2). Reduces to 0.5 for untried items —
        neutral, neither penalized nor rewarded. Only accrues signal under
        oracle/GT feedback (n_trial stays 0 in production/ROUGE1/CF modes).
        """
        return (self.n_pass + 1) / (self.n_trial + 2)

    def live_w_rec(self, lambda_rec: float = 0.1, now: Optional[float] = None) -> float:
        """
        Recompute w_rec based on current age. Called at retrieval, not cached.

        `now` defaults to wall-clock time.time(), which is correct in production
        (memories and queries share real time) but collapses to ~0 age in fast
        eval harnesses that add+query a whole multi-session dataset within
        seconds. Eval code simulating dataset-native timestamps should pass
        `now` explicitly (see MemoirStore.add(created_at=...) / search(now=...)).
        """
        if now is None:
            now = time.time()
        age_hours = (now - self.created_at) / 3600.0
        return math.exp(-lambda_rec * age_hours) * (1.0 + math.log1p(self.retrieval_count))

    def bump_retrieval(self) -> None:
        self.retrieval_count += 1
