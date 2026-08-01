"""
Unit tests for MEMOIR v3 components:
  CCOMMIT  — credible credit commitment (CIFS analog)
  CORE-DLM — diversity-driven core memory bank (DLM analog)
  PRUNE-EFP — credit-aware reversible retrieval pruning (SurgSAM2 analog)

Uses the same unit-norm stub embedder pattern as test_co_cite_graph.py.
"""

import sys
import types

import numpy as np
import pytest

# Stub sentence_transformers before harpo imports (unit-norm — store assumes
# dot == cosine; see test_co_cite_graph.py)
st = types.ModuleType("sentence_transformers")


class _FakeST:
    def __init__(self, *a, **kw):
        pass

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, **kw):
        if isinstance(texts, list):
            vecs = np.random.rand(len(texts), 8).astype("float32")
            return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        vec = np.random.rand(8).astype("float32")
        return vec / np.linalg.norm(vec)


st.SentenceTransformer = _FakeST
sys.modules.setdefault("sentence_transformers", st)

from harpo.memoir.recall import recall_q_with_bm25, COMMIT_N, CORE_DISCOUNT
from harpo.memoir.schema import MemoryItem, WVector
from harpo.memoir.store import (
    CORE_GAMMA_CAND,
    CORE_N_POOL,
    MemoirStore,
)
from harpo.memoir.trace_fb import COMMIT_DELTA, N_COMMIT, apply_trace_feedback

DIM = 8


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _item(direction: int, w_imp: float = 0.0, commit_count: int = 0,
          text: str = "t") -> MemoryItem:
    v = np.zeros(DIM, dtype=np.float32)
    v[direction % DIM] = 1.0
    return MemoryItem(text=text, embedding=v, w=WVector(w_imp=w_imp),
                      commit_count=commit_count)


def _recall(items, q_dir=0, k=5, **flags):
    q = np.zeros(DIM, dtype=np.float32)
    q[q_dir] = 1.0
    matrix = np.stack([m.embedding for m in items])
    return recall_q_with_bm25(
        query="q", query_vec=q, store_items=items, embeddings_matrix=matrix,
        bm25=None, k=k, no_mmr=True, no_w_rec=True, **flags)


# ── CCOMMIT ────────────────────────────────────────────────────────────────────

def test_commit_count_incremented_at_noise_floor():
    store = MemoirStore(use_co_cite=False, use_dedup=False)
    a = _item(0, text="alpha beta gamma")
    retrieved = [(a, 0.9)]
    # attribution above COMMIT_DELTA: answer shares most memory tokens
    apply_trace_feedback(retrieved, "alpha beta gamma", store, alpha=0.3)
    assert a.commit_count == 1
    apply_trace_feedback(retrieved, "alpha beta gamma", store, alpha=0.3)
    assert a.commit_count == 2


def test_commit_not_incremented_below_floor():
    store = MemoirStore(use_co_cite=False, use_dedup=False)
    a = _item(0, text="alpha beta gamma delta epsilon zeta eta theta")
    apply_trace_feedback([(a, 0.9)], "unrelated words entirely", store, alpha=0.3)
    assert a.commit_count == 0


def test_uncommitted_w_imp_invisible_with_flag():
    # Two items equally cosine-similar; one has credit but no commits.
    credited = _item(0, w_imp=0.8, commit_count=0)
    plain = _item(0, w_imp=0.0)
    scored = _recall([credited, plain], use_commit=True)
    scores = dict((id(m), s) for m, s in scored)
    assert abs(scores[id(credited)] - scores[id(plain)]) < 1e-6


def test_committed_w_imp_visible():
    credited = _item(0, w_imp=0.8, commit_count=COMMIT_N)
    plain = _item(0, w_imp=0.0)
    scored = _recall([credited, plain], use_commit=True)
    scores = dict((id(m), s) for m, s in scored)
    assert scores[id(credited)] > scores[id(plain)]


def test_commit_flag_off_is_noop():
    credited = _item(0, w_imp=0.8, commit_count=0)
    plain = _item(0, w_imp=0.0)
    scored = _recall([credited, plain])
    scores = dict((id(m), s) for m, s in scored)
    assert scores[id(credited)] > scores[id(plain)]


def test_commit_constants_consistent():
    assert N_COMMIT == COMMIT_N


# ── CORE-DLM ───────────────────────────────────────────────────────────────────

def _store_with_items(n: int) -> tuple[MemoirStore, list[MemoryItem]]:
    s = MemoirStore(use_co_cite=False, use_dedup=False, use_core_bank=True)
    items = []
    for i in range(n):
        it = _item(i, text=f"memory {i}")
        s._items[it.id] = it
        s._insertion_order.append(it.id)
        items.append(it)
    s._matrix_dirty = True
    return s, items


def test_core_pool_fills_and_promotes():
    s, items = _store_with_items(CORE_N_POOL + 2)
    # feed attributions above the candidate floor one at a time
    for it in items[:CORE_N_POOL - 1]:
        s._update_core_bank({it.id: CORE_GAMMA_CAND + 0.1})
    assert len(s._core_bank) == 0  # pool not yet full
    s._update_core_bank({items[CORE_N_POOL - 1].id: CORE_GAMMA_CAND + 0.1})
    assert len(s._core_bank) == 1
    assert s._core_pool == []  # pool cleared after promotion


def test_first_promotion_is_highest_attribution():
    s, items = _store_with_items(CORE_N_POOL)
    attrs = {it.id: CORE_GAMMA_CAND + 0.01 * (i + 1)
             for i, it in enumerate(items)}
    best_id = items[-1].id  # highest attribution
    for mid, a in attrs.items():
        s._update_core_bank({mid: a})
    assert s._core_bank[0] == best_id
    assert s._core_permanent == best_id


def test_second_promotion_is_most_diverse():
    # First promotion fills the bank with a known member; the next promotion
    # must pick the pool candidate LEAST similar to it.
    s, items = _store_with_items(2 * CORE_N_POOL)
    # First round: same direction items (0..N_POOL-1 are orthogonal here, so
    # instead pin the last-promoted explicitly)
    for it in items[:CORE_N_POOL]:
        s._update_core_bank({it.id: CORE_GAMMA_CAND + 0.1})
    ref = s._items[s._core_last_promoted]
    # Second round pool: one item aligned with ref, others orthogonal.
    aligned = MemoryItem(text="aligned", embedding=ref.embedding.copy(),
                         w=WVector())
    s._items[aligned.id] = aligned
    s._insertion_order.append(aligned.id)
    pool2 = [aligned] + items[CORE_N_POOL: 2 * CORE_N_POOL - 1]
    for it in pool2:
        s._update_core_bank({it.id: CORE_GAMMA_CAND + 0.1})
    assert len(s._core_bank) == 2
    promoted_2nd = s._core_bank[1]
    assert promoted_2nd != aligned.id  # most-diverse, not the aligned clone


def test_core_discount_in_recall():
    core = _item(0, w_imp=0.8)
    non_core = _item(0, w_imp=0.8)
    q = np.zeros(DIM, dtype=np.float32)
    q[0] = 1.0
    matrix = np.stack([core.embedding, non_core.embedding])
    scored = recall_q_with_bm25(
        query="q", query_vec=q, store_items=[core, non_core],
        embeddings_matrix=matrix, bm25=None, k=2, no_mmr=True, no_w_rec=True,
        core_ids={core.id})
    scores = dict((m.id, s) for m, s in scored)
    # non-core gets CORE_DISCOUNT × bonus < full bonus
    assert scores[core.id] > scores[non_core.id]
    expected_gap = 0.8 * (1 - CORE_DISCOUNT)  # cos=1, gamma_imp=1
    assert scores[core.id] - scores[non_core.id] == pytest.approx(expected_gap, abs=1e-5)


# ── PRUNE-EFP ──────────────────────────────────────────────────────────────────

def test_prune_excludes_and_is_reversible():
    s, items = _store_with_items(6)
    n_pruned = s.prune(0.5)
    assert n_pruned == 3
    assert sum(m.pruned for m in items) == 3
    s.unprune_all()
    assert sum(m.pruned for m in items) == 0


def test_prune_keeps_high_credit():
    s, items = _store_with_items(4)
    items[2].w.w_imp = 0.9  # clearly most valuable
    s.prune(0.25)  # keep 1
    assert not items[2].pruned
    assert sum(m.pruned for m in items) == 3


def test_pruned_items_not_retrieved():
    a = _item(0, w_imp=0.0)
    b = _item(0, w_imp=0.0)
    b.pruned = True
    scored = _recall([a, b], k=5)
    ids = {m.id for m, _ in scored}
    assert a.id in ids and b.id not in ids


def test_prune_ratio_one_is_noop():
    s, items = _store_with_items(5)
    assert s.prune(1.0) == 0
    assert sum(m.pruned for m in items) == 0


def test_prune_prefers_diverse_over_redundant():
    # Three copies of one direction + one distinct direction, equal credit:
    # keeping 2 must include the distinct one (redundancy term).
    s = MemoirStore(use_co_cite=False, use_dedup=False)
    dup_items = [_item(0) for _ in range(3)]
    distinct = _item(1)
    for it in dup_items + [distinct]:
        s._items[it.id] = it
        s._insertion_order.append(it.id)
    s.prune(0.5)  # keep 2
    assert not distinct.pruned
