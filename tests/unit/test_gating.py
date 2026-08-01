"""Unit tests for GATE-LinUCB (gating.py) and the UCB w_imp bonus (recall.py)."""

from __future__ import annotations

import numpy as np

from harpo.memoir.gating import DEFAULT_PROFILES, LinUCBGate
from harpo.memoir.recall import recall_q_with_bm25
from harpo.memoir.schema import MemoryItem, WVector

DIM = 16


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


# ── LinUCBGate ─────────────────────────────────────────────────────────────────

def test_select_returns_valid_arm():
    gate = LinUCBGate(dim=DIM)
    q = _unit(np.ones(DIM))
    arm = gate.select(q)
    assert 0 <= arm < len(DEFAULT_PROFILES)


def test_select_deterministic_given_state():
    gate = LinUCBGate(dim=DIM)
    q = _unit(np.arange(1, DIM + 1, dtype=np.float32))
    assert gate.select(q) == gate.select(q)


def test_gate_learns_rewarded_arm():
    gate = LinUCBGate(dim=DIM)
    rng = np.random.default_rng(7)
    target_arm = 2
    # Reward only target_arm for queries near a fixed direction
    direction = _unit(rng.standard_normal(DIM))
    for _ in range(200):
        q = _unit(direction + 0.1 * rng.standard_normal(DIM))
        arm = gate.select(q)
        gate.update(q, arm, 1.0 if arm == target_arm else 0.0)
    # After training, the target arm should dominate selection on-distribution
    wins = sum(
        gate.select(_unit(direction + 0.1 * rng.standard_normal(DIM))) == target_arm
        for _ in range(50)
    )
    assert wins >= 40


def test_update_keeps_a_inv_symmetric():
    gate = LinUCBGate(dim=DIM)
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = _unit(rng.standard_normal(DIM))
        gate.update(q, 0, float(rng.random()))
    A_inv = gate._A_inv[0]
    assert np.allclose(A_inv, A_inv.T, atol=1e-4)


# ── UCB bonus in RECALL-Q ──────────────────────────────────────────────────────

def _item(vec: np.ndarray, w_imp: float = 0.0, retrieval_count: int = 0) -> MemoryItem:
    return MemoryItem(
        text="t", embedding=_unit(vec), w=WVector(w_imp=w_imp),
        retrieval_count=retrieval_count,
    )


def test_ucb_boosts_never_retrieved_item():
    q = np.zeros(DIM, dtype=np.float32)
    q[0] = 1.0
    # Two items equally similar to the query, equal w_imp; one heavily
    # retrieved, one never retrieved. With use_ucb the fresh item must
    # outrank the exploited one.
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    exploited = _item(v, w_imp=0.2, retrieval_count=200)
    fresh = _item(v, w_imp=0.2, retrieval_count=0)
    items = [exploited, fresh]
    matrix = np.stack([m.embedding for m in items])

    scored = recall_q_with_bm25(
        query="q", query_vec=q, store_items=items, embeddings_matrix=matrix,
        bm25=None, k=2, no_mmr=True, no_w_rec=True, use_ucb=True,
    )
    scores = {id(m): s for m, s in [(m, s) for m, s in scored]}
    assert scores[id(fresh)] > scores[id(exploited)]


def test_ucb_flag_off_is_noop():
    q = np.zeros(DIM, dtype=np.float32)
    q[0] = 1.0
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    a = _item(v, w_imp=0.2, retrieval_count=200)
    b = _item(v, w_imp=0.2, retrieval_count=0)
    items = [a, b]
    matrix = np.stack([m.embedding for m in items])
    scored = recall_q_with_bm25(
        query="q", query_vec=q, store_items=items, embeddings_matrix=matrix,
        bm25=None, k=2, no_mmr=True, no_w_rec=True,
    )
    s = [score for _, score in scored]
    assert abs(s[0] - s[1]) < 1e-6
