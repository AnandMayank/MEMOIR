"""Unit tests for TRIAGE-α quality gating."""

import asyncio
import numpy as np
import pytest

from harpo.memoir.schema import MemoryItem, WVector
from harpo.memoir.triage import (
    TriageConfig,
    _compute_w_conf,
    _compute_w_rel,
    _update_centroid,
    triage_alpha,
    _tokenize,
)


def _make_item(text: str, embedding: np.ndarray) -> MemoryItem:
    item = MemoryItem(text=text, embedding=embedding)
    return item


def _unit_vec(dim: int = 4, idx: int = 0) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


class TestUpdateCentroid:
    def test_first_item(self):
        centroid = np.zeros(4, dtype=np.float32)
        emb = _unit_vec(4, 0)
        new_c, n = _update_centroid(centroid, 0, emb, window=20)
        assert n == 1
        assert np.allclose(new_c, emb, atol=1e-6)

    def test_rolling_window(self):
        centroid = _unit_vec(4, 0)
        emb = _unit_vec(4, 1)
        new_c, n = _update_centroid(centroid, 20, emb, window=20)
        assert n == 21
        # centroid should shift toward emb
        assert new_c[1] > 0


class TestComputeWRel:
    def test_identical_embedding(self):
        emb = _unit_vec(4, 0)
        score = _compute_w_rel(emb, emb, n_seen=5)
        assert abs(score - 1.0) < 1e-5

    def test_orthogonal_embedding(self):
        emb     = _unit_vec(4, 0)
        centroid = _unit_vec(4, 1)
        score = _compute_w_rel(emb, centroid, n_seen=5)
        assert score < 0.1

    def test_neutral_on_first_item(self):
        emb = _unit_vec(4, 0)
        centroid = np.zeros(4, dtype=np.float32)
        score = _compute_w_rel(emb, centroid, n_seen=0)
        assert score == 0.5


class TestComputeWConf:
    def test_no_conflict_when_store_empty(self):
        item = _make_item("Python is good", _unit_vec(4, 0))
        assert _compute_w_conf(item, [], threshold=0.85) == 0.0

    def test_conflict_detected_on_negation(self):
        # two items with very similar embeddings; new one has negation
        emb = _unit_vec(4, 0)
        existing = _make_item("Python is fast", emb.copy())
        new_item = _make_item("Python is not fast", emb.copy())
        score = _compute_w_conf(new_item, [existing], threshold=0.5)
        assert score > 0.0

    def test_no_conflict_without_negation(self):
        emb = _unit_vec(4, 0)
        existing = _make_item("Python is fast", emb.copy())
        new_item = _make_item("Python is quick", emb.copy())
        score = _compute_w_conf(new_item, [existing], threshold=0.5)
        assert score == 0.0

    def test_no_embedding_skipped(self):
        item = _make_item("test", None)
        assert _compute_w_conf(item, [], threshold=0.85) == 0.0


class TestTriageAlpha:
    def test_basic_run(self):
        centroid = np.zeros(4, dtype=np.float32)
        emb = _unit_vec(4, 0)
        item = _make_item("test memory", emb)
        cfg = TriageConfig()

        updated, new_centroid, n = asyncio.run(
            triage_alpha(item, [], centroid, 0, cfg)
        )
        assert updated.w.w_rel == 0.5  # neutral on first item
        assert updated.w.w_imp == 0.0  # never set by triage
        assert n == 1

    def test_w_rel_updates_after_centroid_built(self):
        emb0 = _unit_vec(4, 0)
        emb1 = _unit_vec(4, 1)

        item0 = _make_item("memory 0", emb0)
        centroid = np.zeros(4, dtype=np.float32)

        _, centroid, n = asyncio.run(triage_alpha(item0, [], centroid, 0))

        # Second item: same direction as centroid (high w_rel expected)
        item1 = _make_item("memory 1", emb0.copy())
        updated, _, _ = asyncio.run(triage_alpha(item1, [item0], centroid, n))
        assert updated.w.w_rel > 0.8

    def test_no_embedding_is_noop(self):
        centroid = np.zeros(4, dtype=np.float32)
        item = _make_item("no embedding", None)
        updated, new_c, n = asyncio.run(triage_alpha(item, [], centroid, 0))
        assert n == 0
        assert np.allclose(new_c, centroid)
