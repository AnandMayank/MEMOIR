"""Unit tests for RECALL-Q scoring."""

import numpy as np
import pytest

from harpo.memoir.schema import MemoryItem, WVector
from harpo.memoir.recall import recall_q_with_bm25


def _make_item(text: str, w_rel: float = 0.8, w_imp: float = 0.5) -> MemoryItem:
    item = MemoryItem(text=text)
    item.w.w_rel = w_rel
    item.w.w_imp = w_imp
    # Assign a random unit-norm embedding
    np.random.seed(hash(text) % (2**31))
    emb = np.random.randn(4).astype(np.float32)
    item.embedding = emb / np.linalg.norm(emb)
    return item


def _build_matrix(items: list[MemoryItem]) -> np.ndarray:
    return np.stack([i.embedding for i in items], axis=0)


def _build_bm25(texts: list[str]):
    from harpo.memoir.bm25 import BM25Okapi
    return BM25Okapi([t.lower().split() for t in texts])


class TestRecallQ:
    def setup_method(self):
        self.items = [
            _make_item("Python is a fast language", w_rel=0.9, w_imp=0.8),
            _make_item("Java enterprise software", w_rel=0.3, w_imp=0.1),
            _make_item("data science machine learning", w_rel=0.7, w_imp=0.6),
        ]
        self.matrix = _build_matrix(self.items)
        self.bm25 = _build_bm25([i.text for i in self.items])

    def _query_vec(self, idx: int = 0) -> np.ndarray:
        return self.items[idx].embedding

    def test_returns_up_to_k(self):
        q = self._query_vec(0)
        results = recall_q_with_bm25("python", q, self.items, self.matrix, self.bm25, k=2)
        assert len(results) <= 2

    def test_hallucinated_excluded(self):
        self.items[0].hallucinated = True
        q = self._query_vec(0)
        results = recall_q_with_bm25("python", q, self.items, self.matrix, self.bm25, k=3)
        ids = [item.id for item, _ in results]
        assert self.items[0].id not in ids

    def test_flat_w_flag(self):
        """flat_w=True should set w_rel=1, eff_imp=1 — changes scoring but not crash."""
        q = self._query_vec(0)
        results = recall_q_with_bm25(
            "python", q, self.items, self.matrix, self.bm25, k=3, flat_w=True
        )
        assert len(results) > 0

    def test_no_bm25_flag(self):
        q = self._query_vec(0)
        results = recall_q_with_bm25(
            "python", q, self.items, self.matrix, self.bm25, k=3, no_bm25=True
        )
        assert len(results) > 0

    def test_no_mmr_flag(self):
        q = self._query_vec(0)
        results = recall_q_with_bm25(
            "python", q, self.items, self.matrix, self.bm25, k=3, no_mmr=True
        )
        assert len(results) > 0

    def test_empty_store(self):
        q = np.zeros(4, dtype=np.float32)
        results = recall_q_with_bm25("python", q, [], None, None, k=5)
        assert results == []

    def test_retrieval_count_bumped(self):
        q = self._query_vec(0)
        before = self.items[0].retrieval_count
        recall_q_with_bm25("python", q, self.items, self.matrix, self.bm25, k=3)
        # At least one item should have its retrieval_count bumped
        assert any(i.retrieval_count > before for i in self.items)
