"""Unit tests for the certified SUPERSEDE operator (contradiction retirement)."""

import sys
import types

import numpy as np
import pytest

# Patch sentence_transformers before importing the store (no GPU/download).
st = types.ModuleType("sentence_transformers")


class _FakeST:
    def __init__(self, *a, **kw):
        pass

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, **kw):
        def one(t):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.random(8).astype("float32") - 0.5
            return v / np.linalg.norm(v)
        if isinstance(texts, list):
            return np.stack([one(t) for t in texts])
        return one(texts)


st.SentenceTransformer = _FakeST
sys.modules.setdefault("sentence_transformers", st)

from harpo.memoir.consolidate import (
    FaithfulnessJudge,
    is_conflict_pair,
    judge_supersedes,
    propose_supersede_pairs,
    transfer_supersede_credit,
    SUPERSEDE_CREDIT_DISCOUNT,
)
from harpo.memoir.schema import MemoryItem, WVector
from harpo.memoir.store import MemoirStore


def _item(text, created_at, w_imp=0.0, vec=None):
    m = MemoryItem(text=text, created_at=created_at)
    m.w = WVector(w_imp=w_imp)
    if vec is not None:
        v = np.asarray(vec, dtype="float32")
        m.embedding = v / np.linalg.norm(v)
    return m


class TestConflictPair:
    def test_negation_conflict(self):
        assert is_conflict_pair("Python is fast", "Python is not fast")

    def test_correction_marker(self):
        assert is_conflict_pair("She lives in Paris",
                                "Correction: she lives in Lyon actually")

    def test_no_conflict_without_negation(self):
        assert not is_conflict_pair("Python is fast", "Python is popular")

    def test_no_conflict_without_content_overlap(self):
        assert not is_conflict_pair("Python is fast", "The trip is not happening")


class TestProposePairs:
    def test_orders_older_first(self):
        a = _item("The wedding is not cancelled", 100.0, vec=[1, 0, 0, 0, 0, 0, 0, 0])
        b = _item("The wedding is cancelled", 200.0, vec=[1, 0.05, 0, 0, 0, 0, 0, 0])
        pairs = propose_supersede_pairs([b, a])   # insertion order irrelevant
        assert len(pairs) == 1
        older, newer = pairs[0]
        assert older is a and newer is b

    def test_dissimilar_pairs_excluded(self):
        a = _item("The wedding is not cancelled", 100.0, vec=[1, 0, 0, 0, 0, 0, 0, 0])
        b = _item("The wedding is cancelled", 200.0, vec=[0, 1, 0, 0, 0, 0, 0, 0])
        assert propose_supersede_pairs([a, b]) == []

    def test_non_conflicting_similar_pairs_excluded(self):
        a = _item("Alice lives in Paris", 100.0, vec=[1, 0, 0, 0, 0, 0, 0, 0])
        b = _item("Alice resides in Paris", 200.0, vec=[1, 0.05, 0, 0, 0, 0, 0, 0])
        assert propose_supersede_pairs([a, b]) == []

    def test_max_pairs_cap(self):
        items = []
        for i in range(6):
            items.append(_item(f"fact {i} is true", 100.0 + i,
                               vec=[1, 0.01 * i, 0, 0, 0, 0, 0, 0]))
            items.append(_item(f"fact {i} is not true", 200.0 + i,
                               vec=[1, 0.01 * i + 0.005, 0, 0, 0, 0, 0, 0]))
        assert len(propose_supersede_pairs(items, max_pairs=3)) == 3


class TestJudgeSupersedes:
    def test_unanimous(self):
        j = FaithfulnessJudge(lambda p: "SUPERSEDES", n_votes=3)
        assert judge_supersedes(j, "old", "new") == 1.0

    def test_independent(self):
        j = FaithfulnessJudge(lambda p: "INDEPENDENT", n_votes=3)
        assert judge_supersedes(j, "old", "new") == 0.0

    def test_exception_counts_as_independent(self):
        def boom(p):
            raise RuntimeError("api down")
        j = FaithfulnessJudge(boom, n_votes=3)
        assert judge_supersedes(j, "old", "new") == 0.0

    def test_garbage_counts_as_independent(self):
        j = FaithfulnessJudge(lambda p: "maybe? SUPERSEDES later", n_votes=3)
        assert judge_supersedes(j, "old", "new") == 0.0


class TestCreditTransfer:
    def test_discounted_transfer(self):
        older = _item("old fact", 100.0, w_imp=0.8)
        newer = _item("new fact", 200.0, w_imp=0.1)
        transfer_supersede_credit(newer, older, co_cite_graph=None)
        assert abs(newer.w.w_imp - SUPERSEDE_CREDIT_DISCOUNT * 0.8) < 1e-9
        assert older.id in newer.metadata["supersedes"]

    def test_keeps_own_higher_credit(self):
        older = _item("old fact", 100.0, w_imp=0.2)
        newer = _item("new fact", 200.0, w_imp=0.9)
        transfer_supersede_credit(newer, older, co_cite_graph=None)
        assert newer.w.w_imp == 0.9

    def test_edge_migration(self):
        older = _item("old fact", 100.0, w_imp=0.4)
        newer = _item("new fact", 200.0)
        other = _item("unrelated", 50.0)
        graph = {tuple(sorted([older.id, other.id])): 0.6}
        transfer_supersede_credit(newer, older, co_cite_graph=graph)
        key = tuple(sorted([newer.id, other.id]))
        assert key in graph
        assert abs(graph[key] - SUPERSEDE_CREDIT_DISCOUNT * 0.6) < 1e-9
        assert tuple(sorted([older.id, other.id])) not in graph


class TestStoreSupersede:
    def _setup(self, judge_response="SUPERSEDES", **store_kw):
        s = MemoirStore(use_dedup=False, **store_kw)
        id_old = s.add("The wedding is not cancelled")
        id_new = s.add("The wedding is cancelled")
        v = np.zeros(8, dtype="float32"); v[0] = 1.0
        s._items[id_old].embedding = v
        w = np.array([1, 0.05, 0, 0, 0, 0, 0, 0], dtype="float32")
        s._items[id_new].embedding = w / np.linalg.norm(w)
        s._items[id_old].created_at = 100.0
        s._items[id_new].created_at = 200.0
        s._matrix_dirty = True
        judge = FaithfulnessJudge(lambda p: judge_response, n_votes=3,
                                  certify_threshold=1.0)
        return s, id_old, id_new, judge

    def test_certified_supersede_applies(self):
        s, id_old, id_new, judge = self._setup()
        stats = s.consolidate(judge, max_clusters=0, max_prune_checks=0,
                              max_supersede_checks=4)
        assert stats["applied_supersedes"] == 1
        assert s._items[id_old].superseded
        assert not s._items[id_new].superseded
        # superseded item is out of retrieval and out of r(Z)
        texts = [m.text for m in s.search("wedding", k=5)]
        assert "The wedding is not cancelled" not in texts

    def test_independent_verdict_blocks(self):
        s, id_old, id_new, judge = self._setup(judge_response="INDEPENDENT")
        stats = s.consolidate(judge, max_clusters=0, max_prune_checks=0,
                              max_supersede_checks=4)
        assert stats["applied_supersedes"] == 0
        assert not s._items[id_old].superseded

    def test_uncertified_bypasses_judge(self):
        s, id_old, id_new, judge = self._setup(judge_response="INDEPENDENT")
        stats = s.consolidate(judge, certified=False, max_clusters=0,
                              max_prune_checks=0, max_supersede_checks=4)
        assert stats["applied_supersedes"] == 1

    def test_off_by_default(self):
        s, id_old, id_new, judge = self._setup()
        stats = s.consolidate(judge, max_clusters=0, max_prune_checks=0)
        assert stats["applied_supersedes"] == 0
        assert not s._items[id_old].superseded

    def test_credit_protection_requires_unanimity(self):
        s, id_old, id_new, judge = self._setup()
        s._items[id_old].w.w_imp = 0.9   # protected
        # 2-of-3 judge: one framing says INDEPENDENT
        calls = {"n": 0}

        def two_of_three(p):
            calls["n"] += 1
            return "SUPERSEDES" if calls["n"] % 3 else "INDEPENDENT"
        judge = FaithfulnessJudge(two_of_three, n_votes=3, certify_threshold=0.6)
        stats = s.consolidate(judge, max_clusters=0, max_prune_checks=0,
                              max_supersede_checks=4)
        assert stats["applied_supersedes"] == 0   # 2/3 < unanimity for protected

    def test_audit_records_op(self):
        s, id_old, id_new, judge = self._setup()
        s.consolidate(judge, max_clusters=0, max_prune_checks=0,
                      max_supersede_checks=4)
        sup_ops = [o for o in s.consolidation_ops if o.op_type == "supersede"]
        assert len(sup_ops) == 1
        assert sup_ops[0].applied
        assert sup_ops[0].constituent_ids == [id_old]
        assert sup_ops[0].result_id == id_new
