"""Integration tests: MemoirStore.consolidate() with a stub judge/embedder."""

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
        # Unit-norm like the real encoder — the store assumes dot == cosine
        def one(t):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.random(8).astype("float32") - 0.5
            return v / np.linalg.norm(v)
        if isinstance(texts, list):
            return np.stack([one(t) for t in texts])
        return one(texts)


st.SentenceTransformer = _FakeST
sys.modules.setdefault("sentence_transformers", st)

from harpo.memoir.consolidate import FaithfulnessJudge
from harpo.memoir.store import MemoirStore


def _store_with(*texts, **kw):
    s = MemoirStore(use_dedup=False, **kw)
    ids = [s.add(t) for t in texts]
    return s, ids


def _force_embeddings(store, ids, vectors):
    for mid, vec in zip(ids, vectors):
        v = np.asarray(vec, dtype="float32")
        store._items[mid].embedding = v / np.linalg.norm(v)
    store._matrix_dirty = True


LOSSLESS_JUDGE = lambda p: "LOSSLESS"
LOSSY_JUDGE = lambda p: "LOSSY"


def _merge_or_judge(response):
    """generate_fn serving both merge-text generation and judge prompts."""
    def fn(prompt):
        if "consolidation agent" in prompt:
            return "merged: all facts combined"
        return response
    return fn


class TestStoreConsolidate:
    def test_certified_merge_applies_and_compacts(self):
        s, ids = _store_with("Alice lives in Paris", "Alice resides in Paris city",
                             "Bob likes tennis")
        _force_embeddings(s, ids, [[1, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0.05, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 0]])
        judge = FaithfulnessJudge(_merge_or_judge("LOSSLESS"), n_votes=3)
        stats = s.consolidate(judge, retrieval_coverage_min=0.0,
                              max_prune_checks=0)
        assert stats["applied_merges"] == 1
        assert stats["tokens_after"] < stats["tokens_before"]
        live = [m for m in s.all_items() if not m.pruned]
        assert any(m.metadata.get("merged_from") for m in live)
        # constituents reversibly pruned, not deleted
        assert s._items[ids[0]].pruned and s._items[ids[1]].pruned
        assert not s._items[ids[2]].pruned

    def test_lossy_verdict_blocks_merge(self):
        s, ids = _store_with("a b", "a b c")
        _force_embeddings(s, ids, [[1, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0.05, 0, 0, 0, 0, 0, 0]])
        judge = FaithfulnessJudge(_merge_or_judge("LOSSY"), n_votes=3)
        stats = s.consolidate(judge, max_prune_checks=0)
        assert stats["applied_merges"] == 0
        assert not s._items[ids[0]].pruned
        # rejected op still recorded for the audit
        assert len(s.consolidation_ops) == 1
        assert s.consolidation_ops[0].applied is False

    def test_uncertified_mode_applies_despite_lossy(self):
        s, ids = _store_with("a b", "a b c")
        _force_embeddings(s, ids, [[1, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0.05, 0, 0, 0, 0, 0, 0]])
        judge = FaithfulnessJudge(_merge_or_judge("LOSSY"), n_votes=3)
        stats = s.consolidate(judge, certified=False, max_prune_checks=0)
        assert stats["applied_merges"] == 1
        # score still recorded — auditability of the uncertified ablation
        assert s.consolidation_ops[0].judge_score == 0.0

    def test_protected_item_needs_unanimity(self):
        s, ids = _store_with("high credit fact", "its twin")
        _force_embeddings(s, ids, [[1, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0.05, 0, 0, 0, 0, 0, 0]])
        s._items[ids[0]].w.w_imp = 0.8  # > protect_w_imp
        # 2/3 lossless — passes a calibrated 0.6 threshold but NOT unanimity
        responses = iter(["LOSSLESS", "LOSSY", "LOSSLESS"] * 10)

        def gen(prompt):
            if "consolidation agent" in prompt:
                return "merged text"
            return next(responses)

        judge = FaithfulnessJudge(gen, n_votes=3, certify_threshold=0.6)
        stats = s.consolidate(judge, protect_w_imp=0.3,
                              retrieval_coverage_min=0.0, max_prune_checks=0)
        assert stats["applied_merges"] == 0

    def test_dominated_prune_protects_entailer(self):
        s, ids = _store_with("Paris is the capital of France",
                             "France's capital is Paris",
                             "unrelated fact about tennis")
        _force_embeddings(s, ids, [[1, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0.02, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 0]])
        judge = FaithfulnessJudge(LOSSLESS_JUDGE, n_votes=1)
        stats = s.consolidate(judge, max_clusters=0, max_prune_checks=10,
                              prune_neighbor_sim=0.9)
        assert stats["applied_prunes"] == 1
        pruned = [m for m in s.all_items() if m.pruned]
        kept = [m for m in s.all_items() if not m.pruned]
        # exactly one of the twins pruned; its entailer survived the pass
        assert len(pruned) == 1
        assert len(kept) == 2

    def test_search_excludes_consolidated_away_items(self):
        s, ids = _store_with("Alice lives in Paris", "Alice resides in Paris city",
                             "Bob likes tennis")
        _force_embeddings(s, ids, [[1, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0.05, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 0]])
        judge = FaithfulnessJudge(_merge_or_judge("LOSSLESS"), n_votes=1)
        s.consolidate(judge, retrieval_coverage_min=0.0, max_prune_checks=0)
        results = s.search("anything", k=10)
        ids_out = {m.id for m in results}
        assert ids[0] not in ids_out and ids[1] not in ids_out

    def test_rho_hat_counts_ops_touching_support(self):
        s, ids = _store_with("Alice lives in Paris", "Alice resides in Paris city",
                             "Bob likes tennis")
        _force_embeddings(s, ids, [[1, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0.05, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 0]])
        judge = FaithfulnessJudge(_merge_or_judge("LOSSLESS"), n_votes=1)
        s.consolidate(judge, retrieval_coverage_min=0.0, max_prune_checks=0)
        assert s.rho_hat == 0.0
        # feedback whose answer credits the merged item -> k_q >= 1
        s.feedback_answer("where does Alice live",
                          "merged: all facts combined")
        assert s.rho_hat >= 1.0

    def test_audit_jsonl_written(self, tmp_path):
        s, ids = _store_with("a b", "a b c")
        _force_embeddings(s, ids, [[1, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0.05, 0, 0, 0, 0, 0, 0]])
        judge = FaithfulnessJudge(_merge_or_judge("LOSSLESS"), n_votes=1)
        audit = tmp_path / "audit.jsonl"
        s.consolidate(judge, retrieval_coverage_min=0.0, max_prune_checks=0,
                      audit_path=str(audit))
        lines = audit.read_text().strip().splitlines()
        assert len(lines) == len(s.consolidation_ops) == 1

    def test_credit_transfer_zero_ablation(self):
        s, ids = _store_with("a b", "a b c")
        _force_embeddings(s, ids, [[1, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 0.05, 0, 0, 0, 0, 0, 0]])
        s._items[ids[0]].w.w_imp = 0.25
        judge = FaithfulnessJudge(_merge_or_judge("LOSSLESS"), n_votes=1)
        s.consolidate(judge, retrieval_coverage_min=0.0, max_prune_checks=0,
                      credit_transfer="zero", protect_w_imp=0.5)
        merged = [m for m in s.all_items()
                  if not m.pruned and m.metadata.get("merged_from")]
        assert merged and merged[0].w.w_imp == 0.0
