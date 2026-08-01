"""Unit tests for certified consolidation machinery (consolidate.py)."""

import numpy as np
import pytest

from harpo.memoir.schema import MemoryItem, WVector
from harpo.memoir.consolidate import (
    CalibrationResult,
    FaithfulnessJudge,
    calibrate_certify_threshold,
    generate_merge_text,
    hoeffding_ucb,
    propose_clusters,
    retrieval_token_coverage,
    transfer_credit,
)


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def _item(text, vec, w_imp=0.0, dia_id=None):
    meta = {"dia_id": dia_id} if dia_id else {}
    return MemoryItem(text=text, embedding=_unit(vec),
                      w=WVector(w_imp=w_imp), metadata=meta)


# ── Verdict parsing (certificate-critical) ─────────────────────────────────────

class TestParseVerdict:
    def test_plain_lossless(self):
        assert FaithfulnessJudge.parse_verdict("LOSSLESS") is True

    def test_lossless_lowercase(self):
        assert FaithfulnessJudge.parse_verdict("lossless\nbecause...") is True

    def test_lossy(self):
        assert FaithfulnessJudge.parse_verdict("LOSSY") is False

    def test_not_lossless_rejected(self):
        # The certificate-breaking bug: substring test certified this.
        assert FaithfulnessJudge.parse_verdict("NOT LOSSLESS") is False

    def test_lossless_no_rejected(self):
        assert FaithfulnessJudge.parse_verdict("LOSSLESS? NO") is False

    def test_empty_rejected(self):
        assert FaithfulnessJudge.parse_verdict("") is False
        assert FaithfulnessJudge.parse_verdict("   \n  ") is False

    def test_garbage_rejected(self):
        assert FaithfulnessJudge.parse_verdict("42!!") is False

    def test_verdict_from_first_line_only(self):
        assert FaithfulnessJudge.parse_verdict("LOSSY\nLOSSLESS") is False


class TestJudgeEnsemble:
    def test_unanimous_certifies(self):
        judge = FaithfulnessJudge(lambda p: "LOSSLESS", n_votes=3)
        assert judge.is_lossless("merged", ["a", "b"]) is True
        assert judge.call_count == 3

    def test_single_dissent_blocks_at_default_threshold(self):
        responses = iter(["LOSSLESS", "LOSSY", "LOSSLESS"])
        judge = FaithfulnessJudge(lambda p: next(responses), n_votes=3)
        assert judge.is_lossless("merged", ["a"]) is False

    def test_exception_counts_as_lossy_vote(self):
        def boom(p):
            raise RuntimeError("api down")
        judge = FaithfulnessJudge(boom, n_votes=3)
        assert judge.score("merged", ["a"]) == 0.0

    def test_calibrated_threshold_certifies_majority(self):
        responses = iter(["LOSSLESS", "LOSSY", "LOSSLESS"])
        judge = FaithfulnessJudge(lambda p: next(responses), n_votes=3,
                                  certify_threshold=0.6)
        assert judge.is_lossless("merged", ["a"]) is True


# ── Calibration ────────────────────────────────────────────────────────────────

class TestCalibration:
    def test_hoeffding_ucb_bounds(self):
        assert hoeffding_ucb(0, 0) == 1.0
        assert 0.0 < hoeffding_ucb(0, 100) < 0.2
        assert hoeffding_ucb(100, 100) == 1.0

    def test_clean_scores_calibrate(self):
        # 200 unanimous ops, none lossy -> tau=1.0 certifiable at alpha=0.2
        scores = [1.0] * 200
        labels = [False] * 200
        res = calibrate_certify_threshold(scores, labels, alpha=0.2)
        assert isinstance(res, CalibrationResult)
        assert res.tau == 1.0
        assert res.lossy_ucb_at_tau <= 0.2

    def test_permissive_tau_when_low_scores_clean(self):
        scores = [1.0] * 100 + [2 / 3] * 100
        labels = [False] * 200
        res = calibrate_certify_threshold(scores, labels, alpha=0.2)
        assert res is not None and res.tau < 1.0

    def test_uncertifiable_returns_none(self):
        scores = [1.0] * 10
        labels = [True] * 5 + [False] * 5  # 50% lossy even at unanimity
        assert calibrate_certify_threshold(scores, labels, alpha=0.05) is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            calibrate_certify_threshold([1.0], [], alpha=0.1)


# ── Retrieval-safety coverage ──────────────────────────────────────────────────

class TestRetrievalCoverage:
    def test_full_coverage(self):
        assert retrieval_token_coverage("Alice resides in Paris",
                                        ["Alice resides in Paris"]) == 1.0

    def test_paraphrase_drops_coverage(self):
        cov = retrieval_token_coverage("Alice lives in Paris",
                                       ["Alice resides in Paris"])
        assert cov < 1.0

    def test_stopwords_ignored(self):
        assert retrieval_token_coverage("Alice Paris",
                                        ["the Alice of Paris"]) == 1.0


# ── Cluster proposal ───────────────────────────────────────────────────────────

class TestProposeClusters:
    def test_similar_items_cluster(self):
        a = _item("cat A", [1, 0, 0, 0.1])
        b = _item("cat B", [1, 0.05, 0, 0.1])
        c = _item("dog", [0, 1, 0, 0])
        clusters = propose_clusters([a, b, c], sim_threshold=0.9)
        assert len(clusters) == 1
        assert {m.text for m in clusters[0]} == {"cat A", "cat B"}

    def test_co_cite_edge_lowers_threshold(self):
        a = _item("x", [1, 0.5, 0, 0])
        b = _item("y", [0.5, 1, 0, 0])  # cos(a,b) = 0.8
        graph = {tuple(sorted([a.id, b.id])): 0.2}
        # 0.8 < sim_threshold=0.9 -> no cluster without the edge
        assert propose_clusters([a, b], sim_threshold=0.9) == []
        # with edge and floor 0.7 the pair clusters
        clusters = propose_clusters([a, b], sim_threshold=0.9,
                                    co_cite_graph=graph, co_cite_sim_floor=0.7)
        assert len(clusters) == 1

    def test_co_cited_but_dissimilar_not_clustered(self):
        # multi-hop supports: co-cited yet nearly orthogonal — must stay apart
        a = _item("hop 1", [1, 0, 0, 0])
        b = _item("hop 2", [0, 1, 0, 0])
        graph = {tuple(sorted([a.id, b.id])): 0.5}
        assert propose_clusters([a, b], sim_threshold=0.9,
                                co_cite_graph=graph, co_cite_sim_floor=0.45) == []

    def test_credit_protection_excludes_high_w_imp(self):
        a = _item("protected", [1, 0, 0, 0.1], w_imp=0.8)
        b = _item("twin", [1, 0.01, 0, 0.1])
        assert propose_clusters([a, b], sim_threshold=0.9,
                                protect_w_imp=0.3) == []

    def test_pruned_and_hallucinated_excluded(self):
        a = _item("a", [1, 0, 0, 0])
        b = _item("b", [1, 0.01, 0, 0])
        b.pruned = True
        assert propose_clusters([a, b], sim_threshold=0.9) == []


# ── Credit transfer ────────────────────────────────────────────────────────────

class TestTransferCredit:
    def test_max_transfer(self):
        merged = _item("merged", [1, 0, 0, 0])
        c1 = _item("c1", [1, 0, 0, 0], w_imp=0.2, dia_id="D1:3")
        c2 = _item("c2", [1, 0, 0, 0], w_imp=0.6, dia_id="D2:7")
        transfer_credit(merged, [c1, c2], co_cite_graph=None, mode="max")
        assert merged.w.w_imp == 0.6
        assert merged.metadata["dia_ids"] == ["D1:3", "D2:7"]
        assert merged.metadata["dia_id"] == "D1:3"
        assert merged.metadata["merged_from"] == [c1.id, c2.id]

    def test_zero_mode_ablation(self):
        merged = _item("merged", [1, 0, 0, 0])
        c1 = _item("c1", [1, 0, 0, 0], w_imp=0.9)
        transfer_credit(merged, [c1], co_cite_graph=None, mode="zero")
        assert merged.w.w_imp == 0.0

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            transfer_credit(_item("m", [1, 0, 0, 0]), [], None, mode="sum")

    def test_edges_rekeyed_to_merged(self):
        merged = _item("merged", [1, 0, 0, 0])
        c1 = _item("c1", [1, 0, 0, 0])
        c2 = _item("c2", [1, 0, 0, 0])
        outsider = _item("out", [0, 1, 0, 0])
        graph = {
            tuple(sorted([c1.id, outsider.id])): 0.3,
            tuple(sorted([c2.id, outsider.id])): 0.5,
            tuple(sorted([c1.id, c2.id])): 0.9,  # internal — absorbed
        }
        transfer_credit(merged, [c1, c2], co_cite_graph=graph, mode="max")
        assert graph == {tuple(sorted([merged.id, outsider.id])): 0.5}

    def test_nested_merge_inherits_dia_ids(self):
        # a previously-merged constituent carries dia_ids (plural)
        prior = _item("prior merge", [1, 0, 0, 0])
        prior.metadata = {"dia_id": "D1:1", "dia_ids": ["D1:1", "D1:2"]}
        fresh = _item("fresh", [1, 0, 0, 0], dia_id="D3:4")
        merged = _item("merged", [1, 0, 0, 0])
        transfer_credit(merged, [prior, fresh], co_cite_graph=None, mode="max")
        assert merged.metadata["dia_ids"] == ["D1:1", "D1:2", "D3:4"]


# ── Merge text generation ──────────────────────────────────────────────────────

def test_generate_merge_text_formats_entries():
    seen = {}
    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "  merged text  "
    out = generate_merge_text(fake_llm, [_item("alpha", [1, 0, 0, 0]),
                                         _item("beta", [0, 1, 0, 0])])
    assert out == "merged text"
    assert "Entry 1: alpha" in seen["prompt"]
    assert "Entry 2: beta" in seen["prompt"]
