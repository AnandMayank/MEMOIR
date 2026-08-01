"""Unit tests for the verification gate (anti self-reinforcing-error).

Covers: abstention detection, the lp-EMA soft gate math, gate wiring in
apply_trace_feedback (abstention path needs no CF model; the CF-confidence
path uses a monkeypatched CFAttributor stub).
"""

import math

import pytest

from harpo.memoir.schema import MemoryItem, WVector
from harpo.memoir.trace_fb import (
    apply_trace_feedback,
    is_abstention,
    verification_gate,
)


def _item(text: str, w_imp: float = 0.0) -> MemoryItem:
    m = MemoryItem(text=text)
    m.w = WVector(w_imp=w_imp)
    return m


class TestIsAbstention:
    @pytest.mark.parametrize("ans", [
        "",
        "   ",
        "I don't know.",
        "I do not know the answer to that.",
        "I cannot recall any details about this.",
        "The date is not mentioned in the conversation.",
        "There is no information about her sister.",
        "Unable to determine from the given memories.",
        "Unknown",
        "That question is unanswerable from the context.",
        "I'm not sure about this.",
    ])
    def test_abstentions(self, ans):
        assert is_abstention(ans)

    @pytest.mark.parametrize("ans", [
        "Paris",
        "She adopted a dog named Biscuit in May 2023.",
        "3",
        "He knows a lot about painting.",   # 'know' inside a real answer
        "They cancelled the trip.",         # 'can' inside a word must not match
    ])
    def test_real_answers(self, ans):
        assert not is_abstention(ans)


class TestVerificationGateMath:
    def test_noop_without_signal(self):
        assert verification_gate(None, {}) == 1.0
        assert verification_gate(-1.0, None) == 1.0

    def test_first_observation_seeds_baseline(self):
        v = {}
        assert verification_gate(-2.0, v) == 1.0
        assert v["lp"] == -2.0

    def test_confident_answer_full_credit(self):
        v = {"lp": -2.0}
        # more likely than baseline → clipped to 1.0
        assert verification_gate(-1.0, v) == 1.0

    def test_implausible_answer_scaled_down(self):
        v = {"lp": -1.0}
        gate = verification_gate(-3.0, v)
        assert abs(gate - math.exp(-2.0)) < 1e-9
        # baseline EMA moved toward the observation
        assert -3.0 < v["lp"] < -1.0

    def test_gate_monotone_in_lp(self):
        gates = []
        for lp in (-1.0, -2.0, -4.0):
            gates.append(verification_gate(lp, {"lp": -1.0}))
        assert gates[0] >= gates[1] >= gates[2]


class TestGateInFeedback:
    def test_abstention_blocks_credit(self):
        item = _item("Python is a fast language")
        retrieved = [(item, 0.9)]
        res = apply_trace_feedback(
            retrieved, "I don't know about Python", store=None, alpha=0.3,
            use_verification_gate=True,
        )
        assert res.verification_gate == 0.0
        assert item.w.w_imp == 0.0

    def test_same_answer_credits_without_gate(self):
        # sanity: the exact same call WITHOUT the gate does assign credit
        # (the abstention text lexically overlaps the memory)
        item = _item("Python is a fast language")
        res = apply_trace_feedback(
            [(item, 0.9)], "I don't know about Python", store=None, alpha=0.3,
        )
        assert res.verification_gate == 1.0
        assert item.w.w_imp > 0.0

    def test_abstention_gate_spares_oracle(self):
        item = _item("Python is a fast language")
        res = apply_trace_feedback(
            [(item, 0.9)], "I don't know", store=None, alpha=0.3,
            oracle_correct=True, use_verification_gate=True,
        )
        assert item.w.w_imp > 0.0

    def test_abstention_gate_spares_hallucination_reset(self):
        item = _item("Python is a fast language", w_imp=0.7)
        res = apply_trace_feedback(
            [(item, 0.9)], "I don't know", store=None, alpha=0.3,
            hallucinated_ids={item.id}, use_verification_gate=True,
        )
        assert item.id in res.reset
        assert item.w.w_imp == 0.0

    def test_abstention_gate_leaves_alpha_neg_decay_on(self):
        # gate blocks positive credit; retrieved-but-uncited decay still runs
        cited = _item("Python is a fast language", w_imp=0.5)
        uncited = _item("The weather in Bergen is rainy", w_imp=0.5)
        apply_trace_feedback(
            [(cited, 0.9), (uncited, 0.8)], "I don't know about Python",
            store=None, alpha=0.3, alpha_neg=0.06,
            use_verification_gate=True,
        )
        assert cited.w.w_imp == 0.5        # positive credit gated out
        assert uncited.w.w_imp < 0.5       # decay untouched by the gate

    def test_cf_confidence_scales_credit(self, monkeypatch):
        from harpo.memoir import trace_fb as tf

        class _StubCF:
            def __init__(self, lp):
                self._lp = lp

            def score_loo_with_confidence(self, question, answer, memories):
                return [0.8] * len(memories), self._lp

        class _StubGetter:
            lp = -1.0

            @classmethod
            def get(cls, model_name):
                return _StubCF(cls.lp)

        import harpo.memoir.cf_attributor as cfa
        monkeypatch.setattr(cfa, "CFAttributor", _StubGetter)

        # First call seeds the baseline at lp=-1.0 → gate 1.0
        v_running: dict = {}
        item1 = _item("m1")
        apply_trace_feedback(
            [(item1, 0.9)], "Paris", store=None, alpha=0.3, query="q",
            use_cf_attr=True, use_verification_gate=True, v_running=v_running,
        )
        full_delta = item1.w.w_imp
        assert full_delta > 0.0

        # Second call: much less plausible answer → proportionally less credit
        _StubGetter.lp = -3.0
        item2 = _item("m2")
        res2 = apply_trace_feedback(
            [(item2, 0.9)], "Paris", store=None, alpha=0.3, query="q",
            use_cf_attr=True, use_verification_gate=True, v_running=v_running,
        )
        assert res2.verification_gate < 1.0
        assert 0.0 < item2.w.w_imp < full_delta


class TestSparseEvidenceFlag:
    def test_single_citation_on_multihop_answer_flags(self):
        # Two memories credited in one event (multi-evidence answer): each is
        # on its first credible citation → both flagged sparse.
        a = _item("Alice moved to Lyon for the new job")
        b = _item("The new job starts in June")
        apply_trace_feedback(
            [(a, 0.9), (b, 0.8)],
            "Alice moved to Lyon because the new job starts in June",
            store=None, alpha=0.3,
        )
        assert a.commit_count == 1 and a.sparse_evidence
        assert b.commit_count == 1 and b.sparse_evidence

    def test_single_evidence_answer_does_not_flag(self):
        a = _item("Alice moved to Lyon for the new job")
        b = _item("The weather in Bergen is rainy")   # uncredited
        apply_trace_feedback(
            [(a, 0.9), (b, 0.8)],
            "Alice moved to Lyon for the job",
            store=None, alpha=0.3,
        )
        assert a.commit_count == 1 and not a.sparse_evidence

    def test_second_citation_clears_flag(self):
        a = _item("Alice moved to Lyon for the new job")
        b = _item("The new job starts in June")
        ans = "Alice moved to Lyon because the new job starts in June"
        apply_trace_feedback([(a, 0.9), (b, 0.8)], ans, store=None, alpha=0.3)
        assert a.sparse_evidence
        apply_trace_feedback([(a, 0.9), (b, 0.8)], ans, store=None, alpha=0.3)
        assert a.commit_count == 2 and not a.sparse_evidence
        assert b.commit_count == 2 and not b.sparse_evidence
