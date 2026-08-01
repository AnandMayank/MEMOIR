"""Unit tests for TRACE-FB attribution and w_imp update."""

import pytest

from harpo.memoir.schema import MemoryItem, WVector
from harpo.memoir.trace_fb import (
    TraceFBResult,
    apply_trace_feedback,
    hallucinated_ids_from_trajectory,
    position_weight,
    rouge1_recall,
)


class TestRouge1Recall:
    def test_exact_match(self):
        score = rouge1_recall("Python is fast", "Python is fast")
        assert abs(score - 1.0) < 1e-6

    def test_partial_overlap(self):
        score = rouge1_recall("Python fast language", "Python is great")
        # Only "python" overlaps (after stop word removal: "fast", "language", "python" vs "python", "great")
        assert 0 < score < 1.0

    def test_no_overlap(self):
        score = rouge1_recall("Java enterprise software", "Python data science")
        assert score == 0.0

    def test_empty_memory(self):
        score = rouge1_recall("", "some answer")
        assert score == 0.0

    def test_empty_answer(self):
        score = rouge1_recall("Python fast", "")
        assert score == 0.0

    def test_stop_words_excluded(self):
        # Only stop words in memory → no scoreable tokens
        score = rouge1_recall("the is and a", "the answer is here")
        assert score == 0.0

    def test_answer_superset(self):
        # Answer is much longer but contains all memory tokens
        mem = "Python fast"
        ans = "Python is a fast and efficient programming language that scales well"
        score = rouge1_recall(mem, ans)
        assert score == 1.0


class TestPositionWeight:
    def test_rank0_is_highest(self):
        assert position_weight(0) == 1.0

    def test_decreasing(self):
        for r in range(5):
            assert position_weight(r) >= position_weight(r + 1)


class TestApplyTraceFeedback:
    def _make_item(self, text: str, w_imp: float = 0.0) -> MemoryItem:
        item = MemoryItem(text=text)
        item.w.w_imp = w_imp
        return item

    def test_w_imp_increases_on_overlap(self):
        item = self._make_item("Python is a fast language")
        retrieved = [(item, 0.9)]
        result = apply_trace_feedback(retrieved, "Python language", store=None, alpha=0.1)
        assert item.w.w_imp > 0.0
        assert item.id in result.updated

    def test_no_update_on_zero_overlap(self):
        # use_nli=False: isolates ROUGE1 path; NLI would give tiny non-zero entailment
        item = self._make_item("Java enterprise framework")
        retrieved = [(item, 0.5)]
        result = apply_trace_feedback(retrieved, "Python science", store=None, alpha=0.1, use_nli=False)
        assert item.w.w_imp == 0.0
        assert item.id not in result.updated

    def test_hallucination_hard_reset(self):
        item = self._make_item("Python is fast", w_imp=0.8)
        retrieved = [(item, 0.9)]
        result = apply_trace_feedback(
            retrieved,
            "Python fast language",
            store=None,
            alpha=0.1,
            hallucinated_ids={item.id},
        )
        assert item.w.w_imp == 0.0
        assert item.hallucinated is True
        assert item.id in result.reset

    def test_oracle_correct_gives_full_attribution(self):
        item = self._make_item("Java slowly")
        retrieved = [(item, 0.9)]
        result = apply_trace_feedback(
            retrieved, "completely unrelated answer",
            store=None, alpha=0.1, oracle_correct=True
        )
        # Should update despite zero ROUGE because oracle_correct=True
        assert item.w.w_imp > 0.0

    def test_oracle_incorrect_gives_zero_attribution(self):
        item = self._make_item("Python fast language")
        retrieved = [(item, 0.9)]
        result = apply_trace_feedback(
            retrieved, "Python fast language",  # ROUGE would be high
            store=None, alpha=0.1, oracle_correct=False
        )
        assert item.w.w_imp == 0.0

    def test_position_weight_applied(self):
        item0 = self._make_item("Python fast language")
        item1 = self._make_item("Python fast language")
        retrieved = [(item0, 0.9), (item1, 0.8)]
        apply_trace_feedback(retrieved, "Python fast language", store=None, alpha=0.1)
        # rank 0 gets weight 1.0, rank 1 gets weight 0.5 → item0 should have higher w_imp
        assert item0.w.w_imp > item1.w.w_imp

    def test_w_imp_capped_at_1(self):
        item = self._make_item("Python fast language", w_imp=0.99)
        retrieved = [(item, 0.9)]
        apply_trace_feedback(retrieved, "Python fast language", store=None, alpha=1.0)
        assert item.w.w_imp <= 1.0

    def test_empty_retrieved(self):
        result = apply_trace_feedback([], "any answer", store=None)
        assert result.updated == {}
        assert result.reset == []


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skip(reason="NLI path removed from feedback loop after ablation showed regression; nli_attributor.py kept but unused")
class TestNLIAttribution:
    """Tests for the NLI entailment path (ROUGE1 < ROUGE1_THRESHOLD fallback).
    Skipped if torch is not installed."""

    def test_numeric_answer_fires_nli_path(self):
        # "3" has no token overlap with the memory text → ROUGE1=0 → NLI fires
        item = MemoryItem(text="The user visited Paris three times in 2023")
        retrieved = [(item, 0.9)]
        result = apply_trace_feedback(retrieved, "3", store=None, alpha=0.1, use_nli=True)
        assert result.attribution_paths.get(item.id) == "nli"
        assert item.w.w_imp > 0.0  # DeBERTa should give entailment > 0

    def test_verbatim_answer_fires_rouge_path(self):
        item = MemoryItem(text="The user prefers Python for data pipelines")
        retrieved = [(item, 0.9)]
        result = apply_trace_feedback(retrieved, "Python", store=None, alpha=0.1, use_nli=True)
        # "python" token overlaps → ROUGE1 >= threshold → rouge1 path
        assert result.attribution_paths.get(item.id) == "rouge1"

    def test_use_nli_false_skips_nli(self):
        item = MemoryItem(text="The user visited Paris three times in 2023")
        retrieved = [(item, 0.9)]
        result = apply_trace_feedback(retrieved, "3", store=None, alpha=0.1, use_nli=False)
        assert "nli" not in result.attribution_paths.values()
        assert item.w.w_imp == 0.0  # ROUGE1=0, NLI skipped → no update

    def test_batch_scoring_multiple_memories(self):
        items = [
            MemoryItem(text="User bought coffee twice last week"),
            MemoryItem(text="User visited gym three times"),
        ]
        retrieved = [(i, 0.9) for i in items]
        result = apply_trace_feedback(retrieved, "2", store=None, alpha=0.1, use_nli=True)
        # "2" has no token overlap with either memory → both NLI-scored
        nli_count = sum(1 for v in result.attribution_paths.values() if v == "nli")
        assert nli_count == 2

    def test_attribution_in_zero_one(self):
        item = MemoryItem(text="Something happened five times")
        retrieved = [(item, 0.9)]
        result = apply_trace_feedback(retrieved, "5", store=None, alpha=0.1, use_nli=True)
        for v in result.attributions.values():
            assert 0.0 <= v <= 1.0

    def test_attribution_paths_field_exists(self):
        result = apply_trace_feedback([], "any answer", store=None)
        assert hasattr(result, "attribution_paths")
        assert isinstance(result.attribution_paths, dict)


class TestHallucinatedIdsFromTrajectory:
    def test_returns_empty_on_none(self):
        assert hallucinated_ids_from_trajectory(None) == set()
