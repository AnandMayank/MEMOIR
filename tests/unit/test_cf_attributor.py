"""
Unit tests for TRACE-CF (cf_attributor.py) — counterfactual LOO attribution.

Mechanics tests use sshleifer/tiny-gpt2 (~2MB, no chat template) so they run
fast and offline once cached. Behavioral tests that need a real language model
(answer-bearing memory gets high credit) use Qwen2.5-0.5B-Instruct and are
marked slow — they download ~1GB on first run.
"""

from __future__ import annotations

import pytest

from harpo.memoir.cf_attributor import CFAttributor
from harpo.memoir.store import MemoirStore
from harpo.memoir.trace_fb import apply_trace_feedback

TINY = "sshleifer/tiny-gpt2"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tiny_attr() -> CFAttributor:
    return CFAttributor.get(TINY)


# ── Mechanics (tiny model) ──────────────────────────────────────────────────────

def test_scores_bounded(tiny_attr):
    memories = ["Alice went to Paris in June.", "Bob likes cheese.", "The sky is blue."]
    scores = tiny_attr.score_loo("Where did Alice go?", "Paris", memories)
    assert len(scores) == len(memories)
    for s in scores:
        assert 0.0 <= s < 1.0


def test_empty_memories_returns_empty(tiny_attr):
    assert tiny_attr.score_loo("q?", "answer", []) == []


def test_empty_answer_returns_zeros(tiny_attr):
    scores = tiny_attr.score_loo("q?", "   ", ["some memory"])
    assert scores == [0.0]


def test_single_memory(tiny_attr):
    scores = tiny_attr.score_loo("Where did Alice go?", "Paris",
                                 ["Alice went to Paris."])
    assert len(scores) == 1
    assert 0.0 <= scores[0] < 1.0


def test_deterministic(tiny_attr):
    memories = ["Alice went to Paris.", "Bob likes cheese."]
    a = tiny_attr.score_loo("Where did Alice go?", "Paris", memories)
    b = tiny_attr.score_loo("Where did Alice go?", "Paris", memories)
    assert a == pytest.approx(b, abs=1e-6)


def test_singleton_per_model():
    assert CFAttributor.get(TINY) is CFAttributor.get(TINY)


# ── TRACE-FB wiring (tiny model) ───────────────────────────────────────────────

def test_trace_fb_cf_path_labels():
    store = MemoirStore(use_cf_attr=True, cf_model_name=TINY, use_co_cite=False)
    store.add_many(["Alice went to Paris in June.",
                    "Bob adopted a dog named Rex.",
                    "Carol started a pottery class."])
    result = store.feedback_answer("Where did Alice go?", "Paris", k=3)
    assert result.attribution_paths
    assert all(p == "cf" for p in result.attribution_paths.values())
    assert store.cf_attribution_count == len(result.attribution_paths)
    for attr in result.attributions.values():
        assert 0.0 <= attr < 1.0


def test_trace_fb_oracle_bypasses_cf():
    store = MemoirStore(use_cf_attr=True, cf_model_name=TINY, use_co_cite=False)
    store.add_many(["Alice went to Paris.", "Bob likes cheese."])
    result = store.feedback_answer("Where did Alice go?", "Paris", k=2,
                                   oracle_correct=True)
    assert all(p == "oracle" for p in result.attribution_paths.values())
    assert store.cf_attribution_count == 0


def test_trace_fb_cf_updates_w_imp_bounds():
    store = MemoirStore(use_cf_attr=True, cf_model_name=TINY, use_co_cite=False)
    store.add_many(["Alice went to Paris.", "Bob likes cheese."])
    store.feedback_answer("Where did Alice go?", "Paris", k=2)
    for item in store.all_items():
        assert 0.0 <= item.w.w_imp <= 1.0


# ── Behavioral (real model, slow) ──────────────────────────────────────────────

@pytest.mark.slow
def test_answer_bearing_memory_gets_highest_credit():
    attr = CFAttributor.get(QWEN)
    memories = [
        "Melanie mentioned she ran a charity race in the park last Sunday.",
        "Caroline said she adopted a cat named Muffin.",
        "Melanie's favorite color is green.",
    ]
    scores = attr.score_loo("What did Melanie do last Sunday?",
                            "She ran a charity race in the park.", memories)
    assert scores[0] == max(scores)
    assert scores[0] > 0.05


@pytest.mark.slow
def test_numeric_answer_not_blind():
    """The documented ROUGE1 blind spot: count answers whose token appears in
    no memory text. CF attribution must give nonzero credit to the memories
    enumerating the events."""
    attr = CFAttributor.get(QWEN)
    memories = [
        "John went hiking at Mount Rainier in May.",
        "John hiked the Enchantments trail in July.",
        "John did a hike along the coast in August.",
        "Sarah bought a new espresso machine.",
    ]
    scores = attr.score_loo("How many hikes did John go on this year?",
                            "3", memories)
    hike_credit = max(scores[0], scores[1], scores[2])
    assert hike_credit > 0.0
    assert hike_credit >= scores[3]
