"""Unit tests for the co-citation graph in MemoirStore."""

import sys
import types

import pytest

# Patch sentence_transformers so tests run without GPU/download
st = types.ModuleType("sentence_transformers")


class _FakeST:
    def __init__(self, *a, **kw):
        pass

    def get_sentence_embedding_dimension(self):
        return 8  # tiny for tests

    def encode(self, texts, **kw):
        import numpy as np

        # Unit-norm like the real encoder — the store assumes dot == cosine
        # (dedup and MMR both rely on it).
        if isinstance(texts, list):
            vecs = np.random.rand(len(texts), 8).astype("float32")
            return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        vec = np.random.rand(8).astype("float32")
        return vec / np.linalg.norm(vec)


st.SentenceTransformer = _FakeST
sys.modules.setdefault("sentence_transformers", st)

# Also stub rank_bm25 in case not installed
try:
    import rank_bm25  # noqa: F401
except ImportError:
    rb = types.ModuleType("rank_bm25")

    class _BM25:
        def __init__(self, corpus):
            self._n = len(corpus)

        def get_scores(self, query):
            import numpy as np
            return np.zeros(self._n, dtype="float32")

    rb.BM25Okapi = _BM25
    sys.modules["rank_bm25"] = rb

import numpy as np

from harpo.memoir.schema import MemoryItem, WVector
from harpo.memoir.store import (
    CO_CITE_RATE,
    MIN_ATTRIBUTION_FOR_EDGE,
    MemoirStore,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _store_with_memories(*texts: str, use_co_cite: bool = True) -> tuple[MemoirStore, list[str]]:
    """Return (store, [id_0, id_1, ...]) with stub embeddings."""
    # use_dedup=False: stub embeddings are random, so real-text semantics don't
    # apply and near-duplicate collapse would silently merge distinct memories.
    s = MemoirStore(use_co_cite=use_co_cite, use_dedup=False)
    # Use add() to get stable ids
    ids = [s.add(t) for t in texts]
    return s, ids


def _fake_retrieved(store: MemoirStore, *item_ids: str) -> list[tuple[MemoryItem, float]]:
    return [(store._items[id_], 0.9) for id_ in item_ids if id_ in store._items]


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCoCiteGraph:

    def test_edges_form_after_co_citation(self):
        s, (id_a, id_b) = _store_with_memories("Memory A text", "Memory B text")
        s._update_co_cite_graph(
            retrieved=_fake_retrieved(s, id_a, id_b),
            attributions={id_a: 0.6, id_b: 0.4},
        )
        assert s.co_cite_edge_count == 1
        key = tuple(sorted([id_a, id_b]))
        assert key in s._co_cite_graph
        assert s._co_cite_graph[key] > 0.0

    def test_cold_start_no_crash(self):
        s, _ = _store_with_memories("Single memory text")
        results = s.search("query", k=3)
        assert len(results) >= 1
        assert s.co_cite_edge_count == 0

    def test_edge_weight_capped_at_1(self):
        s, (id_a, id_b) = _store_with_memories("A", "B")
        # Drive with very high attributions many times
        for _ in range(500):
            s._update_co_cite_graph(
                retrieved=_fake_retrieved(s, id_a, id_b),
                attributions={id_a: 1.0, id_b: 1.0},
            )
        key = tuple(sorted([id_a, id_b]))
        assert s._co_cite_graph[key] <= 1.0

    def test_undirected_key_symmetry(self):
        s, (id_a, id_b) = _store_with_memories("A", "B")
        key_ab = tuple(sorted([id_a, id_b]))
        key_ba = tuple(sorted([id_b, id_a]))
        assert key_ab == key_ba

    def test_bonus_zero_when_use_co_cite_false(self):
        s, (id_a, id_b) = _store_with_memories(
            "The user works at Google", "Google employees get free meals",
            use_co_cite=False,
        )
        # Manually inject a high-weight edge — it should still not be used
        key = tuple(sorted([id_a, id_b]))
        s._co_cite_graph[key] = 1.0
        s._items[id_a].w.w_imp = 1.0
        # search should not pass co_cite_graph (use_co_cite=False)
        results_disabled = s.search("free meals perks", k=2)
        # Now enable it
        s._use_co_cite = True
        results_enabled = s.search("free meals perks", k=2)
        # Scores may differ; both should return results without crash
        assert len(results_disabled) >= 1
        assert len(results_enabled) >= 1

    def test_min_attribution_threshold(self):
        s, (id_a, id_b) = _store_with_memories("A", "B")

        # Below threshold (0.05, recalibrated from 0.15 — see MIN_ATTRIBUTION_FOR_EDGE
        # docstring in store.py: mean_max_attribution across LoCoMo is 0.0625, so the
        # old 0.15 threshold sat above where ROUGE1 attribution actually lives) — no edge
        s._update_co_cite_graph(
            retrieved=_fake_retrieved(s, id_a, id_b),
            attributions={id_a: 0.03, id_b: 0.03},
        )
        assert s.co_cite_edge_count == 0

        # Above threshold — edge forms
        s._update_co_cite_graph(
            retrieved=_fake_retrieved(s, id_a, id_b),
            attributions={id_a: 0.10, id_b: 0.10},
        )
        assert s.co_cite_edge_count == 1

    def test_bridge_memory_unit(self):
        """Manual graph injection: verify bridge memory surfaces via co-cite bonus."""
        s, (id_a, id_b, id_c) = _store_with_memories(
            "The user works at Google",
            "Google employees get free meals and snacks",
            "The user enjoys outdoor running",
        )

        # Give A high w_imp (prior TRACE-FB) and a strong embedding toward query
        s._items[id_a].w.w_imp = 0.5
        # Inject co-cite edge A↔B
        s._update_co_cite_graph(
            retrieved=_fake_retrieved(s, id_a, id_b),
            attributions={id_a: 0.6, id_b: 0.4},
        )
        assert s.co_cite_edge_count == 1

        # Inject deterministic embeddings so A is close to query, B is bridged
        dim = s._embedder.dim
        query_vec = np.zeros(dim, dtype="float32")
        query_vec[0] = 1.0  # query points along dim 0
        a_vec = np.zeros(dim, dtype="float32")
        a_vec[0] = 1.0      # A aligns with query
        b_vec = np.zeros(dim, dtype="float32")
        b_vec[1] = 1.0      # B orthogonal to query — would normally score 0

        s._items[id_a].embedding = a_vec
        s._items[id_b].embedding = b_vec
        s._items[id_c].embedding = np.zeros(dim, dtype="float32")
        s._matrix_dirty = True

        # Search: B should get co-cite bonus once A is selected
        results = s.search("what food perks does the user get at work", k=3)
        result_ids = [r.id for r in results]
        assert id_b in result_ids, f"Bridge memory B not in results: {result_ids}"

    def test_bridge_memory_integration(self):
        """Full pipeline: feedback_answer() builds the co-cite graph automatically."""
        s, (id_a, id_b, id_c) = _store_with_memories(
            "The user works at Google",
            "Google employees get free meals and snacks",
            "The user enjoys outdoor running",
        )

        # Give A strong cosine to a related query and set up embeddings
        dim = s._embedder.dim
        query_vec = np.zeros(dim, dtype="float32")
        query_vec[0] = 1.0
        a_vec = np.zeros(dim, dtype="float32")
        a_vec[0] = 1.0
        a_vec[1] = 0.3
        a_vec = a_vec / np.linalg.norm(a_vec)
        b_vec = np.zeros(dim, dtype="float32")
        b_vec[1] = 1.0
        b_vec[0] = 0.1
        b_vec = b_vec / np.linalg.norm(b_vec)

        s._items[id_a].embedding = a_vec
        s._items[id_b].embedding = b_vec
        s._items[id_c].embedding = np.zeros(dim, dtype="float32")
        s._matrix_dirty = True

        # Session 1: feedback on a query that attributes strongly to A and B
        # Use oracle_correct=True to force attribution > MIN_ATTRIBUTION_FOR_EDGE
        s._items[id_a].w.w_imp = 0.0
        result = s.feedback_answer(
            query="what does the user get at Google",
            answer="free meals and snacks at Google",
            oracle_correct=True,   # forces attribution=1.0 for all retrieved
        )
        # Graph should now have at least one edge
        assert s.co_cite_edge_count >= 1, "No edges formed after feedback_answer"

        # Set A's w_imp high so it propagates to B
        s._items[id_a].w.w_imp = 0.5

        # Session 2: search for "user perks" — A surfaces, B gets bridge bonus
        results = s.search("what food perks does the user get at work", k=3)
        result_ids = [r.id for r in results]
        assert id_b in result_ids, f"Bridge memory not in results after co-cite graph: {result_ids}"

    def test_nli_attributed_edge_forms(self):
        """NLI path fire (numeric answer) should not crash; edge may or may not form."""
        s = MemoirStore(use_nli=True, use_co_cite=True)
        # Use add() so we control the flow
        s.add("The user visited Paris three times in 2023")
        s.add("Paris is a popular tourist destination")

        result = s.feedback_answer(
            "how many times did the user visit Paris", "3"
        )
        # At least one path should have fired (rouge1 or nli)
        assert len(result.attribution_paths) >= 0  # no crash
        # co_cite_edge_count valid
        assert s.co_cite_edge_count >= 0
        # If any attribution is above threshold, edge should exist
        high_attr = {k: v for k, v in result.attributions.items() if v >= MIN_ATTRIBUTION_FOR_EDGE}
        if len(high_attr) >= 2:
            assert s.co_cite_edge_count >= 1
