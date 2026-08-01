"""
Minimal BM25Okapi implementation — no external dependency.

Drop-in replacement for rank_bm25.BM25Okapi for the MEMOIR use case.
Supports incremental document addition via add_document().

Parameters follow rank_bm25 defaults: k1=1.5, b=0.75, epsilon=0.25.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


class BM25Okapi:
    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.5,
        b:  float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.k1      = k1
        self.b       = b
        self.epsilon = epsilon

        self._corpus:    list[list[str]]    = []
        self._df:        dict[str, int]     = {}  # term → doc frequency
        self._idf:       dict[str, float]   = {}
        self._tf:        list[dict[str, float]] = []
        self._doc_lens:  list[int]          = []
        self._avgdl:     float              = 0.0
        self._N:         int                = 0

        for doc in corpus:
            self._index_doc(doc)
        self._recompute_idf()

    # ── Public interface (matches rank_bm25.BM25Okapi) ────────────────────────

    def get_scores(self, query: list[str]) -> list[float]:
        """Return BM25 score for query against each document."""
        scores = [0.0] * self._N
        for term in query:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for doc_idx, tf_map in enumerate(self._tf):
                tf = tf_map.get(term, 0.0)
                if tf == 0.0:
                    continue
                dl     = self._doc_lens[doc_idx]
                norm   = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1e-9)))
                scores[doc_idx] += idf * norm
        return scores

    def add_document(self, doc: list[str]) -> None:
        """Add one document incrementally and recompute IDF."""
        self._index_doc(doc)
        self._recompute_idf()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _index_doc(self, doc: list[str]) -> None:
        counts = Counter(doc)
        total  = sum(counts.values())

        self._corpus.append(doc)
        self._doc_lens.append(total)
        self._N += 1

        tf_map: dict[str, float] = {}
        for term, freq in counts.items():
            tf_map[term] = freq
            self._df[term] = self._df.get(term, 0) + 1
        self._tf.append(tf_map)

        # Recompute avgdl
        self._avgdl = sum(self._doc_lens) / self._N

    def _recompute_idf(self) -> None:
        """Robertson-Sparck Jones IDF with Okapi smoothing."""
        idf_sum = 0.0
        idf_map: dict[str, float] = {}
        for term, df in self._df.items():
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1)
            idf_map[term] = idf
            idf_sum += idf

        # Epsilon floor to suppress very common terms
        avg_idf = idf_sum / max(len(idf_map), 1)
        floor   = self.epsilon * avg_idf
        for term in idf_map:
            if idf_map[term] < 0:
                idf_map[term] = floor
        self._idf = idf_map
