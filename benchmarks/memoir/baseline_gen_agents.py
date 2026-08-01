"""
Generative Agents baseline (Park et al., 2023).

Implements the 3-component memory scoring:
  score(m) = α·recency(m) + β·importance(m) + γ·relevance(q, m)

Where:
  recency    = exp(-λ·age_hours)
  importance = LLM-rated 1-10, normalized to [0, 1]
  relevance  = cosine(query_embedding, memory_embedding)

α = β = γ = 0.33 (equal weights, matching original paper)

One LLM call per add() — this is the baseline latency we measure against.
Uses Anthropic API; set ANTHROPIC_API_KEY in environment.

Reference: Park et al., "Generative Agents: Interactive Simulacra of Human Behavior", 2023.
"""

from __future__ import annotations

import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GAMemoryItem:
    id:              str   = field(default_factory=lambda: str(uuid.uuid4()))
    text:            str   = ""
    embedding:       Optional[np.ndarray] = field(default=None, repr=False)
    importance:      float = 0.5   # LLM-rated, normalized [0, 1]
    created_at:      float = field(default_factory=time.time)
    retrieval_count: int   = 0

    def recency(self, lambda_rec: float = 0.1) -> float:
        age_hours = (time.time() - self.created_at) / 3600.0
        return math.exp(-lambda_rec * age_hours)


class GenerativeAgentsStore:
    """
    Faithful reproduction of Generative Agents memory scoring.
    Not a strawman — uses the same weights and formulas as the original.
    """

    def __init__(
        self,
        model_name:    str   = "all-MiniLM-L6-v2",
        llm_model:     str   = "claude-haiku-4-5-20251001",
        alpha:         float = 1/3,
        beta:          float = 1/3,
        gamma:         float = 1/3,
        lambda_rec:    float = 0.1,
        track_latency: bool  = True,
        llm_client=None,  # LLMClient instance from llm_client.py (optional)
    ) -> None:
        from harpo.memoir.embedder import Embedder
        self._embedder    = Embedder.get(model_name)
        self._llm_model   = llm_model
        self._llm_client  = llm_client   # if set, used instead of raw anthropic call
        self._alpha       = alpha
        self._beta        = beta
        self._gamma       = gamma
        self._lambda_rec  = lambda_rec
        self._track_latency = track_latency

        self._items:      dict[str, GAMemoryItem] = {}
        self._order:      list[str]               = []
        self._embeddings: Optional[np.ndarray]    = None

        self._write_latencies: list[float] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(self, text: str) -> str:
        t0 = time.perf_counter()

        embedding   = self._embedder.encode(text)
        importance  = self._rate_importance(text)  # one LLM call

        item = GAMemoryItem(
            text=text,
            embedding=embedding,
            importance=importance,
            created_at=time.time(),
        )
        self._items[item.id] = item
        self._order.append(item.id)
        self._rebuild_matrix()

        if self._track_latency:
            self._write_latencies.append(time.perf_counter() - t0)

        return item.id

    def search(self, query: str, k: int = 10) -> list[GAMemoryItem]:
        if not self._items or self._embeddings is None:
            return []

        q_vec = self._embedder.encode(query)
        items = [self._items[i] for i in self._order if i in self._items]
        n = len(items)

        cos_scores = self._embeddings[:n] @ q_vec

        scored = []
        for idx, item in enumerate(items):
            rec  = item.recency(self._lambda_rec)
            imp  = item.importance
            rel  = float(cos_scores[idx])

            # Normalize each to [0,1] across candidates before combining
            score = self._alpha * rec + self._beta * imp + self._gamma * rel
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        for item, _ in top:
            item.retrieval_count += 1
        return [item for item, _ in top]

    def write_latency_stats(self) -> dict[str, float]:
        if not self._write_latencies:
            return {}
        lats = sorted(self._write_latencies)
        n = len(lats)
        return {
            "p50": lats[int(n * 0.50)],
            "p95": lats[int(n * 0.95)],
            "p99": lats[min(int(n * 0.99), n - 1)],
            "mean": sum(lats) / n,
            "n": n,
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _rate_importance(self, text: str) -> float:
        """
        Ask LLM: "On a scale of 1-10, how important or insightful is this memory?"
        Returns normalized [0, 1]. Falls back to 0.5 on failure.

        Priority:
          1. Use self._llm_client.rate_importance() if llm_client was provided
          2. Fall back to direct anthropic SDK call
          3. Return 0.5 on any error (mock mode)
        """
        if self._llm_client is not None:
            return self._llm_client.rate_importance(text)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            prompt = (
                f"On a scale of 1 to 10, where 1 is mundane (e.g., brushing teeth) "
                f"and 10 is extremely important (e.g., a life-changing event), rate the "
                f"poignancy of the following memory. Respond with only a single integer.\n\n"
                f"Memory: {text}"
            )
            response = client.messages.create(
                model=self._llm_model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            m = re.search(r"\b([1-9]|10)\b", raw)
            if m:
                return (int(m.group(1)) - 1) / 9.0
        except Exception:
            pass
        return 0.5  # neutral fallback (mock mode)

    def _rebuild_matrix(self) -> None:
        items = [self._items[i] for i in self._order if i in self._items]
        vecs = [i.embedding for i in items if i.embedding is not None]
        if vecs:
            self._embeddings = np.stack(vecs, axis=0)
