"""
Cosine-only and Recency+Cosine baselines.

CosinStore:         embed + pure cosine, no quality weighting, no BM25
RecencyCosineStore: cosine + recency decay (MemoryBank analog)

Both implement the same interface as GenerativeAgentsStore for the eval harness.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BaseMemoryItem:
    id:              str   = field(default_factory=lambda: str(uuid.uuid4()))
    text:            str   = ""
    embedding:       Optional[np.ndarray] = field(default=None, repr=False)
    created_at:      float = field(default_factory=time.time)
    retrieval_count: int   = 0


class CosineStore:
    """Cosine-only baseline: embed + top-K by cosine similarity. No BM25, no weighting."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from harpo.memoir.embedder import Embedder
        self._embedder    = Embedder.get(model_name)
        self._items:      dict[str, BaseMemoryItem] = {}
        self._order:      list[str]                 = []
        self._embeddings: Optional[np.ndarray]      = None

    def add(self, text: str, metadata: Optional[dict] = None) -> str:
        emb  = self._embedder.encode(text)
        item = BaseMemoryItem(text=text, embedding=emb)
        self._items[item.id] = item
        self._order.append(item.id)
        self._rebuild()
        return item.id

    def search(self, query: str, k: int = 10) -> list[BaseMemoryItem]:
        if not self._items or self._embeddings is None:
            return []
        q_vec = self._embedder.encode(query)
        items = [self._items[i] for i in self._order]
        cos   = self._embeddings[:len(items)] @ q_vec
        top   = np.argpartition(cos, -min(k, len(items)))[-min(k, len(items)):]
        top   = top[np.argsort(cos[top])[::-1]]
        result = [items[i] for i in top]
        for item in result:
            item.retrieval_count += 1
        return result

    def _rebuild(self) -> None:
        items = [self._items[i] for i in self._order]
        vecs  = [i.embedding for i in items if i.embedding is not None]
        if vecs:
            self._embeddings = np.stack(vecs, axis=0)


class RecencyCosineStore(CosineStore):
    """
    Recency + Cosine: score = 0.5 * recency + 0.5 * cosine.
    MemoryBank (2023) analog.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", lambda_rec: float = 0.1) -> None:
        super().__init__(model_name)
        self._lambda_rec = lambda_rec

    def search(self, query: str, k: int = 10) -> list[BaseMemoryItem]:
        if not self._items or self._embeddings is None:
            return []
        q_vec = self._embedder.encode(query)
        items = [self._items[i] for i in self._order]
        cos   = self._embeddings[:len(items)] @ q_vec
        now   = time.time()

        scored = []
        for idx, item in enumerate(items):
            age_h = (now - item.created_at) / 3600.0
            rec   = math.exp(-self._lambda_rec * age_h)
            score = 0.5 * rec + 0.5 * float(cos[idx])
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        for item, _ in top:
            item.retrieval_count += 1
        return [item for item, _ in top]
