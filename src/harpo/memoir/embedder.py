"""
Singleton embedding wrapper for MEMOIR.

Loads all-MiniLM-L6-v2 once and reuses it for all encode calls.
Avoids reloading the ~90MB model on every store instantiation.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class Embedder:
    _instances: dict[str, "Embedder"] = {}
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @classmethod
    def get(cls, model_name: str = "all-MiniLM-L6-v2") -> "Embedder":
        """Return a singleton Embedder per model name/path.
        Supports both HuggingFace model IDs and local paths (e.g. 'models/memoir-encoder-v1').
        """
        if model_name not in cls._instances:
            with cls._lock:
                if model_name not in cls._instances:
                    cls._instances[model_name] = cls(model_name)
        return cls._instances[model_name]

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text. Returns shape (dim,) float32 array."""
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts. Returns shape (n, dim) float32 array."""
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)
