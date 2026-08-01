"""
NLI-based attribution for TRACE-FB blind cases.

Fires only when ROUGE1-recall < threshold — i.e., for numeric/count answers,
temporal inferences, and preference syntheses where the answer token doesn't
appear verbatim in memory text.

Model: cross-encoder/nli-deberta-v3-small
  3-class NLI: contradiction (0), entailment (1), neutral (2)
  ~180MB on CPU; ~180MB GPU VRAM on CUDA.
  Cold load: ~3-5s. P50 batch inference: ~4ms/10 pairs CPU, <1ms GPU.

Design mirrors embedder.py singleton pattern. Model is NOT loaded at import
time — lazy init fires on first score() / score_batch() call only.
"""

from __future__ import annotations

import threading
from typing import Optional


_CACHE_MAX = 256


class NLIAttributor:
    _instance: Optional["NLIAttributor"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._entail_idx: int = 2  # default for DeBERTa-v3; confirmed in _load()
        # Simple FIFO cache keyed on (premise[:100], hypothesis[:50])
        self._cache: dict[tuple[str, str], float] = {}
        self._cache_order: list[tuple[str, str]] = []

    @classmethod
    def get(cls) -> "NLIAttributor":
        """Return singleton NLIAttributor. Thread-safe double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Lazy model load ────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load DeBERTa-v3-small on first use. Called inside _lock-free section
        since repeated loads are idempotent and construction is the bottleneck.
        Sets _unavailable=True if torch/transformers are not installed so callers
        can degrade gracefully (return 0.0) without crashing."""
        if self._model is not None or getattr(self, "_unavailable", False):
            return
        try:
            import torch
        except ImportError:
            self._unavailable = True  # type: ignore[attr-defined]
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = "cross-encoder/nli-deberta-v3-small"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

        # Resolve entailment class index from model config
        id2label = getattr(self._model.config, "id2label", {})
        found = False
        for idx, label in id2label.items():
            if label.lower() == "entailment":
                self._entail_idx = int(idx)
                found = True
                break
        if not found:
            # Model has no entailment label — disable NLI gracefully
            self._unavailable = True  # type: ignore[attr-defined]
            self._model = None

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_key(self, premise: str, hypothesis: str) -> tuple[str, str]:
        return (premise[:100], hypothesis[:50])

    def _cache_get(self, key: tuple[str, str]) -> Optional[float]:
        return self._cache.get(key)

    def _cache_put(self, key: tuple[str, str], value: float) -> None:
        if key not in self._cache:
            if len(self._cache_order) >= _CACHE_MAX:
                oldest = self._cache_order.pop(0)
                self._cache.pop(oldest, None)
            self._cache_order.append(key)
        self._cache[key] = value

    # ── Public scoring API ─────────────────────────────────────────────────────

    def score(self, premise: str, hypothesis: str) -> float:
        """Return P(entailment) ∈ [0, 1] for a single (premise, hypothesis) pair."""
        key = self._cache_key(premise, hypothesis)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        result = self.score_batch([(premise, hypothesis)])[0]
        return result

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Return P(entailment) for each (premise, hypothesis) pair.
        Cached pairs are resolved without model inference.
        Remaining pairs are batched into a single forward pass.
        """
        if not pairs:
            return []

        self._load()

        # Graceful degradation when torch/transformers not installed
        if getattr(self, "_unavailable", False):
            return [0.0] * len(pairs)

        import torch
        import torch.nn.functional as F

        results: list[Optional[float]] = [None] * len(pairs)
        uncached_indices: list[int] = []
        uncached_pairs: list[tuple[str, str]] = []

        for i, (premise, hypothesis) in enumerate(pairs):
            key = self._cache_key(premise, hypothesis)
            cached = self._cache_get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_pairs.append((premise, hypothesis))

        if uncached_pairs:
            premises = [p for p, _ in uncached_pairs]
            hypotheses = [h for _, h in uncached_pairs]

            encoded = self._tokenizer(
                premises,
                hypotheses,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self._model(**encoded).logits  # (batch, 3)

            probs = F.softmax(logits, dim=-1)           # (batch, 3)
            entail_probs = probs[:, self._entail_idx].cpu().tolist()

            for list_pos, (orig_idx, (premise, hypothesis)) in enumerate(
                zip(uncached_indices, uncached_pairs)
            ):
                score = float(entail_probs[list_pos])
                key = self._cache_key(premise, hypothesis)
                self._cache_put(key, score)
                results[orig_idx] = score

        return [r for r in results]  # all slots filled
