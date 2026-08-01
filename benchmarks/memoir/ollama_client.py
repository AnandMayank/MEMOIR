"""
Local Ollama client for the LiCoMemory-matched SOTA eval — open-source
backbone alternative to gemini_client.py.

Uses llama3.1:8b (already pulled locally). NOT the same model LiCoMemory
tested (Llama-3.1-70B-Instruct-Turbo) — 70B needs ~35-40GB VRAM even
quantized; this machine has an 8GB GPU. This is the honest, free,
open-weight substitute: same model family, much smaller size. Label
results accordingly — lower fidelity than a true 70B comparison, not a
reproduction of LiCoMemory's exact Llama condition.

Same generate_answer()/judge_correct() interface as GeminiClient so
eval_locomo_sota.py can swap clients without changes.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import httpx

_ENDPOINT = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = "llama3.1:8b"
_TIMEOUT = 60.0


class OllamaClient:
    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model

    def _call(self, prompt: str, max_tokens: int = 200) -> tuple[str, float]:
        t0 = time.perf_counter()
        try:
            resp = httpx.post(
                _ENDPOINT,
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": max_tokens},
                },
                timeout=_TIMEOUT,
            )
            latency = time.perf_counter() - t0
            if resp.status_code != 200:
                return "", latency
            text = resp.json().get("response", "").strip()
            return text, latency
        except Exception:
            return "", time.perf_counter() - t0

    def generate_answer(self, question: str, context: str) -> tuple[str, float]:
        prompt = (
            f"Based on the context below, write a short phrase answering the question. "
            f"Use exact words from the context wherever possible.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nShort answer:"
        )
        return self._call(prompt, max_tokens=60)

    def judge_correct(self, question: str, generated: str, gold: str) -> tuple[bool, float]:
        prompt = (
            "You are grading whether a candidate answer is correct given the question "
            "and the reference (gold) answer. Minor phrasing differences are fine; "
            "the candidate must convey the same factual content as the reference.\n\n"
            f"Question: {question}\n"
            f"Reference answer: {gold}\n"
            f"Candidate answer: {generated}\n\n"
            "Respond with ONLY \"yes\" or \"no\"."
        )
        text, latency = self._call(prompt, max_tokens=5)
        is_correct = bool(re.search(r"\byes\b", text.lower()))
        return is_correct, latency
