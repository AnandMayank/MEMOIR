"""
Thin Gemini API client for the LiCoMemory-matched SOTA eval.

Uses httpx to call Google's Generative Language API directly — no SDK.
Reads GOOGLE_API_KEY from environment (or .env, loaded manually to avoid
a new dependency). Same used for both generation and LLM-judge scoring,
mirroring how LiCoMemory used a single backbone (GPT-4o-mini) for both
roles in one of their two reported conditions.

This is an eval-harness-time cost only — does not touch MEMOIR's
write-path (0 LLM/write) claim.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import httpx

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# gemini-2.5-flash's free tier is 20 requests/DAY (RESOURCE_EXHAUSTED at that cap) —
# far too low for eval-scale runs. flash-lite has a separate, much larger daily quota.
_DEFAULT_MODEL = "gemini-2.5-flash-lite"
_TIMEOUT = 30.0


class QuotaExhaustedError(RuntimeError):
    """Raised after repeated 429s — signals the caller to stop rather than
    silently accumulate empty-string answers (which look like real 0% scores)."""


def _load_env_file(path: Path) -> None:
    """Minimal .env loader — avoids adding python-dotenv as a dependency."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and key not in os.environ:
            os.environ[key] = val


_load_env_file(Path(__file__).parent.parent.parent / ".env")


class GeminiClient:
    """Generation + LLM-judge client for the SOTA comparison eval."""

    def __init__(self, api_key: Optional[str] = None, model: str = _DEFAULT_MODEL,
                 max_consecutive_429: int = 3) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._model = model
        self._max_consecutive_429 = max_consecutive_429
        self._consecutive_429 = 0
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY not set (checked env and .env)")

    def _call(self, prompt: str, max_tokens: int = 200) -> tuple[str, float]:
        """Returns (response_text, latency_seconds).

        Raises QuotaExhaustedError after max_consecutive_429 429s in a row —
        stops the caller from silently accumulating empty-string answers that
        would otherwise look like genuine 0% accuracy.
        """
        t0 = time.perf_counter()
        try:
            resp = httpx.post(
                _ENDPOINT.format(model=self._model),
                params={"key": self._api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": 0.0,
                    },
                },
                timeout=_TIMEOUT,
            )
            latency = time.perf_counter() - t0
            if resp.status_code == 429:
                self._consecutive_429 += 1
                if self._consecutive_429 >= self._max_consecutive_429:
                    raise QuotaExhaustedError(
                        f"{self._consecutive_429} consecutive 429s from {self._model}: "
                        f"{resp.text[:300]}"
                    )
                return "", latency
            if resp.status_code != 200:
                return "", latency
            self._consecutive_429 = 0
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return "", latency
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            return text, latency
        except QuotaExhaustedError:
            raise
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
        """LLM-judge binary correctness, following LongMemEval's protocol
        (Wu et al. 2024): an LLM makes a binary assessment of answer correctness."""
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


def approx_token_count(text: str) -> int:
    """Cheap approximation (~1.3 tokens/word) — avoids a countTokens API call per item."""
    return int(len(text.split()) * 1.3)
