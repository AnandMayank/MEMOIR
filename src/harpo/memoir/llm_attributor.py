"""
LLM-assisted attribution for TRACE-FB.

Fires only when ROUGE1 attribution < ROUGE1_THRESHOLD (~40% of queries).
Additional w_imp gate: skip if w_imp >= 0.5 (already high-priority, call unnecessary).
Uses claude-haiku-4-5 for cost efficiency.

Training-free: LLM used at inference only, no gradient updates.
Never called at write time — add()/add_many() write-path latency unchanged.

Cost estimate (LoCoMo 10-sample eval, 2433 QA pairs):
  ~40% queries have ROUGE1 < 0.05  → ~973 candidate calls
  ~50% of those pass w_imp < 0.5 gate → ~487 actual LLM calls
  claude-haiku-4-5: ~$0.00025/1K input tokens, max_tokens=20
  Average prompt: ~120 tokens → ~$0.03 per 1K calls → ~$0.015 total
  Round estimate: ~$0.05 total for a full 10-sample LoCoMo eval run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

ATTRIBUTION_PROMPT = """\
A conversational AI agent retrieved the following memory to answer a question.

Memory: {memory}
Question: {query}
Answer given: {answer}

How much did this memory contribute to the answer?
Consider: does this memory contain information that supports, implies, \
or is necessary for the given answer?

Respond with ONLY valid JSON in this exact format:
{{"score": 0.0}}

Where score is 0.0 (no contribution) to 1.0 (directly answers it).
Examples:
- Memory describes 3 events, answer is "3" -> {{"score": 0.85}}
- Memory is about a different topic entirely -> {{"score": 0.02}}
- Memory partially supports but isn't sufficient -> {{"score": 0.4}}"""


class LLMAttributor:
    """
    Singleton LLM attribution scorer. Lazy-loaded on first score() call.
    Thread-safe double-checked locking, same pattern as embedder.py.
    """

    _instance: Optional["LLMAttributor"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._client = None
        self._model = "claude-haiku-4-5-20251001"

    @classmethod
    def get(cls) -> "LLMAttributor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        """Lazy init: load anthropic client on first call."""
        if self._client is not None:
            return
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError(
                "anthropic package required for LLM attribution. "
                "pip install anthropic"
            )

    def score(
        self,
        memory_text: str,
        query: str,
        answer: str,
    ) -> float:
        """
        Score how much memory contributed to answer.
        Returns float in [0.0, 1.0]. Returns 0.0 on any failure.
        """
        self._load()
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=20,
                messages=[{
                    "role": "user",
                    "content": ATTRIBUTION_PROMPT.format(
                        memory=memory_text[:500],
                        query=query[:200],
                        answer=answer[:100],
                    ),
                }],
            )
            text = response.content[0].text.strip()
            parsed = json.loads(text)
            s = float(parsed["score"])
            return max(0.0, min(1.0, s))
        except Exception as e:
            logger.warning("LLM attribution failed: %s, returning 0.0", e)
            return 0.0

    async def score_async(
        self,
        memory_text: str,
        query: str,
        answer: str,
    ) -> float:
        """Async wrapper — runs score() in a thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.score, memory_text, query, answer
        )

    async def score_batch_async(
        self,
        items: list[tuple[str, str, str]],  # [(memory_text, query, answer), ...]
    ) -> list[float]:
        """
        Score multiple (memory, query, answer) triples concurrently.
        asyncio.gather() fires all calls simultaneously — P50 latency
        ~= single-call latency regardless of batch size.
        """
        if not items:
            return []
        tasks = [self.score_async(mem, q, ans) for mem, q, ans in items]
        return list(await asyncio.gather(*tasks))
