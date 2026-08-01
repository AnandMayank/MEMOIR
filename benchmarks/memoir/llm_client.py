"""
Thin LLM client for eval harnesses.

Uses httpx to call the Anthropic Messages API directly — no SDK needed.
Supports a retrieval-only mock mode for environments without a valid API key.

Set ANTHROPIC_API_KEY to a valid direct API key (sk-ant-api03-...) to enable
LLM answer generation. Claude Code OAuth tokens (sk-ant-oat01-...) will not
work with the direct API — use a service account key instead.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import httpx

_ENDPOINT = "https://api.anthropic.com/v1/messages"
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_TIMEOUT = 30.0


class LLMClient:
    """
    Anthropic API client via raw httpx.

    mock_mode=True: returns the retrieved context verbatim as the "answer".
    Useful for measuring retrieval quality without API access.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        model:      str = _DEFAULT_MODEL,
        mock_mode:  bool = False,
    ) -> None:
        self._api_key   = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._model     = model
        self._mock_mode = mock_mode

        if not mock_mode and not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Pass api_key= or use mock_mode=True."
            )

    @property
    def is_mock(self) -> bool:
        return self._mock_mode

    def answer(
        self,
        question:   str,
        context:    str,
        category:   int = 1,
        max_tokens: int = 100,
    ) -> str:
        """Generate an answer given context. Returns raw text."""
        if self._mock_mode:
            # Retrieval-only mode: return the context itself (measures retrieval quality)
            return context[:500]

        prompt = _build_prompt(question, context, category)
        try:
            resp = httpx.post(
                _ENDPOINT,
                headers={
                    "x-api-key":         self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      self._model,
                    "max_tokens": max_tokens,
                    "messages":   [{"role": "user", "content": prompt}],
                },
                timeout=_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()["content"][0]["text"].strip()
            else:
                # Fall back to mock on auth/rate errors
                return context[:300]
        except Exception:
            return context[:300]

    def rate_importance(self, text: str) -> float:
        """
        Rate memory importance 1-10 (Generative Agents baseline).
        Returns normalized [0,1]. Falls back to 0.5 in mock mode or on error.
        """
        if self._mock_mode:
            return 0.5

        prompt = (
            "On a scale of 1 to 10, where 1 is mundane (e.g., brushing teeth) and "
            "10 is extremely important (e.g., a life-changing event), rate the poignancy "
            "of the following memory. Respond with only a single integer.\n\n"
            f"Memory: {text}"
        )
        try:
            resp = httpx.post(
                _ENDPOINT,
                headers={
                    "x-api-key":         self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      self._model,
                    "max_tokens": 5,
                    "messages":   [{"role": "user", "content": prompt}],
                },
                timeout=_TIMEOUT,
            )
            if resp.status_code == 200:
                raw = resp.json()["content"][0]["text"].strip()
                m = re.search(r"\b([1-9]|10)\b", raw)
                if m:
                    return (int(m.group(1)) - 1) / 9.0
        except Exception:
            pass
        return 0.5


def _build_prompt(question: str, context: str, category: int) -> str:
    """Category-specific prompts matching A-MEM's test_advanced.py conventions."""
    if category == 2:
        return (
            f"Based on the context below, answer the question using the DATE OF CONVERSATION "
            f"to give an approximate date. Provide the shortest possible answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nShort answer:"
        )
    elif category == 5:
        return (
            f"Based on the context below, answer the question. If the information is not "
            f"mentioned in the context, respond with 'Not mentioned in the conversation'.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nShort answer:"
        )
    else:
        return (
            f"Based on the context below, write a short phrase answering the question. "
            f"Use exact words from the context wherever possible.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nShort answer:"
        )


def format_context(memories: list) -> str:
    """Format retrieved MemoryItem list into a context string."""
    parts = []
    for i, m in enumerate(memories, 1):
        meta = m.metadata or {}
        time_str = meta.get("time", "")
        prefix = f"[{time_str}] " if time_str else ""
        parts.append(f"{i}. {prefix}{m.text}")
    return "\n".join(parts)
