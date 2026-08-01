"""
RouterClient: cached, budget-capped OpenAI-compatible client for tokenrouter.com.

Mirrors eval_qa_together.TogetherClient exactly (same disk-cache layout, same
BudgetExceeded discipline) but speaks plain OpenAI chat-completions HTTP against
an aggregator base URL, so accuracy rows can run on the backbones the SOTA
papers published with — specifically gpt-4o-mini (LiCoMemory's second backbone;
Llama-70B rows keep using the existing Together client, which already hosts it).

Env:
  TOKEN_ROUTER_API_KEY    required
  TOKEN_ROUTER_BASE_URL   default https://api.tokenrouter.com (code appends
                          /v1/chat/completions and /v1/models itself)

Pricing: aggregators route dynamically, so per-token cost is estimated
conservatively (DEFAULT_PRICE_PER_TOKEN, overridable per instance) — the budget
cap is a safety brake, not an invoice.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).parent.parent.parent
CACHE_DIR = _REPO / "eval_cache" / "router"

# tokenrouter.com requires the provider-prefixed model id (confirmed via
# GET /v1/models -- unprefixed "gpt-4o-mini" 404s as "no available channel").
DEFAULT_MODEL = "openai/gpt-4o-mini"
# Conservative estimate (gpt-4o-mini is ~$0.15/M in, $0.60/M out as of 2026;
# we charge the higher rate on ALL tokens so the brake trips early, never late).
DEFAULT_PRICE_PER_TOKEN = 0.60 / 1_000_000


class BudgetExceeded(RuntimeError):
    """Kept type-compatible with eval_qa_together.BudgetExceeded via duck-name;
    callers that need to catch both should catch (RuntimeError) or import both.
    eval_qa_together.main() catches both explicitly."""
    pass


class RouterClient:
    """Cached, budget-capped OpenAI-compatible chat client (tokenrouter.com)."""

    def __init__(self, budget_usd: float, model: str = DEFAULT_MODEL,
                 price_per_token: float = DEFAULT_PRICE_PER_TOKEN) -> None:
        self._api_key = os.environ.get("TOKEN_ROUTER_API_KEY")
        if not self._api_key:
            raise RuntimeError("TOKEN_ROUTER_API_KEY is not set (add it to .env "
                               "and `set -a; source .env; set +a`)")
        self._base_url = os.environ.get(
            "TOKEN_ROUTER_BASE_URL", "https://api.tokenrouter.com").rstrip("/")
        self._model = model
        self._budget = budget_usd
        self._price = price_per_token
        self.spent_usd = 0.0
        self.calls = 0
        self.cache_hits = 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def chat(self, prompt: str, max_tokens: int = 96) -> tuple[str, float]:
        """Returns (text, api_latency_seconds; 0.0 on cache hit)."""
        key = hashlib.sha256(f"{self._model}||{max_tokens}||{prompt}"
                             .encode()).hexdigest()
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            self.cache_hits += 1
            return json.loads(cache_file.read_text())["text"], 0.0

        if self.spent_usd >= self._budget:
            raise BudgetExceeded(
                f"budget ${self._budget:.2f} reached (spent ${self.spent_usd:.4f})")

        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode())
        latency = time.perf_counter() - t0

        text = resp["choices"][0]["message"]["content"] or ""
        usage = resp.get("usage") or {}
        n_tok = (usage.get("prompt_tokens", len(prompt) // 4)
                 + usage.get("completion_tokens", max_tokens))
        self.spent_usd += n_tok * self._price
        self.calls += 1
        cache_file.write_text(json.dumps({"text": text, "tokens": n_tok}))
        return text, latency
