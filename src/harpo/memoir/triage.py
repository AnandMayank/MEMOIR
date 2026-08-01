"""
TRIAGE-α: Asynchronous write-time quality gating.

Computes w_rel, w_rec (initial), w_conf for a new MemoryItem against the
current session centroid and store snapshot.  Called via asyncio.create_task()
so the MemoirStore.add() return path is never blocked.

All computation is pure — no side effects.  The store applies the result
via _update_item_wvector() when the task completes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .schema import MemoryItem, WVector

# Negation tokens used for conflict detection (syntactic, deterministic).
# Known limitation: semantic conflicts ("X is good" vs "X is bad") are not captured.
_NEGATION = frozenset({
    "not", "no", "never", "isn't", "aren't", "wasn't", "weren't",
    "doesn't", "didn't", "won't", "can't", "cannot", "incorrect",
    "false", "wrong", "actually", "correction", "mistaken", "untrue",
})

_CONFLICT_COSINE_THRESHOLD = 0.85
_SESSION_WINDOW = 20  # items used for rolling centroid


@dataclass
class TriageConfig:
    conflict_threshold: float = _CONFLICT_COSINE_THRESHOLD
    session_window:     int   = _SESSION_WINDOW
    lambda_rec:         float = 0.1  # recency decay rate (age in hours)


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\b\w+\b", text.lower()))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _update_centroid(
    centroid: np.ndarray,
    n_seen: int,
    new_embedding: np.ndarray,
    window: int,
) -> tuple[np.ndarray, int]:
    """Rolling mean centroid over the last `window` items."""
    effective_n = min(n_seen, window)
    new_centroid = (centroid * effective_n + new_embedding) / (effective_n + 1)
    # Re-normalize so centroid stays on unit sphere
    norm = np.linalg.norm(new_centroid)
    if norm > 1e-9:
        new_centroid = new_centroid / norm
    return new_centroid.astype(np.float32), n_seen + 1


def _compute_w_rel(embedding: np.ndarray, centroid: np.ndarray, n_seen: int) -> float:
    """Relevance to session: cosine with centroid, clipped to [0, 1]."""
    if n_seen == 0:
        return 0.5  # neutral for the very first item
    return float(np.clip(_cosine(embedding, centroid), 0.0, 1.0))


def _compute_w_conf(
    new_item: MemoryItem,
    store_snapshot: Sequence[MemoryItem],
    threshold: float,
) -> float:
    """
    Conflict score: fraction of close neighbors (cosine > threshold) whose
    text has negation token overlap with the new item's text.

    Will be low on most non-adversarial data — documented limitation.
    """
    if not store_snapshot or new_item.embedding is None:
        return 0.0

    new_tokens = _tokenize(new_item.text)
    new_negations = new_tokens & _NEGATION

    close_count = 0
    conflict_count = 0

    for existing in store_snapshot:
        if existing.embedding is None:
            continue
        sim = _cosine(new_item.embedding, existing.embedding)
        if sim < threshold:
            continue
        close_count += 1
        existing_tokens = _tokenize(existing.text)
        existing_negations = existing_tokens & _NEGATION
        # Conflict: similar content tokens AND at least one side has negation tokens.
        # Example: "Python is fast" vs "Python is not fast" → same content, new has negation.
        new_content = new_tokens - _NEGATION
        existing_content = existing_tokens - _NEGATION
        content_overlap = new_content & existing_content
        if content_overlap and (new_negations or existing_negations):
            conflict_count += 1

    if close_count == 0:
        return 0.0
    return conflict_count / close_count


async def triage_alpha(
    item: MemoryItem,
    store_snapshot: list[MemoryItem],
    centroid: np.ndarray,
    n_seen: int,
    config: TriageConfig = TriageConfig(),
) -> tuple[MemoryItem, np.ndarray, int]:
    """
    Compute w_rel and w_conf for item; return updated item, new centroid, new n_seen.

    w_rec is set to 1.0 at write time and recomputed live at retrieval.
    w_imp stays 0.0 — only TRACE-FB updates it post-trajectory.
    """
    if item.embedding is None:
        return item, centroid, n_seen

    embedding = item.embedding

    w_rel = _compute_w_rel(embedding, centroid, n_seen)
    w_conf = _compute_w_conf(item, store_snapshot, config.conflict_threshold)

    item.w = WVector(
        w_rel=w_rel,
        w_rec=1.0,    # initial value; live_w_rec() recomputes at retrieval time
        w_conf=w_conf,
        w_imp=0.0,    # populated by TRACE-FB after trajectory completion
    )

    new_centroid, new_n_seen = _update_centroid(
        centroid, n_seen, embedding, config.session_window
    )
    return item, new_centroid, new_n_seen
