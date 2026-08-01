"""
GATE-LinUCB: online-learned, context-dependent retrieval weights.

Replaces RECALL-Q's fixed hyperparameters (alpha_bm25, lambda_mmr, beta_floor,
gamma_imp) with a per-query choice among a small set of weight profiles,
selected by LinUCB (Li et al., 2010) on the query embedding.

Design lineage: CHARM (HARPO, ACL 2026) learns context-dependent weights over
quality dimensions via meta-learning; here the same idea is made training-free —
closed-form ridge updates per arm, no gradients, no labels. The reward is the
signal TRACE-FB already computes at feedback time (max attribution over the
retrieved set), so the gate learns from exactly the same outcome observations
as w_imp itself.

Mechanics:
  context  x = unit-norm random projection of the query embedding (384 → 32)
  per arm  a: ridge state A_a (via its inverse, Sherman–Morrison) and b_a
  select:  argmax_a  θ_a·x + ucb_alpha·sqrt(x^T A_a^{-1} x),  θ_a = A_a^{-1} b_a
  update:  A_a += x x^T,  b_a += reward·x       (closed-form, O(d²))

select() is deterministic given the gate state — feedback_answer()'s internal
re-search of the same query therefore re-selects the same arm it rewards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightProfile:
    """One RECALL-Q hyperparameter configuration (an arm)."""
    name:       str
    alpha_bm25: float
    beta_floor: float
    gamma_imp:  float
    lambda_mmr: float


# Profiles span the qualitative retrieval regimes rather than a dense grid:
# the point is per-query *routing* (verbatim vs semantic vs credit-trusting vs
# diversity-needing queries), not hyperparameter search.
DEFAULT_PROFILES: tuple[WeightProfile, ...] = (
    WeightProfile("baseline",   alpha_bm25=0.3, beta_floor=0.1, gamma_imp=1.0, lambda_mmr=0.1),
    WeightProfile("lexical",    alpha_bm25=0.6, beta_floor=0.1, gamma_imp=1.0, lambda_mmr=0.1),
    WeightProfile("semantic",   alpha_bm25=0.1, beta_floor=0.1, gamma_imp=1.0, lambda_mmr=0.1),
    WeightProfile("importance", alpha_bm25=0.3, beta_floor=0.1, gamma_imp=2.0, lambda_mmr=0.1),
    WeightProfile("diverse",    alpha_bm25=0.3, beta_floor=0.1, gamma_imp=1.0, lambda_mmr=0.3),
)

PROJECTION_DIM: int = 32
UCB_ALPHA: float = 0.5
PROJECTION_SEED: int = 13  # fixed: context features must be stable across runs


class LinUCBGate:
    def __init__(
        self,
        dim: int,
        profiles: tuple[WeightProfile, ...] = DEFAULT_PROFILES,
        proj_dim: int = PROJECTION_DIM,
        ucb_alpha: float = UCB_ALPHA,
    ) -> None:
        self.profiles = profiles
        self._ucb_alpha = ucb_alpha
        rng = np.random.default_rng(PROJECTION_SEED)
        self._proj = (rng.standard_normal((proj_dim, dim)) / math.sqrt(proj_dim)
                      ).astype(np.float32)
        n_arms = len(profiles)
        self._A_inv = [np.eye(proj_dim, dtype=np.float32) for _ in range(n_arms)]
        self._b = [np.zeros(proj_dim, dtype=np.float32) for _ in range(n_arms)]
        self.select_counts = [0] * n_arms
        self.update_counts = [0] * n_arms

    def _context(self, q_vec: np.ndarray) -> np.ndarray:
        x = self._proj @ q_vec.astype(np.float32)
        return x / (np.linalg.norm(x) + 1e-9)

    def select(self, q_vec: np.ndarray) -> int:
        x = self._context(q_vec)
        best_arm, best_score = 0, -np.inf
        for a in range(len(self.profiles)):
            theta = self._A_inv[a] @ self._b[a]
            explore = math.sqrt(max(0.0, float(x @ self._A_inv[a] @ x)))
            score = float(theta @ x) + self._ucb_alpha * explore
            if score > best_score:
                best_arm, best_score = a, score
        self.select_counts[best_arm] += 1
        return best_arm

    def update(self, q_vec: np.ndarray, arm: int, reward: float) -> None:
        x = self._context(q_vec)
        A_inv = self._A_inv[arm]
        # Sherman–Morrison: (A + xx^T)^{-1} = A^{-1} − (A^{-1}x x^T A^{-1}) / (1 + x^T A^{-1} x)
        Ax = A_inv @ x
        self._A_inv[arm] = A_inv - np.outer(Ax, Ax) / (1.0 + float(x @ Ax))
        self._b[arm] += reward * x
        self.update_counts[arm] += 1
