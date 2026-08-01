"""
LoCoMo Evaluation Harness for MEMOIR.

Compares memory systems on the LoCoMo benchmark (10 samples, ~19-32 sessions each).
Data: data/locomo10.json

Systems evaluated:
  cosine         — pure cosine retrieval (flat_w=True, no BM25, no MMR, no TRACE-FB)
  recency_cosine — recency + cosine (MemoryBank analog)
  gen_agents     — Park et al. (2023): recency + LLM-importance + cosine
                   (importance=0.5 in mock mode; set ANTHROPIC_API_KEY for real LLM call)
  memoir_no_trace — MEMOIR w-vector retrieval, no TRACE-FB (≈ analytical analog of GA)
  memoir_trace    — MEMOIR + TRACE-FB (paper contribution)
  oracle_trace    — MEMOIR + binary GT feedback (upper bound; requires GT correctness signal)

Multi-session protocol:
  For session i in 1..N:
    1. Add all turns from session i to the memory system.
    2. Evaluate QA questions attributed to session i.
    3. Run TRACE-FB on session i trajectory → update w_imp.
  Report: context_recall@5 and token_f1 averaged across sessions 2..N
  (session 1 has w_imp=0 — no prior TRACE-FB signal).

Primary metric: context_recall@5 (retrieval quality, no LLM required)
  = fraction of QA pairs where at least one retrieved memory contains a GT answer token
Secondary metric: token_f1 (token overlap between best-retrieved text and GT)
  = F1 between tokenized retrieved-context and GT answer

Full QA accuracy (LLM-generated answers) requires setting:
  ANTHROPIC_API_KEY=<direct api key, not OAuth token>

Usage:
  # Default: all systems, context_recall metric
  python benchmarks/memoir/eval_locomo.py \\
    --dataset data/locomo10.json \\
    --output results/locomo.json

  # Subset of systems
  python benchmarks/memoir/eval_locomo.py --systems cosine,memoir_no_trace,memoir_trace
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent.parent
_LOCAL_DATA = _REPO / "data"
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))  # repo root — makes `benchmarks.memoir.*` importable
                                 # explicitly rather than relying on an editable
                                 # pip install; matters in spawn-context worker
                                 # processes (--workers > 1), which re-run this
                                 # module's top-level code in a fresh interpreter
                                 # that may not be the venv the package was
                                 # installed into.
sys.path.insert(0, str(Path(__file__).parent))  # vendored load_dataset.py

from harpo.memoir.store import MemoirStore
from harpo.memoir.trace_fb import rouge1_recall

# ── lightweight metrics ───────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


def token_f1(pred: str, gt: str) -> float:
    """Token-level F1 between prediction and reference (SQuAD-style)."""
    pred_toks = _tokenize(pred)
    gt_toks   = _tokenize(gt)
    if not pred_toks or not gt_toks:
        return float(pred_toks == gt_toks)
    pred_set = defaultdict(int)
    for t in pred_toks:
        pred_set[t] += 1
    gt_set = defaultdict(int)
    for t in gt_toks:
        gt_set[t] += 1
    common = sum(min(pred_set[t], gt_set[t]) for t in gt_set)
    if common == 0:
        return 0.0
    prec = common / len(pred_toks)
    rec  = common / len(gt_toks)
    return 2 * prec * rec / (prec + rec)


def context_recall_at_k(retrieved_texts: list[str], gt_answer: str) -> float:
    """1.0 if any retrieved memory shares at least one token with GT answer."""
    gt_set = set(_tokenize(gt_answer))
    if not gt_set:
        return 0.0
    for text in retrieved_texts:
        if set(_tokenize(text)) & gt_set:
            return 1.0
    return 0.0


def best_token_f1(retrieved_texts: list[str], gt_answer: str) -> float:
    """Max token_f1 across retrieved memories vs GT (measures retrieval ceiling)."""
    if not retrieved_texts:
        return 0.0
    return max(token_f1(t, gt_answer) for t in retrieved_texts)


# ── Evidence-based retrieval metrics (v7 — replaces lexical ctx@k as headline) ─
# LoCoMo annotates each QA with evidence dia_ids ('D<session>:<turn>') and every
# turn carries its dia_id, so Recall@k is exact and non-lexical: no credit for
# sharing a stray token with the GT, no blindness on numeric answers.

def _item_dia_ids(m) -> set:
    """All evidence dia_ids a memory covers. Consolidation-merged items carry
    the union of their constituents' dia_ids under metadata['dia_ids'];
    plain items carry a single metadata['dia_id']."""
    meta = getattr(m, "metadata", None) or {}
    ids = set(meta.get("dia_ids", ()))
    single = meta.get("dia_id")
    if single:
        ids.add(single)
    return ids


def evidence_recall(retrieved_items, evidence: list[str]) -> float:
    """1.0 if ANY retrieved memory is (or contains) an annotated evidence turn."""
    ev = set(evidence or [])
    if not ev:
        return 0.0
    return 1.0 if any(_item_dia_ids(m) & ev for m in retrieved_items) else 0.0


def full_evidence_recall(retrieved_items, evidence: list[str]) -> float:
    """1.0 only if ALL annotated evidence turns were retrieved (strict —
    the honest multi_hop metric: every hop must be in context)."""
    ev = set(evidence or [])
    if not ev:
        return 0.0
    got: set = set()
    for m in retrieved_items:
        got |= _item_dia_ids(m)
    return 1.0 if ev <= got else 0.0


def retrieved_token_cost(retrieved_texts: list[str]) -> int:
    """Whitespace tokens fed to the answering LLM's context for this query."""
    return sum(len(t.split()) for t in retrieved_texts)


# ── system factory ────────────────────────────────────────────────────────────

def make_system(name: str, llm_client=None, model_name: str = "all-MiniLM-L6-v2",
                commit_calibration=None) -> object:
    """Return an initialized memory system for the given name.
    model_name: embedding model or local path (e.g. 'models/memoir-encoder-v1')
    commit_calibration: Optional[CommitCalibration] (consolidate.py) -- if set,
        memoir_commit/memoir_v3 use the calibrated LLR gate instead of the
        fixed COMMIT_DELTA/N_COMMIT fallback (see calibrate_commit.py).
    """
    if name == "cosine":
        s = MemoirStore(model_name=model_name)
        s._system_name = "cosine"
        s._ablation = {"flat_w": True, "no_bm25": True, "no_mmr": True}
        return s

    if name == "simutil_ucb":
        # RetroAgent (arXiv:2603.08561) memory mechanism adapted: SimUtil-UCB
        # retrieval + uniform-EMA utility. The apples-to-apples baseline for
        # TRACE-CF's per-memory credit — see baseline_simutil.py header.
        from benchmarks.memoir.baseline_simutil import SimUtilStore
        s = SimUtilStore(model_name=model_name)
        s._system_name = name
        return s

    if name == "recency_cosine":
        from benchmarks.memoir.baseline_cosine import RecencyCosineStore
        return RecencyCosineStore()

    if name == "gen_agents":
        from benchmarks.memoir.baseline_gen_agents import GenerativeAgentsStore
        return GenerativeAgentsStore(llm_client=llm_client)

    if name in ("memoir_no_trace", "memoir_trace", "oracle_trace"):
        # alpha_trace=0.3 (up from 0.1) for faster w_imp accumulation across sessions
        s = MemoirStore(model_name=model_name, alpha_trace=0.3, use_co_cite=False)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_co_cite":
        # memoir_trace + co-citation graph: ablation isolating graph contribution
        s = MemoirStore(model_name=model_name, alpha_trace=0.3, use_co_cite=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_trace_rouge1":
        # ROUGE1-only TRACE-FB — explicit ablation name for LLM comparison
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_llm_attr=False)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_trace_llm":
        # TRACE-FB with LLM fallback (claude-haiku-4-5) for ROUGE1-blind queries
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_llm_attr=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_asym":
        # memoir_trace + asymmetric TRACE-FB decay (alpha_neg = alpha*0.2)
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, alpha_neg=0.06)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_foresight":
        # memoir_trace + FORESIGHT-NORM (HFR-inspired path-baseline normalization)
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_foresight_norm=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_weibull":
        # memoir_trace + Weibull temporal decay on w_imp at retrieval time
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_weibull=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_cf_trace":
        # TRACE-CF: counterfactual LOO answer-likelihood attribution (local
        # Qwen2.5-0.5B) instead of ROUGE1. Feedback scan capped at k=16 (not 50):
        # the LOO prompt must hold all scanned memories, and 50 memories both
        # exceeds the scorer's context budget and scales cost linearly.
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_cf_attr=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_ucb":
        # memoir_trace + UCB exploration bonus on w_imp (SimUtil-UCB adapted,
        # RetroAgent arXiv:2603.08561) — counters rich-get-richer lock-in
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_ucb=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_gate":
        # memoir_trace + GATE-LinUCB: per-query learned retrieval weights
        # replacing the fixed alpha/beta/gamma/lambda constants
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_learned_weights=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_commit":
        # memoir_trace + CCOMMIT (CIFS analog): credit retrieval-effective only
        # after the commit criterion is met -- calibrated LLR (calibrate_commit.py)
        # if commit_calibration is provided, else the fixed COMMIT_DELTA/N_COMMIT
        # fallback (diagnosed as mismatched with ROUGE1 -- see PAPER_COMPONENTS.md)
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_commit=True,
                        commit_calibration=commit_calibration)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_dlm":
        # memoir_trace + CORE-DLM: diversity-driven bounded core memory bank
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_core_bank=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_v3":
        # cf_full + CCOMMIT + CORE-DLM: full v3 stack
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_cf_attr=True,
                        use_learned_weights=True, use_ucb=True,
                        use_commit=True, use_core_bank=True,
                        commit_calibration=commit_calibration)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_cf_full":
        # TRACE-CF + gate + UCB: all three new components together
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_cf_attr=True,
                        use_learned_weights=True, use_ucb=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_full":
        # All new blocks combined: asymmetric decay + foresight-norm + weibull + co-cite
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=True, alpha_neg=0.06,
                        use_foresight_norm=True, use_weibull=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_reliability":
        # memoir_trace + RELIABILITY: Laplace-smoothed eta() soft bonus
        # (n_pass/n_trial tracked via reliability_correct — decoupled from
        # ROUGE1 attribution, which stays untouched). MSCE-inspired
        # (arXiv:2607.16621 sec 4.3), oracle-feedback-only signal.
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_reliability=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_anchor":
        # MULTI-ANCHOR (Item 1, Stern 2002 cognitive reserve): memoir_trace +
        # auxiliary query-embedding anchors cached on TRACE-credited items
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_multi_anchor=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_anchor_strict":
        # MULTI-ANCHOR mitigation experiment: memoir_anchor's full-scale run
        # (docs/PAPER_COMPONENTS.md #11) showed a -9.51pp temporal regression,
        # hypothesized to come from the very low SPARSE_CITE_DELTA=0.02 anchor
        # threshold letting weak/noisy credits become retrieval anchors.
        # anchor_min_attr=0.15 (COMMIT_DELTA) requires a much stronger signal
        # before an anchor is recorded.
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_multi_anchor=True,
                        anchor_min_attr=0.15)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_spacing":
        # SPACING (Item 2, Landauer & Bjork 1978 spacing effect): memoir_trace +
        # session-gap saturating multiplier on the TRACE-FB delta
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_spacing=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_spacing_ucb":
        # SPACING + UCB together — the (c) arm of Item 2's ablation design
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_spacing=True, use_ucb=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_stratum":
        # TYPE-EMERGENT TAXONOMY (Item 3, Tulving 1972): memoir_trace +
        # stratum-gated w_rec decay (semantic = certified-merge output decays
        # slower). Requires the eval harness to call store.consolidate() at
        # session boundaries for "semantic" items to ever exist.
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_stratum_decay=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_weibull_cfgate":
        # CF-SHIFT GATE (Item 4) applied to Weibull decay: exempts
        # high-TRACE-CF-shift (load-bearing) items from the friction that
        # otherwise regresses multi_hop (docs/PAPER_COMPONENTS.md CUT #6)
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_cf_attr=True, use_weibull=True,
                        use_cf_shift_gate=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_commit_cfgate":
        # CF-SHIFT GATE applied to CCOMMIT
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_cf_attr=True, use_commit=True,
                        use_cf_shift_gate=True, commit_calibration=commit_calibration)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_dlm_cfgate":
        # CF-SHIFT GATE applied to CORE-DLM
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_cf_attr=True, use_core_bank=True,
                        use_cf_shift_gate=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_ucb_anchor":
        # Stack MULTI-ANCHOR (strict threshold, see memoir_anchor_strict) ON
        # TOP of UCB rather than as a standalone alternative — UCB alone is
        # the strongest single retrieval-side win in the ladder (TABLE 3,
        # +1.84pp all-category-significant); the goal is to add anchor's
        # single_hop/adversarial gains without losing UCB's own contribution.
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_ucb=True, use_multi_anchor=True,
                        anchor_min_attr=0.15)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_ucb_stratum":
        # UCB + stratum-gated decay stacked together. Requires the eval
        # harness to actually call store.consolidate() at session boundaries
        # for "semantic" items to exist — see run_eval's --consolidate flag.
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_ucb=True, use_stratum_decay=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_ucb_all":
        # Full stack: UCB (retrieval, proven) + strict-threshold anchor +
        # spacing + stratum decay, all layered on top of the strongest single
        # baseline rather than compared to it in isolation. Also requires
        # --consolidate for the stratum component to have any effect.
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_ucb=True, use_multi_anchor=True,
                        anchor_min_attr=0.15, use_spacing=True,
                        use_stratum_decay=True)
        s._system_name = name
        s._ablation = {}
        return s

    if name == "memoir_cf_verified":
        # memoir_cf_full + verification gate: TRACE credit scaled by answer
        # plausibility (abstention hard-gate + scorer-lp soft gate) — the
        # anti-self-reinforcing-error ablation. Same config otherwise.
        s = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_cf_attr=True,
                        use_learned_weights=True, use_ucb=True,
                        use_verification_gate=True)
        s._system_name = name
        s._ablation = {}
        return s

    raise ValueError(f"Unknown system: {name!r}")


def _parse_locomo_datetime(date_time: Optional[str]) -> Optional[float]:
    """
    Parse LoCoMo's 'session_N_date_time' strings (e.g. '1:56 pm on 8 May, 2023')
    to an epoch float. Sessions in a sample span real weeks/months — wiring this
    into MemoryItem.created_at lets w_rec / Weibull decay see the dataset's real
    time gaps instead of the near-zero gaps of a fast, same-process eval run.
    Returns None (→ callers fall back to wall-clock time.time()) on parse failure.
    """
    if not date_time:
        return None
    try:
        import datetime as _dt
        return _dt.datetime.strptime(date_time, "%I:%M %p on %d %B, %Y").timestamp()
    except ValueError:
        return None


def system_add(sys_name: str, store, text: str, metadata: dict,
               created_at: Optional[float] = None) -> None:
    if hasattr(store, "add"):
        if sys_name == "gen_agents":
            store.add(text)
        elif created_at is not None:
            store.add(text, metadata=metadata, created_at=created_at)
        else:
            store.add(text, metadata=metadata)


def system_retrieve(sys_name: str, store, query: str, k: int,
                     now: Optional[float] = None) -> list[str]:
    """Return retrieved text strings."""
    if sys_name == "gen_agents":
        items = store.search(query, k=k)
        return [getattr(m, "text", str(m)) for m in items]
    if sys_name in ("recency_cosine", "cosine_only_bare"):
        items = store.search(query, k=k)
        return [m.text for m in items]
    # MemoirStore variants
    ablation = getattr(store, "_ablation", {})
    items = store.search(query, k=k, now=now, **ablation)
    return [m.text for m in items]


_MEMOIR_SCORED_SYSTEMS = (
    "memoir_no_trace", "memoir_trace", "oracle_trace",
    "memoir_co_cite", "memoir_trace_rouge1", "memoir_trace_llm",
    "memoir_asym", "memoir_foresight", "memoir_weibull", "memoir_full",
    "memoir_cf_trace", "memoir_ucb", "memoir_gate", "memoir_cf_full",
    "memoir_commit", "memoir_dlm", "memoir_v3", "memoir_cf_verified",
    "memoir_reliability",
    # Cognition.md extensions (Items 1-4)
    "memoir_anchor", "memoir_anchor_strict", "memoir_spacing", "memoir_spacing_ucb",
    "memoir_stratum",
    "memoir_weibull_cfgate", "memoir_commit_cfgate", "memoir_dlm_cfgate",
    "memoir_ucb_anchor", "memoir_ucb_stratum", "memoir_ucb_all",
)
_MEMOIR_FEEDBACK_SYSTEMS = (
    "memoir_trace", "memoir_co_cite", "memoir_trace_rouge1", "memoir_trace_llm",
    "memoir_asym", "memoir_foresight", "memoir_weibull", "memoir_full",
    "memoir_cf_trace", "memoir_ucb", "memoir_gate", "memoir_cf_full",
    "memoir_commit", "memoir_dlm", "memoir_v3", "memoir_cf_verified",
    "memoir_reliability",
    "memoir_anchor", "memoir_anchor_strict", "memoir_spacing", "memoir_spacing_ucb",
    "memoir_stratum",
    "memoir_weibull_cfgate", "memoir_commit_cfgate", "memoir_dlm_cfgate",
    "memoir_ucb_anchor", "memoir_ucb_stratum", "memoir_ucb_all",
)


def system_retrieve_items(sys_name: str, store, query: str, k: int,
                          now: Optional[float] = None) -> list:
    """Return retrieved item objects (with .text / .metadata) for any system
    whose search() yields items — MemoirStore variants and SimUtilStore."""
    if sys_name == "gen_agents":
        return store.search(query, k=k)
    ablation = getattr(store, "_ablation", {})
    try:
        return store.search(query, k=k, now=now, **ablation)
    except TypeError:  # stores without now/ablation kwargs
        return store.search(query, k=k)


def system_retrieve_with_scores(sys_name: str, store, query: str, k: int,
                                 now: Optional[float] = None):
    """Return (item_or_text, score) list — used for TRACE-FB."""
    if sys_name in _MEMOIR_SCORED_SYSTEMS:
        ablation = getattr(store, "_ablation", {})
        return store.search_with_scores(query, k=k, now=now, **ablation)
    return []


def system_feedback(sys_name: str, store, query: str, answer: str,
                    oracle_correct: Optional[bool] = None,
                    now: Optional[float] = None,
                    session_idx: Optional[int] = None) -> None:
    # k=50: wider scan so memories outside top-5 can accumulate w_imp
    # memoir_trace: ROUGE1 attribution with GT text (simulates LLM answer ≈ GT)
    # oracle_trace: binary oracle — if retrieval succeeded (ctx_recall>0.5), all retrieved
    #   memories get attribution=1.0; otherwise 0.0. This is the true upper bound.
    # session_idx: SPACING (Item 2) — the current LoCoMo session index (sess_id),
    #   threaded through to feedback_answer() for memoir_spacing*/only has an
    #   effect when the store's use_spacing=True.
    if sys_name == "simutil_ucb":
        if oracle_correct is not None:
            store.feedback_uniform(query, 1.0 if oracle_correct else 0.0, k=5)
        return
    if sys_name in _MEMOIR_FEEDBACK_SYSTEMS and answer:
        # cf: scan capped at 16 — the LOO prompt must hold every scanned memory
        fb_k = 16 if sys_name in ("memoir_cf_trace", "memoir_cf_full",
                                  "memoir_v3", "memoir_weibull_cfgate",
                                  "memoir_commit_cfgate", "memoir_dlm_cfgate") else 50
        # reliability_correct: decoupled from oracle_correct so RELIABILITY's
        # n_pass/n_trial tracking can accrue without switching this system's
        # attribution path (ROUGE1/CF stays as-is; only memoir_reliability
        # reads the bonus at retrieval time via use_reliability=True).
        store.feedback_answer(query, answer, k=fb_k, now=now,
                              reliability_correct=oracle_correct,
                              session_idx=session_idx)
    elif sys_name == "oracle_trace" and oracle_correct is not None:
        # Binary oracle: use gt text when correct, empty string when wrong
        # This calls apply_trace_feedback with oracle_correct flag
        store.feedback_answer(
            query,
            answer if oracle_correct else "",
            k=50,
            oracle_correct=oracle_correct,
            now=now,
            session_idx=session_idx,
        )


# ── per-sample evaluation ─────────────────────────────────────────────────────

def evaluate_sample(
    sample,
    sys_name: str,
    store,
    k: int = 5,
    trace_mode: str = "gt",   # "gt" (use GT as TRACE-FB answer) or "none"
    session_split: bool = True,
    verbose: bool = False,
    prune_ratio: float = 1.0,  # PRUNE-EFP: <1.0 prunes store after each session
    perqa_rows: Optional[list] = None,  # if given, raw per-QA rows appended here
    judge=None,  # TYPE-EMERGENT TAXONOMY (Item 3): FaithfulnessJudge instance — when set,
                 # runs store.consolidate() after each session's feedback pass so
                 # certified-merge output can be tagged stratum="semantic" (store.py).
                 # None (default) preserves the exact prior behavior — no consolidation.
    consolidate_max_clusters: int = 4,
) -> dict:
    """
    Run multi-session evaluation for one sample on one memory system.

    Returns dict with per-category metrics.
    """
    sessions = sample.conversation.sessions
    sorted_sess_ids = sorted(sessions.keys())

    # Build a map from evidence string → session id for session-split eval
    # Evidence format: D<day>:<turn> where day = session index
    # When session_split=False, feed all turns then evaluate all QAs.

    if not session_split:
        # Feed all turns, then eval all QAs
        last_epoch = None
        for sess_id in sorted_sess_ids:
            sess = sessions[sess_id]
            sess_epoch = _parse_locomo_datetime(sess.date_time)
            last_epoch = sess_epoch if sess_epoch is not None else last_epoch
            for turn in sess.turns:
                text = f"Speaker {turn.speaker} says: {turn.text}"
                system_add(sys_name, store, text,
                           {"session": sess_id, "time": sess.date_time,
                            "dia_id": turn.dia_id},
                           created_at=sess_epoch)
        qa_results = _eval_qa_batch(sys_name, store, sample.qa, k, trace_mode, verbose,
                                     now=last_epoch)
        return qa_results

    # Session-by-session (multi-session protocol — the paper's evaluation mode)
    all_results = []
    session_seen = set()

    for sess_id in sorted_sess_ids:
        sess = sessions[sess_id]
        # Dataset-native session timestamp (LoCoMo sessions span real weeks/months) —
        # threads through to created_at (write) and now (retrieval/feedback) so
        # w_rec / Weibull decay see real time gaps instead of the fast eval loop's
        # near-zero wall-clock gaps. Falls back to None (→ wall-clock) if unparsable.
        sess_epoch = _parse_locomo_datetime(sess.date_time)
        for turn in sess.turns:
            text = f"Speaker {turn.speaker} says: {turn.text}"
            system_add(sys_name, store, text,
                       {"session": sess_id, "time": sess.date_time,
                        "dia_id": turn.dia_id},
                       created_at=sess_epoch)
        session_seen.add(sess_id)

        # Collect QAs attributed to this session by evidence day
        sess_qa = [
            qa for qa in sample.qa
            if any(ev.startswith(f"D{sess_id}:") for ev in (qa.evidence or []))
        ]

        if sess_id == sorted_sess_ids[0]:
            # Session 1: no prior TRACE-FB signal — evaluate but exclude from paper metric
            _eval_and_feedback(sys_name, store, sess_qa, k, trace_mode, all_results,
                                session=sess_id, count_in_metric=False, verbose=verbose,
                                now=sess_epoch)
        else:
            _eval_and_feedback(sys_name, store, sess_qa, k, trace_mode, all_results,
                                session=sess_id, count_in_metric=True, verbose=verbose,
                                now=sess_epoch)

        # PRUNE-EFP: steady-state store management after each session's feedback
        if prune_ratio < 1.0 and hasattr(store, "prune"):
            store.prune(prune_ratio)

        # TYPE-EMERGENT TAXONOMY (Item 3): sleep-phase certified merge after
        # each session, so certified-merge output can be tagged
        # stratum="semantic" (store.py consolidate()) before the next
        # session's feedback pass evaluates retrieval. Small caps keep judge
        # calls cheap (a handful per session, not per QA).
        if judge is not None and hasattr(store, "consolidate"):
            store.consolidate(judge, max_clusters=consolidate_max_clusters,
                              max_cluster_size=2, retrieval_coverage_min=0.6,
                              certified=True)

    if perqa_rows is not None:
        sid = getattr(sample, "sample_id", None)
        for r in all_results:
            perqa_rows.append({"sample_id": sid, **r})

    return _aggregate(all_results)


def _eval_and_feedback(
    sys_name, store, qa_list, k, trace_mode, results_out,
    session, count_in_metric, verbose, now=None,
) -> None:
    for qa in qa_list:
        gt = str(qa.final_answer)

        t0 = time.perf_counter()
        items5 = system_retrieve_items(sys_name, store, qa.question, k=k, now=now)
        search_ms = (time.perf_counter() - t0) * 1000.0
        retrieved_texts = [getattr(m, "text", str(m)) for m in items5]
        # Separate call: k=15 widens the oversample pool (k_fetch=k*3), so the
        # top-5 of a k=15 call is not identical to the k=5 call — measure both.
        items15 = system_retrieve_items(sys_name, store, qa.question, k=15, now=now)

        ctx_rec = context_recall_at_k(retrieved_texts, gt)   # legacy lexical
        btf1    = best_token_f1(retrieved_texts, gt)
        rec5    = evidence_recall(items5, qa.evidence)
        rec15   = evidence_recall(items15, qa.evidence)
        full5   = full_evidence_recall(items5, qa.evidence)
        full15  = full_evidence_recall(items15, qa.evidence)
        tokens5 = retrieved_token_cost(retrieved_texts)

        # Determine answer and oracle signal for TRACE-FB
        if trace_mode == "gt":
            fb_answer = gt
            # oracle_correct: evidence-based (v7) — was lexical ctx_rec>0.5
            oracle_correct = rec5 > 0.5
        else:
            fb_answer = ""
            oracle_correct = None

        system_feedback(sys_name, store, qa.question, fb_answer,
                         oracle_correct=oracle_correct, now=now,
                         session_idx=session)

        results_out.append({
            "session": session,
            "category": qa.category,
            "question": qa.question,  # pairing key for significance testing
            "ctx_recall": ctx_rec,
            "best_tf1": btf1,
            "rec5": rec5,
            "rec15": rec15,
            "full_ev5": full5,
            "full_ev15": full15,
            "tokens5": tokens5,
            "search_ms": search_ms,
            "count": count_in_metric,
        })

        if verbose:
            print(f"    sess={session} cat={qa.category} "
                  f"rec5={rec5:.0f} rec15={rec15:.0f} ctx={ctx_rec:.0f} "
                  f"gt={gt!r:.30}")


def _eval_qa_batch(sys_name, store, qa_list, k, trace_mode, verbose, now=None) -> dict:
    results = []
    _eval_and_feedback(sys_name, store, qa_list, k, trace_mode, results,
                       session=None, count_in_metric=True, verbose=verbose, now=now)
    return _aggregate(results)


def _aggregate(results: list[dict]) -> dict:
    """Aggregate per-QA results into overall + per-category metrics.

    Dedups by (question, category) before averaging — the online multi-session
    protocol re-evaluates the same question at every session boundary whose
    evidence day it matches (real for multi-hop QAs with cross-session
    evidence), so raw rows over-count repeats as independent trials. This
    mirrors significance.py's load_rows() pairing key (sample_id is constant
    within one evaluate_sample() call, so (question, category) is sufficient
    here) and keeps the LAST occurrence per key, matching load_rows()'s
    dict-overwrite semantics, so _aggregate's n and significance.py's n agree.
    """
    counted_raw = [r for r in results if r.get("count", True)]
    if not counted_raw:
        return {"n": 0, "ctx_recall": 0.0, "best_tf1": 0.0, "by_category": {}}

    dedup: dict[tuple, dict] = {}
    for r in counted_raw:
        dedup[(r["question"], r["category"])] = r
    counted = list(dedup.values())
    n_raw, n_dedup = len(counted_raw), len(counted)
    if n_dedup < n_raw:
        print(f"    [_aggregate] deduped {n_raw} rows -> {n_dedup} unique "
              f"(question, category) pairs ({n_raw - n_dedup} repeats dropped)")

    by_cat: dict[int, list] = defaultdict(list)
    for r in counted:
        by_cat[r["category"]].append(r)

    _EXTRA = ("rec5", "rec15", "full_ev5", "full_ev15", "tokens5", "search_ms")

    def _avg(items: list[dict], key: str) -> float:
        vals = [x[key] for x in items if x.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    cat_metrics = {}
    for cat, items in sorted(by_cat.items()):
        cat_metrics[cat] = {
            "n": len(items),
            "ctx_recall": _avg(items, "ctx_recall"),
            "best_tf1":   _avg(items, "best_tf1"),
            **{key: _avg(items, key) for key in _EXTRA},
        }

    return {
        "n": len(counted),
        "ctx_recall": _avg(counted, "ctx_recall"),
        "best_tf1":   _avg(counted, "best_tf1"),
        **{key: _avg(counted, key) for key in _EXTRA},
        "by_category": cat_metrics,
    }


# ── main evaluation loop ──────────────────────────────────────────────────────

_CATEGORY_NAMES = {
    1: "single_hop_factual",
    2: "temporal",
    3: "multi_hop",
    4: "open_ended",
    5: "adversarial",
}

_ALL_SYSTEMS = [
    "cosine",
    "recency_cosine",
    "gen_agents",
    "memoir_no_trace",
    "memoir_trace",
    "memoir_co_cite",
    "memoir_trace_rouge1",
    "memoir_trace_llm",
    "memoir_cf_trace",
    "oracle_trace",
]


def _eval_one_sample_worker(
    sys_name: str,
    sample,
    k: int,
    trace_mode: str,
    session_split: bool,
    prune_ratio: float,
    model_name: str,
    dump_perqa: bool,
    commit_calibration=None,
) -> tuple:
    """
    Picklable, process-safe unit of work for --workers > 1.

    Each worker process constructs its own store (and, for CF systems, its
    own CFAttributor singleton + CUDA context — spawn start method only,
    never fork, since a forked process would inherit a half-initialized
    CUDA context). Runs on `spawn`: no llm_client (async clients aren't
    picklable) — parallel mode is unavailable for gen_agents/memoir_trace
    with a real LLM feedback answer; run_eval falls back to sequential there.
    commit_calibration is a plain dataclass of floats/lists (consolidate.py:
    CommitCalibration) — picklable, unlike judge_fn closures, so it's passed
    directly rather than rebuilt per-worker from a spec string.
    """
    store = make_system(sys_name, llm_client=None, model_name=model_name,
                        commit_calibration=commit_calibration)
    perqa_rows: Optional[list] = [] if dump_perqa else None
    res = evaluate_sample(
        sample, sys_name, store, k=k,
        trace_mode=trace_mode,
        session_split=session_split,
        verbose=False,
        prune_ratio=prune_ratio,
        perqa_rows=perqa_rows,
    )
    return sample.sample_id, res, (perqa_rows or [])


def run_eval(
    dataset_path: str,
    systems: list[str],
    n_samples: int = 10,
    k: int = 5,
    session_split: bool = True,
    trace_mode: str = "gt",
    verbose: bool = True,
    llm_client=None,
    model_name: str = "all-MiniLM-L6-v2",
    checkpoint_path: Optional[str] = None,
    prune_ratio: float = 1.0,
    dump_perqa: bool = False,
    workers: int = 1,
    commit_calibration=None,
    judge=None,  # TYPE-EMERGENT TAXONOMY (Item 3): FaithfulnessJudge — forces the
                 # serial path (judge closures aren't picklable across worker
                 # processes, same reason llm_client excludes parallel mode).
    consolidate_max_clusters: int = 4,
) -> dict:
    from load_dataset import load_locomo_dataset

    samples = load_locomo_dataset(dataset_path)
    if n_samples > 0:
        samples = samples[:n_samples]

    results: dict[str, dict] = {}

    for sys_name in systems:
        print(f"\n{'─'*60}")
        print(f"System: {sys_name}")
        print(f"{'─'*60}")

        sample_results = []
        perqa_all: list = []
        # Parallel mode: samples are independent (fresh store each), so this
        # parallelizes almost linearly. Unavailable when a real (unpicklable)
        # llm_client feeds TRACE-FB, or for gen_agents which owns one.
        can_parallel = (workers > 1 and llm_client is None and judge is None
                        and sys_name != "gen_agents")

        if can_parallel:
            n_workers = min(workers, len(samples))
            print(f"  running {len(samples)} samples across {n_workers} "
                  f"worker processes (spawn)")
            ctx = mp.get_context("spawn")
            ordered: dict = {}
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {
                    pool.submit(_eval_one_sample_worker, sys_name, sample, k,
                               trace_mode, session_split, prune_ratio,
                               model_name, dump_perqa, commit_calibration): i
                    for i, sample in enumerate(samples)
                }
                done = 0
                for fut in as_completed(futures):
                    i = futures[fut]
                    sample_id, res, perqa_rows = fut.result()
                    ordered[i] = res
                    perqa_all.extend(perqa_rows)
                    done += 1
                    if verbose:
                        print(f"    [{done}/{len(samples)}] sample {sample_id} done "
                              f"rec5={res.get('rec5', 0.0):.4f} n={res['n']}")
            sample_results = [ordered[i] for i in range(len(samples))]
        else:
            perqa_rows: Optional[list] = [] if dump_perqa else None
            for s_idx, sample in enumerate(samples):
                if verbose:
                    print(f"  [{s_idx+1}/{len(samples)}] sample {sample.sample_id}")

                store = make_system(sys_name, llm_client=llm_client, model_name=model_name,
                                   commit_calibration=commit_calibration)
                res = evaluate_sample(
                    sample, sys_name, store, k=k,
                    trace_mode=trace_mode,
                    session_split=session_split,
                    verbose=verbose and s_idx == 0,  # verbose only first sample
                    prune_ratio=prune_ratio,
                    perqa_rows=perqa_rows,
                    judge=judge,
                    consolidate_max_clusters=consolidate_max_clusters,
                )
                sample_results.append(res)
                if verbose:
                    print(f"    → rec5={res.get('rec5', 0.0):.4f} "
                          f"rec15={res.get('rec15', 0.0):.4f} "
                          f"ctx_recall={res['ctx_recall']:.4f} n={res['n']}")
            perqa_all = perqa_rows or []

        # Aggregate across samples
        all_results_flat: list[dict] = []
        for sr in sample_results:
            # Reconstruct flat list from aggregated (approximate re-weighting by n)
            all_results_flat.append(sr)

        agg = _aggregate_samples(sample_results)
        results[sys_name] = agg

        print(f"\n  {sys_name} OVERALL: "
              f"rec5={agg.get('rec5', 0.0):.4f} rec15={agg.get('rec15', 0.0):.4f} "
              f"full_ev5={agg.get('full_ev5', 0.0):.4f} "
              f"ctx_recall={agg['ctx_recall']:.4f} "
              f"tokens5={agg.get('tokens5', 0.0):.0f} "
              f"search={agg.get('search_ms', 0.0):.1f}ms "
              f"n={agg['n']}")
        for cat, m in agg["by_category"].items():
            print(f"    cat={cat} ({_CATEGORY_NAMES.get(cat,'?')}): "
                  f"rec5={m.get('rec5', 0.0):.4f} rec15={m.get('rec15', 0.0):.4f} "
                  f"full_ev5={m.get('full_ev5', 0.0):.4f} "
                  f"ctx={m['ctx_recall']:.4f} n={m['n']}")

        # Checkpoint after every system — a crash in a later (slow) system
        # must not lose hours of completed results.
        if checkpoint_path:
            _write_results(checkpoint_path, results)

        if dump_perqa and perqa_all:
            perqa_dir = Path(checkpoint_path or "results/locomo.json").parent / "perqa"
            perqa_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(checkpoint_path or "results/locomo.json").stem
            with open(perqa_dir / f"{stem}_{sys_name}.jsonl", "w") as f:
                for row in perqa_all:
                    f.write(json.dumps(row) + "\n")

    return results


def _write_results(path: str, results: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _aggregate_samples(sample_results: list[dict]) -> dict:
    """Weighted average across samples (weighted by n QA pairs)."""
    if not sample_results:
        return {"n": 0, "ctx_recall": 0.0, "best_tf1": 0.0, "by_category": {}}

    total_n = sum(r["n"] for r in sample_results)
    if total_n == 0:
        return {"n": 0, "ctx_recall": 0.0, "best_tf1": 0.0, "by_category": {}}

    _KEYS = ("ctx_recall", "best_tf1", "rec5", "rec15",
             "full_ev5", "full_ev15", "tokens5", "search_ms")

    def _wavg(dicts: list[dict], key: str) -> float:
        pairs = [(d.get(key), d["n"]) for d in dicts if d.get(key) is not None]
        n = sum(w for _, w in pairs)
        return float(sum(v * w for v, w in pairs) / n) if n else 0.0

    # Per-category weighted average
    by_cat: dict[int, list[dict]] = defaultdict(list)
    for r in sample_results:
        for cat, m in r.get("by_category", {}).items():
            by_cat[int(cat)].append(m)

    cat_metrics = {}
    for cat, items in sorted(by_cat.items()):
        n_cat = sum(m["n"] for m in items)
        cat_metrics[cat] = {
            "n": n_cat,
            **{key: _wavg(items, key) for key in _KEYS},
            "name": _CATEGORY_NAMES.get(cat, "unknown"),
        }

    return {
        "n": total_n,
        **{key: _wavg(sample_results, key) for key in _KEYS},
        "by_category": cat_metrics,
    }


def print_table(results: dict) -> None:
    """Print a markdown-style comparison table."""
    systems = list(results.keys())
    if not systems:
        return

    print("\n\n## LoCoMo Results (context_recall@5 / best_token_f1)\n")
    header = f"{'System':<22} {'ctx_recall':>12} {'best_tf1':>10} {'n_qa':>6}"
    print(header)
    print("─" * len(header))
    for sys_name in systems:
        r = results[sys_name]
        print(f"{sys_name:<22} {r['ctx_recall']:>12.4f} {r['best_tf1']:>10.4f} {r['n']:>6}")

    # Per-category table for memoir systems
    print("\n## By Category (memoir_trace)\n")
    if "memoir_trace" in results:
        r = results["memoir_trace"]
        for cat, m in r.get("by_category", {}).items():
            print(f"  cat {cat} ({m['name']:<25}): "
                  f"ctx_recall={m['ctx_recall']:.4f} n={m['n']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo eval for MEMOIR")
    parser.add_argument("--dataset",
                        default=str(_LOCAL_DATA / "locomo10.json"))
    parser.add_argument("--systems", default=",".join(_ALL_SYSTEMS),
                        help="Comma-separated list of system names")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--k", type=int, default=5, help="Retrieval k")
    parser.add_argument("--no-session-split", action="store_true",
                        help="Disable multi-session protocol (eval A-MEM style)")
    parser.add_argument("--trace-mode", choices=["gt", "none"], default="gt",
                        help="'gt' uses GT answer for TRACE-FB; 'none' disables feedback")
    parser.add_argument("--output", default="results/locomo.json")
    parser.add_argument("--prune-ratio", type=float, default=1.0,
                        help="PRUNE-EFP: keep-ratio applied after each session "
                             "(1.0 = pruning off)")
    parser.add_argument("--dump-perqa", action="store_true",
                        help="Write per-QA rows to results/perqa/<stem>_<system>.jsonl "
                             "for paired significance testing")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallelize across LoCoMo's independent samples "
                             "using N spawn-context worker processes. Each "
                             "worker loads its own store/CFAttributor, so on "
                             "GPU-backed systems (memoir_cf_*) VRAM use scales "
                             "with N (~1GB fp16 per worker for Qwen2.5-0.5B). "
                             "Falls back to sequential for gen_agents or when "
                             "a real llm_client is supplied.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2",
                        help="Embedding model name or local path (e.g. models/memoir-encoder-v1)")
    parser.add_argument("--commit-calibration", default=None,
                        help="Path to a JSON artifact from calibrate_commit.py. "
                             "Used by memoir_commit/memoir_v3 for the calibrated "
                             "LLR commit gate instead of the fixed COMMIT_DELTA/"
                             "N_COMMIT fallback. Omit to use the fallback.")
    parser.add_argument("--consolidate", action="store_true",
                        help="TYPE-EMERGENT TAXONOMY (Item 3): run store.consolidate() "
                             "after each session's feedback pass, using a real LLM "
                             "judge (RouterClient/gpt-4o-mini via tokenrouter.com, "
                             "needs TOKEN_ROUTER_API_KEY) — required for memoir_stratum/ "
                             "memoir_ucb_stratum/memoir_ucb_all's semantic-stratum "
                             "tagging to have any effect. Forces sequential (non-"
                             "parallel) execution. Off by default — all other systems "
                             "are unaffected either way.")
    parser.add_argument("--consolidate-budget", type=float, default=2.0,
                        help="USD budget cap for the consolidation judge's LLM calls "
                             "(RouterClient safety brake, gpt-4o-mini pricing).")
    parser.add_argument("--consolidate-certify-threshold", type=float, default=1.0,
                        help="FaithfulnessJudge certify_threshold (default 1.0 = "
                             "production unanimous 3-vote gate). Loosen only for a "
                             "diagnostic run isolating whether stratum-decay itself "
                             "helps once semantic-tagged items exist — measured "
                             "gpt-4o-mini LOSSLESS rate on LoCoMo is ~1%%/vote, so the "
                             "unanimous gate certifies ~0 merges regardless of sample "
                             "count (docs/PAPER_COMPONENTS.md #12). Do not use a "
                             "loosened value for any reported production number.")
    parser.add_argument("--consolidate-max-clusters", type=int, default=4,
                        help="Max merge clusters proposed per session in consolidate().")
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    commit_calibration = None
    if args.commit_calibration:
        from harpo.memoir.consolidate import CommitCalibration
        raw = json.loads(Path(args.commit_calibration).read_text())
        cal = raw.get("calibration", raw)  # accept either the fitting script's
                                            # wrapper JSON or a bare artifact
        if cal is None:
            raise SystemExit(f"{args.commit_calibration} has no calibration "
                             "(the fit was not certifiable — rerun calibrate_commit.py "
                             "with more data or a looser alpha)")
        commit_calibration = CommitCalibration.from_dict(cal)
        print(f"Loaded commit calibration: llr_threshold="
              f"{commit_calibration.llr_threshold:.3f} from {args.commit_calibration}")

    judge = None
    if args.consolidate:
        from harpo.memoir.consolidate import FaithfulnessJudge
        from router_client import RouterClient
        client = RouterClient(budget_usd=args.consolidate_budget)
        judge = FaithfulnessJudge(
            generate_fn=lambda prompt: client.chat(prompt, max_tokens=64)[0],
            certify_threshold=args.consolidate_certify_threshold,
        )
        print(f"Consolidation ON: FaithfulnessJudge(gpt-4o-mini via RouterClient, "
              f"budget=${args.consolidate_budget:.2f}, "
              f"certify_threshold={args.consolidate_certify_threshold})")

    results = run_eval(
        dataset_path=args.dataset,
        systems=systems,
        n_samples=args.n_samples,
        k=args.k,
        session_split=not args.no_session_split,
        trace_mode=args.trace_mode,
        commit_calibration=commit_calibration,
        verbose=not args.quiet,
        model_name=args.model_name,
        checkpoint_path=args.output,
        prune_ratio=args.prune_ratio,
        dump_perqa=args.dump_perqa,
        workers=args.workers,
        judge=judge,
        consolidate_max_clusters=args.consolidate_max_clusters,
    )

    print_table(results)

    # Serialize (int keys → string for JSON)
    def _jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(x) for x in obj]
        return obj

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_jsonify(results), f, indent=2)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
