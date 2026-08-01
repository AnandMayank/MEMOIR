"""
Lightweight metric suite for LoCoMo / LongMemEval evaluation.

Drop-in replacement for A-MEM's utils.calculate_metrics / aggregate_metrics
that does not depend on the openai package.

Available metrics:
  exact_match   — lowercased exact string match
  f1            — token-level F1 (SQuAD-style)
  rouge1_f      — ROUGE-1 F-measure
  rouge2_f      — ROUGE-2 F-measure
  rougeL_f      — ROUGE-L F-measure
  bleu1..bleu4  — BLEU-1..4 with add-1 smoothing
  meteor        — METEOR (requires nltk wordnet + punkt)
  sbert_sim     — sentence-BERT cosine similarity (all-MiniLM-L6-v2)

bert_f1 is omitted (requires bert-score which is not installed).
"""

from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

# ── lazy imports (heavy models only loaded if actually called) ────────────────
_rouge_scorer = None
_sbert_model  = None
_nltk_ready   = False


def _get_rouge():
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer as rs
        _rouge_scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _rouge_scorer


def _get_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def _ensure_nltk() -> None:
    global _nltk_ready
    if not _nltk_ready:
        import nltk
        for pkg in ("punkt", "punkt_tab", "wordnet"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        _nltk_ready = True


# ── tokenization ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ── individual metric functions ───────────────────────────────────────────────

def exact_match(pred: str, ref: str) -> float:
    return float(_normalize(pred) == _normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    """Token-level F1 between prediction and reference (SQuAD-style)."""
    pred_toks = _tokenize(pred)
    ref_toks  = _tokenize(ref)
    if not pred_toks or not ref_toks:
        return float(pred_toks == ref_toks)
    pred_cnt = defaultdict(int)
    for t in pred_toks:
        pred_cnt[t] += 1
    ref_cnt = defaultdict(int)
    for t in ref_toks:
        ref_cnt[t] += 1
    common = sum(min(pred_cnt[t], ref_cnt[t]) for t in ref_cnt)
    if common == 0:
        return 0.0
    prec = common / len(pred_toks)
    rec  = common / len(ref_toks)
    return 2 * prec * rec / (prec + rec)


def rouge_scores(pred: str, ref: str) -> dict[str, float]:
    scorer = _get_rouge()
    result = scorer.score(ref, pred)
    return {
        "rouge1_f": result["rouge1"].fmeasure,
        "rouge2_f": result["rouge2"].fmeasure,
        "rougeL_f": result["rougeL"].fmeasure,
    }


def bleu_scores(pred: str, ref: str) -> dict[str, float]:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    hyp   = _tokenize(pred)
    refs  = [_tokenize(ref)]
    sf    = SmoothingFunction().method1
    if not hyp:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    return {
        "bleu1": sentence_bleu(refs, hyp, weights=(1,0,0,0), smoothing_function=sf),
        "bleu2": sentence_bleu(refs, hyp, weights=(0.5,0.5,0,0), smoothing_function=sf),
        "bleu3": sentence_bleu(refs, hyp, weights=(1/3,1/3,1/3,0), smoothing_function=sf),
        "bleu4": sentence_bleu(refs, hyp, weights=(0.25,0.25,0.25,0.25), smoothing_function=sf),
    }


def meteor(pred: str, ref: str) -> float:
    _ensure_nltk()
    from nltk.translate.meteor_score import meteor_score as ms
    hyp  = _tokenize(pred)
    refs = [_tokenize(ref)]
    if not hyp or not any(refs):
        return 0.0
    try:
        return float(ms(refs, hyp))
    except Exception:
        return 0.0


def sbert_similarity(pred: str, ref: str) -> float:
    model = _get_sbert()
    vecs  = model.encode([pred, ref], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


# ── combined calculate_metrics (A-MEM interface) ──────────────────────────────

def calculate_metrics(
    pred: str,
    ref: str,
    compute_sbert: bool = True,
    compute_meteor: bool = True,
    compute_bleu: bool = True,
) -> dict[str, float]:
    """
    Compute all metrics for a single (prediction, reference) pair.
    Compatible with A-MEM's calculate_metrics return format.
    """
    if not ref or str(ref).strip() == "":
        return {
            "exact_match": 0.0, "f1": 0.0,
            "rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0,
            "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
            "meteor": 0.0, "sbert_similarity": 0.0, "bert_f1": 0.0,
        }

    pred_str = str(pred) if pred is not None else ""
    ref_str  = str(ref)

    out: dict[str, float] = {
        "exact_match": exact_match(pred_str, ref_str),
        "f1":          token_f1(pred_str, ref_str),
        "bert_f1":     0.0,  # not computed (bert-score not installed)
    }

    out.update(rouge_scores(pred_str, ref_str))

    if compute_bleu:
        out.update(bleu_scores(pred_str, ref_str))
    else:
        out.update({"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0})

    out["meteor"] = meteor(pred_str, ref_str) if compute_meteor else 0.0

    out["sbert_similarity"] = (
        sbert_similarity(pred_str, ref_str) if compute_sbert else 0.0
    )

    return out


def aggregate_metrics(metrics_list: list[dict]) -> dict[str, float]:
    """
    Aggregate a list of per-sample metric dicts into mean values.
    Compatible with A-MEM's aggregate_metrics return format.
    """
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list if k in m])) for k in keys}
