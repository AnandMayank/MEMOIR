"""
TRACE-CF: counterfactual per-memory attribution via leave-one-out
answer-likelihood shift.

Replaces lexical ROUGE1 attribution with a *behavioral* signal: how much does
removing memory m from the retrieved context drop the probability of the
observed answer under a small frozen scorer LM?

    attribution(m) = 1 - exp(-max(0, lp_full - lp_without_m))

where lp_* is the length-normalized log-likelihood (nats/token) of the answer
teacher-forced after the context. The 1-exp(-x) transform is parameter-free
and maps the shift in [0, inf) to [0, 1).

Why this fixes the documented ROUGE1 blind spot: for a count answer like "3",
no memory text contains "3", so ROUGE1 attribution is structurally 0 forever.
But removing the memory that lists the three events *does* drop P("3" | ctx) —
the signal is causal-behavioral, not lexical.

Cost model: k+1 teacher-forced forward passes per feedback_answer() call,
batched into a single padded forward. No generation, no training, no API.
Write path (add/add_many) is untouched — still 0 LLM calls at write time.

Scorer default: Qwen/Qwen2.5-0.5B-Instruct (~1GB, fp16 on GPU). Any causal LM
works; models without a chat template fall back to a plain completion format.
Singleton per model name, mirroring embedder.Embedder.
"""

from __future__ import annotations

import math
import threading
from typing import Optional

# Hard cap on scorer context length. Retrieved conversational memories are
# short (1-3 sentences); this guard only matters for pathological inputs.
MAX_PROMPT_TOKENS: int = 1536
MAX_ANSWER_TOKENS: int = 256

# Forward-pass mini-batch. The model's output logits are (batch, len, vocab);
# with a 151k vocab even 16 sequences of 512 tokens is ~2.5GB fp16, so large
# k (feedback scans use k=50) must be chunked rather than one padded forward.
SCORER_BATCH_SIZE: int = 8


class CFAttributor:
    _instances: dict[str, "CFAttributor"] = {}
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        ).to(self._device)
        self._model.eval()
        self._has_chat_template = bool(getattr(self._tokenizer, "chat_template", None))

    @classmethod
    def get(cls, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> "CFAttributor":
        if model_name not in cls._instances:
            with cls._lock:
                if model_name not in cls._instances:
                    cls._instances[model_name] = cls(model_name)
        return cls._instances[model_name]

    # ── Prompt construction ────────────────────────────────────────────────────

    def _build_prompt(self, question: str, memories: list[str]) -> str:
        if memories:
            ctx = "\n".join(f"- {m}" for m in memories)
        else:
            ctx = "(no memories retrieved)"
        user = (
            "Answer the question using the retrieved memories.\n"
            f"Memories:\n{ctx}\n"
            f"Question: {question}"
        )
        if self._has_chat_template:
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return user + "\nAnswer: "

    # ── Batched teacher-forced answer log-likelihood ───────────────────────────

    def _answer_logprobs(self, prompts: list[str], answer: str) -> list[float]:
        """
        Length-normalized log P(answer | prompt) in nats/token, one value per
        prompt. Padded forwards run in mini-batches of SCORER_BATCH_SIZE, and
        log_softmax is applied only at the answer positions to keep peak
        memory bounded by (batch, ans_len, vocab) rather than the full length.
        """
        torch = self._torch
        tok = self._tokenizer

        answer_ids = tok(answer, add_special_tokens=False,
                         truncation=True, max_length=MAX_ANSWER_TOKENS).input_ids
        if not answer_ids:
            return [0.0] * len(prompts)
        n_ans = len(answer_ids)

        seqs = []
        for p in prompts:
            p_ids = tok(p, add_special_tokens=False,
                        truncation=True, max_length=MAX_PROMPT_TOKENS).input_ids
            seqs.append(p_ids + answer_ids)

        pad_id = tok.pad_token_id
        out: list[float] = []
        for start in range(0, len(seqs), SCORER_BATCH_SIZE):
            batch_seqs = seqs[start: start + SCORER_BATCH_SIZE]
            max_len = max(len(s) for s in batch_seqs)

            # Left-pad: every sequence's answer tokens then end at the exact
            # same final `n_ans` positions regardless of prompt length, so
            # `logits_to_keep=n_ans` lets HF skip materializing lm_head
            # logits for anything but the answer span. Without this, the
            # model computes logits over the FULL (batch, seq_len, vocab)
            # tensor — for Qwen2.5's ~152k vocab at seq_len~1792, batch=8,
            # fp32, that's ~9GB for one forward pass, the actual OOM cause
            # (not per-worker model weights, which are only ~1GB fp16).
            input_ids = torch.full((len(batch_seqs), max_len), pad_id,
                                   dtype=torch.long)
            attn_mask = torch.zeros((len(batch_seqs), max_len), dtype=torch.long)
            for i, s in enumerate(batch_seqs):
                input_ids[i, max_len - len(s):] = torch.tensor(s, dtype=torch.long)
                attn_mask[i, max_len - len(s):] = 1
            input_ids = input_ids.to(self._device)
            attn_mask = attn_mask.to(self._device)

            def _score_batch() -> list[float]:
                # +1: the window must include the position whose logit
                # predicts the FIRST answer token, i.e. the last prompt
                # position. logits[:, j, :] then predicts full-sequence
                # position (max_len - n_ans - 1 + j) + 1 = max_len - n_ans + j,
                # so for j in [0, n_ans) that's exactly the n_ans answer
                # tokens at input_ids[:, -n_ans:]; the extra kept position
                # (j = n_ans) predicts one token past the sequence — unused.
                with torch.no_grad():
                    logits = self._model(input_ids=input_ids,
                                         attention_mask=attn_mask,
                                         logits_to_keep=n_ans + 1).logits
                ans_logits = logits[:, :n_ans, :].float()      # (batch, n_ans, vocab)
                lps = torch.log_softmax(ans_logits, dim=-1)
                targets = input_ids[:, -n_ans:]                 # (batch, n_ans)
                token_lps = torch.gather(
                    lps, 2, targets.unsqueeze(-1)).squeeze(-1)  # (batch, n_ans)
                vals = token_lps.mean(dim=-1).tolist()
                del logits
                return vals

            try:
                out.extend(_score_batch())
            except RuntimeError as e:
                # Transient CUDA faults (observed on WSL2 during multi-hour
                # runs) recover after a cache flush — retry once, then raise.
                if "CUDA" not in str(e):
                    raise
                # If the CUDA context itself is dead (WSL2 driver reset),
                # even empty_cache raises — surface the original error then;
                # recovery requires a fresh process.
                if self._device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except RuntimeError:
                        raise e
                out.extend(_score_batch())
        return out

    # ── Public API ─────────────────────────────────────────────────────────────

    def score_loo(
        self,
        question: str,
        answer: str,
        memories: list[str],
    ) -> list[float]:
        """
        Leave-one-out attribution for each memory, in input order.
        Returns values in [0, 1). One batched forward over k+1 contexts.
        """
        return self.score_loo_with_confidence(question, answer, memories)[0]

    def score_loo_with_confidence(
        self,
        question: str,
        answer: str,
        memories: list[str],
    ) -> tuple[list[float], float]:
        """
        Like score_loo, but also returns lp_full — the length-normalized
        log-likelihood (nats/token) of the answer under the FULL retrieved
        context. Already computed as the LOO reference point; previously
        discarded. exp(lp_full) is the geometric-mean per-token answer
        probability — a free, training-free answer-confidence proxy used by
        the verification gate (trace_fb.py) to stop crediting memories for
        answers the scorer itself finds implausible (the self-reinforcing-
        error failure mode). Returns (attributions, lp_full); lp_full is
        -inf-safe: 0.0 when there is nothing to score.
        """
        if not memories or not answer.strip():
            return [0.0] * len(memories), 0.0

        prompts = [self._build_prompt(question, memories)]
        for i in range(len(memories)):
            loo = memories[:i] + memories[i + 1:]
            prompts.append(self._build_prompt(question, loo))

        lps = self._answer_logprobs(prompts, answer)
        lp_full = lps[0]
        attrs = [
            1.0 - math.exp(-max(0.0, lp_full - lp_loo))
            for lp_loo in lps[1:]
        ]
        return attrs, lp_full
