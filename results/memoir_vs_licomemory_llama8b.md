# MEMOIR vs LiCoMemory — Same-Model Comparison (llama3.1:8b)

**Status: single-run (n=1 trial) results, NOT yet averaged over repeated runs.**
LiCoMemory's own paper reports "all results averaged over 5 runs, mean ± standard deviation."
The numbers below have no repeated trials and no variance estimate — they are a first-pass,
directional comparison only. 5-run averaging for `group_1` is in progress (see bottom).

---

## Setup

- **Backbone LLM**: `llama3.1:8b` via local Ollama, for both generation and LLM-judge scoring,
  used identically by both systems.
- **Data**: same `/home/user/A-MEM/data/locomo10.json`, samples 0 and 1 (`group_1`=199 QAs,
  `group_2`=105 QAs → 304 unique questions total).
- **Why llama3.1:8b and not their tested backbones**: LiCoMemory's paper tests
  Llama-3.1-70B-Instruct-Turbo and GPT-4o-mini. The 70B model needs ~35-40GB VRAM even
  quantized; this machine has an 8GB GPU. GPT-4o-mini is proprietary (no open weights, no
  free tier). llama3.1:8b is the honest, free, fully-local substitute — NOT a reproduction
  of their tested conditions.
- **Embedding model**: LiCoMemory was configured to use MEMOIR's own fine-tuned encoder
  (`models/memoir-encoder-v1`), not their tested BGE-M3. BGE-M3 is not cached locally, and
  an attempted download corrupted a previously-complete cache of a substitute model
  (`paraphrase-MiniLM-L6-v2`) mid-run in this environment — network access for large model
  downloads is unreliable here. This is a real confound: LiCoMemory's entity/relationship
  merge thresholds (0.85, etc.) were calibrated for BGE-M3's embedding space, not this one.

## Two verified bugs in LiCoMemory's public repo (not introduced by this setup)

1. **Rec. metric always reads 0.0 — FIXED locally.** In `query/query_processor.py`,
   `top_session_ids` was computed internally inside `_calculate_all_summary_scores()` (used
   only to filter `relevant_summaries`) but never attached as a key to the dict
   `_process_unified_query()` returns. `main.py` reads `res.get('top_session_ids', [])`,
   always falling back to `[]`. Confirmed via `grep -n "'top_session_ids'" query_processor.py`
   → zero matches for the dict-key literal before the patch. Their `SessionMatchingEvaluator`
   therefore always computed matching_score=0.0 regardless of true retrieval quality —
   verified across all 304 questions in the first single-run pass (every `matching_scores`
   entry was 0.0).
   **Patch applied** (`query_processor.py`, in `_process_unified_query`, right before the
   `result = {...}` dict is built): derive `top_session_ids` from `relevant_summaries`, which
   already carries `session_id` per entry, and attach it to `result`. No entity/triple/
   retrieval logic was touched — this only exposes information the pipeline already computes.
   The original (unpatched) single-run `group_1`/`group_2` results have been **discarded** and
   are being **re-run from scratch with the fix** (5x for `group_1`, see below), so all
   reported numbers come from one consistent code version.
2. **K_R (token consumption) undercounts — not yet fixed.** `Retrieval Tokens: 0` logged for
   every one of the 304 queries in the original run (`grep "Retrieval Tokens:" ... | sort -u`
   → single value `0`). Their own paper defines K_R as "the number of input tokens in the
   retrieval prompt." What's actually logged is only the answer-generation call's token count.
   Any K_R figure for LiCoMemory should still be treated as an undercount until this is also
   patched.

## Results

**`group_1` numbers below are STALE / DISCARDED** — they came from the pre-fix code (Rec.
always 0.0) and have been superseded by the 5-run patched re-run in progress (see bottom of
this file for live status). Kept here only as a reference point for run-to-run comparison
once the patched results land.

**`group_2` has NOT been re-run** — its numbers below are still from the original unpatched
single run (Rec. genuinely unobtainable for this sample, bug not yet fixed for `group_2`
specifically; only `group_1` is being redone right now).

| System | T_R | Acc. | Rec.@15 | K_R |
|---|---|---|---|---|
| **LiCoMemory** combined (pre-fix, stale) | 14.51s | 41.11% (125/304) | *unobtainable (bug)* | ~1,400 (undercount, bug) |
| MEMOIR cosine | 1.96s | 50.84% | 71.64% | 567 |
| MEMOIR memoir_no_trace | 0.98s | 49.16% | 72.46% | 573 |
| MEMOIR memoir_trace | 0.97s | 48.60% | 68.80% | 559 |
| MEMOIR memoir_full | 0.97s | 48.04% | 72.46% | 571 |

*n=304 for LiCoMemory (one evaluation per unique question). n=358 for MEMOIR — same 304
underlying questions, but MEMOIR's session-split protocol evaluates a question once per
session its evidence spans, so questions with multi-session evidence are counted more than
once. Different counting convention, same source data.*

### LiCoMemory per-sample breakdown (pre-fix single run — group_1 portion now stale)
- `group_1` (199 QAs, **stale, being replaced**): 81/199 correct = 40.70%. Per-category:
  cat1=43.75% (14/32), cat2=16.22% (6/37), cat3=69.23% (9/13), cat4=62.86% (44/70),
  cat5=17.02% (8/47).
- `group_2` (105 QAs, **current/valid, not yet re-run**): 44/105 correct = 41.90%.
  Per-category: cat1=54.55% (6/11), cat2=50.00% (13/26), cat4=47.73% (21/44), cat5=16.67%
  (4/24). (No category 3 questions in this sample.) Rec. still unobtainable for this sample
  (bug not yet fixed here).
- T_R: `group_1` avg 16.85s/query (stale), `group_2` avg 10.07s/query (current).
  Measured as wall-clock gaps between consecutive "Question N/Total" log timestamps —
  includes entity extraction, retrieval, reranking, generation, and LLM-judge eval.
- K_R (undercounted, see bug #2 above): `group_1` avg 1,447 tokens/query (stale), `group_2`
  avg 1,312 tokens/query (current).

### Root cause hypothesis for the gap (not isolated/proven)
LiCoMemory's CogniGraph depends on LLM-quality entity/relation extraction and session
summarization at write time — structured multi-step prompts tuned for a 70B-class or
GPT-4o-mini backbone. An 8B backbone likely degrades that extraction significantly (same
failure mode documented in Mem0 GitHub issue #4573: weaker models follow structured
extraction prompts less reliably). MEMOIR has no equivalent write-time LLM dependency, so a
weak backbone only affects the final answer-generation call, not the memory substrate itself.
This is a plausible mechanism, not a confirmed/isolated cause — the embedding-model
substitution (see confound above) is a second plausible contributor that has not been
separated from the backbone-size effect.

### What this result is and is not
- **Is**: a same-protocol, same-backbone, same-data sanity check with a real, checkable
  mechanism story.
- **Is not**: evidence that MEMOIR beats LiCoMemory's actual published SOTA numbers — those
  used far stronger backbones (Llama-3.1-70B-Instruct-Turbo, GPT-4o-mini) than what's
  available in this environment.
- **Is not yet**: statistically rigorous. Single trial, no variance, 2 of 10 LoCoMo samples.

---

## 5-run averaging (in progress — restarted after Rec. bug fix)

Running `group_1` (199 QAs) 5 times from scratch, all under the patched code (Rec.@15 now
wired correctly), to get mean ± std matching LiCoMemory's own reported protocol. The earlier
pre-fix single run (40.70%, 16.85s/query, Rec. unobtainable) was discarded so all 5 trials
use identical code — no run mixes patched/unpatched behavior. Each run takes ~80 minutes, so
5 runs are expected to take ~6-7 hours total. Root dirs: `llama8b_group1_run1` through
`llama8b_group1_run5`. Results will be appended here as each run completes.

| Run | Acc. | T_R (avg) | Rec.@15 | Notes |
|---|---|---|---|---|
| 1 | *pending* | | | patched code, fresh run |
| 2 | *pending* | | | |
| 3 | *pending* | | | |
| 4 | *pending* | | | |
| 5 | *pending* | | | |
| **Mean ± std** | *pending* | | | |
