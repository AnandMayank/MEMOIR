# MEMOIR — Architecture & Methodology Reference

> Written to be lifted into the paper's Methodology section. Every component
> lists its formula, motivation, lineage (what to cite), and the code flag
> that enables it. This file is the *how*; see `results/` for raw run outputs
> backing the numbers.

---

## 1. Problem setting and notation

An agent accumulates a memory store `M = {m_1, …, m_n}` of short text units
(conversation turns) across sessions. At query time it retrieves `top-k(q, M)`
to place in the answering LLM's context. After an answer `a` is produced, a
**feedback event** occurs — the only learning signal available. Constraints:

- **No ground-truth labels** at feedback time (the answer itself is the signal).
- **No training**: no gradient updates to any model; all adaptation is
  closed-form online state updates.
- **Write path budget**: `add()` must not call any LLM (0 LLM/write —
  contrast: Generative Agents spends 1 LLM call per write for an importance
  score).

Each memory carries a quality vector `w = (w_rel, w_rec, w_conf, w_imp)`
(schema.py). `w_imp` — outcome-conditioned importance — is the quantity the
TRACE family revises retrospectively.

## 2. Write path

### 2.1 TRIAGE-α (`triage.py`) — asynchronous write-time scoring
- `w_rel = cos(e_m, centroid_session)` against a rolling centroid of the last
  `session_window=20` items; `w_conf` = negation-overlap conflict signal
  against the same window. Runs as a background task; `add()` returns first.
- Embedder: frozen `all-MiniLM-L6-v2` singleton (`embedder.py`), unit-norm
  vectors (the store assumes `dot == cosine` throughout).

### 2.2 DEDUP-α (`store.py`)
Write-time near-duplicate collapse: if `cos(e_new, e_recent) ≥ 0.97` within
the session window, the write increments the existing item's
`duplicate_count` instead of inserting. Motivation: write-noise accumulation
(documented in Mem0 issue #4573).

### 2.3 Lazy indexing
BM25 index and the dense matrix are rebuilt lazily at the first `search()`
after writes (`_bm25_dirty`/`_matrix_dirty`) — keeps `add()` O(1).

## 3. Retrieval: RECALL-Q (`recall.py`)

```
score(m) = cos(q, m) · (1 + β·w_rel + δ·w_rec + γ·w_imp_eff)
         + α·BM25(q, m) − λ·MMR(m, selected) + 0.05·co_cite_bonus(m, selected)
defaults: α=0.3, λ=0.1, β=0.1, γ=1.0, δ=0.1;  k_fetch = min(3k, n)
```
- Cosine is the base signal; the w-terms are *proportional bonuses* so fresh
  memories (all-zero w) reduce exactly to the cosine baseline.
- Greedy MMR loop performs selection; the co-cite bonus (Section 5) and
  retrieval-count bump happen there.
- `w_imp_eff` is `w_imp` modified in order by: Weibull decay (optional),
  **CCOMMIT gate** (Section 6.1), **CORE-DLM discount** (Section 6.2), then
  the **UCB exploration bonus** (Section 4.4).
- Filtering: hallucinated and pruned items are excluded *by position index*,
  keeping BM25/embedding rows aligned with items.

## 4. Retrospective credit: the TRACE family (`trace_fb.py`)

All variants share the update rule
```
Δw_imp(m) = α_trace · attr(m, a) · 1/(rank(m)+1),   w_imp ← clip(w_imp+Δ, 0, 1)
α_trace = 0.3;  hallucination ⇒ hard reset w_imp = 0
```
and differ only in the attribution function `attr`:

### 4.1 TRACE-FB (ROUGE1 path) — v1 baseline
`attr = |tokens(m) ∩ tokens(a)| / |tokens(m)|` (stopword-filtered recall).
Fast (6.7ms/feedback) but **lexically blind**: a count answer "3" appearing in
no memory text yields attr = 0 structurally.

### 4.2 TRACE-CF (`cf_attributor.py`) — v2 core contribution
Counterfactual leave-one-out answer-likelihood shift under a small frozen
scorer LM (Qwen2.5-0.5B-Instruct, local GPU):
```
lp(C)   = mean token log-prob of a, teacher-forced after context C
attr(m) = 1 − exp(−max(0, lp(C) − lp(C∖{m})))          ∈ [0, 1)
```
- k+1 contexts scored in chunked batched forwards (`SCORER_BATCH_SIZE=8`,
  `MAX_PROMPT_TOKENS=1536`); log-softmax applied only at answer positions
  (memory-bounded); one retry on transient CUDA faults.
- Cost lands **only** in `feedback_answer()` (67ms P50 at k=5); the write
  path never touches the scorer.
- Fixes the lexical blind spot *causally*: removing the memory that lists
  three events lowers P("3"|ctx) even though "3" appears in no memory.
- Lineage to cite: counterfactual/LOO credit (C3, arXiv:2603.06859 — they
  re-rollout multi-agent messages; we use likelihood shift, no regeneration).

### 4.3 Oracle TRACE (upper bound)
`attr = 1` if the answer was correct else 0 (needs GT; eval only).

### 4.4 UCB exploration bonus (retrieval side, `use_ucb=True`)
```
w_imp_eff = w_imp + κ·sqrt(ln N / (n_m + 1)),  κ=0.05
```
`n_m` = per-item retrieval count, `N` = store-wide total. Counters
rich-get-richer lock-in (early-credited memories keep being retrieved and
keep earning credit; never-retrieved memories can never receive feedback).
Adapted from RetroAgent's SimUtil-UCB Eq. 9 (arXiv:2603.08561) — cite.

## 5. Co-citation graph (`store.py`)

Sparse undirected edges over memories that earn credit **together**:
```
edge(A,B) += min(attr_A, attr_B) · 0.05    if both attr ≥ 0.15
bonus(m)  = Σ_{s ∈ selected, edge(s,m)} edge(s,m)·w_imp(s)   (Loop 2 only)
```
Models joint utility that per-item scalars cannot express (the independence
assumption in SimUtil-UCB-style buffers). Flag `use_co_cite=True`.

## 6. v3 components — credit reliability and store management

Adapted from surgical-video memory management; applicable because our
attribution plays the role of their mask-confidence scores.

### 6.1 CCOMMIT — credible credit commitment (`use_commit=True`)
CIFS analog (ReSurgSAM2, arXiv:2505.08581 Eq. 1: no tracking commitment until
N_w consecutive confident detections). Here:
- `commit_count(m)` += 1 whenever `attr(m) ≥ COMMIT_DELTA=0.15` (the co-cite
  noise floor).
- Retrieval: `w_imp_eff = w_imp · 1[commit_count ≥ N_COMMIT=2]`. Credit still
  *accumulates* while uncommitted — late-committing memories surface with
  full history.
- Co-cite edges require both endpoints committed (stops false-edge seeding).
Targets: one spurious attribution permanently distorting retrieval (the
mechanism behind the measured temporal-category damage).

### 6.2 CORE-DLM — diversity-driven core memory bank (`use_core_bank=True`)
DLM analog (ReSurgSAM2 Eqs. 2–3: candidate pool of confident frames; promote
the most diverse into a bounded long-term bank; initial frame pinned).
- Pool P: items with `attr ≥ 0.15` at any feedback event.
- On `|P| ≥ 8`: promote `argmin_{p∈P} cos(e_p, e_last_promoted)` into core
  bank L (FIFO, capacity 64); first promotion = highest-attribution memory,
  which stays pinned. Pool clears after each promotion.
- Retrieval: core members get the full γ·w_imp bonus; non-core get 0.25×.
Targets: structural rich-get-richer — credit concentrating on redundant or
single-topic clusters (UCB addresses this per-item; DLM bounds it globally
with a diversity constraint).

### 6.3 PRUNE-EFP — credit-aware retrieval pruning (`store.prune(ratio)`)
EFP analog (SurgSAM2 "SAM2S", arXiv:2408.07931: score memory-bank frames by
cosine informativeness, keep the informative subset → 3× speedup).
```
keep_score = w_imp + 0.3·(1 − max_cos_to_kept) + 0.1·w_rec
```
Greedy keep-selection (incremental max-cos, one matvec per kept item, like
MMR); non-kept items get `pruned=True` — excluded from retrieval,
**reversible** (`unprune_all()`), nothing deleted. Applied after each
session's feedback in eval (steady-state store management).
Targets: search latency and prompt-token cost scale with n; the efficiency
claim is the recall-vs-latency/tokens curve at keep ratios {1.0, 0.75, 0.5}.

## 7. GATE-LinUCB — learned per-query retrieval weights (`gating.py`)

Answers the "hand-calibrated constants" critique: the RECALL-Q hyperparameters
(α, β, γ, λ) become a per-query *choice* among five qualitative profiles
(baseline / lexical / semantic / importance-trusting / diversity-seeking),
selected by LinUCB (Li et al., 2010) on a fixed random 384→32 projection of
the query embedding:
```
select: argmax_a θ_a·x + 0.5·sqrt(x^T A_a^{-1} x),   θ_a = A_a^{-1} b_a
update: Sherman–Morrison on A_a^{-1};  b_a += reward·x
reward = max attribution over the retrieved set (no extra labels)
```
Closed-form, gradient-free — bandit learning is the RL-lite answer to
GRPO-scale training. `select()` is deterministic given gate state, so
`feedback_answer()`'s internal re-search re-selects the arm it rewards.
Lineage: CHARM (HARPO, ACL) context-dependent quality weighting, made online.
Flag `use_learned_weights=True`; fixed `PROJECTION_SEED=13`.

## 8. Evaluation methodology

### 8.1 LoCoMo (primary; 10 conversations, 2433 QA pairs, 5 categories)
**Online session-split protocol** (`eval_locomo.py`): sessions ingested in
order with dataset-native timestamps; after each session, its evidenced QAs
are answered and TRACE feedback fires (GT answer as feedback text — the
"LLM answer ≈ GT" convention; oracle systems use the binary signal). Session 1
QAs are excluded from metrics (no prior feedback possible). Feedback scan
k=50 (ROUGE1) / k=16 (CF — the LOO prompt must contain every scanned memory;
protocol difference disclosed).

### 8.2 Metrics (v7 — evidence-based)
LoCoMo annotates every QA with evidence `dia_id`s; every stored memory keeps
its turn's `dia_id`:
- **rec@5 / rec@15**: any retrieved memory is an annotated evidence turn.
  Exact and non-lexical (valid for numeric answers).
- **full-ev@5/@15**: ALL evidence turns retrieved — the honest multi-hop
  metric (every hop must be in context).
- **tokens5**: whitespace tokens of the retrieved top-5 (prompt cost/query).
- **search_ms**: wall-clock retrieval latency (k=5 call).
- Legacy **ctx@5** (any GT-token overlap) kept for continuity; it both
  over-counts (one shared token = hit) and under-counts (numeric answers) —
  shown by cosine: ctx@5 0.679 vs rec@5 0.476.
- The TRACE oracle signal is evidence-based (`rec5 > 0.5`).

### 8.3 LongMemEval-S (cross-benchmark; ceiling finding)
Two-retrieval protocol (`eval_longmemeval.py`): build store → TRACE → search
again; Δ measured against a fresh no-feedback baseline. With the evidence
metric, ev_recall is 0.95–1.00 *before* feedback at the standard slices
(k=5, n=20/type) — retrieval is saturated, so Δ=0 is structurally inevitable
for ANY attribution signal (oracle and CF included). Reported as
"base system works (+5pp over cosine); TRACE does no harm at ceiling."

### 8.4 Correlation gate (`correlation.py`)
Before headline runs, the attribution signal must correlate with retrieval
success: Pearson r(max attribution over retrieved, evidence recall) across
1,986 QAs. CF r=0.4931 > ROUGE1 r=0.4472. Numeric-answer subset is n=23 on
LoCoMo — too small to claim; the numeric case is covered by the LME-S
analysis and the behavioral unit test instead.

### 8.5 Significance (`significance.py`)
Paired bootstrap over QA pairs (pairing key: sample_id + question + category;
10k resamples, two-sided p, 95% CI, per-category breakdown). Produced from
`--dump-perqa` JSONL rows. This addresses "n=10 conversations" directly: the
paired unit is the QA (n≈2,433).

### 8.6 QA accuracy vs SOTA (`eval_qa_together.py`)
End-to-end accuracy with **meta-llama/Llama-3.3-70B-Instruct-Turbo** via
Together ($1.04/M tokens): answer from retrieved top-5, LLM-judge vs gold
(yes/no). Stratified fixed-seed subsample identical across systems; every
request disk-cached; hard `--budget-usd` abort. Comparison rows: Mem0 and
LiCoMemory *reported* LoCoMo numbers (arXiv:2511.01448 Table 1; their
published setting used Llama-3.1-70B-Instruct-Turbo — same family/scale,
difference disclosed).

### 8.7 Latency (`latency_bench.py`)
Write path (add, sync/async), feedback path (ROUGE1 vs CF), P50/P95/P99 over
200 trials, RTX 4060 laptop. Current: write 6.8ms P50 (unchanged by v2/v3);
feedback 6.7ms (ROUGE1) / 67ms (CF, k=5).

## 9. Comparison systems

- `cosine` — flat cosine (ablation flags).
- `simutil_ucb` (`baseline_simutil.py`) — **RetroAgent's memory mechanism
  transplanted** (arXiv:2603.08561 Eqs. 7–10): srel threshold 0.4, score =
  0.5·cos + 0.5·(u + 1.0·sqrt(ln N/n_i)), uniform EMA utility
  (β=0.1, u₀=0.5) with the same success score for every retrieved item.
  RetroAgent itself is an online-RL framework evaluated on ALFWorld/WebShop/
  Sokoban/MineSweeper — its published numbers do not transfer to LoCoMo; the
  transplant isolates the scientific difference (uniform episode credit vs
  per-memory counterfactual credit) under identical store/embedder/data.
- MEMOIR ladder: `memoir_no_trace` → `memoir_trace` (ROUGE1) →
  `memoir_cf_trace` → `+ucb` / `+gate` → `memoir_cf_full` →
  `memoir_commit` / `memoir_dlm` → `memoir_v3` (cf_full + commit + core-bank).

## 10. Positioning (cite + differentiate)

| System | Credit granularity | Training | Setting |
|---|---|---|---|
| RetroAgent (2603.08561) | uniform per episode | GRPO (4×H200) | agentic RL envs |
| C3 (2603.06859) | per message, LOO re-rollout | RL | multi-agent collab |
| Memory-R2 / AgeMem / HiMPO | memory ops / states | RL | agent memory |
| Mem0 / LiCoMemory (2511.01448) | none (write-time importance) | none | conversational |
| **MEMOIR v3** | **per memory, counterfactual LOO likelihood-shift** | **none** | conversational |

## 11. Honest limits (must appear in the paper)

- CF overall ctx-gain over ROUGE1-trace is small; the claims are calibration
  (r=0.493 vs 0.447), the structural numeric fix, per-category damage
  reduction, and training-free-vs-RL positioning.
- CF feedback is 10× ROUGE1 cost (67ms) and caps the feedback scan at k=16
  on LoCoMo (protocol difference).
- LME-S is a ceiling/no-harm result, not headroom.
- v3 introduces new constants (N_COMMIT, N_POOL, N_CORE, prune λs) —
  sensitivity ablations or folding into gate arms are the mitigations.
- Accuracy comparison uses our judge implementation; Mem0/LiCoMemory rows are
  reported numbers from their infrastructure.
- Statistical unit: 10 conversations / 2,433 QAs; significance is at the QA
  level (paired bootstrap); conversation-level (store-clustered / leave-one-
  store-out) resampling is not yet implemented in `significance.py`.
