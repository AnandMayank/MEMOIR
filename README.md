# MEMOIR: A Training-Free Memory Lifecycle for Long-Horizon LLM Agents

Anonymous code release accompanying an anonymous paper submission. Distribution or
public sharing beyond the review process is not intended.

MEMOIR (**Mem**ory **E**vidence via Merge-certified, **O**nline **I**mportance
**R**evision) unifies **judge-gated prospective consolidation** with **retrospective,
outcome-conditioned attribution** for agent memory. It targets two coupled failure
modes of static-importance memory systems: consolidation that silently destroys
evidence future queries need, and retrieval that keeps favoring stale memories once
written — training-free, with no LLM call on the online write path.

## Core mechanisms

- **RECALL-Q** (`src/harpo/memoir/recall.py`) — retrieval scoring combining dense
  similarity, BM25, recency, relevance, and importance, with MMR diversity and
  co-citation bonuses.
- **TRACE** — training-free retrieval attribution and credit estimation, with two
  instantiations: **TRACE-FB** (feedback attribution via answer-to-memory unigram
  recall, `src/harpo/memoir/trace_fb.py`) and **TRACE-CF** (counterfactual leave-one-out
  attribution under a frozen local scorer LM, `src/harpo/memoir/cf_attributor.py`).
- **TRACE-UCB** — a count-based exploration bonus that prevents feedback-driven
  retrieval lock-in on under-exposed memories (`src/harpo/memoir/gating.py`,
  `recall.py`).
- **Judge-gated MERGE** — a certificate-gated consolidation operator (three
  differently framed judge prompts testing factual preservation, future-question
  answerability, and adversarial omissions; retrieval-coverage check; hard length
  ratio) and an optional **SUPERSEDE** operator for reversible temporal/contradiction
  updates (`src/harpo/memoir/consolidate.py`).
- **TRIAGE-α** — lightweight write-time relevance/negation scoring that flags
  conflict signals for SUPERSEDE candidates (`src/harpo/memoir/triage.py`).

## Headline results

On 1,905 deduplicated LoCoMo QA pairs (full formulas and evaluation protocol in
`ARCHITECTURE.md`):

- The full RECALL-Q retrieval stack improves evidence recall@5 by **11.92 percentage
  points** over bare cosine retrieval.
- Confidence-gated, reference-free feedback improves evidence recall@5 by **1.50
  points** over the matched no-TRACE retriever with no reference answers or reranker
  training; a deployment-unavailable oracle-feedback upper bound adds **1.84 points**
  overall and **8.05 points** on multi-hop questions.
- Certified merge changes fixed-budget, text-grounded evidence recall from 58.82% to
  59.08%, reduces the active store by 1.42%, and incurs **0 observed losses in 100
  independently audited merges** (one-sided 95% upper bound: 2.95%).
- The online write path requires **no LLM call** and completes in 6.8ms median
  latency; merge screening is confined to periodic offline consolidation.

## Repository layout

```
src/harpo/memoir/       Core library: retrieval, attribution, consolidation
benchmarks/memoir/      Evaluation harnesses, baselines, plotting scripts
tests/unit/             Unit tests for the mechanisms this paper describes
data/                   Vendored LoCoMo benchmark data
results/                Raw run outputs backing the paper's tables (curated subset —
                         see note below)
ARCHITECTURE.md         Full methodology: formulas, evaluation protocol, honest limits
```

`results/` here is a curated subset of a much larger internal experiment log, kept to
the runs that directly back this paper's reported numbers (RQ1 cross-backbone
comparison, RQ2's oracle-feedback retrieval ladder, RQ3 latency and attribution
correlation, and the SUPERSEDE mechanism). It does not include exploratory ablations,
sweeps, or intermediate iterations that aren't cited in the paper.

## Installation and quick start

```bash
pip install -e ".[memoir,memoir-eval]"

python -c "
from harpo.memoir.store import MemoirStore
s = MemoirStore()
s.add('The user prefers Python for data pipelines')
s.add('Python is slower than C++ but easier to write')
answer = 'Python'
s.feedback_answer('preferred language', answer)
results = s.search('What language does the user prefer?', k=3)
print(results[0].text, results[0].w.w_imp)
assert results[0].w.w_imp > 0
"
```

## Running the evaluations

```bash
python benchmarks/memoir/eval_locomo.py \
  --data data/locomo10.json \
  --systems cosine,memoir_no_trace,memoir_trace \
  --output results/locomo_run.json

python benchmarks/memoir/eval_consolidation.py --help
```

## Tests

```bash
pytest tests/unit/ -v -m "not slow"          # fast suite
pytest tests/unit/test_cf_attributor.py -m slow -v   # downloads a local scorer LM
```

See `ARCHITECTURE.md` for the full retrieval/attribution/consolidation formulas,
evaluation methodology, comparison systems, and honest limits.

## License

MIT — see `LICENSE`.
