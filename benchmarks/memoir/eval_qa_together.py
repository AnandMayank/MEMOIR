"""
End-to-end QA accuracy on LoCoMo via the Together API — the SOTA-comparison
eval (vs Mem0 / LiCoMemory reported numbers, arXiv:2511.01448 Table 1).

Protocol per QA (mirrors the retrieval harness in eval_locomo.py):
  1. Session-split ingestion + TRACE feedback exactly as eval_locomo.
  2. At each subsampled QA: retrieve top-k → generate an answer with
     meta-llama/Llama-3.3-70B-Instruct-Turbo from the retrieved context.
  3. LLM-judge (same model): is the prediction correct given the gold answer?
  Metrics: judge accuracy, evidence rec@5, answer latency, prompt tokens, $.

Cost controls (current recharge: $15; $1.04/M tokens in AND out):
  - every request disk-cached in eval_cache/together/ (reruns are free)
  - running spend printed; hard abort at --budget-usd

Usage (TOGETHER_API_KEY must be set):
  # Ladder 1: pipeline sanity (~$0.08)
  python benchmarks/memoir/eval_qa_together.py --systems memoir_no_trace \
      --n-qa 100 --budget-usd 0.5 --output results/together_sanity.json
  # Ladder 2: subsample comparison (~$1.2)
  python benchmarks/memoir/eval_qa_together.py \
      --systems cosine,memoir_no_trace,memoir_v3 \
      --n-qa 500 --budget-usd 3.0 --output results/together_subsample.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).parent.parent.parent
_LOCAL_DATA = _REPO / "data"
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).parent))  # vendored load_dataset.py

from eval_locomo import (  # noqa: E402
    _parse_locomo_datetime,
    evidence_recall,
    make_system,
    system_add,
    system_feedback,
    system_retrieve_items,
)

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
PRICE_PER_TOKEN = 1.04 / 1_000_000  # input and output, USD
CACHE_DIR = _REPO / "eval_cache" / "together"

_CATEGORY_NAMES = {1: "single_hop", 2: "temporal", 3: "multi_hop",
                   4: "open_ended", 5: "adversarial"}


class BudgetExceeded(RuntimeError):
    pass


def _budget_errors() -> tuple:
    """Both clients' BudgetExceeded types (router imported lazily — it's only
    needed when --backend router is used and requires TOKEN_ROUTER_API_KEY)."""
    try:
        from router_client import BudgetExceeded as RouterBudgetExceeded
        return (BudgetExceeded, RouterBudgetExceeded)
    except ImportError:
        return (BudgetExceeded,)


_BUDGET_ERRORS = _budget_errors()


class TogetherClient:
    """Cached, budget-capped wrapper around the Together SDK."""

    def __init__(self, budget_usd: float, model: str = MODEL) -> None:
        from together import Together
        # Explicit base_url: the SDK's default silently routes chat
        # completions through api-inference.together.ai (a separate
        # dedicated-inference host, X-Together-Routing header confirms it),
        # which 404s "Model not found" for serverless models on this
        # account/region even though api.together.xyz/v1 serves them fine.
        self._client = Together(base_url="https://api.together.xyz/v1")  # reads TOGETHER_API_KEY
        self._model = model
        self._budget = budget_usd
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

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        latency = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        n_tok = ((usage.prompt_tokens if usage else len(prompt) // 4)
                 + (usage.completion_tokens if usage else max_tokens))
        self.spent_usd += n_tok * PRICE_PER_TOKEN
        self.calls += 1
        cache_file.write_text(json.dumps({"text": text, "tokens": n_tok}))
        return text, latency


def answer_prompt(question: str, memories: list[str]) -> str:
    # Validated via benchmarks/memoir/prompt_ablation_test.py (n=80, live 70B):
    # the prior strict wording ("use only... say No information available")
    # over-triggered abstention on raw dialogue-turn context -- 35% no-info
    # rate, all wrong by construction. Permitting reasonable inference across
    # turns recovered +5pp accuracy and cut the no-info rate to 21%.
    ctx = "\n".join(f"- {m}" for m in memories) or "(none)"
    return (
        "You are answering a question about past conversations using the "
        "retrieved memory snippets below. The snippets are raw dialogue turns "
        "-- the answer may require connecting information across turns or "
        "making a reasonable inference from what was said, not just exact "
        "wording. Answer concisely (a short phrase) with the best-supported "
        "answer you can infer from the snippets. Only say \"No information "
        "available\" if the snippets give no relevant basis for any answer "
        "at all.\n\n"
        f"Memories:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )


def judge_prompt(question: str, gold: str, predicted: str) -> str:
    return (
        "You are grading a QA system. Decide if the predicted answer is "
        "semantically correct given the gold answer. Minor wording, formatting "
        "or date-format differences are acceptable; missing or contradictory "
        "content is not.\n\n"
        f"Question: {question}\nGold answer: {gold}\nPredicted answer: {predicted}\n\n"
        "Reply with exactly one word: yes or no."
    )


def stratified_subsample(samples, n_qa: int, seed: int = 13):
    """Deterministic per-category proportional subsample across all samples.
    Returns {sample_id: set(qa_indices)}."""
    all_refs = []  # (category, sample_idx, qa_idx)
    for si, sample in enumerate(samples):
        for qi, qa in enumerate(sample.qa):
            all_refs.append((qa.category, si, qi))
    by_cat = defaultdict(list)
    for ref in all_refs:
        by_cat[ref[0]].append(ref)
    total = len(all_refs)
    rng = random.Random(seed)
    chosen = []
    for cat, refs in sorted(by_cat.items()):
        n_cat = max(1, round(n_qa * len(refs) / total))
        rng.shuffle(refs)
        chosen.extend(refs[:n_cat])
    picked = defaultdict(set)
    for _, si, qi in chosen:
        picked[si].add(qi)
    return picked


def run_system(sys_name: str, samples, picked, client: TogetherClient,
               k: int = 5, verbose: bool = True) -> dict:
    rows = []
    for si, sample in enumerate(samples):
        wanted = picked.get(si, set())
        if not wanted:
            continue
        store = make_system(sys_name)
        qa_index = {id(qa): i for i, qa in enumerate(sample.qa)}
        sessions = sample.conversation.sessions
        for sess_id in sorted(sessions.keys()):
            sess = sessions[sess_id]
            sess_epoch = _parse_locomo_datetime(sess.date_time)
            for turn in sess.turns:
                system_add(sys_name, store,
                           f"Speaker {turn.speaker} says: {turn.text}",
                           {"session": sess_id, "dia_id": turn.dia_id},
                           created_at=sess_epoch)
            # Same online protocol as eval_locomo: QAs evidenced in this session
            sess_qa = [qa for qa in sample.qa
                       if any(ev.startswith(f"D{sess_id}:")
                              for ev in (qa.evidence or []))]
            for qa in sess_qa:
                gt = str(qa.final_answer)
                items = system_retrieve_items(sys_name, store, qa.question,
                                              k=k, now=sess_epoch)
                texts = [m.text for m in items]
                # retrieval-side TRACE feedback (GT protocol, as in eval_locomo)
                rec5 = evidence_recall(items, qa.evidence)
                system_feedback(sys_name, store, qa.question, gt,
                                oracle_correct=rec5 > 0.5, now=sess_epoch)

                if qa_index[id(qa)] not in wanted:
                    continue
                pred, gen_lat = client.chat(answer_prompt(qa.question, texts))
                verdict, _ = client.chat(
                    judge_prompt(qa.question, gt, pred.strip()), max_tokens=4)
                correct = verdict.strip().lower().startswith("yes")
                rows.append({
                    "sample_id": sample.sample_id,
                    "question": qa.question,
                    "category": qa.category,
                    "gold": gt,
                    "predicted": pred.strip(),
                    "correct": correct,
                    "rec5": rec5,
                    "gen_latency_s": gen_lat,
                    "prompt_tokens_est": sum(len(t.split()) for t in texts),
                })
        if verbose:
            done = sum(1 for r in rows)
            print(f"  [{sys_name}] sample {si+1}/{len(samples)} — {done} QAs, "
                  f"spent ${client.spent_usd:.3f} ({client.cache_hits} cache hits)")

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    lat = [r["gen_latency_s"] for r in rows if r["gen_latency_s"] > 0]
    return {
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / max(len(rows), 1),
        "rec5": sum(r["rec5"] for r in rows) / max(len(rows), 1),
        "avg_gen_latency_s": (sum(lat) / len(lat)) if lat else 0.0,
        "avg_prompt_tokens": (sum(r["prompt_tokens_est"] for r in rows)
                              / max(len(rows), 1)),
        "by_category": {
            _CATEGORY_NAMES.get(c, str(c)): {
                "n": len(rs),
                "accuracy": sum(r["correct"] for r in rs) / len(rs),
                "rec5": sum(r["rec5"] for r in rs) / len(rs),
            } for c, rs in sorted(by_cat.items())
        },
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="LoCoMo QA accuracy via Together")
    ap.add_argument("--data", default=str(_LOCAL_DATA / "locomo10.json"))
    ap.add_argument("--systems", default="cosine,memoir_no_trace,memoir_v3")
    ap.add_argument("--n-qa", type=int, default=500,
                    help="stratified subsample size (identical across systems)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--budget-usd", type=float, required=True,
                    help="hard spend cap for this run")
    ap.add_argument("--backend", default="together",
                    choices=["together", "router"],
                    help="'together' = Together SDK (Llama-70B-Turbo etc., "
                         "TOGETHER_API_KEY); 'router' = tokenrouter.com "
                         "OpenAI-compatible aggregator (gpt-4o-mini etc., "
                         "TOKEN_ROUTER_API_KEY) — same disk-cache + budget "
                         "discipline either way")
    ap.add_argument("--model", default=None,
                    help="model id override for the chosen backend "
                         "(defaults: Together=Llama-3.3-70B-Instruct-Turbo, "
                         "router=gpt-4o-mini)")
    ap.add_argument("--output", default="results/together_qa.json")
    args = ap.parse_args()

    from load_dataset import load_locomo_dataset
    samples = load_locomo_dataset(args.data)
    picked = stratified_subsample(samples, args.n_qa, seed=args.seed)
    n_picked = sum(len(v) for v in picked.values())
    print(f"Subsample: {n_picked} QAs (requested {args.n_qa}, seed {args.seed})")

    if args.backend == "router":
        from router_client import RouterClient
        client = RouterClient(budget_usd=args.budget_usd,
                              **({"model": args.model} if args.model else {}))
        model_id = client._model
    else:
        if not os.getenv("TOGETHER_API_KEY"):
            raise SystemExit("TOGETHER_API_KEY is not set")
        client = TogetherClient(budget_usd=args.budget_usd,
                                **({"model": args.model} if args.model else {}))
        model_id = args.model or MODEL
    results: dict = {"model": model_id, "backend": args.backend,
                     "n_qa": n_picked, "seed": args.seed,
                     "systems": {}}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for sys_name in [s.strip() for s in args.systems.split(",") if s.strip()]:
        print(f"\n── {sys_name} ──")
        try:
            results["systems"][sys_name] = run_system(
                sys_name, samples, picked, client, k=args.k)
        except _BUDGET_ERRORS as e:
            print(f"STOPPED: {e}")
            results["budget_exceeded"] = str(e)
            break
        finally:
            results["spent_usd"] = round(client.spent_usd, 4)
            results["api_calls"] = client.calls
            results["cache_hits"] = client.cache_hits
            with open(out_path, "w") as f:  # checkpoint after every system
                json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"{'System':<18} {'acc':>7} {'rec5':>7} {'gen_lat':>8} {'tok':>6} {'n':>5}")
    for name, r in results["systems"].items():
        print(f"{name:<18} {r['accuracy']:>7.4f} {r['rec5']:>7.4f} "
              f"{r['avg_gen_latency_s']:>7.2f}s {r['avg_prompt_tokens']:>6.0f} "
              f"{r['n']:>5}")
    print(f"Total spent: ${client.spent_usd:.3f}  "
          f"({client.calls} calls, {client.cache_hits} cache hits)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
