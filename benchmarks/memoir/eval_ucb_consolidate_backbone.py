"""
MEMOIR's best-so-far method (memoir_ucb retrieval + certified consolidation,
merge+supersede) at real backbone scale, matching the LiCoMemory-comparison
protocol (group_1 = sample 0, n=199 QAs, k=5) already used for the llama3.1:8b
row in docs/EXPERIMENT_TABLES.md T1.

Scope decision (2026-07-22): user asked to compare across backbones (8B done,
70B/gpt-4o-mini pending) but to test only the single best combined method
(consolidate + trace_ucb) rather than the full system ladder -- narrows this
run to one system, keeping API cost/time bounded.

Usage:
  TOGETHER_API_KEY=... python benchmarks/memoir/eval_ucb_consolidate_backbone.py \\
      --backend together --budget-usd 4.0 \\
      --output results/ucb_consolidate_70b.json
  TOKEN_ROUTER_API_KEY=... python benchmarks/memoir/eval_ucb_consolidate_backbone.py \\
      --backend router --budget-usd 4.0 \\
      --output results/ucb_consolidate_gpt4omini.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from harpo.memoir.store import MemoirStore
from harpo.memoir.consolidate import FaithfulnessJudge
from eval_locomo import _parse_locomo_datetime
from eval_locomo_sota import evidence_recall_at_k, format_context, aggregate
from eval_qa_together import TogetherClient, judge_prompt as _judge_prompt_qa
from gemini_client import approx_token_count
from load_dataset import load_locomo_dataset

TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


class BackboneAnswerClient:
    """Adapts TogetherClient/RouterClient's .chat() to the
    generate_answer/judge_correct interface (OllamaClient's shape),
    matching digest_accuracy_test_70b.py's Together70BAnswerClient pattern
    but backend-agnostic (together or router)."""

    def __init__(self, backend: str, budget_usd: float, model: str | None = None):
        if backend == "router":
            from router_client import RouterClient
            self._client = RouterClient(budget_usd=budget_usd,
                                        **({"model": model} if model else {}))
        else:
            self._client = TogetherClient(budget_usd=budget_usd,
                                          model=model or TOGETHER_MODEL)

    @property
    def spent_usd(self) -> float:
        return self._client.spent_usd

    def generate_answer(self, question: str, context: str) -> tuple[str, float]:
        prompt = (
            "Based on the context below, write a short phrase answering the question. "
            "Use exact words from the context wherever possible.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nShort answer:"
        )
        text, lat = self._client.chat(prompt, max_tokens=60)
        return text, lat

    def judge_correct(self, question: str, generated: str, gold: str) -> tuple[bool, float]:
        text, lat = self._client.chat(
            _judge_prompt_qa(question, gold, generated.strip()), max_tokens=4)
        return text.strip().lower().startswith("yes"), lat


def run_ucb_consolidate(sample, k, client, judge, model_name) -> list[dict]:
    """memoir_ucb (TRACE-FB + UCB exploration bonus) retrieval, with certified
    consolidation (merge+supersede, digest off) run once per session boundary
    -- the RQ1 (UCB) + RQ5 (certified consolidation) combination, MEMOIR's
    best validated method so far per docs/COMBINED_PAPER_PLAN.md TABLE 3."""
    store = MemoirStore(model_name=model_name, alpha_trace=0.3,
                        use_co_cite=False, use_ucb=True)
    sessions = sample.conversation.sessions
    results = []
    for sess_id in sorted(sessions.keys()):
        sess = sessions[sess_id]
        # Dataset-native session timestamp -- REQUIRED. Without created_at/now,
        # every item's age collapses to ~0 (fast wall-clock add loop), making
        # w_rec / Weibull decay degenerate no-ops. Confirmed empirically this
        # run: omitting this alone dropped rec@5 on this exact sample from
        # 0.597 to 0.382 (n=238) -- the same failure mode already fixed once
        # for eval_longmemeval.py (docs/COMBINED_PAPER_PLAN.md 2026-07-21) and
        # documented in MemoryItem.live_w_rec's docstring.
        sess_epoch = _parse_locomo_datetime(sess.date_time)
        for turn in sess.turns:
            text = f"Speaker {turn.speaker} says: {turn.text}"
            store.add(text, metadata={"dia_id": turn.dia_id, "session": sess_id},
                     created_at=sess_epoch)

        # Certified consolidation, merge-only "safety-first" config (matches
        # EXPERIMENT_TABLES.md T3's validated certified row, 44/0/0
        # merges/prunes/supersedes). max_prune_checks MUST be 0 here --
        # consolidate()'s own default (32) would turn on the dominated-prune
        # operator, independently audited at 83.9% lossy (T3/A3) and never
        # validated as part of "the best method so far". Supersede stays off
        # too (default 0) for the same reason -- neither has cleared the bar
        # this UCB+merge combination has.
        stats = store.consolidate(judge=judge, max_prune_checks=0)
        print(f"    session {sess_id}: merges={stats.get('applied_merges')} "
              f"prunes={stats.get('applied_prunes')} "
              f"judge_calls_total={judge.call_count} spent=${client.spent_usd:.3f}",
              flush=True)

        sess_qa = [qa for qa in sample.qa
                   if any(ev.startswith(f"D{sess_id}:") for ev in (qa.evidence or []))]
        for qa in sess_qa:
            gt = str(qa.final_answer)
            t0 = time.perf_counter()
            items = store.search(qa.question, k=k, use_ucb=True, now=sess_epoch)
            # Merged items carry metadata["dia_ids"] (plural, list -- set by
            # consolidate.py's transfer_credit) instead of a single "dia_id",
            # since merge collapses several raw turns (each with their own
            # dia_id) into one item and prunes the constituents. Missing this
            # silently undercounts evidence recall for any QA whose evidence
            # got merged -- must union both singular and plural forms.
            retrieved: set[str] = set()
            for m in items:
                did = m.metadata.get("dia_id")
                if did:
                    retrieved.add(did)
                for did in m.metadata.get("dia_ids", []) or []:
                    retrieved.add(did)
            rec = evidence_recall_at_k(retrieved, qa.evidence)
            context = format_context(items)
            k_r = approx_token_count(context)
            answer, gen_latency = client.generate_answer(qa.question, context)
            is_correct, judge_latency = client.judge_correct(qa.question, answer, gt)
            # retrieval-side TRACE feedback (GT protocol, matches eval_locomo/T1 rows)
            store.feedback_answer(qa.question, gt, k=50, use_ucb=True,
                                  oracle_correct=None, now=sess_epoch)
            t_r = time.perf_counter() - t0
            results.append({"category": qa.category, "rec": rec,
                            "acc": 1.0 if is_correct else 0.0,
                            "t_r": t_r, "k_r": k_r,
                            # kept for debugging surprising acc/rec numbers
                            # without needing to re-spend API budget
                            "question": qa.question, "gold": gt,
                            "predicted": answer.strip(),
                            "n_merged_in_context": sum(
                                1 for m in items if m.metadata.get("kind") == "merge"
                                or "merged_from" in m.metadata)})
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MEMOIR ucb+consolidate at 70B/gpt-4o-mini scale, group_1 protocol")
    ap.add_argument("--backend", default="together", choices=["together", "router"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--budget-usd", type=float, default=4.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--output", default="results/ucb_consolidate_backbone.json")
    args = ap.parse_args()

    samples = load_locomo_dataset(str(_REPO / "data" / "locomo10.json"))[:1]  # group_1
    client = BackboneAnswerClient(args.backend, args.budget_usd, args.model)
    judge = FaithfulnessJudge(
        generate_fn=lambda prompt: client._client.chat(prompt, max_tokens=200)[0],
        n_votes=1,
    )

    all_results: list[dict] = []
    for i, sample in enumerate(samples):
        r = run_ucb_consolidate(sample, k=args.k, client=client, judge=judge,
                                model_name="all-MiniLM-L6-v2")
        all_results.extend(r)
        print(f"  [ucb+consolidate/{args.backend}] sample {i+1}/{len(samples)} done, "
              f"n={len(r)}, judge_calls={judge.call_count}, spent=${client.spent_usd:.3f}",
              flush=True)

    out = {"memoir_ucb_consolidate": aggregate(all_results),
          "backend": args.backend, "spent_usd": client.spent_usd}
    print(f"memoir_ucb_consolidate ({args.backend}):", out["memoir_ucb_consolidate"],
         flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")

    perqa_path = out_path.parent / f"{out_path.stem}_perqa.jsonl"
    with open(perqa_path, "w") as f:
        for row in all_results:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(all_results)} per-QA rows to {perqa_path}")


if __name__ == "__main__":
    main()
