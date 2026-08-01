"""
FAST diagnostic: cosine+digest at 70B (Together) on the SAME group_1 sample
already used for the live LiCoMemory comparison (docs/COMBINED_PAPER_PLAN.md,
2026-07-20: LiCoMemory 54.3-54.8% on group_1, n=199, same backbone). The 8B
digest win (43.28%, digest_accuracy_test.py) was a different backbone size
than the LiCoMemory number -- this closes that gap for a fair comparison.
"""
from __future__ import annotations

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

from harpo.memoir.consolidate import FaithfulnessJudge
from eval_locomo import make_system
from eval_locomo_sota import evidence_recall_at_k, format_context, aggregate
from eval_qa_together import TogetherClient, judge_prompt as _judge_prompt_qa
from gemini_client import approx_token_count
from load_dataset import load_locomo_dataset

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


class Together70BAnswerClient:
    """Adapts TogetherClient's .chat() to the generate_answer/judge_correct
    interface eval_locomo_sota.py's run loop expects (OllamaClient's shape)."""

    def __init__(self, budget_usd: float = 4.0):
        self._client = TogetherClient(budget_usd=budget_usd, model=MODEL)

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


def run_cosine_digest(sample, k, client, judge, model_name, max_digest_checks=3):
    store = make_system("cosine", model_name=model_name)
    sessions = sample.conversation.sessions
    results = []
    for sess_id in sorted(sessions.keys()):
        sess = sessions[sess_id]
        for turn in sess.turns:
            text = f"Speaker {turn.speaker} says: {turn.text}"
            store.add(text, metadata={"dia_id": turn.dia_id, "session": sess_id})

        stats = store.consolidate(judge=judge, max_digest_checks=max_digest_checks)
        print(f"    session {sess_id}: digests={stats.get('applied_digests')} "
              f"judge_calls_total={judge.call_count} spent=${client._client.spent_usd:.3f}",
              flush=True)

        sess_qa = [qa for qa in sample.qa
                   if any(ev.startswith(f"D{sess_id}:") for ev in (qa.evidence or []))]
        for qa in sess_qa:
            gt = str(qa.final_answer)
            ablation = getattr(store, "_ablation", {})
            t0 = time.perf_counter()
            items = store.search(qa.question, k=k, **ablation)
            retrieved = {m.metadata.get("dia_id") for m in items if m.metadata.get("dia_id")}
            rec = evidence_recall_at_k(retrieved, qa.evidence)
            context = format_context(items)
            k_r = approx_token_count(context)
            answer, gen_latency = client.generate_answer(qa.question, context)
            is_correct, judge_latency = client.judge_correct(qa.question, answer, gt)
            t_r = time.perf_counter() - t0
            results.append({"category": qa.category, "rec": rec,
                            "acc": 1.0 if is_correct else 0.0,
                            "t_r": t_r, "k_r": k_r})
    return results


def main():
    samples = load_locomo_dataset(str(_REPO / "data" / "locomo10.json"))[:1]  # group_1
    client = Together70BAnswerClient(budget_usd=4.0)
    judge = FaithfulnessJudge(
        generate_fn=lambda prompt: client._client.chat(prompt, max_tokens=200)[0],
        n_votes=1,
    )

    all_results = []
    for i, sample in enumerate(samples):
        r = run_cosine_digest(sample, k=5, client=client, judge=judge,
                              model_name="all-MiniLM-L6-v2", max_digest_checks=3)
        all_results.extend(r)
        print(f"  [cosine+digest-70b] sample {i+1}/{len(samples)} done, n={len(r)}, "
              f"judge_calls={judge.call_count}, spent=${client._client.spent_usd:.3f}",
              flush=True)

    out = {"cosine_digest_70b": aggregate(all_results),
          "spent_usd": client._client.spent_usd}
    print("cosine_digest_70b:", out["cosine_digest_70b"], flush=True)

    with open("results/digest_accuracy_test_70b.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved to results/digest_accuracy_test_70b.json")


if __name__ == "__main__":
    main()
