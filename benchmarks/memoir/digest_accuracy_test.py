"""
FAST diagnostic (not production): does adding certified session digests
(src/harpo/memoir/consolidate.py's new DIGEST operator) to the already-winning
`cosine` baseline improve QA accuracy enough to beat Mem0/A-MEM? cosine alone
already ties A-MEM (38.27%) and beats Mem0 (35.75%) on this protocol
(docs/COMBINED_PAPER_PLAN.md, 2026-07-20/21) -- TRACE-FB's reranking was found
to hurt accuracy even under oracle feedback, so this tests a different lever:
richer (LiCoMemory-style synthesized) context on top of cosine's retrieval,
not a different ranking formula.
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

from harpo.memoir.store import MemoirStore
from harpo.memoir.consolidate import FaithfulnessJudge
from eval_locomo import make_system
from eval_locomo_sota import make_client, evidence_recall_at_k, format_context, aggregate
from gemini_client import approx_token_count
from load_dataset import load_locomo_dataset


def run_cosine_digest(sample, k, client, judge, model_name, max_digest_checks=6,
                      perqa_rows=None):
    store = make_system("cosine", model_name=model_name)
    sessions = sample.conversation.sessions
    sample_id = getattr(sample, "sample_id", None)
    results = []
    for sess_id in sorted(sessions.keys()):
        sess = sessions[sess_id]
        for turn in sess.turns:
            text = f"Speaker {turn.speaker} says: {turn.text}"
            store.add(text, metadata={"dia_id": turn.dia_id, "session": sess_id})

        # certified digest ONLY (no merge/prune/supersede) at each session
        # boundary -- write-time-occasional, not per-query, preserving the
        # zero-per-query-LLM-call retrieval property.
        stats = store.consolidate(judge=judge, max_digest_checks=max_digest_checks)
        print(f"    session {sess_id}: digests={stats.get('applied_digests')} "
              f"judge_calls_total={judge.call_count}", flush=True)

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
            row = {"category": qa.category, "rec": rec,
                   "acc": 1.0 if is_correct else 0.0,
                   "t_r": t_r, "k_r": k_r}
            results.append(row)
            if perqa_rows is not None:
                # pairing key matches significance.py's load_rows():
                # (sample_id, question, category)
                perqa_rows.append({"sample_id": sample_id, "question": qa.question,
                                   "session": sess_id, "count": True, **row})
    return results


def main():
    samples = load_locomo_dataset(str(_REPO / "data" / "locomo10.json"))[:1]
    client = make_client("ollama")
    from ollama_client import OllamaClient
    judge_client = OllamaClient()
    judge = FaithfulnessJudge(
        generate_fn=lambda prompt: judge_client._call(prompt, max_tokens=200)[0],
        n_votes=1,  # speed: single-framing judge (deadline-driven scope cut)
    )

    all_results = []
    perqa_rows: list = []
    for i, sample in enumerate(samples):
        r = run_cosine_digest(sample, k=5, client=client, judge=judge,
                              model_name="all-MiniLM-L6-v2", max_digest_checks=2,
                              perqa_rows=perqa_rows)
        all_results.extend(r)
        print(f"  [cosine+digest] sample {i+1}/{len(samples)} done, n={len(r)}, "
              f"judge_calls={judge.call_count}", flush=True)

    out = {"cosine_digest": aggregate(all_results)}
    print("cosine_digest:", out["cosine_digest"], flush=True)

    with open("results/digest_accuracy_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved to results/digest_accuracy_test.json")

    perqa_dir = Path("results/perqa")
    perqa_dir.mkdir(parents=True, exist_ok=True)
    with open(perqa_dir / "digest_accuracy_test_cosine_digest.jsonl", "w") as f:
        for row in perqa_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(perqa_rows)} per-QA rows to "
          f"{perqa_dir / 'digest_accuracy_test_cosine_digest.jsonl'}")


if __name__ == "__main__":
    main()
