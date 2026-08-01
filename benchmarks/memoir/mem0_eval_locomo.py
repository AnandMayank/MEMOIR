"""
Mem0 baseline eval on the same LiCoMemory-matched protocol used for MEMOIR
and LiCoMemory: same 2 LoCoMo samples, same llama3.1:8b backbone (via Ollama),
same generate+judge approach (reuses ollama_client.py), same evidence-based
Rec@15 metric (matches memory dia_id metadata against QA evidence).

Mem0's add() runs its own LLM-based extraction ("infer=True" — the 2-phase
extraction+update architecture the LiCoMemory paper cites). This means each
add() call is itself an LLM call via Ollama, distinct from MEMOIR's
zero-write-LLM design and from how this eval measures MEMOIR/LiCoMemory.

Usage:
  python benchmarks/memoir/mem0_eval_locomo.py \\
    --dataset data/locomo10.json \\
    --n-samples 2 --k 15 \\
    --output results/mem0_llama8b_2samples.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).parent.parent.parent
_LOCAL_DATA = _REPO / "data"
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # vendored load_dataset.py

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ollama_client import OllamaClient
from gemini_client import approx_token_count

_FINETUNED_ENCODER = _REPO / "models" / "memoir-encoder-v1"
# Fall back to the same base encoder MEMOIR itself uses when the fine-tuned
# checkpoint hasn't been synced to this machine (it's built/lives on the
# vast.ai GPU box via finetune_encoder.py + deploy/rsync_to_vast.sh).
LOCAL_ENCODER = (
    str(_FINETUNED_ENCODER)
    if _FINETUNED_ENCODER.is_dir()
    else "sentence-transformers/all-MiniLM-L6-v2"
)

MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:8b",
            "ollama_base_url": "http://localhost:11434",
            "temperature": 0.0,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": LOCAL_ENCODER,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0_locomo_eval",
            "embedding_model_dims": 384,
            "path": "/tmp/mem0_qdrant_locomo",
            "on_disk": False,
        },
    },
}


def evidence_recall_at_k(retrieved_dia_ids: set[str], evidence: list[str]) -> float:
    ev = [e for e in (evidence or []) if e]
    if not ev:
        return 0.0
    hit = sum(1 for e in ev if e in retrieved_dia_ids)
    return hit / len(ev)


def format_context(memories: list[dict]) -> str:
    parts = []
    for i, m in enumerate(memories, 1):
        parts.append(f"{i}. {m.get('memory', '')}")
    return "\n".join(parts)


def run_sample(mem0_client, sample, k: int, client: OllamaClient, sample_id: str) -> list[dict]:
    sessions = sample.conversation.sessions
    sorted_sess_ids = sorted(sessions.keys())
    results: list[dict] = []

    for sess_id in sorted_sess_ids:
        sess = sessions[sess_id]
        for turn in sess.turns:
            text = f"Speaker {turn.speaker} says: {turn.text}"
            mem0_client.add(
                text,
                user_id=sample_id,
                metadata={"dia_id": turn.dia_id, "session": sess_id},
                infer=True,
            )

        sess_qa = [
            qa for qa in sample.qa
            if any(ev.startswith(f"D{sess_id}:") for ev in (qa.evidence or []))
        ]
        for qa in sess_qa:
            gt = str(qa.final_answer)

            t0 = time.perf_counter()
            search_result = mem0_client.search(
                qa.question, filters={"user_id": sample_id}, top_k=k
            )
            memories = search_result.get("results", search_result) if isinstance(search_result, dict) else search_result
            retrieved_dia_ids = {
                m.get("metadata", {}).get("dia_id")
                for m in memories if m.get("metadata", {}).get("dia_id")
            }
            rec = evidence_recall_at_k(retrieved_dia_ids, qa.evidence)

            context = format_context(memories)
            k_r = approx_token_count(context)
            answer, gen_latency = client.generate_answer(qa.question, context)
            is_correct, judge_latency = client.judge_correct(qa.question, answer, gt)
            t_r = time.perf_counter() - t0

            results.append({
                "category": qa.category,
                "rec": rec,
                "acc": 1.0 if is_correct else 0.0,
                "t_r": t_r,
                "k_r": k_r,
            })

    return results


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {"n": 0, "acc": 0.0, "rec": 0.0, "t_r": 0.0, "k_r": 0.0}
    n = len(results)
    return {
        "n": n,
        "acc": sum(r["acc"] for r in results) / n,
        "rec": sum(r["rec"] for r in results) / n,
        "t_r": sum(r["t_r"] for r in results) / n,
        "k_r": sum(r["k_r"] for r in results) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mem0 baseline eval, same protocol as MEMOIR/LiCoMemory")
    parser.add_argument("--dataset", default=str(_LOCAL_DATA / "locomo10.json"))
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--output", default="results/mem0_llama8b_2samples.json")
    args = parser.parse_args()

    from load_dataset import load_locomo_dataset
    from mem0 import Memory

    samples = load_locomo_dataset(args.dataset)[: args.n_samples]
    mem0_client = Memory.from_config(MEM0_CONFIG)
    ollama_client = OllamaClient()

    flat: list[dict] = []
    for s_idx, sample in enumerate(samples):
        print(f"[{s_idx+1}/{len(samples)}] sample {sample.sample_id} ...", flush=True)
        r = run_sample(mem0_client, sample, k=args.k, client=ollama_client,
                        sample_id=f"locomo_sample_{s_idx}")
        flat.extend(r)
        print(f"  running n={len(flat)}", flush=True)

    agg = aggregate(flat)
    print(f"\nMem0: Acc={agg['acc']:.4f} Rec@{args.k}={agg['rec']:.4f} "
          f"T_R={agg['t_r']:.3f}s K_R={agg['k_r']:.0f} n={agg['n']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
