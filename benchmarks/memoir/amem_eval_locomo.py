"""
A-MEM baseline eval on the same LiCoMemory-matched protocol used for MEMOIR,
LiCoMemory, and Mem0 (mem0_eval_locomo.py): same LoCoMo samples, same
llama3.1:8b backbone via Ollama for both A-MEM's own write-time LLM work
(note construction + memory evolution) and answer generation + judging,
same evidence-based Rec metric via dia_id metadata.

A-MEM (Xu et al., NeurIPS 2025) constructs structured notes with LLM calls at
write time and evolves neighbor links — like Mem0's extraction, its write path
is LLM-dependent, unlike MEMOIR's zero-write-LLM design. Under the same-model
8B protocol this is exactly the hypothesis T1 tests: write-time-LLM systems
degrade with weaker backbones; MEMOIR's write path doesn't.

Requires: /home/user/A-MEM on sys.path (memory_layer.AgenticMemorySystem),
`ollama` python package, local Ollama serving llama3.1:8b.

Usage:
  python benchmarks/memoir/amem_eval_locomo.py \
    --dataset data/locomo10.json --n-samples 2 --k 15 \
    --output results/amem_llama8b_2samples.json
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
# /home/user/A-MEM was the original eval machine's checkout; this machine
# instead vendors the public agiresearch/A-mem repo under external/A-mem,
# with a memory_layer.py shim re-exporting AgenticMemorySystem to match the
# original module path.
_AMEM = Path(os.environ.get("AMEM_PATH", str(_REPO / "external" / "A-mem")))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # vendored load_dataset.py
sys.path.insert(0, str(_AMEM))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ollama_client import OllamaClient
from gemini_client import approx_token_count

_FINETUNED_ENCODER = _REPO / "models" / "memoir-encoder-v1"
LOCAL_ENCODER = (
    str(_FINETUNED_ENCODER)
    if _FINETUNED_ENCODER.is_dir()
    else "all-MiniLM-L6-v2"
)


def evidence_recall_at_k(retrieved_dia_ids: set[str], evidence: list[str]) -> float:
    ev = [e for e in (evidence or []) if e]
    if not ev:
        return 0.0
    hit = sum(1 for e in ev if e in retrieved_dia_ids)
    return hit / len(ev)


def run_sample(amem, sample, k: int, client: OllamaClient) -> list[dict]:
    sessions = sample.conversation.sessions
    sorted_sess_ids = sorted(sessions.keys())
    results: list[dict] = []
    # The public agiresearch/A-mem MemoryNote has no dia_id/session fields and
    # find_related_memories_raw returns a formatted string, not objects — the
    # eval protocol needs per-note dia_id for the evidence-based Rec@k metric,
    # so we track it out-of-band by note id and query the Chroma retriever
    # directly instead of relying on the (incompatible) raw-string helper.
    dia_map: dict[str, str] = {}

    for sess_id in sorted_sess_ids:
        sess = sessions[sess_id]
        for turn in sess.turns:
            text = f"Speaker {turn.speaker} says: {turn.text}"
            # A-MEM's add_note runs its LLM note-construction + (periodically)
            # evolution — the write-time LLM cost T1's latency column measures.
            note_id = amem.add_note(text, time=sess.date_time)
            dia_map[note_id] = turn.dia_id

        sess_qa = [
            qa for qa in sample.qa
            if any(ev.startswith(f"D{sess_id}:") for ev in (qa.evidence or []))
        ]
        for qa in sess_qa:
            gt = str(qa.final_answer)

            t0 = time.perf_counter()
            texts, dia_ids = [], set()
            if amem.memories:
                raw = amem.retriever.search(qa.question, k)
                ids = (raw.get("ids") or [[]])[0]
                metas = (raw.get("metadatas") or [[]])[0]
                for doc_id, meta in zip(ids, metas):
                    texts.append((meta or {}).get("content", ""))
                    d = dia_map.get(doc_id)
                    if d:
                        dia_ids.add(d)
            rec = evidence_recall_at_k(dia_ids, qa.evidence)

            context = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
            k_r = approx_token_count(context)
            answer, _ = client.generate_answer(qa.question, context)
            is_correct, _ = client.judge_correct(qa.question, answer, gt)
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
    ap = argparse.ArgumentParser(
        description="A-MEM baseline eval, same protocol as MEMOIR/LiCoMemory/Mem0")
    ap.add_argument("--dataset", default=str(_LOCAL_DATA / "locomo10.json"))
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--llm-model", default="llama3.1:8b")
    ap.add_argument("--output", default="results/amem_llama8b_2samples.json")
    args = ap.parse_args()

    from load_dataset import load_locomo_dataset
    from memory_layer import AgenticMemorySystem

    samples = load_locomo_dataset(args.dataset)[: args.n_samples]
    ollama_client = OllamaClient()

    flat: list[dict] = []
    for s_idx, sample in enumerate(samples):
        print(f"[{s_idx+1}/{len(samples)}] sample {sample.sample_id} ...", flush=True)
        # fresh A-MEM store per sample (same convention as MEMOIR/Mem0 evals)
        amem = AgenticMemorySystem(
            model_name=LOCAL_ENCODER,
            llm_backend="ollama",
            llm_model=args.llm_model,
        )
        r = run_sample(amem, sample, k=args.k, client=ollama_client)
        flat.extend(r)
        print(f"  running n={len(flat)}", flush=True)

    agg = aggregate(flat)
    print(f"\nA-MEM: Acc={agg['acc']:.4f} Rec@{args.k}={agg['rec']:.4f} "
          f"T_R={agg['t_r']:.3f}s K_R={agg['k_r']:.0f} n={agg['n']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({**agg, "per_qa": flat}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
