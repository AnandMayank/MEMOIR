"""
LiCoMemory-matched SOTA comparison eval for MEMOIR.

Reproduces the metric protocol from LiCoMemory (arXiv:2511.01448) Table 1:
  Acc.  — LLM-judge binary correctness of a generated answer
  Rec.  — proportion of ground-truth evidence items retrieved in top-15
  T_R   — query latency (retrieval + generation + judge)
  K_R   — prompt tokens in the retrieval-augmented generation context

Differs from eval_locomo.py's ctx_recall@5 (token-overlap proxy, no LLM
required) — this script actually generates and judges answers via Gemini,
so it is scoped to a small subset (n_samples) to stay within free-tier
rate limits. Not a replacement for eval_locomo.py; a separate, narrower
protocol run specifically to compare against LiCoMemory's published table.

Usage:
  python benchmarks/memoir/eval_locomo_sota.py \\
    --dataset data/locomo10.json \\
    --systems cosine,memoir_no_trace,memoir_trace,memoir_full \\
    --n-samples 2 --k 15 \\
    --output results/locomo_sota_table.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent.parent
_LOCAL_DATA = _REPO / "data"
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # vendored load_dataset.py

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from harpo.memoir.store import MemoirStore
from gemini_client import approx_token_count
from eval_locomo import make_system, system_add, _MEMOIR_FEEDBACK_SYSTEMS


def make_client(backend: str):
    if backend == "gemini":
        from gemini_client import GeminiClient
        return GeminiClient()
    if backend == "ollama":
        from ollama_client import OllamaClient
        return OllamaClient()
    raise ValueError(f"Unknown backend: {backend!r}")


def evidence_recall_at_k(retrieved_dia_ids: set[str], evidence: list[str]) -> float:
    """Fraction of gold evidence items present among the top-k retrieved items."""
    ev = [e for e in (evidence or []) if e]
    if not ev:
        return 0.0
    hit = sum(1 for e in ev if e in retrieved_dia_ids)
    return hit / len(ev)


def format_context(items: list) -> str:
    parts = []
    for i, m in enumerate(items, 1):
        parts.append(f"{i}. {m.text}")
    return "\n".join(parts)


def run_system(sys_name: str, sample, k: int, client,
               model_name: str) -> list[dict]:
    """Build the store via the multi-session protocol, then evaluate every QA
    with generate+judge. Returns a list of per-QA result dicts."""
    store = make_system(sys_name, model_name=model_name)
    sessions = sample.conversation.sessions
    sorted_sess_ids = sorted(sessions.keys())

    results: list[dict] = []

    for sess_id in sorted_sess_ids:
        sess = sessions[sess_id]
        for turn in sess.turns:
            text = f"Speaker {turn.speaker} says: {turn.text}"
            store.add(text, metadata={"dia_id": turn.dia_id, "session": sess_id})

        sess_qa = [
            qa for qa in sample.qa
            if any(ev.startswith(f"D{sess_id}:") for ev in (qa.evidence or []))
        ]
        for qa in sess_qa:
            gt = str(qa.final_answer)
            ablation = getattr(store, "_ablation", {})

            t0 = time.perf_counter()
            items = store.search(qa.question, k=k, **ablation)
            retrieved_dia_ids = {
                m.metadata.get("dia_id") for m in items if m.metadata.get("dia_id")
            }
            rec = evidence_recall_at_k(retrieved_dia_ids, qa.evidence)

            context = format_context(items)
            k_r = approx_token_count(context)
            try:
                answer, gen_latency = client.generate_answer(qa.question, context)
                is_correct, judge_latency = client.judge_correct(qa.question, answer, gt)
            except Exception as e:
                if type(e).__name__ == "QuotaExhaustedError":
                    print(f"  QUOTA EXHAUSTED after {len(results)} QAs in this system "
                          f"— returning partial results. {e}", flush=True)
                    return results
                raise
            t_r = (time.perf_counter() - t0)

            results.append({
                "category": qa.category,
                "rec": rec,
                "acc": 1.0 if is_correct else 0.0,
                "t_r": t_r,
                "k_r": k_r,
            })

            # TRACE-FB feedback using the actually-generated answer (not GT) —
            # more faithful to production use than the GT-simulated feedback
            # used in eval_locomo.py's cheaper ctx_recall@5 protocol.
            if sys_name in _MEMOIR_FEEDBACK_SYSTEMS and answer:
                store.feedback_answer(qa.question, answer, k=50)

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
    parser = argparse.ArgumentParser(description="LiCoMemory-matched SOTA eval for MEMOIR")
    parser.add_argument("--dataset", default=str(_LOCAL_DATA / "locomo10.json"))
    parser.add_argument("--systems", default="cosine,memoir_no_trace,memoir_trace,memoir_full")
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--k", type=int, default=15)
    # "models/memoir-encoder-v1" (a fine-tuned encoder from the original dev
    # machine) is not present on every machine this script runs on -- fails
    # closed under HF_HUB_OFFLINE=1 (set below) rather than silently falling
    # back, and eval_consolidation.py/eval_qa_together.py/etc. all default to
    # all-MiniLM-L6-v2 already. Match that default here.
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "ollama"],
                        help="gemini = Gemini 2.5 Flash (generate+judge); "
                             "ollama = local llama3.1:8b (open-source, lower-fidelity "
                             "vs LiCoMemory's 70B condition)")
    parser.add_argument("--output", default="results/locomo_sota_table.json")
    args = parser.parse_args()

    from load_dataset import load_locomo_dataset
    samples = load_locomo_dataset(args.dataset)[: args.n_samples]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    client = make_client(args.backend)

    all_results: dict[str, dict] = {}
    for sys_name in systems:
        print(f"\n{'─'*60}\nSystem: {sys_name} (backend={args.backend})\n{'─'*60}")
        flat: list[dict] = []
        for s_idx, sample in enumerate(samples):
            print(f"  [{s_idx+1}/{len(samples)}] sample {sample.sample_id} ...", flush=True)
            r = run_system(sys_name, sample, k=args.k, client=client, model_name=args.model_name)
            flat.extend(r)
            print(f"    running n={len(flat)}", flush=True)
        agg = aggregate(flat)
        all_results[sys_name] = agg
        print(f"  {sys_name}: Acc={agg['acc']:.4f} Rec@{args.k}={agg['rec']:.4f} "
              f"T_R={agg['t_r']:.3f}s K_R={agg['k_r']:.0f} n={agg['n']}")

    print("\n\n## LiCoMemory-matched SOTA Table\n")
    header = f"{'System':<20} {'Acc.':>8} {f'Rec.@{args.k}':>10} {'T_R':>8} {'K_R':>8} {'n':>6}"
    print(header)
    print("─" * len(header))
    for sys_name, agg in all_results.items():
        print(f"{sys_name:<20} {agg['acc']*100:>7.2f}% {agg['rec']*100:>9.2f}% "
              f"{agg['t_r']:>7.2f}s {agg['k_r']:>8.0f} {agg['n']:>6}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
