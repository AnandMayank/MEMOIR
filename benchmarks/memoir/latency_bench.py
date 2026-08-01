"""
Write-path latency benchmark.

Measures the blocking latency of memory.add() for each system:
  memoir    — MEMOIR (embedding only; TRIAGE-α is async/background, not counted)
  gen_agents — Park et al. (2023): embedding + 1 LLM call per write
  amem      — A-MEM: embedding + 2 LLM calls per write (analyze + evolve)

Reports P50 / P95 / P99 over N trials per system.

Note: gen_agents and amem latency requires ANTHROPIC_API_KEY (direct API key).
      memoir latency runs without any API key.
      In mock mode (no valid API key) gen_agents importance=constant 0.5,
      making it as fast as memoir — not a fair comparison. Run with a real key.

Usage:
  python benchmarks/memoir/latency_bench.py --n-trials 1000 --output results/latency.json

  # Without API key: only MEMOIR write path is meaningful
  python benchmarks/memoir/latency_bench.py --systems memoir --n-trials 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Sample texts (realistic conversational turns for the benchmark)
_SAMPLE_TEXTS = [
    "Speaker Caroline says: I went to a LGBTQ support group yesterday and it was so powerful.",
    "Speaker Melanie says: That's amazing, I'm so glad you found that community.",
    "Speaker Caroline says: I've been thinking about going back to school for psychology.",
    "Speaker Melanie says: You'd be great at that, you're so good at listening.",
    "Speaker Caroline says: My cat Whiskers has been acting strange lately.",
    "Speaker Melanie says: Did you take her to the vet? Could be the change in season.",
    "Speaker Caroline says: I've been reading a lot about adoption agencies recently.",
    "Speaker Melanie says: Are you seriously considering adopting? That's a big step.",
    "Speaker Caroline says: Yeah, I think I want to have kids someday.",
    "Speaker Melanie says: That's wonderful. Have you talked to your partner about it?",
    "Speaker Caroline says: We've had some preliminary discussions.",
    "Speaker Melanie says: Well I'm here if you need to talk through any of it.",
    "Speaker Caroline says: Thanks, that means a lot. How are your kids doing?",
    "Speaker Melanie says: Busy as always. School has been intense for them.",
    "Speaker Caroline says: I remember you saying they're both in middle school now.",
    "Speaker Melanie says: Yeah, it's a tough age. Lots of drama.",
    "Speaker Caroline says: I'll keep that in mind when I'm a parent someday.",
    "Speaker Melanie says: You'll do great. You're very nurturing.",
    "Speaker Caroline says: I try. Hey, did you hear about the new community center opening?",
    "Speaker Melanie says: No! When is that happening?",
]


def _percentiles(arr: list[float]) -> dict[str, float]:
    a = np.array(arr)
    return {
        "p50_ms":  float(np.percentile(a, 50) * 1000),
        "p95_ms":  float(np.percentile(a, 95) * 1000),
        "p99_ms":  float(np.percentile(a, 99) * 1000),
        "mean_ms": float(np.mean(a) * 1000),
        "std_ms":  float(np.std(a) * 1000),
        "n":       len(arr),
    }


def bench_memoir(n_trials: int, verbose: bool) -> list[float]:
    """Sync context: asyncio.run() per add() — includes TRIAGE-α blocking."""
    from harpo.memoir.store import MemoirStore
    store = MemoirStore()
    latencies = []
    for i in range(n_trials):
        text = _SAMPLE_TEXTS[i % len(_SAMPLE_TEXTS)]
        t0 = time.perf_counter()
        store.add(text)
        latencies.append(time.perf_counter() - t0)
        if verbose and (i + 1) % 100 == 0:
            print(f"  memoir_sync: {i+1}/{n_trials} done, "
                  f"running P50={np.percentile(latencies,50)*1000:.1f}ms")
    return latencies


async def _async_add_loop(n_trials: int, verbose: bool) -> list[float]:
    """Async context: TRIAGE-α fires as create_task (non-blocking)."""
    import asyncio
    from harpo.memoir.store import MemoirStore
    store = MemoirStore()
    latencies = []
    for i in range(n_trials):
        text = _SAMPLE_TEXTS[i % len(_SAMPLE_TEXTS)]
        t0 = time.perf_counter()
        store.add(text)  # In running loop: create_task fires TRIAGE-α in background
        latencies.append(time.perf_counter() - t0)
        # Yield to event loop so background tasks can run (prevents task backlog)
        if i % 10 == 0:
            await asyncio.sleep(0)
        if verbose and (i + 1) % 100 == 0:
            print(f"  memoir_async: {i+1}/{n_trials} done, "
                  f"running P50={np.percentile(latencies,50)*1000:.1f}ms")
    return latencies


def bench_memoir_async(n_trials: int, verbose: bool) -> list[float]:
    """Async context (actual agent deployment): TRIAGE-α is non-blocking background task."""
    import asyncio
    return asyncio.run(_async_add_loop(n_trials, verbose))


def bench_gen_agents(n_trials: int, llm_client, verbose: bool) -> list[float]:
    sys.path.insert(0, str(Path(__file__).parent))
    from baseline_gen_agents import GenerativeAgentsStore
    store = GenerativeAgentsStore(llm_client=llm_client)
    latencies = []
    for i in range(n_trials):
        text = _SAMPLE_TEXTS[i % len(_SAMPLE_TEXTS)]
        t0 = time.perf_counter()
        store.add(text)
        latencies.append(time.perf_counter() - t0)
        if verbose and (i + 1) % 20 == 0:
            print(f"  gen_agents: {i+1}/{n_trials} done, "
                  f"running P50={np.percentile(latencies,50)*1000:.1f}ms")
    return latencies


def bench_amem(n_trials: int, verbose: bool) -> Optional[list[float]]:
    """Run A-MEM's AgenticMemorySystem.add_note() latency. Requires A-MEM + SGLang/Ollama."""
    try:
        sys.path.insert(0, str(Path("/home/user/A-MEM")))
        from memory_layer import AgenticMemorySystem
        memory = AgenticMemorySystem(model="claude-haiku-4-5-20251001", backend="anthropic")
        latencies = []
        for i in range(n_trials):
            text = _SAMPLE_TEXTS[i % len(_SAMPLE_TEXTS)]
            t0 = time.perf_counter()
            memory.add_note(text)
            latencies.append(time.perf_counter() - t0)
            if verbose and (i + 1) % 10 == 0:
                print(f"  amem: {i+1}/{n_trials} done, "
                      f"running P50={np.percentile(latencies,50)*1000:.1f}ms")
        return latencies
    except Exception as e:
        print(f"  [amem] skipped: {e}")
        return None


def bench_feedback(attr: str, n_trials: int, verbose: bool, k: int = 5) -> list[float]:
    """
    Feedback-path latency: feedback_answer() per call.
      attr="rouge1" — lexical attribution (default TRACE-FB)
      attr="cf"     — TRACE-CF counterfactual LOO (k+1 scorer forwards, local GPU)
    Write path is unaffected by CF; this isolates the feedback-time cost.
    """
    from harpo.memoir.store import MemoirStore
    store = MemoirStore(use_cf_attr=(attr == "cf"), use_co_cite=False)
    # Populate a realistic store (5× the sample turns)
    store.add_many([t for _ in range(5) for t in _SAMPLE_TEXTS])
    questions = [
        "What support group did Caroline attend?",
        "What is Caroline considering studying?",
        "What is the name of Caroline's cat?",
        "How many kids does Melanie have?",
        "What new building is opening in the community?",
    ]
    answers = [
        "an LGBTQ support group",
        "psychology",
        "Whiskers",
        "2",
        "a new community center",
    ]
    if attr == "cf":  # warm up the scorer LM (model load excluded from timing)
        store.feedback_answer(questions[0], answers[0], k=k)
    latencies = []
    for i in range(n_trials):
        q, a = questions[i % len(questions)], answers[i % len(answers)]
        t0 = time.perf_counter()
        store.feedback_answer(q, a, k=k)
        latencies.append(time.perf_counter() - t0)
        if verbose and (i + 1) % 20 == 0:
            print(f"  feedback_{attr}: {i+1}/{n_trials} done, "
                  f"running P50={np.percentile(latencies,50)*1000:.1f}ms")
    return latencies


def print_table(results: dict) -> None:
    print("\n## Write-path latency (add() per item)\n")
    header = f"{'System':<16} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'Mean (ms)':>10} {'n':>6}"
    print(header)
    print("─" * len(header))
    for sys_name, r in results.items():
        if r is None:
            print(f"{sys_name:<16} {'SKIPPED':>10}")
        else:
            print(f"{sys_name:<16} {r['p50_ms']:>10.1f} {r['p95_ms']:>10.1f} "
                  f"{r['p99_ms']:>10.1f} {r['mean_ms']:>10.1f} {r['n']:>6}")

    # Speedup vs gen_agents
    if "memoir" in results and results["memoir"] and "gen_agents" in results and results["gen_agents"]:
        speedup = results["gen_agents"]["p50_ms"] / results["memoir"]["p50_ms"]
        print(f"\nMEMOIR is {speedup:.1f}× faster than gen_agents at P50 write latency.")
        if results["gen_agents"].get("is_mock"):
            print("  (Note: gen_agents ran in mock mode — real LLM latency would be higher.)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write-path latency benchmark")
    parser.add_argument("--systems", default="memoir,gen_agents",
                        help="Comma-separated: memoir,gen_agents,amem,"
                             "feedback_rouge1,feedback_cf")
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--output", default="results/latency.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    verbose = not args.quiet

    # Build LLM client (mock mode if no valid API key)
    sys.path.insert(0, str(Path(__file__).parent))
    from llm_client import LLMClient
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    # OAuth tokens (sk-ant-oat01-*) don't work with the direct API
    is_oauth = "oat" in api_key
    if is_oauth or not api_key:
        print("Note: Running gen_agents in mock mode (no valid direct API key). "
              "Set ANTHROPIC_API_KEY=sk-ant-api03-... for real LLM latency.\n")
        llm_client = LLMClient(mock_mode=True)
        mock_mode = True
    else:
        llm_client = LLMClient(api_key=api_key)
        mock_mode = False

    results: dict[str, Optional[dict]] = {}

    if "memoir" in systems:
        print(f"Benchmarking MEMOIR sync ({args.n_trials} trials)...")
        lat = bench_memoir(args.n_trials, verbose)
        results["memoir_sync"] = _percentiles(lat)
        print(f"Benchmarking MEMOIR async ({args.n_trials} trials)...")
        lat = bench_memoir_async(args.n_trials, verbose)
        results["memoir_async"] = _percentiles(lat)

    if "gen_agents" in systems:
        print(f"\nBenchmarking Generative Agents ({args.n_trials} trials)...")
        lat = bench_gen_agents(args.n_trials, llm_client, verbose)
        r = _percentiles(lat)
        r["is_mock"] = mock_mode
        results["gen_agents"] = r

    if "amem" in systems:
        print(f"\nBenchmarking A-MEM ({args.n_trials} trials)...")
        lat = bench_amem(args.n_trials, verbose)
        results["amem"] = _percentiles(lat) if lat else None

    for fb in ("feedback_rouge1", "feedback_cf"):
        if fb in systems:
            attr = fb.split("_", 1)[1]
            print(f"\nBenchmarking feedback path ({attr}, {args.n_trials} trials)...")
            lat = bench_feedback(attr, args.n_trials, verbose)
            results[fb] = _percentiles(lat)

    print_table(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, float):
            return round(obj, 4)
        return obj

    with open(out_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
