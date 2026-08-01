"""
Certified-consolidation evaluation on LoCoMo (combined-paper prospective half).

Protocol (extends eval_locomo's online session-split): sessions are ingested in
order; each session's evidenced QAs are answered and TRACE feedback fires; every
`--sleep-every` sessions the store runs a consolidation pass. Directly measured:

  eps_hat   sufficiency distortion: over QAs that were answerable (rec5=1) at
            their evaluation time, the fraction no longer answerable when
            re-retrieved right after a sleep. Def. 1 of the certified theory,
            on realized queries instead of synthetic probes.
  tokens    r(Z) live-store token rate at each sleep (compression).
  rec@5 / full-ev@5 etc. — the standard eval_locomo v7 metrics, so the cost of
            consolidation to ongoing retrieval is visible in the same table.
  rho_hat   TRACE-observed support–operation incidence (Bridge 1).
  audit     every op with judge scores -> results/consolidation_audit_*.jsonl
            (delta-hat calibration is a separate independent-auditor pass).

Variants:
  no_sleep          anchor (never consolidates)
  certified         judge-gated merges + dominated prunes (ours)
  uncertified       same operator, judge gate bypassed (MemRefine-style —
                    isolates the certificate; scores still recorded)
  recency_prune     token-matched: prune oldest until certified's budget
  random_prune      token-matched: prune random until certified's budget

Token matching: `certified` runs FIRST per sample and records its post-sleep
compression ratios; heuristic baselines then prune to the same ratios at the
same sleep points (the certified repo's fairness rule).

Judge (--judge):
  stub:lossless | stub:lossy    pipeline testing, no LLM
  ollama:<model>                local Ollama (e.g. ollama:llama3.1:8b)
  together:<model>              Together API, disk-cached, budget-capped

Usage (smoke gate):
  python benchmarks/memoir/eval_consolidation.py \
    --n-samples 2 --base-system memoir_trace --judge ollama:llama3.1:8b \
    --variants no_sleep,certified,uncertified \
    --output results/consolidation_smoke.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).parent))

from harpo.memoir.consolidate import FaithfulnessJudge, summarize_all_to_core
from harpo.memoir.trace_fb import rouge1_recall
from eval_locomo import (
    _CATEGORY_NAMES,
    _aggregate,
    _eval_and_feedback,
    _parse_locomo_datetime,
    evidence_recall,
    make_system,
    system_add,
)


# ── Judge backends ─────────────────────────────────────────────────────────────

def make_generate_fn(spec: str, budget_usd: float = 2.0):
    """Return (callable(prompt)->str, describe_dict)."""
    kind, _, model = spec.partition(":")

    if kind == "stub":
        text = "LOSSLESS" if model != "lossy" else "LOSSY"
        # supersede prompts speak a different verdict vocabulary; map the
        # stub's polarity onto it so the certified_supersede variant is
        # smoke-testable end-to-end (stub:lossless => everything certifies)
        supersede_text = "SUPERSEDES" if model != "lossy" else "INDEPENDENT"

        def stub_fn(prompt):
            if "consolidation agent" in prompt:
                # merge-text stub: concatenate the entry lines (lossless by
                # construction — keeps the pipeline test honest)
                lines = [l.split(": ", 1)[1] for l in prompt.splitlines()
                         if l.startswith("Entry ")]
                return " ".join(lines)
            if "SUPERSEDES or INDEPENDENT" in prompt:
                return supersede_text
            return text
        return stub_fn, {"judge": spec, "cost": 0.0}

    if kind == "ollama":
        import urllib.request

        def ollama_fn(prompt):
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({"model": model, "prompt": prompt,
                                 "stream": False,
                                 "options": {"temperature": 0.0,
                                             "num_predict": 200}}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["response"]
        return ollama_fn, {"judge": spec, "cost": 0.0}

    if kind == "together":
        from eval_qa_together import TogetherClient
        client = TogetherClient(budget_usd=budget_usd, **(
            {"model": model} if model else {}))

        def together_fn(prompt):
            text, _ = client.chat(prompt, max_tokens=200)
            return text
        return together_fn, {"judge": spec, "client": client}

    if kind == "router":
        # tokenrouter.com — OpenAI-compatible aggregator (GPT-4o-mini etc.,
        # matching LiCoMemory's second published backbone). Same disk-cache +
        # budget-cap discipline as the Together client. Base URL/key from env:
        # TOKEN_ROUTER_BASE_URL (default https://www.tokenrouter.com/api) and
        # TOKEN_ROUTER_API_KEY.
        from router_client import RouterClient
        client = RouterClient(budget_usd=budget_usd, **(
            {"model": model} if model else {}))

        def router_fn(prompt):
            text, _ = client.chat(prompt, max_tokens=200)
            return text
        return router_fn, {"judge": spec, "client": client}

    raise ValueError(f"unknown judge spec: {spec!r}")


# ── Token-matched heuristic pruning (baseline variants) ────────────────────────

def prune_to_ratio(store, target_ratio: float, mode: str, rng) -> int:
    """Mark items pruned until live tokens <= target_ratio * current tokens."""
    live = [m for m in store.all_items() if not m.pruned and not m.hallucinated]
    total = sum(len(m.text.split()) for m in live)
    target = target_ratio * total
    if mode == "recency":
        order = sorted(live, key=lambda m: m.created_at)          # oldest first
    elif mode == "random":
        order = list(live)
        rng.shuffle(order)
    else:
        raise ValueError(mode)
    pruned = 0
    for m in order:
        if total <= target:
            break
        m.pruned = True
        total -= len(m.text.split())
        pruned += 1
    if pruned:
        store._matrix_dirty = True
        store._bm25_dirty = True
    return pruned


# ── Sufficiency probe: fresh pre/post snapshot around a single sleep event ─────
#
# Def. 1: eps = Pr[Z fails q | X answered q]. X and Z must be the store
# IMMEDIATELY before and after the SAME sleep event — using a stale
# "was answerable when originally evaluated N sessions ago" flag instead
# confounds consolidation loss with ordinary retrieval drift from unrelated
# session growth in between (caught in the smoke run: eps_hat=0.33 with ZERO
# merges/prunes applied, which is definitionally impossible if the store is
# byte-for-byte unchanged between the two checks).

def _peek_retrieve(sys_name, store, query, k, now):
    """Read-only retrieval for measurement probes — peek=True skips the
    retrieval_count bump search() normally does, so calling this twice
    (pre/post a sleep) doesn't itself perturb UCB/recency state and pollute
    the eps measurement with probe-induced noise (found via smoke-run
    inspection: eps_hat was nonzero even at 0 applied merges/prunes)."""
    if sys_name == "gen_agents":
        return store.search(query, k=k)
    ablation = dict(getattr(store, "_ablation", {}))
    ablation["peek"] = True
    try:
        return store.search(query, k=k, now=now, **ablation)
    except TypeError:
        return store.search(query, k=k)


def rec5_snapshot(sys_name, store, qas, k, now) -> tuple[dict, dict]:
    """Fresh rec5 pass over `qas` against the store's CURRENT state.
    Returns (answerable_by_qa, retrieved_ids_by_qa) — the second map costs
    nothing extra (same retrievals) and feeds the stale_served_rate metric:
    after a sleep applies SUPERSEDE ops, the overlap between these pre-sleep
    retrieved ids and the retired ids measures how much stale/contradicted
    content retrieval was actively serving before supersede retired it
    (post-sleep it is structurally zero, which is the operator's point —
    the merge-only `certified` variant leaves it nonzero)."""
    out = {}
    retrieved_ids = {}
    for qa in qas:
        items = _peek_retrieve(sys_name, store, qa.question, k, now)
        out[id(qa)] = evidence_recall(items, qa.evidence) > 0.5
        retrieved_ids[id(qa)] = {getattr(m, "id", None) for m in items}
    return out, retrieved_ids


def epsilon_delta(sys_name, store, qas, pre_snapshot: dict, k, now):
    """Re-check only the QAs that WERE answerable pre-sleep; count post-sleep
    losses. Returns (n_probes, n_lost) isolating this sleep's own effect."""
    probes = [qa for qa in qas if pre_snapshot.get(id(qa))]
    lost = 0
    for qa in probes:
        items = _peek_retrieve(sys_name, store, qa.question, k, now)
        if evidence_recall(items, qa.evidence) < 0.5:
            lost += 1
    return len(probes), lost


def _pearson(x: list, y: list) -> float:
    if len(x) < 2:
        return 0.0
    xv, yv = np.array(x, dtype=float), np.array(y, dtype=float)
    if xv.std() < 1e-9 or yv.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])


def correlation_snapshot(sys_name, store, qas, k, now) -> tuple[list, list]:
    """
    Bridge-2 signature measurement: for each QA, (max ROUGE1 attribution of
    retrieved memories against the GT answer, evidence_recall hit). Returns
    parallel (attributions, hits) lists — call before AND after a sleep on
    the same qas to see whether the credit-ledger's attribution/retrieval
    correlation survives consolidation (certified) or degrades (uncertified,
    "uncertified consolidation corrupts the credit ledger").
    """
    attrs, hits = [], []
    for qa in qas:
        items = _peek_retrieve(sys_name, store, qa.question, k, now)
        gt = str(qa.final_answer)
        texts = [getattr(m, "text", "") for m in items]
        attrs.append(max((rouge1_recall(t, gt) for t in texts), default=0.0))
        hits.append(evidence_recall(items, qa.evidence))
    return attrs, hits


# ── Per-sample protocol with sleep hook ────────────────────────────────────────

def evaluate_sample_with_sleep(
    sample,
    sys_name: str,
    store,
    variant: str,
    judge_fn,
    k: int = 5,
    sleep_every: int = 5,
    tau: float = 1.0,
    tau_prune: Optional[float] = None,
    protect_w_imp: float = 0.3,
    credit_transfer: str = "max",
    matched_ratios: Optional[list[float]] = None,
    audit_path: Optional[str] = None,
    seed: int = 13,
    max_clusters: int = 16,
    max_prune_checks: int = 32,
    retrieval_coverage_min: float = 0.6,
    use_co_cite_clusters: bool = True,
    track_correlation: bool = False,
    max_merge_ratio: float = 0.9,
    merge_generate_fn=None,
    max_supersede_checks: int = 0,
    tau_supersede: Optional[float] = None,
    target_ratio: float = 0.8,
    dump_perqa: bool = False,
    retrieval_guard: str = "lexical",
    tau_lex: Optional[float] = None,
    tau_anchor: float = 0.9,
    tau_query: float = 0.8,
    min_probes: int = 3,
    max_probes: int = 10,
    merge_regeneration_attempts: int = 0,
    log_counterfactual_probes: Optional[str] = None,
) -> dict:
    """
    Returns eval_locomo-style aggregate dict extended with consolidation stats.
    If variant is certified/uncertified, `achieved_ratios` in the result carries
    the per-sleep compression ratios (used to token-match heuristic baselines).

    track_correlation: Bridge-2 signature measurement — Pearson r(max
    attribution, evidence hit) over `seen_qas`, snapshotted immediately
    before and after each sleep. Adds 2x the retrieval cost of eps_hat alone
    (no LLM calls — pure retrieval), off by default.

    max_supersede_checks: >0 enables the certified SUPERSEDE operator inside
    each sleep (store.consolidate passes it through). The `certified_supersede`
    variant is `certified` with this defaulted on — isolating supersede's
    marginal effect against merge-only certified.
    """
    rng = np.random.default_rng(seed)
    sessions = sample.conversation.sessions
    sorted_sess_ids = sorted(sessions.keys())

    judge = FaithfulnessJudge(judge_fn, n_votes=3, certify_threshold=tau) \
        if judge_fn else None

    all_results: list[dict] = []
    seen_qas: list = []   # all QAs answered so far (any prior session)
    sleep_log: list[dict] = []
    achieved_ratios: list[float] = []
    eps_probes = eps_lost = 0
    corr_log: list[dict] = []
    matched_iter = iter(matched_ratios or [])

    for si, sess_id in enumerate(sorted_sess_ids):
        sess = sessions[sess_id]
        sess_epoch = _parse_locomo_datetime(sess.date_time)
        for turn in sess.turns:
            text = f"Speaker {turn.speaker} says: {turn.text}"
            system_add(sys_name, store, text,
                       {"session": sess_id, "time": sess.date_time,
                        "dia_id": turn.dia_id},
                       created_at=sess_epoch)

        sess_qa = [qa for qa in sample.qa
                   if any(ev.startswith(f"D{sess_id}:")
                          for ev in (qa.evidence or []))]
        count = sess_id != sorted_sess_ids[0]
        before_n = len(all_results)
        _eval_and_feedback(sys_name, store, sess_qa, k, "gt", all_results,
                           session=sess_id, count_in_metric=count,
                           verbose=False, now=sess_epoch)
        seen_qas.extend(sess_qa)

        # ── sleep hook ────────────────────────────────────────────────────────
        is_last = si == len(sorted_sess_ids) - 1
        if variant == "no_sleep" or (si + 1) % sleep_every != 0 or is_last:
            continue

        # Fresh pre-sleep snapshot — X in Def. 1, taken right before this sleep
        # so nothing but the sleep itself changes before the post-check.
        pre_snapshot, pre_retrieved_ids = rec5_snapshot(
            sys_name, store, seen_qas, k, now=sess_epoch)
        pre_attrs = pre_hits = None
        if track_correlation:
            pre_attrs, pre_hits = correlation_snapshot(sys_name, store, seen_qas,
                                                        k, now=sess_epoch)

        tokens_before = store.total_tokens()
        ops_before = len(getattr(store, "consolidation_ops", []))
        t0 = time.perf_counter()
        if variant in ("certified_budget", "uncertified_budget"):
            # Compression-targeted mode (RQ5's actual test): repeated
            # certified passes until target_ratio is hit or the certificate
            # refuses everything left. Aggressive per-pass caps — the
            # safety-first defaults only touch ~2% of store tokens/sleep.
            stats = store.consolidate_to_budget(
                judge,
                target_ratio=target_ratio,
                max_passes=4,
                merge_generate_fn=merge_generate_fn,
                certified=(variant == "certified_budget"),
                protect_w_imp=protect_w_imp,
                credit_transfer=credit_transfer,
                audit_path=audit_path,
                max_clusters=max(max_clusters, 64),
                max_cluster_size=6,
                max_prune_checks=max_prune_checks,
                retrieval_coverage_min=retrieval_coverage_min,
                tau_prune=tau_prune,
                use_co_cite_clusters=use_co_cite_clusters,
                max_merge_ratio=max_merge_ratio,
                max_supersede_checks=max(max_supersede_checks, 16),
                tau_supersede=tau_supersede,
            )
        elif variant in ("certified", "certified_supersede", "uncertified"):
            # certified_supersede = certified merges + the SUPERSEDE operator —
            # w_conf-flagged contradiction pairs where the judge certifies the
            # newer item supersedes the older (its home-turf metric is
            # stale_retrieval_rate below, its home-turf category LME-S
            # knowledge-update). Isolates supersede's marginal effect against
            # merge-only `certified`.
            supersede_n = (max(max_supersede_checks, 8)
                           if variant == "certified_supersede"
                           else max_supersede_checks)
            stats = store.consolidate(
                judge,
                merge_generate_fn=merge_generate_fn,
                certified=(variant != "uncertified"),
                protect_w_imp=protect_w_imp,
                credit_transfer=credit_transfer,
                audit_path=audit_path,
                max_clusters=max_clusters,
                max_prune_checks=max_prune_checks,
                retrieval_coverage_min=retrieval_coverage_min,
                tau_prune=tau_prune,
                use_co_cite_clusters=use_co_cite_clusters,
                max_merge_ratio=max_merge_ratio,
                max_supersede_checks=supersede_n,
                tau_supersede=tau_supersede,
                retrieval_guard=retrieval_guard,
                tau_lex=tau_lex,
                tau_anchor=tau_anchor,
                tau_query=tau_query,
                min_probes=min_probes,
                max_probes=max_probes,
                merge_regeneration_attempts=merge_regeneration_attempts,
                log_counterfactual_probes=log_counterfactual_probes,
            )
        elif variant in ("recency_prune", "random_prune"):
            ratio = next(matched_iter, 1.0)
            n = prune_to_ratio(store, ratio, variant.split("_")[0], rng)
            stats = {"applied_merges": 0, "applied_prunes": n,
                     "tokens_before": tokens_before,
                     "tokens_after": store.total_tokens(), "judge_calls": 0}
        elif variant == "summarize_all":
            stats = summarize_all_to_core(store, judge.generate_fn)
        else:
            raise ValueError(variant)
        sleep_s = time.perf_counter() - t0

        tokens_after = stats["tokens_after"]
        achieved_ratios.append(tokens_after / max(1, tokens_before))

        if track_correlation:
            post_attrs, post_hits = correlation_snapshot(sys_name, store, seen_qas,
                                                          k, now=sess_epoch)
            corr_log.append({
                "after_session": sess_id,
                "r_pre": _pearson(pre_attrs, pre_hits),
                "r_post": _pearson(post_attrs, post_hits),
                "n": len(seen_qas),
            })

        # Z in Def. 1 — same store, right after the sleep, same "now".
        n_probes, n_lost = epsilon_delta(sys_name, store, seen_qas, pre_snapshot,
                                         k, now=sess_epoch)
        eps_probes += n_probes
        eps_lost += n_lost

        # stale_served_rate (SUPERSEDE's home metric): of the queries probed
        # pre-sleep, what fraction were being served at least one item that
        # THIS sleep's certified supersede ops then retired as contradicted/
        # stale? Zero extra retrievals — reuses the pre-sleep snapshot ids.
        retired_ids = {
            op.constituent_ids[0]
            for op in getattr(store, "consolidation_ops", [])[ops_before:]
            if getattr(op, "op_type", "") == "supersede" and op.applied
        } if stats.get("applied_supersedes", 0) else set()
        stale_served = sum(
            1 for qa in seen_qas
            if pre_retrieved_ids.get(id(qa), set()) & retired_ids
        ) if retired_ids else 0
        sleep_log.append({
            "after_session": sess_id, **stats,
            "sleep_seconds": round(sleep_s, 2),
            "eps_probes": n_probes, "eps_lost": n_lost,
            "stale_served": stale_served,
            "stale_served_rate": stale_served / max(1, len(seen_qas)),
        })

    agg = _aggregate(all_results)
    agg["variant"] = variant
    agg["final_tokens"] = store.total_tokens()
    agg["eps_hat"] = (eps_lost / eps_probes) if eps_probes else None
    agg["eps_probes"] = eps_probes
    agg["rho_hat"] = getattr(store, "rho_hat", 0.0)
    agg["achieved_ratios"] = achieved_ratios
    agg["sleeps"] = sleep_log
    agg["applied_merges"] = sum(s["applied_merges"] for s in sleep_log)
    agg["applied_prunes"] = sum(s["applied_prunes"] for s in sleep_log)
    agg["applied_supersedes"] = sum(s.get("applied_supersedes", 0) for s in sleep_log)
    agg["stale_served_total"] = sum(s.get("stale_served", 0) for s in sleep_log)
    if track_correlation:
        agg["correlation"] = corr_log
    if dump_perqa:
        # raw per-QA rows (question = pairing key) for significance.py's
        # paired bootstrap between variants — same row schema as eval_locomo
        sid = str(getattr(sample, "sample_id", ""))
        agg["perqa"] = [{"sample_id": sid, **r} for r in all_results]
    return agg


# ── Parallel worker (--workers > 1) ─────────────────────────────────────────────

def _eval_one_sample_worker(
    sample, base_system: str, variant: str,
    judge_spec: str, budget_usd: float,
    k: int, sleep_every: int, tau: float, tau_prune, protect_w_imp: float,
    credit_transfer: str, matched_ratios, audit_path, max_clusters: int,
    max_prune_checks: int, retrieval_coverage_min: float, model_name: str,
    use_co_cite_clusters: bool = True, track_correlation: bool = False,
    max_merge_ratio: float = 0.9, merge_judge_spec: Optional[str] = None,
    max_supersede_checks: int = 0, tau_supersede: Optional[float] = None,
    target_ratio: float = 0.8, dump_perqa: bool = False,
    retrieval_guard: str = "lexical", tau_lex: Optional[float] = None,
    tau_anchor: float = 0.9, tau_query: float = 0.8, min_probes: int = 3,
    max_probes: int = 10, merge_regeneration_attempts: int = 0,
    log_counterfactual_probes: Optional[str] = None,
) -> dict:
    """
    Picklable, process-safe unit of work for --workers > 1.

    Mirrors eval_locomo.py's _eval_one_sample_worker: each worker rebuilds its
    own store AND its own judge_fn (make_generate_fn's closures aren't
    picklable — the ollama/together fns capture no shared state beyond a
    model name/spec string, so rebuilding is cheap and correct, not a
    workaround). Runs on spawn; matched_ratios (certified's achieved
    compression, feeding the token-matched heuristic baselines) is passed in
    read-only since certified always completes fully before recency_prune/
    random_prune's sample loop starts (see the `order` sort in main()).
    """
    judge_fn, _ = make_generate_fn(judge_spec, budget_usd)
    merge_fn = None
    if merge_judge_spec:
        merge_fn, _ = make_generate_fn(merge_judge_spec, budget_usd)
    store = make_system(base_system, model_name=model_name)
    res = evaluate_sample_with_sleep(
        sample, base_system, store, variant,
        judge_fn=judge_fn, k=k, sleep_every=sleep_every, tau=tau,
        tau_prune=tau_prune, protect_w_imp=protect_w_imp,
        credit_transfer=credit_transfer, matched_ratios=matched_ratios,
        audit_path=audit_path, max_clusters=max_clusters,
        max_prune_checks=max_prune_checks,
        retrieval_coverage_min=retrieval_coverage_min,
        use_co_cite_clusters=use_co_cite_clusters,
        track_correlation=track_correlation,
        max_merge_ratio=max_merge_ratio, merge_generate_fn=merge_fn,
        max_supersede_checks=max_supersede_checks, tau_supersede=tau_supersede,
        target_ratio=target_ratio, dump_perqa=dump_perqa,
        retrieval_guard=retrieval_guard, tau_lex=tau_lex, tau_anchor=tau_anchor,
        tau_query=tau_query, min_probes=min_probes, max_probes=max_probes,
        merge_regeneration_attempts=merge_regeneration_attempts,
        log_counterfactual_probes=log_counterfactual_probes,
    )
    return getattr(sample, "sample_id", None), res


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dataset", default=str(_REPO / "data" / "locomo10.json"))
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--base-system", default="memoir_trace",
                    help="MemoirStore config the variants run on top of")
    ap.add_argument("--variants",
                    default="no_sleep,certified,uncertified,recency_prune,random_prune")
    ap.add_argument("--judge", default="ollama:llama3.1:8b")
    ap.add_argument("--budget-usd", type=float, default=2.0)
    ap.add_argument("--sleep-every", type=int, default=5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tau-prune", type=float, default=None,
                    help="separate certify threshold for dominated-prune; "
                         "defaults to --tau if unset")
    ap.add_argument("--protect-w-imp", type=float, default=0.3)
    ap.add_argument("--credit-transfer", default="max", choices=["max", "zero"])
    ap.add_argument("--max-clusters", type=int, default=16,
                    help="merge-cluster cap per sleep (bounds judge calls)")
    ap.add_argument("--max-prune-checks", type=int, default=10,
                    help="dominated-prune candidate cap per sleep, for uncertified/"
                         "recency_prune/random_prune baselines. The `certified` "
                         "variant ALWAYS runs with prune disabled regardless of this "
                         "flag (see --certified-max-prune-checks): an independent "
                         "auditor found 83.9%% of audited prune ops lossy "
                         "(results/calibration_prune.json) vs merge's 6.25%%.")
    ap.add_argument("--certified-max-prune-checks", type=int, default=0,
                    help="dominated-prune cap specifically for the `certified` "
                         "variant. Defaults to 0 (merge-only certified "
                         "consolidation, see docs/COMBINED_PAPER_PLAN.md). Set >0 "
                         "only for the documented-negative-result prune ablation.")
    ap.add_argument("--retrieval-coverage-min", type=float, default=0.6,
                    help="min fraction of originals' content tokens that must "
                         "survive in merged text (BM25-findability gate)")
    ap.add_argument("--no-co-cite-clusters", action="store_true",
                    help="Bridge-1 ablation (B.8): disable co-cite-seeded merge "
                         "candidates, embedding-similarity clusters only")
    ap.add_argument("--track-correlation", action="store_true",
                    help="Bridge-2 signature measurement (B.9): Pearson "
                         "r(max attribution, evidence hit) immediately before "
                         "and after every sleep, per sample (adds retrieval-only "
                         "cost, no extra LLM calls)")
    ap.add_argument("--max-merge-ratio", type=float, default=0.9,
                    help="length gate: a merge is only certified if "
                         "len(merged) <= this * sum(len(originals)). Measured "
                         "necessary — llama3.1:8b inflated tokens +5.3%% net "
                         "on 44 applied merges without this gate "
                         "(results/consolidation_full10.json)")
    ap.add_argument("--merge-judge", default=None,
                    help="separate model spec for merge-TEXT GENERATION only "
                         "(certification still uses --judge). E.g. "
                         "'together:meta-llama/Llama-3.3-70B-Instruct-Turbo' "
                         "to test whether a stronger model compresses better "
                         "than the local judge model without paying for a "
                         "stronger judge too. Defaults to --judge's model.")
    ap.add_argument("--max-supersede-checks", type=int, default=0,
                    help="SUPERSEDE operator: certified retirement of "
                         "contradicted/stale items (w_conf-flagged pairs), cap "
                         "per sleep. Default 0 = off. The certified_supersede "
                         "variant forces at least 8 regardless of this flag.")
    ap.add_argument("--tau-supersede", type=float, default=None,
                    help="separate certify threshold for supersede; defaults "
                         "to --tau if unset")
    ap.add_argument("--target-ratio", type=float, default=0.8,
                    help="compression target for the certified_budget/"
                         "uncertified_budget variants: repeated certified "
                         "passes until r(Z) <= this * r(X) or the certificate "
                         "refuses everything left (max 4 passes)")
    ap.add_argument("--dump-perqa", action="store_true",
                    help="write per-QA rows to <stem>_<variant>_perqa.jsonl "
                         "for significance.py's paired bootstrap between "
                         "variants (pairing key: sample_id+question+category)")
    ap.add_argument("--retrieval-guard", default="lexical",
                    choices=["none", "lexical", "anchor", "crpc", "hybrid"],
                    help="which retrieval-preservation gate a merge candidate "
                         "must clear (docs/CONSOLIDATION_APPENDIX.md CRPC spec). "
                         "'lexical' is the pre-existing default (kappa_lex >= "
                         "--tau-lex); 'crpc' replays historical queries through "
                         "the live RECALL-Q stack against a temporary "
                         "counterfactual store")
    ap.add_argument("--tau-lex", type=float, default=None,
                    help="lexical-coverage threshold; defaults to "
                         "--retrieval-coverage-min if unset")
    ap.add_argument("--tau-anchor", type=float, default=0.9,
                    help="exact-anchor (entities/dates/numbers/negations/"
                         "speakers) recall threshold for anchor/crpc/hybrid guards")
    ap.add_argument("--tau-query", type=float, default=0.8,
                    help="CRPC historical-query retention threshold")
    ap.add_argument("--min-probes", type=int, default=3,
                    help="minimum historical queries a cluster needs before "
                         "CRPC's query_retention check applies; below this the "
                         "cluster is sparse (crpc mode: rejected; hybrid mode: "
                         "falls back to the lexical gate)")
    ap.add_argument("--max-probes", type=int, default=10,
                    help="cap on historical queries replayed per candidate "
                         "cluster (TRACE-credited queries preferred)")
    ap.add_argument("--merge-regeneration-attempts", type=int, default=0,
                    help="on a retrieval-guard rejection, retry merge-text "
                         "generation this many times with the missing anchors "
                         "appended as a hint, then re-check the same gate")
    ap.add_argument("--log-counterfactual-probes", default=None,
                    help="JSONL path: one line per evaluated merge candidate "
                         "with every CRPC diagnostic, applied or rejected")
    ap.add_argument("--model-name", default="all-MiniLM-L6-v2")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallelize across independent samples within each "
                         "variant (spawn-context process pool, matches "
                         "eval_locomo.py --workers). Each worker hits the "
                         "same judge backend (e.g. Ollama) concurrently -- "
                         "raise OLLAMA_NUM_PARALLEL server-side to benefit, "
                         "see deploy/README_VASTAI.md")
    ap.add_argument("--output", default="results/consolidation.json")
    args = ap.parse_args()

    from load_dataset import load_locomo_dataset
    samples = load_locomo_dataset(args.dataset)[: args.n_samples]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    judge_fn, judge_info = make_generate_fn(args.judge, args.budget_usd)
    merge_fn = None
    if args.merge_judge:
        merge_fn, _ = make_generate_fn(args.merge_judge, args.budget_usd)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_stem = str(out_path.with_suffix(""))

    # certified must run before token-matched baselines
    order = sorted(variants,
                   key=lambda v: 0 if v in ("certified", "certified_budget") else 1)
    results: dict = {"config": {**vars(args), "judge_info": str(judge_info)}}
    matched: dict[str, list[float]] = {}   # sample_id -> certified ratios

    for variant in order:
        print(f"\n{'─'*60}\nVariant: {variant} (base {args.base_system})\n{'─'*60}")
        wants_audit = variant in ("certified", "certified_supersede",
                                   "certified_budget", "uncertified",
                                   "uncertified_budget")
        prune_cap = (args.certified_max_prune_checks
                     if variant in ("certified", "certified_supersede")
                     else args.max_prune_checks)
        n_workers = min(args.workers, len(samples))

        if n_workers > 1:
            print(f"  running {len(samples)} samples across {n_workers} "
                  f"worker processes (spawn)")
            ctx = mp.get_context("spawn")
            ordered: dict = {}
            worker_audit_paths: list[str] = []
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {}
                for i, sample in enumerate(samples):
                    sid = str(getattr(sample, "sample_id", i))
                    # per-sample audit file (concurrent writers to one file
                    # would interleave/corrupt lines) — merged after the pool.
                    ap_i = f"{audit_stem}_{variant}_{sid}_audit.jsonl" \
                        if wants_audit else None
                    if ap_i:
                        worker_audit_paths.append(ap_i)
                    fut = pool.submit(
                        _eval_one_sample_worker, sample, args.base_system,
                        variant, args.judge, args.budget_usd, args.k,
                        args.sleep_every, args.tau, args.tau_prune,
                        args.protect_w_imp, args.credit_transfer,
                        matched.get(sid), ap_i, args.max_clusters, prune_cap,
                        args.retrieval_coverage_min, args.model_name,
                        not args.no_co_cite_clusters, args.track_correlation,
                        args.max_merge_ratio, args.merge_judge,
                        args.max_supersede_checks, args.tau_supersede,
                        args.target_ratio, args.dump_perqa,
                        args.retrieval_guard, args.tau_lex, args.tau_anchor,
                        args.tau_query, args.min_probes, args.max_probes,
                        args.merge_regeneration_attempts,
                        args.log_counterfactual_probes,
                    )
                    futures[fut] = i
                done = 0
                for fut in as_completed(futures):
                    i = futures[fut]
                    sid, res = fut.result()
                    ordered[i] = res
                    done += 1
                    eps = res["eps_hat"]
                    print(f"    [{done}/{len(samples)}] sample {sid} done "
                          f"rec5={res.get('rec5', 0):.4f} "
                          f"tokens={res['final_tokens']} "
                          f"eps={'—' if eps is None else f'{eps:.4f}'} "
                          f"merges={res['applied_merges']} prunes={res['applied_prunes']}")
            per_sample = [ordered[i] for i in range(len(samples))]
            if wants_audit:
                combined = f"{audit_stem}_{variant}_audit.jsonl"
                with open(combined, "w") as out_f:
                    for p in worker_audit_paths:
                        if Path(p).exists():
                            out_f.write(Path(p).read_text())
                            Path(p).unlink()
        else:
            per_sample = []
            for s_idx, sample in enumerate(samples):
                sid = str(getattr(sample, "sample_id", s_idx))
                store = make_system(args.base_system, model_name=args.model_name)
                res = evaluate_sample_with_sleep(
                    sample, args.base_system, store, variant,
                    judge_fn=judge_fn, k=args.k,
                    sleep_every=args.sleep_every, tau=args.tau,
                    tau_prune=args.tau_prune,
                    protect_w_imp=args.protect_w_imp,
                    credit_transfer=args.credit_transfer,
                    matched_ratios=matched.get(sid),
                    # per-variant audit file — certified's rejected ops and
                    # uncertified's unconditionally-applied ops must not share
                    # a file, or `applied` flags from the two runs get
                    # conflated when post-hoc filtering only by op_type/score.
                    audit_path=f"{audit_stem}_{variant}_audit.jsonl"
                    if wants_audit else None,
                    max_clusters=args.max_clusters,
                    max_prune_checks=prune_cap,
                    retrieval_coverage_min=args.retrieval_coverage_min,
                    use_co_cite_clusters=not args.no_co_cite_clusters,
                    track_correlation=args.track_correlation,
                    max_merge_ratio=args.max_merge_ratio,
                    merge_generate_fn=merge_fn,
                    max_supersede_checks=args.max_supersede_checks,
                    tau_supersede=args.tau_supersede,
                    target_ratio=args.target_ratio,
                    dump_perqa=args.dump_perqa,
                    retrieval_guard=args.retrieval_guard,
                    tau_lex=args.tau_lex,
                    tau_anchor=args.tau_anchor,
                    tau_query=args.tau_query,
                    min_probes=args.min_probes,
                    max_probes=args.max_probes,
                    merge_regeneration_attempts=args.merge_regeneration_attempts,
                    log_counterfactual_probes=args.log_counterfactual_probes,
                )
                per_sample.append(res)
                eps = res["eps_hat"]
                print(f"  [{s_idx+1}/{len(samples)}] {sid}: rec5={res.get('rec5', 0):.4f} "
                      f"tokens={res['final_tokens']} "
                      f"eps={'—' if eps is None else f'{eps:.4f}'} "
                      f"merges={res['applied_merges']} prunes={res['applied_prunes']}")

        if variant in ("certified", "certified_budget"):
            # token-match anchor for the heuristic prune baselines; first
            # certified-family variant to run wins (order puts certified first)
            for i, sample in enumerate(samples):
                sid = str(getattr(sample, "sample_id", i))
                matched.setdefault(sid, per_sample[i]["achieved_ratios"])

        # aggregate across samples (QA-weighted for retrieval; op-summed)
        n_total = sum(r["n"] for r in per_sample) or 1
        agg = {
            "n": n_total,
            "rec5": sum(r.get("rec5", 0) * r["n"] for r in per_sample) / n_total,
            "full_ev5": sum(r.get("full_ev5", 0) * r["n"] for r in per_sample) / n_total,
            "ctx_recall": sum(r.get("ctx_recall", 0) * r["n"] for r in per_sample) / n_total,
            "tokens5": sum(r.get("tokens5", 0) * r["n"] for r in per_sample) / n_total,
            "final_tokens_total": sum(r["final_tokens"] for r in per_sample),
            "eps_hat": (sum(r["eps_probes"] * (r["eps_hat"] or 0.0) for r in per_sample)
                        / max(1, sum(r["eps_probes"] for r in per_sample))),
            "eps_probes": sum(r["eps_probes"] for r in per_sample),
            "rho_hat_mean": float(np.mean([r["rho_hat"] for r in per_sample])),
            "applied_merges": sum(r["applied_merges"] for r in per_sample),
            "applied_prunes": sum(r["applied_prunes"] for r in per_sample),
            "applied_supersedes": sum(r.get("applied_supersedes", 0) for r in per_sample),
            "stale_served_total": sum(r.get("stale_served_total", 0) for r in per_sample),
            "per_sample": per_sample,
        }
        if args.dump_perqa:
            perqa_path = f"{audit_stem}_{variant}_perqa.jsonl"
            with open(perqa_path, "w") as f:
                for r in per_sample:
                    for row in r.pop("perqa", []):
                        f.write(json.dumps(row) + "\n")
            print(f"  per-QA rows -> {perqa_path} (significance.py input)")
        results[variant] = agg
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"  {variant} OVERALL: rec5={agg['rec5']:.4f} "
              f"eps_hat={agg['eps_hat']:.4f} ({agg['eps_probes']} probes) "
              f"tokens={agg['final_tokens_total']} "
              f"merges={agg['applied_merges']} prunes={agg['applied_prunes']} "
              f"supersedes={agg['applied_supersedes']} "
              f"stale_served={agg['stale_served_total']}")

    print(f"\n## Consolidation results (rec@5 / eps_hat / tokens)\n")
    print(f"{'variant':<20}{'rec5':>8}{'eps_hat':>9}{'tokens':>10}"
          f"{'merges':>8}{'prunes':>8}{'supersd':>8}{'stale':>7}")
    for v in order:
        a = results[v]
        print(f"{v:<20}{a['rec5']:>8.4f}{a['eps_hat']:>9.4f}"
              f"{a['final_tokens_total']:>10}{a['applied_merges']:>8}"
              f"{a['applied_prunes']:>8}{a['applied_supersedes']:>8}"
              f"{a['stale_served_total']:>7}")
    print(f"\nSaved to {out_path}; audit: {audit_stem}_<variant>_audit.jsonl")


if __name__ == "__main__":
    main()
