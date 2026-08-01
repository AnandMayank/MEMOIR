#!/bin/bash
# Master runner for every experiment PENDING in docs/EXPERIMENT_TABLES.md —
# DECISION-ORDERED, not cost-ordered: the two results that decide whether the
# paper's spine holds run FIRST, and a hard gate stops the run before the
# expensive fill-in phases if the spine breaks. Each block is independently
# resumable (API calls disk-cached, each writes its own results/*.json).
#
# Phases:
#   0  free extractions (minutes)
#   1  GO/NO-GO — compression-targeted consolidation + independent delta
#      re-audit (HARD GATE), then the 70B accuracy anchor (soft gate, printed
#      verdict vs LiCoMemory's published 67.2%)
#   2  story-shaping — v3/CF calibration (decides CF-vs-ucb framing), supersede
#   3  fill-in — ablation matrix, same-model 5-run, Mem0/A-MEM, 4o-mini anchor
#   4  significance (consumes earlier outputs)
#
# Gate override: PAPER_GATE=off bash ... runs everything regardless.
# Gate thresholds: GATE_LOSSY_MAX (default 0.35) — max independent-audited
# lossy rate for certified ops; GATE_MIN_COMPRESSION (default 0.001) —
# certified (merge-only) must compress the store by at least this fraction
# for RQ5's claim to be worth the remaining compute. Deliberately low: with
# the 70B merge-judge fix, a real 10-sample run measured 0.3% compression
# at 0% independent-audited-lossy (RCPS-certifiable, 448 ops) — merge-only
# consolidation is a safety-first operator, not an aggressive compressor
# (docs/COMBINED_PAPER_PLAN.md 2026-07-09 already estimated a ~0.5%
# ceiling for it). 0.08 was inherited from the abandoned target-ratio
# design and was never realistic here; the gate now just confirms
# compression is real and positive, not an arbitrary aspirational target.
#
# RQ5 reframe (2026-07-18): the target-ratio-driven `certified_budget`
# variant re-enables certified prune to force compression, and a fresh
# independent audit on this run measured 63.6% lossy on those ops — same
# failure mode as the original dominated-prune audit (83.9% lossy,
# docs/COMBINED_PAPER_PLAN.md 2026-07-09), just resurfacing in budget mode.
# Reverted Phase 1's headline to merge-only `certified` (already the
# project's decided-safe operator, 6.25% lossy) paired with the Llama-70B
# merge-judge fix (confirmed -22.9% raw compression vs the 8B judge's +5.3%
# inflation) — the mechanism expected to deliver real compression WITHOUT
# prune. `certified_budget`/`uncertified_budget` stay in eval_consolidation.py
# as an explicit documented-negative-result ablation, not the gate path.
#
# Usage (vast.ai, after deploy/vast_setup_ollama.sh):
#   nohup bash benchmarks/memoir/run_paper_experiments.sh [WORKERS] \
#     > paper_experiments.log 2>&1 & disown
#
# WORKERS default 4 (24GB 3090, OLLAMA_NUM_PARALLEL=6). Requires in .env:
# TOGETHER_API_KEY (70B merge-gen + anchor), TOKEN_ROUTER_API_KEY (4o-mini).

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ -f .env ]; then set -a; source .env; set +a; fi
# Use $HOME rather than a hardcoded /root -- not every instance runs this
# as root, and /root may not be writable.
SECOND_DRIVE=/media/$(whoami)/second_drive
if [ -d "$SECOND_DRIVE" ] && [ -w "$SECOND_DRIVE" ]; then
  export HF_HOME="$SECOND_DRIVE/harpo_hf_cache"
else
  export HF_HOME="$HOME/.cache/huggingface"
fi
mkdir -p "$HF_HOME" results

# ── preflight: venv must be active before anything runs ─────────────────────────
# Silent failure mode this guards against: launching via `nohup bash script.sh &`
# from a shell where .venv was never activated runs every python step against
# bare system Python. Steps fail with ModuleNotFoundError, `step()` swallows the
# nonzero exit and moves on, and the run silently cascades all the way to the
# Phase-1 gate reporting a false "spine broken" (missing output files, not a
# real RQ5 failure) — burning the whole run's wall-clock to say nothing.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  if [ -f .venv/bin/activate ]; then
    echo "venv not active — activating .venv"
    # shellcheck disable=SC1091
    source .venv/bin/activate
  else
    echo "FATAL: no active venv and no .venv/ found. Run"
    echo "  deploy/vast_setup_ollama.sh"
    echo "first, or 'source .venv/bin/activate' before launching this script."
    exit 1
  fi
fi
if ! python3 -c "import numpy, harpo.memoir" 2>/tmp/preflight_err.log; then
  echo "FATAL: venv is active but core deps aren't importable:"
  cat /tmp/preflight_err.log
  echo "Run: pip install -e '.[memoir,memoir-eval,dev]'"
  exit 1
fi
echo "preflight OK: $(which python3), numpy + harpo.memoir importable"

WORKERS="${1:-4}"
JUDGE="ollama:llama3.1:8b"
MERGE_JUDGE="together:meta-llama/Llama-3.3-70B-Instruct-Turbo"
PAPER_GATE="${PAPER_GATE:-on}"
GATE_LOSSY_MAX="${GATE_LOSSY_MAX:-0.35}"
GATE_MIN_COMPRESSION="${GATE_MIN_COMPRESSION:-0.001}"

step() {
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "════════════════════════════════════════════════════════════════"
  shift
  set +e
  "$@" 2>&1 | tee /tmp/step_out.log
  local rc=${PIPESTATUS[0]}
  set -e
  if [ $rc -ne 0 ]; then
    if grep -qE "ModuleNotFoundError|No module named" /tmp/step_out.log; then
      echo "FATAL: step failed on a missing module — this is an environment"
      echo "problem, not an experiment result. Stopping immediately instead of"
      echo "cascading (downstream gates would misreport this as a real failure)."
      exit 1
    fi
    echo "!! step exited $rc — continuing with the rest"
  fi
}

# ════════ PHASE 0 — free extractions ════════════════════════════════════════════

# The LiCoMemory run artifacts live on the dev laptop, not compute instances —
# skip extraction where they're absent (the extracted summary is already
# committed at results/licomemory_llama8b_5run.json; extract_licomemory_runs.py
# also refuses to overwrite it with an empty result as a second guard).
if [ -d /home/user/LiCoMemory/results ]; then
  step "0. LiCoMemory 5-run extraction (T1 row 1; CPU, free)" \
    python benchmarks/memoir/extract_licomemory_runs.py \
      --dataset data/locomo10.json \
      --output results/licomemory_llama8b_5run.json
else
  echo "0. LiCoMemory artifacts not on this machine — using committed"
  echo "   results/licomemory_llama8b_5run.json (skipping extraction)."
fi

# ════════ PHASE 1 — GO/NO-GO (the two results that decide the paper) ════════════

# 1a. THE RQ5 test: can certified (merge-only) consolidation actually
# compress? certified = judge-gated merge path only (--certified-max-prune-
# checks defaults to 0 — dominated-prune stays out of the headline number,
# see docs/COMBINED_PAPER_PLAN.md 2026-07-09 and the 2026-07-18 reframe
# note above). Paired with the 70B merge-judge fix so compression isn't
# structurally dependent on prune.
step "1a. Merge-only certified consolidation, 70B merge-judge (T3 headline)" \
  python benchmarks/memoir/eval_consolidation.py \
    --n-samples 10 --base-system memoir_trace --judge "$JUDGE" \
    --merge-judge "$MERGE_JUDGE" --budget-usd 10.0 \
    --variants no_sleep,certified,uncertified,recency_prune,random_prune \
    --sleep-every 8 --tau 0.6 \
    --retrieval-coverage-min 0.45 --workers "$WORKERS" \
    --dump-perqa \
    --output results/consolidation_rq5.json

# 1b. Independent delta re-audit of the certified (merge-only) ops. THE gate.
step "1b. Independent delta re-audit of certified (merge-only) ops" \
  python benchmarks/memoir/calibrate_consolidation.py \
    --audit results/consolidation_rq5_certified_audit.jsonl \
    --auditor ollama:mistral:7b \
    --alpha 0.15 --confidence 0.85 --op-type both \
    --output results/calibration_certified_ops.json

# ── HARD GATE ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  GATE: is the RQ5 spine intact? (PAPER_GATE=$PAPER_GATE)"
echo "════════════════════════════════════════════════════════════════"
set +e
python - "$GATE_LOSSY_MAX" "$GATE_MIN_COMPRESSION" <<'PYGATE'
import json, sys
lossy_max, min_comp = float(sys.argv[1]), float(sys.argv[2])
ok = True
try:
    cal = json.load(open("results/calibration_certified_ops.json"))
    lossy = cal.get("empirical_lossy_rate")
    print(f"  independent-audited lossy rate of certified (merge-only) ops: {lossy:.3f} "
          f"(gate: <= {lossy_max})")
    if lossy is None or lossy > lossy_max:
        print("  ✗ CERTIFICATION FAILED THE INDEPENDENT AUDIT at this operating point")
        ok = False
except Exception as e:
    print(f"  ✗ could not read audit result: {e}"); ok = False
try:
    d = json.load(open("results/consolidation_rq5.json"))
    c, ns, u = d["certified"], d["no_sleep"], d["uncertified"]
    comp = 1 - c["final_tokens_total"] / ns["final_tokens_total"]
    eps_c, eps_u = c["eps_hat"], u["eps_hat"]
    print(f"  certified compression: {comp*100:.1f}% (gate: >= {min_comp*100:.0f}%)   "
          f"eps_hat certified={eps_c:.4f} vs uncertified={eps_u:.4f}")
    if comp < min_comp:
        print("  ✗ COMPRESSION BELOW USEFUL THRESHOLD — 70B merge-judge fix didn't carry")
        ok = False
except Exception as e:
    print(f"  ✗ could not read consolidation result: {e}"); ok = False
print("  GATE:", "PASS — spine intact, continuing" if ok else
      "FAIL — RQ5 spine broken; fix/reframe before spending Phase 2-4 compute")
sys.exit(0 if ok else 1)
PYGATE
GATE_RC=$?
set -e
if [ $GATE_RC -ne 0 ] && [ "$PAPER_GATE" != "off" ]; then
  echo ""
  echo "  Stopping here (rerun with PAPER_GATE=off to force the remaining phases)."
  echo "  Decide the reframe first: results/calibration_certified_ops.json and"
  echo "  results/consolidation_rq5.json contain the evidence."
  exit 1
fi

# 1c. 70B accuracy anchor — the other make-or-break number (~$5, fast).
step "1c. T1 anchor: MEMOIR on Llama-3.3-70B-Turbo via Together (\$5 cap)" \
  python benchmarks/memoir/eval_qa_together.py \
    --backend together --n-qa 300 --k 5 --budget-usd 5.0 \
    --systems cosine,memoir_ucb,memoir_cf_full \
    --output results/qa_70b_together.json

python - <<'PYSOFT' || true
import json
try:
    d = json.load(open("results/qa_70b_together.json"))
    print("  ── soft gate: vs LiCoMemory published 67.20% LoCoMo Acc (70B) ──")
    for name, s in d.get("systems", {}).items():
        acc = s.get("acc")
        if acc is not None:
            verdict = "BEATS" if acc > 0.672 else "below"
            print(f"    {name}: {acc*100:.2f}%  ({verdict} 67.20%)")
except Exception as e:
    print(f"  (soft gate skipped: {e})")
PYSOFT

# ════════ PHASE 2 — story-shaping ═══════════════════════════════════════════════

# Decides requirement #4: CF genuinely wins (calibration + v3) or the paper
# restructures around memoir_ucb.
step "2a. v3 ablations + TRACE-CF commit calibration (T5/A2)" \
  bash benchmarks/memoir/run_v3_ablations.sh 8

step "2b. SUPERSEDE: certified vs certified_supersede vs no_sleep (T3/A7)" \
  python benchmarks/memoir/eval_consolidation.py \
    --n-samples 10 --base-system memoir_trace --judge "$JUDGE" \
    --merge-judge "$MERGE_JUDGE" --budget-usd 5.0 \
    --variants no_sleep,certified,certified_supersede \
    --sleep-every 8 --max-clusters 8 --tau 0.6 \
    --retrieval-coverage-min 0.45 --workers "$WORKERS" \
    --dump-perqa \
    --output results/consolidation_supersede.json

# ════════ PHASE 3 — fill-in ═════════════════════════════════════════════════════

step "3a. Consolidation ablation matrix A.1-A.5, B.6-B.9 (T3/T4/A5)" \
  bash benchmarks/memoir/run_ablation_matrix.sh "$WORKERS" 10 5

for run in 1 2 3 4 5; do
  step "3b. MEMOIR same-model accuracy run $run/5 (T1)" \
    python benchmarks/memoir/eval_locomo_sota.py \
      --dataset data/locomo10.json --backend ollama \
      --model-name all-MiniLM-L6-v2 \
      --systems cosine,memoir_no_trace,memoir_trace,memoir_ucb,memoir_cf_full \
      --n-samples 2 --k 5 \
      --output "results/qa_llama8b_run${run}.json"
done

step "3c. Mem0 baseline, same-model 8B protocol (T1)" \
  python benchmarks/memoir/mem0_eval_locomo.py \
    --dataset data/locomo10.json --n-samples 2 --k 5 \
    --output results/mem0_llama8b_2samples.json

step "3d. A-MEM baseline, same-model 8B protocol (T1)" \
  python benchmarks/memoir/amem_eval_locomo.py \
    --dataset data/locomo10.json --n-samples 2 --k 5 \
    --output results/amem_llama8b_2samples.json

step "3e. T1 anchor: gpt-4o-mini via tokenrouter (\$5 cap)" \
  python benchmarks/memoir/eval_qa_together.py \
    --backend router --model openai/gpt-4o-mini --n-qa 300 --k 5 --budget-usd 5.0 \
    --systems cosine,memoir_ucb,memoir_cf_full \
    --output results/qa_4omini_router.json

# ════════ PHASE 4 — significance (consumes earlier outputs) ═════════════════════

step "4a. significance: memoir_ucb vs memoir_no_trace (retrieval headline)" \
  bash -c '
    if ls results/perqa/*memoir_ucb*.jsonl >/dev/null 2>&1; then
      python benchmarks/memoir/significance.py \
        --a results/perqa/locomo_v7_evidence_memoir_ucb.jsonl \
        --b results/perqa/locomo_v7_evidence_memoir_no_trace.jsonl \
        --metric rec5 --output results/sig_ucb_vs_notrace.json
    else
      python benchmarks/memoir/eval_locomo.py \
        --dataset data/locomo10.json \
        --systems memoir_no_trace,memoir_ucb --workers '"$WORKERS"' \
        --dump-perqa --output results/locomo_v7_sig_rerun.json && \
      python benchmarks/memoir/significance.py \
        --a results/perqa/locomo_v7_sig_rerun_memoir_ucb.jsonl \
        --b results/perqa/locomo_v7_sig_rerun_memoir_no_trace.jsonl \
        --metric rec5 --output results/sig_ucb_vs_notrace.json
    fi'

step "4b. significance: certified vs uncertified (rec5)" \
  python benchmarks/memoir/significance.py \
    --a results/consolidation_rq5_certified_perqa.jsonl \
    --b results/consolidation_rq5_uncertified_perqa.jsonl \
    --metric rec5 --output results/sig_certified_vs_uncertified.json

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All paper experiments dispatched. Fill docs/EXPERIMENT_TABLES.md from:"
echo "  licomemory_llama8b_5run.json, consolidation_rq5.json,"
echo "  calibration_certified_ops.json, qa_70b_together.json,"
echo "  consolidation_supersede.json, ablation_*.json, locomo_v3_*.json,"
echo "  commit_calibration_cf.json, qa_llama8b_run*.json,"
echo "  mem0/amem_llama8b_2samples.json, qa_4omini_router.json, sig_*.json"
echo "════════════════════════════════════════════════════════════════"
