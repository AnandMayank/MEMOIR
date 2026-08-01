#!/bin/bash
# Master ablation-matrix runner for the combined paper's prospective half
# (docs/COMBINED_PAPER_PLAN.md, "Ablation matrix" section). One command,
# sequences every priority row, each to its own results/*.json — safe to
# kick off on a vast.ai instance and walk away (`nohup ... & disown`, or
# just run it in a `screen`/`tmux` session).
#
# Usage:
#   bash benchmarks/memoir/run_ablation_matrix.sh [WORKERS] [N_SAMPLES] [SWEEP_N]
#
#   WORKERS    process-parallelism per variant (default 4 — tuned for a
#              24GB 3090 with OLLAMA_NUM_PARALLEL=6; see deploy/vast_setup_ollama.sh)
#   N_SAMPLES  LoCoMo samples for the core comparisons A.1-A.3, B.6-B.9 (default 10)
#   SWEEP_N    LoCoMo samples for the tau/sleep_every sweeps A.4-A.5 (default 5 —
#              these run 3-4x more configs, so a smaller n keeps total wall-clock
#              reasonable; bump to 10 if you have time/budget)
#
# Does NOT include: D.14 iterated-consolidation stress (needs a dedicated
# protocol wrapper, not yet built — see docs/COMBINED_PAPER_PLAN.md),
# D.15-17 (significance/judge-sensitivity/MemoryAgentBench — separate scripts).
# Continues past individual row failures (set +e per row) so one bad config
# doesn't lose the rest of an hours-long run.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Together needs TOGETHER_API_KEY exported -- .env isn't auto-sourced by
# plain shells. Load it if present so --merge-judge works out of the box.
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# vast.ai's default HF_HOME (often /workspace/.hf_home, network-backed) can
# throw "Stale file handle" mid-download on some instances -- force-redirect
# to local instance disk (unconditional: vast.ai images set HF_HOME to the
# problematic path by default, so a `:-` fallback wouldn't override it).
SECOND_DRIVE=/media/$(whoami)/second_drive
if [ -d "$SECOND_DRIVE" ] && [ -w "$SECOND_DRIVE" ]; then
  export HF_HOME="$SECOND_DRIVE/harpo_hf_cache"
else
  export HF_HOME="$HOME/.cache/huggingface"
fi
mkdir -p "$HF_HOME"

WORKERS="${1:-4}"
N_SAMPLES="${2:-10}"
SWEEP_N="${3:-5}"
JUDGE="ollama:llama3.1:8b"
MERGE_JUDGE="together:meta-llama/Llama-3.3-70B-Instruct-Turbo"
OUT="results"
mkdir -p "$OUT"

# Calibrated defaults from the go/no-go gate + RCPS calibration
# (docs/COMBINED_PAPER_PLAN.md): tau=0.6, retrieval_coverage_min=0.45,
# certified runs merge-only (prune failed independent audit at 83.9% lossy).
# merge-judge=Llama-70B: measured -22.9% net compression vs llama3.1:8b's
# +5.3% inflation (results/consolidation_llama70b_merge_test.json) --
# certification stays on the cheap local judge, only merge-TEXT GENERATION
# uses the stronger (Together-billed, budget-capped, disk-cached) model.
BASE_ARGS=(--base-system memoir_trace --judge "$JUDGE" --merge-judge "$MERGE_JUDGE"
           --budget-usd 5.0 --sleep-every 8
           --max-clusters 8 --max-prune-checks 10 --tau 0.6
           --retrieval-coverage-min 0.45 --workers "$WORKERS")

run() {
  local name="$1"; shift
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  $name"
  echo "════════════════════════════════════════════════════════════════"
  set +e
  python benchmarks/memoir/eval_consolidation.py "$@"
  local rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "!! $name exited $rc — continuing with the rest of the matrix"
  fi
}

# ── A.1-A.2: core headline (no_sleep / certified / uncertified) ────────────────
run "A.1-A.2: no_sleep vs certified vs uncertified" \
  --n-samples "$N_SAMPLES" "${BASE_ARGS[@]}" \
  --variants no_sleep,certified,uncertified \
  --output "$OUT/ablation_A1_A2_core.json"

# ── A.3: token-matched baselines (recency/random already ratio-matched to
#         certified; summarize_all is unconditional, its own extreme point) ───
run "A.3: recency_prune vs random_prune (matched to certified's ratios)" \
  --n-samples "$N_SAMPLES" "${BASE_ARGS[@]}" \
  --variants certified,recency_prune,random_prune \
  --output "$OUT/ablation_A3_baselines.json"

run "A.3: summarize_all (MemGPT-style, unconditional)" \
  --n-samples "$N_SAMPLES" "${BASE_ARGS[@]}" \
  --variants no_sleep,summarize_all \
  --output "$OUT/ablation_A3_summarize_all.json"

# ── A.4: tau sweep (rate-distortion frontier at the certificate threshold) ─────
for tau in 0.4 0.6 0.8 1.0; do
  run "A.4: tau sweep — tau=$tau" \
    --n-samples "$SWEEP_N" "${BASE_ARGS[@]}" --tau "$tau" \
    --variants no_sleep,certified \
    --output "$OUT/ablation_A4_tau${tau}.json"
done

# ── A.5: sleep_every sweep (consolidation frequency sensitivity) ───────────────
for se in 4 8 16; do
  run "A.5: sleep_every sweep — sleep_every=$se" \
    --n-samples "$SWEEP_N" --base-system memoir_trace --judge "$JUDGE" \
    --merge-judge "$MERGE_JUDGE" --budget-usd 5.0 \
    --max-clusters 8 --max-prune-checks 10 --tau 0.6 \
    --retrieval-coverage-min 0.45 --workers "$WORKERS" \
    --sleep-every "$se" \
    --variants no_sleep,certified \
    --output "$OUT/ablation_A5_sleepevery${se}.json"
done

# ── B.6: credit protection on/off (Bridge 1) ────────────────────────────────────
run "B.6: credit protection ON (default, protect_w_imp=0.3)" \
  --n-samples "$N_SAMPLES" "${BASE_ARGS[@]}" --protect-w-imp 0.3 \
  --variants no_sleep,certified \
  --output "$OUT/ablation_B6_protect_on.json"

run "B.6: credit protection OFF (protect_w_imp=1.1, nothing ever protected)" \
  --n-samples "$N_SAMPLES" "${BASE_ARGS[@]}" --protect-w-imp 1.1 \
  --variants no_sleep,certified \
  --output "$OUT/ablation_B6_protect_off.json"

# ── B.7: credit transfer max vs zero (Bridge 2) ─────────────────────────────────
run "B.7: credit transfer = max (default)" \
  --n-samples "$N_SAMPLES" "${BASE_ARGS[@]}" --credit-transfer max \
  --variants no_sleep,certified \
  --output "$OUT/ablation_B7_transfer_max.json"

run "B.7: credit transfer = zero (merged item starts w_imp=0)" \
  --n-samples "$N_SAMPLES" "${BASE_ARGS[@]}" --credit-transfer zero \
  --variants no_sleep,certified \
  --output "$OUT/ablation_B7_transfer_zero.json"

# ── B.8: co-cite-proposed vs embedding-only clusters (Bridge 1) ────────────────
run "B.8: co-cite-seeded clusters (default)" \
  --n-samples "$N_SAMPLES" "${BASE_ARGS[@]}" \
  --variants no_sleep,certified \
  --output "$OUT/ablation_B8_cocite_on.json"

run "B.8: embedding-only clusters (--no-co-cite-clusters)" \
  --n-samples "$N_SAMPLES" "${BASE_ARGS[@]}" --no-co-cite-clusters \
  --variants no_sleep,certified \
  --output "$OUT/ablation_B8_cocite_off.json"

# ── B.9: attribution-correlation pre/post, certified vs uncertified ────────────
# THE signature figure: does uncertified consolidation corrupt the credit
# ledger (r_post << r_pre) while certified preserves it (r_post ~= r_pre)?
run "B.9: attribution correlation pre/post consolidation" \
  --n-samples "$N_SAMPLES" "${BASE_ARGS[@]}" --track-correlation \
  --variants certified,uncertified \
  --output "$OUT/ablation_B9_correlation.json"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Ablation matrix complete. Results in $OUT/ablation_*.json"
echo "════════════════════════════════════════════════════════════════"
