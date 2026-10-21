#!/bin/bash
# Two-stage CFG x EMA sweep on the full MIMIC-CXR test set, split over 2 GPUs.
#
#   stage-a  ema1 / ema2 / raw at the trained cfg=3.0   (15 configs)
#   stage-b  CFG grid at each model's best EMA variant  (25 configs)
#
# Usage: bash scripts/run_sweep_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_ROOT="${CACHE_ROOT:-/mnt/nvme1/rct_sweep}"
LOG_DIR="${REPO_ROOT}/output/sweep_logs"
SWEEP="${SCRIPT_DIR}/sweep_cfg_ema.py"

mkdir -p "$LOG_DIR" "$CACHE_ROOT"

# exp5 is 74M vs 42M for the rest, so it gets a lighter share of the split.
GPU0_MODELS="exp1_baseline exp2_no_conv exp3_no_skip"
GPU1_MODELS="exp4_no_conv_no_skip exp5_h512_no_conv"

echo "[0/3] Preparing full MIMIC test reference set..."
python "$SWEEP" prepare --cache_root "$CACHE_ROOT" 2>&1 | tail -3

echo ""
echo "[1/3] Stage A — EMA variant sweep (ema1 / ema2 / raw) @ cfg=3.0"
CUDA_VISIBLE_DEVICES=0 python "$SWEEP" stage-a --cache_root "$CACHE_ROOT" --gpu 0 \
  --models $GPU0_MODELS > "${LOG_DIR}/stage_a_gpu0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python "$SWEEP" stage-a --cache_root "$CACHE_ROOT" --gpu 1 \
  --models $GPU1_MODELS > "${LOG_DIR}/stage_a_gpu1.log" 2>&1 &
P1=$!
wait $P0 || { echo "stage-a GPU0 failed"; exit 1; }
wait $P1 || { echo "stage-a GPU1 failed"; exit 1; }
echo "  ✓ stage A done"

# Pick each model's best EMA variant, then use the majority choice for stage B
# (stage B sweeps one EMA variant per model; --ema_choice is per-invocation).
BEST_EMA=$(python - "$CACHE_ROOT" <<'PY'
import json, sys, glob, collections
merged = {}
for p in glob.glob(f"{sys.argv[1]}/sweep_a_gpu*.json"):
    merged.update(json.load(open(p)))
per_model = {}
for k, v in merged.items():
    m, e, c = k.split("|")
    per_model.setdefault(m, {})[e] = v
votes = collections.Counter(min(d, key=d.get) for d in per_model.values())
print(votes.most_common(1)[0][0] if votes else "ema1")
PY
)
echo "  best EMA variant (majority across models): $BEST_EMA"

echo ""
echo "[2/3] Stage B — CFG sweep @ ${BEST_EMA}"
CUDA_VISIBLE_DEVICES=0 python "$SWEEP" stage-b --cache_root "$CACHE_ROOT" --gpu 0 \
  --models $GPU0_MODELS --ema_choice "$BEST_EMA" > "${LOG_DIR}/stage_b_gpu0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python "$SWEEP" stage-b --cache_root "$CACHE_ROOT" --gpu 1 \
  --models $GPU1_MODELS --ema_choice "$BEST_EMA" > "${LOG_DIR}/stage_b_gpu1.log" 2>&1 &
P1=$!
wait $P0 || { echo "stage-b GPU0 failed"; exit 1; }
wait $P1 || { echo "stage-b GPU1 failed"; exit 1; }
echo "  ✓ stage B done"

echo ""
echo "[3/3] Report"
python "$SWEEP" report --cache_root "$CACHE_ROOT" 2>&1 | tee "${LOG_DIR}/report.log"
