#!/bin/bash
# Companion to run_token_cfg_ema_sweep.sh: once all 8 gen sets are complete,
# compute FID (GPU) for each into metrics_fid.txt. Overlaps with the sweep's
# FRD (CPU-bound radiomics) stage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/output"
REAL_DIR="${OUTPUT_ROOT}/rcdm_chexworld_cxr/fid_eval/real"
SWEEP_ROOT="${OUTPUT_ROOT}/token_cfg_ema_sweep"
LOG_DIR="${OUTPUT_ROOT}/eval_logs"
N=3403

CONFIGS=()
for ema in ema1 ema2; do for cfg in 1 1.5 3 5; do CONFIGS+=("cfg${cfg}_${ema}"); done; done

# wait for the last-generated set to finish
last="${SWEEP_ROOT}/cfg5_ema2/gen"
echo "waiting for generation to finish ($last == $N)..."
until [ "$(ls "$last" 2>/dev/null | wc -l)" -ge "$N" ]; do sleep 30; done
echo "all generation done -- computing FID"

for tag in "${CONFIGS[@]}"; do
  gen="${SWEEP_ROOT}/${tag}/gen"
  [ -d "$gen" ] || continue
  CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/quality_metrics.py" \
    --real_dir "$REAL_DIR" --gen_dir "$gen" \
    --metrics fid --out "${SWEEP_ROOT}/${tag}/metrics_fid.txt" \
    > "${LOG_DIR}/fidonly_token_${tag}.log" 2>&1
  echo "  ${tag}: $(grep '^FID' "${SWEEP_ROOT}/${tag}/metrics_fid.txt")"
done
echo "FID sweep done"
