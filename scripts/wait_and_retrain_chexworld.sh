#!/bin/bash
# Wait for rct_pooled_224_raddino training to finish (300k steps),
# then sequentially retrain CheXWorld spatial and pooled with the corrected
# encoder (faithful normalization + no synthetic CLS token).
#
# Run in background: nohup bash scripts/wait_and_retrain_chexworld.sh > logs/retrain_chexworld.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WAIT_CKPT="${REPO_ROOT}/output/rct_pooled_224_raddino/checkpoints/stepstep=0300000.ckpt"

echo "=========================================="
echo "Waiting for rct_pooled_224_raddino 300k checkpoint..."
echo "  target: $WAIT_CKPT"
echo "  start:  $(date)"
echo "=========================================="

while [ ! -f "$WAIT_CKPT" ]; do
  LATEST=$(ls -t "${REPO_ROOT}/output/rct_pooled_224_raddino/checkpoints"/stepstep=*.ckpt 2>/dev/null | head -1)
  echo "  $(date '+%H:%M:%S') latest pooled-raddino: $(basename "${LATEST:-none}")"
  sleep 300
done

echo ""
echo "rct_pooled_224_raddino training complete at $(date). Starting CheXWorld retraining..."

# ---- 1. CheXWorld spatial (token/patch_only) ----
echo ""
echo "=========================================="
echo "[1/2] CheXWorld spatial (token, patch_only, faithful norm)"
echo "  start: $(date)"
echo "=========================================="

bash "${SCRIPT_DIR}/train_token_224_p14_chexworld.sh" 1.5 ema2 \
  --output_dir "${REPO_ROOT}/output/rct_token_224_p14_chexworld_v2" \
  --wandb_run_name "token_224_p14_300k_cfg1.5_ema2_faithful" \
  --chexworld_compat faithful

echo ""
echo "[1/2] CheXWorld spatial done at $(date)"

# ---- 2. CheXWorld pooled (cls_only, mean pool) ----
echo ""
echo "=========================================="
echo "[2/2] CheXWorld pooled (cls_only, mean pool, faithful norm)"
echo "  start: $(date)"
echo "=========================================="

bash "${SCRIPT_DIR}/train_pooled_224_chexworld.sh" \
  --output_dir "${REPO_ROOT}/output/rct_pooled_224_chexworld_v2" \
  --wandb_run_name "pooled_224_meanpool_adaln_300k_cfg1.5_ema2_faithful" \
  --chexworld_compat faithful

echo ""
echo "=========================================="
echo "ALL DONE at $(date)"
echo "CheXWorld spatial v2: output/rct_token_224_p14_chexworld_v2"
echo "CheXWorld pooled v2:  output/rct_pooled_224_chexworld_v2"
echo "=========================================="
