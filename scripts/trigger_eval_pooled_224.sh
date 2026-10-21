#!/bin/bash
# Wait for the pooled 224 300k run to finish, then run the final eval pipeline.
# Launch detached: nohup bash scripts/trigger_eval_pooled_224.sh &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/output/rct_pooled_224_chexworld"
CKPT="${MODEL_DIR}/checkpoints/stepstep=0300000.ckpt"

echo "[trigger] waiting for 300k checkpoint: $CKPT"
until [ -f "$CKPT" ]; do sleep 120; done
echo "[trigger] 300k present at $(date); waiting for train.py to exit..."
while pgrep -f "train.py.*rct_pooled_224_chexworld" >/dev/null 2>&1; do sleep 60; done
echo "[trigger] training gone at $(date); starting eval"

sleep 30
bash "${SCRIPT_DIR}/eval_rct_final.sh" "$MODEL_DIR" 1.5 ema2
echo "[trigger] pooled eval finished at $(date)"
