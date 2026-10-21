#!/bin/bash
# Wait for the token 224/p14 300k run to finish, then run the final eval pipeline.
# Meant to be launched detached (nohup ... &) so it survives across sessions.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CKPT="${REPO_ROOT}/output/rct_token_224_p14_chexworld/checkpoints/stepstep=0300000.ckpt"

echo "[trigger] waiting for 300k checkpoint: $CKPT"
until [ -f "$CKPT" ]; do sleep 120; done
echo "[trigger] 300k checkpoint present at $(date)"

# wait for the training process to release the GPUs
echo "[trigger] waiting for train.py to exit..."
while pgrep -f "train.py.*rct_token_224_p14_chexworld" >/dev/null 2>&1; do sleep 60; done
echo "[trigger] training process gone at $(date); starting eval"

sleep 30  # small settle margin
bash "${SCRIPT_DIR}/eval_token_224_final.sh"
echo "[trigger] eval pipeline finished at $(date)"
