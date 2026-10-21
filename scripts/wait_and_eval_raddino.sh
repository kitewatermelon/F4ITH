#!/bin/bash
# Wait for rct_token_224_p14_raddino training to finish (300k steps),
# then run full evaluation pipeline.
set -euo pipefail

CKPT_DIR="/home/yspark/RCT/output/rct_token_224_p14_raddino/checkpoints"
TARGET_CKPT="${CKPT_DIR}/stepstep=0300000.ckpt"

echo "Waiting for training to finish (target: $TARGET_CKPT)..."
while [ ! -f "$TARGET_CKPT" ]; do
  LATEST=$(ls -t "$CKPT_DIR"/stepstep=*.ckpt 2>/dev/null | head -1)
  echo "  $(date '+%H:%M:%S') latest: $(basename "${LATEST:-none}")"
  sleep 300
done

# Wait a bit more for checkpoint to be fully written
sleep 30
echo "Training finished at $(date). Starting full evaluation..."

bash /home/yspark/RCT/scripts/eval_raddino_full.sh
