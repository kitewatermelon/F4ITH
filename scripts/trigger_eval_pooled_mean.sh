#!/bin/bash
# After the mean-pool pooled 300k run finishes, eval it on MIMIC / NIH(full) /
# CheXpert. Deletes the stale CLS-model fid_eval dirs first (their gen + feature
# caches are from the old CLS checkpoint; caches are keyed by filename so they
# would be silently reused for the new gen otherwise). NIH sharded across 2 GPUs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/output/rct_pooled_224_chexworld"
CKPT="${MODEL_DIR}/checkpoints/stepstep=0300000.ckpt"
LOG="${REPO_ROOT}/output/eval_logs"
N=30000

echo "[pm] waiting for mean-pool 300k checkpoint: $CKPT"
until [ -f "$CKPT" ]; do sleep 120; done
echo "[pm] 300k present at $(date); waiting for train.py to exit..."
while pgrep -f "train.py.*rct_pooled_224_chexworld" >/dev/null 2>&1; do sleep 60; done
echo "[pm] training gone; deleting stale CLS eval dirs and re-evaluating"

# drop stale CLS-model results + feature caches so nothing is reused
for ds in "" _nih _chexpert; do rm -rf "${MODEL_DIR}/fid_eval${ds}"; done

run_ds() {  # <name> <dataset> <save_subdir>
  local name="$1" ds="$2" sp="${MODEL_DIR}/$3"
  echo "[pm] === ${name}: sharded gen ==="
  CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CKPT" \
    --save_path "$sp" --dataset "$ds" --shard 0/2 --cfg 1.5 --ema ema2 --num_images "$N" --batch_size 32 --gpu 0 \
    > "${LOG}/gen_pooledmean_${name}_s0.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct --ckpt_path "$CKPT" \
    --save_path "$sp" --dataset "$ds" --shard 1/2 --cfg 1.5 --ema ema2 --num_images "$N" --batch_size 32 --gpu 0 \
    > "${LOG}/gen_pooledmean_${name}_s1.log" 2>&1 &
  wait
  echo "[pm] ${name} gen done ($(ls ${sp}/gen 2>/dev/null | wc -l)); metrics"
  python "${SCRIPT_DIR}/quality_metrics.py" --real_dir "${sp}/real" --gen_dir "${sp}/gen" \
    --out "${sp}/metrics.txt" > "${LOG}/metrics_pooledmean_${name}.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" --real_dir "${sp}/real" --gen_dir "${sp}/gen" \
    --gpu 0 --out "${sp}/distance_metrics.txt" > "${LOG}/distance_pooledmean_${name}.log" 2>&1
  echo "[pm] ${name}: $(grep '^FID' ${sp}/metrics.txt 2>/dev/null) | $(grep '^Rank_gen2cond_cosine' ${sp}/distance_metrics.txt 2>/dev/null)"
}

run_ds mimic    mimic    fid_eval
run_ds chexpert chexpert fid_eval_chexpert
run_ds nih      nih      fid_eval_nih
echo "[pm] ALL DONE at $(date)"
