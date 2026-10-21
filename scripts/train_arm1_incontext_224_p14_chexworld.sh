#!/bin/bash
# arm1 (best conditioning-fidelity arm of the in-context ablation) scaled up:
#   cls_only + adaLN-Zero + JiT in-context tokens (16 @ block 4), CheXWorld
#   pooled-mean conditioning, long skip, 224px / patch 14 (16x16 = 256 tokens,
#   same token count as the 128/p8 ablation), 300k steps. Periodic sample-grid
#   eval every 10k uses the given CFG x EMA.
#
# Usage: bash scripts/train_arm1_incontext_224_p14_chexworld.sh [cfg] [sample_ema] [extra args...]
#   e.g. bash scripts/train_arm1_incontext_224_p14_chexworld.sh 1.5 ema2

set -euo pipefail

CFG="${1:-1.5}"
SAMPLE_EMA="${2:-ema2}"
[ "$#" -ge 1 ] && shift
[ "$#" -ge 1 ] && shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/src/rct/train.py" \
  --data_dir           /mnt/nvme1/mimic-cxr-jpg \
  --nih_dir            /mnt/nvme1/nih-chest-x-ray-14 \
  --chexpert_dir       /mnt/nvme1/chexpert \
  --output_dir         "${REPO_ROOT}/output/arm1_incontext_224_p14_chexworld" \
  --model_variant      S_16 \
  --hidden_size        512 \
  --num_heads          8 \
  --img_size           224 \
  --patch_size         14 \
  --batch_size         128 \
  --num_workers        12 \
  --num_gpus           -1 \
  --lr                 5e-5 \
  --weight_decay       0.0 \
  --max_steps          300000 \
  --warmup_steps       10000 \
  --cond_drop_prob     0.1 \
  --cfg                "$CFG" \
  --sample_ema         "$SAMPLE_EMA" \
  --ode_steps          50 \
  --ode_method         heun \
  --interval_min       0.0 \
  --interval_max       1.0 \
  --save_every         10000 \
  --sample_every       10000 \
  --num_samples        8 \
  --ema_decay1         0.9999 \
  --ema_decay2         0.9996 \
  --P_mean             -0.8 \
  --P_std              0.8 \
  --noise_scale        1.0 \
  --t_eps              0.05 \
  --log_every          50 \
  --seed               42 \
  --conditioning_mode  cls_only \
  --pooled_cond_source mean \
  --backbone           chexworld \
  --t_embed_mode       adaln \
  --in_context_len     16 \
  --in_context_start   4 \
  --long_skip_connection \
  --no_final_conv \
  --wandb_project      rct-arm1-incontext-224-chexworld \
  --wandb_run_name     "arm1_incontext_224_p14_300k_cfg${CFG}_${SAMPLE_EMA}" \
  --compile \
  "$@"
