#!/bin/bash
# RCT training: t_embed_mode=token, 224px / patch 14 (16x16 = 256 tokens),
# RAD-DINO patch_only conditioning, 300k steps. Identical config to the
# main-result CheXWorld run (train_token_224_p14_chexworld.sh) with the
# backbone swapped to RAD-DINO. Periodic eval (sample grids) every 10k steps
# uses the CFG x EMA optimum found by the cfg sweep.
#
# Usage: bash scripts/train_token_224_p14_raddino.sh <cfg> <sample_ema> [extra args...]
#   e.g. bash scripts/train_token_224_p14_raddino.sh 1.5 ema2

set -euo pipefail

CFG="${1:?usage: train_token_224_p14_raddino.sh <cfg> <sample_ema>}"
SAMPLE_EMA="${2:?usage: train_token_224_p14_raddino.sh <cfg> <sample_ema>}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/src/rct/train.py" \
  --data_dir           /mnt/nvme1/mimic-cxr-jpg \
  --nih_dir            /mnt/nvme1/nih-chest-x-ray-14 \
  --chexpert_dir       /mnt/nvme1/chexpert \
  --output_dir         "${REPO_ROOT}/output/rct_token_224_p14_raddino" \
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
  --conditioning_mode  patch_only \
  --backbone           raddino \
  --t_embed_mode       token \
  --long_skip_connection \
  --no_final_conv \
  --wandb_project      rct-token-224-raddino \
  --wandb_run_name     "token_224_p14_300k_cfg${CFG}_${SAMPLE_EMA}" \
  --compile \
  "$@"
