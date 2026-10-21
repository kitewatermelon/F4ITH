#!/bin/bash
# RCT training: POOLED conditioning (conditioning_mode=cls_only) at 224px,
# TorchXRayVision DenseNet-121 backbone. The CNN has no true CLS token, so
# pooled_cond_source=mean GAP-pools the 7x7 feature map (same as the distance
# metrics use). t_embed_mode=adaln is required: token mode leaves `c` unused so
# pooled conditioning would never enter the model.
# Mirrors train_pooled_224_chexworld.sh except for the backbone.
# 300k steps, seed 42. Periodic eval every 10k steps.
#
# Usage: bash scripts/train_pooled_224_xrv_densenet.sh [extra args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/src/rct/train.py" \
  --data_dir           /mnt/nvme1/mimic-cxr-jpg \
  --nih_dir            /mnt/nvme1/nih-chest-x-ray-14 \
  --chexpert_dir       /mnt/nvme1/chexpert \
  --output_dir         "${REPO_ROOT}/output/rct_pooled_224_xrv_densenet" \
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
  --cfg                1.5 \
  --sample_ema         ema2 \
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
  --backbone           xrv_densenet121 \
  --t_embed_mode       adaln \
  --long_skip_connection \
  --no_final_conv \
  --wandb_project      rct-pooled-224-xrv-densenet \
  --wandb_run_name     "pooled_224_meanpool_adaln_300k_cfg1.5_ema2" \
  --compile \
  "$@"
