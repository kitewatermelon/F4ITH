#!/bin/bash
# In-context ablation, arm 1/3: pooled CheXWorld CLS injected via adaLN-Zero
# AND JiT-style in-context tokens (cls_only + t_embed_mode=adaln + in_context).
# CheXWorld 128x128 / patch 8 (16x16 = 256 tokens), 100k steps.
# Usage: bash scripts/ablation_incontext_arm1_adaln.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/src/rct/train.py" \
  --data_dir           /mnt/nvme1/mimic-cxr-jpg \
  --nih_dir            /mnt/nvme1/nih-chest-x-ray-14 \
  --chexpert_dir       /mnt/nvme1/chexpert \
  --output_dir         "${REPO_ROOT}/output/ablation_incontext_arm1_adaln" \
  --model_variant      S_16 \
  --hidden_size        512 \
  --num_heads          8 \
  --img_size           128 \
  --patch_size         8 \
  --batch_size         128 \
  --num_workers        12 \
  --num_gpus           -1 \
  --lr                 5e-5 \
  --weight_decay       0.0 \
  --max_steps          100000 \
  --warmup_steps       10000 \
  --cond_drop_prob     0.1 \
  --cfg                1.5 \
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
  --conditioning_mode  cls_only \
  --pooled_cond_source mean \
  --backbone           chexworld \
  --t_embed_mode       adaln \
  --in_context_len     16 \
  --in_context_start   4 \
  --long_skip_connection \
  --no_final_conv \
  --wandb_project      ablation-incontext-chexworld-128 \
  --wandb_run_name     "arm1_adaln_incontext" \
  --compile \
  "$@"
