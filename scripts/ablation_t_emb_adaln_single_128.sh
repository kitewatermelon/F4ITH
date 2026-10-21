#!/bin/bash
# t_embed_mode ablation @ 128px, arm 3/3: adaln_single (PixArt-alpha style,
# shared modulation MLP + per-block offset table).
# Same as ablation_t_emb_adaln_128.sh except t_embed_mode=adaln_single.
# Usage: bash scripts/ablation_t_emb_adaln_single_128.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/src/rct/train.py" \
  --data_dir           /mnt/nvme1/mimic-cxr-jpg \
  --nih_dir            /mnt/nvme1/nih-chest-x-ray-14 \
  --chexpert_dir       /mnt/nvme1/chexpert \
  --output_dir         "${REPO_ROOT}/output/ablation_t_emb_adaln_single_128_chexworld" \
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
  --cfg                3.0 \
  --sample_ema         raw \
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
  --conditioning_mode  patch_only \
  --backbone           chexworld \
  --t_embed_mode       adaln_single \
  --long_skip_connection \
  --no_final_conv \
  --wandb_project      ablation-t_emb-chexworld-128 \
  --wandb_run_name     "t_emb_adaln_single_128" \
  --compile \
  "$@"
