#!/bin/bash
# t_embed_mode ablation @ 128px, arm 1/3: adaln (DiT-style, per-block modulation MLP).
# Fast variant of ablation_t_emb_adaln.sh: img_size=128 (instead of 224) for
# quicker iteration, cfg=3.0 + raw (non-EMA) periodic sampling (matching the
# older exp5 protocol) instead of cfg=1.5/EMA2.
# patch_size=8 (not the S_16 default of 16): num_patches=(128/8)^2=256 for
# our own self-attention. Note the ChexWorld cross-attention context is
# unaffected by this -- it's sized by CHEXWORLD_PATCH_SIZE=16 (fixed) and
# img_size, giving (128/16)^2=64 context tokens regardless of our patch_size.
# batch_size=128: unlike the 224/patch14 fixed-token resolution ablation
# (which OOM'd at batch=128 with 196 cross-attn context tokens from
# ChexWorld's 224/16 grid), this config's cross-attn context is only 64
# tokens (128/16), which keeps peak memory low enough to fit batch=128.
# Usage: bash scripts/ablation_t_emb_adaln_128.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REPO_ROOT}/src/rct/train.py" \
  --data_dir           /mnt/nvme1/mimic-cxr-jpg \
  --nih_dir            /mnt/nvme1/nih-chest-x-ray-14 \
  --chexpert_dir       /mnt/nvme1/chexpert \
  --output_dir         "${REPO_ROOT}/output/ablation_t_emb_adaln_128_chexworld" \
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
  --t_embed_mode       adaln \
  --long_skip_connection \
  --no_final_conv \
  --wandb_project      ablation-t_emb-chexworld-128 \
  --wandb_run_name     "t_emb_adaln_128" \
  --compile \
  "$@"
