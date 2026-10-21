#!/bin/bash
# CFG sweep on NIH + CheXpert for pooled & token models.
# cfg={1, 2, 2.5, 3} (cfg=1.5 already in fid_eval_{nih,chexpert}).
# Distance metrics only (CosSim, Rank, MRR) + CosSim_patch. No FID/FRD.
# GPU 0 only (GPU 1 occupied).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${REPO_ROOT}/output/main_result"
LOG="${REPO_ROOT}/output/logs/cfg_sweep_cross_dataset"
mkdir -p "$LOG"

CFGS=(1 2 2.5 3)

run_one() {
  local label="$1" model_dir="$2" ckpt="$3" ds="$4" num_images="$5"
  local base="${model_dir}/fid_eval_${ds}"
  local real="${base}/real"
  local sweep="${model_dir}/cfg_sweep_${ds}"
  mkdir -p "$sweep"

  echo "=========================================="
  echo "${label} / ${ds} CFG sweep (ema2, n=${num_images})"
  echo "=========================================="

  for cfg in "${CFGS[@]}"; do
    tag="cfg${cfg}_ema2"
    sp="${sweep}/${tag}"
    if [ -f "${sp}/distance_metrics.txt" ] && [ -f "${sp}/cossim_patch.txt" ]; then
      echo "[sweep] ${label}/${ds} ${tag} already done — skipping"
      continue
    fi
    echo "[sweep] ${label}/${ds} ${tag}: gen at $(date)"
    CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_generation.py" --model_type rct \
      --ckpt_path "$ckpt" --save_path "$sp" --real_from_dir "$real" \
      --cfg "$cfg" --ema ema2 --num_images "$num_images" --batch_size 512 --gpu 0 \
      > "${LOG}/gen_${label}_${ds}_${tag}.log" 2>&1
    ng=$(ls "${sp}/gen" 2>/dev/null | wc -l)
    echo "[sweep] ${label}/${ds} ${tag}: gen done (${ng} imgs); metrics"
    [ "$ng" -eq 0 ] && { echo "  no gen, skipping"; continue; }

    CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/distance_metrics.py" --real_dir "$real" --gen_dir "${sp}/gen" \
      --gpu 0 --out "${sp}/distance_metrics.txt" > "${LOG}/dist_${label}_${ds}_${tag}.log" 2>&1

    CUDA_VISIBLE_DEVICES=0 python -c "
import sys, os, torch as th, torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'baseline', 'RCDM'))
from guided_diffusion_rcdm.get_ssl_models import get_model
from torch.utils.data import Dataset, DataLoader
class PairDS(Dataset):
    def __init__(self, rd, gd, files, rs=224):
        self.rd, self.gd, self.files, self.rs = rd, gd, files, rs
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        f = self.files[i]
        def ld(d):
            t = TF.to_tensor(Image.open(os.path.join(d, f)).convert('RGB'))
            if t.shape[-1] != self.rs: t = TF.resize(t, [self.rs, self.rs], antialias=True)
            return t * 2.0 - 1.0
        return ld(self.rd), ld(self.gd)
rd, gd = '${real}', '${sp}/gen'
rf = set(f for f in os.listdir(rd) if f.endswith('.png'))
gf = set(f for f in os.listdir(gd) if f.endswith('.png'))
files = sorted(rf & gf)
enc = get_model('chexworld', pool=False).cuda().eval()
for p in enc.parameters(): p.requires_grad_(False)
dl = DataLoader(PairDS(rd, gd, files), batch_size=32, num_workers=4, pin_memory=True)
per_img = []
with th.no_grad():
    for r, g in dl:
        hr = enc(r.cuda()); hg = enc(g.cuda())
        hr = F.normalize(hr, dim=2); hg = F.normalize(hg, dim=2)
        per_img.append((hr * hg).sum(dim=2).mean(dim=1).float().cpu())
v = th.cat(per_img)
out = os.path.join('${sp}', 'cossim_patch.txt')
with open(out, 'w') as f:
    f.write(f'metric=position_aligned_patch_cosine_mean\nencoder=chexworld\nreal_dir={rd}\ngen_dir={gd}\nn={len(files)}\nCosSim_patch_mean={v.mean().item():.6f}\nCosSim_patch_std={v.std().item():.6f}\n')
print(f'CosSim_patch={v.mean():.4f} (n={len(files)})')
" > "${LOG}/cossim_${label}_${ds}_${tag}.log" 2>&1
    echo "  ${tag}: $(grep '^CosSim_mean' ${sp}/distance_metrics.txt 2>/dev/null) | $(grep '^Rank_gen2cond_cosine' ${sp}/distance_metrics.txt 2>/dev/null)"
  done

  # summary
  echo ""
  echo "--- ${label} / ${ds} summary ---"
  printf "%-6s %14s %18s %16s %10s %16s %10s\n" cfg CosSim CosSim_patch Rank_g2c MRR_g2c Rank_c2g MRR_c2g
  for cfg in 1 1.5 2 2.5 3; do
    if [ "$cfg" = "1.5" ]; then
      d="${base}/distance_metrics.txt"
      cp="${base}/cossim_patch.txt"
    else
      d="${sweep}/cfg${cfg}_ema2/distance_metrics.txt"
      cp="${sweep}/cfg${cfg}_ema2/cossim_patch.txt"
    fi
    [ -f "$d" ] || { printf "%-6s %14s\n" "$cfg" "(pending)"; continue; }
    cos=$(grep '^CosSim_mean' "$d" 2>/dev/null | awk '{print $3}')
    cospatch=$(grep 'CosSim_patch_mean' "$cp" 2>/dev/null | awk -F= '{print $2}')
    rkg=$(grep '^Rank_gen2cond_cosine' "$d" 2>/dev/null | awk '{print $3}')
    mrrg=$(grep '^MRR_gen2cond_cosine' "$d" 2>/dev/null | awk '{print $3}')
    rkc=$(grep '^Rank_cond2gen_cosine' "$d" 2>/dev/null | awk '{print $3}')
    mrrc=$(grep '^MRR_cond2gen_cosine' "$d" 2>/dev/null | awk '{print $3}')
    printf "%-6s %14s %18s %16s %10s %16s %10s\n" "$cfg" "$cos" "$cospatch" "$rkg" "$mrrg" "$rkc" "$mrrc"
  done | tee "${sweep}/summary.txt"
  echo ""
}

POOLED="${OUT}/rct_pooled_224_chexworld_v2"
POOLED_CK="${POOLED}/checkpoints/stepstep=0300000.ckpt"
TOKEN="${OUT}/rct_token_224_p14_chexworld_v2"
TOKEN_CK="${TOKEN}/checkpoints/stepstep=0300000.ckpt"

# NIH: 25596 images
run_one "pooled" "$POOLED" "$POOLED_CK" "nih" 25596
run_one "token"  "$TOKEN"  "$TOKEN_CK"  "nih" 25596

# CheXpert: 202 images
run_one "pooled" "$POOLED" "$POOLED_CK" "chexpert" 202
run_one "token"  "$TOKEN"  "$TOKEN_CK"  "chexpert" 202

echo "[sweep] ALL DONE at $(date)"
