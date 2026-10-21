#!/usr/bin/env python
"""Fixed-feature sampling for the ILS top-20 test studies (discard figures).

For each of the 20 MIMIC-CXR-Ext-ILS top-studies, take the frontal (PA/AP)
conditioning image the model actually sees (the 256^2 npy, resized to img_size),
feed it through the requested model(s), and draw N=100 samples with the feature
fixed. Per (model, study) we save the same artifacts as label_sampling_all.py:
raw 100 samples, individual row PNGs, and npy maps (mean/std/err*).

Config is identical to label_sampling_all.py (N=100, batch 50, seed 0, ema2,
cfg from ckpt). Reuses that module's helpers.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/gen_discard_top20.py pooled_chexworld pooled_raddino pooled_densenet
  CUDA_VISIBLE_DEVICES=1 python scripts/gen_discard_top20.py patch_chexworld  patch_raddino  patch_densenet
"""
import sys, os
sys.path.insert(0, '/home/yspark/RCT/')
sys.path.insert(0, '/home/yspark/RCT/src')
import numpy as np
import torch as th
import torchvision.transforms.functional as TF
from PIL import Image

from scripts.eval_generation import build_rct_model
from scripts.label_sampling_all import (
    sample_stack, chw_to_uint8, map_to_rgb, signed_to_rgb,
    ckpt, MODELS, N_SAMPLES,
)
from rct.dataset import MIMICCXRDataset

MIMIC_ROOT = "/mnt/nvme1/mimic-cxr-jpg"
OUT_ROOT   = "/mnt/nvme1/RCT/main_result/discard_top20"

# grid order matches top20_test_studies_cxr.png (row-major)
STUDIES = [
    "s59039129","s58666319","s57907009","s55177624","s54432661",
    "s59480739","s58936335","s58611036","s58464159","s58155125",
    "s57446197","s57013017","s56121920","s56010471","s55921730",
    "s55303396","s54523680","s54233043","s53881360","s53247313",
]


def resolve_conditioning():
    """study id -> PIL RGB conditioning image (the frontal 256^2 npy)."""
    ds = MIMICCXRDataset(root=MIMIC_ROOT, split="test",
                         frontal_only=True, uncertainty="zero")
    df = ds.df
    out = {}
    for s in STUDIES:
        rows = df[df["study_id"] == int(s[1:])]
        assert len(rows) >= 1, f"{s}: no frontal row"
        idx = rows.index[0]
        out[s] = ds._load_image(idx).convert("RGB")   # PIL RGB, 256x256
    return out


def load_cond_pil(im, img_size):
    x = TF.to_tensor(TF.resize(im, [img_size, img_size], antialias=True))
    return (x * 2.0 - 1.0).unsqueeze(0).to("cuda", th.float16), x  # cond, real[0,1]


def run_model(tag, cond_imgs):
    print(f"\n===== MODEL: {tag} ({MODELS[tag]}) =====", flush=True)
    rct, _ = build_rct_model(ckpt_path=ckpt(MODELS[tag]), device="cuda",
                             ema_variant="ema2")
    img_size = rct.img_size
    out_dir = os.path.join(OUT_ROOT, tag)

    cache = {}
    for s in STUDIES:
        cond, real = load_cond_pil(cond_imgs[s], img_size)   # real: [3,H,W] in [0,1]
        with th.no_grad(), th.amp.autocast("cuda"):
            pooled, patches = rct.encode_cond(cond)
        stack = sample_stack(rct, pooled, patches)            # [n,3,H,W] in [0,1]
        mean_img = stack.mean(0)
        std_map  = stack.std(0).mean(0)
        err = stack - real.unsqueeze(0)
        err_signed = err.mean(0).mean(0)
        err_abs    = err.abs().mean(0).mean(0)
        err_std    = err.std(0).mean(0)
        cache[s] = dict(stack=stack, real=real, mean=mean_img, std=std_map,
                        err_signed=err_signed, err_abs=err_abs, err_std=err_std)
        print(f"  {s} | std={float(std_map.mean()):.4f} "
              f"MAE={float(err_abs.mean()):.4f} bias={float(err_signed.mean()):+.4f}",
              flush=True)

    # per-model shared ranges (for the saved PNG colorizations)
    all_std = th.stack([cache[s]["std"] for s in STUDIES])
    all_abs = th.stack([cache[s]["err_abs"] for s in STUDIES])
    all_sig = th.stack([cache[s]["err_signed"] for s in STUDIES])
    smin, smax = float(all_std.min()), float(all_std.max())
    amin, amax = float(all_abs.min()), float(all_abs.max())
    sig_abs = float(all_sig.abs().max())

    for s in STUDIES:
        c = cache[s]
        d = os.path.join(out_dir, s)
        os.makedirs(d, exist_ok=True)
        for k in range(c["stack"].shape[0]):
            Image.fromarray(chw_to_uint8(c["stack"][k])).save(
                os.path.join(d, f"sample_{k:03d}.png"))
        Image.fromarray(chw_to_uint8(c["real"])).save(os.path.join(d, "real.png"))
        Image.fromarray(chw_to_uint8(c["mean"])).save(os.path.join(d, "mean.png"))
        Image.fromarray(map_to_rgb(c["std"].numpy(), smin, smax)).save(os.path.join(d, "std.png"))
        Image.fromarray(signed_to_rgb(c["err_signed"].numpy(), sig_abs)).save(os.path.join(d, "err_signed.png"))
        Image.fromarray(map_to_rgb(c["err_abs"].numpy(), amin, amax)).save(os.path.join(d, "err_abs.png"))
        Image.fromarray(map_to_rgb(c["err_std"].numpy(), smin, smax)).save(os.path.join(d, "err_std.png"))
        np.save(os.path.join(d, "mean.npy"), c["mean"].numpy())
        np.save(os.path.join(d, "std.npy"), c["std"].numpy())
        np.save(os.path.join(d, "err_signed.npy"), c["err_signed"].numpy())
        np.save(os.path.join(d, "err_abs.npy"), c["err_abs"].numpy())
        np.save(os.path.join(d, "err_std.npy"), c["err_std"].numpy())

    del rct, cache
    th.cuda.empty_cache()
    print(f"  wrote -> {out_dir}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or list(MODELS.keys())
    cond_imgs = resolve_conditioning()
    print(f"resolved {len(cond_imgs)} conditioning images; running {which}", flush=True)
    for tag in which:
        run_model(tag, cond_imgs)
    print("\nDONE.")
