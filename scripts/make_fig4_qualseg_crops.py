#!/usr/bin/env python
"""Regenerate Fig.4 (qualseg) crops for chosen samples.

qual/{DS}_{id}_{suf}.png : reconstruction (grayscale), copied from gen at native
    resolution; RCDM (native 128) is upsampled to 224 for display.
seg/{DS}_{id}_{suf}.png  : PSPNet/ChestX-Det 14-structure masks rendered as solid
    tab20 colors on black, painted largest-area first (seg input resized to 224 to
    match the r224 agreement CSVs). GT is the segmented conditioning image.

Only the three rows shown in Fig.~\\ref{fig2} are (re)generated.
"""
import os
import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/yspark/RCT"
QUAL = os.path.join(BASE, "docs/paper/figs/qual")
SEG  = os.path.join(BASE, "docs/paper/figs/seg")
os.makedirs(QUAL, exist_ok=True); os.makedirs(SEG, exist_ok=True)

# chosen samples: dataset -> (fid subdir, id)
SAMPLES = {"MIMIC": ("fid_eval", "02973"),
           "NIH": ("fid_eval_nih", "14764"),
           "CheXpert": ("fid_eval_chexpert", "00092")}

REAL_DIR = "output/main_result/rct_token_224_p14_chexworld_v2"  # Original/GT source
# suffix -> (model dir, upsample128)
METHODS = {
    "rcdm":       ("output/main_result/rcdm_chexworld_cxr", True),
    "featinv":    ("baseline/FeatInv/output", False),
    "rctpool":    ("output/main_result/rct_pooled_224_chexworld_v2", False),
    "rctspatial": ("output/main_result/rct_token_224_p14_chexworld_v2", False),
    "rdpool":     ("output/main_result/rct_pooled_224_raddino", False),
    "rdspatial":  ("output/main_result/rct_token_224_p14_raddino", False),
    "dnpool":     ("output/main_result/rct_pooled_224_xrv_densenet", False),
    "dnspatial":  ("output/main_result/rct_token_224_p14_xrv_densenet", False),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()


def seg_masks(path, resize=224):
    im = Image.open(path).convert("RGB").resize((resize, resize), Image.BILINEAR)
    arr = xrv.datasets.normalize(np.asarray(im, np.float32).mean(-1), 255)
    x = torch.from_numpy(arr)[None, None].to(device)
    with torch.no_grad():
        return (torch.sigmoid(_model(x)) > 0.5).to(torch.uint8).cpu().numpy()[0]  # [14,512,512]


def render_seg(masks, out):
    cmap = plt.get_cmap("tab20")
    H = W = masks.shape[1]
    canvas = np.zeros((H, W, 3), np.float32)
    for c in sorted(range(masks.shape[0]), key=lambda k: -masks[k].sum()):  # largest first
        m = masks[c].astype(bool)
        if m.any():
            canvas[m] = cmap(c % 20)[:3]
    Image.fromarray((canvas * 255).astype(np.uint8)).save(out)


def save_qual(src, out, upsample128):
    im = Image.open(src)
    if upsample128:
        im = im.convert("L").resize((224, 224), Image.BILINEAR)
    im.save(out)


for ds, (sub, idx) in SAMPLES.items():
    real = os.path.join(BASE, REAL_DIR, sub, "real", f"{idx}.png")
    # Original (qual) + GT (seg)
    save_qual(real, os.path.join(QUAL, f"{ds}_{idx}_orig.png"), False)
    render_seg(seg_masks(real), os.path.join(SEG, f"{ds}_{idx}_gt.png"))
    for suf, (root, up) in METHODS.items():
        gen = os.path.join(BASE, root, sub, "gen", f"{idx}.png")
        save_qual(gen, os.path.join(QUAL, f"{ds}_{idx}_{suf}.png"), up)
        render_seg(seg_masks(gen), os.path.join(SEG, f"{ds}_{idx}_{suf}.png"))
    print(f"{ds} {idx}: qual+seg done", flush=True)

print("DONE")
