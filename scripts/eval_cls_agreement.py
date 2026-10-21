#!/usr/bin/env python
"""Classification-prediction agreement (gen vs real) across xrv DenseNet models.

For each pathology classifier, runs gen and real images through the model and
measures how well the gen prediction follows the real prediction (paired across
the dataset): Pearson r, Spearman rho, MAE, and binary agreement @0.5.

Usage:
    python scripts/eval_cls_agreement.py --root <dir with gen/ real/> \
        [--resize 224] [--batch 64] [--out-name cls_agreement]
"""
import argparse
import glob
import os

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
from scipy.stats import pearsonr, spearmanr, wilcoxon

WEIGHTS = [
    "densenet121-res224-all",
    "densenet121-res224-rsna",
    "densenet121-res224-nih",
    "densenet121-res224-pc",
    "densenet121-res224-chex",
    "densenet121-res224-mimic_nb",
    "densenet121-res224-mimic_ch",
]


def load_batch(paths, resize):
    """Load PNGs -> tensor [B,1,H,W] in xrv range [-1024,1024]."""
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if resize:
            im = im.resize((resize, resize), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32).mean(-1)
        out.append(xrv.datasets.normalize(a, 255))
    return torch.from_numpy(np.stack(out))[:, None]


@torch.no_grad()
def predict_all(models, paths, device, batch, resize):
    """Stream images from disk once; run every model. Memory-safe for large N.

    models: dict weight -> model. Returns dict weight -> probs [N, n_path].
    """
    acc = {w: [] for w in models}
    for i in range(0, len(paths), batch):
        x = load_batch(paths[i : i + batch], resize).to(device)
        for w, m in models.items():
            acc[w].append(m(x).cpu().numpy())
    return {w: np.concatenate(v) for w, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-name", default="cls_agreement")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--std-thresh", type=float, default=0.02,
                    help="min std of real probs for a pathology to count as informative")
    args = ap.parse_args()

    gen_paths = sorted(glob.glob(os.path.join(args.root, "gen", "*.png")))
    real_paths = sorted(glob.glob(os.path.join(args.root, "real", "*.png")))
    assert [os.path.basename(p) for p in gen_paths] == [os.path.basename(p) for p in real_paths]
    if args.limit:
        gen_paths, real_paths = gen_paths[: args.limit], real_paths[: args.limit]
    n = len(gen_paths)
    print(f"{n} pairs | resize={args.resize}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(args.root, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    print("loading models...")
    models = {w: xrv.models.DenseNet(weights=w).to(device).eval() for w in WEIGHTS}
    print("predicting (streaming images once, all models)...")
    gen_pred = predict_all(models, gen_paths, device, args.batch, args.resize)
    real_pred = predict_all(models, real_paths, device, args.batch, args.resize)

    import csv

    rows = []          # per (model, pathology)
    model_means = []   # per model, over informative pathologies

    for w in WEIGHTS:
        paths_names = models[w].pathologies
        gp = gen_pred[w]   # [n,18]
        rp = real_pred[w]

        per_r, per_mae, per_wp = [], [], []
        for k, path in enumerate(paths_names):
            g, r = gp[:, k], rp[:, k]
            std = r.std()
            informative = std >= args.std_thresh
            if std < 1e-8:
                pr = sr = np.nan
            else:
                pr = pearsonr(g, r)[0]
                sr = spearmanr(g, r)[0]
            mae = np.abs(g - r).mean()
            agree = ((g > 0.5) == (r > 0.5)).mean()
            # paired Wilcoxon: is gen systematically biased vs real?
            try:
                wp = wilcoxon(g, r)[1]
            except ValueError:
                wp = np.nan
            rows.append([w, path, f"{pr:.4f}", f"{sr:.4f}", f"{mae:.4f}",
                         f"{agree:.4f}", f"{r.std():.4f}", f"{wp:.2e}", int(informative)])
            if informative:
                per_r.append(pr); per_mae.append(mae); per_wp.append(agree)
        mr = np.nanmean(per_r) if per_r else np.nan
        mm = np.nanmean(per_mae) if per_mae else np.nan
        ma = np.nanmean(per_wp) if per_wp else np.nan
        model_means.append([w, len(per_r), f"{mr:.4f}", f"{mm:.4f}", f"{ma:.4f}"])
        print(f"{w:<28} informative={len(per_r):>2}  meanPearson={mr:.3f}  MAE={mm:.3f}  binAgree={ma:.3f}")

    # write per-pathology csv
    with open(os.path.join(out_dir, "per_pathology.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["model", "pathology", "pearson", "spearman", "mae",
                      "bin_agree@0.5", "real_std", "wilcoxon_p", "informative"])
        wtr.writerows(rows)

    with open(os.path.join(out_dir, "model_summary.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["model", "n_informative_path", "mean_pearson", "mean_mae", "mean_bin_agree"])
        wtr.writerows(model_means)

    print(f"\nWrote:\n  {out_dir}/per_pathology.csv\n  {out_dir}/model_summary.csv")


if __name__ == "__main__":
    main()
