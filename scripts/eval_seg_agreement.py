#!/usr/bin/env python
"""Anatomical segmentation agreement (gen vs real) via PSPNet.

Segments 14 anatomical structures on paired gen/real chest X-rays with
torchxrayvision's ChestX-Det PSPNet and reports per-organ Dice/IoU agreement.

Usage:
    python scripts/eval_seg_agreement.py \
        --root output/main_result/rct_token_224_p14_chexworld/fid_eval \
        [--batch 16] [--n-overlays 8] [--limit N]
"""
import argparse
import glob
import os

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image


def load_batch(paths, resize=0):
    """Load PNGs -> tensor [B,1,H,W] in xrv range [-1024, 1024].

    If resize > 0, each image is resized to (resize, resize) (bilinear) before
    normalization, so different-resolution sets can be compared apples-to-apples.
    """
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if resize:
            im = im.resize((resize, resize), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32).mean(-1)  # HxW, [0,255]
        arr = xrv.datasets.normalize(arr, 255)  # -> [-1024, 1024]
        imgs.append(arr)
    x = torch.from_numpy(np.stack(imgs))[:, None]  # [B,1,H,W]
    return x


@torch.no_grad()
def segment(model, paths, device, batch, resize=0):
    """Return binary masks [N,14,512,512] (uint8) for the given image paths."""
    out = []
    for i in range(0, len(paths), batch):
        x = load_batch(paths[i : i + batch], resize).to(device)
        logits = model(x)  # [b,14,512,512]
        masks = (torch.sigmoid(logits) > 0.5).to(torch.uint8).cpu()
        out.append(masks)
    return torch.cat(out).numpy()


def dice_iou(a, b):
    """a,b: bool HxW. Returns (dice, iou); NaN if both empty."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    sa, sb = a.sum(), b.sum()
    union = sa + sb - inter
    if sa == 0 and sb == 0:
        return np.nan, np.nan
    dice = 2.0 * inter / (sa + sb)
    iou = inter / union
    return dice, iou


def save_overlay(gen_path, real_path, gm, rm, out_path, targets, mean_dice):
    """Side-by-side gen|real with per-organ mask contours overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure

    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, path, masks, title in [
        (axes[0], gen_path, gm, "gen"),
        (axes[1], real_path, rm, "real"),
    ]:
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        ax.imshow(img, cmap="gray")
        H, W = img.shape
        for c in range(masks.shape[0]):
            m = masks[c]
            if m.sum() == 0:
                continue
            # masks are 512x512; scale contour coords to image size
            for contour in measure.find_contours(m.astype(float), 0.5):
                ys = contour[:, 0] * H / m.shape[0]
                xs = contour[:, 1] * W / m.shape[1]
                ax.plot(xs, ys, lw=0.8, color=cmap(c % 20))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{os.path.basename(gen_path)}  mean Dice={mean_dice:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing gen/ and real/")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-overlays", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="limit #pairs (0 = all)")
    ap.add_argument("--resize", type=int, default=0, help="resize inputs to NxN before seg (0 = native)")
    ap.add_argument("--out-name", default="seg_agreement", help="output subdir name under root")
    args = ap.parse_args()

    gen_paths = sorted(glob.glob(os.path.join(args.root, "gen", "*.png")))
    real_paths = sorted(glob.glob(os.path.join(args.root, "real", "*.png")))
    assert [os.path.basename(p) for p in gen_paths] == [
        os.path.basename(p) for p in real_paths
    ], "gen/real filename sets differ"
    if args.limit:
        gen_paths, real_paths = gen_paths[: args.limit], real_paths[: args.limit]
    n = len(gen_paths)
    print(f"{n} pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    targets = model.targets
    n_org = len(targets)

    out_dir = os.path.join(args.root, args.out_name)
    os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)

    dice = np.full((n, n_org), np.nan, dtype=np.float32)
    iou = np.full((n, n_org), np.nan, dtype=np.float32)
    overlay_step = max(1, n // max(1, args.n_overlays))

    for i in range(0, n, args.batch):
        gp = gen_paths[i : i + args.batch]
        rp = real_paths[i : i + args.batch]
        gm = segment(model, gp, device, args.batch, args.resize)
        rm = segment(model, rp, device, args.batch, args.resize)
        for j in range(len(gp)):
            idx = i + j
            for c in range(n_org):
                dice[idx, c], iou[idx, c] = dice_iou(gm[j, c], rm[j, c])
            if idx % overlay_step == 0 and (idx // overlay_step) < args.n_overlays:
                md = np.nanmean(dice[idx])
                save_overlay(
                    gp[j], rp[j], gm[j], rm[j],
                    os.path.join(out_dir, "overlays", f"{os.path.basename(gp[j])}"),
                    targets, md,
                )
        print(f"  {min(i + args.batch, n)}/{n}", end="\r")
    print()

    # per-image CSV
    import csv

    per_img = os.path.join(out_dir, "per_image.csv")
    with open(per_img, "w", newline="") as f:
        w = csv.writer(f)
        header = ["file"] + [f"dice_{t}" for t in targets] + [f"iou_{t}" for t in targets] + ["mean_dice", "mean_iou"]
        w.writerow(header)
        for idx in range(n):
            row = [os.path.basename(gen_paths[idx])]
            row += [f"{v:.4f}" if not np.isnan(v) else "" for v in dice[idx]]
            row += [f"{v:.4f}" if not np.isnan(v) else "" for v in iou[idx]]
            row += [f"{np.nanmean(dice[idx]):.4f}", f"{np.nanmean(iou[idx]):.4f}"]
            w.writerow(row)

    # summary
    summary = os.path.join(out_dir, "summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["organ", "mean_dice", "mean_iou", "n_valid"])
        print(f"\n{'organ':<22} {'Dice':>7} {'IoU':>7} {'N':>6}")
        print("-" * 45)
        for c, t in enumerate(targets):
            valid = ~np.isnan(dice[:, c])
            md = np.nanmean(dice[:, c]) if valid.any() else float("nan")
            mi = np.nanmean(iou[:, c]) if valid.any() else float("nan")
            w.writerow([t, f"{md:.4f}", f"{mi:.4f}", int(valid.sum())])
            print(f"{t:<22} {md:>7.4f} {mi:>7.4f} {int(valid.sum()):>6}")
        macro_d = np.nanmean(dice)
        macro_i = np.nanmean(iou)
        w.writerow(["MACRO_MEAN", f"{macro_d:.4f}", f"{macro_i:.4f}", n])
        print("-" * 45)
        print(f"{'MACRO_MEAN':<22} {macro_d:>7.4f} {macro_i:>7.4f} {n:>6}")

    print(f"\nWrote:\n  {summary}\n  {per_img}\n  {out_dir}/overlays/")


if __name__ == "__main__":
    main()
