#!/usr/bin/env python
"""Per-image segmentation overlays: {dataset}_{method}_{index}.png"""

import csv
import os
import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import measure

BASE = "/home/yspark/RCT"

MODELS = {
    "RCDM": os.path.join(BASE, "output/main_result/rcdm_chexworld_cxr"),
    "FeatInv": os.path.join(BASE, "baseline/FeatInv/output"),
    "RCT_patch": os.path.join(BASE, "output/main_result/rct_token_224_p14_chexworld_v2"),
    "RCT_pool": os.path.join(BASE, "output/main_result/rct_pooled_224_chexworld_v2"),
}

DATASETS = {
    "MIMIC": "fid_eval",
    "CheXpert": "fid_eval_chexpert",
    "NIH": "fid_eval_nih",
}

OUT_DIR = os.path.join(BASE, "output/segmentation_vis")
os.makedirs(OUT_DIR, exist_ok=True)


def load_per_image(csv_path):
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            md = row.get("mean_dice", "")
            if md:
                data[row["file"]] = float(md)
    return data


def pick_examples(model_scores, n=10):
    common = set(model_scores["RCDM"].keys())
    for m in model_scores:
        common &= set(model_scores[m].keys())

    candidates = []
    for fname in common:
        rcdm = model_scores["RCDM"][fname]
        fi = model_scores["FeatInv"][fname]
        patch = model_scores["RCT_patch"][fname]
        pool = model_scores["RCT_pool"][fname]
        if patch > 0.55 and pool > 0.50 and (rcdm < 0.50 or fi < 0.50):
            gap = (patch + pool) - (rcdm + fi)
            candidates.append((fname, gap, rcdm, fi, patch, pool))

    candidates.sort(key=lambda x: -x[1])

    if len(candidates) < n:
        candidates_all = []
        for fname in common:
            rcdm = model_scores["RCDM"][fname]
            fi = model_scores["FeatInv"][fname]
            patch = model_scores["RCT_patch"][fname]
            pool = model_scores["RCT_pool"][fname]
            gap = (patch + pool) - (rcdm + fi)
            candidates_all.append((fname, gap, rcdm, fi, patch, pool))
        candidates_all.sort(key=lambda x: -x[1])
        candidates = candidates_all

    return candidates[:n]


@torch.no_grad()
def segment_image(seg_model, img_path, device, resize=224):
    im = Image.open(img_path).convert("RGB")
    if resize:
        im = im.resize((resize, resize), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32).mean(-1)
    arr = xrv.datasets.normalize(arr, 255)
    x = torch.from_numpy(arr)[None, None].to(device)
    logits = seg_model(x)
    masks = (torch.sigmoid(logits) > 0.5).to(torch.uint8).cpu().numpy()[0]
    return masks


def save_overlay(img_path, masks, out_path):
    from matplotlib.colors import to_rgba
    cmap = plt.get_cmap("tab20")
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    H, W = img.shape

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img, cmap="gray")

    for c in range(masks.shape[0]):
        m = masks[c]
        if m.sum() == 0:
            continue
        color = to_rgba(cmap(c % 20))
        # Resize mask to image size for filled overlay
        from scipy.ndimage import zoom
        scale_y, scale_x = H / m.shape[0], W / m.shape[1]
        m_resized = zoom(m.astype(float), (scale_y, scale_x), order=0) > 0.5
        # Semi-transparent filled mask
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[m_resized, :3] = color[:3]
        rgba[m_resized, 3] = 0.25
        ax.imshow(rgba, extent=[0, W, H, 0])
        # Bold contour
        for contour in measure.find_contours(m.astype(float), 0.5):
            ys = contour[:, 0] * H / m.shape[0]
            xs = contour[:, 1] * W / m.shape[1]
            ax.plot(xs, ys, lw=2.0, color=color[:3])

    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    seg_model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()

    all_results = []

    for ds_name, fid_subdir in DATASETS.items():
        print(f"\n=== {ds_name} ===")

        model_scores = {}
        for mname, mroot in MODELS.items():
            csv_path = os.path.join(mroot, fid_subdir, "seg_agreement_r224", "per_image.csv")
            if os.path.exists(csv_path):
                model_scores[mname] = load_per_image(csv_path)
                print(f"  {mname}: {len(model_scores[mname])} images")
            else:
                print(f"  {mname}: NOT FOUND")

        if len(model_scores) < 4:
            continue

        examples = pick_examples(model_scores, n=10)
        print(f"  Selected {len(examples)} examples")

        for fname, gap, rcdm_d, fi_d, patch_d, pool_d in examples:
            idx = os.path.splitext(fname)[0]  # e.g. "02973"
            print(f"  Processing {fname} ...")

            # GT (real image, segment it)
            real_path = os.path.join(MODELS["RCT_patch"], fid_subdir, "real", fname)
            if not os.path.exists(real_path):
                real_path = os.path.join(MODELS["RCDM"], fid_subdir, "real", fname)
            real_masks = segment_image(seg_model, real_path, device)
            save_overlay(real_path, real_masks, os.path.join(OUT_DIR, f"{ds_name}_GT_{idx}.png"))

            # Each model's generated image
            for mname, mroot in MODELS.items():
                gen_path = os.path.join(mroot, fid_subdir, "gen", fname)
                if not os.path.exists(gen_path):
                    print(f"    {mname} gen not found: {gen_path}")
                    continue
                gen_masks = segment_image(seg_model, gen_path, device)
                save_overlay(gen_path, gen_masks, os.path.join(OUT_DIR, f"{ds_name}_{mname}_{idx}.png"))

            all_results.append({
                "dataset": ds_name,
                "file": fname,
                "RCDM_dice": f"{rcdm_d:.4f}",
                "FeatInv_dice": f"{fi_d:.4f}",
                "RCT_patch_dice": f"{patch_d:.4f}",
                "RCT_pool_dice": f"{pool_d:.4f}",
            })

    csv_path = os.path.join(OUT_DIR, "dice_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "file", "RCDM_dice", "FeatInv_dice", "RCT_patch_dice", "RCT_pool_dice"])
        w.writeheader()
        w.writerows(all_results)
    print(f"\nCSV: {csv_path}")
    print(f"Total images: {len(all_results) * 5}")


if __name__ == "__main__":
    main()
