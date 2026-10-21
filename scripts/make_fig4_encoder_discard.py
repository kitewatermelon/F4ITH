#!/usr/bin/env python
"""Paper Figure 4: renders the encoder-discard byMode figure for the study used in
the manuscript (s53247313) into docs/paper/legacy/figs/fig4_encoder_discard.png with
the hedged, operational title ("tightly constrained (low std) vs. left free (high
std)", "one of 20 lesion-annotated studies"). Derived from
make_discard_top20_figures.py; GPU-free (reads precomputed std/mean/err_signed .npy).

Layout per study (columns = 6 models, grouped patch(spatial) | pooled(global)):
  left reference column, 2 cells:
    GT   : clean 224 conditioning CXR
    mask : same CXR with ILS lesion masks overlaid (top-20 palette)
  right grid, 3 rows x 6 models:
    sample-mean / per-pixel std / signed-error (sample-mean - GT)

std colorbar (magma) and signed-error colorbar (RdBu_r) are both self-normalized
per study. Reads real/mean/std/err_signed from main_result/discard_top20/, and
lesion masks from mimic_ils/lesion_mask_test/. GPU-free.
Writes only {study}_byMode.png (overwrites in place).
"""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from PIL import Image

SRC  = "/mnt/nvme1/RCT/main_result/discard_top20"
OUT  = "/home/yspark/RCT/docs/paper/legacy/figs"
MASK = "/mnt/nvme1/RCT/mimic_ils/lesion_mask_test"
os.makedirs(OUT, exist_ok=True)

STUDIES = ["s53247313"]
# byMode order: patch{CW,RAD,DN} | pooled{CW,RAD,DN}
MODELS  = ["patch_chexworld", "patch_raddino", "patch_densenet",
           "pooled_chexworld", "pooled_raddino", "pooled_densenet"]
HEADERS = ["CheXWorld", "RAD-DINO", "DenseNet", "CheXWorld", "RAD-DINO", "DenseNet"]
GROUPS  = [("patch (spatial)", 0, 2), ("pooled (global)", 3, 5)]

# top-20 overlay palette (matplotlib tab10 == overlay legend colors)
PALETTE = {
    "cardiomegaly": "#1f77b4", "pneumonia": "#ff7f0e", "atelectasis": "#2ca02c",
    "opacity": "#d62728", "consolidation": "#9467bd", "edema": "#8c564b",
    "effusion": "#e377c2",
}
ABBR = {"cardiomegaly":"card","atelectasis":"atel","opacity":"opac",
        "consolidation":"cons","edema":"edem","effusion":"effu","pneumonia":"pneu"}


def load_maps(model, study):
    d = os.path.join(SRC, model, study)
    mean = np.transpose(np.load(os.path.join(d, "mean.npy")), (1, 2, 0))  # [H,W,3]
    std  = np.load(os.path.join(d, "std.npy"))                            # [H,W]
    sig  = np.load(os.path.join(d, "err_signed.npy"))                     # [H,W]
    return mean, std, sig


def gt_rgb(study):
    real = plt.imread(os.path.join(SRC, MODELS[0], study, "real.png"))[..., :3]
    return np.clip(real, 0, 1)


def overlay_masks(study, size, alpha=1.0):
    """CXR with lesion masks overlaid; smaller masks are drawn last (on top) so a
    large mask never hides a small one. Returns (rgb, labels_present alphabetical)."""
    base = gt_rgb(study).copy()
    H, W = base.shape[:2]
    items = []
    for f in sorted(glob.glob(os.path.join(MASK, f"{study}_*.png"))):
        m = re.match(rf"{study}_([a-z]+)_\d+\.png", os.path.basename(f))
        if not m:
            continue
        lab = m.group(1)
        mk = np.array(Image.open(f).convert("L").resize((W, H), Image.NEAREST)) > 127
        items.append((int(mk.sum()), lab, mk))
    # largest area first -> smallest painted last (topmost)
    for area, lab, mk in sorted(items, key=lambda t: t[0], reverse=True):
        col = np.array(matplotlib.colors.to_rgb(PALETTE.get(lab, "#ffffff")))
        base[mk] = (1 - alpha) * base[mk] + alpha * col
    present = sorted({lab for _, lab, _ in items})
    return np.clip(base, 0, 1), present


def build(study):
    data = {m: load_maps(m, study) for m in MODELS}
    smax = max(float(data[m][1].max()) for m in MODELS)                    # std self-norm
    sigmax = max(float(np.abs(data[m][2]).max()) for m in MODELS)          # signed self-norm
    gt = gt_rgb(study)
    mask_rgb, present = overlay_masks(study, gt.shape[0])

    ncol = len(MODELS)
    fig = plt.figure(figsize=(2.0 * (ncol + 1) + 0.6, 7.2))
    # 6 stacked units high; col0=ref, col1=gap, col2..=models
    gs = GridSpec(6, ncol + 2, figure=fig,
                  width_ratios=[1.0, 0.18] + [1.0] * ncol,
                  height_ratios=[1] * 6, wspace=0.06, hspace=0.06,
                  left=0.055, right=0.9, top=0.86, bottom=0.10)

    # left reference column: GT (rows 0-2), mask (rows 3-5)
    axGT = fig.add_subplot(gs[0:3, 0]);  axMK = fig.add_subplot(gs[3:6, 0])
    axGT.imshow(gt[..., 0], cmap="gray"); axGT.set_title("GT", fontsize=11, pad=4)
    axMK.imshow(mask_rgb);               axMK.set_title("GT + lesion mask", fontsize=10, pad=4)
    for ax in (axGT, axMK):
        ax.set_xticks([]); ax.set_yticks([])

    # right grid: rows mean(0:2)/std(2:4)/signed(4:6) x models
    rows = [("sample-mean", 0), ("per-pixel std", 2), ("signed-error", 4)]
    top_axes = {}
    for c, m in enumerate(MODELS):
        mean, std, sig = data[m]
        col = 2 + c
        a0 = fig.add_subplot(gs[0:2, col]); a0.imshow(np.clip(mean, 0, 1))
        a1 = fig.add_subplot(gs[2:4, col]); a1.imshow(std, cmap="magma", vmin=0.0, vmax=smax)
        a2 = fig.add_subplot(gs[4:6, col]); a2.imshow(sig, cmap="RdBu_r", vmin=-sigmax, vmax=sigmax)
        a0.set_title(HEADERS[c], fontsize=10, pad=4)
        top_axes[c] = a0
        for r, ax in enumerate((a0, a1, a2)):
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(rows[r][0], fontsize=11)

    # group bars above model columns (reuse the stored top-row axes)
    for gl, a, b in GROUPS:
        x0 = top_axes[a].get_position().x0
        x1 = top_axes[b].get_position().x1
        y  = top_axes[a].get_position().y1
        fig.lines.append(plt.Line2D([x0, x1], [y + 0.03, y + 0.03],
                                    transform=fig.transFigure, color="k", lw=1.2))
        fig.text((x0 + x1) / 2, y + 0.038, gl, ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    # two colorbars on the right (std over the std row, signed over the signed row)
    cax_s = fig.add_axes([0.915, 0.44, 0.013, 0.16])
    cb_s = fig.colorbar(ScalarMappable(Normalize(0, smax), "magma"), cax=cax_s)
    cb_s.set_label("std", fontsize=9); cb_s.ax.tick_params(labelsize=8)
    cax_e = fig.add_axes([0.915, 0.12, 0.013, 0.16])
    cb_e = fig.colorbar(ScalarMappable(Normalize(-sigmax, sigmax), "RdBu_r"), cax=cax_e)
    cb_e.set_label("signed err", fontsize=9); cb_e.ax.tick_params(labelsize=8)

    # legend of lesion labels present in this study
    if present:
        handles = [Patch(facecolor=PALETTE[l], edgecolor="none", label=l) for l in present]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))

    lab = ", ".join(sorted(ABBR.get(l, l) for l in present))
    fig.suptitle(f"Fixed-feature sampling variance — {study} ({lab}), N=100  "
                 f"(one of 20 lesion-annotated studies): pixels tightly constrained "
                 f"(low std) vs. left free (high std)", fontsize=12, y=0.97)
    fig.savefig(os.path.join(OUT, "fig4_encoder_discard.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{study}: byMode.png (smax={smax:.4f} sigmax={sigmax:.4f} labels={present})", flush=True)


def main():
    for study in STUDIES:
        build(study)
    print("DONE")


if __name__ == "__main__":
    main()
