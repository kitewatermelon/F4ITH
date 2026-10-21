#!/usr/bin/env python
"""Figure: what each encoder encodes vs discards, via fixed-feature sampling variance.

One conditioning image (00059, Pneumothorax) is fed through every model. Rows are
real / sample-mean / per-pixel std; columns are the six models. Where samples agree
(low std) the feature pins the content (encoded); where they disagree (high std) the
feature does not constrain it (discarded). The std row shares one colorbar so
mode/encoder differences are directly comparable.

Two column orderings are produced:
  A) grouped by encoder:  CheXWorld{patch,pool} | RAD-DINO{patch,pool} | DenseNet{patch,pool}
  B) grouped by mode:     patch{CW,RAD,DN} | pooled{CW,RAD,DN}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = "/mnt/nvme1/RCT/main_result"
OUT  = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

FIDX  = "00059"
LABEL = "Pneumothorax"

def paths(model):
    d = os.path.join(ROOT, model, LABEL, FIDX)
    return d

def load(model):
    d = paths(model)
    real = plt.imread(os.path.join(d, "real.png"))        # HxW or HxWx3/4
    if real.ndim == 3:
        real = real[..., 0]
    mean = np.load(os.path.join(d, "mean.npy"))           # [3,H,W] in [0,1]
    mean = np.transpose(mean, (1, 2, 0))
    std  = np.load(os.path.join(d, "std.npy"))            # [H,W]
    return real, mean, std

MODELS = ["patch_chexworld", "pooled_chexworld",
          "patch_raddino",   "pooled_raddino",
          "patch_densenet",  "pooled_densenet"]
DATA = {m: load(m) for m in MODELS}

# shared std range across all six panels (fair comparison)
SMAX = max(float(DATA[m][2].max()) for m in MODELS)
SMIN = 0.0
CMAP = "magma"
print(f"shared std range: [{SMIN:.3f}, {SMAX:.3f}]")


def build(order, headers, group_spans, fname, title):
    """order: list of models (6). headers: per-column short label.
    group_spans: list of (label, start, end) for a top group bar."""
    ncol = len(order)
    fig, axes = plt.subplots(3, ncol, figsize=(2.0 * ncol, 6.6), squeeze=False)
    row_names = ["real", "sample-mean", "per-pixel std"]
    for c, m in enumerate(order):
        real, mean, std = DATA[m]
        axes[0][c].imshow(real, cmap="gray")
        axes[1][c].imshow(np.clip(mean, 0, 1))
        axes[2][c].imshow(std, cmap=CMAP, vmin=SMIN, vmax=SMAX)
        axes[0][c].set_title(headers[c], fontsize=10, pad=4)
        for r in range(3):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
    for r, nm in enumerate(row_names):
        axes[r][0].set_ylabel(nm, fontsize=11)

    fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.02,
                        wspace=0.04, hspace=0.04)

    # group bars above the column titles
    for gl, a, b in group_spans:
        x0 = axes[0][a].get_position().x0
        x1 = axes[0][b].get_position().x1
        y  = axes[0][a].get_position().y1
        fig.lines.append(plt.Line2D([x0, x1], [y + 0.035, y + 0.035],
                                    transform=fig.transFigure, color="k", lw=1.2))
        fig.text((x0 + x1) / 2, y + 0.045, gl, ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    # shared colorbar for the std row
    cax = fig.add_axes([0.915, 0.04, 0.015, 0.24])
    sm = ScalarMappable(norm=Normalize(SMIN, SMAX), cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("std", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=12, y=0.985)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}.png / .pdf")


# A) grouped by encoder
build(
    order=["patch_chexworld", "pooled_chexworld",
           "patch_raddino",   "pooled_raddino",
           "patch_densenet",  "pooled_densenet"],
    headers=["patch", "pool", "patch", "pool", "patch", "pool"],
    group_spans=[("CheXWorld", 0, 1), ("RAD-DINO", 2, 3), ("DenseNet", 4, 5)],
    fname="fig_encoder_discard_byEncoder",
    title=f"Fixed-feature sampling variance ({LABEL}, N=100): encoded (low std) vs discarded (high std)",
)

# B) grouped by mode
build(
    order=["patch_chexworld", "patch_raddino", "patch_densenet",
           "pooled_chexworld", "pooled_raddino", "pooled_densenet"],
    headers=["CheXWorld", "RAD-DINO", "DenseNet", "CheXWorld", "RAD-DINO", "DenseNet"],
    group_spans=[("patch (spatial)", 0, 2), ("pooled (global)", 3, 5)],
    fname="fig_encoder_discard_byMode",
    title=f"Fixed-feature sampling variance ({LABEL}, N=100): encoded (low std) vs discarded (high std)",
)
print("DONE")
