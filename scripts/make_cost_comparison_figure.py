#!/usr/bin/env python
"""Cost-comparison figure for the RCT manuscript. Two panels:
  (a) trainable vs frozen parameters as a stacked bar (M)
  (b) compute (GFLOPs)
across RCDM, FeatInv, RCT-global, RCT-spatial.

Per the paper's framing, the frozen vision encoder (CheXWorld, 85.8 M) is
*swappable and common to every method* ("across three encoders"), so it is
excluded from the parameter bars: RCDM and both RCT variants are therefore
trainable-only, and FeatInv's frozen segment is its external pretrained
generator (dagger). GPU-free; writes docs/paper/figs/fig_cost_comparison.png.

Colors are the validated dataviz categorical slots (CVD worst-adjacent dE 13.3,
light surface): RCDM red, FeatInv green, and the two RCT variants a cool blue/
aqua pair so "ours" reads as one family. Aqua sits below 3:1 contrast, so the
relief rule is met by direct value labels on every bar.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "/home/yspark/RCT/docs/paper/figs"
os.makedirs(OUT, exist_ok=True)

MODELS = ["RCDM\n(128x128)", "FeatInv", "RCT-global\n(Ours)", "RCT-spatial\n(Ours)"]
COLORS = ["#e34948", "#008300", "#2a78d6", "#1baf7a"]  # RCDM, FeatInv, RCT-g, RCT-s

# Parameters (M) -- CheXWorld encoder (85.8 M, swappable) excluded from Frozen.
TRAINABLE = np.array([421.29, 367.57, 65.90, 58.95])
FROZEN    = np.array([  0.00, 1152.03, 0.00,  0.00])   # FeatInv: external generator (dagger)
FROZEN_DAGGER = [False, True, False, False]

# Compute
GFLOPS = np.array([614.61, 242.42, 22.78, 29.16])

# Fold factors vs the lightest model (RCT-spatial), from the source table.
PARAM_FOLD = ["7.1x", "6.2x", "1.1x", "1.0x"]   # trainable
FLOP_FOLD  = ["21.1x", "8.3x", "0.8x", "1.0x"]

# --- chrome ---
INK   = "#0b0b0b"
MUTED = "#898781"
GRID  = "#e1e0d9"
BASE  = "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": BASE,
})

x = np.arange(len(MODELS))
WIDTH = 0.62  # slim bars
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8, 8))


def style_axis(ax, ylabel, ymax):
    ax.set_ylabel(ylabel, fontsize=11, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9.5, color=INK)
    ax.set_ylim(0, ymax)
    ax.tick_params(axis="y", labelsize=9, color=BASE, labelcolor=MUTED)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)


# ---------- (a) parameters, stacked ----------
total = TRAINABLE + FROZEN
axA.bar(x, TRAINABLE, width=WIDTH, color=COLORS, edgecolor="white", lw=1.2, zorder=3)
axA.bar(x, FROZEN, width=WIDTH, bottom=TRAINABLE, color=COLORS, edgecolor="white", lw=1.2,
        hatch="////", alpha=0.45, zorder=3)

ymaxA = total.max() * 1.16
style_axis(axA, "Parameters (M)", ymaxA)

for i in range(len(MODELS)):
    # value above the bar (folds omitted here: a trainable-fold over a stacked
    # total bar would be ambiguous; the split values already carry the story)
    axA.text(x[i], total[i] + ymaxA * 0.015, f"{total[i]:,.0f}",
             ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
    if FROZEN[i] > 0:
        # split values as white labels centered inside each segment
        axA.text(x[i], TRAINABLE[i] * 0.5, f"{TRAINABLE[i]:,.0f}",
                 ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        lbl = f"{FROZEN[i]:,.0f}" + ("†" if FROZEN_DAGGER[i] else "")
        axA.text(x[i], TRAINABLE[i] + FROZEN[i] * 0.5, lbl,
                 ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

axA.set_title("(a) Parameters", fontsize=12, color=INK, pad=8, loc="left")
legend = axA.legend(handles=[
    Patch(facecolor=MUTED, edgecolor="white", label="Trainable"),
    Patch(facecolor=MUTED, edgecolor="white", hatch="////", alpha=0.55,
          label="Frozen (external generator, †)"),
], fontsize=8.5, frameon=False, loc="upper right", handlelength=1.4,
   borderaxespad=0.4)

# ---------- (b) GFLOPs ----------
ymaxB = GFLOPS.max() * 1.16
axB.bar(x, GFLOPS, width=WIDTH, color=COLORS, edgecolor="white", lw=1.2, zorder=3)
style_axis(axB, "Compute (GFLOPs)", ymaxB)
for i in range(len(MODELS)):
    axB.text(x[i], GFLOPS[i] + ymaxB * 0.015, f"{GFLOPS[i]:,.1f}",
             ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
    axB.text(x[i], GFLOPS[i] + ymaxB * 0.058, FLOP_FOLD[i],
             ha="center", va="bottom", fontsize=7.5, color=MUTED)
axB.set_title("(b) Compute", fontsize=12, color=INK, pad=8, loc="left")

# fig.text(0.5, 0.005,
#          "Frozen vision encoder (CheXWorld, 85.8 M) is swappable and common to all "
#          "methods, so it is excluded here. † FeatInv's frozen segment is its external "
#          "pretrained generator. Fold factors are relative to RCT-spatial.",
#          ha="center", va="bottom", fontsize=7.5, color=MUTED)

fig.tight_layout(rect=(0, 0.045, 1, 1))
out = os.path.join(OUT, "fig_cost_comparison.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig)
print("wrote", out)
