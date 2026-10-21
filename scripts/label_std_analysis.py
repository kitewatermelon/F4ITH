#!/usr/bin/env python
"""Label-stratified per-pixel mean/std analysis (StyleGAN Fig.4c style).

For each CheXpert pathology label, pick 5 real conditioning images that are
positive for that label, condition the RCT model on each, sample N times
(noise varies), and compute the per-pixel mean image + per-pixel std map.

Runs for both the pooled (cls_only) and token (patch_only) checkpoints.
Labels are reconstructed by replaying the seed-0 permutation of the MIMIC-CXR
test split used by build_real_set in scripts/eval_generation.py.
"""
import sys, os, csv
sys.path.insert(0, '/home/yspark/RCT/')
sys.path.insert(0, '/home/yspark/RCT/src')
import numpy as np
import torch as th
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.eval_generation import build_rct_model
from rct.dataset import MIMICCXRDataset

# ---- config ---------------------------------------------------------------
SEED_PERM   = 0        # matches build_real_set --seed default
NUM_IMAGES  = 3403
N_SAMPLES   = 100      # samplings per image
PER_LABEL   = 5        # images per label
BATCH       = 50       # samples per generate call
NOISE_SEED  = 0        # reproducible sampling noise
MIMIC_ROOT  = "/mnt/nvme1/mimic-cxr-jpg"
OUT_ROOT    = "output/main_result/label_std_analysis"

MODELS = {
    "pooled": {
        "ckpt": "output/main_result/rct_pooled_224_chexworld_v2/checkpoints/last.ckpt",
        "real": "output/main_result/rct_pooled_224_chexworld_v2/fid_eval/real",
    },
    "token": {
        "ckpt": "output/main_result/rct_token_224_p14_chexworld_v2/checkpoints/last.ckpt",
        "real": "output/main_result/rct_token_224_p14_chexworld_v2/fid_eval/real",
    },
}
LABELS = MIMICCXRDataset.CHEXPERT_LABELS


def build_label_matrix(real_dir):
    """Return (file_indices sorted, label_matrix [n,14] binary)."""
    ds = MIMICCXRDataset(root=MIMIC_ROOT, split="test",
                         frontal_only=True, uncertainty="zero")
    g = th.Generator().manual_seed(SEED_PERM)
    perm = th.randperm(len(ds), generator=g)[:NUM_IMAGES].tolist()
    files = sorted(int(os.path.splitext(f)[0])
                   for f in os.listdir(real_dir) if f.endswith(".png"))
    mat = np.zeros((len(files), len(LABELS)), dtype=int)
    for row, i in enumerate(files):
        lab = ds.df.iloc[perm[i]][LABELS].values.astype(float)
        mat[row] = (lab == 1).astype(int)
    return files, mat


def select_per_label(files, mat):
    """label -> list of up to PER_LABEL file indices positive for it."""
    sel = {}
    for j, name in enumerate(LABELS):
        pos = [files[r] for r in range(len(files)) if mat[r, j] == 1]
        sel[name] = pos[:PER_LABEL]
    return sel


def load_cond(path, img_size):
    im = Image.open(path).convert("RGB")
    x = TF.to_tensor(TF.resize(im, [img_size, img_size], antialias=True))
    return (x * 2.0 - 1.0).unsqueeze(0).to("cuda", th.float16)


@th.no_grad()
def sample_mean_std(rct, pooled, patches, n=N_SAMPLES, batch=BATCH, seed=NOISE_SEED):
    """n samplings (noise varies) -> (mean_img[3,H,W], std_map[H,W]) in [0,1]."""
    th.manual_seed(seed)
    parts, done = [], 0
    with th.amp.autocast("cuda"):
        while done < n:
            b = min(batch, n - done)
            dc = pooled[:1].expand(b, *pooled.shape[1:])
            pt = patches[:1].expand(b, *patches.shape[1:]).contiguous()
            g = rct.generate_from_features(dc, pt)          # [b,3,H,W]
            parts.append(g.float().cpu())
            done += b
    imgs = (th.cat(parts) + 1) / 2                           # [n,3,H,W]
    return imgs.mean(0), imgs.std(0).mean(0)                 # [3,H,W], [H,W]


def to_uint8(chw):
    a = chw.clamp(0, 1).permute(1, 2, 0).numpy()
    return (a * 255).round().astype(np.uint8)


def std_to_rgb(std_np, smin, smax, cmap="magma"):
    """Colormap a std map to an RGB uint8 image using a shared range."""
    norm = np.clip((std_np - smin) / max(smax - smin, 1e-8), 0, 1)
    rgba = matplotlib.colormaps[cmap](norm)
    return (rgba[..., :3] * 255).round().astype(np.uint8)


def save_individual(ld, fidx, real_dir, mean_img, std_map, smin, smax):
    """Save real / mean / std as three separate image files for one panel."""
    real = Image.open(os.path.join(real_dir, f"{fidx:05d}.png")).convert("L")
    real.save(os.path.join(ld, f"{fidx:05d}_real.png"))
    Image.fromarray(to_uint8(mean_img)).save(os.path.join(ld, f"{fidx:05d}_mean.png"))
    Image.fromarray(std_to_rgb(std_map.numpy(), smin, smax)).save(
        os.path.join(ld, f"{fidx:05d}_std.png"))


def run_model(tag, cfg):
    print(f"\n===== MODEL: {tag} =====", flush=True)
    files, mat = build_label_matrix(cfg["real"])
    sel = select_per_label(files, mat)

    rct, _ = build_rct_model(ckpt_path=cfg["ckpt"], device="cuda", ema_variant="ema2")
    img_size = rct.img_size
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    # cache encodings so images shared across labels are encoded once
    enc_cache = {}
    summary = []

    # global std range for a shared colorbar across all panels of this model
    std_store = {}   # (label, fidx) -> (mean_img, std_map)

    for name in LABELS:
        for fidx in sel[name]:
            key = fidx
            if key not in enc_cache:
                cond = load_cond(os.path.join(cfg["real"], f"{fidx:05d}.png"), img_size)
                with th.no_grad(), th.amp.autocast("cuda"):
                    enc_cache[key] = rct.encode_cond(cond)
            pooled, patches = enc_cache[key]
            mean_img, std_map = sample_mean_std(rct, pooled, patches)
            std_store[(name, fidx)] = (mean_img, std_map)
            sm, sx = float(std_map.mean()), float(std_map.max())
            summary.append((name, fidx, sm, sx))
            print(f"  {name:28s} {fidx:05d} | std mean={sm:.4f} max={sx:.4f}", flush=True)

    all_std = th.stack([v[1] for v in std_store.values()])
    smin, smax = float(all_std.min()), float(all_std.max())
    print(f"  shared std range: [{smin:.4f}, {smax:.4f}]", flush=True)

    # save raw npy + individual per-panel pngs (real / mean / std saved separately)
    for (name, fidx), (mean_img, std_map) in std_store.items():
        ld = os.path.join(out_dir, name.replace(" ", "_"))
        os.makedirs(ld, exist_ok=True)
        np.save(os.path.join(ld, f"{fidx:05d}_mean.npy"), mean_img.numpy())
        np.save(os.path.join(ld, f"{fidx:05d}_std.npy"), std_map.numpy())
        save_individual(ld, fidx, cfg["real"], mean_img, std_map, smin, smax)

    # per-label montage: rows = [real cond, sample-mean, std map], cols = 5 images
    for name in LABELS:
        fidxs = sel[name]
        if not fidxs:
            continue
        ncol = len(fidxs)
        fig, axes = plt.subplots(3, ncol, figsize=(2.4 * ncol, 7.2), squeeze=False)
        for c, fidx in enumerate(fidxs):
            real = Image.open(os.path.join(cfg["real"], f"{fidx:05d}.png")).convert("L")
            mean_img, std_map = std_store[(name, fidx)]
            axes[0][c].imshow(real, cmap="gray"); axes[0][c].set_title(f"{fidx:05d}", fontsize=8)
            axes[1][c].imshow(to_uint8(mean_img))
            im = axes[2][c].imshow(std_map.numpy(), cmap="magma", vmin=smin, vmax=smax)
            for r in range(3):
                axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        axes[0][0].set_ylabel("real cond", fontsize=9)
        axes[1][0].set_ylabel(f"mean of {N_SAMPLES}", fontsize=9)
        axes[2][0].set_ylabel("per-pixel std", fontsize=9)
        fig.suptitle(f"[{tag}] {name}  (N={N_SAMPLES} samples/image)", fontsize=11)
        fig.colorbar(im, ax=axes[2, :].tolist(), fraction=0.046, pad=0.02)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(out_dir, f"montage_{name.replace(' ', '_')}.png"),
                    dpi=110, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "file_idx", "std_mean", "std_max"])
        w.writerows(summary)

    # per-label aggregate (mean +/- std of the scalar std_mean over the 5 images)
    with open(os.path.join(out_dir, "per_label_std.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n", "std_mean_avg", "std_mean_sd"])
        for name in LABELS:
            vals = [s[2] for s in summary if s[0] == name]
            if vals:
                w.writerow([name, len(vals), f"{np.mean(vals):.5f}", f"{np.std(vals):.5f}"])
    print(f"  wrote -> {out_dir}", flush=True)


def post_model(tag, cfg):
    """Regenerate individual real/mean/std pngs + montages from saved .npy (no GPU)."""
    print(f"\n===== POST: {tag} =====", flush=True)
    files, mat = build_label_matrix(cfg["real"])
    sel = select_per_label(files, mat)
    out_dir = os.path.join(OUT_ROOT, tag)

    store = {}
    for name in LABELS:
        for fidx in sel[name]:
            ld = os.path.join(out_dir, name.replace(" ", "_"))
            mean_img = th.from_numpy(np.load(os.path.join(ld, f"{fidx:05d}_mean.npy")))
            std_map = th.from_numpy(np.load(os.path.join(ld, f"{fidx:05d}_std.npy")))
            store[(name, fidx)] = (mean_img, std_map)

    all_std = th.stack([v[1] for v in store.values()])
    smin, smax = float(all_std.min()), float(all_std.max())

    for (name, fidx), (mean_img, std_map) in store.items():
        ld = os.path.join(out_dir, name.replace(" ", "_"))
        save_individual(ld, fidx, cfg["real"], mean_img, std_map, smin, smax)

    for name in LABELS:
        fidxs = sel[name]
        if not fidxs:
            continue
        ncol = len(fidxs)
        fig, axes = plt.subplots(3, ncol, figsize=(2.4 * ncol, 7.2), squeeze=False)
        for c, fidx in enumerate(fidxs):
            real = Image.open(os.path.join(cfg["real"], f"{fidx:05d}.png")).convert("L")
            mean_img, std_map = store[(name, fidx)]
            axes[0][c].imshow(real, cmap="gray"); axes[0][c].set_title(f"{fidx:05d}", fontsize=8)
            axes[1][c].imshow(to_uint8(mean_img))
            im = axes[2][c].imshow(std_map.numpy(), cmap="magma", vmin=smin, vmax=smax)
            for r in range(3):
                axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        axes[0][0].set_ylabel("real cond", fontsize=9)
        axes[1][0].set_ylabel(f"mean of {N_SAMPLES}", fontsize=9)
        axes[2][0].set_ylabel("per-pixel std", fontsize=9)
        fig.suptitle(f"[{tag}] {name}  (N={N_SAMPLES} samples/image)", fontsize=11)
        fig.colorbar(im, ax=axes[2, :].tolist(), fraction=0.046, pad=0.02)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(out_dir, f"montage_{name.replace(' ', '_')}.png"),
                    dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"  post-processed -> {out_dir}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    post = "--post" in args
    args = [a for a in args if a != "--post"]
    which = args or list(MODELS.keys())
    for tag in which:
        (post_model if post else run_model)(tag, MODELS[tag])
    print("\nDONE.")
