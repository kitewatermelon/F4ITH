#!/usr/bin/env python
"""Label-stratified 100x sampling for all RCT encoder/conditioning variants.

For each of the 14 CheXpert labels, pick the first 5 real conditioning images
(by filename) that are positive for that label -> up to 70 conditioning images,
shared across every model (real dirs are byte-identical). Fixing the noise seed
per conditioning image, generate 100 samples (noise varies within the 100) and
compute, per conditioning image:

  sample-mean   = mean_i(sample_i)                         (RGB image)
  sample-std    = std_i(sample_i)                          (map)
  err-signed    = mean_i(sample_i) - real                  (map, diverging)
  err-abs (MAE) = mean_i |sample_i - real|                 (map)
  err-std       = std_i(sample_i - real)  (== sample-std)  (map)

Raw 100 samples, individual full-res PNGs per row-type, npy maps, a 6-row x
5-col montage per label, and per-model summaries are written under
/mnt/nvme1/RCT/main_result/. Labels of every selected image go to
selected_labels.csv at the top level.

Sampling config: last.ckpt, ema2, cfg from checkpoint (1.5), 224px / 50 steps.
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
SEED_PERM   = 0          # matches build_real_set --seed default
NUM_IMAGES  = 3403
N_SAMPLES   = 100        # samplings per conditioning image
PER_LABEL   = 5          # conditioning images per label
BATCH       = 50         # samples per generate call
NOISE_SEED  = 0          # per-image noise seed (shared across models)
MIMIC_ROOT  = "/mnt/nvme1/mimic-cxr-jpg"
OUT_ROOT    = "/mnt/nvme1/RCT/main_result"
SRC_ROOT    = "output/main_result"

MODELS = {
    "pooled_chexworld":  "rct_pooled_224_chexworld_v2",
    "pooled_raddino":    "rct_pooled_224_raddino",
    "pooled_densenet":   "rct_pooled_224_xrv_densenet",
    "patch_chexworld":   "rct_token_224_p14_chexworld_v2",
    "patch_raddino":     "rct_token_224_p14_raddino",
    "patch_densenet":    "rct_token_224_p14_xrv_densenet",
}
LABELS = MIMICCXRDataset.CHEXPERT_LABELS


def ckpt(dirname):
    return os.path.join(SRC_ROOT, dirname, "checkpoints", "last.ckpt")


def real_dir(dirname):
    return os.path.join(SRC_ROOT, dirname, "fid_eval", "real")


# ---- selection + labels (computed once, shared across models) --------------
def build_label_matrix(ref_real_dir):
    ds = MIMICCXRDataset(root=MIMIC_ROOT, split="test",
                         frontal_only=True, uncertainty="zero")
    g = th.Generator().manual_seed(SEED_PERM)
    perm = th.randperm(len(ds), generator=g)[:NUM_IMAGES].tolist()
    files = sorted(int(os.path.splitext(f)[0])
                   for f in os.listdir(ref_real_dir) if f.endswith(".png"))
    mat = np.zeros((len(files), len(LABELS)), dtype=int)
    for row, i in enumerate(files):
        lab = ds.df.iloc[perm[i]][LABELS].values.astype(float)
        mat[row] = (lab == 1).astype(int)
    return files, mat


def select_per_label(files, mat):
    sel = {}
    for j, name in enumerate(LABELS):
        pos = [files[r] for r in range(len(files)) if mat[r, j] == 1]
        sel[name] = pos[:PER_LABEL]
    return sel


def write_selected_labels(files, mat, sel):
    os.makedirs(OUT_ROOT, exist_ok=True)
    fidx_to_row = {f: r for r, f in enumerate(files)}
    # every (label, fidx) chosen; a fidx may recur across labels (multi-label)
    rows = []
    seen = set()
    for name in LABELS:
        for fidx in sel[name]:
            key = fidx
            r = fidx_to_row[fidx]
            positives = [LABELS[j] for j in range(len(LABELS)) if mat[r, j] == 1]
            rows.append((name, fidx, mat[r], positives))
    with open(os.path.join(OUT_ROOT, "selected_labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sel_label", "file_idx"] + LABELS + ["positive_labels"])
        for name, fidx, vec, positives in rows:
            w.writerow([name, f"{fidx:05d}"] + [int(v) for v in vec] + ["|".join(positives)])
    print(f"  selected {len(rows)} (label,image) entries; "
          f"{len(set(f for _, f, _, _ in rows))} unique images -> selected_labels.csv",
          flush=True)


# ---- sampling -------------------------------------------------------------
def load_cond(path, img_size):
    im = Image.open(path).convert("RGB")
    x = TF.to_tensor(TF.resize(im, [img_size, img_size], antialias=True))
    return (x * 2.0 - 1.0).unsqueeze(0).to("cuda", th.float16)


@th.no_grad()
def sample_stack(rct, pooled, patches, n=N_SAMPLES, batch=BATCH, seed=NOISE_SEED):
    """n samplings (noise varies) -> raw stack [n,3,H,W] in [0,1]."""
    th.manual_seed(seed)
    parts, done = [], 0
    with th.amp.autocast("cuda"):
        while done < n:
            b = min(batch, n - done)
            dc = pooled[:1].expand(b, *pooled.shape[1:])
            pt = patches[:1].expand(b, *patches.shape[1:]).contiguous()
            g = rct.generate_from_features(dc, pt)              # [b,3,H,W]
            parts.append(((g.float() + 1) / 2).clamp(0, 1).cpu())
            done += b
    return th.cat(parts)                                        # [n,3,H,W]


# ---- image helpers --------------------------------------------------------
def chw_to_uint8(chw):
    a = chw.clamp(0, 1).permute(1, 2, 0).numpy()
    return (a * 255).round().astype(np.uint8)


def map_to_rgb(m, vmin, vmax, cmap="magma"):
    norm = np.clip((m - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    return (matplotlib.colormaps[cmap](norm)[..., :3] * 255).round().astype(np.uint8)


def signed_to_rgb(m, vabs, cmap="RdBu_r"):
    norm = np.clip((m + vabs) / max(2 * vabs, 1e-8), 0.0, 1.0)
    return (matplotlib.colormaps[cmap](norm)[..., :3] * 255).round().astype(np.uint8)


# ---- per-model run --------------------------------------------------------
def run_model(tag, dirname, sel):
    print(f"\n===== MODEL: {tag} ({dirname}) =====", flush=True)
    rdir = real_dir(dirname)
    rct, _ = build_rct_model(ckpt_path=ckpt(dirname), device="cuda", ema_variant="ema2")
    img_size = rct.img_size
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    # compute once per unique fidx (deterministic: same noise seed)
    cache = {}   # fidx -> dict of maps + real tensor
    for name in LABELS:
        for fidx in sel[name]:
            if fidx in cache:
                continue
            cond = load_cond(os.path.join(rdir, f"{fidx:05d}.png"), img_size)
            with th.no_grad(), th.amp.autocast("cuda"):
                pooled, patches = rct.encode_cond(cond)
            stack = sample_stack(rct, pooled, patches)          # [n,3,H,W]
            real = TF.to_tensor(TF.resize(
                Image.open(os.path.join(rdir, f"{fidx:05d}.png")).convert("RGB"),
                [img_size, img_size], antialias=True))          # [3,H,W] in [0,1]
            mean_img = stack.mean(0)                             # [3,H,W]
            std_map  = stack.std(0).mean(0)                      # [H,W]
            err = stack - real.unsqueeze(0)                     # [n,3,H,W]
            err_signed = err.mean(0).mean(0)                    # [H,W] signed
            err_abs    = err.abs().mean(0).mean(0)              # [H,W] MAE
            err_std    = err.std(0).mean(0)                     # [H,W] (== std_map)
            cache[fidx] = dict(stack=stack, real=real, mean=mean_img, std=std_map,
                               err_signed=err_signed, err_abs=err_abs, err_std=err_std)
            print(f"  {fidx:05d} | std={float(std_map.mean()):.4f} "
                  f"MAE={float(err_abs.mean()):.4f} "
                  f"bias={float(err_signed.mean()):+.4f}", flush=True)

    # shared colormap ranges for this model
    all_std = th.stack([cache[f]["std"] for f in cache])
    all_abs = th.stack([cache[f]["err_abs"] for f in cache])
    all_sig = th.stack([cache[f]["err_signed"] for f in cache])
    smin, smax = float(all_std.min()), float(all_std.max())
    amin, amax = float(all_abs.min()), float(all_abs.max())
    sig_abs = float(all_sig.abs().max())
    print(f"  ranges: std[{smin:.4f},{smax:.4f}] mae[{amin:.4f},{amax:.4f}] "
          f"signed +/-{sig_abs:.4f}", flush=True)

    # write per (label, fidx): raw stack + individual pngs + npy
    summary = []
    for name in LABELS:
        for fidx in sel[name]:
            c = cache[fidx]
            d = os.path.join(out_dir, name.replace(" ", "_"), f"{fidx:05d}")
            os.makedirs(d, exist_ok=True)
            # raw 100 samples
            for k in range(c["stack"].shape[0]):
                Image.fromarray(chw_to_uint8(c["stack"][k])).save(
                    os.path.join(d, f"sample_{k:03d}.png"))
            # individual row pngs (full-res)
            Image.fromarray(chw_to_uint8(c["real"]).squeeze()).save(os.path.join(d, "real.png"))
            Image.fromarray(chw_to_uint8(c["mean"])).save(os.path.join(d, "mean.png"))
            Image.fromarray(map_to_rgb(c["std"].numpy(), smin, smax)).save(os.path.join(d, "std.png"))
            Image.fromarray(signed_to_rgb(c["err_signed"].numpy(), sig_abs)).save(os.path.join(d, "err_signed.png"))
            Image.fromarray(map_to_rgb(c["err_abs"].numpy(), amin, amax)).save(os.path.join(d, "err_abs.png"))
            Image.fromarray(map_to_rgb(c["err_std"].numpy(), smin, smax)).save(os.path.join(d, "err_std.png"))
            # npy maps
            np.save(os.path.join(d, "mean.npy"), c["mean"].numpy())
            np.save(os.path.join(d, "std.npy"), c["std"].numpy())
            np.save(os.path.join(d, "err_signed.npy"), c["err_signed"].numpy())
            np.save(os.path.join(d, "err_abs.npy"), c["err_abs"].numpy())
            np.save(os.path.join(d, "err_std.npy"), c["err_std"].numpy())
            summary.append((name, fidx, float(c["std"].mean()),
                            float(c["err_abs"].mean()), float(c["err_signed"].mean())))

    # per-label montage: 6 rows x 5 cols
    row_specs = [
        ("real", "gray", None), ("sample-mean", "rgb", None), ("sample-std", "magma", (smin, smax)),
        ("err-signed", "RdBu_r", (-sig_abs, sig_abs)), ("err-abs(MAE)", "magma", (amin, amax)),
        ("err-std", "magma", (smin, smax)),
    ]
    for name in LABELS:
        fidxs = sel[name]
        if not fidxs:
            continue
        ncol = len(fidxs)
        fig, axes = plt.subplots(6, ncol, figsize=(2.4 * ncol, 14.4), squeeze=False)
        last_im = {}
        for cix, fidx in enumerate(fidxs):
            c = cache[fidx]
            panels = [c["real"], c["mean"], c["std"], c["err_signed"], c["err_abs"], c["err_std"]]
            for rix, (label, cmap, rng) in enumerate(row_specs):
                ax = axes[rix][cix]
                if cmap == "gray":
                    ax.imshow(chw_to_uint8(panels[rix]).squeeze(), cmap="gray")
                elif cmap == "rgb":
                    ax.imshow(chw_to_uint8(panels[rix]))
                else:
                    vmin, vmax = rng
                    last_im[rix] = ax.imshow(panels[rix].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if rix == 0:
                    ax.set_title(f"{fidx:05d}", fontsize=8)
        for rix, (label, _, _) in enumerate(row_specs):
            axes[rix][0].set_ylabel(label, fontsize=9)
        fig.suptitle(f"[{tag}] {name}  (N={N_SAMPLES} samples/image)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(os.path.join(out_dir, f"montage_{name.replace(' ', '_')}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "file_idx", "std_mean", "mae_mean", "bias_mean"])
        for name, fidx, sm, mae, bias in summary:
            w.writerow([name, f"{fidx:05d}", f"{sm:.5f}", f"{mae:.5f}", f"{bias:+.5f}"])

    with open(os.path.join(out_dir, "per_label_std.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n", "std_mean_avg", "std_mean_sd", "mae_avg", "mae_sd"])
        for name in LABELS:
            svals = [s[2] for s in summary if s[0] == name]
            mvals = [s[3] for s in summary if s[0] == name]
            if svals:
                w.writerow([name, len(svals), f"{np.mean(svals):.5f}", f"{np.std(svals):.5f}",
                            f"{np.mean(mvals):.5f}", f"{np.std(mvals):.5f}"])

    del rct, cache
    th.cuda.empty_cache()
    print(f"  wrote -> {out_dir}", flush=True)


if __name__ == "__main__":
    ref = real_dir(MODELS["pooled_chexworld"])
    files, mat = build_label_matrix(ref)
    sel = select_per_label(files, mat)
    write_selected_labels(files, mat, sel)

    which = sys.argv[1:] or list(MODELS.keys())
    for tag in which:
        run_model(tag, MODELS[tag], sel)
    print("\nDONE.")
