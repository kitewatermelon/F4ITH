#!/usr/bin/env python3
"""
Representation-space distance metrics between a real-image directory and its
reconstructions (generated conditioned on the encoder's feature of each real
image). Complements scripts/quality_metrics.py: instead of distributional
realism it measures per-image representation *preservation*.

Measure in the space the model was conditioned on: --ssl_model must name the
same encoder the checkpoint used (chexworld for the CheXWorld runs,
xrv_densenet121 for the DenseNet runs). Features cache per encoder, so switching
does not reuse another encoder's vectors.

Let f = the pooled encoder, h = f(x) for real image x, h' = f(x') for its
reconstruction x'. Reports:
    1. CosSim(h, h')                 -- mean +/- std per-pair cosine similarity
    2. Mean pairwise L2 among {h}    -- spread of conditioning representations
    3. Mean paired L2 (h_i, h'_i)    -- reconstruction drift (smaller = better)
    4. Mean rank  (both directions x {cosine, L2})
    5. MRR        (both directions x {cosine, L2})

Rank directions:
    gen->cond : for each h'_i, rank of the true h_i among all {h_j}
    cond->gen : for each h_i, rank of the true h'_i among all {h'_j}

Usage:
    # RCT / RCDM (native resolution)
    python scripts/distance_metrics.py \
        --real_dir output/ablation_t_emb_token_128_chexworld/fid_eval/real \
        --gen_dir  output/ablation_t_emb_token_128_chexworld/fid_eval/gen \
        --out      output/ablation_t_emb_token_128_chexworld/fid_eval/distance_metrics.txt

    # TorchXRayVision DenseNet-121 conditioned run
    python scripts/distance_metrics.py \
        --real_dir output/rct_token_224_p14_xrv_densenet/fid_eval/real \
        --gen_dir  output/rct_token_224_p14_xrv_densenet/fid_eval/gen \
        --ssl_model xrv_densenet121

    # FeatInv (resize both sets to 224 for the CheXWorld encoder)
    python scripts/distance_metrics.py \
        --real_dir baseline/FeatInv/output/fid_eval/real \
        --gen_dir  baseline/FeatInv/output/fid_eval/gen \
        --resize 224 \
        --out      baseline/FeatInv/output/fid_eval/distance_metrics.txt
"""
import os
import sys
import argparse

import numpy as np
import torch as th
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
RCDM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baseline", "RCDM")
sys.path.insert(0, RCDM_ROOT)


def list_pngs(png_dir: str) -> list:
    return sorted(f for f in os.listdir(png_dir) if f.endswith(".png"))


# ---------------------------------------------------------------------------
# feature extraction (with on-disk cache)
# ---------------------------------------------------------------------------

@th.no_grad()
def _encode_dir(png_dir: str, ssl_model, device: th.device, batch_size: int,
                resize: int) -> np.ndarray:
    files = list_pngs(png_dir)
    feats = []
    for start in range(0, len(files), batch_size):
        imgs = []
        for f in files[start:start + batch_size]:
            img = Image.open(os.path.join(png_dir, f)).convert("RGB")
            t = TF.to_tensor(img)                      # [0,1]
            if resize and (t.shape[-1] != resize or t.shape[-2] != resize):
                t = TF.resize(t, [resize, resize], antialias=True)
            imgs.append(t * 2.0 - 1.0)                 # [-1,1] (encoder contract)
        x = th.stack(imgs).to(device)
        feats.append(ssl_model(x).float().cpu().numpy())
        if (start // batch_size) % 20 == 0:
            print(f"    encoded {min(start + batch_size, len(files))}/{len(files)}", flush=True)
    return np.concatenate(feats, axis=0)


def _cache_paths(png_dir: str, tag: str, resize: int, model: str):
    # Keyed by encoder: two encoders' features must never share a cache file.
    # "chexworld" reproduces the historical chexworld_feats_* names.
    parent = os.path.dirname(png_dir.rstrip("/"))
    suffix = f"_r{resize}" if resize else ""
    base = os.path.join(parent, f"{model}_feats_{tag}{suffix}")
    return base + ".npy", base + ".txt"


def get_features(png_dir: str, tag: str, ssl_model, device: th.device,
                 batch_size: int, resize: int, recompute: bool, model: str) -> np.ndarray:
    """Encode a dir to (N, D) pooled features, caching to <parent>/<model>_feats_<tag>.npy.
    The cache is reused only when its stored filename list matches the current dir."""
    npy_path, txt_path = _cache_paths(png_dir, tag, resize, model)
    files = list_pngs(png_dir)

    if not recompute and os.path.exists(npy_path) and os.path.exists(txt_path):
        with open(txt_path) as fh:
            cached_files = fh.read().split()
        if cached_files == files:
            print(f"  [{tag}] loaded cached features <- {npy_path}", flush=True)
            return np.load(npy_path)
        print(f"  [{tag}] cache stale (filename list changed), re-encoding", flush=True)

    print(f"  [{tag}] encoding {len(files)} images from {png_dir}"
          f"{f' (resize {resize})' if resize else ''}", flush=True)
    feats = _encode_dir(png_dir, ssl_model, device, batch_size, resize)
    np.save(npy_path, feats)
    with open(txt_path, "w") as fh:
        fh.write("\n".join(files))
    print(f"  [{tag}] cached features -> {npy_path}", flush=True)
    return feats


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def rank_and_mrr(H: th.Tensor, Hp: th.Tensor, metric: str, chunk: int = 2048) -> tuple:
    """Rank of each true pair (i,i) when, for every query row in the first arg,
    the candidates in the second arg are sorted ascending by distance.
    metric: "cosine" (1 - cos sim, on L2-normalized rows) or "l2".
    Computed in row chunks so memory is O(chunk x N), not O(N^2) -- required for
    large candidate pools (e.g. the full NIH test set, ~25.6k).
    Returns (mean_rank, mrr); rank is 1-indexed = 1 + (#candidates strictly closer)."""
    n = H.shape[0]
    A = F.normalize(H, dim=1) if metric == "cosine" else H
    B = F.normalize(Hp, dim=1) if metric == "cosine" else Hp
    ranks = th.empty(n, device=H.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = 1.0 - A[s:e] @ B.T if metric == "cosine" else th.cdist(A[s:e], B)  # (b, N)
        true = d[th.arange(e - s, device=H.device), th.arange(s, e, device=H.device)].unsqueeze(1)
        ranks[s:e] = 1 + (d < true).sum(dim=1).float()
    return ranks.mean().item(), (1.0 / ranks).mean().item()


def _pairwise_l2_mean(H: th.Tensor, chunk: int = 2048) -> float:
    """Mean Euclidean distance over all i<j pairs, in row chunks (O(chunk x N) memory)."""
    n = H.shape[0]
    total, count = 0.0, 0
    ar = th.arange(n, device=H.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = th.cdist(H[s:e], H)                                   # (b, N)
        mask = ar.unsqueeze(0) > ar[s:e].unsqueeze(1)             # keep j > i only
        total += d[mask].sum().item()
        count += int(mask.sum().item())
    return total / count


def compute_metrics(H: np.ndarray, Hp: np.ndarray, device: th.device, chunk: int = 2048) -> dict:
    H_t = th.from_numpy(H).float().to(device)
    Hp_t = th.from_numpy(Hp).float().to(device)
    res = {}

    # 1. cosine similarity of paired (h_i, h'_i)
    cos = F.cosine_similarity(H_t, Hp_t, dim=1)
    res["CosSim_mean"] = cos.mean().item()
    res["CosSim_std"] = cos.std().item()

    # 2. mean pairwise L2 among conditioning reps {h} (off-diagonal, i<j)
    res["PairwiseL2_realreal"] = _pairwise_l2_mean(H_t, chunk)

    # 3. mean paired L2 of (h_i, h'_i) (diagonal)
    res["PairedL2_realgen"] = (H_t - Hp_t).norm(dim=1).mean().item()

    # 4/5. rank + MRR, both directions x {cosine, l2}
    for metric in ("cosine", "l2"):
        # gen->cond: query = h', candidates = h  -> rank of true h_i
        mr, mrr = rank_and_mrr(Hp_t, H_t, metric, chunk)
        res[f"Rank_gen2cond_{metric}"] = mr
        res[f"MRR_gen2cond_{metric}"] = mrr
        # cond->gen: query = h, candidates = h'  -> rank of true h'_i
        mr, mrr = rank_and_mrr(H_t, Hp_t, metric, chunk)
        res[f"Rank_cond2gen_{metric}"] = mr
        res[f"MRR_cond2gen_{metric}"] = mrr

    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", required=True)
    p.add_argument("--gen_dir", required=True)
    p.add_argument("--out", default=None, help="Results text file (default: <gen_dir>/../distance_metrics.txt)")
    p.add_argument("--ssl_model", default="chexworld",
                   choices=["chexworld", "xrv_densenet121", "raddino"],
                   help="Pooled encoder defining the representation space. Use the same "
                        "backbone the evaluated checkpoint was conditioned on (default: chexworld)")
    p.add_argument("--resize", type=int, default=None,
                   help="Resize both sets to NxN before the encoder (FeatInv: 224). Default: native.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--recompute", action="store_true", help="Ignore cached features and re-encode")
    p.add_argument("--chunk", type=int, default=2048,
                   help="Row-chunk size for the O(NxN) rank/pairwise ops (keeps memory O(chunk x N))")
    args = p.parse_args()

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    out_path = args.out or os.path.join(os.path.dirname(args.gen_dir.rstrip("/")), "distance_metrics.txt")

    real_files, gen_files = list_pngs(args.real_dir), list_pngs(args.gen_dir)
    print(f"real: {len(real_files)} images ({args.real_dir})")
    print(f"gen:  {len(gen_files)} images ({args.gen_dir})")
    if real_files != gen_files:
        raise SystemExit("real_dir and gen_dir must contain the same filenames (paired reconstructions).")

    print(f"\n=== Loading {args.ssl_model} encoder ===", flush=True)
    from guided_diffusion_rcdm.get_ssl_models import get_model
    ssl_model = get_model(args.ssl_model, pool=True).to(device).eval()
    for pm in ssl_model.parameters():
        pm.requires_grad_(False)

    print(f"\n=== Extracting features{f' (resize {args.resize})' if args.resize else ''} ===", flush=True)
    H = get_features(args.real_dir, "real", ssl_model, device, args.batch_size, args.resize,
                     args.recompute, args.ssl_model)
    Hp = get_features(args.gen_dir, "gen", ssl_model, device, args.batch_size, args.resize,
                      args.recompute, args.ssl_model)

    print(f"\n=== Computing metrics (N={len(H)}, dim={H.shape[1]}) ===", flush=True)
    results = compute_metrics(H, Hp, device, args.chunk)

    print("\n" + "=" * 46)
    print("Summary")
    print("=" * 46)
    for k, v in results.items():
        print(f"  {k:24s} {v:12.6f}")

    with open(out_path, "w") as f:
        f.write(f"real_dir={args.real_dir} ({len(real_files)} images)\n")
        f.write(f"gen_dir={args.gen_dir} ({len(gen_files)} images)\n")
        f.write(f"ssl_model={args.ssl_model}\n")
        if args.resize:
            f.write(f"resize={args.resize}\n")
        for k, v in results.items():
            f.write(f"{k} = {v:.6f}\n")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
