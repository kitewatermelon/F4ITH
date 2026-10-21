#!/usr/bin/env python3
"""
Compute FID and FRD between a real-image
directory and a generated-image directory of matching PNGs (same filenames).
Model-agnostic: works on the output of scripts/eval_generation.py for any
model type (RCDM, RCT, ...).

Usage:
    python scripts/quality_metrics.py \
        --real_dir /home/yspark/RCT/output/rcdm_chexworld_cxr/fid_eval/real \
        --gen_dir  /home/yspark/RCT/output/rcdm_chexworld_cxr/fid_eval/gen \
        --out      /home/yspark/RCT/output/rcdm_chexworld_cxr/fid_eval/metrics.txt
"""
import os
import argparse
import shutil
import tempfile

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

def list_pngs(png_dir: str) -> list:
    return sorted(f for f in os.listdir(png_dir) if f.endswith(".png"))


def resize_dir(src_dir: str, dst_dir: str, size: int) -> str:
    """Bicubic-resize every PNG in src_dir to size x size under dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    for f in list_pngs(src_dir):
        img = Image.open(os.path.join(src_dir, f))
        if img.size != (size, size):
            img = img.resize((size, size), Image.BICUBIC)
        img.save(os.path.join(dst_dir, f))
    return dst_dir


# ---------------------------------------------------------------------------
# FID (Inception features via torch_fidelity)
# ---------------------------------------------------------------------------

def compute_fid(real_dir: str, gen_dir: str, cuda: bool) -> float:
    import torch_fidelity
    metrics = torch_fidelity.calculate_metrics(
        input1=real_dir, input2=gen_dir, fid=True, cuda=cuda, verbose=False,
    )
    return metrics["frechet_inception_distance"]


# ---------------------------------------------------------------------------
# FRD (Frechet Radiomic Distance) -- reference implementation from
# https://github.com/RichardObi/frd-score (pip: frd-score), FRDv1 default:
# PyRadiomics Original+LoG+Wavelet features, z-score normalized against the
# real set, log-transformed Frechet distance.
# ---------------------------------------------------------------------------

def compute_frd(real_dir: str, gen_dir: str) -> float:
    import frd_score
    return frd_score.compute_frd([real_dir, gen_dir])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", required=True)
    p.add_argument("--gen_dir", required=True)
    p.add_argument("--out", default=None, help="Results text file (default: <gen_dir>/../metrics.txt)")
    p.add_argument("--resize", type=int, default=None,
                   help="Resize both sets to NxN before computing metrics, so runs at "
                        "different native resolutions are comparable (e.g. 224)")
    p.add_argument("--metrics", nargs="*", default=["fid", "frd"],
                    choices=["fid", "frd"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = args.out or os.path.join(os.path.dirname(args.gen_dir.rstrip("/")), "metrics.txt")

    n_real, n_gen = len(list_pngs(args.real_dir)), len(list_pngs(args.gen_dir))
    print(f"real: {n_real} images ({args.real_dir})")
    print(f"gen:  {n_gen} images ({args.gen_dir})")

    src_real_dir, src_gen_dir = args.real_dir, args.gen_dir
    tmp_dir = None
    if args.resize:
        tmp_dir = tempfile.mkdtemp(prefix=f"qm_resize{args.resize}_")
        print(f"\nResizing both sets to {args.resize}x{args.resize} -> {tmp_dir}")
        args.real_dir = resize_dir(src_real_dir, os.path.join(tmp_dir, "real"), args.resize)
        args.gen_dir = resize_dir(src_gen_dir, os.path.join(tmp_dir, "gen"), args.resize)

    results = {}

    if "fid" in args.metrics:
        print("\n=== FID (Inception) ===")
        results["FID"] = compute_fid(args.real_dir, args.gen_dir, cuda=device.type == "cuda")
        print(f"  FID = {results['FID']:.4f}")

    if "frd" in args.metrics:
        print("\n=== FRD (pyradiomics) ===")
        results["FRD"] = compute_frd(args.real_dir, args.gen_dir)
        print(f"  FRD = {results['FRD']:.4f}")

    print("\n" + "=" * 40)
    print("Summary")
    print("=" * 40)
    for k, v in results.items():
        print(f"  {k:12s} {v:10.4f}")

    with open(out_path, "w") as f:
        f.write(f"real_dir={src_real_dir} ({n_real} images)\n")
        f.write(f"gen_dir={src_gen_dir} ({n_gen} images)\n")
        if args.resize:
            f.write(f"resize={args.resize}\n")
        for k, v in results.items():
            f.write(f"{k} = {v:.6f}\n")
    print(f"\nSaved results to {out_path}")

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
