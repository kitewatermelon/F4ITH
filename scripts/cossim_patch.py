#!/usr/bin/env python
"""Position-aligned per-patch CosSim (new metric).

For each (real, gen) image pair: encode BOTH through the same encoder in
spatial mode (pool=False -> (N, D) patch tokens), L2-normalize each patch,
take the cosine of same-position patches, average over the N patches -> one
per-image score; then average over images. Real and gen are both resized to
the encoder's native input (224) so their patch grids align position-by-position.

Usage: python scripts/cossim_patch.py --dataset chexpert   # or mimic / nih
Writes per-job cossim_patch.txt next to each gen dir and prints a summary.
"""
import os, sys, argparse, json
import numpy as np
import torch as th
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline", "RCDM"))
from guided_diffusion_rcdm.get_ssl_models import get_model

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def A(*p): return os.path.join(BASE, *p)

# dataset -> dir suffix for the standard {method}/{suffix}/{real,gen} layout
SUF = {"mimic": "fid_eval", "nih": "fid_eval_nih", "chexpert": "fid_eval_chexpert"}
# FeatInv real is always the cfg9 conditioning set; gen comes from the variant dir
FI = A("baseline", "FeatInv", "output")

def jobs_for(ds):
    s = SUF[ds]
    fi_real = f"{FI}/{s}/real"
    fi_pool_gen = {"mimic":"fid_eval_pooled","nih":"fid_eval_nih_pooled","chexpert":"fid_eval_chexpert_pooled"}[ds]
    fi_unp_gen  = {"mimic":"fid_eval_cfg1.75","nih":"fid_eval_nih_cfg1.75","chexpert":"fid_eval_chexpert_cfg1.75"}[ds]
    J = []
    # label, encoder, real_dir, gen_dir
    J += [("RCDM","chexworld", A("output/main_result/rcdm_chexworld_cxr",s,"real"), A("output/main_result/rcdm_chexworld_cxr",s,"gen"))]
    J += [("FeatInv-pooled","chexworld", fi_real, f"{FI}/{fi_pool_gen}/gen")]
    J += [("F4ITH-GFC","chexworld", A("output/main_result/rct_pooled_224_chexworld_v2",s,"real"), A("output/main_result/rct_pooled_224_chexworld_v2",s,"gen"))]
    J += [("FeatInv-unpooled","chexworld", fi_real, f"{FI}/{fi_unp_gen}/gen")]
    J += [("F4ITH-SFC","chexworld", A("output/main_result/rct_token_224_p14_chexworld_v2",s,"real"), A("output/main_result/rct_token_224_p14_chexworld_v2",s,"gen"))]
    J += [("RAD-DINO-GFC","raddino", A("output/main_result/rct_pooled_224_raddino",s,"real"), A("output/main_result/rct_pooled_224_raddino",s,"gen"))]
    J += [("RAD-DINO-SFC","raddino", A("output/main_result/rct_token_224_p14_raddino",s,"real"), A("output/main_result/rct_token_224_p14_raddino",s,"gen"))]
    J += [("XRV-GFC","xrv_densenet121", A("output/main_result/rct_pooled_224_xrv_densenet",s,"real"), A("output/main_result/rct_pooled_224_xrv_densenet",s,"gen"))]
    J += [("XRV-SFC","xrv_densenet121", A("output/main_result/rct_token_224_p14_xrv_densenet",s,"real"), A("output/main_result/rct_token_224_p14_xrv_densenet",s,"gen"))]
    return J

def _load(dir_, f, resize=224):
    t = TF.to_tensor(Image.open(os.path.join(dir_, f)).convert("RGB"))
    if t.shape[-1] != resize or t.shape[-2] != resize:
        t = TF.resize(t, [resize, resize], antialias=True)
    return t * 2.0 - 1.0

class PairDS(Dataset):
    def __init__(self, real_dir, gen_dir, files, resize=224):
        self.rd, self.gd, self.files, self.rs = real_dir, gen_dir, files, resize
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        f = self.files[i]
        return _load(self.rd, f, self.rs), _load(self.gd, f, self.rs)

@th.no_grad()
def score(enc, real_dir, gen_dir, device, bs=64, workers=8):
    rf = set(f for f in os.listdir(real_dir) if f.endswith(".png"))
    gf = set(f for f in os.listdir(gen_dir) if f.endswith(".png"))
    files = sorted(rf & gf)
    assert files, f"no shared files: {real_dir} vs {gen_dir}"
    dl = DataLoader(PairDS(real_dir, gen_dir, files), batch_size=bs,
                    num_workers=workers, pin_memory=True, persistent_workers=False)
    per_img = []; done = 0
    for r, g in dl:
        hr = enc(r.to(device, non_blocking=True))        # (B,N,D)
        hg = enc(g.to(device, non_blocking=True))         # (B,N,D)
        hr = F.normalize(hr, dim=2); hg = F.normalize(hg, dim=2)
        cos_patch = (hr * hg).sum(dim=2)                  # (B,N)
        per_img.append(cos_patch.mean(dim=1).float().cpu())
        done += r.shape[0]
        if (done // bs) % 20 == 0:
            print(f"    {done}/{len(files)}", flush=True)
    v = th.cat(per_img)
    return v.mean().item(), v.std().item(), len(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SUF))
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    device = "cuda" if th.cuda.is_available() else "cpu"
    J = jobs_for(args.dataset)
    # group by encoder to load each once
    results = {}
    for enc_name in ["chexworld", "raddino", "xrv_densenet121"]:
        sub = [j for j in J if j[1] == enc_name]
        if not sub: continue
        print(f"\n=== load encoder {enc_name} (pool=False) ===", flush=True)
        enc = get_model(enc_name, pool=False).to(device).eval()
        for p in enc.parameters(): p.requires_grad_(False)
        for label, _, rd, gd in sub:
            print(f"[{args.dataset}] {label}: {gd}", flush=True)
            m, sd, n = score(enc, rd, gd, device, args.bs)
            results[label] = {"cossim_patch_mean": m, "cossim_patch_std": sd, "n": n}
            out = os.path.join(os.path.dirname(gd.rstrip("/")), "cossim_patch.txt")
            with open(out, "w") as f:
                f.write(f"metric=position_aligned_patch_cosine_mean\nencoder={enc_name}\n"
                        f"real_dir={rd}\ngen_dir={gd}\nn={n}\n"
                        f"CosSim_patch_mean={m:.6f}\nCosSim_patch_std={sd:.6f}\n")
            print(f"    -> CosSim_patch={m:.4f} (n={n}) written {out}", flush=True)
        del enc; th.cuda.empty_cache()
    outj = A("output", f"cossim_patch_{args.dataset}.json")
    with open(outj, "w") as f: json.dump(results, f, indent=2)
    print("\n==== SUMMARY", args.dataset, "====")
    for k, v in results.items():
        print(f"  {k:20s} {v['cossim_patch_mean']:.4f}  (n={v['n']})")
    print("written", outj)

if __name__ == "__main__":
    main()
