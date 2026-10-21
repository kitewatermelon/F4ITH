#!/usr/bin/env python3
"""
CFG x EMA sweep on the full MIMIC-CXR test set (3403 frontal images).

Two-stage, factorized design (EMA affects weights, CFG affects sampling, so
they are treated as near-independent):

  stage-a   cfg fixed at the training value, sweep ema1 / ema2 / raw
            -> 5 models x 3 variants = 15 configs
  stage-b   best EMA variant per model, sweep the CFG grid
            -> 5 models x 5 cfg = 25 configs

Real Inception features are extracted once and cached, so each config only
pays for its own generated set (torch_fidelity would otherwise re-encode the
3403 real images on every single call).

Usage:
    python scripts/sweep_cfg_ema.py stage-a --gpu 0 --models exp1_baseline exp2_no_conv
    python scripts/sweep_cfg_ema.py stage-b --gpu 0 --models ... --ema_choice ema1
    python scripts/sweep_cfg_ema.py report
"""
import os
import sys
import json
import shutil
import argparse

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rct.dataset import MIMICCXRDataset
from rct.denoiser import RCTDenoiser, resolve_chexworld_compat
from quality_metrics import frechet_distance
from eval_generation import save_png


MODELS = {
    "exp1_baseline":        "ablation_exp1_baseline_chexworld",
    "exp2_no_conv":         "ablation_exp2_no_conv_chexworld",
    "exp3_no_skip":         "ablation_exp3_no_skip_chexworld",
    "exp4_no_conv_no_skip": "ablation_exp4_no_conv_no_skip_chexworld",
    "exp5_h512_no_conv":    "ablation_exp5_h512_no_conv_chexworld",
}

EMA_VARIANTS = ["ema1", "ema2", "raw"]
CFG_GRID = [1.0, 1.5, 2.0, 3.0, 5.0]
IMG_SIZE = 224
MIMIC_ROOT = "/mnt/nvme1/mimic-cxr-jpg"


# ---------------------------------------------------------------------------

def load_denoiser(ckpt_path, device, ema_variant="ema1", cfg=None):
    """ema_variant: 'ema1' | 'ema2' | 'raw'. cfg=None keeps the trained value."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    net_args = argparse.Namespace(
        img_size=hp["img_size"], model_variant=hp["model_variant"],
        attn_dropout=hp["attn_dropout"], proj_dropout=hp["proj_dropout"],
        cond_drop_prob=hp["cond_drop_prob"], P_mean=hp["P_mean"], P_std=hp["P_std"],
        noise_scale=hp["noise_scale"], t_eps=hp["t_eps"],
        ema_decay1=hp["ema_decay1"], ema_decay2=hp["ema_decay2"],
        sampling_method=hp["ode_method"], num_sampling_steps=hp["ode_steps"],
        cfg=hp["cfg"] if cfg is None else cfg,
        interval_min=hp["interval_min"], interval_max=hp["interval_max"],
        long_skip_connection=hp["long_skip_connection"], depth=hp["depth"],
        conditioning_mode=hp["conditioning_mode"], backbone=hp["backbone"],
        chexworld_compat=resolve_chexworld_compat(hp),
        use_final_conv=not hp["no_final_conv"],
        hidden_size=hp.get("hidden_size"), num_heads=hp.get("num_heads"),
    )
    denoiser = RCTDenoiser(net_args)

    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if not k.startswith("denoiser."):
            continue
        state_dict[k[len("denoiser."):].replace("net._orig_mod.", "net.")] = v
    denoiser.load_state_dict(state_dict, strict=True)

    if ema_variant != "raw":
        key = "ema_params1" if ema_variant == "ema1" else "ema_params2"
        ema = ckpt[key]
        with torch.no_grad():
            for n, p in denoiser.named_parameters():
                if p.requires_grad:
                    p.copy_(ema[n.replace("net.", "net._orig_mod.", 1) if n.startswith("net.") else n])

    denoiser.eval()
    return denoiser.to(device)


def build_real(cache_root, force=False):
    """Full MIMIC test set -> PNGs + conditioning tensors."""
    real_dir = os.path.join(cache_root, "real_mimic_full")
    pt_path = os.path.join(cache_root, "real_mimic_full.pt")
    if os.path.exists(pt_path) and not force:
        return real_dir, torch.load(pt_path, map_location="cpu")

    os.makedirs(real_dir, exist_ok=True)
    ds = MIMICCXRDataset(root=MIMIC_ROOT, split="test", transform=None,
                          frontal_only=True, uncertainty="zero")
    print(f"building full MIMIC test reference: {len(ds)} frontal images")

    tensors = []
    for i in range(len(ds)):
        image, _ = ds[i]
        t = TF.to_tensor(TF.resize(image.convert("RGB"), [IMG_SIZE, IMG_SIZE], antialias=True)) * 2 - 1
        tensors.append(t)
        save_png(t, os.path.join(real_dir, f"{i:05d}.png"))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(ds)}")

    stacked = torch.stack(tensors)
    torch.save(stacked, pt_path)
    return real_dir, stacked


# ---------------------------------------------------------------------------
# Inception features (cached for the real set, recomputed per generated set)
# ---------------------------------------------------------------------------

@torch.no_grad()
def inception_features(png_dir, device, batch_size=64):
    """torch_fidelity's InceptionV3 pool3 features — the same backbone its FID uses."""
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    fe = FeatureExtractorInceptionV3(
        "inception", features_list=["2048"],
    ).to(device).eval()

    files = sorted(f for f in os.listdir(png_dir) if f.endswith(".png"))
    feats = []
    for start in range(0, len(files), batch_size):
        imgs = [torch.from_numpy(
                    np.array(Image.open(os.path.join(png_dir, f)).convert("RGB"), dtype=np.uint8)
                ).permute(2, 0, 1) for f in files[start:start + batch_size]]
        x = torch.stack(imgs).to(device)          # uint8 (B,3,H,W) — the extractor rescales internally
        feats.append(fe(x)[0].float().cpu().numpy())

    del fe
    torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0)


def fid_from_features(f_real, f_fake):
    return frechet_distance(f_real.mean(0), np.cov(f_real, rowvar=False),
                             f_fake.mean(0), np.cov(f_fake, rowvar=False))


# ---------------------------------------------------------------------------

@torch.no_grad()
def run_config(model_name, ema_variant, cfg, real, cache_root, device,
               batch_size, real_feats, keep_pngs=False):
    tag = f"{model_name}__{ema_variant}__cfg{cfg}"
    gen_dir = os.path.join(cache_root, "sweep_gen", tag)
    if os.path.exists(gen_dir):
        shutil.rmtree(gen_dir)
    os.makedirs(gen_dir, exist_ok=True)

    ckpt = os.path.join("/home/yspark/RCT/output", MODELS[model_name], "checkpoints", "stepstep=0100000.ckpt")
    denoiser = load_denoiser(ckpt, device, ema_variant=ema_variant, cfg=cfg)

    idx = 0
    for start in range(0, len(real), batch_size):
        batch = real[start:start + batch_size].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen = denoiser.generate(batch)
        for t in gen:
            save_png(t.float(), os.path.join(gen_dir, f"{idx:05d}.png"))
            idx += 1
        print(f"    {tag}: {idx}/{len(real)}", end="\r")
    print()

    del denoiser
    torch.cuda.empty_cache()

    fid = fid_from_features(real_feats, inception_features(gen_dir, device))
    if not keep_pngs:
        shutil.rmtree(gen_dir)   # 75 x 3403 PNGs would be ~40GB otherwise
    return fid


def results_path(cache_root, stage, gpu):
    return os.path.join(cache_root, f"sweep_{stage}_gpu{gpu}.json")


def cmd_stage(args, stage):
    device = torch.device("cuda")
    real_dir, real = build_real(args.cache_root)

    feat_cache = os.path.join(args.cache_root, "real_mimic_full_inception.npy")
    if os.path.exists(feat_cache):
        real_feats = np.load(feat_cache)
    else:
        print("extracting Inception features for the real set (one time)...")
        real_feats = inception_features(real_dir, device)
        np.save(feat_cache, real_feats)
    print(f"real features: {real_feats.shape}")

    out_path = results_path(args.cache_root, stage, args.gpu)
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    for model_name in args.models:
        combos = ([(v, None) for v in EMA_VARIANTS] if stage == "a"
                  else [(args.ema_choice, c) for c in CFG_GRID])
        for ema_variant, cfg in combos:
            cfg_val = 3.0 if cfg is None else cfg   # stage-a uses the trained cfg
            key = f"{model_name}|{ema_variant}|{cfg_val}"
            if key in results and not args.force:
                print(f"  {key} cached — skipping")
                continue
            fid = run_config(model_name, ema_variant, cfg_val, real, args.cache_root,
                             device, args.batch_size, real_feats)
            results[key] = fid
            print(f"  {key}  FID={fid:.3f}")
            json.dump(results, open(out_path, "w"), indent=2)


def cmd_report(args):
    merged = {}
    for stage in ("a", "b"):
        for gpu in (0, 1):
            p = results_path(args.cache_root, stage, gpu)
            if os.path.exists(p):
                merged.update(json.load(open(p)))

    if not merged:
        print("no sweep results found")
        return

    rows = {}
    for key, fid in merged.items():
        model, ema, cfg = key.split("|")
        rows.setdefault(model, {})[(ema, float(cfg))] = fid

    print("\n" + "=" * 78)
    print("MIMIC-CXR full test set (n=3403) — FID sweep")
    print("=" * 78)

    print("\n### Stage A: EMA variant @ cfg=3.0")
    print(f"{'model':<24}{'ema1':>10}{'ema2':>10}{'raw':>10}   best")
    print("-" * 62)
    best_ema = {}
    for model in MODELS:
        if model not in rows:
            continue
        vals = {e: rows[model].get((e, 3.0)) for e in EMA_VARIANTS}
        avail = {e: v for e, v in vals.items() if v is not None}
        if not avail:
            continue
        b = min(avail, key=avail.get)
        best_ema[model] = b
        cells = "".join(f"{vals[e]:>10.2f}" if vals[e] is not None else f"{'-':>10}" for e in EMA_VARIANTS)
        print(f"{model:<24}{cells}   {b}")

    print("\n### Stage B: CFG sweep (best EMA per model)")
    header = "".join(f"{c:>9}" for c in CFG_GRID)
    print(f"{'model':<24}{header}   best")
    print("-" * (24 + 9 * len(CFG_GRID) + 12))
    final = {}
    for model in MODELS:
        if model not in rows:
            continue
        vals = {c: next((v for (e, cc), v in rows[model].items() if cc == c), None) for c in CFG_GRID}
        avail = {c: v for c, v in vals.items() if v is not None}
        if not avail:
            continue
        bc = min(avail, key=avail.get)
        final[model] = (bc, avail[bc])
        cells = "".join(f"{vals[c]:>9.2f}" if vals[c] is not None else f"{'-':>9}" for c in CFG_GRID)
        print(f"{model:<24}{cells}   cfg={bc}")

    if final:
        print("\n### Final ranking (each model at its own best CFG)")
        print(f"{'model':<24}{'FID':>10}{'cfg':>8}{'ema':>8}")
        print("-" * 50)
        for model, (cfg, fid) in sorted(final.items(), key=lambda kv: kv[1][1]):
            print(f"{model:<24}{fid:>10.2f}{cfg:>8}{best_ema.get(model, '?'):>8}")
        winner = min(final, key=lambda m: final[m][1])
        print(f"\nBest overall: {winner}  FID={final[winner][1]:.2f} @ cfg={final[winner][0]}")

    out = os.path.join(args.cache_root, "sweep_summary.json")
    json.dump({"raw": merged, "best_ema": best_ema,
               "best_cfg": {m: c for m, (c, _) in final.items()}}, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["stage-a", "stage-b", "report", "prepare"])
    p.add_argument("--cache_root", default="/mnt/nvme1/rct_sweep")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--ema_choice", default="ema1", choices=EMA_VARIANTS,
                    help="stage-b only: which EMA variant to sweep CFG with")
    p.add_argument("--gpu", type=int, default=0, help="label for the results file (set CUDA_VISIBLE_DEVICES separately)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    os.makedirs(args.cache_root, exist_ok=True)
    if args.mode == "prepare":
        build_real(args.cache_root)
    elif args.mode == "report":
        cmd_report(args)
    else:
        cmd_stage(args, args.mode.split("-")[1])


if __name__ == "__main__":
    main()
