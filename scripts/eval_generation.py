#!/usr/bin/env python3
"""
Generate real/reconstructed PNG sets on the MIMIC-CXR test split for a
trained checkpoint (RCDM, RCT or FeatInv), for use with scripts/quality_metrics.py.

RCDM: --ckpt_path points to a raw EMA state_dict (e.g. ema_0.9999_300000.pt)
from baseline/RCDM/scripts/train_chexworld_cxr.py. Reconstruction is a
~1000-step diffusion sample conditioned on a CheXWorld feature vector of the
real image, so generation is expensive -- use --shard to split across GPUs.

RCT: --ckpt_path points to a src/rct Lightning checkpoint (e.g.
stepstep=0100000.ckpt). Architecture is read from the checkpoint's saved
hyperparameters. Reconstruction is a cheap ~50-step Heun ODE, no sharding
needed.

FeatInv: --ckpt_path points to a baseline/FeatInv ControlNet checkpoint (e.g.
control_featinv_chexworld_fin.ckpt). Reconstruction is a 50-step DDIM sample at
256x256 conditioned on CheXWorld spatial patch tokens, with CFG -- expensive, so
use --shard to split across GPUs.

Usage:
    # RCDM, split across 2 GPUs
    python scripts/eval_generation.py --model_type rcdm \
        --ckpt_path output/rcdm_chexworld_cxr/ema_0.9999_300000.pt \
        --save_path output/rcdm_chexworld_cxr/fid_eval \
        --gpu 0 --shard 0/2
    python scripts/eval_generation.py --model_type rcdm \
        --ckpt_path output/rcdm_chexworld_cxr/ema_0.9999_300000.pt \
        --save_path output/rcdm_chexworld_cxr/fid_eval \
        --gpu 1 --shard 1/2

    # RCT
    python scripts/eval_generation.py --model_type rct \
        --ckpt_path output/ablation_exp1_baseline_chexworld/checkpoints/stepstep=0100000.ckpt \
        --save_path output/ablation_exp1_baseline_chexworld/fid_eval --gpu 0

    # FeatInv, split across 2 GPUs
    python scripts/eval_generation.py --model_type featinv \
        --ckpt_path baseline/FeatInv/output/control_featinv_chexworld_fin.ckpt \
        --save_path baseline/FeatInv/output/fid_eval --gpu 0 --shard 0/2
"""
import os
import sys
import argparse

import torch as th
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
RCDM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baseline", "RCDM")
sys.path.insert(0, RCDM_ROOT)
FEATINV_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baseline", "FeatInv", "code")

from rct.dataset import MIMICCXRDataset, NIHChestXray14Dataset, CheXpertDataset


# Eval split per dataset. CheXpert has no public test labels, so we use its
# validation split; MIMIC/NIH use their held-out test splits.
def make_eval_dataset(dataset: str, mimic_dir: str, nih_dir: str, chexpert_dir: str):
    if dataset == "mimic":
        return MIMICCXRDataset(root=mimic_dir, split="test", frontal_only=True, uncertainty="zero")
    if dataset == "nih":
        return NIHChestXray14Dataset(root=nih_dir, split="test", frontal_only=True)
    if dataset == "chexpert":
        return CheXpertDataset(root=chexpert_dir, split="validate", frontal_only=True, uncertainty="zero")
    raise ValueError(f"unknown dataset {dataset!r}")


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

def save_png(t: th.Tensor, path: str):
    """t: (3, H, W) in [-1, 1] -> uint8 PNG."""
    img = ((t.clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu()
    Image.fromarray(img.permute(1, 2, 0).numpy(), mode="RGB").save(path)


def build_real_set(ds, out_dir: str, num_images: int, img_size: int, seed: int,
                    shard_idx: int = 0, num_shards: int = 1) -> list:
    """Resize `num_images` test images from `ds` to [-1, 1] tensors on disk; return
    [(global_i, tensor), ...] for only this shard's slice of the (seed-deterministic)
    index order, so multiple shards can run in parallel against disjoint filenames."""
    os.makedirs(out_dir, exist_ok=True)

    g = th.Generator().manual_seed(seed)
    n = min(num_images, len(ds))
    indices = th.randperm(len(ds), generator=g)[:n].tolist()

    tensors = []
    for i, idx in enumerate(indices):
        if i % num_shards != shard_idx:
            continue
        image, _ = ds[idx]
        image = image.convert("RGB")
        t = TF.to_tensor(TF.resize(image, [img_size, img_size], antialias=True))  # [0,1]
        t = t * 2.0 - 1.0
        tensors.append((i, t))
        save_png(t, os.path.join(out_dir, f"{i:05d}.png"))
        if len(tensors) % 200 == 0:
            print(f"  {len(tensors)}/{n // num_shards} (shard {shard_idx}/{num_shards})", flush=True)

    print(f"  Cached {len(tensors)} real test images (shard {shard_idx}/{num_shards}) -> {out_dir}", flush=True)
    return tensors


class RealImageDataset(Dataset):
    """Lazy-loading dataset over a directory of {i:05d}.png real images."""
    def __init__(self, real_dir: str, img_size: int, num_images: int,
                 shard_idx: int = 0, num_shards: int = 1):
        all_files = sorted(f for f in os.listdir(real_dir) if f.endswith(".png"))
        self.real_dir = real_dir
        self.img_size = img_size
        self.items = []
        for f in all_files:
            i = int(os.path.splitext(f)[0])
            if i >= num_images or i % num_shards != shard_idx:
                continue
            self.items.append((i, f))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, f = self.items[idx]
        img = Image.open(os.path.join(self.real_dir, f)).convert("RGB")
        t = TF.to_tensor(img)
        if t.shape[-1] != self.img_size or t.shape[-2] != self.img_size:
            t = TF.resize(t, [self.img_size, self.img_size], antialias=True)
        t = t * 2.0 - 1.0
        return i, t


def load_real_set(real_dir: str, img_size: int, num_images: int,
                  shard_idx: int = 0, num_shards: int = 1) -> list:
    """Legacy: load all into memory. Use make_real_dataloader for large sets."""
    ds = RealImageDataset(real_dir, img_size, num_images, shard_idx, num_shards)
    tensors = [(i, t) for i, t in ds]
    print(f"  Loaded {len(tensors)} real conditioning images (shard {shard_idx}/{num_shards}) <- {real_dir}", flush=True)
    return tensors


def make_real_dataloader(real_dir: str, img_size: int, num_images: int,
                         batch_size: int, num_workers: int = 8,
                         shard_idx: int = 0, num_shards: int = 1):
    ds = RealImageDataset(real_dir, img_size, num_images, shard_idx, num_shards)
    print(f"  RealImageDataset: {len(ds)} images (shard {shard_idx}/{num_shards}) <- {real_dir}", flush=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# ---------------------------------------------------------------------------
# RCDM (baseline/RCDM) -- 1000-step diffusion, CheXWorld feature conditioning
# ---------------------------------------------------------------------------

def build_rcdm_model(ckpt_path: str, image_size: int, timestep_respacing: str, ssl_dim: int, device: th.device):
    from guided_diffusion_rcdm.script_util import (
        model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict,
    )
    defaults = model_and_diffusion_defaults()
    defaults.update(
        image_size=image_size, attention_resolutions="32,16,8", num_channels=256, num_heads=4,
        num_res_blocks=2, resblock_updown=True, use_scale_shift_norm=True, learn_sigma=True,
        diffusion_steps=1000, noise_schedule="linear", class_cond=False, use_fp16=False,
        timestep_respacing=timestep_respacing,
    )
    args = argparse.Namespace(**defaults)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        G_shared=True, feat_cond=True, ssl_dim=ssl_dim,
    )
    state_dict = th.load(ckpt_path, map_location="cpu")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), diffusion


@th.no_grad()
def generate_rcdm(args, real_tensors: list, gen_dir: str, device: th.device):
    from guided_diffusion_rcdm.get_ssl_models import get_model

    os.makedirs(gen_dir, exist_ok=True)

    ssl_model = get_model("chexworld", use_head=False).to(device).eval()
    for pm in ssl_model.parameters():
        pm.requires_grad_(False)
    ssl_dim = ssl_model(th.zeros(1, 3, args.image_size, args.image_size, device=device)).size(1)

    model, diffusion = build_rcdm_model(args.ckpt_path, args.image_size, args.timestep_respacing, ssl_dim, device)
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    done = 0
    for start in range(0, len(real_tensors), args.batch_size):
        chunk = real_tensors[start:start + args.batch_size]
        ids = [i for i, _ in chunk]
        batch = th.stack([t for _, t in chunk]).to(device)
        feat = ssl_model(batch).detach()
        sample = sample_fn(
            model, (batch.shape[0], 3, args.image_size, args.image_size),
            clip_denoised=True, model_kwargs={"feat": feat},
        )
        for gid, t in zip(ids, sample):
            save_png(t.float(), os.path.join(gen_dir, f"{gid:05d}.png"))
        done += len(chunk)
        print(f"  generated {done}/{len(real_tensors)} (shard {args.shard_idx}/{args.num_shards})", flush=True)


# ---------------------------------------------------------------------------
# RCT (src/rct) -- ~50-step Heun ODE, DINO/CheXWorld conditioning
# ---------------------------------------------------------------------------

def build_rct_model(ckpt_path: str, device: th.device, ema_variant: str = "ema2", cfg_override: float = None):
    from rct.denoiser import RCTDenoiser, resolve_chexworld_compat

    ckpt = th.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    cfg = hp["cfg"] if cfg_override is None else cfg_override
    if cfg_override is not None:
        print(f"  CFG override: {hp['cfg']} -> {cfg}", flush=True)
    args = argparse.Namespace(
        img_size=hp["img_size"], model_variant=hp["model_variant"],
        attn_dropout=hp["attn_dropout"], proj_dropout=hp["proj_dropout"],
        cond_drop_prob=hp["cond_drop_prob"], P_mean=hp["P_mean"], P_std=hp["P_std"],
        noise_scale=hp["noise_scale"], t_eps=hp["t_eps"],
        ema_decay1=hp["ema_decay1"], ema_decay2=hp["ema_decay2"],
        sampling_method=hp["ode_method"], num_sampling_steps=hp["ode_steps"],
        cfg=cfg, interval_min=hp["interval_min"], interval_max=hp["interval_max"],
        long_skip_connection=hp["long_skip_connection"], depth=hp["depth"],
        conditioning_mode=hp["conditioning_mode"], backbone=hp["backbone"],
        chexworld_compat=resolve_chexworld_compat(hp),
        use_final_conv=not hp["no_final_conv"],
        hidden_size=hp.get("hidden_size"), num_heads=hp.get("num_heads"),
        t_embed_mode=hp.get("t_embed_mode", "adaln"), patch_size=hp.get("patch_size"),
        pooled_cond_source=hp.get("pooled_cond_source", "cls"),
        in_context_len=hp.get("in_context_len", 0),
        in_context_start=hp.get("in_context_start", 4),
    )
    denoiser = RCTDenoiser(args)

    from rct.denoiser import remap_legacy_state_dict

    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if not k.startswith("denoiser."):
            continue
        new_k = k[len("denoiser."):].replace("net._orig_mod.", "net.")
        state_dict[new_k] = v
    # Compat: rename legacy dino-prefixed keys (pre encoder/cond rename).
    state_dict = remap_legacy_state_dict(state_dict)
    denoiser.load_state_dict(state_dict, strict=True)

    if ema_variant != "raw":
        key = "ema_params1" if ema_variant == "ema1" else "ema_params2"
        if key not in ckpt:
            raise KeyError(f"{key} not found in checkpoint (available: {[k for k in ckpt if 'ema' in k]})")
        print(f"  Using {key} (decay {hp.get('ema_decay1' if ema_variant=='ema1' else 'ema_decay2')})", flush=True)
        ema = remap_legacy_state_dict(ckpt[key])  # dino_proj.* -> cond_proj.*
        with th.no_grad():
            for n, p in denoiser.named_parameters():
                if not p.requires_grad:
                    continue
                ema_key = n.replace("net.", "net._orig_mod.", 1) if n.startswith("net.") else n
                p.copy_(ema[ema_key])

    denoiser.eval()
    return denoiser.to(device), hp["img_size"]


@th.no_grad()
def generate_rct(args, real_tensors, gen_dir: str, device: th.device):
    ema_variant = "raw" if args.no_ema else args.ema
    denoiser, _ = build_rct_model(args.ckpt_path, device, ema_variant=ema_variant, cfg_override=args.cfg)
    os.makedirs(gen_dir, exist_ok=True)

    if isinstance(real_tensors, DataLoader):
        total = len(real_tensors.dataset)
        done = 0
        for ids, batch in real_tensors:
            batch = batch.to(device, non_blocking=True)
            with th.amp.autocast("cuda", dtype=th.bfloat16):
                gen = denoiser.generate(batch)
            for gid, t in zip(ids.tolist(), gen):
                save_png(t.float(), os.path.join(gen_dir, f"{gid:05d}.png"))
            done += batch.shape[0]
            print(f"  generated {done}/{total} (shard {args.shard_idx}/{args.num_shards})", flush=True)
    else:
        total = len(real_tensors)
        done = 0
        for start in range(0, total, args.batch_size):
            chunk = real_tensors[start:start + args.batch_size]
            ids = [i for i, _ in chunk]
            batch = th.stack([t for _, t in chunk]).to(device)
            with th.amp.autocast("cuda", dtype=th.bfloat16):
                gen = denoiser.generate(batch)
            for gid, t in zip(ids, gen):
                save_png(t.float(), os.path.join(gen_dir, f"{gid:05d}.png"))
            done += len(chunk)
            print(f"  generated {done}/{total} (shard {args.shard_idx}/{args.num_shards})", flush=True)


# ---------------------------------------------------------------------------
# FeatInv (baseline/FeatInv) -- 50-step DDIM ControlNet, CheXWorld patch tokens
# ---------------------------------------------------------------------------

FEATINV_PROMPT = "a high-quality, detailed, and professional image"


def build_featinv_model(ckpt_path: str, config_path: str, device: th.device):
    from cldm.model import create_model, load_state_dict

    model = create_model(config_path).cpu()
    state_dict = load_state_dict(ckpt_path, location="cpu")
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    # FrozenCLIPEmbedder stores a hardcoded "cuda" and moves its token ids there,
    # which breaks on any GPU other than cuda:0.
    model.cond_stage_model.device = device
    return model


@th.no_grad()
def generate_featinv(args, real_tensors: list, gen_dir: str, device: th.device):
    sys.path.insert(0, FEATINV_ROOT)
    from guided_diffusion_rcdm.get_ssl_models import get_model
    from cldm.ddim_hacked import DDIMSampler

    os.makedirs(gen_dir, exist_ok=True)

    ssl_model = get_model("chexworld", pool=False).to(device).eval()
    for pm in ssl_model.parameters():
        pm.requires_grad_(False)

    model = build_featinv_model(args.ckpt_path, args.featinv_config, device)
    # Control strength scales the ControlNet residuals injected into the UNet.
    # FeatInv paper: 1.0 for unpooled, 1.5 for the pooled variant.
    model.control_scales = [args.featinv_control_strength] * 13
    sampler = DDIMSampler(model)
    latent = args.image_size // 8

    done = 0
    for start in range(0, len(real_tensors), args.batch_size):
        chunk = real_tensors[start:start + args.batch_size]
        ids = [i for i, _ in chunk]
        batch = th.stack([t for _, t in chunk]).to(device)          # (B,3,S,S) in [-1,1]

        hint_in = TF.resize(batch, [args.hint_resolution, args.hint_resolution], antialias=True)
        tokens = ssl_model(hint_in)                                 # (B, N, 768)
        h = int(round(tokens.shape[1] ** 0.5))
        control = tokens.reshape(tokens.shape[0], h, h, -1).permute(0, 3, 1, 2).float().contiguous()
        if args.featinv_pool:
            # Pooled variant: mean-pool the grid to a global descriptor and
            # broadcast it back, discarding spatial detail (matches training).
            control = control.mean(dim=(2, 3), keepdim=True).expand(-1, -1, h, h).contiguous()

        b = batch.shape[0]
        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([FEATINV_PROMPT] * b)]}
        # Mirrors ControlLDM.log_images: unconditional branch keeps the control, drops the text.
        un_cond = {"c_concat": [control],
                   "c_crossattn": [model.get_unconditional_conditioning(b)]}
        samples, _ = sampler.sample(
            args.ddim_steps, b, (model.channels, latent, latent), cond, verbose=False, eta=args.eta,
            unconditional_guidance_scale=args.cfg_scale, unconditional_conditioning=un_cond,
        )
        gen = model.decode_first_stage(samples)                     # (B,3,S,S) in [-1,1]

        for gid, t in zip(ids, gen):
            save_png(t.float(), os.path.join(gen_dir, f"{gid:05d}.png"))
        done += len(chunk)
        print(f"  generated {done}/{len(real_tensors)} (shard {args.shard_idx}/{args.num_shards})", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=["rcdm", "rct", "featinv"])
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--dataset", default="mimic", choices=["mimic", "nih", "chexpert"],
                   help="Which test set to build the real reference from (chexpert uses its valid split)")
    p.add_argument("--data_path", default="/mnt/nvme1/mimic-cxr-jpg", help="MIMIC-CXR root")
    p.add_argument("--nih_dir", default="/mnt/nvme1/nih-chest-x-ray-14")
    p.add_argument("--chexpert_dir", default="/mnt/nvme1/chexpert")
    p.add_argument("--save_path", required=True, help="Writes <save_path>/real/ and <save_path>/gen/")
    p.add_argument("--real_from_dir", default=None,
                   help="Load conditioning images from this existing real PNG dir instead of "
                        "rebuilding the real set from the dataset. Only <save_path>/gen/ is written.")
    p.add_argument("--num_images", type=int, default=3403)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0, help="Selects which test images form the eval subset")
    p.add_argument("--gen_seed", type=int, default=42, help="Seeds the sampling noise")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard", default="0/1", help='"i/n" -- run on a slice of the images, for parallel GPUs')
    p.add_argument("--no_ema", action="store_true", help="rct: use raw trained weights (alias for --ema raw)")
    p.add_argument("--ema", default="ema2", choices=["raw", "ema1", "ema2"],
                   help="rct: which weights to sample from (ema1=decay1, ema2=decay2, raw=trained)")
    p.add_argument("--cfg", type=float, default=None,
                   help="rct only: override the checkpoint's saved CFG scale (default: use hp value)")
    p.add_argument("--image_size", type=int, default=None,
                   help="rcdm/featinv only (defaults 128/256); rct reads this from the checkpoint")
    # rcdm-only
    p.add_argument("--use_ddim", action="store_true", default=True)
    p.add_argument("--no_ddim", dest="use_ddim", action="store_false")
    p.add_argument("--timestep_respacing", default="ddim1000", help="rcdm only")
    # featinv-only
    p.add_argument("--featinv_config",
                   default=os.path.join(FEATINV_ROOT, "models", "cldm_featinv_chexworld.yaml"))
    p.add_argument("--hint_resolution", type=int, default=224, help="featinv only")
    p.add_argument("--ddim_steps", type=int, default=50, help="featinv only")
    p.add_argument("--cfg_scale", type=float, default=9.0, help="featinv only")
    p.add_argument("--featinv_pool", action="store_true",
                   help="featinv only: condition on the mean-pooled descriptor (pooled variant) "
                        "instead of spatial patch tokens (unpooled). Paper setting: pooled uses "
                        "control strength 1.5 + cfg 8.0; unpooled uses 1.0 + cfg 1.75.")
    p.add_argument("--featinv_control_strength", type=float, default=1.0,
                   help="featinv only: ControlNet strength (paper: 1.0 unpooled, 1.5 pooled)")
    p.add_argument("--eta", type=float, default=0.0, help="featinv only")
    args = p.parse_args()

    args.shard_idx, args.num_shards = (int(x) for x in args.shard.split("/"))

    device = th.device(f"cuda:{args.gpu}" if th.cuda.is_available() else "cpu")
    real_dir = os.path.join(args.save_path, "real")
    gen_dir = os.path.join(args.save_path, "gen")

    image_size = args.image_size or {"rcdm": 128, "featinv": 256}.get(args.model_type)
    if args.model_type == "rct":
        # RCT stores its own resolution in the checkpoint; peek at it before building the real set.
        ckpt = th.load(args.ckpt_path, map_location="cpu", weights_only=False)
        image_size = ckpt["hyper_parameters"]["img_size"]
        del ckpt

    if args.real_from_dir:
        print(f"=== Loading real conditioning set from {args.real_from_dir} (shard {args.shard_idx}/{args.num_shards}) ===", flush=True)
        if args.model_type == "rct":
            real_tensors = make_real_dataloader(
                args.real_from_dir, image_size, args.num_images, args.batch_size,
                shard_idx=args.shard_idx, num_shards=args.num_shards)
        else:
            real_tensors = load_real_set(args.real_from_dir, image_size, args.num_images,
                                         args.shard_idx, args.num_shards)
        real_dir = args.real_from_dir
    else:
        print(f"=== Building real reference set [{args.dataset}] ({args.num_images} images, shard {args.shard_idx}/{args.num_shards}) ===", flush=True)
        ds = make_eval_dataset(args.dataset, args.data_path, args.nih_dir, args.chexpert_dir)
        real_tensors = build_real_set(ds, real_dir, args.num_images, image_size, args.seed,
                                       args.shard_idx, args.num_shards)

    print(f"\n=== [{args.model_type}] Loading {args.ckpt_path} ===", flush=True)
    print(f"\n=== [{args.model_type}] Generating reconstructions -> {gen_dir} ===", flush=True)
    args.image_size = image_size
    # Reproducible sampling noise. Offset per shard so shards don't share a noise stream.
    th.manual_seed(args.gen_seed + args.shard_idx)
    th.cuda.manual_seed_all(args.gen_seed + args.shard_idx)
    if args.model_type == "rcdm":
        generate_rcdm(args, real_tensors, gen_dir, device)
    elif args.model_type == "featinv":
        generate_featinv(args, real_tensors, gen_dir, device)
    else:
        generate_rct(args, real_tensors, gen_dir, device)

    print(f"\ndone. real_dir={real_dir} gen_dir={gen_dir}", flush=True)
    print(f"Run: python scripts/quality_metrics.py --real_dir {real_dir} --gen_dir {gen_dir} "
          f"--out {args.save_path}/metrics.txt")


if __name__ == "__main__":
    main()
