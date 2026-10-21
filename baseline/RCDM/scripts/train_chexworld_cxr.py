"""
Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
Train a conditional (representation based) diffusion model on images.
"""

import argparse
import random

import numpy as np
import torch as th
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None

from guided_diffusion_rcdm import dist_util, logger
from guided_diffusion_rcdm.resample import create_named_schedule_sampler
from guided_diffusion_rcdm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion_rcdm.train_util import TrainLoop
from guided_diffusion_rcdm.get_ssl_models import get_model
from guided_diffusion_rcdm.rct_cxr_data import load_rct_cxr_data, build_sample_batch
from torch.cuda.amp import autocast, GradScaler

def main(args):
    # Init distributed setup
    dist_util.init_distributed_mode(args)

    # Reproducibility: fixed seed on all ranks (mirrors RCT's seed_everything).
    # Data order is already deterministic via DistributedSampler(seed=0); this
    # pins model init, diffusion timestep/noise sampling (numpy + torch).
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    logger.configure(dir=args.out_dir)

    # Init wandb (rank 0 only)
    if wandb is not None and dist_util.get_rank() == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.out_dir,
        )

    # Load SSL model
    if args.feat_cond:
        ssl_model = get_model(args.type_model, args.use_head).to(args.gpu).eval()
        ssl_dim = ssl_model(th.zeros(1,3,224,224).to(args.gpu)).size(1)
        print("SSL DIM:", ssl_dim)
        for _,p in ssl_model.named_parameters():
            p.requires_grad_(False)
    else:
        ssl_model = None
        ssl_dim = 2048
        print("No SSL models")

    # Create RCDM
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()), G_shared=args.no_shared, feat_cond=args.feat_cond, ssl_dim=ssl_dim
    )

    model.to(args.gpu)

    # Compile model for performance (mirrors src/rct: plain torch.compile, no
    # reduce-overhead/CUDA-graphs which are fragile with DDP + periodic sampling).
    # Checkpoint save/load unwraps the resulting _orig_mod (see TrainLoop).
    if args.compile:
        logger.log("compiling model with torch.compile...")
        th._dynamo.config.cache_size_limit = 128
        th._dynamo.config.optimize_ddp = False
        model = th.compile(model)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Model params: {total_params:,} ({total_params/1e6:.1f}M)")
    logger.log(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    logger.log(f"Model size (bf16): {(total_params*2)/(1024**3):.2f}GB")

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Create the dataloader
    logger.log("creating data loader...")
    data = load_ssl_data(
        args,
        ssl_model=ssl_model,
    )

    # Prepare a fixed batch of real CXR images for in-training visualization,
    # conditioned on their backbone features (mirrors src/rct SampleCallback).
    sample_imgs = build_sample_batch(
        mimic_dir=args.mimic_dir,
        nih_dir=args.nih_dir,
        chexpert_dir=args.chexpert_dir,
        resolution=args.image_size,
        num_samples=args.num_samples,
    ).to(args.gpu)
    if ssl_model is not None:
        with th.no_grad():
            with autocast(dtype=th.bfloat16 if args.use_bf16 else th.float32):
                sample_feat = ssl_model(sample_imgs).detach()
    else:
        sample_feat = None

    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_bf16=args.use_bf16,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        sample_imgs=sample_imgs,
        sample_feat=sample_feat,
        sample_interval=args.sample_every,
        wandb_enabled=wandb is not None and dist_util.get_rank() == 0,
    ).run_loop()

def load_ssl_data(args, ssl_model=None):
    data = load_rct_cxr_data(
        mimic_dir=args.mimic_dir,
        nih_dir=args.nih_dir,
        chexpert_dir=args.chexpert_dir,
        batch_size=args.batch_size,
        resolution=args.image_size,
    )
    for batch, batch_big, model_kwargs in data:
        # We add the conditioning in conditional mode
        if ssl_model is not None:
            with th.no_grad():
                with autocast(dtype=th.bfloat16 if args.use_bf16 else th.float32):
                    # single-resolution design: batch_big is the same crop as batch (both at args.image_size), not a separate 224x224 crop
                    model_kwargs["feat"] = ssl_model(batch_big.to(args.gpu)).detach()
            yield batch, model_kwargs
        else:
            yield batch, model_kwargs

def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=300000,  # Train for 300K steps
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_bf16=True,
        submitit=False,
        local_rank=0,
        dist_url="env://",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()  # add_help=True (unlike image_train.py's add_help=False) — this script isn't wrapped by run_with_submitit.py's parents=[], and --help needs to work standalone for testing

    # Handle torch.distributed.launch's --local-rank flag
    parser.add_argument('--local-rank', type=int, default=0, dest='local_rank_dash')
    parser.add_argument('--out_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--feat_cond', action='store_true', default=False,
                        help='Activate conditional RCDM.')
    parser.add_argument('--use_head', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--no_shared', action='store_false', default=True,
                        help='Disable the shared lower dimensional projection of the representation.')
    parser.add_argument('--type_model', type=str, default="dino",
                    help='Select the type of model to use.')
    parser.add_argument('--mimic_dir', default="/mnt/nvme1/mimic-cxr-jpg", type=str)
    parser.add_argument('--nih_dir', default="/mnt/nvme1/nih-chest-x-ray-14", type=str)
    parser.add_argument('--chexpert_dir', default="/mnt/nvme1/chexpert", type=str)
    parser.add_argument('--wandb_project', default="rcdm-chexworld", type=str, help='W&B project name')
    parser.add_argument('--wandb_run_name', default=None, type=str, help='W&B run name')
    parser.add_argument('--compile', action='store_true', default=False, help='Compile model with torch.compile for performance')
    parser.add_argument('--sample_every', default=10000, type=int, help='Generate + log sample grid every N steps (rank 0)')
    parser.add_argument('--num_samples', default=8, type=int, help='Number of fixed conditioning images for sampling viz')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility (matches RCT default 42).')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
