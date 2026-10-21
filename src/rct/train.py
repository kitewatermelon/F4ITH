"""
Cross-attention JiT training on MIMIC-CXR, conditioned on a frozen backbone encoder
(RAD-DINO or ChexWorld) via a required `--conditioning_mode`: `cls_only` drives adaLN
with a pooled CLS embedding (global), `patch_only` drives per-block cross-attention with
dense patch tokens (spatial) — see
docs/superpowers/specs/2026-08-03-cross-attn-jit-design.md and
docs/superpowers/specs/2026-08-04-conditioning-mode-design.md.

Usage (RAD-DINO):
    python src/rct/train.py \
        --data_dir     /mnt/nvme1/mimic-cxr-jpg \
        --output_dir   output/rct_s16_raddino_cxr \
        --conditioning_mode cls_only

Usage (ChexWorld):
    python src/rct/train.py \
        --data_dir     /mnt/nvme1/mimic-cxr-jpg \
        --output_dir   output/rct_s16_chexworld_cxr \
        --conditioning_mode cls_only \
        --backbone chexworld
"""

import os
import argparse
from contextlib import nullcontext

import torch
import torchvision.utils as vutils
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from stable_pretraining.data import transforms
from stable_pretraining.data import gpu_transforms as gt
from stable_pretraining.data.datasets import FromTorchDataset

from rct.denoiser import RCTDenoiser
from rct.dataset import MIMICCXRDataset, NIHChestXray14Dataset, CheXpertDataset


# ---------------------------------------------------------------------------
# Augmentation — images in [-1, 1] for flow-matching
# ---------------------------------------------------------------------------

def get_rct_aug(device, img_size=224):
    """CPU resize -> GPU crop/flip/normalize to [-1, 1]."""
    cpu_transform = transforms.Compose(
        transforms.Resize((256, 256)),
        transforms.ToImage(rgb=True, scale=True),   # PIL L -> [0,1] RGB tensor
    )
    gpu_aug = gt.GPUCompose([
        gt.GPURandomResizedCrop(size=img_size, scale=(0.8, 1.2)),
        gt.GPUColorJitter(brightness=0.2, contrast=0.2, p=0.5),   # exposure/detector-gain variation
        gt.GPUNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),   # -> [-1, 1]
    ]).to(device)
    return cpu_transform, gpu_aug


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class CombinedCXRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, nih_dir: str, chexpert_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.nih_dir = nih_dir
        self.chexpert_dir = chexpert_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        cpu_transform, _ = get_rct_aug(torch.device("cpu"))

        mimic_ds = MIMICCXRDataset(
            root=self.data_dir,
            split="train",
            label_csv="chexpert",
            transform=None,
            frontal_only=True,
            uncertainty="zero",
        )
        nih_ds = NIHChestXray14Dataset(
            root=self.nih_dir,
            split="train",
            transform=None,
            frontal_only=True,
        )
        chexpert_ds = CheXpertDataset(
            root=self.chexpert_dir,
            split="train",
            transform=None,
            frontal_only=True,
            uncertainty="zero",
        )
        combined = torch.utils.data.ConcatDataset([mimic_ds, nih_ds, chexpert_ds])

        self.train_ds = FromTorchDataset(
            dataset=combined,
            names=["image"],
            transform=cpu_transform,
            gpu_transform=None,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
            multiprocessing_context="fork",
            prefetch_factor=4,
        )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _warmup_constant_lambda(warmup_steps: int):
    """Linear warmup then constant LR, matching the original JiT schedule."""
    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return lr_lambda


# ---------------------------------------------------------------------------
# Sample callback
# ---------------------------------------------------------------------------

class SampleCallback(pl.Callback):
    def __init__(self, sample_imgs: torch.Tensor, every_n_steps: int, ode_steps: int, cfg: float,
                 sample_ema: str = "ema2"):
        super().__init__()
        self.sample_imgs = sample_imgs   # (N, 3, 224, 224) in [-1, 1]
        self.every_n_steps = every_n_steps
        self.ode_steps = ode_steps
        self.cfg = cfg
        self.sample_ema = sample_ema

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0 or trainer.global_step % self.every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return

        denoiser = pl_module.denoiser
        denoiser.eval()
        cond = self.sample_imgs.to(pl_module.device)

        orig_steps, orig_cfg = denoiser.steps, denoiser.cfg_scale
        denoiser.steps, denoiser.cfg_scale = self.ode_steps, self.cfg

        if self.sample_ema == "raw":
            ema_ctx = nullcontext()
        elif self.sample_ema == "ema1":
            ema_ctx = denoiser.ema_scope(1)
        else:
            ema_ctx = denoiser.ema_scope(2)

        with ema_ctx, torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen = denoiser.generate(cond)

        denoiser.steps, denoiser.cfg_scale = orig_steps, orig_cfg
        denoiser.train()

        cond_vis = (cond.clamp(-1, 1) + 1) / 2
        gen_vis  = (gen .clamp(-1, 1) + 1) / 2
        grid = vutils.make_grid(
            torch.cat([cond_vis.cpu(), gen_vis.cpu()], dim=0),
            nrow=cond.size(0), padding=2, pad_value=1.0,
        )
        import wandb
        trainer.logger.experiment.log(
            {"samples": wandb.Image(grid, caption=f"step {trainer.global_step} | top: cond, bottom: gen")},
            step=trainer.global_step,
        )


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class RCTModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args
        # Built on CPU here (not on_fit_start) so _gpu_aug exists as a real
        # submodule at construction time, before Lightning restores a
        # checkpoint's state_dict -- otherwise a fresh module has no
        # _gpu_aug.* keys yet and --resume_from fails with "Unexpected
        # key(s)". Lightning moves it to the right device along with the
        # rest of the module automatically.
        _, self._gpu_aug = get_rct_aug(torch.device("cpu"), img_size=args.img_size)

        denoiser_args = argparse.Namespace(
            img_size           = args.img_size,
            model_variant       = args.model_variant,
            attn_dropout        = args.attn_dropout,
            proj_dropout        = args.proj_dropout,
            cond_drop_prob      = args.cond_drop_prob,
            P_mean              = args.P_mean,
            P_std               = args.P_std,
            noise_scale         = args.noise_scale,
            t_eps               = args.t_eps,
            ema_decay1          = args.ema_decay1,
            ema_decay2          = args.ema_decay2,
            sampling_method     = args.ode_method,
            num_sampling_steps  = args.ode_steps,
            cfg                 = args.cfg,
            interval_min        = args.interval_min,
            interval_max        = args.interval_max,
            long_skip_connection = args.long_skip_connection,
            depth               = args.depth,
            conditioning_mode   = args.conditioning_mode,
            pooled_cond_source  = args.pooled_cond_source,
            use_final_conv      = not args.no_final_conv,
            backbone            = args.backbone,
            hidden_size         = args.hidden_size,
            num_heads           = args.num_heads,
            patch_size          = args.patch_size,
            t_embed_mode        = args.t_embed_mode,
            in_context_len      = args.in_context_len,
            in_context_start    = args.in_context_start,
        )
        denoiser = RCTDenoiser(denoiser_args)
        if args.compile:
            torch._dynamo.config.cache_size_limit = 128
            torch._dynamo.config.optimize_ddp = False
            denoiser.net = torch.compile(denoiser.net)
        self.denoiser = denoiser

    def on_fit_start(self):
        import copy
        if self.denoiser.ema_params1 is None:
            params = self.denoiser._trainable_params()
            self.denoiser.ema_params1 = copy.deepcopy(params)
            self.denoiser.ema_params2 = copy.deepcopy(params)
        else:
            # on_load_checkpoint restores these from CPU tensors (they're
            # saved via .cpu() in on_save_checkpoint) before the module has
            # been placed on its training device -- move them to match now.
            self.denoiser.ema_params1 = [p.to(self.device) for p in self.denoiser.ema_params1]
            self.denoiser.ema_params2 = [p.to(self.device) for p in self.denoiser.ema_params2]

    def on_save_checkpoint(self, checkpoint):
        names = [n for n, p in self.denoiser.named_parameters() if p.requires_grad]
        if self.denoiser.ema_params1 is not None:
            checkpoint["ema_params1"] = {n: p.cpu() for n, p in zip(names, self.denoiser.ema_params1)}
        if self.denoiser.ema_params2 is not None:
            checkpoint["ema_params2"] = {n: p.cpu() for n, p in zip(names, self.denoiser.ema_params2)}

    def on_load_checkpoint(self, checkpoint):
        # Compat: checkpoints from before the dino->encoder/cond rename carry the
        # old key prefixes. Remap the module weights (called before Lightning's
        # strict load) and the name-keyed EMA dicts so resume keeps working.
        from rct.denoiser import remap_legacy_state_dict
        if "state_dict" in checkpoint:
            checkpoint["state_dict"] = remap_legacy_state_dict(checkpoint["state_dict"], prefix="denoiser.")
        for ema_key in ("ema_params1", "ema_params2"):
            if ema_key in checkpoint:
                checkpoint[ema_key] = remap_legacy_state_dict(checkpoint[ema_key])
        names = [n for n, p in self.denoiser.named_parameters() if p.requires_grad]
        if "ema_params1" in checkpoint:
            self.denoiser.ema_params1 = [checkpoint["ema_params1"][n] for n in names]
        if "ema_params2" in checkpoint:
            self.denoiser.ema_params2 = [checkpoint["ema_params2"][n] for n in names]

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        x = self._gpu_aug({"image": x})["image"]   # -> [-1, 1]
        loss = self.denoiser(x, cond_img=x)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        trainable = [p for p in self.denoiser.parameters() if p.requires_grad and p.grad is not None]
        if trainable:
            grad_norm  = torch.stack([p.grad.detach().norm() for p in trainable]).norm()
            param_norm = torch.stack([p.detach().norm()      for p in trainable]).norm()
            self.log("train/grad_norm",  grad_norm,  on_step=True, on_epoch=False, sync_dist=True)
            self.log("train/param_norm", param_norm, on_step=True, on_epoch=False, sync_dist=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.denoiser.update_ema()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.denoiser.parameters() if p.requires_grad],
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=_warmup_constant_lambda(self.args.warmup_steps),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Cross-attention JiT trainer")

    p.add_argument("--data_dir",    default="/mnt/nvme1/mimic-cxr-jpg")
    p.add_argument("--nih_dir",      default="/mnt/nvme1/nih-chest-x-ray-14")
    p.add_argument("--chexpert_dir", default="/mnt/nvme1/chexpert")
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--model_variant", default="S_16", choices=["S_16", "B_16"])
    p.add_argument("--long_skip_connection", action="store_true")
    p.add_argument("--no_final_conv", action="store_true", help="Disable the final 3x3 conv residual smoothing layer")
    p.add_argument("--depth", type=int, default=13)
    p.add_argument("--hidden_size", type=int, default=None, help="Override the model variant's default channel width (e.g. 512)")
    p.add_argument("--num_heads", type=int, default=None, help="Override the model variant's default head count (must divide hidden_size)")
    p.add_argument("--patch_size", type=int, default=None, help="Override the model variant's default patch size (e.g. 14 to keep num_patches fixed at a non-224 img_size)")
    p.add_argument("--conditioning_mode", required=True, choices=["cls_only", "patch_only", "cls_cross"])
    p.add_argument("--in_context_len", type=int, default=0,
                   help="Number of in-context conditioning tokens (JiT-style) prepended mid-network for "
                        "cls_only; 0 disables. The pooled CLS is repeated in_context_len times + a learned "
                        "position embedding, injected at block --in_context_start and dropped before the "
                        "final layer. Excluded from the long-skip concat (skips carry image patches only).")
    p.add_argument("--in_context_start", type=int, default=4,
                   help="0-indexed block at which in-context tokens are prepended (must be < depth//2 when "
                        "--long_skip_connection is set).")
    p.add_argument("--pooled_cond_source", default="cls", choices=["cls", "mean"],
                   help="cls_only conditioning source: 'cls' token (default) or 'mean' over patch tokens "
                        "(faithful pooled rep; matches the distance-metric pool=True feature)")
    p.add_argument("--backbone", default="raddino",
                   choices=["raddino", "chexworld", "xrv_densenet121"],
                   help="Frozen encoder for conditioning: RAD-DINO (ViT-B/14 SSL), CheXWorld "
                        "(ViT-B/16 IWM SSL) or TorchXRayVision DenseNet-121 (CNN, supervised "
                        "on public CXR labels; 7x7=49 conditioning tokens at 224)")
    p.add_argument("--chexworld_compat", default="faithful", choices=["faithful", "legacy"],
                   help="Which ChexWorld encoder to build: 'faithful' matches its pretraining "
                        "(ImageNet normalization, no class token) and the distance-metric "
                        "encoder; 'legacy' reproduces the pre-fix encoder (0.5/0.5 stats + a "
                        "synthetic CLS token) for continuing a pre-fix run. Only applies to "
                        "--backbone chexworld")
    p.add_argument("--t_embed_mode", default="adaln", choices=["adaln", "token", "adaln_single"],
                   help="How the timestep reaches the network: adaLN-Zero (DiT-style, default), a shared-MLP adaLN-Zero variant (PixArt-alpha style), or a prepended sequence token (U-ViT style)")

    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_steps",    type=int,   default=300_000)
    p.add_argument("--warmup_steps", type=int,   default=10_000)
    p.add_argument("--num_gpus",     type=int,   default=-1)

    p.add_argument("--cond_drop_prob", type=float, default=0.1)
    p.add_argument("--P_mean",         type=float, default=-0.8)
    p.add_argument("--P_std",          type=float, default=0.8)
    p.add_argument("--noise_scale",    type=float, default=1.0)
    p.add_argument("--t_eps",          type=float, default=0.05)
    p.add_argument("--ema_decay1",     type=float, default=0.9999)
    p.add_argument("--ema_decay2",     type=float, default=0.9996)
    p.add_argument("--attn_dropout",   type=float, default=0.0)
    p.add_argument("--proj_dropout",   type=float, default=0.0)

    p.add_argument("--cfg",          type=float, default=1.5)
    p.add_argument("--ode_steps",    type=int,   default=50)
    p.add_argument("--ode_method",   default="heun", choices=["euler", "heun"])
    p.add_argument("--interval_min", type=float, default=0.0)
    p.add_argument("--interval_max", type=float, default=1.0)
    p.add_argument("--sample_every", type=int,   default=10_000)
    p.add_argument("--num_samples",  type=int,   default=8)
    p.add_argument("--sample_ema",   default="ema2", choices=["ema2", "ema1", "raw"],
                   help="Which weights the periodic W&B preview-sample callback uses: ema2 (default), ema1, or raw (live, non-EMA) weights")

    p.add_argument("--output_dir",  default="output/rct_s16_raddino_cxr")
    p.add_argument("--save_every",  type=int, default=10_000)
    p.add_argument("--resume_from", default=None)

    p.add_argument("--wandb_project",  default="rct-raddino-cxr")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--log_every",      type=int, default=50)

    p.add_argument("--compile", action="store_true", default=True, help="Compile model with torch.compile")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    dm = CombinedCXRDataModule(
        data_dir=args.data_dir,
        nih_dir=args.nih_dir,
        chexpert_dir=args.chexpert_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    import torchvision.transforms.functional as TF
    sample_imgs_raw = [dm.train_ds[i]["image"] for i in range(args.num_samples)]
    sample_imgs = torch.stack(sample_imgs_raw)          # (N, 3, 256, 256) in [0,1]
    sample_imgs = TF.resize(sample_imgs, [args.img_size, args.img_size], antialias=True)  # full FOV, matches scale=1.0 GPURandomResizedCrop
    sample_imgs = sample_imgs * 2.0 - 1.0              # -> [-1, 1]

    lit = RCTModule(args)

    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        save_dir=args.output_dir,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="step{step:07d}",
            every_n_train_steps=args.save_every,
            save_top_k=-1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        SampleCallback(
            sample_imgs=sample_imgs,
            every_n_steps=args.sample_every,
            ode_steps=args.ode_steps,
            cfg=args.cfg,
            sample_ema=args.sample_ema,
        ),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="bf16-mixed",
        max_epochs=-1,
        max_steps=args.max_steps,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=args.log_every,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        enable_progress_bar=True,
    )

    trainer.fit(lit, datamodule=dm, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
