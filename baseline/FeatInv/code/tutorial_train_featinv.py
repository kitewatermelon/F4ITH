import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from cldm.model import create_model, load_state_dict
from rct_cxr_featinv_data import RCTCXRFeatInvDataset, build_sample_batch
from rct_sample_logger import RCTSampleLogger


class RCTGradMetrics(pl.Callback):
    """Mirror src/rct/train.py: log train/grad_norm and train/param_norm over
    trainable params before each optimizer step (ControlLDM already logs
    train/loss; LearningRateMonitor covers lr). Gives W&B parity with RCT."""
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        trainable = [p for p in pl_module.parameters() if p.requires_grad and p.grad is not None]
        if not trainable:
            return
        grad_norm  = torch.stack([p.grad.detach().norm() for p in trainable]).norm()
        param_norm = torch.stack([p.detach().norm()      for p in trainable]).norm()
        pl_module.log("train/grad_norm",  grad_norm,  on_step=True, on_epoch=False, sync_dist=True)
        pl_module.log("train/param_norm", param_norm, on_step=True, on_epoch=False, sync_dist=True)

# Reproducibility: fixed seed across all ranks + workers (matches RCT).
pl.seed_everything(int(os.environ.get("FEATINV_SEED", 42)), workers=True)

# --- Configs (original FeatInv hyperparameters preserved) ---
SCRIPT_DIR = Path(__file__).parent
# All run artifacts (init/final ckpt, wandb, PL logs, samples) live under output/.
OUTPUT_DIR = Path(os.environ.get("FEATINV_OUTPUT_DIR", SCRIPT_DIR.parent / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config_path = SCRIPT_DIR / "models" / "cldm_featinv_chexworld.yaml"
# ControlNet init merged from miniSD (create with download_minisd.py).
resume_path = OUTPUT_DIR / "control_featinv_chexworld_init.ckpt"
# Pooled variant (Table-1 baseline): conditions the ControlNet on the mean-pooled
# CheXWorld descriptor broadcast over the grid instead of the spatial patch tokens.
# Same config/init/hyperparameters as unpooled for a fair comparison; only the
# hint content, the final-ckpt name, and the W&B run name differ.
pool_mode = os.environ.get("FEATINV_POOL", "0") == "1"
variant = "pooled" if pool_mode else "unpooled"
final_ckpt_name = ("control_featinv_chexworld_pooled_fin.ckpt" if pool_mode
                   else "control_featinv_chexworld_fin.ckpt")
run_name = "featinv_chexworld_pooled_cxr" if pool_mode else "featinv_chexworld_cxr"
print(f"[featinv-train] variant={variant}  final_ckpt={final_ckpt_name}", flush=True)
batch_size = int(os.environ.get("FEATINV_BATCH_SIZE", 8))
sample_every = int(os.environ.get("FEATINV_SAMPLE_EVERY", 10000))   # match RCT/RCDM
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
use_bf16 = True
# compile ON by default. NOTE: compile the inner UNet only (NOT the whole
# LightningModule — that breaks Lightning hooks / log_images).
use_compile = os.environ.get("FEATINV_COMPILE", "1") == "1"

mimic_dir = os.environ.get("MIMIC_DIR", "/mnt/nvme1/mimic-cxr-jpg")
nih_dir = os.environ.get("NIH_DIR", "/mnt/nvme1/nih-chest-x-ray-14")
chexpert_dir = os.environ.get("CHEXPERT_DIR", "/mnt/nvme1/chexpert")
# Same fixed sample batch as RCT/RCDM: first 8 items of ConcatDataset([mimic,nih,chexpert]).
num_samples = 8

# Smoke/config hooks (defaults preserve the original experiment).
devices = int(os.environ.get("FEATINV_DEVICES", 2))
# 3 epochs over the combined dataset (515,523 imgs) at global batch 16 (8/GPU x 2 GPUs).
max_steps = int(os.environ.get("FEATINV_MAX_STEPS", 96_660))
limit_train_batches = float(os.environ.get("FEATINV_LIMIT_BATCHES", 1.0))

# --- Data ---
dataset = RCTCXRFeatInvDataset(mimic_dir, nih_dir, chexpert_dir, pool=pool_mode)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

# --- Model ---
model = create_model(str(config_path)).cpu()
if resume_path.exists():
    model.load_state_dict(load_state_dict(str(resume_path), location="cpu"))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

if use_compile and torch.cuda.is_available():
    # Compile the inner UNet only (mirrors src/rct compiling denoiser.net),
    # NOT the whole LightningModule — keeps Lightning hooks / log_images intact.
    model.model.diffusion_model = torch.compile(model.model.diffusion_model)

# --- Fixed sample batch (rank 0 builds; RCT no-crop preprocessing) ---
sample_batch = build_sample_batch(mimic_dir, nih_dir, chexpert_dir, num_samples=num_samples,
                                  pool=pool_mode)

# --- Logging / callbacks (all under output/) ---
wandb_logger = WandbLogger(project="featinv-chexworld", name=run_name,
                           save_dir=str(OUTPUT_DIR))
sample_logger = RCTSampleLogger(sample_batch=sample_batch, every_n_steps=sample_every,
                                max_images=num_samples)   # show all 8, like RCT/RCDM
# W&B parity with RCT: train/loss (ControlLDM) + grad/param norms + lr + samples.
grad_metrics = RCTGradMetrics()
lr_monitor = LearningRateMonitor(logging_interval="step")

# --- Trainer (original: 2 GPUs, bf16, ddp, max_epochs=3) ---
precision = "bf16-mixed" if use_bf16 else 32
trainer = pl.Trainer(
    default_root_dir=str(OUTPUT_DIR),
    precision=precision,
    callbacks=[sample_logger, grad_metrics, lr_monitor],
    max_epochs=-1,
    max_steps=max_steps,
    limit_train_batches=limit_train_batches,
    accelerator="gpu",
    devices=devices,
    enable_progress_bar=True,
    log_every_n_steps=10,
    logger=wandb_logger,
    # sd_locked freezes the UNet, so some params don't produce grads (matches RCT).
    strategy=DDPStrategy(find_unused_parameters=True),
)

trainer.fit(model, dataloader)
trainer.save_checkpoint(str(OUTPUT_DIR / final_ckpt_name))
