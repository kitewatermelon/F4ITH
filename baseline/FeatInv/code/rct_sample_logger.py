"""RCT-style in-training sampling for FeatInv: fixed real batch -> generate
conditioned on its CheXWorld hint -> log [top: cond | bottom: gen] to W&B.
Mirrors src/rct/train.py::SampleCallback. Replaces ControlNet's disk-only
ImageLogger for W&B visualization."""
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils


class RCTSampleLogger(pl.Callback):
    def __init__(self, sample_batch, every_n_steps=5000, ddim_steps=50, cfg_scale=9.0, max_images=4):
        super().__init__()
        self.sample_batch = sample_batch
        self.every_n_steps = every_n_steps
        self.ddim_steps = ddim_steps
        self.cfg_scale = cfg_scale
        self.max_images = max_images

    def _should_run(self, step, is_global_zero):
        return bool(is_global_zero and step > 0 and step % self.every_n_steps == 0)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if not self._should_run(step, trainer.is_global_zero):
            return

        was_training = pl_module.training
        pl_module.eval()
        dev = pl_module.device
        n = min(self.max_images, self.sample_batch["jpg"].shape[0])
        b = {
            "jpg": self.sample_batch["jpg"][:n].to(dev),
            "hint": self.sample_batch["hint"][:n].to(dev),
            "txt": self.sample_batch["txt"][:n],
        }

        sample_only = self.cfg_scale <= 1.0
        log = pl_module.log_images(
            b, N=n, sample=sample_only, ddim_steps=self.ddim_steps,
            unconditional_guidance_scale=self.cfg_scale,
        )
        key = "samples" if sample_only else f"samples_cfg_scale_{self.cfg_scale:.2f}"
        gen = log[key].detach().cpu().float().clamp(-1, 1)                # (n,3,256,256)
        cond = b["jpg"].permute(0, 3, 1, 2).detach().cpu().float().clamp(-1, 1)

        grid = vutils.make_grid(
            torch.cat([(cond + 1) / 2, (gen + 1) / 2], dim=0),
            nrow=n, padding=2, pad_value=1.0,
        )
        if was_training:
            pl_module.train()

        import wandb
        trainer.logger.experiment.log(
            {"samples": wandb.Image(grid, caption=f"step {step} | top: cond, bottom: gen")},
            step=step,
        )
