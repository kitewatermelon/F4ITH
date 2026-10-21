import copy
import functools
import os
import re

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

try:
    import wandb
except ImportError:
    wandb = None

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_bf16=False,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        sample_imgs=None,
        sample_feat=None,
        sample_interval=10000,
        wandb_enabled=False,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_bf16 = use_bf16
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        # bf16 has fp32's exponent range, so loss scaling is unnecessary (and
        # harmful over long runs — the scale grows until fp32 grads overflow).
        # Keep the scaler object for a uniform API, but disabled = pass-through.
        self.scaler = GradScaler(enabled=False)

        self.sample_imgs = sample_imgs
        self.sample_feat = sample_feat
        self.sample_interval = sample_interval
        self.wandb_enabled = wandb_enabled

        self._load_and_sync_parameters()
        # With bf16, we don't need manual loss scaling; just use the model as-is
        self.mp_trainer = None

        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(list(self.model.parameters()))
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _unwrapped_model(self):
        """Return the underlying nn.Module, unwrapping torch.compile's
        OptimizedModule (``_orig_mod``) so state_dict / named_parameters use
        clean keys (no ``_orig_mod.`` prefix) for checkpoint save/load/EMA."""
        return getattr(self.model, "_orig_mod", self.model)

    def _load_and_sync_parameters(self):
        if self.resume_checkpoint != "":
            resume_checkpoint = self.resume_checkpoint
        else:
            resume_checkpoint, self.resume_step = find_resume_checkpoint()

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                trained_model = th.load(resume_checkpoint, map_location=dist_util.dev())
                self._unwrapped_model().load_state_dict(trained_model)

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(list(self.model.parameters()))

        if self.resume_checkpoint != "":
            main_checkpoint = self.resume_checkpoint
        else:
            main_checkpoint, self.resume_step = find_resume_checkpoint()
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
                for param, (name, _) in zip(ema_params, self._unwrapped_model().named_parameters()):
                    if name in state_dict:
                        param.data.copy_(state_dict[name])

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        if self.resume_checkpoint != "":
            main_checkpoint = self.resume_checkpoint
        else:
            main_checkpoint, self.resume_step = find_resume_checkpoint()
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                kvs = logger.dumpkvs()
                # dumpkvs() returns the just-flushed metric dict; forward it to
                # W&B here (rank 0). NOTE: logger.get_current() returns a Logger
                # object, not a dict — logging that never worked. Per-timestep
                # quartile diagnostics (*_q0..q3) stay in the local log/CSV but
                # are filtered out of W&B to keep the dashboard readable.
                if wandb is not None and self.wandb_enabled and kvs:
                    wandb.log(
                        {
                            f"train/{k}": v
                            for k, v in kvs.items()
                            if not re.search(r"_q\d$", k)
                        },
                        step=self.step + self.resume_step,
                    )
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)

        # Unscale grads (no-op with bf16 where the scaler is disabled), then log
        # the true PRE-clip grad/param norms and clip to max_norm=1.0 (matches
        # src/rct). clip_grad_norm_ returns the total 2-norm before clipping.
        self.scaler.unscale_(self.opt)
        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if params:
            grad_norm = th.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            with th.no_grad():
                param_norm = th.norm(th.stack([p.detach().norm() for p in params]))
            logger.logkv_mean("grad_norm", grad_norm.item())
            logger.logkv_mean("param_norm", param_norm.item())

        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()
        self._update_ema()
        self._anneal_lr()
        self.log_step()
        self._sample_if_needed()

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            micro_noise = None

            def compute_forward():
                with autocast(enabled=self.use_bf16, dtype=th.bfloat16):
                    losses = self.diffusion.training_losses(
                        self.ddp_model,
                        micro,
                        t,
                        model_kwargs=micro_cond,
                        noise=micro_noise
                    )
                return losses

            if last_batch or not self.use_ddp:
                losses = compute_forward()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_forward()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.scaler.scale(loss).backward()

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, list(self.model.parameters()), rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("lr", self.opt.param_groups[0]["lr"])
        # W&B logging happens in run_loop where dumpkvs() returns the flushed dict.

    def _sample_if_needed(self):
        """Conditioning-aware sampling, mirroring src/rct's SampleCallback:
        skip step 0, rank 0 only, generate conditioned on a fixed real batch
        (via its backbone `feat`), and log a [top: cond | bottom: gen] grid."""
        step = self.step + self.resume_step
        if (
            self.sample_imgs is None
            or dist.get_rank() != 0
            or step == 0
            or step % self.sample_interval != 0
        ):
            return

        logger.log(f"generating samples at step {step}...")
        # Sample on the underlying (uncompiled) module: avoids DDP forward-side
        # effects on a single rank and torch.compile eval-mode recompilation.
        model = self._unwrapped_model()
        was_training = model.training
        model.eval()

        model_kwargs = {"feat": self.sample_feat} if self.sample_feat is not None else {}
        with th.no_grad():
            with autocast(enabled=self.use_bf16, dtype=th.bfloat16):
                samples = self.diffusion.p_sample_loop(
                    model,
                    self.sample_imgs.shape,
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    progress=True,
                )

        if was_training:
            model.train()

        # Visualize: top row = real conditioning images, bottom row = generated
        cond_vis = (self.sample_imgs.clamp(-1, 1) + 1) / 2  # [-1, 1] -> [0, 1]
        samples_vis = (samples.clamp(-1, 1) + 1) / 2

        import torchvision.utils as vutils
        grid_img = vutils.make_grid(
            th.cat([cond_vis.cpu(), samples_vis.cpu()], dim=0),
            nrow=self.sample_imgs.shape[0], padding=2, pad_value=1.0,
        )

        if wandb is not None and self.wandb_enabled:
            wandb.log(
                {"samples": wandb.Image(grid_img, caption=f"step {step} | top: cond, bottom: gen")},
                step=step,
            )

    def save(self):
        def save_checkpoint(rate, params):
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                unwrapped = self._unwrapped_model()
                if rate == 0:
                    state_dict = unwrapped.state_dict()
                    with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                        th.save(state_dict, f)
                else:
                    ema_state = {name: p.clone().detach() for name, p in zip(
                        [n for n, _ in unwrapped.named_parameters()],
                        params
                    )}
                    with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                        th.save(ema_state, f)

        save_checkpoint(0, None)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()

def extract_number(f):
    s = re.findall("(\d+).pt",f)
    return (int(s[0]) if s else -1,f)

def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    filelistall = os.listdir(get_blob_logdir())
    max_filename = max(filelistall, key=extract_number)
    resume_step = extract_number(max_filename)[0]
    if resume_step == -1 or resume_step == 000000:
        return None, 0
    filename = f"model{(resume_step):06d}.pt"
    path = bf.join(get_blob_logdir(), filename)
    if bf.exists(path):
        return path, resume_step
    return None, 0

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
