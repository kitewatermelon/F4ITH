"""
Smoke test: build RCTDenoiser (model_variant="S_16") for both conditioning
modes ("cls_only", "patch_only"), run one forward pass (finite flow-matching
loss) and one generate() pass on dummy images.

Requires network access to download RAD-DINO from the HF hub (cached after
the first run — cxr-ssl has already downloaded it on this machine).

Run from the RCT repo root: python scripts/smoke_test_denoiser.py
"""
import argparse
import torch

from rct.denoiser import RCTDenoiser


def make_args(conditioning_mode):
    return argparse.Namespace(
        img_size=224,
        model_variant="S_16",
        conditioning_mode=conditioning_mode,
        attn_dropout=0.0,
        proj_dropout=0.0,
        cond_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        noise_scale=1.0,
        t_eps=0.05,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        sampling_method="heun",
        num_sampling_steps=4,
        cfg=1.5,
        interval_min=0.0,
        interval_max=1.0,
    )


for mode in ("cls_only", "patch_only"):
    denoiser = RCTDenoiser(make_args(mode))
    denoiser.eval()

    x = torch.randn(2, 3, 224, 224).clamp(-1, 1)   # dummy [-1, 1] images

    with torch.no_grad():
        cond_pooled, patches = denoiser.encode_cond(x)
        assert cond_pooled.shape == (2, 768), f"[{mode}] unexpected cls shape: {cond_pooled.shape}"
        assert patches.shape == (2, 256, 768), f"[{mode}] unexpected patches shape: {patches.shape}"
        if mode == "patch_only":
            assert denoiser.null_patch_tokens.shape == (1, 256, 768), (
                f"[{mode}] unexpected null_patch_tokens shape: {denoiser.null_patch_tokens.shape}"
            )
        else:
            assert denoiser.null_cond.shape == (1, denoiser.net.hidden_size), (
                f"[{mode}] unexpected null_cond shape: {denoiser.null_cond.shape}"
            )

        loss = denoiser(x, cond_img=x)
        assert torch.isfinite(loss), f"[{mode}] loss is not finite: {loss}"

        gen = denoiser.generate(x)
        assert gen.shape == x.shape, f"[{mode}] unexpected generate() output shape: {gen.shape}"

    print(f"OK: [{mode}] loss =", loss.item(), "| patches =", patches.shape, "| gen =", gen.shape)
