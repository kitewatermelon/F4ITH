"""Backward-compat: legacy dino-prefixed checkpoint keys remap to encoder/cond."""
import torch

from rct.denoiser import remap_legacy_state_dict


def test_remaps_legacy_prefixes():
    legacy = {
        "dino.model.blocks.0.weight": torch.zeros(1),
        "dino_proj.weight": torch.zeros(1),
        "dino_proj.bias": torch.zeros(1),
        "net.blocks.0.attn.qkv.weight": torch.zeros(1),  # untouched
        "null_patch_tokens": torch.zeros(1),             # untouched
    }
    out = remap_legacy_state_dict(legacy)
    assert set(out) == {
        "encoder.model.blocks.0.weight",
        "cond_proj.weight",
        "cond_proj.bias",
        "net.blocks.0.attn.qkv.weight",
        "null_patch_tokens",
    }


def test_dino_proj_matched_before_dino():
    # "dino_proj." must not be mangled into "encoder.proj." by the "dino." rule
    out = remap_legacy_state_dict({"dino_proj.weight": torch.zeros(1)})
    assert "cond_proj.weight" in out


def test_new_names_pass_through_unchanged():
    already = {"encoder.model.x": torch.zeros(1), "cond_proj.weight": torch.zeros(1)}
    assert set(remap_legacy_state_dict(already)) == set(already)


def test_prefix_aware_remap_for_lightning_keys():
    legacy = {"denoiser.dino_proj.weight": torch.zeros(1), "denoiser.net.x": torch.zeros(1)}
    out = remap_legacy_state_dict(legacy, prefix="denoiser.")
    assert set(out) == {"denoiser.cond_proj.weight", "denoiser.net.x"}
