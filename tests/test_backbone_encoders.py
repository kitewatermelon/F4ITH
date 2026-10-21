"""Conditioning-backbone wiring: encoder contract, normalization, legacy compat."""
import argparse

import pytest
import torch
import torch.nn as nn

import rct.denoiser as denoiser_mod
from rct.denoiser import (
    CHEXWORLD_MEAN,
    CHEXWORLD_MEAN_LEGACY,
    CHEXWORLD_STD,
    CHEXWORLD_STD_LEGACY,
    XRV_DENSENET_DIM,
    XRV_DENSENET_MEAN,
    XRV_DENSENET_STD,
    RCTDenoiser,
    _CondEncoder,
    chexworld_norm_stats,
    resolve_chexworld_compat,
)


def _minimal_args(**overrides):
    args = argparse.Namespace(
        model_variant="S_16", img_size=32, attn_dropout=0.0, proj_dropout=0.0,
        long_skip_connection=False, depth=2, conditioning_mode="patch_only",
        cond_drop_prob=0.1, P_mean=-0.8, P_std=0.8, t_eps=0.05, noise_scale=1.0,
        ema_decay1=0.9999, ema_decay2=0.9996, sampling_method="heun",
        num_sampling_steps=10, cfg=1.5, interval_min=0.0, interval_max=1.0,
        backbone="chexworld",
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class _FakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 16

    def forward(self, z, t, cond_emb, patch_tokens):
        return z


class _FakeEncoder(_CondEncoder):
    """Minimal encoder honouring the encode() -> (pooled, patches) contract."""
    def __init__(self, n=196, dim=768):
        super().__init__()
        self.n, self.dim = n, dim

    def encode(self, x):
        patches = torch.randn(x.size(0), self.n, self.dim)
        return patches.mean(dim=1), patches


def _patch(monkeypatch, **kw):
    monkeypatch.setitem(denoiser_mod._MODEL_REGISTRY, "S_16", lambda **k: _FakeNet())
    monkeypatch.setattr(denoiser_mod, "_load_chexworld_encoder",
                        lambda compat="faithful": _FakeEncoder(**kw))


def test_faithful_stats_match_chexworld_pretraining():
    # CheXWorld's checkpoint records norm_type="default" -> ImageNet stats, which is
    # also what baseline/RCDM's get_ssl_models wrapper (and the distance metrics) use.
    mean, std = chexworld_norm_stats("faithful")
    assert torch.allclose(mean, torch.tensor([0.485, 0.456, 0.406]))
    assert torch.allclose(std, torch.tensor([0.229, 0.224, 0.225]))


def test_legacy_stats_are_the_pre_fix_halves():
    mean, std = chexworld_norm_stats("legacy")
    assert torch.allclose(mean, torch.full((3,), 0.5))
    assert torch.allclose(std, torch.full((3,), 0.5))


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown chexworld_compat"):
        chexworld_norm_stats("imagenet")


def test_missing_hparam_means_legacy():
    # Pre-fix checkpoints carry no flag: 0.5/0.5 stats plus the synthetic CLS token.
    assert resolve_chexworld_compat({"backbone": "chexworld"}) == "legacy"
    assert resolve_chexworld_compat({"chexworld_compat": "faithful"}) == "faithful"


def test_denoiser_defaults_to_faithful_stats(monkeypatch):
    _patch(monkeypatch)
    d = RCTDenoiser(_minimal_args())
    assert torch.allclose(d.enc_mean, CHEXWORLD_MEAN)
    assert torch.allclose(d.enc_std, CHEXWORLD_STD)


def test_denoiser_honours_legacy_flag(monkeypatch):
    _patch(monkeypatch)
    d = RCTDenoiser(_minimal_args(chexworld_compat="legacy"))
    assert torch.allclose(d.enc_mean, CHEXWORLD_MEAN_LEGACY)
    assert torch.allclose(d.enc_std, CHEXWORLD_STD_LEGACY)


def test_normalize_for_encoder_uses_selected_stats(monkeypatch):
    _patch(monkeypatch)
    img = torch.full((1, 3, 8, 8), -1.0)   # [-1,1] black -> 0.0 in [0,1]
    new = RCTDenoiser(_minimal_args())._normalize_for_encoder(img)
    old = RCTDenoiser(_minimal_args(chexworld_compat="legacy"))._normalize_for_encoder(img)
    assert torch.allclose(new[0, :, 0, 0], -CHEXWORLD_MEAN / CHEXWORLD_STD)
    assert torch.allclose(old[0, :, 0, 0], torch.full((3,), -1.0))


def test_raddino_backbone_unaffected(monkeypatch):
    monkeypatch.setitem(denoiser_mod._MODEL_REGISTRY, "S_16", lambda **kw: _FakeNet())
    monkeypatch.setattr(denoiser_mod, "_load_raddino_encoder", lambda: _FakeEncoder(256, 768))
    d = RCTDenoiser(_minimal_args(backbone="raddino", chexworld_compat="legacy"))
    assert torch.allclose(d.enc_mean, denoiser_mod.RADDINO_MEAN)


def test_unknown_backbone_rejected(monkeypatch):
    monkeypatch.setitem(denoiser_mod._MODEL_REGISTRY, "S_16", lambda **kw: _FakeNet())
    with pytest.raises(ValueError, match="unknown backbone"):
        RCTDenoiser(_minimal_args(backbone="resnet50"))


def test_cnn_backbone_sets_token_grid_and_cond_dim(monkeypatch):
    captured = {}

    def fake_model_fn(**kwargs):
        captured.update(kwargs)
        return _FakeNet()

    monkeypatch.setitem(denoiser_mod._MODEL_REGISTRY, "S_16", fake_model_fn)
    monkeypatch.setattr(denoiser_mod, "_load_xrv_densenet_encoder",
                        lambda: _FakeEncoder(49, XRV_DENSENET_DIM))
    d = RCTDenoiser(_minimal_args(backbone="xrv_densenet121"))
    # 224 / stride 32 -> 7x7 = 49 conditioning tokens, 1024-wide
    assert d.null_patch_tokens.shape == (1, 49, XRV_DENSENET_DIM)
    # cross-attention must project from the encoder's width, not a hardcoded 768
    assert captured["cond_dim"] == XRV_DENSENET_DIM
    assert torch.allclose(d.enc_mean, XRV_DENSENET_MEAN)
    assert torch.allclose(d.enc_std, XRV_DENSENET_STD)


def test_frozen_encoder_stays_in_eval_mode(monkeypatch):
    # BatchNorm in a CNN backbone would otherwise keep updating its running stats.
    _patch(monkeypatch)
    d = RCTDenoiser(_minimal_args())
    d.train()
    assert not d.encoder.training and d.net.training


def test_pooled_source_mean_overrides_encoder_pooling(monkeypatch):
    _patch(monkeypatch)
    d = RCTDenoiser(_minimal_args(conditioning_mode="cls_only", pooled_cond_source="mean"))
    pooled, patches = d.encode_cond(torch.zeros(2, 3, 32, 32))
    assert torch.allclose(pooled, patches.mean(dim=1))
