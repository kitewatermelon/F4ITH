import argparse

import pytest
import torch
import torch.nn as nn

import rct.denoiser as denoiser_mod
from rct.denoiser import RCTDenoiser


def _minimal_args(conditioning_mode, **overrides):
    args = argparse.Namespace(
        model_variant="S_16",
        img_size=32,
        attn_dropout=0.0,
        proj_dropout=0.0,
        long_skip_connection=False,
        depth=2,
        conditioning_mode=conditioning_mode,
        cond_drop_prob=0.1,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=0.05,
        noise_scale=1.0,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        sampling_method="heun",
        num_sampling_steps=10,
        cfg=1.5,
        interval_min=0.0,
        interval_max=1.0,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class _FakeNet(nn.Module):
    """Records the (cond_emb, patch_tokens) it was called with; returns z
    unchanged so the flow-matching loss computation has valid shapes."""
    def __init__(self):
        super().__init__()
        self.hidden_size = 16
        self.received = None

    def forward(self, z, t, cond_emb, patch_tokens):
        self.received = (cond_emb, patch_tokens)
        return z


class _FakeDino(denoiser_mod._CondEncoder):
    def encode(self, x):
        B = x.size(0)
        tokens = torch.randn(B, 1 + 256, 768)   # CLS + 256 patches, matches RAD-DINO patch14/224
        return tokens[:, 0], tokens[:, 1:]


def _patch(monkeypatch, net=None):
    net = net or _FakeNet()
    captured = {}

    def fake_model_fn(**kwargs):
        captured.update(kwargs)
        return net

    monkeypatch.setitem(denoiser_mod._MODEL_REGISTRY, "S_16", fake_model_fn)
    monkeypatch.setattr(denoiser_mod, "_load_raddino_encoder", lambda: _FakeDino())
    return captured, net


def test_cls_only_builds_cond_proj_and_null_cond_not_null_patch_tokens(monkeypatch):
    captured, _ = _patch(monkeypatch)
    denoiser = RCTDenoiser(_minimal_args("cls_only"))
    assert hasattr(denoiser, "cond_proj")
    assert hasattr(denoiser, "null_cond")
    assert not hasattr(denoiser, "null_patch_tokens")
    assert captured["conditioning_mode"] == "cls_only"


def test_patch_only_builds_null_patch_tokens_not_cond_proj_or_null_cond(monkeypatch):
    captured, _ = _patch(monkeypatch)
    denoiser = RCTDenoiser(_minimal_args("patch_only"))
    assert hasattr(denoiser, "null_patch_tokens")
    assert not hasattr(denoiser, "cond_proj")
    assert not hasattr(denoiser, "null_cond")
    assert captured["conditioning_mode"] == "patch_only"


def test_missing_conditioning_mode_raises(monkeypatch):
    _patch(monkeypatch)
    args = _minimal_args("cls_only")
    del args.conditioning_mode
    with pytest.raises(AttributeError):
        RCTDenoiser(args)


def test_forward_passes_only_cond_emb_in_cls_only_mode(monkeypatch):
    _, net = _patch(monkeypatch)
    denoiser = RCTDenoiser(_minimal_args("cls_only"))
    x = torch.randn(2, 3, 32, 32)
    cond_img = torch.randn(2, 3, 32, 32)
    denoiser(x, cond_img)
    cond_emb, patch_tokens = net.received
    assert cond_emb is not None
    assert patch_tokens is None


def test_forward_passes_only_patch_tokens_in_patch_only_mode(monkeypatch):
    _, net = _patch(monkeypatch)
    denoiser = RCTDenoiser(_minimal_args("patch_only"))
    x = torch.randn(2, 3, 32, 32)
    cond_img = torch.randn(2, 3, 32, 32)
    denoiser(x, cond_img)
    cond_emb, patch_tokens = net.received
    assert cond_emb is None
    assert patch_tokens is not None
