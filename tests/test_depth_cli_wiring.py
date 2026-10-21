import argparse
import sys

import torch.nn as nn

import rct.denoiser as denoiser_mod
from rct.denoiser import RCTDenoiser
from rct.train import parse_args


def test_parse_args_depth_defaults_to_13(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--conditioning_mode", "cls_only"])
    args = parse_args()
    assert args.depth == 13


def test_parse_args_depth_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--depth", "13", "--conditioning_mode", "cls_only"])
    args = parse_args()
    assert args.depth == 13


def _minimal_denoiser_args(**overrides):
    args = argparse.Namespace(
        model_variant="S_16",
        img_size=64,
        attn_dropout=0.0,
        proj_dropout=0.0,
        long_skip_connection=True,
        depth=13,
        conditioning_mode="cls_only",
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
    def __init__(self):
        super().__init__()
        self.hidden_size = 16


def _patch_model_fn_and_dino(monkeypatch):
    """Avoid a real model build and a real (network-fetching) RAD-DINO load —
    this test is only checking that `depth` reaches the model_fn kwargs."""
    captured = {}

    def fake_model_fn(**kwargs):
        captured.update(kwargs)
        return _FakeNet()

    monkeypatch.setitem(denoiser_mod._MODEL_REGISTRY, "S_16", fake_model_fn)
    monkeypatch.setattr(denoiser_mod, "_load_raddino_encoder", lambda: nn.Module())
    return captured


def test_rctdenoiser_passes_depth_to_model_fn(monkeypatch):
    captured = _patch_model_fn_and_dino(monkeypatch)
    RCTDenoiser(_minimal_denoiser_args(depth=13))
    assert captured["depth"] == 13


def test_rctdenoiser_defaults_depth_to_13_when_args_lacks_it(monkeypatch):
    captured = _patch_model_fn_and_dino(monkeypatch)
    args = _minimal_denoiser_args()
    del args.depth
    RCTDenoiser(args)
    assert captured["depth"] == 13
