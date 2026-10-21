import sys

from rct.train import parse_args


def test_parse_args_accepts_adaln_single(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "--conditioning_mode", "cls_only", "--t_embed_mode", "adaln_single"],
    )
    args = parse_args()
    assert args.t_embed_mode == "adaln_single"
