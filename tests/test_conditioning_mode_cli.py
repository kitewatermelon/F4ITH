import sys

import pytest

from rct.train import parse_args


def test_conditioning_mode_is_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    with pytest.raises(SystemExit):
        parse_args()


def test_conditioning_mode_accepts_cls_only(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--conditioning_mode", "cls_only"])
    args = parse_args()
    assert args.conditioning_mode == "cls_only"


def test_conditioning_mode_accepts_patch_only(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--conditioning_mode", "patch_only"])
    args = parse_args()
    assert args.conditioning_mode == "patch_only"


def test_conditioning_mode_rejects_invalid_value(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--conditioning_mode", "bogus"])
    with pytest.raises(SystemExit):
        parse_args()
