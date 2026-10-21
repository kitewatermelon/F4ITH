import os
import sys

import pytest

from rct.train import CombinedCXRDataModule, parse_args
from rct.dataset import MIMICCXRDataset, NIHChestXray14Dataset, CheXpertDataset

MIMIC_ROOT = "/mnt/nvme1/mimic-cxr-jpg"
NIH_ROOT = "/mnt/nvme1/nih-chest-x-ray-14"
CHEXPERT_ROOT = "/mnt/nvme1/chexpert"

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(MIMIC_ROOT) and os.path.isdir(NIH_ROOT) and os.path.isdir(CHEXPERT_ROOT)),
    reason="one or more chest X-ray datasets not available on this machine",
)


def _make_dm(batch_size=8):
    dm = CombinedCXRDataModule(
        data_dir=MIMIC_ROOT, nih_dir=NIH_ROOT, chexpert_dir=CHEXPERT_ROOT,
        batch_size=batch_size, num_workers=0,
    )
    dm.setup()
    return dm


def test_combined_length_is_sum_of_three_frontal_train_splits():
    dm = _make_dm()
    mimic_len = len(MIMICCXRDataset(root=MIMIC_ROOT, split="train", label_csv="chexpert", frontal_only=True))
    nih_len = len(NIHChestXray14Dataset(root=NIH_ROOT, split="train", frontal_only=True))
    chexpert_len = len(CheXpertDataset(root=CHEXPERT_ROOT, split="train", frontal_only=True))
    assert len(dm.train_ds) == mimic_len + nih_len + chexpert_len


def test_sample_dict_has_no_labels_key():
    dm = _make_dm()
    sample = dm.train_ds[0]
    assert "image" in sample
    assert "labels" not in sample


def test_dataloader_batches_without_crashing_across_dataset_boundary():
    """Regression test for the real bug this spec exists to avoid: label
    dims differ across datasets (14 vs 15) and would crash default_collate
    if labels were included in the batch.

    Uses a plain DataLoader directly (not dm.train_dataloader()) because
    train_dataloader() unconditionally sets prefetch_factor=4, which
    torch's DataLoader rejects when num_workers=0 (as used here for test
    speed) — that's pre-existing, unrelated-to-this-task behavior in
    train_dataloader(), not something to work around by adding workers to
    a unit test or by changing train_dataloader() itself.
    """
    import torch as _torch

    dm = _make_dm(batch_size=8)
    loader = _torch.utils.data.DataLoader(dm.train_ds, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    assert batch["image"].shape == (8, 3, 256, 256)


def test_parse_args_nih_and_chexpert_dir_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--conditioning_mode", "cls_only"])
    args = parse_args()
    assert args.nih_dir == "/mnt/nvme1/nih-chest-x-ray-14"
    assert args.chexpert_dir == "/mnt/nvme1/chexpert"


def test_parse_args_nih_and_chexpert_dir_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--nih_dir", "/tmp/nih", "--chexpert_dir", "/tmp/chexpert",
        "--conditioning_mode", "cls_only",
    ])
    args = parse_args()
    assert args.nih_dir == "/tmp/nih"
    assert args.chexpert_dir == "/tmp/chexpert"
