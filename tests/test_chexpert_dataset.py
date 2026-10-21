import os
import numpy as np
import pytest

from rct.dataset import CheXpertDataset

ROOT = "/mnt/nvme1/chexpert"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="CheXpert dataset not available on this machine")


def test_train_frontal_only_count():
    ds = CheXpertDataset(root=ROOT, split="train", frontal_only=True)
    assert len(ds) == 191027


def test_validate_frontal_only_count():
    ds = CheXpertDataset(root=ROOT, split="validate", frontal_only=True)
    assert len(ds) == 202


def test_frontal_only_false_includes_laterals():
    ds_all = CheXpertDataset(root=ROOT, split="validate", frontal_only=False)
    ds_frontal = CheXpertDataset(root=ROOT, split="validate", frontal_only=True)
    assert len(ds_all) == 234
    assert len(ds_all) > len(ds_frontal)


def test_test_split_raises():
    with pytest.raises(ValueError):
        CheXpertDataset(root=ROOT, split="test")


def test_getitem_opens_real_file_and_strips_prefix():
    ds = CheXpertDataset(root=ROOT, split="validate", frontal_only=True)
    image, labels = ds[0]
    assert image.mode == "L"
    assert labels.shape == (14,)
    assert labels.dtype.name == "float32"


def test_uncertainty_zero_removes_negative_ones():
    ds = CheXpertDataset(root=ROOT, split="train", frontal_only=False, uncertainty="zero")
    step = max(1, len(ds) // 200)
    labels_sample = np.stack([ds[i][1] for i in range(0, len(ds), step)])
    assert not (labels_sample == -1).any()
