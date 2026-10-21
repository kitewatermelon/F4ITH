import os
import pytest

from rct.dataset import NIHChestXray14Dataset

ROOT = "/mnt/nvme1/nih-chest-x-ray-14"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="NIH dataset not available on this machine")


def test_train_split_uses_train_val_list():
    ds = NIHChestXray14Dataset(root=ROOT, split="train")
    assert len(ds) == 86524


def test_validate_split_uses_test_list():
    ds = NIHChestXray14Dataset(root=ROOT, split="validate")
    assert len(ds) == 25596


def test_test_split_is_same_as_validate():
    ds_test = NIHChestXray14Dataset(root=ROOT, split="test")
    ds_validate = NIHChestXray14Dataset(root=ROOT, split="validate")
    assert len(ds_test) == len(ds_validate)


def test_frontal_only_is_a_noop_since_nih_has_no_laterals():
    ds_all = NIHChestXray14Dataset(root=ROOT, split="train", frontal_only=False)
    ds_frontal = NIHChestXray14Dataset(root=ROOT, split="train", frontal_only=True)
    assert len(ds_all) == len(ds_frontal)


def test_getitem_shape_and_labels():
    ds = NIHChestXray14Dataset(root=ROOT, split="train")
    image, labels = ds[0]
    assert image.size == (224, 224)
    assert image.mode == "L"
    assert labels.shape == (15,)
    assert labels.dtype.name == "float32"
    assert set(labels.tolist()) <= {0.0, 1.0}
