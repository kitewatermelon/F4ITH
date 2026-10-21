import os

import pytest
import torch
from torch.utils.data import ConcatDataset

from rct.dataset import CheXpertDataset, MIMICCXRDataset, NIHChestXray14Dataset
from guided_diffusion_rcdm.rct_cxr_data import _SingleResolutionCXR, load_rct_cxr_data

MIMIC_ROOT = "/mnt/nvme1/mimic-cxr-jpg"
NIH_ROOT = "/mnt/nvme1/nih-chest-x-ray-14"
CHEXPERT_ROOT = "/mnt/nvme1/chexpert"

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(MIMIC_ROOT) and os.path.isdir(NIH_ROOT) and os.path.isdir(CHEXPERT_ROOT)),
    reason="one or more chest X-ray datasets not available on this machine",
)


def test_generator_yields_correct_shapes_ranges_and_equal_batches():
    gen = load_rct_cxr_data(
        mimic_dir=MIMIC_ROOT, nih_dir=NIH_ROOT, chexpert_dir=CHEXPERT_ROOT,
        batch_size=4, resolution=128, num_workers=0,
    )
    batch, batch_big, model_kwargs = next(gen)
    assert batch.shape == (4, 3, 128, 128)
    assert batch_big.shape == (4, 3, 128, 128)
    assert torch.equal(batch, batch_big)   # single-resolution: identical content
    assert batch.min() >= -1.0 - 1e-4 and batch.max() <= 1.0 + 1e-4
    assert model_kwargs == {}


def test_dataset_length_is_sum_of_three_frontal_train_splits():
    mimic_ds = MIMICCXRDataset(root=MIMIC_ROOT, split="train", label_csv="chexpert", frontal_only=True)
    nih_ds = NIHChestXray14Dataset(root=NIH_ROOT, split="train", frontal_only=True)
    chexpert_ds = CheXpertDataset(root=CHEXPERT_ROOT, split="train", frontal_only=True)
    combined = ConcatDataset([mimic_ds, nih_ds, chexpert_ds])
    wrapped = _SingleResolutionCXR(combined, resolution=128)
    assert len(wrapped) == len(mimic_ds) + len(nih_ds) + len(chexpert_ds)


def test_no_random_flip_deterministic_across_repeated_reads():
    mimic_ds = MIMICCXRDataset(root=MIMIC_ROOT, split="train", label_csv="chexpert", frontal_only=True)
    wrapped = _SingleResolutionCXR(mimic_ds, resolution=128)
    arr_a, arr2_a, _ = wrapped[0]
    arr_b, arr2_b, _ = wrapped[0]
    assert (arr_a == arr_b).all()
    assert (arr2_a == arr2_b).all()
