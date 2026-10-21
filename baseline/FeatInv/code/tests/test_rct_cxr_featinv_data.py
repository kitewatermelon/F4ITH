import os
import numpy as np
import pytest

from rct_cxr_featinv_data import RCTCXRFeatInvDataset, build_sample_batch

MIMIC = "/mnt/nvme1/mimic-cxr-jpg"
NIH = "/mnt/nvme1/nih-chest-x-ray-14"
CHEX = "/mnt/nvme1/chexpert"

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(MIMIC) and os.path.isdir(NIH) and os.path.isdir(CHEX)),
    reason="CXR datasets not available",
)


def test_getitem_shapes_and_ranges():
    ds = RCTCXRFeatInvDataset(MIMIC, NIH, CHEX)
    item = ds[0]
    assert item["jpg"].shape == (256, 256, 3)
    assert item["jpg"].dtype == np.float32
    assert item["jpg"].min() >= -1.0 - 1e-4 and item["jpg"].max() <= 1.0 + 1e-4
    assert item["hint"].shape == (14, 14, 768)
    assert isinstance(item["txt"], str)


def test_build_sample_batch_collated_shapes():
    b = build_sample_batch(MIMIC, NIH, CHEX, num_samples=4)
    assert tuple(b["jpg"].shape) == (4, 256, 256, 3)
    assert tuple(b["hint"].shape) == (4, 14, 14, 768)
    assert len(b["txt"]) == 4


def test_pooled_hint_is_spatially_constant():
    """Pooled dataset item keeps the (14,14,768) shape but every grid cell must
    hold the SAME vector (spatial detail discarded)."""
    ds_pl = RCTCXRFeatInvDataset(MIMIC, NIH, CHEX, cond_drop_prob=0.0, pool=True)
    hint_pl = ds_pl[0]["hint"]
    assert hint_pl.shape == (14, 14, 768)
    assert np.allclose(hint_pl, hint_pl[0, 0][None, None, :], atol=1e-5)


def test_pooled_hint_equals_mean_of_unpooled():
    """On the SAME (deterministic, no-crop) sample batch, the pooled hint at every
    cell must equal the mean of the unpooled grid over its 14x14 positions."""
    sp = build_sample_batch(MIMIC, NIH, CHEX, num_samples=2, pool=False)["hint"]  # (2,14,14,768)
    pl = build_sample_batch(MIMIC, NIH, CHEX, num_samples=2, pool=True)["hint"]
    assert tuple(pl.shape) == (2, 14, 14, 768)
    for i in range(sp.shape[0]):
        mean_sp = sp[i].reshape(-1, 768).mean(axis=0)
        assert np.allclose(pl[i, 0, 0], mean_sp, atol=1e-4)
        assert np.allclose(pl[i], mean_sp[None, None, :], atol=1e-5)
