"""Adapter feeding RCT's combined CXR datasets into RCDM's training loop,
replacing image_datasets.load_data (which expects a flat image-file
directory — incompatible with our CSV-driven, multi-dataset pipeline)."""
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from rct.dataset import CheXpertDataset, MIMICCXRDataset, NIHChestXray14Dataset


def _center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class _SingleResolutionCXR(Dataset):
    """Wraps a raw (image, labels)-returning dataset, center-cropped and
    normalized to [-1,1], no horizontal flip (anatomically invalid for CXR
    — see RCT's own get_rct_aug). Single resolution: the same crop feeds
    both the diffusion UNet and CheXWorld conditioning."""
    def __init__(self, raw_dataset, resolution):
        self.raw_dataset = raw_dataset
        self.resolution = resolution

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        image, _ = self.raw_dataset[idx]   # PIL "L" image; labels unused (matches RCT)
        image = image.convert("RGB")

        arr = _center_crop_arr(image, self.resolution)
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2, 0, 1])

        return arr, arr, {}   # (batch, batch_big, model_kwargs) -- same array twice, single resolution


def _build_combined_raw(mimic_dir, nih_dir, chexpert_dir):
    """Raw ConcatDataset yielding (PIL image, labels) — before the
    center-crop/normalize wrapper."""
    mimic_ds = MIMICCXRDataset(root=mimic_dir, split="train", label_csv="chexpert", frontal_only=True)
    nih_ds = NIHChestXray14Dataset(root=nih_dir, split="train", frontal_only=True)
    chexpert_ds = CheXpertDataset(root=chexpert_dir, split="train", frontal_only=True)
    return ConcatDataset([mimic_ds, nih_ds, chexpert_ds])


def _build_combined_dataset(mimic_dir, nih_dir, chexpert_dir, resolution):
    return _SingleResolutionCXR(
        _build_combined_raw(mimic_dir, nih_dir, chexpert_dir), resolution
    )


def load_rct_cxr_data(
    *,
    mimic_dir: str,
    nih_dir: str,
    chexpert_dir: str,
    batch_size: int,
    resolution: int,
    num_workers: int = 12,
):
    """Generator over (batch, batch_big, model_kwargs) — drop-in replacement
    for guided_diffusion_rcdm.image_datasets.load_data in image_train.py.

    Under DDP (world_size > 1) each rank sees a disjoint shard via
    DistributedSampler; set_epoch is advanced each pass so shuffling differs
    across epochs. Without DDP it falls back to a plain shuffled loader."""
    dataset = _build_combined_dataset(mimic_dir, nih_dir, chexpert_dir, resolution)

    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    sampler = (
        DistributedSampler(dataset, shuffle=True, drop_last=True)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def build_sample_batch(*, mimic_dir, nih_dir, chexpert_dir, resolution, num_samples):
    """Return a fixed (num_samples, 3, resolution, resolution) tensor of real
    CXR images in [-1, 1] — the first `num_samples` dataset items, used as
    conditioning for in-training visualization (mirrors src/rct SampleCallback).

    Preprocessing: full-image resize to (resolution, resolution) with NO crop
    (full field of view), matching src/rct's validation/sampling protocol
    (TF.resize([H, W])). This intentionally differs from the center-cropped
    training pipeline (_SingleResolutionCXR)."""
    import torchvision.transforms.functional as TF

    raw = _build_combined_raw(mimic_dir, nih_dir, chexpert_dir)
    n = min(num_samples, len(raw))
    tensors = []
    for i in range(n):
        image, _ = raw[i]
        image = image.convert("RGB")
        t = TF.to_tensor(TF.resize(image, [resolution, resolution], antialias=True))  # (3,H,W) in [0,1]
        tensors.append(t * 2.0 - 1.0)   # -> [-1, 1]
    return torch.stack(tensors)
