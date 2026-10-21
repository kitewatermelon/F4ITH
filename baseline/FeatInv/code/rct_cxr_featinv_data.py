"""RCT combined-CXR data feeding FeatInv's ControlNet {jpg,txt,hint} contract,
conditioned on CheXWorld spatial patch tokens. RCT-style augmentation (no
flip); CheXWorld features computed on cuda:{LOCAL_RANK} (DDP-safe)."""
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "RCDM"))
from guided_diffusion_rcdm.get_ssl_models import get_model as get_ssl_model  # noqa: E402
from rct.dataset import CheXpertDataset, MIMICCXRDataset, NIHChestXray14Dataset  # noqa: E402

DEFAULT_PROMPT = "a high-quality, detailed, and professional image"


def _combined_raw(mimic_dir, nih_dir, chexpert_dir):
    mimic = MIMICCXRDataset(root=mimic_dir, split="train", label_csv="chexpert", frontal_only=True)
    nih = NIHChestXray14Dataset(root=nih_dir, split="train", frontal_only=True)
    chex = CheXpertDataset(root=chexpert_dir, split="train", frontal_only=True)
    return ConcatDataset([mimic, nih, chex])


def _chexworld_hint(feature_model, crop_t, hint_resolution, device, pool=False):
    """crop_t: (3,H,W) in [0,1]. Returns (h,h,768) HWC float32 numpy.

    pool=False: spatial patch-token grid (unpooled FeatInv, default).
    pool=True:  the mean-pooled global descriptor broadcast to every grid cell,
                so the hint carries no finegrained spatial detail -- the pooled
                variant used as a Table-1 baseline (matches the FeatInv paper's
                pooled feature maps and keeps the ControlNet hint shape identical)."""
    hint_in = TF.resize(crop_t, [hint_resolution, hint_resolution], antialias=True)
    hint_in = (hint_in * 2.0 - 1.0).unsqueeze(0).to(device)   # (1,3,R,R) in [-1,1]
    with torch.no_grad():
        tokens = feature_model(hint_in)                       # (1, N, 768)
    n = tokens.shape[1]
    h = int(round(n ** 0.5))
    grid = tokens.squeeze(0).reshape(h, h, -1).float()        # (h,h,768)
    if pool:
        grid = grid.mean(dim=(0, 1), keepdim=True).expand(h, h, -1).contiguous()
    return grid.cpu().numpy()


class RCTCXRFeatInvDataset(Dataset):
    def __init__(self, mimic_dir, nih_dir, chexpert_dir, hint_resolution=224,
                 target_resolution=256, cond_drop_prob=0.15, prompt=DEFAULT_PROMPT,
                 pool=False):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.hint_resolution = hint_resolution
        self.target_resolution = target_resolution
        self.cond_drop_prob = cond_drop_prob
        self.prompt = prompt
        self.pool = pool
        self.raw = _combined_raw(mimic_dir, nih_dir, chexpert_dir)
        # RCT-style aug: Resize(256) -> RandomResizedCrop(scale 0.8-1.2), NO flip.
        self.aug = transforms.Compose([
            transforms.Resize((target_resolution, target_resolution)),
            transforms.RandomResizedCrop(target_resolution, scale=(0.8, 1.2), antialias=True),
        ])
        self.feature_model = get_ssl_model("chexworld", pool=False).to(self.device).eval()
        for p in self.feature_model.parameters():
            p.requires_grad_(False)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        image, _ = self.raw[idx]
        image = image.convert("RGB")
        crop = self.aug(image)                       # PIL 256x256
        crop_t = TF.to_tensor(crop)                  # (3,256,256) in [0,1]
        jpg = (crop_t * 2.0 - 1.0).permute(1, 2, 0).numpy().astype(np.float32)  # (256,256,3) HWC
        hint = _chexworld_hint(self.feature_model, crop_t, self.hint_resolution, self.device,
                               pool=self.pool)
        txt = self.prompt
        if random.random() < self.cond_drop_prob:
            hint = np.zeros_like(hint)
            txt = ""
        return {"jpg": jpg, "txt": txt, "hint": hint}


def build_sample_batch(mimic_dir, nih_dir, chexpert_dir, num_samples,
                       hint_resolution=224, target_resolution=256, prompt=DEFAULT_PROMPT,
                       pool=False):
    """Fixed batch for in-training sampling. NO-crop full-resize (RCT validation
    preprocessing), no cond-drop."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    raw = _combined_raw(mimic_dir, nih_dir, chexpert_dir)
    feature_model = get_ssl_model("chexworld", pool=False).to(device).eval()
    n = min(num_samples, len(raw))
    jpgs, hints = [], []
    for i in range(n):
        image, _ = raw[i]
        image = image.convert("RGB")
        crop_t = TF.to_tensor(TF.resize(image, [target_resolution, target_resolution], antialias=True))
        jpgs.append((crop_t * 2.0 - 1.0).permute(1, 2, 0))                 # (256,256,3)
        hints.append(torch.from_numpy(_chexworld_hint(feature_model, crop_t, hint_resolution, device, pool=pool)))
    return {"jpg": torch.stack(jpgs), "txt": [prompt] * n, "hint": torch.stack(hints)}
