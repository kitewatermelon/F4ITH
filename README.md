# F4ITH

**An Efficient and Faithful Inversion Transformer for Feature-Agnostic Probing of Chest X-ray Encoder Representations**

Yeon Su Park, Ji-In Choi, Ji Hye Won

> F4ITH is a lightweight flow-matching generator that reconstructs chest X-ray images conditioned solely on the spatial or global features of a frozen medical-image encoder — without a pretrained external generator or text prompt.

<!-- ![Architecture](docs/arch.png) -->

## Highlights

- **Conditioning purity** — reconstruction is explained only by the encoder features; no text prompt or pretrained natural-image prior is involved.
- **Dual conditioning modes** — spatial features via per-block cross-attention (`patch_only`), or global features via adaptive layer normalization (`cls_only`).
- **Lightweight** — 1/7 the trainable parameters and several-fold less inference compute than prior methods (RCDM, FeatInv).
- **Encoder-agnostic** — swap the frozen encoder to compare what different representations preserve; tested on CheXWorld, RAD-DINO, and TorchXRayVision DenseNet-121.
- **Faithful** — under identical global conditioning, lowers generation-to-conditioning retrieval rank from 61.53 to 1.66.

## Repository Structure

```
F4ITH/
├── src/rct/                  # Core model
│   ├── model.py              #   CrossAttnJiT transformer (U-ViT + cross-attn + adaLN)
│   ├── denoiser.py           #   Flow-matching denoiser & conditioning encoders
│   ├── dataset.py            #   MIMIC-CXR, NIH, CheXpert dataloaders
│   ├── train.py              #   Training entry point (PyTorch Lightning)
│   └── rope_util.py          #   2D RoPE & RMSNorm utilities
├── baseline/
│   ├── FeatInv/              # FeatInv baseline (ControlNet + LDM)
│   └── RCDM/                 # RCDM baseline (conditional diffusion)
├── backbone/
│   └── CheXWorld/            # CheXWorld encoder (I-JEPA ViT)
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
# 1. PyTorch (match your CUDA version)
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130

# 2. stable-pretraining (CheXWorld encoder loader, not on PyPI)
pip install "stable-pretraining @ git+https://github.com/rbalestr-lab/stable-pretraining"

# 3. This package
pip install -e .
```

## Training

F4ITH requires [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) and a frozen encoder checkpoint.

```bash
# Spatial conditioning (cross-attention on patch tokens)
python src/rct/train.py \
    --data_dir /path/to/mimic-cxr-jpg \
    --output_dir output/f4ith_spatial_chexworld \
    --conditioning_mode patch_only \
    --backbone chexworld

# Global conditioning (adaLN on CLS token)
python src/rct/train.py \
    --data_dir /path/to/mimic-cxr-jpg \
    --output_dir output/f4ith_global_chexworld \
    --conditioning_mode cls_only \
    --backbone chexworld
```

Backbone options: `chexworld`, `raddino`, `xrv_densenet`.

## Supported Encoders

| Encoder | Architecture | Training | Spatial tokens | Global token |
|---|---|---|---|---|
| [CheXWorld](https://github.com/RPIDIAL/CheXWorld) | ViT-B/16 | I-JEPA (self-supervised) | 14 x 14 patch | Mean-pooled |
| [RAD-DINO](https://huggingface.co/microsoft/rad-dino) | ViT-B/14 | DINOv2 (self-supervised) | 16 x 16 patch | [CLS] token |
| [TorchXRayVision](https://github.com/mlmed/torchxrayvision) | DenseNet-121 | Supervised (multi-label) | 7 x 7 feature map | GAP |

## Citation

```bibtex
@article{park2026f4ith,
  title   = {F4ITH: An Efficient and Faithful Inversion Transformer for Feature-Agnostic Probing of Chest X-ray Encoder Representations},
  author  = {Park, Yeon Su and Choi, Ji-In and Won, Ji Hye},
  year    = {2026},
  note    = {Preprint}
}
```

## License

This repository is for research purposes. See individual baseline directories for their respective licenses.
