from contextlib import contextmanager
from typing import NamedTuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from rct.model import _MODEL_REGISTRY


RADDINO_MODEL = "microsoft/rad-dino"
RADDINO_DIM = 768
RADDINO_IMG_SIZE = 224
RADDINO_PATCH_SIZE = 14
RADDINO_MEAN = torch.tensor([0.5307, 0.5307, 0.5307])
RADDINO_STD = torch.tensor([0.2583, 0.2583, 0.2583])

CHEXWORLD_MODEL = "/home/yspark/RCT/backbone/weights/chexworld_pretrained.tar"
CHEXWORLD_DIM = 768
CHEXWORLD_IMG_SIZE = 224
CHEXWORLD_PATCH_SIZE = 16
# CheXWorld was pretrained with norm_type="default" (the checkpoint's own args, resolved
# by backbone/CheXWorld/data_utils/xray_transform.py:get_mean_std) -> ImageNet stats.
# Feeding it 0.5/0.5 puts the frozen encoder off-distribution: its pooled features then
# match the faithfully-normalised ones at cos 0.45 on real MIMIC images, barely above the
# 0.30 that unrelated image pairs score, versus cos 0.9985 once normalised correctly.
# baseline/RCDM/guided_diffusion_rcdm/get_ssl_models.py (used by RCDM, FeatInv and the
# distance metrics) has always used these stats, so this keeps RCT in the same space.
CHEXWORLD_MEAN = torch.tensor([0.485, 0.456, 0.406])
CHEXWORLD_STD = torch.tensor([0.229, 0.224, 0.225])
# Pre-fix checkpoints were trained on 0.5/0.5 stats *and* on a synthetic CLS token (see
# _build_chexworld_vit); they learned to invert those features, so loading one has to
# reproduce both. See resolve_chexworld_compat().
CHEXWORLD_MEAN_LEGACY = torch.tensor([0.5, 0.5, 0.5])
CHEXWORLD_STD_LEGACY = torch.tensor([0.5, 0.5, 0.5])

CHEXWORLD_COMPAT_MODES = ("faithful", "legacy")

# TorchXRayVision DenseNet-121 supervised on the union of the public CXR label sets.
# denseblock4 is stride 32, so 224 inputs give a 7x7 = 49 token grid -- coarser than the
# ViT backbones, which cross-attention does not care about (no positional embedding on
# the keys, and the token count only has to be consistent between train and inference).
XRV_DENSENET_WEIGHTS = "densenet121-res224-all"
XRV_DENSENET_DIM = 1024
XRV_DENSENET_IMG_SIZE = 224
XRV_DENSENET_PATCH_SIZE = 32
# xrv.utils.normalize scales [0, 1] to roughly [-1024, 1024] as (2*u - 1) * 1024, which is
# exactly (u - 0.5) / (1/2048) -- so it fits the same mean/std path as the other backbones.
XRV_DENSENET_MEAN = torch.tensor([0.5, 0.5, 0.5])
XRV_DENSENET_STD = torch.tensor([1.0 / 2048, 1.0 / 2048, 1.0 / 2048])


def chexworld_norm_stats(mode: str):
    """(mean, std) tensors for the ChexWorld encoder's input normalization.

    "faithful" matches CheXWorld's own norm_type="default"; "legacy" is the
    0.5/0.5 stats baked into pre-fix checkpoints.
    """
    if mode == "faithful":
        return CHEXWORLD_MEAN, CHEXWORLD_STD
    if mode == "legacy":
        return CHEXWORLD_MEAN_LEGACY, CHEXWORLD_STD_LEGACY
    raise ValueError(f"unknown chexworld_compat {mode!r}, expected one of {CHEXWORLD_COMPAT_MODES}")


def resolve_chexworld_compat(hparams: dict) -> str:
    """Which ChexWorld encoder variant a saved checkpoint was trained against.

    Checkpoints written before the flag existed carry no key and were trained on
    the legacy encoder (0.5/0.5 normalization + synthetic CLS token), so a missing
    key means "legacy". Only for loading saved runs -- fresh training defaults to
    "faithful" via the CLI.
    """
    return hparams.get("chexworld_compat", "legacy")


class _CondEncoder(nn.Module):
    """A frozen conditioning encoder.

    Subclasses implement ``encode(x) -> (pooled (B, D), patches (B, N, D))``.

    Returning the two explicitly lets every backbone state how its pooled vector
    is formed -- a genuine CLS token for RAD-DINO, global average pooling for the
    ones that never had one -- instead of forcing each to emit a ViT-shaped
    (B, 1+N, D) sequence with a pooled token at index 0. That older contract is
    what pushed the ChexWorld loader into fabricating a CLS token its pretraining
    never had, and it has no meaning at all for a CNN feature map.
    """

    def encode(self, x: torch.Tensor):
        raise NotImplementedError


class _RadDinoEncoder(_CondEncoder):
    """Wraps HF `Dinov2Model` (microsoft/rad-dino).

    RAD-DINO has a real pretrained CLS token, so `pooled` is that token and the
    dense patch tokens follow it in the sequence.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode(self, x: torch.Tensor):
        tokens = self.model(pixel_values=x).last_hidden_state   # (B, 1+N, 768)
        return tokens[:, 0], tokens[:, 1:]


def _load_raddino_encoder() -> _RadDinoEncoder:
    """Load RAD-DINO from the HF hub (no local checkpoint required)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(RADDINO_MODEL)
    model.eval()
    return _RadDinoEncoder(model)


class _ChexWorldEncoder(_CondEncoder):
    """Wraps a pretrained ChexWorld ViT-Base/16 encoder (IWM SSL on chest X-rays).

    CheXWorld was pretrained *without* a class token, so `pooled` is the mean over
    patch tokens -- the same pooled feature the distance metrics use. `has_cls` is
    only ever true for the legacy compat encoder, which carries the synthetic CLS
    that pre-fix checkpoints were trained against.
    """
    def __init__(self, encoder_model, has_cls: bool = False):
        super().__init__()
        self.model = encoder_model
        self.model.eval()
        self.has_cls = has_cls

    def encode(self, x: torch.Tensor):
        tokens = self.model.forward_features(x)
        if self.has_cls:
            # Legacy: the pre-fix encoder reported its synthetic CLS as the pooled
            # vector, so a checkpoint trained with pooled_cond_source="cls" keeps
            # receiving exactly that.
            return tokens[:, 0], tokens[:, 1:]
        return tokens.mean(dim=1), tokens


class _XRVDenseNetEncoder(_CondEncoder):
    """Wraps a TorchXRayVision DenseNet-121 (supervised on public CXR labels).

    A CNN has no token sequence: the last dense block's feature map is flattened
    into (B, h*w, C) "patch" tokens and `pooled` is their global average, which is
    exactly the embedding xrv's own `features2()` returns (it is relu + GAP over
    the same map).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def encode(self, x: torch.Tensor):
        # xrv models take a single channel; CXRs reach us as replicated grayscale.
        x = x.mean(dim=1, keepdim=True)
        feat = F.relu(self.model.features(x))          # (B, C, h, w), relu as in features2()
        patches = feat.flatten(2).transpose(1, 2)      # (B, h*w, C)
        return patches.mean(dim=1), patches


def _load_xrv_densenet_encoder() -> _XRVDenseNetEncoder:
    """Load TorchXRayVision DenseNet-121 (weights downloaded on first use)."""
    import torchxrayvision as xrv
    model = xrv.models.DenseNet(weights=XRV_DENSENET_WEIGHTS)
    model.eval()
    return _XRVDenseNetEncoder(model)


def _load_chexworld_encoder(compat: str = "faithful") -> _ChexWorldEncoder:
    """
    Load ChexWorld ViT-Base/16 target_encoder (EMA teacher) from checkpoint.

    We use target_encoder (not encoder) because:
    - More stable and smooth (EMA updated during training)
    - Better generalization for downstream tasks
    - Less prone to overfitting

    ChexWorld was trained WITHOUT a class token (with_cls_token=False), so
    pos_embed has shape (1, 196, 768) for patches only and `compat="faithful"`
    builds the ViT that way. `compat="legacy"` reproduces the pre-fix encoder,
    which prepended a synthetic CLS token so the backbone could pretend to be a
    RAD-DINO-shaped (B, 1+N, D) sequence; checkpoints trained against it need it
    back. New runs should never use it.

    Requires: timm library (should be installed with stable_pretraining)
    """
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])

    if compat not in CHEXWORLD_COMPAT_MODES:
        raise ValueError(f"unknown chexworld_compat {compat!r}, expected one of {CHEXWORLD_COMPAT_MODES}")

    # Load checkpoint
    checkpoint = torch.load(CHEXWORLD_MODEL, map_location='cpu', weights_only=False)
    model_state = checkpoint['model']

    # Extract target_encoder state dict (EMA teacher - more stable for downstream tasks)
    # Using target_encoder instead of encoder for better generalization
    encoder_state = {}
    for k, v in model_state.items():
        if k.startswith('target_encoder.'):
            new_k = k.replace('target_encoder.', '', 1)
            encoder_state[new_k] = v

    has_cls = compat == "legacy"
    try:
        import timm
        model = timm.models.vision_transformer.VisionTransformer(
            img_size=CHEXWORLD_IMG_SIZE,
            patch_size=CHEXWORLD_PATCH_SIZE,
            in_chans=3,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            # Faithful: no class token, matching pretraining and the vendored encoder in
            # baseline/RCDM/guided_diffusion_rcdm/get_ssl_models.py.
            class_token=has_cls,
            # NOT 'avg': that makes timm swap the final LayerNorm for fc_norm, silently
            # dropping CheXWorld's pretrained `norm` weights. We only ever call
            # forward_features, so no pooling head is needed at all.
            global_pool='token' if has_cls else '',
            # img_size above only sizes the pretrained pos_embed for checkpoint
            # loading; dynamic_img_size lets forward() run at any resolution by
            # interpolating pos_embed to the actual input's patch grid (e.g. for
            # --img_size 128 ablations run natively instead of upsampled to 224).
            dynamic_img_size=True,
        )

        if has_cls:
            # Legacy only: ChexWorld's pos_embed is (1, 196, 768) but a class-token ViT
            # wants (1, 197, 768), so the pre-fix loader invented both the extra position
            # and the token itself out of the mean of the pretrained positions. That
            # phantom token then took part in all 12 self-attention layers.
            if 'pos_embed' in encoder_state and encoder_state['pos_embed'].shape[1] == 196:
                original_pos = encoder_state['pos_embed']
                cls_pos = original_pos.mean(dim=1, keepdim=True)          # (1, 1, 768)
                encoder_state['pos_embed'] = torch.cat([cls_pos, original_pos], dim=1)
            if 'cls_token' not in encoder_state:
                encoder_state['cls_token'] = encoder_state['pos_embed'][:, :1, :]

        # strict=True on purpose: a silently skipped key means part of the frozen
        # encoder is randomly initialised, which is exactly the failure this loader
        # hid before (global_pool='avg' dropped the pretrained final LayerNorm).
        model.load_state_dict(encoder_state, strict=True)
    except ImportError:
        raise ImportError(
            "ChexWorld loader requires 'timm' library. "
            "Install with: pip install timm"
        )

    return _ChexWorldEncoder(model, has_cls=has_cls)


class _EncoderConfig(NamedTuple):
    """Everything RCTDenoiser needs to know about a frozen conditioning encoder."""
    encoder: _CondEncoder
    img_size: int          # resolution the encoder runs at
    patch_size: int        # output stride, so (img_size // patch_size) ** 2 = token count
    dim: int               # token width
    mean: torch.Tensor     # input normalization, applied to [0, 1] images
    std: torch.Tensor


def _configure_raddino(args) -> _EncoderConfig:
    return _EncoderConfig(
        encoder=_load_raddino_encoder(),
        img_size=RADDINO_IMG_SIZE,
        patch_size=RADDINO_PATCH_SIZE,
        dim=RADDINO_DIM,
        mean=RADDINO_MEAN,
        std=RADDINO_STD,
    )


def _configure_chexworld(args) -> _EncoderConfig:
    compat = getattr(args, 'chexworld_compat', 'faithful')
    if compat == "legacy":
        print("⚠️  ChexWorld encoder running in legacy compat mode (0.5/0.5 normalization "
              "+ synthetic CLS token); that is for pre-fix checkpoints only.", flush=True)
    mean, std = chexworld_norm_stats(compat)
    return _EncoderConfig(
        encoder=_load_chexworld_encoder(compat),
        # ChexWorld's ViT interpolates its pos_embed to any input grid (see jepa_vit.py's
        # interpolate_pos_encoding), so the encoder can run natively at the denoiser's own
        # resolution instead of always being upsampled to its pretraining size -- saves
        # compute at smaller --img_size (e.g. 128 ablations).
        img_size=args.img_size,
        patch_size=CHEXWORLD_PATCH_SIZE,
        dim=CHEXWORLD_DIM,
        mean=mean,
        std=std,
    )


def _configure_xrv_densenet121(args) -> _EncoderConfig:
    return _EncoderConfig(
        encoder=_load_xrv_densenet_encoder(),
        img_size=XRV_DENSENET_IMG_SIZE,
        patch_size=XRV_DENSENET_PATCH_SIZE,
        dim=XRV_DENSENET_DIM,
        mean=XRV_DENSENET_MEAN,
        std=XRV_DENSENET_STD,
    )


# Conditioning backbones. Adding one means adding a _CondEncoder subclass with an
# encode() method plus an entry here; nothing downstream is backbone-specific.
_BACKBONES = {
    "raddino": _configure_raddino,
    "chexworld": _configure_chexworld,
    "xrv_densenet121": _configure_xrv_densenet121,
}


# Backward-compat: checkpoints trained before the dino->encoder/cond rename carry
# the old parameter/submodule prefixes. Remap them at load time so existing
# checkpoints keep loading with strict=True. Longer prefixes first so
# "dino_proj." is matched before "dino.".
_LEGACY_KEY_MAP = {"dino_proj.": "cond_proj.", "dino.": "encoder."}


def remap_legacy_state_dict(state_dict: dict, prefix: str = "") -> dict:
    """Rename legacy dino-prefixed keys to their encoder/cond equivalents.

    `prefix` is any leading namespace before the renamed attribute (e.g.
    "denoiser." for a LightningModule checkpoint, "" for an already-stripped
    denoiser state dict). Keys already using the new names pass through
    unchanged, so this is safe to apply unconditionally.
    """
    out = {}
    for k, v in state_dict.items():
        if prefix and k.startswith(prefix):
            head, tail = prefix, k[len(prefix):]
        else:
            head, tail = "", k
        for old, new in _LEGACY_KEY_MAP.items():
            if tail.startswith(old):
                tail = new + tail[len(old):]
                break
        out[head + tail] = v
    return out


class RCTDenoiser(nn.Module):
    """
    Flow-matching denoiser: CrossAttnJiT conditioned on a frozen backbone encoder (RAD-DINO or ChexWorld), via
    a pooled CLS embedding driving adaLN-Zero (`conditioning_mode="cls_only"`,
    global), the same pooled CLS as a single cross-attention memory token
    (`conditioning_mode="cls_cross"`), or dense patch tokens
    (`conditioning_mode="patch_only"`, cross-attention, spatial).

    During training  : forward(x, cond_img) -> scalar loss
    During inference  : generate(cond_img)   -> generated image tensor
    """

    def __init__(self, args):
        super().__init__()

        model_variant = getattr(args, 'model_variant', 'S_16')
        model_fn = _MODEL_REGISTRY[model_variant]
        in_channels = getattr(args, 'in_channels', 3)
        self.conditioning_mode = args.conditioning_mode
        # For cls_only ("pooled") conditioning: "cls" takes whatever the encoder calls
        # its pooled output, "mean" always averages the patch tokens. They differ only
        # for RAD-DINO, whose pooled output is a real CLS token; the backbones without
        # one already pool by averaging, which is also the distance metric's pool=True
        # feature.
        self.pooled_cond_source = getattr(args, 'pooled_cond_source', 'cls')
        # The encoder is built first: its token width sets the model's cond_dim, which
        # is what cross-attention projects the conditioning tokens from.
        backbone_type = getattr(args, 'backbone', 'raddino').lower()
        if backbone_type not in _BACKBONES:
            raise ValueError(f"unknown backbone {backbone_type!r}, expected one of {tuple(_BACKBONES)}")
        enc_cfg = _BACKBONES[backbone_type](args)
        self.backbone = backbone_type
        self.encoder = enc_cfg.encoder
        self.enc_img_size = enc_cfg.img_size
        self.enc_patch_size = enc_cfg.patch_size
        self.enc_dim = enc_cfg.dim
        self.enc_mean = enc_cfg.mean
        self.enc_std = enc_cfg.std
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        model_kwargs = dict(
            conditioning_mode=self.conditioning_mode,
            cond_dim=self.enc_dim,
            input_size=args.img_size,
            in_channels=in_channels,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            long_skip_connection=getattr(args, 'long_skip_connection', False),
            depth=getattr(args, 'depth', 13),
            use_final_conv=getattr(args, 'use_final_conv', True),
            t_embed_mode=getattr(args, 't_embed_mode', 'adaln'),
            in_context_len=getattr(args, 'in_context_len', 0),
            in_context_start=getattr(args, 'in_context_start', 4),
        )
        # Override the variant's default width/heads (e.g. --hidden_size 512 --num_heads 8)
        if getattr(args, 'hidden_size', None) is not None:
            model_kwargs['hidden_size'] = args.hidden_size
        if getattr(args, 'num_heads', None) is not None:
            model_kwargs['num_heads'] = args.num_heads
        if getattr(args, 'patch_size', None) is not None:
            model_kwargs['patch_size'] = args.patch_size
        self.net = model_fn(**model_kwargs)
        self.img_size = args.img_size
        self.in_channels = in_channels
        hidden_size = self.net.hidden_size

        if self.conditioning_mode in ("cls_only", "cls_cross"):
            # Both drive off the pooled CLS: cls_only feeds it to adaLN, cls_cross
            # feeds it as a single cross-attention memory token.
            self.cond_proj = nn.Linear(self.enc_dim, hidden_size)
            self.null_cond = nn.Parameter(torch.zeros(1, hidden_size))
        else:
            num_enc_patches = (self.enc_img_size // self.enc_patch_size) ** 2
            self.null_patch_tokens = nn.Parameter(torch.zeros(1, num_enc_patches, self.enc_dim))

        self.cond_drop_prob = args.cond_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def train(self, mode: bool = True):
        """Keep the frozen encoder in eval mode.

        nn.Module.train() recurses into children, which would put the backbone's
        BatchNorm layers back into training mode and let their running statistics
        drift on the conditioning images -- silently changing a "frozen" encoder's
        output as training goes on. Harmless for the ViT backbones, not for a CNN.
        """
        super().train(mode)
        self.encoder.eval()
        return self

    def _normalize_for_encoder(self, img: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] image to backbone encoder's expected normalization."""
        img = (img * 0.5 + 0.5).clamp(0, 1)               # [0, 1]
        mean = self.enc_mean.to(img.device, img.dtype).view(1, 3, 1, 1)
        std  = self.enc_std .to(img.device, img.dtype).view(1, 3, 1, 1)
        return (img - mean) / std

    @torch.no_grad()
    def encode_cond(self, cond_img: torch.Tensor):
        """
        cond_img: (B, 3, H, W) in [-1, 1]
        returns:  cond_pooled (B, D), patches (B, N, D)

        Each backbone's encode() reports its own pooled vector alongside the dense
        tokens (see _CondEncoder); pooled_cond_source="mean" overrides it with the
        patch-token mean.
        """
        x = self._normalize_for_encoder(cond_img)
        if x.shape[-2] != self.enc_img_size or x.shape[-1] != self.enc_img_size:
            x = F.interpolate(x, size=self.enc_img_size, mode='bilinear', align_corners=False)
        pooled, patches = self.encoder.encode(x)
        if self.pooled_cond_source == "mean":
            pooled = patches.mean(dim=1)
        return pooled, patches

    def get_cond_emb(self, cond_pooled: torch.Tensor, drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond = self.cond_proj(cond_pooled)
        if drop_mask is not None and drop_mask.any():
            null = self.null_cond.expand(cond.size(0), -1)
            cond = torch.where(drop_mask.unsqueeze(-1), null, cond)
        return cond

    def get_patch_tokens(self, patches: torch.Tensor, drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if drop_mask is not None and drop_mask.any():
            null = self.null_patch_tokens.expand(patches.size(0), -1, -1)
            patches = torch.where(drop_mask.view(-1, 1, 1), null, patches)
        return patches

    def _sample_t(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x: torch.Tensor, cond_img: torch.Tensor) -> torch.Tensor:
        """
        x:        (B, 3, H, W) clean image in [-1, 1]
        cond_img: (B, 3, H, W) conditioning image in [-1, 1]
        returns:  scalar flow-matching loss
        """
        cond_pooled, patches = self.encode_cond(cond_img)

        drop_mask = torch.rand(x.size(0), device=x.device) < self.cond_drop_prob
        if self.conditioning_mode in ("cls_only", "cls_cross"):
            cond_emb = self.get_cond_emb(cond_pooled, drop_mask)
            patch_tokens = None
        else:
            cond_emb = None
            patch_tokens = self.get_patch_tokens(patches, drop_mask)

        t = self._sample_t(x.size(0), x.device).view(-1, 1, 1, 1)
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), cond_emb, patch_tokens)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        return ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()

    @torch.no_grad()
    def generate(self, cond_img: torch.Tensor) -> torch.Tensor:
        cond_pooled, patches = self.encode_cond(cond_img)
        return self.generate_from_features(cond_pooled, patches)

    @torch.no_grad()
    def generate_from_features(
        self,
        cond_pooled: torch.Tensor,
        patches: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = cond_pooled.device
        B = cond_pooled.size(0)

        if self.conditioning_mode in ("cls_only", "cls_cross"):
            cond_emb = self.get_cond_emb(cond_pooled)
            uncond_emb = self.null_cond.expand(B, -1)
            cond_patches = None
            uncond_patches = None
        else:
            cond_emb = None
            uncond_emb = None
            cond_patches = patches
            uncond_patches = self.null_patch_tokens.expand(B, -1, -1)

        if noise is None:
            z = self.noise_scale * torch.randn(B, self.in_channels, self.img_size, self.img_size, device=device)
        else:
            z = noise.to(device)

        ts = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        ts = ts.view(-1, *([1] * z.ndim)).expand(-1, B, *([1] * (z.ndim - 1)))

        stepper = self._euler_step if self.method == "euler" else self._heun_step

        for i in range(self.steps - 1):
            z = stepper(z, ts[i], ts[i + 1], cond_emb, uncond_emb, cond_patches, uncond_patches)
        z = self._euler_step(z, ts[-2], ts[-1], cond_emb, uncond_emb, cond_patches, uncond_patches)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, cond_emb, uncond_emb, cond_patches, uncond_patches) -> torch.Tensor:
        x_c = self.net(z, t.flatten(), cond_emb, cond_patches)
        v_c = (x_c - z) / (1.0 - t).clamp_min(self.t_eps)

        x_u = self.net(z, t.flatten(), uncond_emb, uncond_patches)
        v_u = (x_u - z) / (1.0 - t).clamp_min(self.t_eps)

        lo, hi = self.cfg_interval
        in_interval = (t < hi) & ((lo == 0) | (t > lo))
        scale = torch.where(in_interval, torch.tensor(self.cfg_scale, device=t.device), torch.ones_like(t))
        return v_u + scale * (v_c - v_u)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, cond_emb, uncond_emb, cond_patches, uncond_patches):
        v = self._forward_sample(z, t, cond_emb, uncond_emb, cond_patches, uncond_patches)
        return z + (t_next - t) * v

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, cond_emb, uncond_emb, cond_patches, uncond_patches):
        v_t = self._forward_sample(z, t, cond_emb, uncond_emb, cond_patches, uncond_patches)
        z_e = z + (t_next - t) * v_t
        v_tn = self._forward_sample(z_e, t_next, cond_emb, uncond_emb, cond_patches, uncond_patches)
        return z + (t_next - t) * 0.5 * (v_t + v_tn)

    def _trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def update_ema(self):
        src = self._trainable_params()
        for t, s in zip(self.ema_params1, src):
            t.detach().mul_(self.ema_decay1).add_(s, alpha=1 - self.ema_decay1)
        for t, s in zip(self.ema_params2, src):
            t.detach().mul_(self.ema_decay2).add_(s, alpha=1 - self.ema_decay2)

    @contextmanager
    def ema_scope(self, which=2):
        """Temporarily swap trainable params with ema_params{which} for eval/sampling,
        restoring the live (raw) weights on exit. Param mutation uses .data.copy_,
        which bypasses autograd, so no explicit no_grad is needed here."""
        ema = self.ema_params2 if which == 2 else self.ema_params1
        assert ema is not None, f"ema_params{which} not initialized yet (on_fit_start hasn't run)"
        params = self._trainable_params()
        with torch.no_grad():
            backup = [p.detach().clone() for p in params]
            for p, e in zip(params, ema):
                p.data.copy_(e)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(params, backup):
                    p.data.copy_(b)
