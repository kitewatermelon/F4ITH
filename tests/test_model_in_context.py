import pytest
import torch

from rct.model import CrossAttnJiT


# Small config: 64px / patch16 = 4x4 = 16 patch tokens, odd depth for long skip.
COMMON = dict(input_size=64, patch_size=16, depth=5, hidden_size=64,
              num_heads=4, bottleneck_dim=32, cond_dim=32)


def _cls_inputs(b=2):
    return torch.randn(b, 3, 64, 64), torch.rand(b), torch.randn(b, 64), None


@pytest.mark.parametrize("t_embed_mode", ["adaln", "adaln_single"])
@pytest.mark.parametrize("long_skip", [True, False])
def test_in_context_forward_shape_and_finite(t_embed_mode, long_skip):
    model = CrossAttnJiT("cls_only", t_embed_mode=t_embed_mode, long_skip_connection=long_skip,
                         in_context_len=4, in_context_start=1, **COMMON)
    x, t, cond_emb, patch_tokens = _cls_inputs()
    out = model(x, t, cond_emb, patch_tokens)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_in_context_posemb_created_only_when_active():
    active = CrossAttnJiT("cls_only", in_context_len=4, in_context_start=1, **COMMON)
    assert hasattr(active, "in_context_posemb")
    assert active.in_context_posemb.shape == (1, 4, 64)
    assert hasattr(active, "feat_rope_incontext")

    off = CrossAttnJiT("cls_only", in_context_len=0, **COMMON)
    assert not hasattr(off, "in_context_posemb")
    assert not hasattr(off, "feat_rope_incontext")


def test_in_context_zero_len_is_param_identical_to_baseline():
    # in_context_len=0 must not add any parameters vs a plain cls_only model.
    base = CrossAttnJiT("cls_only", long_skip_connection=True, **COMMON)
    off = CrossAttnJiT("cls_only", long_skip_connection=True, in_context_len=0, **COMMON)
    base_keys = {k for k, _ in base.named_parameters()}
    off_keys = {k for k, _ in off.named_parameters()}
    assert base_keys == off_keys


def test_in_context_start_must_be_in_encoder_with_long_skip():
    # depth=5 -> depth//2 = 2, so in_context_start must be < 2.
    with pytest.raises(ValueError):
        CrossAttnJiT("cls_only", long_skip_connection=True,
                     in_context_len=4, in_context_start=2, **COMMON)


def test_in_context_rejected_with_token_mode():
    with pytest.raises(ValueError):
        CrossAttnJiT("cls_only", t_embed_mode="token",
                     in_context_len=4, in_context_start=1, **COMMON)


def test_in_context_forced_off_for_non_cls_only():
    m = CrossAttnJiT("patch_only", in_context_len=4, in_context_start=1, **COMMON)
    assert m.in_context_len == 0
    assert not hasattr(m, "in_context_posemb")


# --- cls_cross conditioning mode (ablation arm 2) ---

@pytest.mark.parametrize("t_embed_mode", ["adaln", "token"])
@pytest.mark.parametrize("long_skip", [True, False])
def test_cls_cross_forward_shape_and_finite(t_embed_mode, long_skip):
    model = CrossAttnJiT("cls_cross", t_embed_mode=t_embed_mode,
                         long_skip_connection=long_skip, **COMMON)
    x, t, cond_emb, patch_tokens = _cls_inputs()
    out = model(x, t, cond_emb, patch_tokens)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_cls_cross_blocks_have_cross_attention():
    model = CrossAttnJiT("cls_cross", t_embed_mode="token", **COMMON)
    for block in model.blocks:
        assert hasattr(block, "cross_attn")
        # zero-init output projection: sublayer starts as a no-op
        assert torch.all(block.cross_attn.out_proj.weight == 0)


def test_cls_cross_null_cond_forward_is_safe():
    # CFG unconditional path passes an all-null pooled embedding.
    model = CrossAttnJiT("cls_cross", t_embed_mode="token", long_skip_connection=True, **COMMON)
    out = model(torch.randn(2, 3, 64, 64), torch.rand(2), torch.zeros(2, 64), None)
    assert out.shape == (2, 3, 64, 64)
    assert torch.isfinite(out).all()
