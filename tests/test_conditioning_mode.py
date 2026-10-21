import pytest
import torch

from rct.model import CrossAttnJiT, CrossAttnJiTBlock, CrossAttnJiT_S_16


def _make_model(conditioning_mode, depth=2, hidden_size=64, num_heads=4):
    return CrossAttnJiT(
        conditioning_mode, input_size=64, patch_size=16, depth=depth,
        hidden_size=hidden_size, num_heads=num_heads, bottleneck_dim=32,
        cond_dim=32, long_skip_connection=False,
    )


def test_cls_only_block_has_no_cross_attn():
    block = CrossAttnJiTBlock(64, 4, conditioning_mode="cls_only")
    assert not hasattr(block, "cross_attn")
    assert not hasattr(block, "norm3")


def test_patch_only_block_has_cross_attn():
    block = CrossAttnJiTBlock(64, 4, conditioning_mode="patch_only")
    assert hasattr(block, "cross_attn")
    assert hasattr(block, "norm3")


def test_adaln_modulation_is_six_chunks_in_both_modes():
    for mode in ("cls_only", "patch_only"):
        block = CrossAttnJiTBlock(64, 4, conditioning_mode=mode)
        assert block.adaLN_modulation[-1].out_features == 6 * 64


def test_cls_only_ignores_patch_tokens():
    model = _make_model("cls_only")
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    cond_emb = torch.randn(2, 64)
    patch_tokens_a = torch.randn(2, 16, 32)
    patch_tokens_b = torch.randn(2, 16, 32) * 100.0
    with torch.no_grad():
        out_a = model(x, t, cond_emb, patch_tokens_a)
        out_b = model(x, t, cond_emb, patch_tokens_b)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_patch_only_ignores_cond_emb():
    model = _make_model("patch_only")
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    cond_emb_a = torch.randn(2, 64)
    cond_emb_b = torch.randn(2, 64) * 100.0
    patch_tokens = torch.randn(2, 16, 32)
    with torch.no_grad():
        out_a = model(x, t, cond_emb_a, patch_tokens)
        out_b = model(x, t, cond_emb_b, patch_tokens)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_patch_only_cross_attn_out_proj_is_zero_init():
    model = _make_model("patch_only")
    for block in model.blocks:
        assert torch.all(block.cross_attn.out_proj.weight == 0)
        assert torch.all(block.cross_attn.out_proj.bias == 0)


def test_cls_only_forward_shape():
    model = _make_model("cls_only")
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    cond_emb = torch.randn(2, 64)
    patch_tokens = torch.randn(2, 16, 32)
    out = model(x, t, cond_emb, patch_tokens)
    assert out.shape == x.shape


def test_patch_only_forward_shape():
    model = _make_model("patch_only")
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    cond_emb = torch.randn(2, 64)
    patch_tokens = torch.randn(2, 16, 32)
    out = model(x, t, cond_emb, patch_tokens)
    assert out.shape == x.shape


def test_factory_requires_conditioning_mode():
    with pytest.raises(TypeError):
        CrossAttnJiT_S_16(input_size=64)


def test_factory_accepts_conditioning_mode():
    model = CrossAttnJiT_S_16(input_size=64, conditioning_mode="cls_only")
    assert model.conditioning_mode == "cls_only"
