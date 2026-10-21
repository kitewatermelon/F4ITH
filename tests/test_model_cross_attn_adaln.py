import torch

from rct.model import CrossAttnJiT


def test_full_model_adaln_modulation_is_zero_init():
    model = CrossAttnJiT("patch_only", input_size=64, patch_size=16, depth=2, hidden_size=64,
                          num_heads=4, bottleneck_dim=32, cond_dim=32,
                          long_skip_connection=False)
    for block in model.blocks:
        assert torch.all(block.adaLN_modulation[-1].weight == 0)
        assert torch.all(block.adaLN_modulation[-1].bias == 0)


def test_full_model_forward_shape_unchanged():
    model = CrossAttnJiT("patch_only", input_size=64, patch_size=16, depth=2, hidden_size=64,
                          num_heads=4, bottleneck_dim=32, cond_dim=32,
                          long_skip_connection=False)
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    cond_emb = torch.randn(2, 64)
    patch_tokens = torch.randn(2, 16, 32)
    out = model(x, t, cond_emb, patch_tokens)
    assert out.shape == x.shape


def test_full_model_forward_shape_unchanged_long_skip_connection():
    model = CrossAttnJiT("patch_only", input_size=64, patch_size=16, depth=3, hidden_size=64,
                          num_heads=4, bottleneck_dim=32, cond_dim=32,
                          long_skip_connection=True)
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    cond_emb = torch.randn(2, 64)
    patch_tokens = torch.randn(2, 16, 32)
    out = model(x, t, cond_emb, patch_tokens)
    assert out.shape == x.shape
