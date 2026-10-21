import torch

from rct.model import CrossAttnJiT


def _build(t_embed_mode, depth=3, long_skip_connection=False):
    return CrossAttnJiT(
        "patch_only", input_size=64, patch_size=16, depth=depth, hidden_size=64,
        num_heads=4, bottleneck_dim=32, cond_dim=32,
        long_skip_connection=long_skip_connection, t_embed_mode=t_embed_mode,
    )


def _forward(model):
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    cond_emb = torch.randn(2, 64)
    patch_tokens = torch.randn(2, 16, 32)
    return model(x, t, cond_emb, patch_tokens), x


def test_adaln_single_block_has_zero_init_scale_shift_table():
    model = _build("adaln_single")
    for block in model.blocks:
        assert block.scale_shift_table.shape == (6, 64)
        assert torch.all(block.scale_shift_table == 0)


def test_adaln_single_shared_mlp_is_zero_init():
    model = _build("adaln_single")
    assert torch.all(model.shared_adaLN[-1].weight == 0)
    assert torch.all(model.shared_adaLN[-1].bias == 0)


def test_adaln_single_forward_shape_unchanged():
    model = _build("adaln_single")
    out, x = _forward(model)
    assert out.shape == x.shape


def test_adaln_single_forward_shape_unchanged_with_long_skip():
    model = _build("adaln_single", depth=3, long_skip_connection=True)
    out, x = _forward(model)
    assert out.shape == x.shape


def test_adaln_single_has_fewer_params_than_adaln():
    adaln_params = sum(p.numel() for p in _build("adaln").parameters())
    adaln_single_params = sum(p.numel() for p in _build("adaln_single").parameters())
    assert adaln_single_params < adaln_params


def test_adaln_single_final_layer_still_has_own_adaln():
    model = _build("adaln_single")
    assert hasattr(model.final_layer, "adaLN_modulation")
    assert model.final_layer.use_adaln is True


def test_invalid_t_embed_mode_still_rejected():
    try:
        _build("not_a_real_mode")
        assert False, "expected AssertionError"
    except AssertionError:
        pass
