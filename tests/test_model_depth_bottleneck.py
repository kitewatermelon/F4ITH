import torch

from rct.model import CrossAttnJiT, CrossAttnJiT_S_16, CrossAttnJiT_B_16


def _count_block_calls(model):
    """Wrap every block's forward to count invocations, return the counter list."""
    counts = [0] * len(model.blocks)
    for i, block in enumerate(model.blocks):
        orig_forward = block.forward

        def wrapper(*args, _i=i, _orig=orig_forward, **kwargs):
            counts[_i] += 1
            return _orig(*args, **kwargs)

        block.forward = wrapper
    return counts


def _run_forward(model, hidden_size, cond_dim):
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    cond_emb = torch.randn(2, hidden_size)
    patch_tokens = torch.randn(2, 16, cond_dim)
    return model(x, t, cond_emb, patch_tokens)


def test_odd_depth_runs_every_block_exactly_once():
    """depth=13: encoder(6) + bottleneck(1) + decoder(6) = 13 blocks, all
    must execute exactly once. Before the fix, zip() silently dropped the
    13th block (0 calls) while still allocating it as a live nn.Module."""
    model = CrossAttnJiT("patch_only", input_size=64, patch_size=16, depth=13, hidden_size=32,
                          num_heads=4, bottleneck_dim=16, cond_dim=16,
                          long_skip_connection=True)
    counts = _count_block_calls(model)
    _run_forward(model, hidden_size=32, cond_dim=16)
    assert counts == [1] * 13


def test_even_depth_still_runs_every_block_exactly_once():
    """Regression guard: depth=12 without long_skip_connection (the only
    even-depth configuration still allowed) must keep working identically
    to before this change. Even depth + long_skip_connection is now
    rejected at construction time (see test_even_depth_long_skip_raises)."""
    model = CrossAttnJiT("patch_only", input_size=64, patch_size=16, depth=12, hidden_size=32,
                          num_heads=4, bottleneck_dim=16, cond_dim=16,
                          long_skip_connection=False)
    counts = _count_block_calls(model)
    _run_forward(model, hidden_size=32, cond_dim=16)
    assert counts == [1] * 12


def test_even_depth_long_skip_raises():
    """Even depth has no bottleneck block, so the last encoder block's
    output would double as both the decoder's input and its first skip --
    a degenerate self-concat. Construction must reject this combination."""
    try:
        CrossAttnJiT("patch_only", input_size=64, patch_size=16, depth=12, hidden_size=32,
                      num_heads=4, bottleneck_dim=16, cond_dim=16,
                      long_skip_connection=True)
        assert False, "expected ValueError for even depth + long_skip_connection"
    except ValueError:
        pass


def test_odd_depth_forward_shape():
    model = CrossAttnJiT("patch_only", input_size=64, patch_size=16, depth=13, hidden_size=32,
                          num_heads=4, bottleneck_dim=16, cond_dim=16,
                          long_skip_connection=True)
    x = torch.randn(2, 3, 64, 64)
    out = _run_forward(model, hidden_size=32, cond_dim=16)
    assert out.shape == x.shape


def test_s16_factory_accepts_depth_override():
    model = CrossAttnJiT_S_16(input_size=64, depth=13, long_skip_connection=True, conditioning_mode="patch_only")
    assert model.depth == 13
    assert len(model.blocks) == 13


def test_s16_factory_default_depth_is_13():
    model = CrossAttnJiT_S_16(input_size=64, long_skip_connection=True, conditioning_mode="patch_only")
    assert model.depth == 13


def test_b16_factory_accepts_depth_override():
    model = CrossAttnJiT_B_16(input_size=64, depth=13, long_skip_connection=True, conditioning_mode="patch_only")
    assert model.depth == 13
    assert len(model.blocks) == 13
