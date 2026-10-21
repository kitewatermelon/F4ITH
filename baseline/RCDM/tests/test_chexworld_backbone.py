import torch

from guided_diffusion_rcdm.get_ssl_models import get_model


def test_chexworld_model_output_shape_at_128():
    model = get_model("chexworld")
    model.eval()
    x = torch.randn(2, 3, 128, 128).clamp(-1, 1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 768)


def test_chexworld_model_output_shape_at_224():
    """Confirms the same loaded model handles its native pretrain
    resolution too (interpolate_pos_encoding is a no-op here, N==npatch)."""
    model = get_model("chexworld")
    model.eval()
    x = torch.randn(2, 3, 224, 224).clamp(-1, 1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 768)


def test_chexworld_ssl_dim_matches_image_train_zero_input_pattern():
    """image_train.py's main() does exactly this to auto-detect ssl_dim."""
    model = get_model("chexworld")
    dim = model(torch.zeros(1, 3, 128, 128)).size(1)
    assert dim == 768


def test_chexworld_pool_false_returns_spatial_tokens_at_224():
    model = get_model("chexworld", pool=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224).clamp(-1, 1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 196, 768)   # 14x14 patch grid, no CLS token


def test_chexworld_pool_true_default_still_pooled():
    model = get_model("chexworld")   # default pool=True
    model.eval()
    x = torch.randn(2, 3, 128, 128).clamp(-1, 1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 768)
