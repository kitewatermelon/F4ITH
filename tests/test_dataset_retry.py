import pytest

from rct.dataset import _open_image_with_retry


def test_open_image_with_retry_skips_single_bad_index():
    length = 5
    bad_indices = {2}

    def open_fn(idx):
        if idx in bad_indices:
            raise OSError("corrupt file")
        return f"image-{idx}"

    image, resolved_idx = _open_image_with_retry(open_fn, 2, length)

    assert image == "image-3"
    assert resolved_idx == 3


def test_open_image_with_retry_returns_original_index_when_good():
    length = 5

    def open_fn(idx):
        return f"image-{idx}"

    image, resolved_idx = _open_image_with_retry(open_fn, 1, length)

    assert image == "image-1"
    assert resolved_idx == 1


def test_open_image_with_retry_raises_after_max_retries():
    length = 100

    def open_fn(idx):
        raise OSError("always corrupt")

    with pytest.raises(RuntimeError):
        _open_image_with_retry(open_fn, 0, length, max_retries=10)
