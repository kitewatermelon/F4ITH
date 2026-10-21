"""Feature caches are keyed by encoder, so two backbones never share vectors."""
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "distance_metrics",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "distance_metrics.py"),
)
distance_metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(distance_metrics)


def test_cache_path_is_per_encoder():
    npy_cw, _ = distance_metrics._cache_paths("/run/fid_eval/real", "real", None, "chexworld")
    npy_xrv, _ = distance_metrics._cache_paths("/run/fid_eval/real", "real", None, "xrv_densenet121")
    assert npy_cw != npy_xrv


def test_chexworld_keeps_its_historical_cache_name():
    # Pre-existing caches on disk are named chexworld_feats_<tag>.npy; renaming
    # them would silently re-encode every evaluated run.
    npy, txt = distance_metrics._cache_paths("/run/fid_eval/real", "real", None, "chexworld")
    assert npy.endswith("/run/fid_eval/chexworld_feats_real.npy")
    assert txt.endswith("/run/fid_eval/chexworld_feats_real.txt")


def test_resize_variant_gets_its_own_cache():
    plain, _ = distance_metrics._cache_paths("/run/fid_eval/gen", "gen", None, "chexworld")
    resized, _ = distance_metrics._cache_paths("/run/fid_eval/gen", "gen", 224, "chexworld")
    assert resized.endswith("chexworld_feats_gen_r224.npy") and plain != resized
