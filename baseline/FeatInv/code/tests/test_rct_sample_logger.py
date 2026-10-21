from rct_sample_logger import RCTSampleLogger


def _cb():
    return RCTSampleLogger(sample_batch={"jpg": None, "txt": [], "hint": None}, every_n_steps=5000)


def test_skips_step_zero():
    assert _cb()._should_run(0, is_global_zero=True) is False


def test_skips_non_zero_rank():
    assert _cb()._should_run(5000, is_global_zero=False) is False


def test_skips_off_cadence():
    assert _cb()._should_run(4999, is_global_zero=True) is False


def test_runs_on_cadence_rank_zero():
    assert _cb()._should_run(5000, is_global_zero=True) is True
    assert _cb()._should_run(10000, is_global_zero=True) is True
