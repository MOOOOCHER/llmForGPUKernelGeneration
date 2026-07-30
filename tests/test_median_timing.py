from kernelgym.toolkit.kernelbench.timing import get_timing_stats


def test_timing_stats_save_median_as_runtime_anchor():
    stats = get_timing_stats([1.0, 1.0, 1.0, 100.0])

    assert stats["mean"] == 25.8
    assert stats["median"] == 1.0
    assert stats["runtime_anchor"] == "median"
