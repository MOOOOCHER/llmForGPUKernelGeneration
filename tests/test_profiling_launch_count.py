from kernelgym.toolkit.kernelbench.profiling import annotate_launch_count


def test_launch_count_uses_event_counts_not_distinct_kernel_names():
    metrics = {
        "kernel_count": 2,
        "kernels": [
            {"name": "same_kernel", "count": 20},
            {"name": "memcpy", "count": 10},
        ],
    }
    result = annotate_launch_count(metrics, 10)
    assert result["kernel_count"] == 2
    assert result["kernel_launch_count_total"] == 30
    assert result["kernel_launch_count"] == 3
    assert result["profiling_num_trials"] == 10
