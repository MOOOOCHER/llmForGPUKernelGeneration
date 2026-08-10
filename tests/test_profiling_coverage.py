import pytest
import torch.profiler as torch_profiler

from kernelgym.toolkit.kernelbench.profiling import (
    compute_triton_kernel_coverage,
    extract_profiling_metrics,
)


def test_time_coverage_uses_cuda_time_for_both_numerator_and_denominator():
    profiling_result = {
        "kernels": [
            {
                "name": "generated_kernel",
                "cpu_time_us": 100.0,
                "cuda_time_us": 20.0,
            },
            {
                "name": "aten::sum",
                "cpu_time_us": 900.0,
                "cuda_time_us": 30.0,
            },
            {
                "name": "Lazy Function Loading",
                "cpu_time_us": 50.0,
                "cuda_time_us": 10.0,
            },
        ]
    }

    coverage = compute_triton_kernel_coverage(
        ["generated_kernel grid=(1,)", "function grid=(1,)"], profiling_result
    )

    assert coverage["custom_kernel_cuda_time_in_profiling_us"] == pytest.approx(20.0)
    assert coverage["total_kernel_run_time_in_profiling_us"] == pytest.approx(60.0)
    assert (
        coverage["custom_kernel_cuda_time_in_profiling_us"]
        / coverage["total_kernel_run_time_in_profiling_us"]
    ) == pytest.approx(1.0 / 3.0)


def test_profiling_metrics_exclude_cpu_annotations_with_child_cuda_time():
    class Event:
        def __init__(self, key, device_type, cuda_time_us, cpu_time_us):
            self.key = key
            self.device_type = device_type
            self.device_time_total = cuda_time_us
            self.cpu_time_total = cpu_time_us
            self.self_cuda_time_total = 0.0
            self.count = 1

    class Profile:
        def key_averages(self):
            return [
                Event("aten::sum", torch_profiler.DeviceType.CPU, 20.0, 100.0),
                Event("generated_kernel", torch_profiler.DeviceType.CUDA, 20.0, 0.0),
            ]

    metrics = extract_profiling_metrics(Profile())

    assert metrics["kernel_count"] == 1
    assert metrics["cuda_device_event_count"] == 1
    assert metrics["total_cuda_time_us"] == pytest.approx(20.0)
    assert metrics["kernels"] == [
        {
            "name": "generated_kernel",
            "cuda_time_us": 20.0,
            "cpu_time_us": 0.0,
            "count": 1,
        }
    ]
