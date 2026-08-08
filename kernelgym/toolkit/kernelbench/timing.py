"""KernelBench timing helpers (toolkit layer)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from kernelgym.toolkit.kernelbench.profiling import (
    extract_profiling_metrics,
    profiling_context,
)
def clear_l2_cache(device: torch.device | int | str = "cuda") -> None:
    """Thrash the GPU L2 cache to match KernelBench's cold-cache protocol."""
    # Keep this byte-for-byte equivalent in size and behavior to
    # KernelBench/src/kernelbench/timing.py::clear_l2_cache:
    # 32 * 1024 * 1024 int64 values = 256 MiB, larger than H100's L2.
    dummy = torch.empty(
        (32, 1024, 1024),
        dtype=torch.int64,
        device=device,
    )
    dummy.fill_(42)
    del dummy


def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1,
    verbose: bool = True,
    device: torch.device = None,
    enable_profiling: bool = False,
) -> Tuple[List[float], Dict[str, Any]]:
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)
    torch.cuda.empty_cache()

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    for trial in range(num_trials + discard_first):
        torch.cuda.synchronize(device=device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        clear_l2_cache(device=device)

        start_event.record()
        kernel_fn(*args)
        end_event.record()

        torch.cuda.synchronize(device=device)

        if trial >= discard_first:
            elapsed_time_ms = start_event.elapsed_time(end_event)
            if verbose:
                logical_idx = trial - discard_first + 1
                print(f"Trial {logical_idx}: {elapsed_time_ms:.3g} ms")
            elapsed_times.append(elapsed_time_ms)

    profiling_metrics: Dict[str, Any] = {}
    if enable_profiling:
        try:
            torch.cuda.synchronize(device=device)

            num_profiling_trials = min(10, num_trials)
            print(
                f"[Profiling] Running {num_profiling_trials} additional iterations for profiling..."
            )

            with profiling_context(True) as prof:
                for _ in range(num_profiling_trials):
                    kernel_fn(*args)
                torch.cuda.synchronize(device=device)

            profiling_metrics = extract_profiling_metrics(prof)
            if profiling_metrics:
                print(
                    f"[Profiling] Captured {profiling_metrics.get('kernel_count', 0)} CUDA kernels"
                )
                print(
                    f"[Profiling] Total CUDA time: {profiling_metrics.get('total_cuda_time_us', 0):.2f} us"
                )

        except Exception as e:
            print(f"[Profiling] Warning: Profiling failed: {e}")
            profiling_metrics = {"profiling_error": str(e)}

    return elapsed_times, profiling_metrics


def run_profiling_only(
    kernel_fn: callable,
    *args,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> Dict[str, Any]:
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    profiling_metrics: Dict[str, Any] = {}
    try:
        torch.cuda.synchronize(device=device)
        print(f"[Profiling] Running {num_trials} iterations (profiling-only)...")
        with profiling_context(True) as prof:
            for _ in range(num_trials):
                kernel_fn(*args)
            torch.cuda.synchronize(device=device)
        profiling_metrics = extract_profiling_metrics(prof)
        if profiling_metrics:
            print(
                f"[Profiling] Captured {profiling_metrics.get('kernel_count', 0)} CUDA kernels"
            )
    except Exception as e:
        print(f"[Profiling] Warning: Profiling-only failed: {e}")
        profiling_metrics = {"profiling_error": str(e)}

    return profiling_metrics


def get_timing_stats(elapsed_times: List[float], device: torch.device = None) -> dict:
    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "median": float(f"{np.median(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
        "runtime_anchor": "median",
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)

    return stats
