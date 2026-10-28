"""Evaluator-owned floating-point math-mode controls."""

from __future__ import annotations

import os
import re
from typing import Any, MutableMapping

import torch


_STRICT_FP32_SOURCE_REWRITES = [
    (
        re.compile(
            r"(\.to\s*\(\s*)(torch|tl)\.(?:float16|half|bfloat16)\b",
            re.IGNORECASE,
        ),
        r"\1\2.float32",
        "explicit_lower_precision_to_cast",
    ),
    (
        re.compile(
            r"(\.to\s*\(\s*dtype\s*=\s*)torch\.(?:float16|half|bfloat16)\b",
            re.IGNORECASE,
        ),
        r"\1torch.float32",
        "explicit_lower_precision_to_dtype_cast",
    ),
    (
        re.compile(
            r"(\bdtype\s*=\s*)torch\.(?:float16|half|bfloat16)\b",
            re.IGNORECASE,
        ),
        r"\1torch.float32",
        "explicit_lower_precision_dtype",
    ),
    (
        re.compile(
            r"(tl\.astype\s*\(\s*[^,]+,\s*)tl\.(?:float16|bfloat16)\b",
            re.IGNORECASE,
        ),
        r"\1tl.float32",
        "triton_lower_precision_astype",
    ),
    (
        re.compile(r"\.half\s*\(\s*\)", re.IGNORECASE),
        ".float()",
        "torch_half_method",
    ),
    (
        re.compile(
            r"(set_float32_matmul_precision\s*\(\s*)['\"](?:high|medium)['\"](\s*\))",
            re.IGNORECASE,
        ),
        r'\1"highest"\2',
        "torch_float32_matmul_precision",
    ),
    (
        re.compile(
            r"((?:cuda\.matmul|cudnn)\.allow_tf32\s*=\s*)True\b",
            re.IGNORECASE,
        ),
        r"\1False",
        "torch_backend_allow_tf32",
    ),
    (
        re.compile(
            r"(\binput_precision\s*=\s*)['\"](?:tf32|tf32x3)['\"]",
            re.IGNORECASE,
        ),
        r'\1"ieee"',
        "triton_dot_input_precision",
    ),
    (
        re.compile(r"(\ballow_tf32\s*=\s*)True\b", re.IGNORECASE),
        r"\1False",
        "triton_dot_allow_tf32",
    ),
    (
        re.compile(
            r"((?:os\.)?environ\s*\[\s*['\"]TRITON_F32_DEFAULT['\"]\s*\]\s*=\s*)"
            r"['\"](?:tf32|tf32x3)['\"]",
            re.IGNORECASE,
        ),
        r'\1"ieee"',
        "triton_f32_environment",
    ),
]


def sanitize_strict_fp32_source(source: str) -> tuple[str, list[str]]:
    """Rewrite safe reduced-precision opt-ins to their IEEE FP32 equivalents.

    This handles explicit PyTorch/Triton FP16 and BF16 casts as well as TF32
    math-mode requests. Native CUDA/CUTLASS half operations are intentionally
    not rewritten because token substitution cannot preserve their semantics.
    """
    rewritten = source
    applied: list[str] = []
    for pattern, replacement, label in _STRICT_FP32_SOURCE_REWRITES:
        rewritten, count = pattern.subn(replacement, rewritten)
        if count:
            applied.append(f"{label}:{count}")
    return rewritten, applied


def get_strict_fp32_state() -> dict[str, Any]:
    """Return the process-global settings relevant to IEEE FP32 evaluation."""
    return {
        "fp32_math_mode": "ieee",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "triton_f32_default": os.environ.get("TRITON_F32_DEFAULT"),
    }


def enforce_strict_fp32(
    metadata: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Force IEEE FP32 matmul/convolution semantics for an evaluation process.

    Candidate modules execute arbitrary import-time Python and can mutate
    PyTorch's process-global math mode. Call this before and after importing a
    candidate, and immediately before each correctness or timing phase.

    Triton 3.x defaults FP32 ``tl.dot`` to TF32. ``TRITON_F32_DEFAULT=ieee``
    changes that default; standalone KernelBench sanitizes explicit candidate
    TF32 requests before this state is enforced.
    """
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    state = get_strict_fp32_state()
    if metadata is not None:
        metadata.update(state)
    return state


def verify_strict_fp32() -> None:
    """Fail if candidate execution changed evaluator-owned math-mode state."""
    state = get_strict_fp32_state()
    valid = (
        state["float32_matmul_precision"] == "highest"
        and state["cuda_matmul_allow_tf32"] is False
        and state["cudnn_allow_tf32"] is False
        and state["triton_f32_default"] == "ieee"
    )
    if not valid:
        enforce_strict_fp32()
        raise RuntimeError(
            f"Candidate changed evaluator-owned IEEE FP32 math mode: {state}"
        )
