import os

import torch

import pytest

from kernelbench.precision import (
    enforce_strict_fp32,
    sanitize_strict_fp32_source,
    verify_strict_fp32,
)


def test_enforce_strict_fp32_restores_process_global_state():
    os.environ["TRITON_F32_DEFAULT"] = "tf32"
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    metadata = {}
    state = enforce_strict_fp32(metadata)

    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert os.environ["TRITON_F32_DEFAULT"] == "ieee"
    assert state == metadata
    assert state["fp32_math_mode"] == "ieee"


def test_verify_strict_fp32_rejects_runtime_mutation_and_restores_state():
    enforce_strict_fp32()
    torch.set_float32_matmul_precision("high")

    with pytest.raises(RuntimeError, match="changed evaluator-owned"):
        verify_strict_fp32()

    assert torch.get_float32_matmul_precision() == "highest"


def test_sanitize_strict_fp32_source_rewrites_explicit_opt_ins():
    source = """
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["TRITON_F32_DEFAULT"] = "tf32"
x = tl.dot(a, b, input_precision="tf32x3")
y = tl.dot(c, d, allow_tf32=True)
"""
    rewritten, applied = sanitize_strict_fp32_source(source)

    assert 'set_float32_matmul_precision("highest")' in rewritten
    assert rewritten.count("allow_tf32 = False") == 2
    assert 'TRITON_F32_DEFAULT"] = "ieee"' in rewritten
    assert 'input_precision="ieee"' in rewritten
    assert "allow_tf32=False" in rewritten
    assert len(applied) == 5


def test_sanitize_strict_fp32_source_leaves_default_dot_unchanged():
    source = "x = tl.dot(a, b)"
    rewritten, applied = sanitize_strict_fp32_source(source)
    assert rewritten == source
    assert applied == []


def test_sanitize_strict_fp32_source_rewrites_fp16_and_bf16_casts():
    source = """
a = x.to(torch.float16)
b = x.to(dtype=torch.bfloat16)
c = tl.load(ptr).to(tl.bfloat16)
d = tl.astype(c, tl.float16)
e = torch.empty(4, dtype=torch.half)
f = x.half()
"""
    rewritten, applied = sanitize_strict_fp32_source(source)

    assert "float16" not in rewritten
    assert "bfloat16" not in rewritten
    assert "torch.half" not in rewritten
    assert ".half()" not in rewritten
    assert rewritten.count("torch.float32") == 4
    assert rewritten.count("tl.float32") == 2
    assert len(applied) == 5
