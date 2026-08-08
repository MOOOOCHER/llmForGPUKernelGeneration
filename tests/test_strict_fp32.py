import torch

from kernelgym.toolkit.validation import validate_code, validate_strict_fp32_code


def test_kernelbench_toolkit_configures_ieee_fp32_once():
    import os

    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert os.environ["TRITON_F32_DEFAULT"] == "ieee"


def test_kernel_validation_rejects_explicit_tf32():
    code = '''
class ModelNew:
    pass

value = tl.dot(a, b, input_precision="tf32")
'''
    valid, error = validate_code(code, "ModelNew")
    assert not valid
    assert "Non-IEEE FP32" in error


def test_reference_validation_is_not_subject_to_candidate_check():
    code = '''
class Model:
    pass

torch.set_float32_matmul_precision("high")
'''
    valid, error = validate_code(code, "Model")
    assert valid, error


def test_default_triton_dot_is_allowed():
    valid, error = validate_strict_fp32_code("value = tl.dot(a, b)")
    assert valid, error
