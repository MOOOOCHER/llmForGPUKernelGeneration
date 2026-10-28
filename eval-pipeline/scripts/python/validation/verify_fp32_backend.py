#!/usr/bin/env python3
import os

import torch

from kernelgym.toolkit.kernelbench.correctness import get_tolerance_for_precision
from kernelgym.toolkit.validation import validate_code


good = "class ModelNew:\n    pass\n"
bad = (
    "class ModelNew:\n"
    "    pass\n"
    'torch.set_float32_matmul_precision("high")\n'
)

assert validate_code(good, "ModelNew")[0]
valid, message = validate_code(bad, "ModelNew")
assert not valid
assert "Non-IEEE FP32" in message
assert get_tolerance_for_precision(torch.float32) == 1e-4
assert os.environ["TRITON_F32_DEFAULT"] == "ieee"
assert torch.get_float32_matmul_precision() == "highest"
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cudnn.allow_tf32 is False

print("FP32_BACKEND_OK tolerance=1e-4 tf32_rejected=true triton_default=ieee")
