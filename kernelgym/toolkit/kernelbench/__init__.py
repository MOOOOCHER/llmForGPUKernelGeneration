"""KernelBench toolkit package and process-wide FP32 defaults.

Keep this module import-light: `kernelgym.toolkit.kernelbench.*` is imported from
low-level modules (e.g. schema/result), so importing heavy modules here can
create circular imports.
"""

import os

import torch


# KernelGym treats generated kernels as cooperative: source validation rejects
# the common explicit TF32 opt-ins, while these defaults are established once
# before Triton kernels are imported or compiled.
os.environ["TRITON_F32_DEFAULT"] = "ieee"
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

__all__ = []
