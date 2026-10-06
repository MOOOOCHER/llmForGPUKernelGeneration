from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class Operator:
    name: str
    category: str
    in_count: int
    input_ranks: Sequence[Sequence[int]]
    output_rank_rule: str
    emitter: str
    type: str = "compute"  # compute | create
    attributes: Dict[str, Any] = field(default_factory=dict)


# input_ranks: one list per input, containing allowed ranks.
OPERATORS: Dict[str, Operator] = {
    # Creation operators.
    "randn": Operator("randn", "OpCreate", 0, [], "create", "randn", "create"),
    "ones": Operator("ones", "OpCreate", 0, [], "create", "ones", "create"),
    "zeros": Operator("zeros", "OpCreate", 0, [], "create", "zeros", "create"),
    "positive_rand": Operator("positive_rand", "OpCreate", 0, [], "create", "positive_rand", "create"),

    # Shape-preserving unary pointwise.
    "relu": Operator("relu", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "relu"),
    "leaky_relu": Operator("leaky_relu", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "leaky_relu"),
    "sigmoid": Operator("sigmoid", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "sigmoid"),
    "tanh": Operator("tanh", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "tanh"),
    "silu": Operator("silu", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "silu"),
    "gelu": Operator("gelu", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "gelu"),
    "elu": Operator("elu", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "elu"),
    "selu": Operator("selu", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "selu"),
    "softplus": Operator("softplus", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "softplus"),
    "softsign": Operator("softsign", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "softsign"),
    "hardsigmoid": Operator("hardsigmoid", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "hardsigmoid"),
    "hardtanh": Operator("hardtanh", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "hardtanh"),
    "logsigmoid": Operator("logsigmoid", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "logsigmoid"),
    "clamp": Operator("clamp", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "clamp"),
    "abs": Operator("abs", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "abs"),
    "sin": Operator("sin", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "sin"),
    "cos": Operator("cos", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "cos"),
    "exp2": Operator("exp2", "UnaryPointwise", 1, [[1, 2, 3, 4, 5]], "same", "exp2"),

    # Exact-shape binary pointwise. Broadcasting can be added later.
    "add": Operator("add", "BinaryPointwise", 2, [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], "same", "add"),
    "sub": Operator("sub", "BinaryPointwise", 2, [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], "same", "sub"),
    "mul": Operator("mul", "BinaryPointwise", 2, [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], "same", "mul"),
    "div": Operator("div", "BinaryPointwise", 2, [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], "same", "div"),
    "maximum": Operator("maximum", "BinaryPointwise", 2, [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], "same", "maximum"),
    "minimum": Operator("minimum", "BinaryPointwise", 2, [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], "same", "minimum"),

    # Reductions. dim and keepdim are node attributes.
    "sum": Operator("sum", "Reduction", 1, [[2, 3, 4, 5]], "reduction", "sum"),
    "mean": Operator("mean", "Reduction", 1, [[2, 3, 4, 5]], "reduction", "mean"),
    "max": Operator("max", "Reduction", 1, [[2, 3, 4, 5]], "reduction", "max_values"),
    "min": Operator("min", "Reduction", 1, [[2, 3, 4, 5]], "reduction", "min_values"),
    "var": Operator("var", "Reduction", 1, [[2, 3, 4, 5]], "reduction", "var"),
    "norm": Operator("norm", "Reduction", 1, [[2, 3, 4, 5]], "reduction", "norm"),

    # Softmax / cumulative.
    "softmax": Operator("softmax", "Softmax", 1, [[2, 3, 4, 5]], "same", "softmax"),
    "log_softmax": Operator("log_softmax", "Softmax", 1, [[2, 3, 4, 5]], "same", "log_softmax"),
    "cumsum": Operator("cumsum", "Cumulative", 1, [[2, 3, 4, 5]], "same", "cumsum"),
    "cummax": Operator("cummax", "Cumulative", 1, [[2, 3, 4, 5]], "same", "cummax_values"),
    "cummin": Operator("cummin", "Cumulative", 1, [[2, 3, 4, 5]], "same", "cummin_values"),

    # Matrix ops.
    "matmul": Operator("matmul", "Matmul", 2, [[2], [2]], "matmul", "matmul"),
    "bmm": Operator("bmm", "BMM", 2, [[3], [3]], "bmm", "bmm"),
    "linear": Operator("linear", "Linear", 1, [[2]], "linear", "linear"),

    # Normalization.
    "layer_norm": Operator("layer_norm", "LayerNorm", 1, [[2, 3, 4, 5]], "same", "layer_norm"),
    "batch_norm": Operator("batch_norm", "BatchNorm", 1, [[2, 3, 4, 5]], "same", "batch_norm"),
    "group_norm": Operator("group_norm", "GroupNorm", 1, [[3, 4, 5]], "same", "group_norm"),
    "instance_norm": Operator("instance_norm", "InstanceNorm", 1, [[3, 4, 5]], "same", "instance_norm"),

    # Pooling.
    "avg_pool1d": Operator("avg_pool1d", "Pool1D", 1, [[3]], "pool1d", "avg_pool1d"),
    "max_pool1d": Operator("max_pool1d", "Pool1D", 1, [[3]], "pool1d", "max_pool1d"),
    "avg_pool2d": Operator("avg_pool2d", "Pool2D", 1, [[4]], "pool2d", "avg_pool2d"),
    "max_pool2d": Operator("max_pool2d", "Pool2D", 1, [[4]], "pool2d", "max_pool2d"),
    "avg_pool3d": Operator("avg_pool3d", "Pool3D", 1, [[5]], "pool3d", "avg_pool3d"),
    "max_pool3d": Operator("max_pool3d", "Pool3D", 1, [[5]], "pool3d", "max_pool3d"),

    # Convolutions are represented as modules and consume one tensor edge.
    "conv1d": Operator("conv1d", "Conv1D", 1, [[3]], "conv1d", "conv1d"),
    "conv2d": Operator("conv2d", "Conv2D", 1, [[4]], "conv2d", "conv2d"),
    "conv3d": Operator("conv3d", "Conv3D", 1, [[5]], "conv3d", "conv3d"),
    "conv_transpose1d": Operator("conv_transpose1d", "ConvTranspose1D", 1, [[3]], "conv_transpose1d", "conv_transpose1d"),
    "conv_transpose2d": Operator("conv_transpose2d", "ConvTranspose2D", 1, [[4]], "conv_transpose2d", "conv_transpose2d"),
    "conv_transpose3d": Operator("conv_transpose3d", "ConvTranspose3D", 1, [[5]], "conv_transpose3d", "conv_transpose3d"),
}


POINTWISE_OPS = [
    name for name, op in OPERATORS.items()
    if op.category in {"UnaryPointwise", "BinaryPointwise"}
]
REDUCTION_OPS = ["sum", "mean", "max", "min", "var", "norm"]

ANCHOR_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "pointwise": {
        "anchors": POINTWISE_OPS,
        "fusion_possibilities": ["chain"],
        "category_type": "pointwise",
    },
    "reduction": {
        "anchors": REDUCTION_OPS,
        "fusion_possibilities": ["chain", "prologue", "epilogue", "both"],
        "category_type": "reduction",
    },
    "normalization": {
        "anchors": ["layer_norm", "batch_norm", "group_norm", "instance_norm"],
        "fusion_possibilities": ["prologue", "epilogue", "both"],
        "category_type": "normalization",
    },
    "softmax": {
        "anchors": ["softmax", "log_softmax"],
        "fusion_possibilities": ["prologue", "epilogue", "both"],
        "category_type": "softmax",
    },
    "matrix": {
        "anchors": ["linear", "matmul", "bmm"],
        "fusion_possibilities": ["prologue", "epilogue", "both"],
        "category_type": "matrix",
    },
    "pooling": {
        "anchors": ["avg_pool1d", "max_pool1d", "avg_pool2d", "max_pool2d", "avg_pool3d", "max_pool3d"],
        "fusion_possibilities": ["prologue", "epilogue", "both"],
        "category_type": "pooling",
    },
    "conv": {
        "anchors": ["conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"],
        "fusion_possibilities": ["prologue", "epilogue", "both"],
        "category_type": "conv",
    },
    "cumulative": {
        "anchors": ["cumsum", "cummax", "cummin"],
        "fusion_possibilities": ["prologue", "epilogue", "both"],
        "category_type": "cumulative",
    },
}

COMPUTE_OPS = [name for name, op in OPERATORS.items() if op.type == "compute"]
CREATE_OPS = [name for name, op in OPERATORS.items() if op.type == "create"]

__all__ = [
    "Operator", "OPERATORS", "POINTWISE_OPS", "REDUCTION_OPS",
    "ANCHOR_CATEGORIES", "COMPUTE_OPS", "CREATE_OPS",
]
