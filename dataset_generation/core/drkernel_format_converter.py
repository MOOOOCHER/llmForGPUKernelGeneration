from __future__ import annotations

import uuid
from typing import Any, Dict, List

_PROMPT_PREFIX = """You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities, or algorithmic changes.

You are given the following architecture:
"""

_PROMPT_SUFFIX = """

Note: The kernels should be optimized for FP32 (32-bit floating point) precision.

Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!"""


_LIBRARY_AWARE_FAMILIES = {"convolution", "matrix"}


def _render_fusion_region(program: Dict[str, Any]) -> str:
    if not bool(program.get("include_fusion_plan", program.get("is_structured", True))):
        return ""

    chain = [str(op) for op in _list(program.get("chain")) if str(op).strip()]
    if len(chain) < 2:
        return ""

    anchor_family = str(
        program.get("anchor_category")
        or program.get("fusion_target", {}).get("anchor_family", "")
    ).lower()
    library_aware = anchor_family in _LIBRARY_AWARE_FAMILIES

    lines = [
        "",
        "The following ordered operator region is a candidate fusion opportunity:",
        "",
        f"  {' -> '.join(chain)}",
        "",
        "Evaluate these boundaries, but fuse only the subset that gives a clear end-to-end speedup while preserving exact numerical correctness.",
        "",
        "A sequence of operators counts as fused only when they are computed within the same executed Triton/CUDA kernel invocation, without materializing intermediate results for consumption by a later kernel. Kernel names, comments, wrappers, and helper-function names do not demonstrate fusion.",
        "",
    ]

    if library_aware:
        lines.append(
            "This region contains convolution or matrix multiplication. Retain the optimized library operation by default; replace it only when the concrete shapes and surrounding fusion provide a clear end-to-end benefit. Focus on fusing surrounding operations and epilogues, and do not interpret the listed region as requiring a custom implementation of the expensive library operation."
        )
    else:
        lines.append(
            "For pointwise, normalization, pooling, reduction, softmax, or cumulative operations, fuse profitable adjacent boundaries when legal; leaving an operation separate is acceptable when fusion would be unprofitable or risk correctness."
        )

    return "\n".join(lines) + "\n"


def _list(value: Any) -> List[Any]:
    if value is None: return []
    if isinstance(value, list): return value
    if isinstance(value, tuple): return list(value)
    return [value]


def _bucket(n: int) -> str:
    if n <= 1: return "single"
    if n <= 3: return "short"
    if n <= 5: return "medium"
    return "long"


class DrKernelRLFormat:
    @staticmethod
    def create_prompt(program: Dict[str, Any]) -> str:
        fusion_region = _render_fusion_region(program)
        return (
            f"{_PROMPT_PREFIX}\n{program['pytorch_code'].strip()}"
            f"{fusion_region}{_PROMPT_SUFFIX}"
        )

    @staticmethod
    def generate_sample(program: Dict[str, Any], idx: int, data_source: str = "balanced_csp_dag") -> Dict[str, Any]:
        prompt = DrKernelRLFormat.create_prompt(program)
        target = program.get("fusion_target", {})
        operators = _list(program.get("operators"))
        fusion_count = int(program.get("fusion_op_count", len(target.get("all_nodes", []))))
        total_count = int(program.get("program_op_count", len(operators)))
        chain = _list(program.get("chain"))
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "kernel_optimization",
            "reward_model": {"style": "rule", "ground_truth": program["pytorch_code"]},
            "extra_info": {
                "entry_point": "Model",
                "module_name": "Model",
                "uuid": f"{data_source}_{idx:06d}_{uuid.uuid4().hex[:8]}",
                "ops": str(operators),
                "operators": operators,
                "original_prompt": [{"role": "user", "content": prompt}],
                "fusion_anchor": program.get("anchor", target.get("anchor_op", "none")),
                "fusion_anchor_category": program.get("anchor_category", target.get("anchor_family", "mixed")),
                "fusion_anchor_category_type": program.get("anchor_category_type", target.get("anchor_family", "mixed")),
                "fusion_chain": chain,
                "fusion_chain_length": len(chain),
                "fusion_chain_length_category": _bucket(len(chain)),
                "fusion_type": program.get("fusion_type", target.get("pattern", "none")),
                "fusion_description": program.get("description", ""),
                "total_operators": total_count,
                "program_op_count": total_count,
                "fusion_op_count": fusion_count,
                "context_op_count": int(program.get("context_op_count", max(0, total_count - fusion_count))),
                "is_structured": bool(program.get("is_structured", True)),
                "fusion_target": target,
                "curriculum_source": program.get("curriculum_source", "unknown"),
                "recipe": _list(program.get("recipe")),
                "program_graph": program.get("program", {}),
                "solved_shapes": program.get("solved_shapes", {}),
                "csp_seed": program.get("csp_seed"),
                "csp_status": program.get("csp_status", "unknown"),
                "workload_bucket": program.get("workload_bucket", "unknown"),
            },
        }
