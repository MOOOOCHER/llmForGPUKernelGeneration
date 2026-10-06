from __future__ import annotations

import ast
import json
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence

from .drkernel_format_converter import DrKernelRLFormat
from .emitter import PyTorchEmitter
from .generator import BalancedCellScheduler, GraphGenerator, STAGES
from .shape_solver import ShapeSolver
from .runtime_validation import validate_generated_program


class DatasetGenerator:
    def __init__(self, stage: str, seed: int = 42, data_source: str = "balanced_csp_dag") -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage}")
        self.stage = stage
        self.seed = seed
        self.data_source = data_source
        self.graph_generator = GraphGenerator(seed)
        self.emitter = PyTorchEmitter()

    def generate(
        self,
        total_samples: int,
        max_attempts_per_sample: int = 50,
        *,
        initial_samples: Optional[Sequence[Dict[str, Any]]] = None,
        on_sample_accepted: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        runtime_validate: bool = False,
        validation_device: str = "auto",
        validation_timeout_seconds: float = 20.0,
        validation_repeats: int = 2,
        check_intermediates: bool = True,
    ) -> List[Dict[str, Any]]:
        scheduler = BalancedCellScheduler(self.stage, total_samples, self.seed)
        samples: List[Dict[str, Any]] = list(initial_samples or [])
        if len(samples) > total_samples:
            raise ValueError(
                f"checkpoint has {len(samples)} samples, more than requested {total_samples}"
            )
        for _ in range(len(samples)):
            resumed_cell = scheduler.next_cell()
            if resumed_cell is None:
                raise ValueError("checkpoint is longer than the generated schedule")
            scheduler.accept(resumed_cell)
        config = STAGES[self.stage]

        while len(samples) < total_samples:
            cell = scheduler.next_cell()
            if cell is None:
                break
            accepted = False
            validation_rejections: Counter[str] = Counter()
            attempts_for_cell = max_attempts_per_sample * (2 if runtime_validate else 1)
            for attempt in range(attempts_for_cell):
                csp_seed = self.seed + len(samples) * 1000 + attempt
                try:
                    graph, target = self.graph_generator.generate(cell, config.allow_branches)
                    if cell.structured and config.structured_full_program:
                        target_nodes = set(target.get("all_nodes", []))
                        if target_nodes != set(graph.nodes):
                            raise ValueError(
                                "structured Stage 1 fusion target must cover every program op"
                            )
                    if not self.graph_generator.validate_output_connectivity(graph):
                        continue
                    if not self.graph_generator.validate_numerical_safety(graph):
                        continue
                    shapes = ShapeSolver(
                        seed=csp_seed,
                        workload_bucket=cell.workload_bucket,
                        stage=self.stage,
                    ).solve(graph)
                except (ValueError, IndexError):
                    # Randomized graph/operator choices can occasionally violate
                    # a rank contract. They are rejected without consuming the
                    # scheduled quota cell.
                    continue
                if shapes is None:
                    continue
                code = self.emitter.emit(graph, shapes)
                try:
                    ast.parse(code)
                except SyntaxError:
                    continue
                if runtime_validate:
                    validation = validate_generated_program(
                        code,
                        device=validation_device,
                        timeout_seconds=validation_timeout_seconds,
                        repeats=validation_repeats,
                        check_intermediates=check_intermediates,
                    )
                    if not validation.valid:
                        reason = validation.reason.strip().splitlines()[0] if validation.reason else "unknown runtime-validation failure"
                        validation_rejections[reason] += 1
                        continue
                operators = [n.op_name for n in graph.topological_nodes()]
                fusion_ids = set(target.get("all_nodes", []))
                fusion_ops = [n.op_name for n in graph.topological_nodes() if n.node_id in fusion_ids]
                anchor_op = target.get("anchor_op", "none")
                program = {
                    "anchor": anchor_op,
                    "anchor_category": target.get("anchor_family", "mixed"),
                    "anchor_category_type": target.get("anchor_family", "mixed"),
                    "fusion_type": target.get("pattern", "none"),
                    "chain": fusion_ops,
                    "operators": operators,
                    "op_count": len(operators),
                    "program_op_count": len(operators),
                    "fusion_op_count": len(fusion_ids),
                    "context_op_count": len(operators) - len(fusion_ids),
                    "fusion_target": target,
                    "program": graph.to_dict(),
                    "solved_shapes": shapes,
                    "csp_seed": csp_seed,
                    "csp_status": "feasible",
                    "workload_bucket": cell.workload_bucket,
                    "pytorch_code": code,
                    "description": (
                        f"{self.stage}: {target.get('anchor_family')}/{target.get('pattern')}"
                        if cell.structured else f"{self.stage}: random_dag"
                    ),
                    "is_structured": cell.structured,
                    # Construction metadata must not become a fusion instruction.
                    "include_fusion_plan": False,
                    "curriculum_source": ("stage2_recipe_single" if target.get("pattern") == "stage2_recipe_single" else ("stage2_long_recipe" if self.stage == "stage2" else "stage1_base")),
                    "recipe": target.get("recipe", []),
                }
                samples.append(DrKernelRLFormat.generate_sample(program, len(samples), self.data_source))
                scheduler.accept(cell)
                if on_sample_accepted is not None:
                    on_sample_accepted(samples)
                accepted = True
                break
            if not accepted:
                details = ""
                if validation_rejections:
                    most_common = "; ".join(
                        f"{count}x {reason}"
                        for reason, count in validation_rejections.most_common(5)
                    )
                    details = f". Runtime-validation rejections: {most_common}"
                raise RuntimeError(
                    f"could not generate a valid sample for cell {cell} "
                    f"after {attempts_for_cell} attempts{details}"
                )
        return samples


def statistics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    extras = [s["extra_info"] for s in samples]
    return {
        "total": len(samples),
        "by_program_op_count": dict(Counter(x["program_op_count"] for x in extras)),
        "by_structured": dict(Counter(str(x["is_structured"]) for x in extras)),
        "by_anchor_family": dict(Counter(x["fusion_anchor_category"] for x in extras)),
        "by_fusion_type": dict(Counter(x["fusion_type"] for x in extras)),
        "by_curriculum_source": dict(Counter(x.get("curriculum_source", "unknown") for x in extras)),
        "by_workload_bucket": dict(Counter(x.get("workload_bucket", "unknown") for x in extras)),
    }
