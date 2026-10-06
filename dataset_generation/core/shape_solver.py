from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover
    cp_model = None

from .graph import OperatorNode, ProgramGraph
from .operators import OPERATORS


# KernelBench-inspired semantic domains. These are deliberately broader than
# the original generic domains and include common vision/model dimensions as
# well as irregular sizes used to discourage power-of-two overfitting.
BATCH_DIMS = [1, 2, 4, 8, 16, 32, 64, 128]
ROW_DIMS = [2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192]
CHANNEL_DIMS = [
    1, 3, 4, 6, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256,
    384, 512, 768, 1024, 2048,
]
SPATIAL_DIMS = [
    4, 7, 8, 12, 14, 16, 24, 28, 32, 48, 56, 64, 96, 112,
    128, 192, 224, 256,
]
FEATURE_DIMS = [
    16, 32, 48, 64, 96, 127, 128, 192, 255, 256, 384, 512,
    768, 1000, 1024, 1536, 2048, 3072, 4096,
]
SCAN_DIMS = [
    32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
]
MATMUL_M_DIMS = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
MATMUL_N_DIMS = [1, 16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096, 8192]
MATMUL_K_DIMS = [
    1, 16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096, 8192, 16384, 32768,
]
GENERIC_DIMS = [
    2, 3, 4, 5, 7, 8, 12, 14, 16, 24, 28, 31, 32, 48, 56, 63,
    64, 96, 112, 127, 128, 192, 224, 255, 256, 384, 512, 768,
    1000, 1024, 1536, 2048, 3072, 4096,
]

# Each workload bucket exposes only a bounded subset of the full semantic
# domains. This keeps generated samples affordable while still allowing
# the large bucket to exercise KernelBench-like regimes.
DOMAIN_CAPS = {
    "small": {
        "batch": 32, "row": 1024, "channel": 256, "spatial": 64,
        "feature": 512, "scan": 1024, "mat_m": 512, "mat_n": 512, "mat_k": 1024,
    },
    "medium": {
        "batch": 64, "row": 2048, "channel": 512, "spatial": 128,
        "feature": 1024, "scan": 4096, "mat_m": 1024, "mat_n": 1024, "mat_k": 4096,
    },
    "large": {
        "batch": 128, "row": 4096, "channel": 1024, "spatial": 224,
        "feature": 8192, "scan": 16384, "mat_m": 32768, "mat_n": 8192, "mat_k": 32768,
    },
}



# Hard caps apply to every edge and to the sum of all graph tensors. They make
# workload buckets describe the complete program, not only its anchor.
MAX_EDGE_NUMEL = {
    "small": 65_536,
    "medium": 262_144,
    "large": 16_777_216,
}

MAX_PROGRAM_NUMEL = {
    "small": 262_144,
    "medium": 1_048_576,
    "large": 33_554_432,
}

WORKLOAD_TARGETS = {
    "small": {
        "numel": 4_096, "rows": 64, "width": 32,
        "matrix_out": 1_024, "pool_out": 1_024, "conv_out": 4_096,
    },
    "medium": {
        "numel": 16_384, "rows": 128, "width": 64,
        "matrix_out": 4_096, "pool_out": 4_096, "conv_out": 16_384,
    },
    "large": {
        "numel": 65_536, "rows": 256, "width": 1024,
        "matrix_out": 8_388_608, "pool_out": 16_384, "conv_out": 65_536,
    },
}


@dataclass
class TensorShapeVars:
    dims: List


class ShapeSolver:
    """Edge-based CP-SAT solver with semantic axis domains and workload buckets."""

    def __init__(
        self,
        seed: int = 42,
        timeout_seconds: float = 5.0,
        min_total_numel: int = 256,
        max_total_numel: int = 100_000_000,
        workload_bucket: str = "small",
        stage: Optional[str] = None,
    ) -> None:
        if cp_model is None:
            raise RuntimeError("OR-Tools is required. Install with: pip install ortools")
        if workload_bucket not in WORKLOAD_TARGETS:
            raise ValueError(f"unknown workload bucket: {workload_bucket}")
        self.seed = seed
        self.timeout_seconds = timeout_seconds
        self.min_total_numel = min_total_numel
        self.stage = stage
        # Long Stage 2 chains can legitimately carry a KernelBench-scale matrix
        # through many shape-preserving context edges. Keep this exception local
        # to that regime; Stage 1 and all per-edge limits remain unchanged.
        if stage == "stage1" and workload_bucket == "large":
            max_total_numel = max(max_total_numel, 100_663_296)
        elif stage == "stage2" and workload_bucket == "large":
            max_total_numel = max(max_total_numel, 201_326_592)
        self.max_total_numel = max_total_numel
        self.workload_bucket = workload_bucket
        self.targets = WORKLOAD_TARGETS[workload_bucket]
        self.rng = random.Random(seed)

    def solve(self, graph: ProgramGraph) -> Optional[Dict[str, List[int]]]:
        model = cp_model.CpModel()
        edge_vars: Dict[str, TensorShapeVars] = {}
        semantic_family = self._semantic_family(graph)

        for edge in graph.edges.values():
            domains = self._axis_domains(edge.rank, semantic_family)
            dims = [
                model.NewIntVarFromDomain(cp_model.Domain.FromValues(values), f"{edge.edge_id}_d{i}")
                for i, values in enumerate(domains)
            ]
            edge_vars[edge.edge_id] = TensorShapeVars(dims)

        for node in graph.topological_nodes():
            self._add_node_constraints(model, node, edge_vars)

        self._add_workload_constraints(model, graph, edge_vars)

        edge_cap = MAX_EDGE_NUMEL[self.workload_bucket]
        program_cap = MAX_PROGRAM_NUMEL[self.workload_bucket]
        if self.workload_bucket == "large":
            if self.stage == "stage1":
                program_cap = 100_663_296
            elif self.stage == "stage2":
                program_cap = 201_326_592
        numel_vars = []
        for edge_id, shape in edge_vars.items():
            edge_numel = self._product(model, shape.dims, f"{edge_id}_numel", edge_cap)
            model.Add(edge_numel <= edge_cap)
            numel_vars.append(edge_numel)
        model.Add(sum(numel_vars) >= self.min_total_numel)
        model.Add(sum(numel_vars) <= min(program_cap, self.max_total_numel))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout_seconds
        solver.parameters.random_seed = self.seed
        solver.parameters.randomize_search = True
        solver.parameters.num_search_workers = 1

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        return {
            edge_id: [solver.Value(dim) for dim in shape.dims]
            for edge_id, shape in edge_vars.items()
        }

    def _semantic_family(self, graph: ProgramGraph) -> str:
        target_nodes = [n for n in graph.topological_nodes() if n.is_fusion_target]
        nodes = target_nodes or graph.topological_nodes()
        priority = [
            ("Conv", "conv"), ("Pool", "spatial"), ("BMM", "bmm"),
            ("Matmul", "matrix"), ("Linear", "matrix"),
            ("BatchNorm", "spatial"), ("GroupNorm", "spatial"), ("InstanceNorm", "spatial"),
            ("Cumulative", "scan"), ("Reduction", "reduction"),
            ("Softmax", "reduction"), ("LayerNorm", "reduction"),
        ]
        categories = [OPERATORS[n.op_name].category for n in nodes]
        for prefix, family in priority:
            if any(cat.startswith(prefix) for cat in categories):
                return family
        return "generic"

    def _bounded(self, values: Sequence[int], key: str) -> List[int]:
        cap = DOMAIN_CAPS[self.workload_bucket][key]
        bounded = [value for value in values if value <= cap]
        if not bounded:
            raise ValueError(f"empty semantic domain for {key} in bucket {self.workload_bucket}")
        return bounded

    def _axis_domains(self, rank: int, family: str) -> List[Sequence[int]]:
        batch = self._bounded(BATCH_DIMS, "batch")
        rows = self._bounded(ROW_DIMS, "row")
        channels = self._bounded(CHANNEL_DIMS, "channel")
        spatial = self._bounded(SPATIAL_DIMS, "spatial")
        features = self._bounded(FEATURE_DIMS, "feature")
        scans = self._bounded(SCAN_DIMS, "scan")
        mat_m = self._bounded(MATMUL_M_DIMS, "mat_m")
        mat_n = self._bounded(MATMUL_N_DIMS, "mat_n")
        mat_k = self._bounded(MATMUL_K_DIMS, "mat_k")

        if family in {"conv", "spatial"} and rank >= 3:
            return [batch, channels] + [spatial] * (rank - 2)
        if family == "bmm" and rank == 3:
            matrix_axis = sorted(set(mat_m) | set(mat_n) | set(mat_k))
            return [batch, matrix_axis, matrix_axis]
        if family == "matrix" and rank == 2:
            # Any rank-2 edge can be the left input, right input, or output of
            # a matrix operator. Use the union here; node constraints assign
            # the M/K/N roles and enforce compatibility.
            matrix_axis = sorted(set(mat_m) | set(mat_n) | set(mat_k))
            return [matrix_axis, matrix_axis]
        if family == "scan" and rank >= 2:
            return [rows] * (rank - 1) + [scans]
        if family == "reduction":
            # Reduction chains repeatedly remove the reduced axis. A dimension
            # that begins as a leading/row axis can therefore become the last
            # axis of a lower-rank intermediate. Using a rank-position-only
            # feature domain for every last axis makes valid chains infeasible
            # (for example [2, 4, 8, 32] -> [2, 4, 8] -> [2, 4] -> [2]).
            # Use the union for reduction-family edges; the node workload
            # constraints still enforce meaningful widths on the axes that are
            # actually reduced.
            reduction_axis = sorted(set(rows) | set(features))
            return [reduction_axis] * rank
        if rank == 1:
            return [features]
        if rank == 2:
            return [rows, features]
        return [batch, channels] + [spatial] * (rank - 2)

    def _add_workload_constraints(self, model, graph: ProgramGraph, edge_vars: Dict[str, TensorShapeVars]) -> None:
        target_nodes = [n for n in graph.topological_nodes() if n.is_fusion_target]
        if not target_nodes:
            for edge_id in graph.output_edges:
                out_numel = self._product(model, edge_vars[edge_id].dims, f"{edge_id}_workload", self.max_total_numel)
                model.Add(out_numel >= self.targets["numel"])
            return

        # Constrain the anchor(s), not every pointwise member of a fusion block.
        anchor_nodes = [n for n in target_nodes if n.fusion_role in {"anchor", "chain"}]
        if not anchor_nodes:
            anchor_nodes = target_nodes

        is_reduction_chain = (
            len(anchor_nodes) > 1
            and all(OPERATORS[n.op_name].category == "Reduction" for n in anchor_nodes)
            and all(n.fusion_role == "chain" for n in anchor_nodes)
        )
        followup_width = {"small": 4, "medium": 8, "large": 16}[self.workload_bucket]

        for anchor_index, node in enumerate(anchor_nodes):
            cat = OPERATORS[node.op_name].category
            x = edge_vars[node.input_edges[0]].dims
            out = edge_vars[node.output_edge].dims

            if cat == "Reduction":
                dim = int(node.attributes["dim"]) % len(x)
                rows = self._product(model, [d for i, d in enumerate(x) if i != dim], f"{node.node_id}_rows", self.max_total_numel)
                if is_reduction_chain and anchor_index > 0:
                    # A chain's workload is defined by the first reduction.
                    # Applying the full bucket target to every successive
                    # reduction makes rank-shrinking chains infeasible. Keep
                    # later reductions non-trivial without multiplying the
                    # complete large-bucket row/width requirements.
                    model.Add(x[dim] >= followup_width)
                else:
                    model.Add(x[dim] >= self.targets["width"])
                    model.Add(rows >= self.targets["rows"])
            elif cat in {"Softmax", "Cumulative", "LayerNorm"}:
                dim = int(node.attributes.get("dim", -1)) % len(x)
                rows = self._product(model, [d for i, d in enumerate(x) if i != dim], f"{node.node_id}_rows", self.max_total_numel)
                model.Add(x[dim] >= self.targets["width"])
                model.Add(rows >= self.targets["rows"])
            elif cat in {"BatchNorm", "GroupNorm", "InstanceNorm"}:
                out_numel = self._product(model, out, f"{node.node_id}_out_numel", self.max_total_numel)
                model.Add(x[1] >= 4)
                model.Add(out_numel >= self.targets["numel"])
            elif cat in {"Matmul", "BMM", "Linear"}:
                out_numel = self._product(model, out, f"{node.node_id}_out_numel", self.max_total_numel)
                model.Add(out_numel >= self.targets["matrix_out"])
                if cat == "Matmul":
                    model.Add(x[0] >= (1024 if self.workload_bucket == "large" else 16))
                    model.Add(x[1] >= self.targets["width"])
                elif cat == "BMM":
                    if self.workload_bucket == "large":
                        model.Add(x[0] >= 4)
                        model.Add(x[1] >= 512)
                    else:
                        model.Add(x[1] >= 16)
                    model.Add(x[2] >= self.targets["width"])
            elif cat.startswith("Pool"):
                out_numel = self._product(model, out, f"{node.node_id}_out_numel", self.max_total_numel)
                for spatial in x[2:]:
                    model.Add(spatial >= 8)
                model.Add(out_numel >= self.targets["pool_out"])
            elif cat.startswith("Conv"):
                out_numel = self._product(model, out, f"{node.node_id}_out_numel", self.max_total_numel)
                for spatial in x[2:]:
                    model.Add(spatial >= 8)
                model.Add(out_numel >= self.targets["conv_out"])
            elif cat in {"UnaryPointwise", "BinaryPointwise"}:
                out_numel = self._product(model, out, f"{node.node_id}_out_numel", self.max_total_numel)
                model.Add(out_numel >= self.targets["numel"])

    def _add_node_constraints(self, model, node: OperatorNode, edge_vars: Dict[str, TensorShapeVars]) -> None:
        op = OPERATORS[node.op_name]
        ins = [edge_vars[e].dims for e in node.input_edges]
        out = edge_vars[node.output_edge].dims
        cat = op.category

        if cat in {"UnaryPointwise", "Softmax", "Cumulative", "LayerNorm", "BatchNorm", "GroupNorm", "InstanceNorm"}:
            self._same_shape(model, ins[0], out)
            if cat in {"Softmax", "Cumulative"}:
                dim = int(node.attributes["dim"])
                if not -len(ins[0]) <= dim < len(ins[0]):
                    raise ValueError(f"invalid dim {dim} for {node.node_id}")
            if cat == "GroupNorm":
                groups = int(node.attributes["num_groups"])
                model.AddModuloEquality(0, ins[0][1], groups)
            return

        if cat == "BinaryPointwise":
            if len(ins[0]) != len(ins[1]) or len(out) != len(ins[0]):
                raise ValueError("exact-shape binary operator rank mismatch")
            for a, b, y in zip(ins[0], ins[1], out):
                model.Add(a == b)
                model.Add(y == a)
            return

        if cat == "Reduction":
            x = ins[0]
            dim = int(node.attributes["dim"]) % len(x)
            keepdim = bool(node.attributes.get("keepdim", False))
            if keepdim:
                if len(out) != len(x):
                    raise ValueError("keepdim reduction output rank mismatch")
                for i, (a, y) in enumerate(zip(x, out)):
                    model.Add(y == (1 if i == dim else a))
            else:
                if len(out) != len(x) - 1:
                    raise ValueError("reduction output rank mismatch")
                j = 0
                for i, a in enumerate(x):
                    if i != dim:
                        model.Add(out[j] == a)
                        j += 1
            return

        if cat == "Matmul":
            a, b = ins
            model.Add(a[1] == b[0])
            model.Add(out[0] == a[0])
            model.Add(out[1] == b[1])
            return

        if cat == "BMM":
            a, b = ins
            model.Add(a[0] == b[0])
            model.Add(a[2] == b[1])
            model.Add(out[0] == a[0])
            model.Add(out[1] == a[1])
            model.Add(out[2] == b[2])
            return

        if cat == "Linear":
            x = ins[0]
            model.Add(x[-1] == int(node.attributes["in_features"]))
            model.Add(out[0] == x[0])
            model.Add(out[1] == int(node.attributes["out_features"]))
            return

        if cat.startswith("Pool"):
            x = ins[0]
            kernel = int(node.attributes["kernel_size"])
            stride = int(node.attributes["stride"])
            model.Add(out[0] == x[0])
            model.Add(out[1] == x[1])
            for i in range(2, len(x)):
                model.AddModuloEquality(0, x[i], stride)
                model.Add(out[i] * stride == x[i])
                model.Add(x[i] >= kernel)
            return

        if cat.startswith("Conv"):
            x = ins[0]
            in_channels = int(node.attributes["in_channels"])
            out_channels = int(node.attributes["out_channels"])
            stride = int(node.attributes["stride"])
            transposed = "Transpose" in cat
            if len(x) < 3 or len(out) != len(x):
                raise ValueError(f"{node.op_name} requires matching rank >= 3")
            model.Add(x[1] == in_channels)
            model.Add(out[0] == x[0])
            model.Add(out[1] == out_channels)
            for i in range(2, len(x)):
                if transposed:
                    model.Add(out[i] == x[i] * stride)
                else:
                    model.AddModuloEquality(0, x[i], stride)
                    model.Add(out[i] * stride == x[i])
            return

        raise NotImplementedError(f"No constraints for {node.op_name}/{cat}")

    @staticmethod
    def _same_shape(model, a: List, b: List) -> None:
        if len(a) != len(b):
            raise ValueError("shape-preserving rank mismatch")
        for x, y in zip(a, b):
            model.Add(x == y)

    @staticmethod
    def _product(model, dims: List, name: str, upper: int):
        if not dims:
            constant = model.NewIntVar(1, 1, f"{name}_scalar")
            model.Add(constant == 1)
            return constant
        if len(dims) == 1:
            return dims[0]
        current = dims[0]
        for i, dim in enumerate(dims[1:], start=1):
            nxt = model.NewIntVar(1, upper, f"{name}_partial_{i}")
            model.AddMultiplicationEquality(nxt, [current, dim])
            current = nxt
        return current
