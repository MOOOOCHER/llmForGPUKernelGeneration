from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .graph import OperatorNode, ProgramGraph
from .operators import ANCHOR_CATEGORIES, COMPUTE_OPS, OPERATORS, POINTWISE_OPS, REDUCTION_OPS


WORKLOAD_BUCKETS = ("small", "medium", "large")

# Stage1 and Stage2 intentionally use the same workload distribution. The
# curriculum difference comes from graph size, structure, and branching.
WORKLOAD_WEIGHTS = {
    # Preserve simple cases while making KernelBench-scale anchors common enough
    # to train strategy selection for library-sensitive heavy operators.
    "stage1": {"small": 0.20, "medium": 0.40, "large": 0.40},
    "stage2": {"small": 0.20, "medium": 0.40, "large": 0.40},
}

# Exponential functions after unbounded accumulations can overflow before a
# later bounded activation hides the invalid intermediate.
UNSAFE_ADJACENCIES = {
    ("cumsum", "exp2"),
    ("cummax", "exp2"),
    ("sum", "exp2"),
}


@dataclass(frozen=True)
class StageConfig:
    program_op_min: int
    program_op_max: int
    fusion_op_min: int
    fusion_op_max: int
    structured_ratio: float
    allow_branches: bool
    structured_full_program: bool = False


STAGES = {
    # Stage 1 is the fully structured short-chain curriculum: every sample
    # receives one scheduled anchor and an explicit full-region plan.
    "stage1": StageConfig(2, 5, 2, 5, 1.00, False, True),
    # Stage 2 uses only recipe-built graphs.  These are structured for
    # generation validity and coverage, but do not inject a fusion plan into
    # the RL prompt.
    "stage2": StageConfig(6, 15, 6, 15, 1.00, True),
}


# Shared by Stage 2 and compact Stage 1 recipe motifs.
STAGE2_RECIPE_WEIGHTS = {
    "matrix": 0.35,
    "conv": 0.25,
    "pooling": 0.15,
    "normalization": 0.10,
    "softmax": 0.10,
    "cumulative": 0.05,
}
STAGE1_BASE_RATIO = 0.60


@dataclass(frozen=True)
class SampleCell:
    structured: bool
    anchor_family: str
    fusion_pattern: str
    program_op_count: int
    fusion_op_count: int
    workload_bucket: str


class BalancedCellScheduler:
    """Precompute an exactly sized, balanced schedule.

    A cell is removed only after ``accept``. Rejected CSP attempts therefore do
    not change the final dataset distribution.
    """

    def __init__(self, stage: str, total_samples: int, seed: int = 42) -> None:
        self.stage = stage
        self.config = STAGES[stage]
        self.total_samples = total_samples
        self.rng = random.Random(seed)
        self.schedule = self._build_schedule()
        self.index = 0

    @staticmethod
    def _balanced_sequence(values: List[Any], count: int, offset: int = 0) -> List[Any]:
        return [values[(i + offset) % len(values)] for i in range(count)]

    def _weighted_workload_sequence(self, count: int, offset: int = 0) -> List[str]:
        """Return an exactly sized deterministic sequence using largest remainders."""
        weights = WORKLOAD_WEIGHTS[self.stage]
        raw = {bucket: count * weights[bucket] for bucket in WORKLOAD_BUCKETS}
        quotas = {bucket: int(raw[bucket]) for bucket in WORKLOAD_BUCKETS}
        remaining = count - sum(quotas.values())
        order = sorted(
            WORKLOAD_BUCKETS,
            key=lambda bucket: (raw[bucket] - quotas[bucket], weights[bucket]),
            reverse=True,
        )
        for bucket in order[:remaining]:
            quotas[bucket] += 1

        # Interleave buckets instead of placing all small samples together.
        pool = []
        for bucket in WORKLOAD_BUCKETS:
            pool.extend([bucket] * quotas[bucket])
        rng = random.Random(self.rng.random() + offset)
        rng.shuffle(pool)
        return pool

    @staticmethod
    def _weighted_sequence(weights: Dict[str, float], count: int) -> List[str]:
        """Return exact largest-remainder quotas for a named weighted pool."""
        raw = {name: count * weight for name, weight in weights.items()}
        quotas = {name: int(value) for name, value in raw.items()}
        remaining = count - sum(quotas.values())
        order = sorted(
            weights,
            key=lambda name: (raw[name] - quotas[name], weights[name]),
            reverse=True,
        )
        for name in order[:remaining]:
            quotas[name] += 1
        return [name for name in weights for _ in range(quotas[name])]

    @staticmethod
    def _weighted_pattern_pool(count: int) -> List[str]:
        """Return a 40/40/20 pool for the three anchored fusion patterns."""
        patterns = ("prologue", "epilogue", "prologue_anchor_epilogue")
        weights = {
            "prologue": 0.40,
            "epilogue": 0.40,
            "prologue_anchor_epilogue": 0.20,
        }
        raw = {pattern: count * weights[pattern] for pattern in patterns}
        quotas = {pattern: int(raw[pattern]) for pattern in patterns}
        remaining = count - sum(quotas.values())
        order = sorted(
            patterns,
            key=lambda pattern: (raw[pattern] - quotas[pattern], weights[pattern]),
            reverse=True,
        )
        for pattern in order[:remaining]:
            quotas[pattern] += 1
        return [pattern for pattern in patterns for _ in range(quotas[pattern])]

    def _build_schedule(self) -> List[SampleCell]:
        structured_total = round(self.total_samples * self.config.structured_ratio)
        random_total = self.total_samples - structured_total
        stage1_recipe_total = 0
        if self.stage == "stage1":
            # 60% base short chains; 40% one unpadded Stage 2 grammar motif.
            stage1_recipe_total = self.total_samples - round(self.total_samples * STAGE1_BASE_RATIO)
            structured_total -= stage1_recipe_total
        families = list(ANCHOR_CATEGORIES)
        op_counts = list(range(self.config.program_op_min, self.config.program_op_max + 1))

        # Compute workload quotas once over the full dataset so Stage1 and
        # Stage2 both realize the requested 30/45/25 distribution exactly
        # whenever the sample count permits it.
        all_workloads = self._weighted_workload_sequence(self.total_samples, offset=2)
        structured_workloads = all_workloads[:structured_total]
        recipe_workloads = all_workloads[structured_total:structured_total + stage1_recipe_total]
        random_workloads = all_workloads[structured_total + stage1_recipe_total:]

        # Stage 2 is fully recipe-built: every long chain receives an exact
        # primary-family quota, but no fusion plan is exposed to the model.
        if self.stage == "stage2":
            family_seq = self._weighted_sequence(STAGE2_RECIPE_WEIGHTS, structured_total)
            self.rng.shuffle(family_seq)
            op_seq = self._balanced_sequence(op_counts, structured_total, offset=1)
            schedule = [
                SampleCell(True, family, "mixed_recipe", n_ops, 0, workload)
                for family, n_ops, workload in zip(family_seq, op_seq, structured_workloads)
            ]
            self.rng.shuffle(schedule)
            return schedule

        # Balance the Cartesian product instead of cycling family and operator
        # count independently. Independent cycles correlated the two axes and
        # prevented some families from ever receiving a legal three-part cell.
        family_op_pairs = [
            (family, n_ops) for family in families for n_ops in op_counts
        ]
        pair_seq = self._balanced_sequence(family_op_pairs, structured_total)
        pattern_by_index: Dict[int, str] = {}
        two_anchor_indices = set()

        # Only four- and five-op Stage 1 cells may contain two scheduled
        # anchors. Select exactly half of each family's eligible cells;
        # two- and three-op cells always retain one anchor.
        if self.stage == "stage1":
            for family in families:
                for n_ops in (4, 5):
                    candidates = [
                        i for i, (candidate, count) in enumerate(pair_seq)
                        # Chained reductions are not reliably feasible in the
                        # existing Stage 1 shape solver; retain their
                        # established single-anchor curriculum instead.
                        if candidate == family and family != "reduction" and count == n_ops
                    ]
                    self.rng.shuffle(candidates)
                    two_anchor_indices.update(candidates[:len(candidates) // 2])
            for i in two_anchor_indices:
                pattern_by_index[i] = "two_anchor_region"

        for family in families:
            family_indices = [
                i for i, (candidate, _) in enumerate(pair_seq) if candidate == family
            ]
            single_anchor_indices = [i for i in family_indices if i not in two_anchor_indices]
            if family == "pointwise":
                for i in single_anchor_indices:
                    pattern_by_index[i] = "chain"
                continue

            anchored_indices = list(single_anchor_indices)
            if family == "reduction":
                # Preserve the historical 25% reduction-chain share. Chain is
                # separate from the 40/40/20 anchored-pattern distribution.
                chain_total = round(len(single_anchor_indices) * 0.25)
                chain_candidates = [i for i in single_anchor_indices if pair_seq[i][1] <= 4]
                self.rng.shuffle(chain_candidates)
                chain_indices = set(chain_candidates[:chain_total])
                for i in chain_indices:
                    pattern_by_index[i] = "chain"
                anchored_indices = [i for i in single_anchor_indices if i not in chain_indices]

            pattern_pool = self._weighted_pattern_pool(len(anchored_indices))
            self.rng.shuffle(pattern_pool)

            # A two-op program cannot contain both a prologue and epilogue.
            # Swap illegal assignments with legal 3+-op assignments while
            # retaining the exact global quotas for the family.
            illegal_positions = [
                position
                for position, (i, pattern) in enumerate(zip(anchored_indices, pattern_pool))
                if pair_seq[i][1] < 3 and pattern == "prologue_anchor_epilogue"
            ]
            swap_positions = [
                position
                for position, (i, pattern) in enumerate(zip(anchored_indices, pattern_pool))
                if pair_seq[i][1] >= 3 and pattern != "prologue_anchor_epilogue"
            ]
            if len(swap_positions) < len(illegal_positions):
                raise ValueError(
                    f"not enough 3+-op {family} cells for the combined fusion quota"
                )
            for illegal, replacement in zip(illegal_positions, swap_positions):
                pattern_pool[illegal], pattern_pool[replacement] = (
                    pattern_pool[replacement], pattern_pool[illegal]
                )
            for i, pattern in zip(anchored_indices, pattern_pool):
                pattern_by_index[i] = pattern

        structured: List[SampleCell] = []
        per_family_seen = {family: 0 for family in families}
        for i, ((family, n_ops), workload) in enumerate(
            zip(pair_seq, structured_workloads)
        ):
            pattern = pattern_by_index[i]
            family_index = per_family_seen[family]
            per_family_seen[family] += 1

            if self.config.structured_full_program:
                f_ops = n_ops
            else:
                max_fusion = min(self.config.fusion_op_max, n_ops)
                min_fusion = (
                    3
                    if pattern in {"both", "prologue_anchor_epilogue"}
                    else self.config.fusion_op_min
                )
                valid_fusion_counts = list(range(min_fusion, max_fusion + 1))
                if family == "reduction" and pattern == "chain":
                    valid_fusion_counts = [v for v in valid_fusion_counts if v <= 4]
                f_ops = valid_fusion_counts[(family_index + i) % len(valid_fusion_counts)]
            structured.append(
                SampleCell(True, family, pattern, n_ops, f_ops, workload)
            )

        if stage1_recipe_total:
            recipe_families = self._weighted_sequence(STAGE2_RECIPE_WEIGHTS, stage1_recipe_total)
            self.rng.shuffle(recipe_families)
            recipe_op_counts = self._balanced_sequence(list(range(3, 6)), stage1_recipe_total, offset=3)
            structured.extend(
                SampleCell(True, family, "stage2_recipe_single", n_ops, n_ops, workload)
                for family, n_ops, workload in zip(recipe_families, recipe_op_counts, recipe_workloads)
            )

        random_cells: List[SampleCell] = []
        random_op_seq = self._balanced_sequence(op_counts, random_total, offset=2)
        for i in range(random_total):
            random_cells.append(
                SampleCell(False, "none", "none", random_op_seq[i], 0, random_workloads[i])
            )

        schedule = structured + random_cells
        self.rng.shuffle(schedule)
        return schedule

    def next_cell(self) -> Optional[SampleCell]:
        return None if self.index >= len(self.schedule) else self.schedule[self.index]

    def accept(self, cell: SampleCell) -> None:
        expected = self.next_cell()
        if expected != cell:
            raise ValueError(f"accepted cell {cell} does not match scheduled cell {expected}")
        self.index += 1


class GraphGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.node_idx = 0
        self.edge_idx = 0

    def generate(self, cell: SampleCell, allow_branches: bool) -> Tuple[ProgramGraph, Dict[str, Any]]:
        self.node_idx = 0
        self.edge_idx = 0
        if cell.structured:
            return self._generate_structured(cell, allow_branches)
        return self._generate_random(cell.program_op_count, allow_branches)

    def _generate_structured(self, cell: SampleCell, allow_branches: bool) -> Tuple[ProgramGraph, Dict[str, Any]]:
        if cell.fusion_pattern == "mixed_recipe":
            return self._generate_stage2_recipe(cell)
        if cell.fusion_pattern == "stage2_recipe_single":
            return self._generate_stage1_recipe_single(cell)
        if cell.fusion_pattern == "two_anchor_region":
            return self._generate_stage1_two_anchor_region(cell)
        graph = ProgramGraph()
        anchor = self.rng.choice(ANCHOR_CATEGORIES[cell.anchor_family]["anchors"])
        fusion_ops, roles = self._fusion_ops(cell.anchor_family, anchor, cell.fusion_pattern, cell.fusion_op_count)
        context_count = cell.program_op_count - len(fusion_ops)

        first_rank = self._required_primary_rank(anchor)
        if cell.anchor_family == "reduction" and cell.fusion_pattern == "chain":
            first_rank = max(first_rank, len(fusion_ops) + 1)
            if first_rank > 5:
                raise ValueError("reduction chain is too long for the supported maximum rank")
        live = self._build_context(graph, context_count, first_rank, allow_branches)
        primary = self._choose_edge_of_rank(graph, live, first_rank)
        if primary is None:
            primary = self._new_input(graph, first_rank)

        fusion_nodes: List[str] = []
        pre_nodes: List[str] = []
        anchor_nodes: List[str] = []
        scheduled_anchor_nodes: List[str] = []
        post_nodes: List[str] = []
        current = primary

        for op_name, role in zip(fusion_ops, roles):
            node, current = self._append_operator(
                graph, op_name, current, role=role, target=True,
                workload_bucket=cell.workload_bucket,
            )
            fusion_nodes.append(node.node_id)
            if role == "pre":
                pre_nodes.append(node.node_id)
            elif role == "anchor":
                anchor_nodes.append(node.node_id)
                scheduled_anchor_nodes.append(node.node_id)
            elif role == "post":
                post_nodes.append(node.node_id)
            else:
                anchor_nodes.append(node.node_id)
                # ``chain`` labels legacy pointwise/reduction regions.  The
                # first node remains the one scheduled anchor for Stage 1.
                if not scheduled_anchor_nodes:
                    scheduled_anchor_nodes.append(node.node_id)

        graph.output_edges = [current]
        target = {
            "anchor_family": cell.anchor_family,
            "anchor_op": anchor,
            "pattern": cell.fusion_pattern,
            "pre_nodes": pre_nodes,
            "anchor_nodes": anchor_nodes,
            "scheduled_anchor_nodes": scheduled_anchor_nodes,
            "post_nodes": post_nodes,
            "all_nodes": fusion_nodes,
        }
        return graph, target

    def _generate_stage1_two_anchor_region(self, cell: SampleCell) -> Tuple[ProgramGraph, Dict[str, Any]]:
        """Build a full four- or five-op target region with two anchors."""
        graph = ProgramGraph()
        first_anchor, second_anchor, first_rank = self._stage1_anchor_pair(cell.anchor_family)
        context_count = cell.program_op_count - 2
        pre = self.rng.randint(0, context_count)
        between = self.rng.randint(0, context_count - pre)
        post = context_count - pre - between
        ops = (
            [self.rng.choice(POINTWISE_OPS) for _ in range(pre)]
            + [first_anchor]
            + [self.rng.choice(POINTWISE_OPS) for _ in range(between)]
            + [second_anchor]
            + [self.rng.choice(POINTWISE_OPS) for _ in range(post)]
        )
        roles = ["context"] * pre + ["anchor"] + ["context"] * between + ["anchor"] + ["context"] * post

        current = self._new_input(graph, first_rank)
        fusion_nodes: List[str] = []
        anchor_nodes: List[str] = []
        last_linear: Optional[OperatorNode] = None
        last_conv: Optional[OperatorNode] = None
        for op_name, role in zip(ops, roles):
            node, current = self._append_operator(
                graph, op_name, current, role=role, target=True,
                workload_bucket=cell.workload_bucket,
            )
            if op_name == "linear":
                if last_linear is not None:
                    node.attributes["in_features"] = last_linear.attributes["out_features"]
                last_linear = node
            if OPERATORS[op_name].category.startswith("Conv"):
                if last_conv is not None:
                    node.attributes["in_channels"] = last_conv.attributes["out_channels"]
                last_conv = node
            fusion_nodes.append(node.node_id)
            if role == "anchor":
                anchor_nodes.append(node.node_id)

        graph.output_edges = [current]
        return graph, {
            "anchor_family": cell.anchor_family,
            "anchor_op": first_anchor,
            "pattern": "two_anchor_region",
            "pre_nodes": [],
            "anchor_nodes": anchor_nodes,
            "scheduled_anchor_nodes": anchor_nodes,
            "post_nodes": [],
            "all_nodes": fusion_nodes,
        }

    def _stage1_anchor_pair(self, family: str) -> Tuple[str, str, int]:
        """Sample compatible (not necessarily distinct) anchors from one region."""
        if family == "matrix":
            if self.rng.random() < 0.25:
                return "bmm", "bmm", 3
            return self.rng.choice(["linear", "matmul"]), self.rng.choice(["linear", "matmul"]), 2
        if family == "conv":
            suffix = self.rng.choice(["1d", "2d", "3d"])
            options = [f"conv{suffix}", f"conv_transpose{suffix}"]
            return self.rng.choice(options), self.rng.choice(options), int(suffix[0]) + 2
        if family == "pooling":
            suffix = self.rng.choice(["1d", "2d", "3d"])
            options = [f"avg_pool{suffix}", f"max_pool{suffix}"]
            return self.rng.choice(options), self.rng.choice(options), int(suffix[0]) + 2
        if family == "normalization":
            rank = self.rng.choice([3, 4, 5])
            options = [op for op in ANCHOR_CATEGORIES[family]["anchors"] if rank in OPERATORS[op].input_ranks[0]]
            return self.rng.choice(options), self.rng.choice(options), rank
        if family == "softmax":
            return self.rng.choice(["softmax", "log_softmax"]), self.rng.choice(["softmax", "log_softmax"]), self.rng.choice([2, 3, 4, 5])
        if family == "reduction":
            return self.rng.choice(REDUCTION_OPS), self.rng.choice(REDUCTION_OPS), 3
        if family == "cumulative":
            ops = ["cumsum", "cummax", "cummin"]
            return self.rng.choice(ops), self.rng.choice(ops), self.rng.choice([2, 3, 4, 5])
        if family == "pointwise":
            return self.rng.choice(POINTWISE_OPS), self.rng.choice(POINTWISE_OPS), self.rng.choice([1, 2, 3, 4, 5])
        raise ValueError(f"unknown Stage 1 anchor family: {family}")

    def _generate_stage2_recipe(self, cell: SampleCell) -> Tuple[ProgramGraph, Dict[str, Any]]:
        """Build a long, valid chain from a primary-family grammar.

        The returned target deliberately has no fusion nodes: Stage 2 uses the
        construction metadata for auditing, not as an instruction in the RL
        prompt.  The primary family controls exact dataset coverage.
        """
        graph = ProgramGraph()
        recipe = self._stage2_recipe_ops(cell.anchor_family, cell.program_op_count)
        # Reduction motifs begin with pointwise context, so choose enough rank
        # for their one scheduled reduction explicitly.
        first_rank = (
            self.rng.choice([3, 4, 5])
            if cell.anchor_family == "reduction"
            else self._required_primary_rank(recipe[0])
        )
        current = self._new_input(graph, first_rank)
        last_linear: Optional[OperatorNode] = None
        last_conv: Optional[OperatorNode] = None

        for op_name in recipe:
            node, current = self._append_operator(
                graph, op_name, current, workload_bucket=cell.workload_bucket,
            )
            # Pointwise middle chains preserve shape, so link the next heavy
            # operator to the latest compatible heavy producer.
            if op_name == "linear":
                if last_linear is not None:
                    node.attributes["in_features"] = last_linear.attributes["out_features"]
                last_linear = node
            if OPERATORS[op_name].category.startswith("Conv"):
                if last_conv is not None:
                    node.attributes["in_channels"] = last_conv.attributes["out_channels"]
                last_conv = node

        graph.output_edges = [current]
        return graph, {
            "anchor_family": cell.anchor_family,
            "anchor_op": recipe[0],
            "pattern": "mixed_recipe",
            "recipe": recipe,
            "pre_nodes": [],
            "anchor_nodes": [],
            "post_nodes": [],
            "all_nodes": [],
        }

    def _generate_stage1_recipe_single(self, cell: SampleCell) -> Tuple[ProgramGraph, Dict[str, Any]]:
        """Emit one compact Stage 2 grammar motif, never long-chain padding."""
        graph = ProgramGraph()
        recipe = self._stage2_single_recipe_ops(cell.anchor_family, cell.program_op_count)
        first_rank = self.rng.choice([3, 4, 5]) if cell.anchor_family == "reduction" else self._required_primary_rank(recipe[0])
        current = self._new_input(graph, first_rank)
        nodes: List[str] = []
        last_linear: Optional[OperatorNode] = None
        last_conv: Optional[OperatorNode] = None
        for op_name in recipe:
            node, current = self._append_operator(graph, op_name, current, role="recipe", target=True, workload_bucket=cell.workload_bucket)
            if op_name == "linear":
                if last_linear is not None: node.attributes["in_features"] = last_linear.attributes["out_features"]
                last_linear = node
            if OPERATORS[op_name].category.startswith("Conv"):
                if last_conv is not None: node.attributes["in_channels"] = last_conv.attributes["out_channels"]
                last_conv = node
            nodes.append(node.node_id)
        graph.output_edges = [current]
        return graph, {"anchor_family": cell.anchor_family, "anchor_op": recipe[0], "pattern": "stage2_recipe_single", "recipe": recipe, "pre_nodes": [], "anchor_nodes": nodes, "scheduled_anchor_nodes": [nodes[0]], "post_nodes": [], "all_nodes": nodes}

    def _stage2_single_recipe_ops(self, family: str, n_ops: int) -> List[str]:
        """Use the Stage 2 family grammar once at compact 3--5-op length."""
        if n_ops not in (3, 4, 5): raise ValueError("compact Stage 2 motifs must contain 3--5 operators")
        def pw(n: int) -> List[str]: return [self.rng.choice(POINTWISE_OPS) for _ in range(n)]
        mid = pw(n_ops - 2)
        if family == "matrix":
            x=self.rng.choices(["linear", "matmul", "bmm"], weights=[50,30,20])[0]; return [x]+mid+[x]
        if family == "conv":
            x=self.rng.choice(ANCHOR_CATEGORIES["conv"]["anchors"]); return [x]+mid+[self.rng.choices([x,self._pool_for_conv(x)], weights=[65,35])[0]]
        if family == "pooling":
            x=self.rng.choice(ANCHOR_CATEGORIES["conv"]["anchors"]); return [x]+mid+[self._pool_for_conv(x)]
        if family == "normalization":
            x=self.rng.choice(ANCHOR_CATEGORIES["normalization"]["anchors"]); return [x]+mid+[x]
        if family == "softmax":
            return [self.rng.choice(["linear","matmul","bmm"])]+mid+[self.rng.choice(["softmax","log_softmax"])]
        if family == "reduction":
            k=self.rng.randint(1,n_ops-2); return pw(k)+[self.rng.choice(REDUCTION_OPS)]+pw(n_ops-k-1)
        if family == "cumulative":
            return [self.rng.choice(["cumsum","cummax","cummin"])]+mid+[self.rng.choice(["layer_norm","batch_norm"])]
        raise ValueError(f"unknown Stage 2 recipe family: {family}")

    def _stage2_recipe_ops(self, family: str, n_ops: int) -> List[str]:
        """Repeat complete compatible motifs, then take one partial motif."""
        if n_ops < 6:
            raise ValueError("Stage 2 recipe length must be at least six")

        def pointwise(count: int) -> List[str]:
            return [self.rng.choice(POINTWISE_OPS) for _ in range(count)]

        # Keep the concrete anchor subtype fixed across repeated motifs so
        # rank and feature/channel contracts remain valid.
        if family == "matrix":
            heavy = self.rng.choices(["linear", "matmul", "bmm"], weights=[50, 30, 20])[0]
            motif = lambda: [heavy] + pointwise(self.rng.randint(1, 3)) + [heavy]
        elif family == "conv":
            conv = self.rng.choice(ANCHOR_CATEGORIES["conv"]["anchors"])
            motif = lambda: [conv] + pointwise(self.rng.randint(1, 3)) + [
                self.rng.choices([conv, self._pool_for_conv(conv)], weights=[65, 35])[0]
            ]
        elif family == "pooling":
            conv = self.rng.choice(ANCHOR_CATEGORIES["conv"]["anchors"])
            pool = self._pool_for_conv(conv)
            motif = lambda: [conv] + pointwise(self.rng.randint(1, 3)) + [pool]
        elif family == "normalization":
            norm = self.rng.choice(ANCHOR_CATEGORIES["normalization"]["anchors"])
            motif = lambda: [norm] + pointwise(self.rng.randint(1, 3)) + [norm]
        elif family == "softmax":
            heavy = self.rng.choice(["linear", "matmul", "bmm"])
            softmax = self.rng.choice(["softmax", "log_softmax"])
            motif = lambda: [heavy] + pointwise(self.rng.randint(1, 3)) + [softmax]
        elif family == "reduction":
            # One reduction only.  Subsequent motifs preserve rank.
            reduction = self.rng.choice(REDUCTION_OPS)
            norm = self.rng.choice(["layer_norm", "batch_norm"])
            core = pointwise(self.rng.randint(1, 3)) + [reduction] + pointwise(self.rng.randint(1, 3))
            post_reduction_motif = lambda: pointwise(self.rng.randint(1, 3)) + [norm]
        elif family == "cumulative":
            cumulative = self.rng.choice(["cumsum", "cummax", "cummin"])
            norm = self.rng.choice(["layer_norm", "batch_norm"])
            motif = lambda: [cumulative] + pointwise(self.rng.randint(1, 3)) + [norm]
        else:
            raise ValueError(f"unknown Stage 2 recipe family: {family}")

        if family == "reduction":
            ops = core
            while len(ops) < n_ops:
                partial = post_reduction_motif()
                ops.extend(partial[:n_ops - len(ops)])
            return ops

        ops: List[str] = []
        while len(ops) < n_ops:
            partial = motif()
            ops.extend(partial[:n_ops - len(ops)])
        return ops

    def _pool_for_conv(self, conv: str) -> str:
        suffix = conv[-2:] if conv.startswith("conv") and conv[-2:] in {"1d", "2d", "3d"} else conv[-2:]
        if suffix not in {"1d", "2d", "3d"}:
            raise ValueError(f"cannot choose pooling companion for {conv}")
        return f"{self.rng.choice(['avg_pool', 'max_pool'])}{suffix}"

    def _generate_random(self, n_ops: int, allow_branches: bool) -> Tuple[ProgramGraph, Dict[str, Any]]:
        graph = ProgramGraph()
        random_pool = [
            op for op in COMPUTE_OPS
            if OPERATORS[op].category in {
                "UnaryPointwise", "BinaryPointwise", "Reduction", "Softmax", "Cumulative"
            }
        ]
        current: Optional[str] = None
        current_rank: Optional[int] = None
        for _ in range(n_ops):
            if current is None:
                op_name = self.rng.choice(random_pool)
                current_rank = self._required_primary_rank(op_name)
                current = self._new_input(graph, current_rank)
            else:
                compatible = [
                    op_name for op_name in random_pool
                    if current_rank in OPERATORS[op_name].input_ranks[0]
                    and not (OPERATORS[op_name].category == "Reduction" and current_rank <= 1)
                ]
                if not compatible:
                    compatible = [
                        op_name for op_name in POINTWISE_OPS
                        if current_rank in OPERATORS[op_name].input_ranks[0]
                    ]
                op_name = self.rng.choice(compatible)
            _, current = self._append_operator(graph, op_name, current)
            current_rank = graph.edges[current].rank

        assert current is not None
        graph.output_edges = [current]
        return graph, {}

    def _build_context(self, graph: ProgramGraph, count: int, desired_rank: int, allow_branches: bool) -> List[str]:
        if count == 0:
            return []
        context_pool = [
            op for op in COMPUTE_OPS
            if OPERATORS[op].category in {"UnaryPointwise", "BinaryPointwise", "Softmax", "Cumulative"}
            and desired_rank in OPERATORS[op].input_ranks[0]
        ]
        if not context_pool:
            raise ValueError(f"no context operators support rank {desired_rank}")
        current = self._new_input(graph, desired_rank)
        for _ in range(count):
            op_name = self.rng.choice(context_pool)
            _, current = self._append_operator(graph, op_name, current)
        return [current]

    def _fusion_ops(self, family: str, anchor: str, pattern: str, count: int) -> Tuple[List[str], List[str]]:
        if pattern == "chain":
            if family == "pointwise":
                ops = [anchor] + [self.rng.choice(POINTWISE_OPS) for _ in range(count - 1)]
            elif family == "reduction":
                ops = [anchor] + [self.rng.choice(REDUCTION_OPS) for _ in range(count - 1)]
            else:
                raise ValueError("chain is only legal for pointwise or reduction")
            return ops, ["chain"] * len(ops)

        surrounding = count - 1
        if pattern == "prologue":
            pre, post = surrounding, 0
        elif pattern == "epilogue":
            pre, post = 0, surrounding
        elif pattern in {"both", "prologue_anchor_epilogue"}:
            pre = max(1, surrounding // 2)
            post = surrounding - pre
        else:
            raise ValueError(f"unknown pattern: {pattern}")
        ops = [self.rng.choice(POINTWISE_OPS) for _ in range(pre)] + [anchor] + [self.rng.choice(POINTWISE_OPS) for _ in range(post)]
        roles = ["pre"] * pre + ["anchor"] + ["post"] * post
        return ops, roles

    def _append_operator(self, graph: ProgramGraph, op_name: str, primary_edge: str,
                         role: Optional[str] = None, target: bool = False,
                         workload_bucket: str = "small") -> Tuple[OperatorNode, str]:
        op = OPERATORS[op_name]
        primary_rank = graph.edges[primary_edge].rank
        allowed_primary_ranks = set(op.input_ranks[0])
        if primary_rank not in allowed_primary_ranks:
            raise ValueError(
                f"{op_name} requires primary rank in {sorted(allowed_primary_ranks)}, "
                f"got rank {primary_rank} from edge {primary_edge}"
            )
        attrs = self._sample_attributes(op_name, primary_rank, workload_bucket)
        input_edges = [primary_edge]
        if op.in_count == 2:
            second_rank = self._second_input_rank(op_name, primary_rank)
            second = self._new_input(graph, second_rank, create_op="positive_rand" if op_name == "div" else None)
            input_edges.append(second)
        out_rank = self._infer_output_rank(op_name, [graph.edges[e].rank for e in input_edges], attrs)
        node_id = self._new_node_id()
        output_edge = self._new_edge_id()
        node = OperatorNode(node_id, op_name, input_edges, output_edge, attrs, target, role)
        graph.add_node(node, out_rank)
        return node, output_edge

    def _sample_attributes(
        self, op_name: str, input_rank: int, workload_bucket: str = "small"
    ) -> Dict[str, Any]:
        cat = OPERATORS[op_name].category
        if cat == "Reduction":
            return {"dim": self.rng.randrange(input_rank), "keepdim": False}
        if cat in {"Softmax", "Cumulative"}:
            return {"dim": self.rng.randrange(input_rank)}
        if cat == "Linear":
            features = [32, 64, 96, 128, 192, 256, 512]
            if workload_bucket == "large":
                features = [512, 1024, 2048, 4096, 8192]
            return {
                "in_features": self.rng.choice(features),
                "out_features": self.rng.choice(features),
                "bias": True,
            }
        if cat.startswith("Pool"):
            return {"kernel_size": 2, "stride": 2}
        if cat.startswith("Conv"):
            return {
                "in_channels": self.rng.choice([4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]),
                "out_channels": self.rng.choice([8, 16, 24, 32, 48, 64, 96, 128, 192, 256]),
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "bias": True,
            }
        if cat == "GroupNorm":
            return {"num_groups": 1}
        if cat == "UnaryPointwise" and op_name == "clamp":
            return {"min": -1.0, "max": 1.0}
        return {}

    def _infer_output_rank(self, op_name: str, input_ranks: List[int], attrs: Dict[str, Any]) -> int:
        rule = OPERATORS[op_name].output_rank_rule
        if rule == "same": return input_ranks[0]
        if rule == "reduction": return input_ranks[0] if attrs.get("keepdim") else input_ranks[0] - 1
        if rule == "matmul": return 2
        if rule == "bmm": return 3
        if rule == "linear": return 2
        if rule.startswith("pool") or rule.startswith("conv"): return input_ranks[0]
        raise ValueError(f"unsupported output rule: {rule}")

    def _required_primary_rank(self, op_name: str) -> int:
        return self.rng.choice(list(OPERATORS[op_name].input_ranks[0]))

    @staticmethod
    def _second_input_rank(op_name: str, primary_rank: int) -> int:
        cat = OPERATORS[op_name].category
        if cat == "BinaryPointwise": return primary_rank
        if cat == "Matmul": return 2
        if cat == "BMM": return 3
        raise ValueError(f"unexpected binary op: {op_name}")

    def _choose_edge_of_rank(self, graph: ProgramGraph, live: List[str], rank: int) -> Optional[str]:
        candidates = [e for e in live if graph.edges[e].rank == rank]
        return self.rng.choice(candidates) if candidates else None

    def _new_input(self, graph: ProgramGraph, rank: int, create_op: Optional[str] = None) -> str:
        edge_id = self._new_edge_id()
        if create_op is None:
            create_op = self.rng.choices(["randn", "ones", "zeros"], weights=[90, 5, 5], k=1)[0]
        graph.add_input_edge(edge_id, rank, create_op)
        return edge_id

    @staticmethod
    def validate_output_connectivity(graph: ProgramGraph) -> bool:
        producer_by_edge = {edge.edge_id: edge.producer for edge in graph.edges.values()}
        reachable_nodes = set()
        stack = list(graph.output_edges)
        while stack:
            edge_id = stack.pop()
            producer = producer_by_edge.get(edge_id)
            if producer is None or producer in reachable_nodes:
                continue
            reachable_nodes.add(producer)
            stack.extend(graph.nodes[producer].input_edges)
        return reachable_nodes == set(graph.nodes)

    @staticmethod
    def validate_numerical_safety(graph: ProgramGraph) -> bool:
        """Reject direct producer/consumer pairs likely to overflow internally."""
        for node in graph.topological_nodes():
            for edge_id in node.input_edges:
                producer_id = graph.edges[edge_id].producer
                if producer_id is None:
                    continue
                producer_op = graph.nodes[producer_id].op_name
                if (producer_op, node.op_name) in UNSAFE_ADJACENCIES:
                    return False
        return True

    def _new_node_id(self) -> str:
        value = f"n{self.node_idx}"
        self.node_idx += 1
        return value

    def _new_edge_id(self) -> str:
        value = f"e{self.edge_idx}"
        self.edge_idx += 1
        return value
