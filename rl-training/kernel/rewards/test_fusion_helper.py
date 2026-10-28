import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).with_name("fusion_helper.py")
SPEC = importlib.util.spec_from_file_location("fusion_helper_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
FUSION_HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FUSION_HELPER
SPEC.loader.exec_module(FUSION_HELPER)

classify_fusion_boundaries = FUSION_HELPER.classify_fusion_boundaries
compute_fusion_reward = FUSION_HELPER.compute_fusion_reward
analyze_fusion_source = FUSION_HELPER.analyze_fusion_source
reference_fusion_plans_text = FUSION_HELPER.reference_fusion_plans_text
classify_profiled_boundaries = FUSION_HELPER._classify_profiled_boundaries
matched_profiler_launch_names = FUSION_HELPER._matched_profiler_launch_names
FusionDetectionContext = FUSION_HELPER.FusionDetectionContext
ProfiledExportFusionDetector = FUSION_HELPER.ProfiledExportFusionDetector


def custom_node(node_id, name="whatever", semantics=None):
    lifted_semantics = list(semantics) if semantics is not None else ["unknown"]
    return {
        "node_id": node_id,
        "name": name,
        "backend": "triton",
        "original_kind": "triton_wrapper",
        "semantics": lifted_semantics,
        "opacity": "opaque" if lifted_semantics == ["unknown"] else "partial",
        "op_category": "unknown" if lifted_semantics == ["unknown"] else "pointwise",
        "evidence": ["no_semantic_match"] if lifted_semantics == ["unknown"] else ["name_pattern"],
    }


def visible_node(node_id, semantic):
    return {
        "node_id": node_id,
        "backend": "torch",
        "original_kind": "torch_op",
        "semantics": [semantic],
        "opacity": "transparent",
        "op_category": "reduction" if semantic == "sum" else "pointwise",
    }


def graph(nodes):
    return {
        "lifted_nodes": nodes,
        "edges": [
            {"src": nodes[index]["node_id"], "dst": nodes[index + 1]["node_id"]}
            for index in range(len(nodes) - 1)
        ],
    }


CHAIN_PLAN = {
    "fragment_id": "f0",
    "action": "merge",
    "node_ids": ["r0", "r1", "r2"],
    "semantics": ["relu", "add", "sum"],
}


class FusionBoundaryTest(unittest.TestCase):
    def test_prompt_renders_unique_adjacent_boundaries(self):
        overlapping_plan = {
            "action": "merge",
            "semantics": ["sigmoid", "mul"],
            "fallback_for": "p0",
        }
        reference_fusion_plans_text.cache_clear()
        with patch.object(
            FUSION_HELPER,
            "reference_fusion_context",
            return_value={
                "prompt_boundaries": [
                    ["relu", "add"],
                    ["add", "sum"],
                ]
            },
        ):
            text = reference_fusion_plans_text("prompt-boundary-test")

        self.assertEqual(text, "- relu -> add\n- add -> sum")

    def test_real_parser_assigns_opaque_prefix_before_visible_reduction(self):
        reference = """
import torch

class Model:
    def forward(self, x, residual):
        x = torch.relu(x)
        x = torch.add(x, residual)
        return torch.sum(x, dim=1)
"""
        generated = """
import torch
import triton

@triton.jit
def whatever(x, residual, out):
    pass

class ModelNew:
    def forward(self, x, residual):
        out = torch.empty_like(x)
        whatever[(1,)](x, residual, out)
        return torch.sum(out, dim=1)
"""
        _, reference_plans = analyze_fusion_source(reference)
        generated_graph, _ = analyze_fusion_source(generated)

        boundaries = classify_fusion_boundaries(reference_plans, generated_graph)

        self.assertEqual(
            [(item["left_semantic"], item["right_semantic"], item["proven_fused"])
             for item in boundaries],
            [("relu", "add", True), ("add", "sum", False)],
        )

    def test_real_parser_uses_names_for_two_custom_launches(self):
        reference = """
import torch

class Model:
    def forward(self, x, residual):
        x = torch.relu(x)
        x = torch.add(x, residual)
        return torch.sum(x, dim=1)
"""
        generated = """
import torch
import triton

@triton.jit
def relu_add_kernel(x, residual, temporary):
    pass

@triton.jit
def reduction_kernel(temporary, output):
    pass

class ModelNew:
    def forward(self, x, residual):
        temporary = torch.empty_like(x)
        relu_add_kernel[(1,)](x, residual, temporary)
        output = torch.empty_like(x)
        reduction_kernel[(1,)](temporary, output)
        return output
"""
        _, reference_plans = analyze_fusion_source(reference)
        generated_graph, _ = analyze_fusion_source(generated)

        boundaries = classify_fusion_boundaries(reference_plans, generated_graph)

        self.assertEqual(
            [(item["left_semantic"], item["right_semantic"], item["proven_fused"])
             for item in boundaries],
            [("relu", "add", True), ("add", "sum", False)],
        )

    def test_residual_assignment_proves_only_prefix_fusion(self):
        boundaries = classify_fusion_boundaries(
            [CHAIN_PLAN], graph([custom_node("g0"), visible_node("g1", "sum")])
        )

        self.assertEqual(
            [(item["left_semantic"], item["right_semantic"], item["proven_fused"])
             for item in boundaries],
            [("relu", "add", True), ("add", "sum", False)],
        )

    def test_single_custom_launch_proves_all_boundaries(self):
        boundaries = classify_fusion_boundaries([CHAIN_PLAN], graph([custom_node("g0")]))

        self.assertTrue(all(item["proven_fused"] for item in boundaries))

    def test_separate_custom_launches_are_ambiguous_and_not_proven(self):
        boundaries = classify_fusion_boundaries(
            [CHAIN_PLAN], graph([custom_node("g0"), custom_node("g1")])
        )

        self.assertTrue(all(not item["proven_fused"] for item in boundaries))
        self.assertTrue(all(item["status"] == "not_fused" for item in boundaries))

    def test_lifted_semantics_disambiguate_sequential_custom_launches(self):
        boundaries = classify_fusion_boundaries(
            [CHAIN_PLAN],
            graph(
                [
                    custom_node("g0", "relu_add_kernel", ["relu", "add"]),
                    custom_node("g1", "reduction_kernel"),
                ]
            ),
        )

        self.assertEqual(
            [(item["left_semantic"], item["right_semantic"], item["proven_fused"])
             for item in boundaries],
            [("relu", "add", True), ("add", "sum", False)],
        )

    def test_contradictory_lifted_semantics_are_not_proven(self):
        boundaries = classify_fusion_boundaries(
            [CHAIN_PLAN],
            graph(
                [
                    custom_node("g0", "sum_kernel", ["sum"]),
                    custom_node("g1", "relu_add_kernel", ["relu", "add"]),
                ]
            ),
        )

        self.assertTrue(all(not item["proven_fused"] for item in boundaries))

    def test_lifted_semantics_cannot_hide_duplicate_visible_operation(self):
        boundaries = classify_fusion_boundaries(
            [CHAIN_PLAN],
            graph(
                [
                    custom_node(
                        "g0", "relu_add_sum_kernel", ["relu", "add", "sum"]
                    ),
                    visible_node("g1", "sum"),
                ]
            ),
        )

        self.assertTrue(all(not item["proven_fused"] for item in boundaries))

    def test_visible_softplus_prevents_false_conv_fusion_credit(self):
        plan = {
            "fragment_id": "f0",
            "action": "fuse_epilogue",
            "node_ids": ["r0", "r1"],
            "semantics": ["conv2d", "softplus"],
        }
        boundaries = classify_fusion_boundaries(
            [plan], graph([custom_node("g0"), visible_node("g1", "softplus")])
        )

        self.assertFalse(boundaries[0]["proven_fused"])
        self.assertEqual(boundaries[0]["status"], "not_fused")

    def test_path_truncation_prevents_proof(self):
        path = [custom_node("g0")]
        with patch.object(
            FUSION_HELPER,
            "_enumerate_execution_paths",
            return_value=([path], True),
        ):
            boundaries = classify_fusion_boundaries([CHAIN_PLAN], graph(path))

        self.assertTrue(all(item["status"] == "unknown" for item in boundaries))
        self.assertTrue(all(item["analysis_truncated"] for item in boundaries))

    def test_alignment_truncation_prevents_proof(self):
        path = [custom_node("g0")]
        with patch.object(
            FUSION_HELPER,
            "_align_semantics_to_path",
            return_value=([("g0", "g0", "g0")], True),
        ):
            boundaries = classify_fusion_boundaries([CHAIN_PLAN], graph(path))

        self.assertTrue(all(item["status"] == "unknown" for item in boundaries))
        self.assertTrue(all(item["analysis_truncated"] for item in boundaries))

    def test_exact_ordered_name_proves_unknown_boundary(self):
        plan = {
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1"],
            "semantics": ["maximum", "mul"],
        }
        boundaries = classify_fusion_boundaries(
            [plan],
            graph([custom_node("g0", "max_mul_fwd_kernel", ["max", "mul"])]),
        )

        self.assertEqual(boundaries[0]["status"], "proven_fused")
        self.assertEqual(boundaries[0]["proof_source"], "name")

    def test_name_fallback_requires_ordered_tokens(self):
        plan = {
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1"],
            "semantics": ["maximum", "mul"],
        }
        boundaries = classify_fusion_boundaries(
            [plan],
            graph([custom_node("g0", "mul_max_kernel", ["max", "mul"])]),
        )

        self.assertEqual(boundaries[0]["status"], "unknown")

    def test_name_fallback_never_overrides_explicit_split(self):
        pair_plan = {
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1"],
            "semantics": ["relu", "clamp"],
        }
        boundaries = classify_fusion_boundaries(
            [pair_plan],
            graph(
                [
                    custom_node("g0", "relu_kernel"),
                    custom_node("g1", "relu_clamp_kernel"),
                ]
            ),
        )

        self.assertEqual(boundaries[0]["status"], "not_fused")

    def test_overlapping_plans_do_not_duplicate_boundaries(self):
        pair_plan = {
            "fragment_id": "f1",
            "action": "merge",
            "node_ids": ["r0", "r1"],
            "semantics": ["relu", "add"],
        }
        boundaries = classify_fusion_boundaries(
            [CHAIN_PLAN, pair_plan], graph([custom_node("g0"), visible_node("g1", "sum")])
        )

        self.assertEqual(len(boundaries), 2)

    def test_generated_parse_failure_is_neutral(self):
        with patch.object(
            FUSION_HELPER,
            "analyze_fusion_source",
            side_effect=[({}, [CHAIN_PLAN]), ValueError("bad generated source")],
        ):
            result = compute_fusion_reward(
                "reference",
                "generated",
                resolved_weight=0.3,
                unresolved_weight=-0.1,
                detector="ast",
            )

        self.assertEqual(result["fusion_proven_fused_count"], 0)
        self.assertEqual(result["fusion_not_proven_fused_count"], 2)
        self.assertEqual(result["fusion_unknown_count"], 2)
        self.assertAlmostEqual(result["fusion_reward"], 0.0)
        self.assertIn("generated plan generation failed", result["fusion_error"])

    def test_only_explicit_not_fused_boundaries_are_penalized(self):
        generated_graph = graph(
            [custom_node("g0"), custom_node("g1"), visible_node("g2", "sum")]
        )
        with patch.object(
            FUSION_HELPER,
            "analyze_fusion_source",
            side_effect=[({}, [CHAIN_PLAN]), (generated_graph, [])],
        ):
            result = compute_fusion_reward(
                "reference",
                "generated",
                resolved_weight=0.3,
                unresolved_weight=-0.1,
                detector="ast",
            )

        self.assertEqual(result["fusion_not_fused_count"], 2)
        self.assertAlmostEqual(result["fusion_reward"], -0.1)

    def test_reward_metadata_uses_proven_fused_names_only(self):
        with patch.object(
            FUSION_HELPER,
            "analyze_fusion_source",
            side_effect=[({}, [CHAIN_PLAN]), (graph([custom_node("g0")]), [])],
        ):
            result = compute_fusion_reward(
                "reference", "generated", detector="ast"
            )

        legacy_keys = {
            "fusion_resolved_ratio",
            "fusion_unresolved_ratio",
            "fusion_resolved_count",
            "fusion_unresolved_count",
            "fusion_resolved_plans",
            "fusion_unresolved_plans",
            "fusion_resolved_anchor_count",
        }
        self.assertTrue(legacy_keys.isdisjoint(result))
        self.assertIn("fusion_proven_fused_ratio", result)
        self.assertIn("fusion_not_proven_fused_ratio", result)
        self.assertIn("fusion_proven_fused_anchor_count", result)

    def test_profiled_export_combines_name_and_visible_anchor(self):
        generated_graph = graph(
            [
                custom_node("g0", "triton_kernel::relu_add__abcd1234"),
                visible_node("g1", "sum"),
            ]
        )

        boundaries, alignments, inferred = classify_profiled_boundaries(
            [CHAIN_PLAN],
            generated_graph,
            ["relu_add_kernel"],
            True,
        )

        self.assertEqual(
            [item["status"] for item in boundaries],
            ["proven_fused", "not_fused"],
        )
        self.assertEqual(alignments[0]["assignment_count"], 1)
        self.assertEqual(len(inferred), 1)

    def test_profiled_export_requires_profiler_confirmation(self):
        generated_graph = graph(
            [custom_node("g0", "triton_kernel::relu_add_sum__abcd1234")]
        )

        boundaries, _, _ = classify_profiled_boundaries(
            [CHAIN_PLAN],
            generated_graph,
            [],
            True,
        )

        self.assertTrue(all(item["status"] == "unknown" for item in boundaries))

    def test_draft_export_disables_opaque_wildcard_proof(self):
        generated_graph = graph(
            [custom_node("g0", "triton_kernel::opaque__abcd1234")]
        )

        boundaries, _, _ = classify_profiled_boundaries(
            [CHAIN_PLAN],
            generated_graph,
            ["opaque_kernel"],
            True,
            False,
        )

        self.assertTrue(all(item["status"] == "unknown" for item in boundaries))

    def test_profiled_export_keeps_confirmed_custom_terminal_not_returned(self):
        generated_graph = {
            "lifted_nodes": [
                custom_node("g0", "triton_kernel::relu_add_sum__abcd1234"),
                visible_node("g1", "mean"),
            ],
            "edges": [],
            "returns": ["g1"],
        }

        boundaries, alignments, _ = classify_profiled_boundaries(
            [CHAIN_PLAN],
            generated_graph,
            ["relu_add_sum_kernel"],
            True,
        )

        self.assertEqual(
            [item["status"] for item in boundaries],
            ["proven_fused", "proven_fused"],
        )
        self.assertEqual(alignments[0]["assignment_count"], 1)

    def test_incomplete_dependency_path_does_not_override_valid_alignment(self):
        primary = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "fuse_epilogue",
            "node_ids": ["r0", "r1", "r2", "r3"],
            "semantics": ["linear", "elu", "abs", "hardswish"],
        }
        generated_graph = {
            "lifted_nodes": [
                visible_node("g0", "linear"),
                {
                    "node_id": "g1",
                    "name": "aten.empty_like.default",
                    "backend": "torch",
                    "original_kind": "torch_op",
                    "semantics": ["unknown"],
                    "opacity": "opaque",
                    "op_category": "opaque",
                },
                custom_node(
                    "g2", "triton_kernel::elu_abs_hardswish__abcd1234"
                ),
                visible_node("g3", "view"),
            ],
            "edges": [
                {"src": "g0", "dst": "g1"},
                {"src": "g0", "dst": "g2"},
                {"src": "g1", "dst": "g2"},
                {"src": "g2", "dst": "g3"},
                {"src": "g0", "dst": "g3"},
            ],
            "returns": ["g3"],
        }

        boundaries, _, _ = classify_profiled_boundaries(
            [primary],
            generated_graph,
            ["_elu_abs_hardswish_kernel"],
            True,
        )

        self.assertEqual(
            [item["status"] for item in boundaries],
            ["not_fused", "proven_fused", "proven_fused"],
        )

    def test_profiled_export_ignores_fallback_plan_for_classification(self):
        primary = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1", "r2"],
            "semantics": ["elu", "abs", "hardswish"],
        }
        fallback = {
            "plan_id": "p1",
            "fragment_id": "f1",
            "action": "merge",
            "node_ids": ["r0", "r1", "r2"],
            "semantics": ["elu", "abs", "hardswish"],
            "fallback_for": "p0",
        }
        generated_graph = graph(
            [
                custom_node(
                    "g0", "triton_kernel::elu_abs_hardswish__abcd1234"
                )
            ]
        )

        boundaries, alignments, _ = classify_profiled_boundaries(
            [primary, fallback],
            generated_graph,
            ["_elu_abs_hardswish_kernel"],
            True,
        )

        self.assertTrue(
            all(item["status"] == "proven_fused" for item in boundaries)
        )
        self.assertEqual(len(alignments), 1)
        self.assertEqual(alignments[0]["plan_id"], "p0")

    def test_distinct_confirmed_named_launches_prove_split(self):
        generated_graph = {
            "lifted_nodes": [
                custom_node("g0", "triton_kernel::elu__aaaa1111"),
                custom_node("g1", "triton_kernel::abs__bbbb2222"),
                custom_node("g2", "triton_kernel::hardswish__cccc3333"),
            ],
            "edges": [],
        }
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1", "r2"],
            "semantics": ["elu", "abs", "hardswish"],
        }

        boundaries, _, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["elu_kernel", "abs_kernel", "hardswish_kernel"],
            True,
        )

        self.assertEqual(
            [item["status"] for item in boundaries],
            ["not_fused", "not_fused"],
        )

    def test_overlapping_named_launches_prove_partial_boundaries(self):
        generated_graph = {
            "lifted_nodes": [
                custom_node(
                    "g0", "triton_kernel::conv_silu_write_z__aaaa1111"
                ),
                custom_node(
                    "g1", "triton_kernel::silu_add_bias__bbbb2222"
                ),
            ],
            "edges": [],
        }
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "fuse_epilogue",
            "node_ids": ["r0", "r1", "r2"],
            "semantics": ["conv1d", "silu", "add"],
        }

        boundaries, alignments, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["conv_silu_write_z_kernel", "silu_add_bias_kernel"],
            True,
        )

        self.assertEqual(
            [item["status"] for item in boundaries],
            ["proven_fused", "proven_fused"],
        )
        self.assertTrue(
            all(item["proof_source"] == "profiled_name" for item in boundaries)
        )
        self.assertGreater(alignments[0]["assignment_count"], 0)

    def test_single_profiled_slot_absorbs_missing_contiguous_span(self):
        generated_graph = graph(
            [
                visible_node("g0", "linear"),
                custom_node(
                    "g1",
                    "triton_kernel::relu_add__aaaa1111",
                    semantics=["add"],
                ),
                visible_node("g2", "sum"),
            ]
        )
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1", "r2", "r3"],
            "semantics": ["linear", "relu", "add", "sum"],
        }

        boundaries, alignments, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["relu_add_kernel"],
            True,
        )

        self.assertEqual(
            [item["status"] for item in boundaries],
            ["not_fused", "proven_fused", "not_fused"],
        )
        self.assertEqual(alignments[0]["assignment_count"], 1)

    def test_two_opaque_profiled_slots_leave_mixed_boundaries_unknown(self):
        generated_graph = graph(
            [
                custom_node("g0", "triton_kernel::first__aaaa1111"),
                custom_node("g1", "triton_kernel::second__bbbb2222"),
            ]
        )
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1", "r2"],
            "semantics": ["relu", "add", "sum"],
        }

        boundaries, alignments, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["first_kernel", "second_kernel"],
            False,
        )

        self.assertEqual(
            [item["status"] for item in boundaries],
            ["unknown", "unknown"],
        )
        self.assertEqual(alignments[0]["assignment_count"], 2)

    def test_repeated_profiler_names_remain_distinct_occurrences(self):
        generated_graph = graph(
            [
                custom_node("g0", "triton_kernel::relu__aaaa1111"),
                custom_node("g1", "triton_kernel::relu__bbbb2222"),
            ]
        )
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1"],
            "semantics": ["relu", "relu"],
        }

        boundaries, alignments, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["relu_kernel", "relu_kernel"],
            True,
        )

        self.assertEqual(boundaries[0]["status"], "not_fused")
        assignment = alignments[0]["assignments"][0]
        self.assertNotEqual(assignment[0], assignment[1])
        self.assertEqual(
            matched_profiler_launch_names(
                generated_graph,
                {"triton_kernel_in_profiling": ["relu_kernel", "relu_kernel"]},
            ),
            ["relu_kernel", "relu_kernel"],
        )

    def test_branch_path_cannot_hide_distinct_named_launches(self):
        generated_graph = {
            "lifted_nodes": [
                {
                    "node_id": "g0",
                    "name": "aten.empty.memory_format",
                    "backend": "torch",
                    "original_kind": "torch_op",
                    "semantics": ["unknown"],
                    "opacity": "opaque",
                    "op_category": "unknown",
                },
                custom_node(
                    "g1",
                    "triton_kernel::upsample_log1p__aaaa1111",
                    semantics=["log1p"],
                ),
                custom_node(
                    "g2",
                    "triton_kernel::bias_add__bbbb2222",
                    semantics=["add"],
                ),
            ],
            "edges": [
                {"src": "g0", "dst": "g1"},
                {"src": "g0", "dst": "g2"},
            ],
        }
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1"],
            "semantics": ["log1p", "add"],
        }

        boundaries, _, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["upsample_log1p_kernel", "bias_add_kernel"],
            True,
        )

        self.assertEqual(boundaries[0]["status"], "not_fused")

    def test_name_derived_semantics_are_unordered_membership_constraints(self):
        generated_graph = graph(
            [
                custom_node(
                    "g0",
                    "triton_kernel::neg_min_max__aaaa1111",
                    semantics=["max", "min"],
                )
            ]
        )
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1", "r2"],
            "semantics": ["neg", "minimum", "maximum"],
        }

        boundaries, alignments, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["neg_min_max_kernel"],
            True,
        )

        self.assertEqual(
            [item["status"] for item in boundaries],
            ["proven_fused", "proven_fused"],
        )
        self.assertEqual(alignments[0]["assignment_count"], 1)

    def test_export_order_overrides_runtime_sorted_profiler_names(self):
        generated_graph = {
            "lifted_nodes": [
                custom_node(
                    "g0", "triton_kernel::preprocess__aaaa1111"
                ),
                custom_node(
                    "g1",
                    "triton_kernel::maxpool1d__bbbb2222",
                    semantics=["max_pool1d"],
                ),
            ],
            "edges": [],
        }
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "fuse_prologue",
            "node_ids": ["r0", "r1", "r2", "r3"],
            "semantics": ["add", "log", "minimum", "max_pool1d"],
        }

        boundaries, _, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["maxpool1d_kernel", "preprocess_kernel"],
            True,
        )

        self.assertTrue(all(item["status"] == "unknown" for item in boundaries))
        assignments = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["maxpool1d_kernel", "preprocess_kernel"],
            True,
        )[1][0]["assignments"]
        self.assertTrue(
            all(
                assignment.index("g0@export:0")
                <= assignment.index("g1@export:1")
                for assignment in assignments
            )
        )

    def test_reference_context_assigns_neighboring_reduction_launch(self):
        reference_graph = graph(
            [
                visible_node("r0", "ge"),
                visible_node("r1", "mul"),
                visible_node("r2", "sum"),
            ]
        )
        generated_graph = graph(
            [
                custom_node(
                    "g0",
                    "triton_kernel::masked_dist__aaaa1111",
                    semantics=["ge", "mul"],
                ),
                custom_node(
                    "g1",
                    "triton_kernel::final_sum__bbbb2222",
                    semantics=["sum"],
                ),
            ]
        )
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1"],
            "semantics": ["ge", "mul"],
        }

        boundaries, alignments, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["masked_dist_kernel", "final_sum_kernel"],
            True,
            True,
            reference_graph,
        )

        self.assertEqual(boundaries[0]["status"], "proven_fused")
        self.assertEqual(alignments[0]["assignment_count"], 1)

    def test_unique_visible_anchor_prevents_wildcard_false_proof(self):
        generated_graph = {
            "lifted_nodes": [
                visible_node("g0", "linear"),
                custom_node("g1", "triton_kernel::add_bias__aaaa1111"),
                custom_node("g2", "triton_kernel::gelu__bbbb2222"),
            ],
            "edges": [{"src": "g0", "dst": "g1"}],
        }
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "fuse_epilogue",
            "node_ids": ["r0", "r1"],
            "semantics": ["linear", "add"],
        }

        boundaries, _, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["add_bias_kernel", "gelu_kernel"],
            True,
        )

        self.assertEqual(boundaries[0]["status"], "not_fused")

    def test_visible_and_named_locations_prove_split_without_full_alignment(self):
        generated_graph = graph(
            [
                visible_node("g0", "linear"),
                custom_node("g1", "triton_kernel::mish_twice__aaaa1111"),
            ]
        )
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "fuse_epilogue",
            "node_ids": ["r0", "r1", "r2"],
            "semantics": ["linear", "mish", "mish"],
        }

        boundaries, _, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["mish_twice_kernel"],
            True,
        )

        self.assertEqual(
            [item["status"] for item in boundaries],
            ["not_fused", "proven_fused"],
        )

    def test_visible_exported_op_overrides_fused_launch_name(self):
        generated_graph = graph(
            [
                visible_node("g0", "gelu"),
                custom_node(
                    "g1",
                    "triton_kernel::gelu_ne__aaaa1111",
                    semantics=["ne"],
                ),
            ]
        )
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1"],
            "semantics": ["gelu", "ne"],
        }

        boundaries, _, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["gelu_ne_kernel"],
            True,
        )

        self.assertEqual(boundaries[0]["status"], "not_fused")

    def test_repeated_semantics_use_ordered_name_positions(self):
        generated_graph = graph(
            [
                custom_node("g0", "triton_kernel::tanh_relu__aaaa1111"),
                custom_node("g1", "triton_kernel::tanh_only__bbbb2222"),
            ]
        )
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1", "r2"],
            "semantics": ["tanh", "relu", "tanh"],
        }

        boundaries, _, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["tanh_relu_kernel", "tanh_only_kernel"],
            True,
            False,
        )

        self.assertEqual(
            [item["status"] for item in boundaries],
            ["proven_fused", "not_fused"],
        )

    def test_name_fallback_does_not_override_visible_anchor(self):
        generated_graph = graph(
            [
                visible_node("g0", "batch_norm"),
                custom_node(
                    "g1",
                    "triton_kernel::masked_fill_from_bn__aaaa1111",
                ),
            ]
        )
        plan = {
            "plan_id": "p0",
            "fragment_id": "f0",
            "action": "merge",
            "node_ids": ["r0", "r1"],
            "semantics": ["batch_norm", "masked_fill"],
        }

        boundaries, _, _ = classify_profiled_boundaries(
            [plan],
            generated_graph,
            ["masked_fill_from_bn_kernel"],
            True,
            False,
        )

        self.assertEqual(boundaries[0]["status"], "not_fused")

    def test_incorrect_kernel_skips_profiled_export(self):
        with patch.object(
            ProfiledExportFusionDetector, "detect"
        ) as detector:
            result = compute_fusion_reward(
                "reference",
                "generated",
                correctness=False,
                detector="profiled_export",
            )

        detector.assert_not_called()
        self.assertEqual(result["fusion_skipped_reason"], "incorrect_kernel")
        self.assertEqual(
            result["fusion_plan_artifact"]["skipped_reason"],
            "incorrect_kernel",
        )

    def test_export_failure_returns_unknown_boundaries(self):
        with patch.object(
            FUSION_HELPER,
            "reference_fusion_context",
            return_value={
                "fusion_plans": [CHAIN_PLAN],
                "lifted_graph": {"lifted_nodes": [], "edges": []},
                "analysis_mode": "export",
                "export_error": None,
            },
        ), patch.object(
            FUSION_HELPER,
            "_export_generated_lifted_graph",
            side_effect=RuntimeError("not exportable"),
        ):
            result = ProfiledExportFusionDetector().detect(
                FusionDetectionContext(
                    reference_code="reference",
                    kernel_code="generated",
                    correctness=True,
                )
            )

        self.assertEqual(len(result.boundaries), 2)
        self.assertTrue(
            all(item["status"] == "unknown" for item in result.boundaries)
        )
        self.assertTrue(
            all(
                item["unknown_reason"] == "export_failure"
                for item in result.boundaries
            )
        )
        self.assertIsNotNone(result.reference_graph)
        self.assertIsNone(result.generated_graph)

    def test_generated_export_failure_uses_ast_graph_fallback(self):
        ast_graph = graph(
            [custom_node("g0", "triton_kernel::relu_add__abcd1234")]
        )
        with patch.object(
            FUSION_HELPER,
            "_load_defrag_modules",
            return_value={
                "export_file": Mock(
                    side_effect=RuntimeError("data dependent")
                )
            },
        ), patch.object(
            FUSION_HELPER,
            "analyze_fusion_source",
            return_value=(ast_graph, []),
        ):
            generated_graph, metadata = (
                FUSION_HELPER._export_generated_lifted_graph(
                    FusionDetectionContext(
                        reference_code="reference",
                        kernel_code="generated",
                        correctness=True,
                    )
                )
            )

        self.assertEqual(generated_graph, ast_graph)
        self.assertEqual(metadata["mode"], "ast_fallback")
        self.assertIn("data dependent", metadata["warning"])

    def test_profiler_artifact_keeps_only_matched_custom_launches(self):
        generated_graph = graph(
            [custom_node("g0", "triton_kernel::relu_add__abcd1234")]
        )
        metadata = {
            "profiling": {
                "kernels": [
                    {"name": "relu_add_kernel"},
                    {"name": "aten::sum"},
                    {"name": "cudaLaunchKernel"},
                ]
            }
        }

        launches = matched_profiler_launch_names(generated_graph, metadata)

        self.assertEqual(launches, ["relu_add_kernel"])


if __name__ == "__main__":
    unittest.main()
