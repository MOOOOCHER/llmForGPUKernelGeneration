import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


MODULE_PATH = Path(__file__).with_name("fusion_helper.py")
SPEC = importlib.util.spec_from_file_location("fusion_helper_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
FUSION_HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FUSION_HELPER
SPEC.loader.exec_module(FUSION_HELPER)


class FullProgramPlanTest(unittest.TestCase):
    def test_unstructured_sample_uses_complete_operator_chain(self):
        plan, reason = FUSION_HELPER._dataset_fusion_plan(
            {
                "is_structured": False,
                "operators": ["cummin", "sigmoid"],
                "fusion_chain": [],
                "program_graph": {
                    "nodes": [
                        {"node_id": "n0", "op_name": "cummin"},
                        {"node_id": "n1", "op_name": "sigmoid"},
                    ]
                },
            }
        )

        self.assertIsNone(reason)
        self.assertEqual(plan["semantics"], ["cummin", "sigmoid"])
        self.assertEqual(plan["node_ids"], ["n0", "n1"])

    def test_structured_sample_does_not_narrow_to_fusion_chain(self):
        plan, reason = FUSION_HELPER._dataset_fusion_plan(
            {
                "is_structured": True,
                "operators": ["tanh", "relu", "sum"],
                "fusion_chain": ["relu", "sum"],
                "program_graph": {
                    "nodes": [
                        {"node_id": "n0", "op_name": "tanh"},
                        {"node_id": "n1", "op_name": "relu"},
                        {"node_id": "n2", "op_name": "sum"},
                    ]
                },
            }
        )

        self.assertIsNone(reason)
        self.assertEqual(plan["semantics"], ["tanh", "relu", "sum"])
        self.assertEqual(plan["node_ids"], ["n0", "n1", "n2"])

    def test_parquet_numpy_operator_chain_is_normalized(self):
        plan, reason = FUSION_HELPER._dataset_fusion_plan(
            {
                "operators": np.array(
                    ["tanh", "relu", "minimum", "softplus", "cummin"],
                    dtype=object,
                ),
                "program_graph": {
                    "nodes": [
                        {"node_id": "n0", "op_name": "tanh"},
                        {"node_id": "n1", "op_name": "relu"},
                        {"node_id": "n2", "op_name": "minimum"},
                        {"node_id": "n3", "op_name": "softplus"},
                        {"node_id": "n4", "op_name": "cummin"},
                    ]
                },
            }
        )

        self.assertIsNone(reason)
        self.assertEqual(
            plan["semantics"],
            ["tanh", "relu", "minimum", "softplus", "cummin"],
        )
        self.assertEqual(plan["node_ids"], ["n0", "n1", "n2", "n3", "n4"])

    def test_single_operator_sample_is_not_compressible(self):
        plan, reason = FUSION_HELPER._dataset_fusion_plan(
            {"operators": ["relu"]}
        )

        self.assertIsNone(plan)
        self.assertEqual(reason, "operator_chain_too_short")

    def test_fusion_reward_uses_sub_one_speedup_without_gating(self):
        detection = FUSION_HELPER.FusionDetectionResult(
            detector="profiled_export",
            reference_plans=[{"semantics": ["relu", "sigmoid"]}],
            plan_alignments=[{"execution_group_counts": [1]}],
        )
        with patch.object(
            FUSION_HELPER.ProfiledExportFusionDetector,
            "detect",
            return_value=detection,
        ):
            result = FUSION_HELPER.compute_fusion_reward(
                "reference",
                "generated",
                fusion_weight=0.5,
                speedup=0.5,
                correctness=True,
            )

        self.assertAlmostEqual(result["compression_gain"], 1.0)
        self.assertAlmostEqual(result["compression_speed_factor"], 0.5)
        self.assertAlmostEqual(result["fusion_reward"], 0.25)

    def test_fusion_reward_caps_speedup_at_three(self):
        detection = FUSION_HELPER.FusionDetectionResult(
            detector="profiled_export",
            reference_plans=[{"semantics": ["relu", "sigmoid"]}],
            plan_alignments=[{"execution_group_counts": [1]}],
        )
        with patch.object(
            FUSION_HELPER.ProfiledExportFusionDetector,
            "detect",
            return_value=detection,
        ):
            result = FUSION_HELPER.compute_fusion_reward(
                "reference",
                "generated",
                fusion_weight=0.5,
                speedup=5.0,
                correctness=True,
            )

        self.assertAlmostEqual(result["compression_speed_factor"], 3.0)
        self.assertAlmostEqual(result["fusion_reward"], 1.5)


if __name__ == "__main__":
    unittest.main()
