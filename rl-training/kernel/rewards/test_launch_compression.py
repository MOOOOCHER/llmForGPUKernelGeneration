import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("fusion_helper.py")
SPEC = importlib.util.spec_from_file_location("launch_compression_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
compute_fusion_reward = MODULE.compute_fusion_reward


def metadata(reference_launches, generated_launches):
    return {
        "reference_profiling": {"kernel_launch_count": reference_launches},
        "profiling": {"kernel_launch_count": generated_launches},
    }


class LaunchCompressionRewardTest(unittest.TestCase):
    def test_complete_compression_is_one_for_different_chain_lengths(self):
        for reference_launches in (2, 3, 6):
            result = compute_fusion_reward(
                "",
                "",
                correctness=True,
                speedup=1.0,
                fusion_weight=1.0,
                profiler_metadata=metadata(reference_launches, 1),
            )
            self.assertEqual(result["compression_gain"], 1.0)
            self.assertEqual(result["fusion_reward"], 1.0)

    def test_partial_compression(self):
        result = compute_fusion_reward(
            "",
            "",
            correctness=True,
            speedup=2.0,
            fusion_weight=0.5,
            profiler_metadata=metadata(3, 2),
        )
        self.assertEqual(result["compression_gain"], 0.5)
        self.assertEqual(result["fusion_reward"], 0.5)

    def test_no_or_negative_compression_is_zero(self):
        for generated_launches in (3, 4):
            result = compute_fusion_reward(
                "",
                "",
                correctness=True,
                speedup=1.0,
                profiler_metadata=metadata(3, generated_launches),
            )
            self.assertEqual(result["compression_gain"], 0.0)
            self.assertEqual(result["fusion_reward"], 0.0)

    def test_sub_one_speedup_is_gated(self):
        result = compute_fusion_reward(
            "",
            "",
            correctness=True,
            speedup=0.99,
            fusion_weight=0.5,
            profiler_metadata=metadata(4, 1),
        )
        self.assertEqual(result["compression_speed_factor"], 0.99)
        self.assertEqual(result["fusion_reward"], 0.495)

    def test_above_one_speedup_uses_clipped_magnitude(self):
        result = compute_fusion_reward(
            "",
            "",
            correctness=True,
            speedup=9.0,
            fusion_weight=0.5,
            profiler_metadata=metadata(4, 1),
        )
        self.assertEqual(result["compression_speed_factor"], 1.0)
        self.assertEqual(result["fusion_reward"], 0.5)

    def test_incorrect_and_missing_profiles_are_neutral(self):
        incorrect = compute_fusion_reward(
            "", "", correctness=False, profiler_metadata=metadata(3, 1)
        )
        missing = compute_fusion_reward("", "", correctness=True, profiler_metadata={})
        self.assertEqual(incorrect["fusion_reward"], 0.0)
        self.assertEqual(incorrect["compression_skipped_reason"], "incorrect_kernel")
        self.assertEqual(missing["fusion_reward"], 0.0)
        self.assertEqual(
            missing["compression_skipped_reason"], "missing_reference_launch_count"
        )


if __name__ == "__main__":
    unittest.main()
