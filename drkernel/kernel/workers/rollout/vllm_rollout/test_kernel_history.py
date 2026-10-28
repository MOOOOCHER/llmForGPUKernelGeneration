import unittest
import ast
import re
from pathlib import Path
from copy import deepcopy

from kernel_history import build_kernel_history


def extract(text):
    if "```python\n" in text:
        return text.split("```python\n", 1)[1].split("```", 1)[0]
    return None


class KernelHistoryTests(unittest.TestCase):
    def test_actual_agent_prefers_kernel_over_trailing_usage(self):
        agent_path = Path(__file__).with_name("kernel_agent.py")
        if not agent_path.exists():
            agent_path = Path(__file__).resolve().parents[2] / "agent" / "kernel_agent.py"
        tree = ast.parse(agent_path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KernelAgent")
        namespace = {"re": re, "BaseAgent": object}
        # __init__ only needs a base class accepting a tokenizer.
        class StubBase:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
        namespace["BaseAgent"] = StubBase
        exec(compile(ast.Module(body=[node], type_ignores=[]), "kernel_agent.py", "exec"), namespace)
        agent = namespace["KernelAgent"](None)
        response = "Reasoning\n```python\nclass ModelNew: pass\n```\n```python\nprint('usage')\n```"
        context = build_kernel_history([
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": response},
            {"role": "user", "content": "feedback"},
        ], [1], lambda text: agent._extract_answer_block(text) or agent._extract_python_code(text))
        self.assertEqual(context[1]["content"], "```python\nclass ModelNew: pass\n```")

    def test_selection_is_outside_prompt_rule_branch(self):
        tree = ast.parse(Path(__file__).with_name("vllm_async_engine.py").read_text())
        engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiTurnAsyncvLLMEngine")
        method = next(n for n in engine.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_process_single_turn")
        selection = [n for n in method.body if isinstance(n, ast.If) and "build_kernel_history" in ast.unparse(n)]
        self.assertEqual(len(selection), 1)
        self.assertIn("initial+last_kernel", ast.unparse(selection[0].test))

    def test_six_turns_preserve_prefix_and_only_latest_kernel_and_feedback(self):
        messages = [
            {"role": "system", "content": "SYSTEM and SKILLS"},
            {"role": "user", "content": "TASK and REFERENCE and INSTRUCTIONS"},
        ]
        positions = []
        for turn in range(6):
            original = deepcopy(messages)
            context = build_kernel_history(messages, positions, extract)
            self.assertEqual(messages, original)
            self.assertEqual(context[:2], messages[:2])
            self.assertEqual(len(context), 2 if turn == 0 else 4)
            if turn:
                self.assertEqual(context[2]["content"], f"```python\nkernel_{turn-1}\n```")
                self.assertEqual(context[3]["content"], f"feedback_{turn-1}")
                self.assertNotIn("REASONING", str(context))
            positions.append(len(messages))
            messages.extend([
                {"role": "assistant", "content": f"REASONING ```python\nkernel_{turn}\n```"},
                {"role": "user", "content": f"feedback_{turn}"},
            ])
        self.assertEqual(len(messages), 14)

    def test_missing_kernel_does_not_retain_reasoning(self):
        context = build_kernel_history([
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "reasoning only"},
            {"role": "user", "content": "compilation failed"},
        ], [1], extract)
        self.assertNotIn("reasoning only", str(context))
        self.assertEqual(context[-1]["content"], "compilation failed")

    def test_returns_independent_messages(self):
        messages = [{"role": "user", "content": "task"}]
        context = build_kernel_history(messages, [], extract)
        context[0]["content"] = "changed"
        self.assertEqual(messages[0]["content"], "task")

    def test_invalid_positions_fail_explicitly(self):
        with self.assertRaises(ValueError):
            build_kernel_history([{"role": "user", "content": "task"}], [1], extract)


if __name__ == "__main__":
    unittest.main()
