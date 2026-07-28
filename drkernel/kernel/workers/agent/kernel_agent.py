import re

from verl_patch.workers.code.agent.base_agent import BaseAgent
from verl_patch.workers.code.agent_env.base_env import FinishReasonTypeEnum


class KernelAgent(BaseAgent):
    """
    Agent that supports multi-turn code generation, capable of handling code execution, self-test, and final answer extraction
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.answer_block_re = re.compile(
            r"""
            (?P<block>
                ```answer[ \t]*(?:\r?\n)?         # Opening fence: ```answer
                (?P<code>.*?)                     # Answer content
                (?:\r?\n)?```                     # Closing fence
            )
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )

        # Patterns borrowed from kernel/rewards/kernel_reward.py::extract_kernel_code
        kernel_markers = [
            r"#\s*Kernel\s+Implementation\s*\n(.*?)(?=\#\s*End\b|$)",
            r"```python\s*#\s*Kernel\s*\n(.*?)```",
            r"#\s*Your\s+implementation:\s*\n(.*?)(?=\#\s*End\b|$)",
            r"#\s*Generated\s+kernel:\s*\n(.*?)(?=\#\s*End\b|$)",
        ]
        self.kernel_code_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in kernel_markers]
        self.generic_code_block_re = re.compile(r"```(?:[\w+-]+)?\s*\n?(.*?)```", re.DOTALL)
        self.fence_open_re = re.compile(r"```[^\S\n]*[\w+-]*[^\S\n]*\n")
        # The evaluated kernel must define class {entry_point}New (entry point is
        # "Model" throughout the RL dataset, so match any *New class here).
        self.kernel_class_re = re.compile(r"\bclass\s+\w*New\b")

    async def generate_thought_and_action(
        self, response_token_ids: list[int], response_truncation: str
    ) -> tuple[str | None, str | None, bool | None, dict]:
        # remove padding token ids
        response_token_ids = [id for id in response_token_ids if id != self.tokenizer.pad_token_id]
        # translate result_token_id back to string
        response = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)

        if response is None:
            return None, None, None, True, {}

        # Always treat any extracted block as final answer; otherwise no tool call
        answer_block = self._extract_answer_block(response)
        if answer_block is not None:
            return response, response_token_ids, answer_block, True, {
                'finish_type': FinishReasonTypeEnum.ANSWER
            }

        python_code = self._extract_python_code(response)
        if python_code is not None:
            if python_code.strip().startswith("```"):
                code_block = python_code.strip()
            else:
                code_block = f"```python\n{python_code.strip()}\n```"
            return response, response_token_ids, code_block, True, {
                'finish_type': FinishReasonTypeEnum.ANSWER
            }

        return response, response_token_ids, None, True, {
            'finish_type': FinishReasonTypeEnum.NO_TOOL_CALL
        }

    def _extract_answer_block(self, response: str) -> str | None:
        match = self.answer_block_re.search(response)
        if match:
            return match.group("block")
        return None

    def _extract_python_code(self, response: str) -> str | None:
        code_blocks = self.generic_code_block_re.findall(response)
        # Recover a trailing unclosed fence (truncated response or missing ```).
        last_open = None
        for m in self.fence_open_re.finditer(response):
            last_open = m
        if last_open and "```" not in response[last_open.end():]:
            tail = response[last_open.end():].strip()
            if tail:
                code_blocks.append(tail)

        # Prefer the last block that defines the kernel class; a trailing usage
        # snippet or partial patch must not shadow the full implementation.
        for block in reversed(code_blocks):
            if self.kernel_class_re.search(block):
                return block.strip()

        for pattern in self.kernel_code_patterns:
            match = pattern.search(response)
            if match:
                return match.group(1).strip()

        if code_blocks:
            # Return the last discovered block, similar to kernel_reward.extract_kernel_code
            return code_blocks[-1].strip()

        return None
