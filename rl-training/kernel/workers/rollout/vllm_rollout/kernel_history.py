"""Bounded generation context; never mutate the stored full trajectory."""

from copy import deepcopy


def build_kernel_history(messages, response_turns, extract_kernel):
    """Keep all initial messages, latest kernel, and its subsequent feedback.

    response_turns records each assistant's position in the full conversation.
    Its first entry is the original-prompt boundary, including first-turn
    instructions. Extraction reuses KernelAgent's evaluation code selection.
    """
    if not response_turns:
        return deepcopy(messages)
    initial_end, latest_index = response_turns[0], response_turns[-1]
    if not (0 <= initial_end <= latest_index < len(messages)):
        raise ValueError("Invalid assistant positions for kernel history")
    latest = messages[latest_index]
    if latest["role"] != "assistant":
        raise ValueError("Latest response position must identify an assistant")
    code = extract_kernel(latest["content"])
    if code:
        code = code.strip()
        content = code if code.startswith("```") else f"```python\n{code}\n```"
    else:
        # Do not retain unbounded reasoning or silently reuse an older kernel.
        content = "No extractable kernel was produced in the previous attempt."
    return (
        deepcopy(messages[:initial_end])
        + [{"role": "assistant", "content": content}]
        + deepcopy(messages[latest_index + 1:])
    )
