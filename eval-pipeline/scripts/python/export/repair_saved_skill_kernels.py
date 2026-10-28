#!/usr/bin/env python3
"""Repair saved turn kernels using the reward path's canonical code extractor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def assistant_response(state_path: Path) -> str:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    messages = state.get("messages", [])
    responses = [
        message.get("content", "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not responses:
        raise ValueError(f"No assistant response in {state_path}")
    return responses[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--davinci-code-root", type=Path, required=True)
    parser.add_argument("--eval-outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.davinci_code_root))
    from kernel.rewards.kernel_reward import extract_kernel_code

    checked = changed = missing_model = errors = 0
    for root in args.eval_outputs:
        for state_path in sorted(root.glob("problem_*_sample_*/turn_*_state.json")):
            checked += 1
            kernel_path = state_path.with_name(
                state_path.name.replace("_state.json", "_kernel.py")
            )
            try:
                code = extract_kernel_code(assistant_response(state_path)).strip()
                if "class ModelNew" not in code:
                    missing_model += 1
                old_code = (
                    kernel_path.read_text(encoding="utf-8").strip()
                    if kernel_path.exists()
                    else ""
                )
                if code != old_code:
                    changed += 1
                    if args.apply:
                        kernel_path.write_text(code + "\n", encoding="utf-8")
            except Exception as exc:
                errors += 1
                print(f"ERROR {state_path}: {exc}", file=sys.stderr)

    print(
        json.dumps(
            {
                "checked": checked,
                "changed": changed,
                "missing_modelnew_after_extraction": missing_model,
                "errors": errors,
                "applied": args.apply,
            },
            sort_keys=True,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
