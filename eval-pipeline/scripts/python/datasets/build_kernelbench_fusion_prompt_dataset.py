#!/usr/bin/env python3
"""Append compact DefragmentationFramework fusion plans to KernelBench prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--defrag-root", type=Path, required=True)
    return parser.parse_args()


def compact_plans(source: str) -> tuple[list[dict], str | None]:
    try:
        from scripts.ast_based.program_trace import parse_program
        from scripts.common.defrag_plan import plan_defragmentation
        from scripts.common.fragment_analyze import analyze_fragments
        from scripts.common.semantic_lift import lift_graph

        graph = parse_program(source, mode="defrag").to_dict()
        lifted = lift_graph(graph)
        fragments = analyze_fragments(lifted)
        payload = plan_defragmentation(fragments, lifted, fusion_only=True)
        return payload.get("plans", []), None
    except Exception as exc:  # Preserve the dataset and record planner failures.
        return [], f"{type(exc).__name__}: {exc}"


def guidance(plans: list[dict]) -> str:
    lines = [
        "Fusion-plan guidance from the DefragmentationFramework static analysis:",
    ]
    if plans:
        for plan in plans:
            semantics = " -> ".join(plan.get("semantics", [])) or "unknown"
            lines.append(f"- {plan.get('action', 'fuse')}: {semantics}")
    else:
        lines.append("- No non-library fusion opportunity was identified.")
    lines.extend(
        [
            "Treat these as optimization candidates, not correctness assumptions.",
            "Do not replace an isolated convolution or linear operator merely to claim fusion; optimized library calls may be faster.",
            "Preserve exact FP32 semantics, including IEEE precision for tl.dot reductions.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.defrag_root))

    table = pq.read_table(args.input)
    rows = table.to_pylist()
    stats = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "rows_with_plans": 0,
        "plan_count": 0,
        "planner_failures": [],
    }

    for row in rows:
        plans, error = compact_plans(row["ground_truth"])
        if plans:
            stats["rows_with_plans"] += 1
            stats["plan_count"] += len(plans)
        if error:
            stats["planner_failures"].append(
                {"uid": row.get("uid"), "error": error}
            )

        messages = [dict(message) for message in row["prompt"]]
        target = next(
            (message for message in reversed(messages) if message.get("role") == "user"),
            None,
        )
        if target is None:
            raise ValueError(f"row {row.get('uid')} has no user prompt")
        target["content"] = target["content"].rstrip() + "\n\n" + guidance(plans)
        row["prompt"] = messages

        extra = dict(row.get("extra_info") or {})
        extra["fusion_prompt_injected"] = True
        extra["fusion_plans"] = plans
        if error:
            extra["fusion_plan_error"] = error
        row["extra_info"] = extra

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), args.output)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
