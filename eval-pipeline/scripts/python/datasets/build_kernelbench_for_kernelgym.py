#!/usr/bin/env python3
"""
Build KernelGYM grading datasets from local KernelBench problems.

The output parquet follows KernelGYM's RLHFDataset shape:
  - prompt: chat-style list of {role, content}
  - ground_truth: reference PyTorch source
  - entry_point: normally "Model"
  - reward_model: compatibility dict with ground_truth/style
  - extra_info: problem metadata used by rollout/reward paths

Example:
  KernelGYM/.venv/bin/python scripts/kernelbench_evaluation/build_kernelbench_for_kernelgym.py \
    --levels 1 2 3 \
    --backend triton \
    --prompt-option one_shot \
    --precision fp32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KERNELGYM_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = KERNELGYM_ROOT.parent
DEFAULT_KERNELBENCH_ROOT = KERNELGYM_ROOT / "kernelbench"
DEFAULT_OUTPUT_DIR = KERNELGYM_ROOT / "eval-pipeline" / "datasets"


def add_kernelbench_to_path(kernelbench_root: Path) -> None:
    src = kernelbench_root / "src"
    if not src.exists():
        raise FileNotFoundError(f"KernelBench src directory not found: {src}")
    sys.path.insert(0, str(src))

    # Importing kernelbench pulls the litellm/openai LLM-client chain, which
    # the drkernel venv doesn't ship. Only prompt construction and dataset
    # listing are needed here, so stub the client imports when absent instead
    # of requiring a second venv. tomli is stdlib tomllib on Python 3.11+.
    import types

    try:
        import litellm  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("litellm")
        stub.completion = None
        stub.text_completion = None
        sys.modules["litellm"] = stub
    try:
        import tomli  # noqa: F401
    except ModuleNotFoundError:
        import tomllib

        sys.modules["tomli"] = tomllib


def parse_problem_ids(raw_values: list[str] | None) -> list[int] | None:
    if not raw_values:
        return None
    parsed: list[int] = []
    for value in raw_values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.append(int(part))
    return sorted(set(parsed))


def parse_id_range(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    if ":" not in raw:
        raise ValueError("--id-range must use START:END, inclusive")
    start_s, end_s = raw.split(":", 1)
    start, end = int(start_s), int(end_s)
    if start > end:
        raise ValueError("--id-range START must be <= END")
    return start, end


def render_kernelbench_prompt(
    *,
    reference_code: str,
    backend: str,
    prompt_option: str,
    precision: str | None,
    custom_prompt_key: str | None,
    include_hardware: bool,
    gpu_name: str | None,
) -> str:
    from kernelbench.prompt_constructor_toml import get_custom_prompt, get_prompt_for_backend

    if custom_prompt_key:
        return get_custom_prompt(
            custom_prompt_key,
            ref_arch_src=reference_code,
            backend=backend,
            option=prompt_option,
            precision=precision,
            include_hardware=include_hardware,
            gpu_name=gpu_name,
        )

    return get_prompt_for_backend(
        ref_arch_src=reference_code,
        backend=backend,
        option=prompt_option,
        precision=precision,
        include_hardware=include_hardware,
        gpu_name=gpu_name,
    )


def build_rows(args: argparse.Namespace, level: int) -> list[dict[str, Any]]:
    from kernelbench.dataset import LocalKernelBenchDataset

    dataset = LocalKernelBenchDataset(
        level=level,
        base_path=str(args.kernelbench_root / "KernelBench"),
        problem_ids=args.problem_ids,
        id_range=args.id_range,
    )

    rows: list[dict[str, Any]] = []
    for problem in dataset:
        uid = f"kernelbench_level{level}_problem{problem.problem_id:04d}"
        prompt_text = render_kernelbench_prompt(
            reference_code=problem.code,
            backend=args.backend,
            prompt_option=args.prompt_option,
            precision=args.precision,
            custom_prompt_key=args.custom_prompt_key,
            include_hardware=args.include_hardware,
            gpu_name=args.gpu_name,
        )
        extra_info = {
            "source": "kernelbench",
            "data_source": "kernelbench",
            "uuid": uid,
            "uid": uid,
            "index": len(rows),
            "level": level,
            "problem_id": problem.problem_id,
            "problem_name": problem.name,
            "problem_hash": problem.hash,
            "path": os.path.relpath(problem.path or "", WORKSPACE_ROOT),
            "entry_point": args.entry_point,
            "backend": args.backend,
            "prompt_option": args.prompt_option,
            "custom_prompt_key": args.custom_prompt_key,
            "precision": args.precision,
            "include_hardware": args.include_hardware,
            "gpu_name": args.gpu_name,
        }
        reward_model = {
            "style": "rule",
            "ground_truth": problem.code,
            "entry_point": args.entry_point,
        }
        rows.append(
            {
                "uid": uid,
                "data_source": "kernelbench",
                "level": level,
                "problem_id": problem.problem_id,
                "problem_name": problem.name,
                "prompt": [{"role": "user", "content": prompt_text}],
                "ground_truth": problem.code,
                "entry_point": args.entry_point,
                "reward_model": reward_model,
                "extra_info": extra_info,
            }
        )
        if args.max_problems is not None and len(rows) >= args.max_problems:
            break

    return rows


def write_dataset(rows: list[dict[str, Any]], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe.to_parquet(output_path, index=False)


def write_metadata(
    *,
    args: argparse.Namespace,
    per_level_counts: dict[int, int],
    output_files: list[Path],
    combined_output: Path | None,
) -> Path:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "path_base": "repository_root",
        "script": os.path.relpath(__file__, KERNELGYM_ROOT),
        "kernelbench_root": os.path.relpath(args.kernelbench_root, KERNELGYM_ROOT),
        "output_dir": os.path.relpath(args.output_dir, KERNELGYM_ROOT),
        "levels": args.levels,
        "per_level_counts": per_level_counts,
        "total_rows": sum(per_level_counts.values()),
        "output_files": [os.path.relpath(path, KERNELGYM_ROOT) for path in output_files],
        "combined_output": os.path.relpath(combined_output, KERNELGYM_ROOT) if combined_output else None,
        "parameters": {
            "backend": args.backend,
            "prompt_option": args.prompt_option,
            "custom_prompt_key": args.custom_prompt_key,
            "precision": args.precision,
            "include_hardware": args.include_hardware,
            "gpu_name": args.gpu_name,
            "entry_point": args.entry_point,
            "problem_ids": args.problem_ids,
            "id_range": args.id_range,
            "max_problems": args.max_problems,
        },
    }
    metadata_path = args.output_dir / "kernelbench_for_kernelgym_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KernelGYM parquet datasets from local KernelBench problems."
    )
    parser.add_argument(
        "--kernelbench-root",
        type=Path,
        default=DEFAULT_KERNELBENCH_ROOT,
        help="Path to the KernelBench repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated parquet datasets.",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--backend", default="triton")
    parser.add_argument("--prompt-option", default="one_shot")
    parser.add_argument("--custom-prompt-key", default=None)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--include-hardware", action="store_true")
    parser.add_argument("--gpu-name", default=None)
    parser.add_argument("--entry-point", default="Model")
    parser.add_argument(
        "--problem-ids",
        nargs="*",
        default=None,
        help="Optional problem IDs, either space-separated or comma-separated.",
    )
    parser.add_argument(
        "--id-range",
        default=None,
        help="Optional inclusive range START:END.",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Optional cap per level for quick smoke-test datasets.",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Only write per-level parquet files.",
    )
    args = parser.parse_args()

    args.kernelbench_root = args.kernelbench_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.problem_ids = parse_problem_ids(args.problem_ids)
    args.id_range = parse_id_range(args.id_range)
    if args.include_hardware and not args.gpu_name:
        parser.error("--include-hardware requires --gpu-name")
    if args.max_problems is not None and args.max_problems < 1:
        parser.error("--max-problems must be positive")
    return args


def main() -> int:
    args = parse_args()
    add_kernelbench_to_path(args.kernelbench_root)

    all_rows: list[dict[str, Any]] = []
    per_level_counts: dict[int, int] = {}
    output_files: list[Path] = []

    for level in args.levels:
        rows = build_rows(args, level)
        per_level_counts[level] = len(rows)
        all_rows.extend(rows)
        output_path = args.output_dir / f"kernelbench_level{level}_for_kernelgym.parquet"
        write_dataset(rows, output_path)
        output_files.append(output_path)
        print(f"Wrote {len(rows)} rows: {output_path}")

    combined_output = None
    if not args.no_combined:
        combined_output = args.output_dir / "kernelbench_all_levels_for_kernelgym.parquet"
        write_dataset(all_rows, combined_output)
        print(f"Wrote {len(all_rows)} rows: {combined_output}")

    metadata_path = write_metadata(
        args=args,
        per_level_counts=per_level_counts,
        output_files=output_files,
        combined_output=combined_output,
    )
    print(f"Wrote metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
