#!/usr/bin/env python3
"""
Export KernelGYM multi-turn TTS kernels into a KernelBench-compatible run directory.

KernelGYM stores turn artifacts like:
  eval_outputs/level_1_problem_1_sample_0/turn_1_kernel.py

KernelBench expects generated kernels directly under a run directory shaped like:
  KernelBench/runs/<run_name>/level_<level>_problem_<problem_id>_sample_<sample_id>_kernel.py

By default this script stages exports under KernelBench/runs/__exported_kernels__, while
the KernelBench evaluator writes evaluated runs directly under KernelBench/runs.
Use --output-root to override that location.

This script uses the KernelGYM input parquet to map zero-based KernelGYM
problem indices back to KernelBench levels/problem IDs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KERNELGYM_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = KERNELGYM_ROOT.parent
DEFAULT_KERNELBENCH_ROOT = KERNELGYM_ROOT / "kernelbench"
DEFAULT_OUTPUT_ROOT = DEFAULT_KERNELBENCH_ROOT / "runs"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "Results" / "Kernelbench" / "tts_runs"
DEFAULT_DATASET = (
    KERNELGYM_ROOT
    / "eval-pipeline"
    / "datasets"
    / "kernelbench_level1_for_kernelgym.parquet"
)


@dataclass(frozen=True)
class DatasetRow:
    index: int
    level: int
    problem_id: int
    uid: str
    problem_name: str
    extra_info: dict[str, Any]


@dataclass(frozen=True)
class Candidate:
    turn: int
    kernel_path: Path
    eval_path: Path | None
    eval_data: dict[str, Any]
    sample_id: int | None = None


def load_dataset_map(dataset_path: Path) -> dict[int, DatasetRow]:
    import pandas as pd

    dataframe = pd.read_parquet(dataset_path)
    mapping: dict[int, DatasetRow] = {}

    for row_index, row in dataframe.reset_index(drop=True).iterrows():
        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        if not isinstance(extra_info, dict):
            extra_info = {}

        kg_index = int(extra_info.get("index", row_index))
        level = int(extra_info.get("level", row.get("level")))
        problem_id = int(extra_info.get("problem_id", row.get("problem_id")))
        mapping[kg_index] = DatasetRow(
            index=kg_index,
            level=level,
            problem_id=problem_id,
            uid=str(row.get("uid", extra_info.get("uid", ""))),
            problem_name=str(row.get("problem_name", extra_info.get("problem_name", ""))),
            extra_info=extra_info,
        )

    return mapping


def load_dataset_level_problem_map(dataset_path: Path) -> dict[tuple[int, int], DatasetRow]:
    rows_by_index = load_dataset_map(dataset_path)
    return {
        (row.level, row.problem_id): row
        for row in rows_by_index.values()
    }


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def discover_candidates(problem_dir: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for kernel_path in sorted(problem_dir.glob("turn_*_kernel.py")):
        match = re.fullmatch(r"turn_(\d+)_kernel\.py", kernel_path.name)
        if not match:
            continue
        turn = int(match.group(1))
        eval_path = problem_dir / f"turn_{turn}_eval.json"
        candidates.append(
            Candidate(
                turn=turn,
                kernel_path=kernel_path,
                eval_path=eval_path if eval_path.exists() else None,
                eval_data=load_json(eval_path),
            )
        )
    return candidates


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate_score(candidate: Candidate) -> float:
    data = candidate.eval_data
    if "score" in data:
        return numeric(data.get("score"))
    return numeric(data.get("performance"))


def candidate_is_decoy(candidate: Candidate) -> bool:
    return bool(candidate.eval_data.get("is_decoy_kernel", False))


def choose_candidate(candidates: list[Candidate], mode: str) -> Candidate:
    if not candidates:
        raise ValueError("No turn kernels found")

    if mode == "last":
        return max(candidates, key=lambda item: item.turn)

    if mode == "best_score":
        return max(
            candidates,
            key=lambda item: (
                not candidate_is_decoy(item),
                candidate_score(item),
                bool(item.eval_data.get("correctness", False)),
                bool(item.eval_data.get("compilation", False)),
                numeric(item.eval_data.get("performance")),
                item.turn,
            ),
        )

    if mode == "best_correct":
        return max(
            candidates,
            key=lambda item: (
                bool(item.eval_data.get("correctness", False)),
                not candidate_is_decoy(item),
                numeric(item.eval_data.get("performance")),
                candidate_score(item),
                bool(item.eval_data.get("compilation", False)),
                item.turn,
            ),
        )

    if mode == "best_correct_or_compiled":
        return max(
            candidates,
            key=lambda item: (
                bool(item.eval_data.get("correctness", False)),
                bool(item.eval_data.get("compilation", False)),
                not candidate_is_decoy(item),
                numeric(item.eval_data.get("performance")),
                candidate_score(item),
                item.turn,
            ),
        )

    if mode == "best_compiled":
        return max(
            candidates,
            key=lambda item: (
                bool(item.eval_data.get("compilation", False)),
                bool(item.eval_data.get("correctness", False)),
                not candidate_is_decoy(item),
                numeric(item.eval_data.get("performance")),
                candidate_score(item),
                item.turn,
            ),
        )

    raise ValueError(f"Unknown selection mode: {mode}")


def choose_candidates(
    candidates: list[Candidate],
    mode: str,
    count: int,
) -> list[Candidate]:
    remaining = list(candidates)
    selected: list[Candidate] = []
    while remaining and len(selected) < count:
        candidate = choose_candidate(remaining, mode)
        selected.append(candidate)
        remaining.remove(candidate)

    if not selected:
        raise ValueError("No turn kernels found")

    while len(selected) < count:
        selected.append(selected[-1])
    return selected


def parse_problem_dir_name(path: Path) -> tuple[int | None, int, int]:
    match = re.fullmatch(r"level_(\d+)_problem_(\d+)_sample_(\d+)", path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))

    match = re.fullmatch(r"problem_(\d+)_sample_(\d+)", path.name)
    if not match:
        raise ValueError(f"Unexpected KernelGYM problem directory name: {path.name}")
    return None, int(match.group(1)), int(match.group(2))


def export_run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_map = load_dataset_map(args.dataset)
    dataset_level_problem_map = load_dataset_level_problem_map(args.dataset)
    eval_outputs = args.kernelgym_run / "eval_outputs"
    if not eval_outputs.exists():
        raise FileNotFoundError(f"eval_outputs not found: {eval_outputs}")

    output_run_dir = args.output_root / args.output_run_name
    output_run_dir.mkdir(parents=True, exist_ok=True)

    exports: list[dict[str, Any]] = []
    problem_dirs = [
        path for path in eval_outputs.iterdir()
        if path.is_dir()
        and (
            re.fullmatch(r"level_\d+_problem_\d+_sample_\d+", path.name)
            or re.fullmatch(r"problem_\d+_sample_\d+", path.name)
        )
    ]

    grouped_exports: dict[tuple[int, int], list[tuple[DatasetRow, int, int, Candidate]]] = {}

    for problem_dir in sorted(problem_dirs):
        if not problem_dir.is_dir():
            continue

        kg_level, kg_problem_or_index, kg_sample_id = parse_problem_dir_name(problem_dir)
        if kg_level is not None:
            dataset_row = dataset_level_problem_map.get((kg_level, kg_problem_or_index))
            if dataset_row is None:
                raise KeyError(
                    f"KernelGYM level/problem {(kg_level, kg_problem_or_index)} is missing from dataset map {args.dataset}"
                )
            kg_problem_index = dataset_row.index
        elif kg_problem_or_index not in dataset_map:
            raise KeyError(
                f"KernelGYM problem index {kg_problem_or_index} is missing from dataset map {args.dataset}"
            )
        else:
            kg_problem_index = kg_problem_or_index
            dataset_row = dataset_map[kg_problem_index]

        candidates = [
            Candidate(
                turn=candidate.turn,
                kernel_path=candidate.kernel_path,
                eval_path=candidate.eval_path,
                eval_data=candidate.eval_data,
                sample_id=kg_sample_id,
            )
            for candidate in discover_candidates(problem_dir)
        ]

        if args.best_per_problem:
            grouped_exports.setdefault(
                (dataset_row.level, dataset_row.problem_id),
                [],
            ).extend(
                (dataset_row, kg_problem_index, kg_sample_id, candidate)
                for candidate in candidates
            )
            continue

        selected_candidates = choose_candidates(
            candidates,
            args.select,
            args.turns_per_trajectory,
        )
        for selection_rank, selected in enumerate(selected_candidates):
            if args.preserve_sample_ids:
                sample_id = (
                    kg_sample_id * args.turns_per_trajectory + selection_rank
                )
            else:
                sample_id = selection_rank
            output_name = (
                f"level_{dataset_row.level}_problem_{dataset_row.problem_id}_"
                f"sample_{sample_id}_kernel.py"
            )
            output_path = output_run_dir / output_name
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing kernel: {output_path}"
                )
            shutil.copyfile(selected.kernel_path, output_path)

            exports.append(
                {
                    "kernelbench_level": dataset_row.level,
                    "kernelbench_problem_id": dataset_row.problem_id,
                    "kernelbench_problem_name": dataset_row.problem_name,
                    "kernelbench_sample_id": sample_id,
                    "kernelbench_output": str(output_path.relative_to(WORKSPACE_ROOT)),
                    "kernelgym_problem_index": kg_problem_index,
                    "kernelgym_sample_id": kg_sample_id,
                    "kernelgym_uid": dataset_row.uid,
                    "trajectory_selection_rank": selection_rank,
                    "selection_mode": args.select,
                    "selected_turn": selected.turn,
                    "selected_kernel": str(selected.kernel_path.relative_to(WORKSPACE_ROOT)),
                    "selected_eval": str(selected.eval_path.relative_to(WORKSPACE_ROOT)) if selected.eval_path else None,
                    "selected_correctness": selected.eval_data.get("correctness"),
                    "selected_compilation": selected.eval_data.get("compilation"),
                    "selected_success": selected.eval_data.get("success"),
                    "selected_is_decoy": candidate_is_decoy(selected),
                    "selected_performance": selected.eval_data.get("performance"),
                    "selected_score": selected.eval_data.get("score"),
                    "available_turns": [
                        {
                            "turn": candidate.turn,
                            "correctness": candidate.eval_data.get("correctness"),
                            "compilation": candidate.eval_data.get("compilation"),
                            "is_decoy": candidate_is_decoy(candidate),
                            "performance": candidate.eval_data.get("performance"),
                            "score": candidate.eval_data.get("score"),
                        }
                        for candidate in candidates
                    ],
                }
            )

    if args.best_per_problem:
        for (_level, _problem_id), grouped_candidates in sorted(grouped_exports.items()):
            dataset_row = grouped_candidates[0][0]
            selected = choose_candidate(
                [candidate for (_row, _index, _sample_id, candidate) in grouped_candidates],
                args.select,
            )
            selected_row, kg_problem_index, kg_sample_id, _ = next(
                item for item in grouped_candidates if item[3] == selected
            )
            dataset_row = selected_row
            sample_id = 0
            output_name = (
                f"level_{dataset_row.level}_problem_{dataset_row.problem_id}_"
                f"sample_{sample_id}_kernel.py"
            )
            output_path = output_run_dir / output_name
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing kernel: {output_path}")
            shutil.copyfile(selected.kernel_path, output_path)

            exports.append(
                {
                    "kernelbench_level": dataset_row.level,
                    "kernelbench_problem_id": dataset_row.problem_id,
                    "kernelbench_problem_name": dataset_row.problem_name,
                    "kernelbench_sample_id": sample_id,
                    "kernelbench_output": str(output_path.relative_to(WORKSPACE_ROOT)),
                    "kernelgym_problem_index": kg_problem_index,
                    "kernelgym_sample_id": kg_sample_id,
                    "kernelgym_uid": dataset_row.uid,
                    "selection_mode": args.select,
                    "selected_turn": selected.turn,
                    "selected_kernel": str(selected.kernel_path.relative_to(WORKSPACE_ROOT)),
                    "selected_eval": str(selected.eval_path.relative_to(WORKSPACE_ROOT)) if selected.eval_path else None,
                    "selected_correctness": selected.eval_data.get("correctness"),
                    "selected_compilation": selected.eval_data.get("compilation"),
                    "selected_success": selected.eval_data.get("success"),
                    "selected_is_decoy": candidate_is_decoy(selected),
                    "selected_performance": selected.eval_data.get("performance"),
                    "selected_score": selected.eval_data.get("score"),
                    "available_turns": [
                        {
                            "sample_id": candidate.sample_id,
                            "turn": candidate.turn,
                            "correctness": candidate.eval_data.get("correctness"),
                            "compilation": candidate.eval_data.get("compilation"),
                            "is_decoy": candidate_is_decoy(candidate),
                            "performance": candidate.eval_data.get("performance"),
                            "score": candidate.eval_data.get("score"),
                        }
                        for (_row, _index, _sample, candidate) in grouped_candidates
                    ],
                }
            )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(WORKSPACE_ROOT)),
        "kernelgym_run": str(args.kernelgym_run.relative_to(WORKSPACE_ROOT)),
        "dataset": str(args.dataset.relative_to(WORKSPACE_ROOT)),
        "kernelbench_root": str(args.kernelbench_root.relative_to(WORKSPACE_ROOT)),
        "output_root": str(args.output_root.relative_to(WORKSPACE_ROOT)),
        "output_run_name": args.output_run_name,
        "output_run_dir": str(output_run_dir.relative_to(WORKSPACE_ROOT)),
        "selection_mode": args.select,
        "turns_per_trajectory": args.turns_per_trajectory,
        "preserve_sample_ids": args.preserve_sample_ids,
        "best_per_problem": args.best_per_problem,
        "num_exported": len(exports),
        "num_selected_decoys": sum(
            bool(item.get("selected_is_decoy", False)) for item in exports
        ),
        "num_selected_kernelgym_rejected": sum(
            not bool(item.get("selected_correctness", False)) for item in exports
        ),
        "exports": exports,
    }

    metadata_path = output_run_dir / "kernelgym_tts_export_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export KernelGYM TTS turn kernels into a KernelBench run directory."
    )
    parser.add_argument("--kernelgym-run", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--kernelbench-root", type=Path, default=DEFAULT_KERNELBENCH_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will contain <output-run-name>. Defaults to KernelBench/runs/__exported_kernels__.",
    )
    parser.add_argument(
        "--output-run-name",
        default=None,
        help="KernelBench run name. Defaults to <kernelgym_run_name>_<select>.",
    )
    parser.add_argument(
        "--select",
        choices=["last", "best_correct", "best_correct_or_compiled", "best_score", "best_compiled"],
        default="best_correct",
    )
    parser.add_argument(
        "--preserve-sample-ids",
        action="store_true",
        help="Use KernelGYM sample IDs in KernelBench filenames. Default rewrites selected sample as sample_0.",
    )
    parser.add_argument(
        "--best-per-problem",
        action="store_true",
        help="Select one best kernel across all samples and turns for each KernelBench problem.",
    )
    parser.add_argument(
        "--turns-per-trajectory",
        type=int,
        default=1,
        help="Number of ranked turns to export from each trajectory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.kernelgym_run = args.kernelgym_run.resolve()
    args.dataset = args.dataset.resolve()
    args.kernelbench_root = args.kernelbench_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.output_run_name is None:
        args.output_run_name = f"{args.kernelgym_run.name}_{args.select}"
    if args.turns_per_trajectory < 1:
        parser.error("--turns-per-trajectory must be at least 1")
    if args.best_per_problem and args.turns_per_trajectory != 1:
        parser.error("--best-per-problem requires --turns-per-trajectory=1")
    return args


def main() -> int:
    args = parse_args()
    metadata = export_run(args)
    print(
        f"Exported {metadata['num_exported']} kernel(s) to "
        f"{metadata['output_run_dir']} using selection={metadata['selection_mode']}"
    )
    print(f"Metadata: {metadata['output_run_dir']}/kernelgym_tts_export_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
