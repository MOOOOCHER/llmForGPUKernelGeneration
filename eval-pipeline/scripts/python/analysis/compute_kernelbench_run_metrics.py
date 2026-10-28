#!/usr/bin/env python3
"""Compute KernelBench paper-style metrics for a multi-level run."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_FAST_P = (1.0, 1.2, 1.5, 2.0)
DEFAULT_BASELINE_NAME = "torch.eager"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the flat generation_config.yaml files without a YAML dependency."""
    if not path.exists():
        return {}

    data: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("- ") and current_key is not None:
            current_value = data.setdefault(current_key, [])
            if isinstance(current_value, list):
                current_value.append(parse_scalar(line[2:]))
            continue
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_key = key
        data[key] = [] if value == "" else parse_scalar(value)
    return data


def file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_unix": stat.st_mtime,
    }


def select_relevant_parameters(config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "run_name",
        "model_name",
        "server_type",
        "backend",
        "dataset_name",
        "dataset_src",
        "level",
        "num_samples",
        "num_workers",
        "temperature",
        "max_tokens",
        "budget_tokens",
        "prompt_option",
        "custom_prompt_key",
        "precision",
        "hardware_gpu_name",
        "include_hardware_info",
        "check_kernel",
        "is_reasoning_model",
        "reasoning_effort",
        "store_type",
        "subset",
    ]
    return {key: config[key] for key in keys if key in config}


def problem_id_from_name(name: str) -> int | None:
    prefix = name.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def build_problem_name_map(kernelbench_root: Path, level: int) -> dict[int, str]:
    level_dir = kernelbench_root / "KernelBench" / f"level{level}"
    if not level_dir.exists():
        return {}

    mapping: dict[int, str] = {}
    for problem_file in sorted(level_dir.glob("*.py")):
        problem_id = problem_id_from_name(problem_file.name)
        if problem_id is not None:
            mapping[problem_id] = problem_file.name
    return mapping


def runtime_for_selection(entry: dict[str, Any]) -> float:
    runtime = entry.get("runtime", -1.0)
    try:
        runtime = float(runtime)
    except (TypeError, ValueError):
        return math.inf
    return runtime if runtime > 0 else math.inf


def positive_runtime(value: Any) -> float | None:
    try:
        runtime = float(value)
    except (TypeError, ValueError):
        return None
    return runtime if runtime > 0 else None


def sample_selection_key(entry: dict[str, Any]) -> tuple[int, int, float, int]:
    correct = bool(entry.get("correctness", False))
    compiled = bool(entry.get("compiled", False))
    runtime = runtime_for_selection(entry)
    is_decoy = bool(entry.get("_kernelgym_is_decoy", False))

    if not is_decoy and correct and runtime < math.inf:
        tier = 0
    elif not is_decoy and correct:
        tier = 1
    elif not is_decoy and compiled:
        tier = 2
    elif not is_decoy:
        tier = 3
    elif correct and runtime < math.inf:
        tier = 4
    elif correct:
        tier = 5
    elif compiled:
        tier = 6
    else:
        tier = 7

    sample_id = entry.get("sample_id", 0)
    try:
        sample_id = int(sample_id)
    except (TypeError, ValueError):
        sample_id = 0
    return tier, int(is_decoy), runtime, sample_id


def load_export_annotations(
    run_dir: Path,
) -> tuple[dict[tuple[int, int], dict[str, Any]], Path | None]:
    metadata_path = run_dir / "kernelgym_tts_export_metadata.json"
    if not metadata_path.exists():
        return {}, None

    metadata = load_json(metadata_path)
    annotations: dict[tuple[int, int], dict[str, Any]] = {}
    for item in metadata.get("exports", []):
        try:
            key = (
                int(item["kernelbench_problem_id"]),
                int(item["kernelbench_sample_id"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        is_decoy = item.get("selected_is_decoy")
        if is_decoy is None and item.get("selected_eval"):
            selected_eval_path = Path(item["selected_eval"])
            if selected_eval_path.exists():
                selected_eval = load_json(selected_eval_path)
                is_decoy = selected_eval.get("is_decoy_kernel", False)
        annotations[key] = {
            "is_decoy": bool(is_decoy),
            "kernelgym_correctness": bool(
                item.get("selected_correctness", False)
            ),
            "trajectory_id": item.get(
                "kernelgym_sample_id", key[1]
            ),
        }
    return annotations, metadata_path


def normalize_eval_results(
    raw_results: dict[str, Any],
    annotations: dict[tuple[int, int], dict[str, Any]] | None = None,
    *,
    prefer_non_decoy: bool = True,
) -> dict[int, dict[str, Any]]:
    normalized: dict[int, dict[str, Any]] = {}
    annotations = annotations or {}
    for problem_id_text, entries in raw_results.items():
        try:
            problem_id = int(problem_id_text)
        except ValueError:
            continue

        if isinstance(entries, list):
            candidates = [entry for entry in entries if isinstance(entry, dict)]
        elif isinstance(entries, dict):
            candidates = [entries]
        else:
            candidates = []

        candidates = [dict(entry) for entry in candidates]
        for entry in candidates:
            try:
                sample_id = int(entry.get("sample_id", 0))
            except (TypeError, ValueError):
                sample_id = 0
            annotation = annotations.get((problem_id, sample_id), {})
            entry["_kernelgym_is_decoy"] = bool(
                annotation.get("is_decoy", False)
            )
            entry["_kernelgym_correctness"] = annotation.get(
                "kernelgym_correctness"
            )

        if candidates:
            if prefer_non_decoy:
                selected = dict(min(candidates, key=sample_selection_key))
            else:
                selected = dict(
                    min(
                        candidates,
                        key=lambda entry: sample_selection_key(
                            {**entry, "_kernelgym_is_decoy": False}
                        ),
                    )
                )
            selected["_selected_sample_id"] = selected.get("sample_id", 0)
            selected["_available_sample_ids"] = sorted(
                {
                    entry.get("sample_id", 0)
                    for entry in candidates
                },
                key=lambda sample_id: str(sample_id),
            )
            selected["_num_evaluated_samples"] = len(candidates)
        else:
            selected = {}

        normalized[problem_id] = selected
    return normalized


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def geometric_mean(values: list[float]) -> float:
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        return 0.0
    return math.exp(sum(math.log(value) for value in positive_values) / len(positive_values))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def summarize_speedups(
    speedups: list[float],
    total_count: int,
    fast_p: tuple[float, ...],
) -> dict[str, Any]:
    fast_p_counts = {
        str(threshold): sum(1 for speedup in speedups if speedup > threshold)
        for threshold in fast_p
    }
    return {
        "fast_p": {
            str(threshold): safe_rate(count, total_count)
            for threshold, count in fast_p_counts.items()
        },
        "fast_p_counts": fast_p_counts,
        "geo_mean_speedup_correct": geometric_mean(speedups),
        "median_speedup_correct": median(speedups),
        "mean_speedup_correct": sum(speedups) / len(speedups) if speedups else 0.0,
        "max_speedup_correct": max(speedups) if speedups else 0.0,
        "speedup_count": len(speedups),
    }


def parse_named_baseline(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Comparison baselines must be formatted as NAME=PATH"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise argparse.ArgumentTypeError("Baseline name cannot be empty")
    if not path:
        raise argparse.ArgumentTypeError("Baseline path cannot be empty")
    return name, Path(path)


def compute_level_metrics(
    *,
    level: int,
    eval_file: Path,
    baseline_name: str,
    baselines: dict[str, dict[str, Any]],
    problem_names: dict[int, str],
    fast_p: tuple[float, ...],
) -> dict[str, Any]:
    run_dir = eval_file.parent
    config_file = run_dir / "generation_config.yaml"
    generation_config = load_simple_yaml(config_file)
    raw_eval_results = load_json(eval_file)
    if problem_names:
        raw_eval_results = {key: value for key, value in raw_eval_results.items()
                            if str(key).isdigit() and int(key) in problem_names}
    export_annotations, export_metadata_path = load_export_annotations(run_dir)
    eval_results = normalize_eval_results(raw_eval_results, export_annotations)
    sanitized_eval_results = normalize_eval_results(
        raw_eval_results,
        export_annotations,
        prefer_non_decoy=False,
    )
    baseline_levels = {
        name: results.get(f"level{level}", {}) for name, results in baselines.items()
    }
    primary_baseline_level = baseline_levels[baseline_name]

    problem_ids = sorted(problem_names) if problem_names else sorted(eval_results)
    total_count = len(problem_ids)
    evaluated_count = len(eval_results)

    compiled_count = 0
    correct_count = 0
    runnable_count = 0
    speedups: list[float] = []
    speedups_by_baseline: dict[str, list[float]] = {
        name: [] for name in baselines
    }
    correct_problem_ids: list[int] = []
    compiled_problem_ids: list[int] = []
    excessive_speedup_problem_ids: list[int] = []
    missing_baseline_problem_ids: list[int] = []
    missing_baseline_problem_ids_by_baseline: dict[str, list[int]] = {
        name: [] for name in baselines
    }
    selected_decoy_problem_ids: list[int] = []

    per_problem: dict[str, Any] = {}
    for problem_id in problem_ids:
        result = eval_results.get(problem_id, {})
        problem_name = problem_names.get(problem_id, str(problem_id))
        compiled = bool(result.get("compiled", False))
        correct = bool(result.get("correctness", False))
        runtime = result.get("runtime", -1.0)
        runtime = float(runtime) if isinstance(runtime, int | float) else -1.0
        metadata = result.get("metadata", {}) or {}
        excessive_speedup = bool(metadata.get("excessive_speedup", False))
        selected_is_decoy = bool(result.get("_kernelgym_is_decoy", False))
        if selected_is_decoy:
            selected_decoy_problem_ids.append(problem_id)
            correct = False

        if compiled:
            compiled_count += 1
            compiled_problem_ids.append(problem_id)
        if correct:
            correct_count += 1
            correct_problem_ids.append(problem_id)
        if excessive_speedup:
            excessive_speedup_problem_ids.append(problem_id)
        if runtime >= 0:
            runnable_count += 1

        baseline_runtimes: dict[str, float | None] = {}
        baseline_runtime_sources: dict[str, str | None] = {}
        speedups_for_problem: dict[str, float | None] = {}
        for name, baseline_level in baseline_levels.items():
            baseline_entry = baseline_level.get(problem_name)
            baseline_runtime = None
            baseline_runtime_source = None
            if name == baseline_name:
                baseline_runtime = positive_runtime(result.get("ref_runtime"))
                if baseline_runtime is not None:
                    baseline_runtime_source = "paired_eager"
            elif name == "torch.compile":
                baseline_runtime = positive_runtime(
                    result.get("ref_compile_runtime")
                )
                if baseline_runtime is not None:
                    baseline_runtime_source = "paired_torch_compile"

            speedup = None
            if baseline_runtime is None and baseline_entry is not None:
                baseline_runtime = float(baseline_entry.get("mean", 0.0))
                baseline_runtime_source = "historical_baseline"
            elif baseline_runtime is None and problem_id in eval_results:
                missing_baseline_problem_ids_by_baseline[name].append(problem_id)

            if correct and runtime > 0 and baseline_runtime and baseline_runtime > 0:
                speedup = baseline_runtime / runtime
                speedups_by_baseline[name].append(speedup)
            baseline_runtimes[name] = baseline_runtime
            baseline_runtime_sources[name] = baseline_runtime_source
            speedups_for_problem[name] = speedup

        baseline_runtime = baseline_runtimes[baseline_name]
        speedup = speedups_for_problem[baseline_name]
        if problem_id in missing_baseline_problem_ids_by_baseline[baseline_name]:
            missing_baseline_problem_ids.append(problem_id)
        if speedup is not None:
            speedups.append(speedup)

        per_problem[str(problem_id)] = {
            "problem_name": problem_name,
            "selected_sample_id": result.get("_selected_sample_id"),
            "available_sample_ids": result.get("_available_sample_ids", []),
            "num_evaluated_samples": result.get("_num_evaluated_samples", 0),
            "compiled": compiled,
            "correctness": correct,
            "kernelbench_sanitized_correctness": bool(
                sanitized_eval_results.get(problem_id, {}).get(
                    "correctness", False
                )
            ),
            "selected_is_decoy": selected_is_decoy,
            "selected_kernelgym_correctness": result.get(
                "_kernelgym_correctness"
            ),
            "runtime": runtime,
            "excessive_speedup": excessive_speedup,
            "speedup_excluded_from_aggregates": False,
            "baseline_runtime": baseline_runtime,
            "baseline_runtimes": baseline_runtimes,
            "baseline_runtime_sources": baseline_runtime_sources,
            "paired_ref_runtime": positive_runtime(result.get("ref_runtime")),
            "paired_ref_compile_runtime": positive_runtime(
                result.get("ref_compile_runtime")
            ),
            "speedup": speedup,
            "speedups": speedups_for_problem,
            "error_name": (
                metadata.get("runtime_error_name")
                or metadata.get("compilation_error_name")
                or metadata.get("error")
            ),
        }

    sanitized_correct_problem_ids = sorted(
        problem_id
        for problem_id, result in sanitized_eval_results.items()
        if bool(result.get("correctness", False))
    )

    trajectory_total = 0
    trajectory_decoy_count = 0
    trajectory_correct_count = 0
    trajectory_sanitized_correct_count = 0
    trajectory_speedups: list[float] = []
    trajectory_sanitized_speedups: list[float] = []
    trajectory_baseline_speedups: dict[int, dict[str, list[float]]] = {}
    for problem_id_text, entries in raw_eval_results.items():
        try:
            problem_id = int(problem_id_text)
        except ValueError:
            continue
        candidates = entries if isinstance(entries, list) else [entries]
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            trajectory_total += 1
            try:
                sample_id = int(entry.get("sample_id", 0))
            except (TypeError, ValueError):
                sample_id = 0
            is_decoy = bool(
                export_annotations.get((problem_id, sample_id), {}).get(
                    "is_decoy", False
                )
            )
            trajectory_id = export_annotations.get(
                (problem_id, sample_id), {}
            ).get("trajectory_id", sample_id)
            try:
                trajectory_id = int(trajectory_id)
            except (TypeError, ValueError):
                trajectory_id = sample_id
            per_baseline_speedups = trajectory_baseline_speedups.setdefault(
                trajectory_id,
                {name: [] for name in baselines},
            )
            trajectory_decoy_count += int(is_decoy)
            sanitized_correct = bool(entry.get("correctness", False))
            policy_correct = sanitized_correct and not is_decoy
            trajectory_sanitized_correct_count += int(sanitized_correct)
            trajectory_correct_count += int(policy_correct)
            runtime = positive_runtime(entry.get("runtime"))
            baseline_runtime = positive_runtime(entry.get("ref_runtime"))
            if runtime and baseline_runtime:
                speedup = baseline_runtime / runtime
                if sanitized_correct:
                    trajectory_sanitized_speedups.append(speedup)
                if policy_correct:
                    trajectory_speedups.append(speedup)
            runtime = positive_runtime(entry.get("runtime"))
            metadata = entry.get("metadata", {}) or {}
            problem_name = problem_names.get(problem_id, str(problem_id))
            for name, baseline_level in baseline_levels.items():
                baseline_runtime = None
                if name == baseline_name:
                    baseline_runtime = positive_runtime(
                        entry.get("ref_runtime")
                    )
                elif name == "torch.compile":
                    baseline_runtime = positive_runtime(
                        entry.get("ref_compile_runtime")
                    )
                if baseline_runtime is None:
                    baseline_entry = baseline_level.get(problem_name)
                    if baseline_entry is not None:
                        baseline_runtime = positive_runtime(
                            baseline_entry.get("mean")
                        )
                if (
                    policy_correct
                    and runtime
                    and baseline_runtime
                ):
                    per_baseline_speedups[name].append(
                        baseline_runtime / runtime
                    )

    per_trajectory_baseline_metrics = {
        str(trajectory_id): {
            name: summarize_speedups(values, total_count, fast_p)
            for name, values in per_baseline.items()
        }
        for trajectory_id, per_baseline in sorted(
            trajectory_baseline_speedups.items()
        )
    }
    trajectory_ids = sorted(per_trajectory_baseline_metrics)
    trajectory_average_baseline_metrics: dict[str, Any] = {}
    for name in baselines:
        summaries = [
            per_trajectory_baseline_metrics[trajectory_id][name]
            for trajectory_id in trajectory_ids
        ]
        if not summaries:
            trajectory_average_baseline_metrics[name] = summarize_speedups(
                [], total_count, fast_p
            )
            continue
        trajectory_average_baseline_metrics[name] = {
            "fast_p": {
                str(threshold): sum(
                    summary["fast_p"][str(threshold)]
                    for summary in summaries
                )
                / len(summaries)
                for threshold in fast_p
            },
            "fast_p_counts": {
                str(threshold): sum(
                    summary["fast_p_counts"][str(threshold)]
                    for summary in summaries
                )
                / len(summaries)
                for threshold in fast_p
            },
            "geo_mean_speedup_correct": sum(
                summary["geo_mean_speedup_correct"]
                for summary in summaries
            )
            / len(summaries),
            "median_speedup_correct": sum(
                summary["median_speedup_correct"]
                for summary in summaries
            )
            / len(summaries),
            "mean_speedup_correct": sum(
                summary["mean_speedup_correct"]
                for summary in summaries
            )
            / len(summaries),
            "max_speedup_correct": sum(
                summary["max_speedup_correct"]
                for summary in summaries
            )
            / len(summaries),
            "speedup_count": sum(
                summary["speedup_count"] for summary in summaries
            )
            / len(summaries),
        }

    return {
        "level": level,
        "run_dir": str(run_dir),
        "eval_file": str(eval_file),
        "files": {
            "eval_results": file_metadata(eval_file),
            "generation_config": file_metadata(config_file) if config_file.exists() else None,
            "kernelgym_tts_export_metadata": (
                file_metadata(export_metadata_path)
                if export_metadata_path
                else None
            ),
        },
        "generation_config": generation_config,
        "relevant_parameters": select_relevant_parameters(generation_config),
        "sample_selection": "fastest_enroot_correct_non_decoy",
        "decoy_policy": (
            "exported_for_trajectory_accounting_but_failed_for_selection_"
            "and_policy_metrics"
        ),
        "total_count": total_count,
        "evaluated_count": evaluated_count,
        "compiled_count": compiled_count,
        "correct_count": correct_count,
        "runnable_count": runnable_count,
        "compilation_rate": safe_rate(compiled_count, total_count),
        "correctness_rate": safe_rate(correct_count, total_count),
        "runnable_rate": safe_rate(runnable_count, total_count),
        **summarize_speedups(speedups, total_count, fast_p),
        "baseline_metrics": {
            name: summarize_speedups(values, total_count, fast_p)
            for name, values in speedups_by_baseline.items()
        },
        "correct_problem_ids": correct_problem_ids,
        "selected_decoy_count": len(selected_decoy_problem_ids),
        "selected_decoy_problem_ids": sorted(selected_decoy_problem_ids),
        "kernelbench_sanitized": {
            "correct_count": len(sanitized_correct_problem_ids),
            "correctness_rate": safe_rate(
                len(sanitized_correct_problem_ids), total_count
            ),
            "correct_problem_ids": sanitized_correct_problem_ids,
        },
        "trajectory_mean": {
            "sample_count": trajectory_total,
            "decoy_count": trajectory_decoy_count,
            "correct_count": trajectory_correct_count,
            "correctness_rate": safe_rate(
                trajectory_correct_count, trajectory_total
            ),
            **summarize_speedups(
                trajectory_speedups, trajectory_total, fast_p
            ),
        },
        "kernelbench_sanitized_trajectory_mean": {
            "sample_count": trajectory_total,
            "correct_count": trajectory_sanitized_correct_count,
            "correctness_rate": safe_rate(
                trajectory_sanitized_correct_count, trajectory_total
            ),
            **summarize_speedups(
                trajectory_sanitized_speedups,
                trajectory_total,
                fast_p,
            ),
        },
        "trajectory_average_baseline_metrics": {
            "aggregation": "arithmetic_mean_of_per_trajectory_metrics",
            "num_trajectories": len(trajectory_ids),
            "trajectory_ids": trajectory_ids,
            "baselines": trajectory_average_baseline_metrics,
            "per_trajectory": per_trajectory_baseline_metrics,
        },
        "compiled_problem_ids": compiled_problem_ids,
        "excessive_speedup_count": len(excessive_speedup_problem_ids),
        "excessive_speedup_problem_ids": excessive_speedup_problem_ids,
        "missing_baseline_problem_ids": sorted(set(missing_baseline_problem_ids)),
        "missing_baseline_problem_ids_by_baseline": {
            name: sorted(set(problem_ids))
            for name, problem_ids in missing_baseline_problem_ids_by_baseline.items()
        },
        "per_problem": per_problem,
    }


def infer_level_from_dir(path: Path) -> int | None:
    for token in path.name.replace("-", "_").split("_"):
        if token.startswith("level"):
            suffix = token.removeprefix("level")
            if suffix.isdigit():
                return int(suffix)
    return None


def find_level_eval_files(run_root: Path) -> dict[int, Path]:
    eval_files: dict[int, Path] = {}
    for eval_file in sorted(run_root.glob("*level*/eval_results.json")):
        level = infer_level_from_dir(eval_file.parent)
        if level is not None:
            eval_files[level] = eval_file
    return eval_files


def write_markdown(path: Path, metrics: dict[str, Any], fast_p: tuple[float, ...]) -> None:
    lines = [
        f"# KernelBench Metrics: {metrics['run_name']}",
        "",
        f"- Run root: `{metrics['run_root']}`",
        f"- Primary baseline: `{metrics['baseline_name']}` (`{metrics['baseline_file']}`)",
        f"- Working directory: `{metrics['working_directory']}`",
        "",
        "| Level | Tasks | Compiled | Correct | Fast@1 | Fast@1.2 | Fast@1.5 | Fast@2 | GeoMean Speedup (Correct Only) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for level in sorted(metrics["levels"], key=int):
        level_metrics = metrics["levels"][level]
        fast_1 = level_metrics["fast_p"].get("1.0", 0.0)
        fast_12 = level_metrics["fast_p"].get("1.2", 0.0)
        fast_15 = level_metrics["fast_p"].get("1.5", 0.0)
        fast_2 = level_metrics["fast_p"].get("2.0", 0.0)
        lines.append(
            "| {level} | {total} | {compiled} ({compiled_rate:.2%}) | "
            "{correct} ({correct_rate:.2%}) | {fast_1:.2%} | {fast_12:.2%} | "
            "{fast_15:.2%} | {fast_2:.2%} | {geo:.4f} |".format(
                level=level,
                total=level_metrics["total_count"],
                compiled=level_metrics["compiled_count"],
                compiled_rate=level_metrics["compilation_rate"],
                correct=level_metrics["correct_count"],
                correct_rate=level_metrics["correctness_rate"],
                fast_1=fast_1,
                fast_12=fast_12,
                fast_15=fast_15,
                fast_2=fast_2,
                geo=level_metrics["geo_mean_speedup_correct"],
            )
        )

    overall = metrics["overall"]
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Tasks: {overall['total_count']}",
            f"- Compiled: {overall['compiled_count']} ({overall['compilation_rate']:.2%})",
            f"- Correct: {overall['correct_count']} ({overall['correctness_rate']:.2%})",
            f"- Excessive-speedup flags (included; manual review): {overall['excessive_speedup_count']}",
        ]
    )
    for threshold in fast_p:
        key = str(threshold)
        lines.append(f"- Fast@{threshold:g}: {overall['fast_p'][key]:.2%}")
    lines.append(f"- Correct-sample geometric mean speedup: {overall['geo_mean_speedup_correct']:.4f}")
    lines.append(
        "- Speedup aggregates include all policy-correct non-decoy kernels; "
        "excessive-speedup flags do not exclude results. Fast@ rates use all tasks."
    )
    lines.append("")

    lines.extend(
        [
            "## Decoy Policy and Trajectory Mean",
            "",
            "Decoys remain exported for trajectory accounting, but count as failures in "
            "policy metrics and cannot win Best-of-N selection over a non-decoy candidate. "
            "Sanitized columns show standalone KernelBench functional results before that policy filter.",
            "",
            "| Level | Policy Best Correct | Sanitized Best Correct | Exported Decoys | "
            "Mean-of-N Correct | Sanitized Mean-of-N Correct | Mean-of-N Fast@1 | "
            "Mean-of-N GeoMean |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for level in sorted(metrics["levels"], key=int):
        level_metrics = metrics["levels"][level]
        trajectory = level_metrics["trajectory_mean"]
        sanitized_trajectory = level_metrics[
            "kernelbench_sanitized_trajectory_mean"
        ]
        sanitized = level_metrics["kernelbench_sanitized"]
        lines.append(
            "| {level} | {policy_correct}/{tasks} | {sanitized_correct}/{tasks} | "
            "{decoys} | {mean_correct}/{samples} ({mean_rate:.2%}) | "
            "{sanitized_mean_correct}/{samples} ({sanitized_mean_rate:.2%}) | "
            "{fast_1:.2%} | {geo:.4f} |".format(
                level=level,
                policy_correct=level_metrics["correct_count"],
                sanitized_correct=sanitized["correct_count"],
                tasks=level_metrics["total_count"],
                decoys=trajectory["decoy_count"],
                mean_correct=trajectory["correct_count"],
                sanitized_mean_correct=sanitized_trajectory["correct_count"],
                samples=trajectory["sample_count"],
                mean_rate=trajectory["correctness_rate"],
                sanitized_mean_rate=sanitized_trajectory["correctness_rate"],
                fast_1=trajectory["fast_p"].get("1.0", 0.0),
                geo=trajectory["geo_mean_speedup_correct"],
            )
        )
    lines.append("")

    if len(metrics.get("baselines", {})) > 1:
        lines.extend(
            [
                "## Baseline Comparison",
                "",
                "| Scope | Baseline | Fast@1 | Fast@1.2 | Fast@1.5 | Fast@2 | GeoMean Speedup (Correct Only) | Median Speedup (Correct Only) | Mean Speedup (Correct Only) | Max Speedup (Correct Only) | Clean Correct Kernels |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for level in sorted(metrics["levels"], key=int):
            level_metrics = metrics["levels"][level]
            for baseline in metrics["baselines"]:
                baseline_metrics = level_metrics["baseline_metrics"][baseline]
                lines.append(
                    "| Level {level} | `{baseline}` | {fast_1:.2%} | {fast_12:.2%} | "
                    "{fast_15:.2%} | {fast_2:.2%} | {geo:.4f} | {median:.4f} | "
                    "{mean:.4f} | {max_speedup:.4f} | {samples} |".format(
                        level=level,
                        baseline=baseline,
                        fast_1=baseline_metrics["fast_p"].get("1.0", 0.0),
                        fast_12=baseline_metrics["fast_p"].get("1.2", 0.0),
                        fast_15=baseline_metrics["fast_p"].get("1.5", 0.0),
                        fast_2=baseline_metrics["fast_p"].get("2.0", 0.0),
                        geo=baseline_metrics["geo_mean_speedup_correct"],
                        median=baseline_metrics["median_speedup_correct"],
                        mean=baseline_metrics["mean_speedup_correct"],
                        max_speedup=baseline_metrics["max_speedup_correct"],
                        samples=baseline_metrics["speedup_count"],
                    )
                )
        for baseline in metrics["baselines"]:
            baseline_metrics = overall["baseline_metrics"][baseline]
            lines.append(
                "| Overall | `{baseline}` | {fast_1:.2%} | {fast_12:.2%} | "
                "{fast_15:.2%} | {fast_2:.2%} | {geo:.4f} | {median:.4f} | "
                "{mean:.4f} | {max_speedup:.4f} | {samples} |".format(
                    baseline=baseline,
                    fast_1=baseline_metrics["fast_p"].get("1.0", 0.0),
                    fast_12=baseline_metrics["fast_p"].get("1.2", 0.0),
                    fast_15=baseline_metrics["fast_p"].get("1.5", 0.0),
                    fast_2=baseline_metrics["fast_p"].get("2.0", 0.0),
                    geo=baseline_metrics["geo_mean_speedup_correct"],
                    median=baseline_metrics["median_speedup_correct"],
                    mean=baseline_metrics["mean_speedup_correct"],
                    max_speedup=baseline_metrics["max_speedup_correct"],
                    samples=baseline_metrics["speedup_count"],
                )
            )
        lines.append("")

        lines.extend(
            [
                "## Trajectory-Average Baseline Comparison",
                "",
                "Each trajectory is scored independently over the full level. "
                "Every value below is the arithmetic mean of the corresponding "
                "per-trajectory metric; decoys count as failures.",
                "",
                "| Scope | Baseline | Fast@1 | Fast@1.2 | Fast@1.5 | Fast@2 | GeoMean Speedup (Correct Only) | Median Speedup (Correct Only) | Mean Speedup (Correct Only) | Max Speedup (Correct Only) | Clean Correct Kernels |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        sorted_levels = sorted(metrics["levels"], key=int)
        for level in sorted_levels:
            trajectory_metrics = metrics["levels"][level][
                "trajectory_average_baseline_metrics"
            ]
            for baseline in metrics["baselines"]:
                baseline_metrics = trajectory_metrics["baselines"][baseline]
                lines.append(
                    "| Level {level} | `{baseline}` | {fast_1:.2%} | {fast_12:.2%} | "
                    "{fast_15:.2%} | {fast_2:.2%} | {geo:.4f} | {median:.4f} | "
                    "{mean:.4f} | {max_speedup:.4f} | {samples:.2f} |".format(
                        level=level,
                        baseline=baseline,
                        fast_1=baseline_metrics["fast_p"].get("1.0", 0.0),
                        fast_12=baseline_metrics["fast_p"].get("1.2", 0.0),
                        fast_15=baseline_metrics["fast_p"].get("1.5", 0.0),
                        fast_2=baseline_metrics["fast_p"].get("2.0", 0.0),
                        geo=baseline_metrics["geo_mean_speedup_correct"],
                        median=baseline_metrics["median_speedup_correct"],
                        mean=baseline_metrics["mean_speedup_correct"],
                        max_speedup=baseline_metrics["max_speedup_correct"],
                        samples=baseline_metrics["speedup_count"],
                    )
                )
        if len(sorted_levels) == 1:
            only_level = metrics["levels"][sorted_levels[0]][
                "trajectory_average_baseline_metrics"
            ]
            for baseline in metrics["baselines"]:
                baseline_metrics = only_level["baselines"][baseline]
                lines.append(
                    "| Overall | `{baseline}` | {fast_1:.2%} | {fast_12:.2%} | "
                    "{fast_15:.2%} | {fast_2:.2%} | {geo:.4f} | {median:.4f} | "
                    "{mean:.4f} | {max_speedup:.4f} | {samples:.2f} |".format(
                        baseline=baseline,
                        fast_1=baseline_metrics["fast_p"].get("1.0", 0.0),
                        fast_12=baseline_metrics["fast_p"].get("1.2", 0.0),
                        fast_15=baseline_metrics["fast_p"].get("1.5", 0.0),
                        fast_2=baseline_metrics["fast_p"].get("2.0", 0.0),
                        geo=baseline_metrics["geo_mean_speedup_correct"],
                        median=baseline_metrics["median_speedup_correct"],
                        mean=baseline_metrics["mean_speedup_correct"],
                        max_speedup=baseline_metrics["max_speedup_correct"],
                        samples=baseline_metrics["speedup_count"],
                    )
                )
        lines.append("")

    lines.extend(["## Run Parameters", ""])
    for level in sorted(metrics["levels"], key=int):
        params = metrics["levels"][level]["relevant_parameters"]
        lines.append(f"### Level {level}")
        for key, value in params.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def compute_overall(levels: dict[str, Any], fast_p: tuple[float, ...]) -> dict[str, Any]:
    total_count = sum(level["total_count"] for level in levels.values())
    compiled_count = sum(level["compiled_count"] for level in levels.values())
    correct_count = sum(level["correct_count"] for level in levels.values())
    runnable_count = sum(level["runnable_count"] for level in levels.values())
    excessive_speedup_count = sum(
        level["excessive_speedup_count"] for level in levels.values()
    )
    speedups = [
        problem["speedup"]
        for level in levels.values()
        for problem in level["per_problem"].values()
        if problem["speedup"] is not None
    ]
    baseline_names = next(iter(levels.values()))["baseline_metrics"].keys() if levels else []
    speedups_by_baseline = {
        name: [
            problem["speedups"].get(name)
            for level in levels.values()
            for problem in level["per_problem"].values()
            if problem["speedups"].get(name) is not None
        ]
        for name in baseline_names
    }

    return {
        "total_count": total_count,
        "compiled_count": compiled_count,
        "correct_count": correct_count,
        "runnable_count": runnable_count,
        "excessive_speedup_count": excessive_speedup_count,
        "compilation_rate": safe_rate(compiled_count, total_count),
        "correctness_rate": safe_rate(correct_count, total_count),
        "runnable_rate": safe_rate(runnable_count, total_count),
        **summarize_speedups(speedups, total_count, fast_p),
        "baseline_metrics": {
            name: summarize_speedups(values, total_count, fast_p)
            for name, values in speedups_by_baseline.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("KernelBench/runs/drkernel_sft_coldstart_h100"),
        help="KernelBench run root containing per-level run directories.",
    )
    parser.add_argument(
        "--kernelbench-root",
        type=Path,
        default=Path("KernelBench"),
        help="KernelBench repository root.",
    )
    parser.add_argument(
        "--baseline-name",
        default=DEFAULT_BASELINE_NAME,
        help="Name for the primary baseline. Existing top-level speedup fields use this baseline.",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("KernelBench/results/timing/H100_Modal/baseline_time_torch.json"),
        help="Baseline timing JSON used to compute speedups and Fast@p.",
    )
    parser.add_argument(
        "--extra-baseline",
        action="append",
        type=parse_named_baseline,
        default=[],
        metavar="NAME=PATH",
        help="Additional baseline timing JSON to include in comparison metrics.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "results/kernelbench/evaluation_results/sft_coldstart/"
            "kernelbench_drsft_coldstart_h100_metrics.json"
        ),
        help="Where to write metrics JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "results/kernelbench/evaluation_results/sft_coldstart/"
            "kernelbench_drsft_coldstart_h100_metrics.md"
        ),
        help="Where to write a Markdown summary.",
    )
    parser.add_argument(
        "--fast-p",
        type=float,
        nargs="+",
        default=list(DEFAULT_FAST_P),
        help="Speedup thresholds for Fast@p.",
    )
    parser.add_argument("--problem-ids", type=int, nargs="+", default=None,
                        help="Restrict metrics to these problem IDs for a partial evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fast_p = tuple(args.fast_p)
    baseline_paths = {args.baseline_name: args.baseline_file}
    for name, path in args.extra_baseline:
        if name in baseline_paths:
            raise SystemExit(f"Duplicate baseline name: {name}")
        baseline_paths[name] = path

    baselines = {
        name: load_json(path)
        for name, path in baseline_paths.items()
    }
    eval_files = find_level_eval_files(args.run_root)
    if not eval_files:
        raise SystemExit(f"No per-level eval_results.json files found under {args.run_root}")

    levels: dict[str, Any] = {}
    for level, eval_file in sorted(eval_files.items()):
        problem_names = build_problem_name_map(args.kernelbench_root, level)
        if args.problem_ids is not None:
            unknown = set(args.problem_ids) - set(problem_names)
            if unknown:
                raise SystemExit(f"Unknown problem IDs for level {level}: {sorted(unknown)}")
            problem_names = {key: problem_names[key] for key in args.problem_ids}
        levels[str(level)] = compute_level_metrics(
            level=level,
            eval_file=eval_file,
            baseline_name=args.baseline_name,
            baselines=baselines,
            problem_names=problem_names,
            fast_p=fast_p,
        )

    metrics = {
        "run_name": args.run_root.name,
        "run_root": str(args.run_root),
        "working_directory": os.getcwd(),
        "problem_ids": args.problem_ids,
        "baseline_name": args.baseline_name,
        "baseline_file": str(args.baseline_file),
        "baseline_metadata": file_metadata(args.baseline_file),
        "baselines": {
            name: {
                "path": str(path),
                "metadata": file_metadata(path),
                "primary": name == args.baseline_name,
            }
            for name, path in baseline_paths.items()
        },
        "fast_p_thresholds": list(fast_p),
        "levels": levels,
        "overall": compute_overall(levels, fast_p),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_json, metrics)
    write_markdown(args.output_md, metrics, fast_p)

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
