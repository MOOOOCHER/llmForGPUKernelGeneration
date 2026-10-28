#!/usr/bin/env python3
"""Summarize KernelGYM TTS compile/correct/speedup metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def fmt_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.4f}"


def pick(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = as_float(metrics.get(key))
        if value is not None:
            return value
    return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_problem_rows(run_dir: Path) -> list[dict[str, Any]]:
    # Historical quirk: this file is JSONL even though it has a .parquet suffix.
    results_path = run_dir / "graded_results.parquet"
    if not results_path.exists():
        return []
    rows = []
    with results_path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_best_per_problem_from_eval_outputs(run_dir: Path) -> dict[str, Any] | None:
    eval_root = run_dir / "eval_outputs"
    if not eval_root.exists():
        return None

    problem_rows = read_problem_rows(run_dir)
    problems_by_id = {int(row["problem_id"]): row for row in problem_rows if row.get("problem_id") is not None}
    best_by_problem: dict[int, tuple[tuple[bool, float, bool], dict[str, Any]]] = {}

    for eval_path in sorted(eval_root.glob("level_*_problem_*_sample_*/turn_*_eval.json")):
        try:
            row = json.loads(eval_path.read_text())
        except json.JSONDecodeError:
            continue

        problem_id = row.get("problem_id")
        if problem_id is None:
            continue
        problem_id = int(problem_id)
        problem = problems_by_id.get(problem_id, {})
        sample_id = row.get("sample_id")
        turn = row.get("turn_id")

        candidate = {
            "problem_index": problem_id - 1,
            "problem_id": problem_id,
            "problem_uid": problem.get("uid") or f"kernelbench_level{row.get('level')}_problem{problem_id:04d}",
            "problem_name": problem.get("problem_name"),
            "problem_path": (problem.get("extra_info") or {}).get("path"),
            "sample_uid": row.get("uid"),
            "sample_id": sample_id,
            "turn": int(turn or 0),
            "compiled": parse_bool(row.get("compilation")),
            "correct": parse_bool(row.get("correctness")) or parse_bool(row.get("success")),
            "speedup": parse_float(row.get("performance")),
            "speedup_positive": parse_bool(row.get("is_speedup_positive")),
            "eval_path": str(eval_path.relative_to(run_dir)),
        }
        key = (candidate["correct"], candidate["speedup"], candidate["compiled"])
        current = best_by_problem.get(problem_id)
        if current is None or key > current[0]:
            best_by_problem[problem_id] = (key, candidate)

    selected = [best_by_problem[index][1] for index in sorted(best_by_problem)]
    if not selected:
        return None

    count = len(selected)
    correct = sum(item["correct"] for item in selected)
    compiled = sum(item["compiled"] for item in selected)
    speedup_positive = sum(item["speedup_positive"] for item in selected)
    speedups = [item["speedup"] for item in selected]
    correct_speedups = [item["speedup"] for item in selected if item["correct"]]
    turn_counts = Counter(item["turn"] for item in selected)

    return {
        "selection_rule": "best candidate per problem across all samples and turns by correctness, then speedup, then compilation (from eval_outputs)",
        "count": count,
        "compilation_rate": compiled / count,
        "correctness_rate": correct / count,
        "mean_speedup_all": sum(speedups) / count,
        "mean_speedup_correct_only": (sum(correct_speedups) / len(correct_speedups)) if correct_speedups else None,
        "max_speedup": max(speedups),
        "speedup_positive_rate": speedup_positive / count,
        "selected_turn_counts": dict(sorted(turn_counts.items())),
        "num_problem_rows": len(problem_rows) if problem_rows else count,
        "top_speedups": sorted(selected, key=lambda item: item["speedup"], reverse=True)[:10],
    }


def summarize_best_per_problem(run_dir: Path) -> dict[str, Any] | None:
    eval_summary = summarize_best_per_problem_from_eval_outputs(run_dir)
    if eval_summary is not None:
        return eval_summary

    conversations_path = run_dir / "graded_results_conversations_conversations.jsonl"
    problem_rows = read_problem_rows(run_dir)
    if not conversations_path.exists():
        return None
    if not problem_rows:
        return None

    num_problems = len(problem_rows)
    best_by_problem: dict[int, tuple[tuple[bool, float, bool], dict[str, Any]]] = {}
    with conversations_path.open() as handle:
        for sample_index, line in enumerate(handle):
            if not line.strip():
                continue
            problem_index = sample_index % num_problems
            problem = problem_rows[problem_index]
            sample = json.loads(line)
            for turn in sample.get("turns", []):
                metrics = turn.get("metrics") or {}
                candidate = {
                    "problem_index": problem_index,
                    "problem_id": problem.get("problem_id"),
                    "problem_uid": problem.get("uid"),
                    "problem_name": problem.get("problem_name"),
                    "problem_path": (problem.get("extra_info") or {}).get("path"),
                    "sample_uid": sample.get("uid"),
                    "sample_id": None,
                    "sample_index": sample_index,
                    "turn": int(turn.get("turn_id", 0) or 0),
                    "compiled": parse_bool(metrics.get("compilation")),
                    "correct": parse_bool(metrics.get("correctness")) or parse_bool(metrics.get("success")),
                    "speedup": parse_float(metrics.get("performance")),
                    "speedup_positive": parse_bool(metrics.get("is_speedup_positive")),
                }
                key = (candidate["correct"], candidate["speedup"], candidate["compiled"])
                current = best_by_problem.get(problem_index)
                if current is None or key > current[0]:
                    best_by_problem[problem_index] = (key, candidate)

    selected = [best_by_problem[index][1] for index in sorted(best_by_problem)]
    if not selected:
        return None

    count = len(selected)
    correct = sum(item["correct"] for item in selected)
    compiled = sum(item["compiled"] for item in selected)
    speedup_positive = sum(item["speedup_positive"] for item in selected)
    speedups = [item["speedup"] for item in selected]
    correct_speedups = [item["speedup"] for item in selected if item["correct"]]
    turn_counts = Counter(item["turn"] for item in selected)

    return {
        "selection_rule": "best candidate per problem across all samples and turns by correctness, then speedup, then compilation",
        "count": count,
        "compilation_rate": compiled / count,
        "correctness_rate": correct / count,
        "mean_speedup_all": sum(speedups) / count,
        "mean_speedup_correct_only": (sum(correct_speedups) / len(correct_speedups)) if correct_speedups else None,
        "max_speedup": max(speedups),
        "speedup_positive_rate": speedup_positive / count,
        "selected_turn_counts": dict(sorted(turn_counts.items())),
        "num_problem_rows": num_problems,
        "top_speedups": sorted(selected, key=lambda item: item["speedup"], reverse=True)[:10],
    }


def build_summary(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.json: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    overall = {
        "score": pick(metrics, "val/test_score/kernelbench"),
        "pass_at_1": pick(metrics, "val/test_score/kernelbench_pass@1"),
        "compilation_rate": pick(
            metrics,
            "val/test_score_extra/compilation_kernelbench",
            "val/test_score_extra/compilation_tensor_kernelbench",
        ),
        "correctness_rate": pick(
            metrics,
            "val/test_score_extra/correctness_kernelbench",
            "val/test_score_extra/correctness_tensor_kernelbench",
        ),
        "success_rate": pick(metrics, "val/test_score_extra/success_kernelbench"),
        "mean_speedup_reward": pick(
            metrics,
            "val/test_score_extra/performance_kernelbench",
            "val/test_score_extra/performance_tensor_kernelbench",
        ),
        "speedup_positive_rate": pick(metrics, "val/test_score_extra/is_speedup_positive_kernelbench"),
        "num_samples": pick(metrics, "val/kernel/data/unique_samples"),
        "valid_turns": pick(metrics, "val/kernel/data/valid_turns"),
        "total_turns": pick(metrics, "val/kernel/data/total_actual_turns"),
    }

    turns: list[dict[str, Any]] = []
    turn = 1
    while f"val/kernel/turn_{turn}/count" in metrics:
        turns.append(
            {
                "turn": turn,
                "count": pick(metrics, f"val/kernel/turn_{turn}/count"),
                "compilation_rate": pick(metrics, f"val/kernel/turn_{turn}/compilation_rate"),
                "correctness_rate": pick(metrics, f"val/kernel/turn_{turn}/correctness_rate"),
                "mean_speedup_correct_only": pick(metrics, f"val/kernel/turn_{turn}/mean_performance"),
                "mean_speedup_all": pick(metrics, f"val/kernel/turn_{turn}/mean_performance_in_all"),
                "max_speedup": pick(metrics, f"val/kernel/turn_{turn}/max_performance"),
                "fast_at_1_all": pick(metrics, f"val/kernel/turn_{turn}/fast@1_in_all"),
                "fast_at_1_2_all": pick(metrics, f"val/kernel/turn_{turn}/fast@1.2_in_all"),
                "fast_at_1_5_all": pick(metrics, f"val/kernel/turn_{turn}/fast@1.5_in_all"),
                "fast_at_2_all": pick(metrics, f"val/kernel/turn_{turn}/fast@2_in_all"),
            }
        )
        turn += 1

    best_turn = None
    if turns:
        best_turn = max(turns, key=lambda item: item.get("correctness_rate") or -1.0)

    best_per_problem = summarize_best_per_problem(run_dir)
    if best_per_problem is None and turns:
        final_best = turns[-1]
        best_per_problem = {
            "selection_rule": "fallback from metrics val/kernel/best_by_turn_N",
            "count": final_best["count"],
            "compilation_rate": None,
            "correctness_rate": pick(metrics, f"val/kernel/best_by_turn_{final_best['turn']}/correctness_rate"),
            "mean_speedup_all": pick(metrics, f"val/kernel/best_by_turn_{final_best['turn']}/mean_performance_in_all"),
            "mean_speedup_correct_only": pick(metrics, f"val/kernel/best_by_turn_{final_best['turn']}/mean_performance"),
            "max_speedup": pick(metrics, f"val/kernel/best_by_turn_{final_best['turn']}/max_performance"),
            "speedup_positive_rate": pick(metrics, f"val/kernel/best_by_turn_{final_best['turn']}/fast@1_in_all"),
            "selected_turn_counts": None,
        }

    return {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "best_per_problem": best_per_problem,
        "overall": overall,
        "turns": turns,
        "best_turn_by_correctness": best_turn,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    best_per_problem = summary.get("best_per_problem") or {}
    lines = [
        "# KernelGYM TTS Summary",
        "",
        f"Run: `{summary['run_dir']}`",
        "",
        "## Best Per Problem",
        "",
        f"Selection: {best_per_problem.get('selection_rule', 'n/a')}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| problems | {fmt_num(best_per_problem.get('count'))} |",
        f"| compilation | {fmt_pct(best_per_problem.get('compilation_rate'))} |",
        f"| correctness | {fmt_pct(best_per_problem.get('correctness_rate'))} |",
        f"| mean speedup all | {fmt_num(best_per_problem.get('mean_speedup_all'))} |",
        f"| mean speedup correct-only | {fmt_num(best_per_problem.get('mean_speedup_correct_only'))} |",
        f"| max speedup | {fmt_num(best_per_problem.get('max_speedup'))} |",
        f"| speedup-positive | {fmt_pct(best_per_problem.get('speedup_positive_rate'))} |",
        "",
        "## Highest Speedup Problems",
        "",
        "| rank | problem | name | sample | turn | compiled | correct | speedup | eval |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(best_per_problem.get("top_speedups") or [], start=1):
        sample_label = item.get("sample_id")
        if sample_label is None:
            sample_label = item.get("sample_uid")
        elif item.get("sample_uid"):
            sample_label = f"{sample_label} ({item.get('sample_uid')})"
        lines.append(
            "| {rank} | {problem} | {name} | {sample} | {turn} | {compiled} | {correct} | {speedup} | {eval_path} |".format(
                rank=rank,
                problem=item.get("problem_uid") or item.get("problem_id"),
                name=item.get("problem_name") or "",
                sample=sample_label,
                turn=item.get("turn"),
                compiled="yes" if item.get("compiled") else "no",
                correct="yes" if item.get("correct") else "no",
                speedup=fmt_num(item.get("speedup")),
                eval_path=item.get("eval_path") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| score | {fmt_num(overall['score'])} |",
            f"| pass@1 | {fmt_pct(overall['pass_at_1'])} |",
            f"| compilation | {fmt_pct(overall['compilation_rate'])} |",
            f"| correctness | {fmt_pct(overall['correctness_rate'])} |",
            f"| success | {fmt_pct(overall['success_rate'])} |",
            f"| mean speedup reward | {fmt_num(overall['mean_speedup_reward'])} |",
            f"| speedup-positive | {fmt_pct(overall['speedup_positive_rate'])} |",
            f"| samples | {fmt_num(overall['num_samples'])} |",
            f"| valid turns | {fmt_num(overall['valid_turns'])} / {fmt_num(overall['total_turns'])} |",
            "",
            "## Per Turn",
            "",
            "| turn | compile | correct | mean speedup all | mean speedup correct-only | max speedup | fast@1 all | fast@1.2 all | fast@1.5 all | fast@2 all |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for turn in summary["turns"]:
        lines.append(
            "| {turn} | {compile} | {correct} | {mean_all} | {mean_correct} | {max_speedup} | {fast1} | {fast12} | {fast15} | {fast2} |".format(
                turn=turn["turn"],
                compile=fmt_pct(turn["compilation_rate"]),
                correct=fmt_pct(turn["correctness_rate"]),
                mean_all=fmt_num(turn["mean_speedup_all"]),
                mean_correct=fmt_num(turn["mean_speedup_correct_only"]),
                max_speedup=fmt_num(turn["max_speedup"]),
                fast1=fmt_pct(turn["fast_at_1_all"]),
                fast12=fmt_pct(turn["fast_at_1_2_all"]),
                fast15=fmt_pct(turn["fast_at_1_5_all"]),
                fast2=fmt_pct(turn["fast_at_2_all"]),
            )
        )
    best_turn = summary.get("best_turn_by_correctness")
    if best_turn:
        lines.extend(
            [
                "",
                "## Best Turn",
                "",
                f"Best turn by correctness: `{best_turn['turn']}` "
                f"({fmt_pct(best_turn['correctness_rate'])} correct, "
                f"{fmt_pct(best_turn['compilation_rate'])} compiled, "
                f"{fmt_num(best_turn['mean_speedup_all'])} mean speedup over all samples).",
            ]
        )
    turn_counts = best_per_problem.get("selected_turn_counts")
    if turn_counts:
        lines.extend(
            [
                "",
                "Best-per-problem selected turn counts: "
                + ", ".join(f"turn {turn}: {count}" for turn, count in turn_counts.items()),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="KernelGYM TTS output directory containing metrics.json.")
    parser.add_argument("--json", type=Path, help="Output JSON path. Defaults to RUN_DIR/tts_summary.json.")
    parser.add_argument("--md", type=Path, help="Output Markdown path. Defaults to RUN_DIR/tts_summary.md.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary = build_summary(run_dir)
    json_path = args.json or (run_dir / "tts_summary.json")
    md_path = args.md or (run_dir / "tts_summary.md")

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(summary))

    overall = summary["overall"]
    best_per_problem = summary.get("best_per_problem") or {}
    print("KernelGYM TTS Summary")
    print("=====================")
    print(f"Run: {run_dir}")
    print("Best per problem:")
    print(f"  Compilation:       {fmt_pct(best_per_problem.get('compilation_rate'))}")
    print(f"  Correctness:       {fmt_pct(best_per_problem.get('correctness_rate'))}")
    print(f"  Mean speedup all:  {fmt_num(best_per_problem.get('mean_speedup_all'))}")
    print(f"  Speedup-positive:  {fmt_pct(best_per_problem.get('speedup_positive_rate'))}")
    print("Overall all turns:")
    print(f"  Compilation:       {fmt_pct(overall['compilation_rate'])}")
    print(f"  Correctness:       {fmt_pct(overall['correctness_rate'])}")
    print(f"  Pass@1:            {fmt_pct(overall['pass_at_1'])}")
    print(f"  Mean speedup:      {fmt_num(overall['mean_speedup_reward'])}")
    print(f"  Speedup-positive:  {fmt_pct(overall['speedup_positive_rate'])}")
    best_turn = summary.get("best_turn_by_correctness")
    if best_turn:
        print(
            f"Best turn:         {best_turn['turn']} "
            f"({fmt_pct(best_turn['correctness_rate'])} correct, "
            f"{fmt_num(best_turn['mean_speedup_all'])} mean speedup all)"
        )
    print(f"Wrote JSON:        {json_path}")
    print(f"Wrote Markdown:    {md_path}")


if __name__ == "__main__":
    main()
