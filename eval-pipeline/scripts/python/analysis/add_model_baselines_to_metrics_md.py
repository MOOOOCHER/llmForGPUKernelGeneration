#!/usr/bin/env python3
"""Append matching DaVinci/DrKernel baseline rows to a KernelBench metrics Markdown file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_level(path: Path, level: str) -> dict:
    candidate = path / f"l{level}/kernelbench_metrics.json" if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    data = json.loads(candidate.read_text())
    levels = data.get("levels", {})
    if level in levels:
        return levels[level]
    if len(levels) == 1:
        return next(iter(levels.values()))
    raise KeyError(f"level {level} not found in {candidate}")


def parse_named(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use NAME=PATH")
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-md", type=Path, required=True)
    parser.add_argument("--level", action="append", required=True)
    parser.add_argument("--baseline-run", action="append", type=parse_named, required=True)
    args = parser.parse_args()

    text = args.metrics_md.read_text()
    marker = "## Corresponding Model Baselines"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n\n"

    lines = [
        marker,
        "",
        "Matching DaVinci and base DrKernel-8B results for this KernelBench generation and level. Speedups use the named reference baseline; clean-correct counts are policy-correct timed kernels.",
        "",
        "| Level | Model | Reference | Correct | Fast@1 | Fast@1.2 | Fast@1.5 | Fast@2 | GeoMean | Median | Mean | Max | Clean correct |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in sorted(set(args.level), key=int):
        for model, run_path in args.baseline_run:
            level_metrics = load_level(run_path, level)
            for reference, baseline in level_metrics.get("baseline_metrics", {}).items():
                lines.append(
                    "| L{level} | {model} | {reference} | {correct}/{tasks} ({rate:.2%}) | "
                    "{f1:.2%} | {f12:.2%} | {f15:.2%} | {f2:.2%} | "
                    "{geo:.4f} | {median:.4f} | {mean:.4f} | {maximum:.4f} | {samples} |".format(
                        level=level,
                        model=model,
                        reference=reference,
                        correct=level_metrics.get("correct_count", 0),
                        tasks=level_metrics.get("total_count", 0),
                        rate=level_metrics.get("correctness_rate", 0.0),
                        f1=baseline["fast_p"].get("1.0", 0.0),
                        f12=baseline["fast_p"].get("1.2", 0.0),
                        f15=baseline["fast_p"].get("1.5", 0.0),
                        f2=baseline["fast_p"].get("2.0", 0.0),
                        geo=baseline["geo_mean_speedup_correct"],
                        median=baseline["median_speedup_correct"],
                        mean=baseline["mean_speedup_correct"],
                        maximum=baseline["max_speedup_correct"],
                        samples=baseline["speedup_count"],
                    )
                )
    args.metrics_md.write_text(text + "\n".join(lines) + "\n")
    print(args.metrics_md)


if __name__ == "__main__":
    main()

