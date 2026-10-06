from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from core.dataset import DatasetGenerator, statistics

DatasetKind = Literal["stage1", "stage2"]

GENERATOR_PROFILES = {
    "stage1": "stage1-60base-40compact-stage2-recipes-v1-no-fusion-prompt",
    "stage2": "stage2-recipes-v10-softmax-10-no-reduction-primary",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_parquet(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty Parquet dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # Parquet cannot represent an empty struct. Empty operator-attribute and
    # fusion-target mappings are semantically absent, so encode them as null.
    sanitized = [_sanitize_for_arrow(row) for row in rows]
    pq.write_table(pa.Table.from_pylist(sanitized), temporary, compression="zstd")
    os.replace(temporary, path)


def _sanitize_for_arrow(value: Any) -> Any:
    if isinstance(value, dict):
        if not value:
            return None
        return {key: _sanitize_for_arrow(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_arrow(item) for item in value]
    return value


class ParquetCheckpoints:
    """Store immutable Parquet shards and recover a contiguous accepted prefix."""

    _PART_RE = re.compile(r"^part-(\d{6})-(\d{6})\.parquet$")

    def __init__(
        self,
        output: Path,
        *,
        kind: str,
        seed: int,
        total_samples: int,
        data_source: str,
        resume: bool,
    ) -> None:
        self.output = output
        self.directory = output.with_suffix(".checkpoints")
        self.metadata_path = self.directory / "metadata.json"
        self.metadata = {
            "version": 2,
            "kind": kind,
            "generator_profile": GENERATOR_PROFILES[kind],
            "seed": seed,
            "total_samples": total_samples,
            "data_source": data_source,
        }
        self.persisted = 0

        if self.directory.exists() and not resume:
            raise FileExistsError(
                f"checkpoint directory already exists: {self.directory}; use --resume"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists():
            actual = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if actual != self.metadata:
                actual_target = actual.get("total_samples")
                same_run = all(
                    actual.get(key) == expected
                    for key, expected in self.metadata.items()
                    if key != "total_samples"
                )
                can_extend = (
                    resume
                    and same_run
                    and isinstance(actual_target, int)
                    and actual_target < total_samples
                )
                if not can_extend:
                    raise ValueError(
                        f"checkpoint metadata mismatch in {self.metadata_path}: "
                        f"expected {self.metadata}, found {actual}"
                    )
                print(
                    f"[{kind}] extending checkpoint target from "
                    f"{actual_target} to {total_samples} samples",
                    flush=True,
                )
                _write_json(self.metadata_path, self.metadata)
        else:
            existing_parts = list(self.directory.glob("part-*.parquet"))
            if existing_parts:
                raise ValueError(f"checkpoint parts exist without metadata in {self.directory}")
            _write_json(self.metadata_path, self.metadata)

    def load(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        expected_start = 0
        parts = sorted(self.directory.glob("part-*.parquet"))
        for part in parts:
            match = self._PART_RE.match(part.name)
            if match is None:
                continue
            start, end = map(int, match.groups())
            if start != expected_start or end <= start:
                raise ValueError(
                    f"non-contiguous checkpoint {part}: expected start {expected_start}"
                )
            rows = pq.read_table(part).to_pylist()
            if len(rows) != end - start:
                raise ValueError(
                    f"checkpoint row-count mismatch for {part}: "
                    f"filename says {end - start}, file has {len(rows)}"
                )
            samples.extend(rows)
            expected_start = end
        self.persisted = len(samples)
        return samples

    def save_through(self, samples: List[Dict[str, Any]], count: int) -> None:
        if count <= self.persisted:
            return
        if count > len(samples):
            raise ValueError(f"cannot checkpoint {count} rows from {len(samples)} samples")
        part = self.directory / f"part-{self.persisted:06d}-{count:06d}.parquet"
        _write_parquet(part, samples[self.persisted:count])
        self.persisted = count


def generate_dataset(
    kind: DatasetKind,
    samples: int,
    *,
    seed: int = 42,
    output: str | Path | None = None,
    stats_output: str | Path | None = None,
    data_source: str = "balanced_csp_dag",
    max_attempts_per_sample: int = 50,
    runtime_validate: bool = False,
    validation_device: str = "auto",
    validation_timeout_seconds: float = 20.0,
    validation_repeats: int = 2,
    check_intermediates: bool = True,
    output_format: str = "json",
    checkpoint_every: int = 0,
    progress_every: int = 0,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    """Generate a Stage1 or Stage2 dataset in DrKernel RL format."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if kind not in {"stage1", "stage2"}:
        raise ValueError(f"unknown dataset kind: {kind!r}")
    if output_format not in {"json", "parquet"}:
        raise ValueError(f"unknown output format: {output_format!r}")
    if checkpoint_every < 0 or progress_every < 0:
        raise ValueError("checkpoint and progress intervals must be non-negative")

    output_path = Path(output) if output is not None else None
    checkpoints: Optional[ParquetCheckpoints] = None
    initial_samples: List[Dict[str, Any]] = []
    if checkpoint_every:
        if output_format != "parquet" or output_path is None:
            raise ValueError("checkpoints require a Parquet output path")
        checkpoints = ParquetCheckpoints(
            output_path,
            kind=kind,
            seed=seed,
            total_samples=samples,
            data_source=data_source,
            resume=resume,
        )
        initial_samples = checkpoints.load()
        if initial_samples:
            print(
                f"[{kind}] resumed {len(initial_samples)}/{samples} accepted samples "
                f"from {checkpoints.directory}",
                flush=True,
            )

    resumed_count = len(initial_samples)
    started_at = time.monotonic()

    def on_sample_accepted(current: List[Dict[str, Any]]) -> None:
        accepted = len(current)
        if checkpoints is not None and accepted % checkpoint_every == 0:
            checkpoints.save_through(current, accepted)
        if progress_every and accepted % progress_every == 0:
            elapsed = max(time.monotonic() - started_at, 1e-9)
            generated_now = accepted - resumed_count
            rate = generated_now / elapsed
            remaining = samples - accepted
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            print(
                f"[{kind}] progress: {accepted}/{samples} accepted "
                f"({100.0 * accepted / samples:.1f}%), "
                f"run_rate={rate:.3f} samples/s, eta={eta_seconds / 3600:.2f}h",
                flush=True,
            )

    generator = DatasetGenerator(stage=kind, seed=seed, data_source=data_source)
    generated = generator.generate(
        total_samples=samples,
        max_attempts_per_sample=max_attempts_per_sample,
        initial_samples=initial_samples,
        on_sample_accepted=on_sample_accepted,
        runtime_validate=runtime_validate,
        validation_device=validation_device,
        validation_timeout_seconds=validation_timeout_seconds,
        validation_repeats=validation_repeats,
        check_intermediates=check_intermediates,
    )
    summary = statistics(generated)

    if checkpoints is not None:
        checkpoints.save_through(generated, len(generated))
    if output_path is not None:
        if output_format == "parquet":
            _write_parquet(output_path, generated)
        else:
            _write_json(output_path, generated)
    if stats_output is not None:
        _write_json(Path(stats_output), summary)
    return generated


def generate_stage1_data(
    samples: int,
    *,
    seed: int = 42,
    output: str | Path = "stage1.json",
    stats_output: str | Path | None = "stage1_stats.json",
    max_attempts_per_sample: int = 50,
    runtime_validate: bool = False,
    validation_device: str = "auto",
    validation_timeout_seconds: float = 20.0,
    validation_repeats: int = 2,
    check_intermediates: bool = True,
    output_format: str = "json",
    checkpoint_every: int = 0,
    progress_every: int = 0,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    """Generate Stage1 data: 2–5 ops; structured regions span every op."""
    return generate_dataset(
        "stage1", samples, seed=seed, output=output,
        stats_output=stats_output, max_attempts_per_sample=max_attempts_per_sample,
        runtime_validate=runtime_validate, validation_device=validation_device,
        validation_timeout_seconds=validation_timeout_seconds,
        validation_repeats=validation_repeats, check_intermediates=check_intermediates,
        output_format=output_format, checkpoint_every=checkpoint_every,
        progress_every=progress_every, resume=resume,
    )


def generate_stage2_data(
    samples: int,
    *,
    seed: int = 42,
    output: str | Path = "stage2.json",
    stats_output: str | Path | None = "stage2_stats.json",
    max_attempts_per_sample: int = 100,
    runtime_validate: bool = False,
    validation_device: str = "auto",
    validation_timeout_seconds: float = 20.0,
    validation_repeats: int = 2,
    check_intermediates: bool = True,
    output_format: str = "json",
    checkpoint_every: int = 0,
    progress_every: int = 0,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    """Generate Stage2 data: 6–15 ops, 50% structured, branches enabled."""
    return generate_dataset(
        "stage2", samples, seed=seed, output=output,
        stats_output=stats_output, max_attempts_per_sample=max_attempts_per_sample,
        runtime_validate=runtime_validate, validation_device=validation_device,
        validation_timeout_seconds=validation_timeout_seconds,
        validation_repeats=validation_repeats, check_intermediates=check_intermediates,
        output_format=output_format, checkpoint_every=checkpoint_every,
        progress_every=progress_every, resume=resume,
    )


# Backward-compatible aliases for existing callers.
def generate_warmup_data(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    return generate_stage1_data(*args, **kwargs)


def generate_main_rl_data(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    return generate_stage2_data(*args, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate balanced DrKernel-format Stage1 and Stage2 datasets."
    )
    parser.add_argument(
        "kind", choices=("stage1", "stage2", "all"),
        help="Curriculum stage to generate.",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--stage1-samples", type=int, default=None)
    parser.add_argument("--stage2-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument(
        "--runtime-validate", action="store_true",
        help="Execute each emitted reference model before accepting it.",
    )
    parser.add_argument(
        "--validation-device", choices=("auto", "cpu", "cuda"), default="auto",
    )
    parser.add_argument("--validation-timeout", type=float, default=20.0)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--output-format", choices=("json", "parquet"), default="json")
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="Persist an immutable Parquet checkpoint shard every N accepted samples.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=0,
        help="Print accepted-sample progress every N samples.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from compatible checkpoint shards in the output directory.",
    )
    parser.add_argument(
        "--no-intermediate-check", action="store_true",
        help="Check only returned outputs, not every forward assignment.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir: Path = args.output_dir
    summaries: Dict[str, Dict[str, Any]] = {}

    if args.kind in {"stage1", "all"}:
        count = (
            args.stage1_samples
            if args.kind == "all" and args.stage1_samples is not None
            else args.samples
        )
        data = generate_stage1_data(
            count, seed=args.seed,
            output=output_dir / f"stage1.{args.output_format}",
            stats_output=output_dir / "stage1_stats.json",
            max_attempts_per_sample=args.max_attempts or 50,
            runtime_validate=args.runtime_validate,
            validation_device=args.validation_device,
            validation_timeout_seconds=args.validation_timeout,
            validation_repeats=args.validation_repeats,
            check_intermediates=not args.no_intermediate_check,
            output_format=args.output_format,
            checkpoint_every=args.checkpoint_every,
            progress_every=args.progress_every,
            resume=args.resume,
        )
        summaries["stage1"] = statistics(data)

    if args.kind in {"stage2", "all"}:
        count = (
            args.stage2_samples
            if args.kind == "all" and args.stage2_samples is not None
            else args.samples
        )
        seed = args.seed + 1 if args.kind == "all" else args.seed
        data = generate_stage2_data(
            count, seed=seed,
            output=output_dir / f"stage2.{args.output_format}",
            stats_output=output_dir / "stage2_stats.json",
            max_attempts_per_sample=args.max_attempts or 100,
            runtime_validate=args.runtime_validate,
            validation_device=args.validation_device,
            validation_timeout_seconds=args.validation_timeout,
            validation_repeats=args.validation_repeats,
            check_intermediates=not args.no_intermediate_check,
            output_format=args.output_format,
            checkpoint_every=args.checkpoint_every,
            progress_every=args.progress_every,
            resume=args.resume,
        )
        summaries["stage2"] = statistics(data)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
