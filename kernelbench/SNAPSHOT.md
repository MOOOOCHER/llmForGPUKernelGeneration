# Thesis KernelBench evaluator snapshot

This directory contains the KernelBench source checkout preserved for the thesis
on 28 October 2026, including local evaluator changes and their tests.

- Base Git revision: `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`.
- Original HPI checkout: `/sc/scratch/zongxiong.chen/daniel/KernelBench`.
- Origin: `https://github.com/ScalingIntelligence/KernelBench.git`.
- Upstream documentation: [README.md](README.md) and [EVAL.md](EVAL.md).
- Upstream license: [LICENSE](LICENSE), preserved unchanged.

`snapshot_manifest.json` records the source revision, checkout status, and the
size and SHA-256 hash of every copied file. `local_changes.patch` records the
tracked modifications and three added source/test files relative to the base
revision. The preserved snapshot includes the applied changes; no patching step
is needed to use it.

## Thesis modifications

The local changes cover evaluation/export metadata handling, evaluator-owned
FP32 math-mode controls and source sanitization, precision/static policy checks,
median timing statistics, and associated tests. The exact changed paths are in
the manifest and patch.

## Contents and boundaries

The snapshot includes upstream tracked benchmark tasks, evaluator source,
scripts, tests, package metadata, documentation, assets, notebooks, and reference
timing fixtures. Its additional files are `precision.py`,
`test_strict_fp32.py`, and `test_median_timing.py`.

The original `.git` directory, interpreter/pytest/Triton caches, runtime `runs/`,
and untracked HPI campaign results are outside this snapshot. Upstream tracked
reference timing fixtures are retained as supplied; they do not establish the
matching reference for a thesis campaign. The HPI-specific
`H100_aisc_fp32`/`H100_aisc_bf16` timing collections remain external artifacts.

The package source lives in `kernelbench/src/kernelbench`, relative to the
KernelGYM repository root. Its distribution name is `kernelbench`; the benchmark
task directory is `kernelbench/KernelBench/`. Evaluation wrappers should resolve
these repository-relative locations rather than the original HPI checkout.

Use the environment appropriate to the evaluation campaign. The preserved
`pyproject.toml` declares Python 3.10; historical campaign environments may differ.
This snapshot captures the current HPI source and is not evidence that every
historical campaign used identical source or software versions. Campaign
configuration, generation/export orchestration, containers, and result artifacts
must be indexed separately by the evaluation pipeline.
