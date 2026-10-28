# Thesis training recipes

Slurm launchers for the five DrKernel-based thesis approaches.
Submit from the repository root. Each allocation uses **four H100s**
shared by training, rollout, and KernelGYM evaluation, with sequence parallelism
4, 48 CPUs, 450 GB RAM, and a default 24-hour limit. Cluster resource limits
can be overridden with `sbatch` options.

## Settings

| Recipe | Coverage-retention ramp | Fusion weight | Evaluated step |
|---|---|---:|---:|
| `coverage-threshold.slurm` | 10–40% | 0.5 | 400 |
| `no-fusion.slurm` | 10–40% | 0 | 400 |
| `no-filter.slurm` | Disabled | 0.5 | 400 |
| `proportional-coverage.slurm` | 0–30% | 0.5 | 400 |
| `original-coverage-rs.slurm` | 30–40% | 0.5 | 400 |

The five DrKernel recipes start from `models/sft-cold-start/` and use the
10,000-record `dataset/stage1/stage1.parquet`. They use TRLOO, two turns,
eight rollouts per prompt, batch/minibatch 8, learning rate 1e-6, 14,336 prompt
tokens and 8,192 response tokens, 50 training timing trials, profiling-reward
weight 0.25, and a 1.2× eager-speedup exemption from coverage rejection.
MRS uses the geometric rule, importance-ratio interval 0.995–1.005, and token
veto 1e-4. Checkpoints are saved every two steps with two retained.

`_stage1.sh` preserves the existing HPI service/training launcher shared by
these five recipes. The recipe explicitly supplies the fusion weight to Hydra;
the fusion reward function activates compression independently of the nested
`fusion_reward.enable` default. `no-fusion.slurm` sets its weight to zero.

DaVinci was trained in the separate `daVinci-kernel/` repository. Its training
recipe is outside this repository; the [root README](../../README.md#thesis-models-and-results)
indexes its baseline model and evaluation results.

## Launch

The [root README](../../README.md#external-dependencies) lists the external
Python environment, Enroot image, framework, and model files. The original
100-record validation data is expected at
`rl-training/data/hkust-nlp/drkernel-validation-data.parquet`. This upstream
validation dataset is not part of the tracked synthetic dataset bundle.

```bash
mkdir -p logs/rl-training
sbatch rl-training/recipes/coverage-threshold.slurm
sbatch rl-training/recipes/no-fusion.slurm
sbatch rl-training/recipes/no-filter.slurm
sbatch rl-training/recipes/proportional-coverage.slurm
sbatch rl-training/recipes/original-coverage-rs.slurm
```

Each command submits a separate run. `KERNELGYM_ROOT` defaults to the submission
directory, and `REPO_ROOT` to its parent workspace. Model, dataset, environment,
and checkpoint paths can be overridden through their named environment
variables; see each launcher. Default outputs are isolated under
`../checkpoints/_other_runs/recipes/<approach>/` and fresh launches refuse
existing checkpoint roots. To continue an allocation intentionally:

```bash
sbatch --export=ALL,ALLOW_RESUME=1 rl-training/recipes/coverage-threshold.slurm
```

Use the same checkpoint root and settings, and wait for the previous writer to
finish. Resume loads the latest saved checkpoint. The historical boundaries
below describe the evaluated models; reruns need not time out at identical steps.

## Historical resume lineage

`Resume` is the loaded optimizer step; overlapping logged steps were recomputed.
Step ranges describe completed updates, including those later overwritten.

| Approach | Job | Resume | Logged completed steps |
|---|---:|---:|---|
| Coverage Threshold | 2513033 | 0 | 1–93 |
| Coverage Threshold | 2515357 | 92 | 93–103 |
| Coverage Threshold | 2520744 | 102 | 103–278 |
| Coverage Threshold | 2527207 | 278 | 279–400 |
| No-Fusion | 2637148 | 0 | 1–141 |
| No-Fusion | 2664788 | 140 | 141–376 |
| No-Fusion | 2706524 | 376 | 377–400 |
| No Filter | 2478587 | 0 | 1–121 |
| No Filter | 2483114 | 120 | 121–200 |
| No Filter | 2488792 | 200 | 201–388 |
| No Filter | 2610264 | 388 | 389–400 |
| Proportional Coverage | 2545989 | 0 | 1–104 |
| Proportional Coverage | 2556088 | 104 | 105–279 |
| Proportional Coverage | 2561852 | 278 | 279–400 |
| Original Coverage RS | 2589249 | 0 | 1–10 |
| Original Coverage RS | 2626215 | 10 | 11–151 |
| Original Coverage RS | 2647416 | 150 | 151–372 |
| Original Coverage RS | 2680272 | 372 | 373–400 |

Coverage Threshold's earlier job 2511490 ended before a completed update.
Coverage ramps follow the corrected `declared_coverage_rs` records: 10–40%
for Coverage Threshold and 0–30% for Proportional Coverage throughout.
The No Filter ceiling was initially 200, then extended to 400. Coverage
Threshold's skip budget changed from 50 to 15 in job 2527207; this is a failure
budget, not a reward/filter change. Its recipe defaults to 50 and accepts
`MAX_SKIP_STEPS=15`. No-Fusion uses 50; the other three DrKernel recipes use 15.

Sources: the HPI Stage-1 launchers under `slurm/RL/`, their recorded
arguments, and grouped training manifests under `Results/RL-Training/` in the
external workspace. Existing final thesis checkpoints are kept separately from
these recipe output directories.
