# GPU kernel generation with reinforcement learning

Code and datasets for the thesis experiments on training language models to
generate efficient GPU kernels. The work extends KernelGYM/DrKernel with
fusion-aware rewards and coverage-based rejection sampling, and evaluates
generated Triton kernels against PyTorch eager and `torch.compile`.

```text
Synthetic PyTorch programs → RL training → multi-turn kernel generation → KernelBench evaluation
```

## Repository layout

| Directory | Contents |
|---|---|
| [`dataset/`](dataset/README.md) | Stage 1 and Stage 2 RL datasets, statistics, and generation settings |
| [`dataset_generation/`](dataset_generation/README.md) | Synthetic program generator, CUDA validation, and Parquet export |
| [`rl-training/`](rl-training/README.md) | RL training code, reward functions, configurations, and VERL integration |
| [`kernelgym/`](kernelgym/) | GPU execution service used for correctness checks and training feedback |
| [`kernelbench/`](kernelbench/SNAPSHOT.md) | Benchmark tasks and evaluator, including FP32 and median-timing changes |
| [`eval-pipeline/`](eval-pipeline/README.md) | Checkpoint merging, generation, candidate export, and final evaluation |

## Datasets

| Dataset | Programs | Operators per program | Use |
|---|---:|---:|---|
| Stage 1 | 10,000 | 2–5 | Dataset used for the thesis training experiments |
| Stage 2 | 25,000 | 6–15 | Longer operator chains for continuation experiments |

Each record contains a model prompt, reference PyTorch code, and metadata about
the computation graph and tensor shapes. Prompts ask for an optimized FP32
Triton implementation without providing an explicit fusion plan.
See the [dataset description](dataset/README.md) for workload composition and
file formats. Existing datasets can be used directly without regeneration.

## Setup

Model training and inference require Linux, CUDA-enabled PyTorch, and compatible
Triton and vLLM installations. Training also uses Redis and the KernelGYM service.
The HPI evaluation launchers require Slurm, Enroot, and the evaluation image.

VERL is included directly under `rl-training/verl/`. Its
[snapshot notes](rl-training/verl/SNAPSHOT.md) record the upstream revision and
local changes; no submodule initialization is required.

The [service setup script](setup.sh) and [training guide](rl-training/README.md)
describe the base installation. The generation and evaluation guides describe
their additional dependencies. HPI launchers contain cluster-specific paths
and resource settings; adapt them for another machine or cluster.

## Generate data

With a CUDA-enabled Python environment active, run a small generation test:

```bash
python -m pip install -r dataset_generation/requirements.txt
./dataset_generation/generate.sh stage1 --samples 4 --checkpoint-every 2 \
  --output-dir dataset/test_stage1
```

Use `stage2` for longer programs. Without `--samples`, the script generates
the full dataset size listed above. New runs write into `dataset/` and support
checkpointing and resuming. See [dataset generation](dataset_generation/README.md)
for full runs, validation, and Slurm commands.

## Train

The [training guide](rl-training/README.md) covers the base RL workflow and service
setup. Thesis training changes are implemented in
[`rl-training/kernel/rewards/`](rl-training/kernel/rewards/) and
[`rl-training/kernel/config/`](rl-training/kernel/config/).

The primary approach is **Coverage Threshold**. The thesis compares it with
**No-Fusion**, **No Filter**, **Proportional Coverage**, and
**Original Coverage RS**, alongside the model baselines. The five
[thesis training recipes](rl-training/recipes/README.md) include the HPI
launchers, explicit approach settings, submission commands, and resume history.
DaVinci was trained in the separate `daVinci-kernel/` repository and is included
here as a comparison baseline; its training implementation and recipe are outside
this repository.

## Evaluate

On HPI, evaluate the main step-400 checkpoint from the repository root:

```bash
CHECKPOINT_STEP=400 N_SAMPLES=3 MAX_USER_TURNS=3 \
bash eval-pipeline/scripts/shell/submission/submit_kernelbench_tts_per_level.sh \
  ../checkpoints/coverage-threshold coverage-threshold
```

The launcher submits checkpoint merging, multi-turn generation, and dependent
export/evaluation jobs for KernelBench Levels 1–3. Final evaluation uses FP32
correctness checks and median kernel timings, reporting speedups against eager
and compiled PyTorch. See the [evaluation guide](eval-pipeline/README.md) for model
exports, small tests, outputs, and configuration overrides.

**DaVinci skill-aware evaluation requires the separate DaVinci codebase.**
The workflow documented here uses KernelGYM/DrKernel without skill retrieval.

## External dependencies

The HPI workspace root is `/sc/scratch/zongxiong.chen/daniel`; this repository
is its `KernelGYM/` directory. Paths in this section are relative to that
workspace root. These artifacts are maintained outside Git and must be supplied
separately for a fresh checkout.

| Dependency | HPI location / configuration |
|---|---|
| Training and generation environment | `.venv-drkernel/`; launchers use its Python executable |
| Evaluation container | `images/kernelbench-torch2.9-cu128.sqsh`; override with `CONTAINER` |
| H100 reference timings | `KernelBench/results/timing/H100_aisc_fp32/`; override with `BASELINE_DIR` |
| Fusion-analysis framework | `archive/DefragmentationFramework/`; override with `DEFRAG_FRAMEWORK_PATH` |
| DaVinci training and skill-aware generation | Separate `daVinci-kernel/` checkout |
| Trained DaVinci skill library | `checkpoints/davinci/skill_library/global_step_84.jsonl` |

The training/generation environment recorded on **28 October 2026** uses
Python 3.12.14, PyTorch 2.8.0, vLLM 0.10.2, Triton 3.4.0, Transformers 4.56.0,
and FlashAttention 2.8.3. Dataset generation uses OR-Tools 9.14.6206 and
PyArrow 25.0.0. These describe the current environment; historical campaigns
do not all have complete package-version inventories.

The Enroot image is based on
`pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel` and installs the KernelBench
runtime dependencies, including `litellm`. The external build recipe is
`slurm/Enroot/build_kernelbench_image.sbatch`. It imports the base image,
installs dependencies, and exports the `.sqsh` file. Building requires at least
40 GB of node-local scratch; adjust its cluster paths and scheduler settings
before using it elsewhere. KernelBench source is mounted from this repository
at evaluation time. Use the container environment for evaluator checks.

The saved reference-timing directory contains `baseline_time_torch.json` and
`baseline_time_torch_compile_inductor_default.json`. Evaluation metrics use
paired measurements when available and these saved timings as fallback.

## Thesis models and results

Model exports are in `models/<export>/` under the HPI workspace root.
The five trained approaches use `checkpoints/<export>/global_step_400/`.
Trained DaVinci uses step 84 in its nested run directory under
`checkpoints/davinci/`; both skill conditions share the same weights.

| Thesis approach | Model export | Checkpoint step | H100 result label |
|---|---|---:|---|
| Coverage Threshold (Main) | `coverage-threshold` | 400 | `Coverage Threshold (Main)` |
| No-Fusion | `no-fusion` | 400 | `No Compression` |
| No Filter | `no-filter` | 400 | `No Filter` |
| Proportional Coverage | `proportional-coverage` | 400 | `Proportional Coverage` |
| Original Coverage RS | `original-coverage-rs` | 400 | `Original Coverage RS` |
| SFT Cold Start | `sft-cold-start` | Released model | `SFT Cold Start` |
| DrKernel-8B | `drkernel-8b` | Released RL model | `DrKernel-8B` |
| DaVinci without skills | `trained_davinci` | 84 | `DaVinci (no skills)` |
| DaVinci with skills | `trained_davinci` | 84 | `DaVinci + skills` |

`No Compression` is the existing artifact label for the thesis's **No-Fusion**
ablation. The released DaVinci SFT and RL exports remain separately available
as `models/daVinci-kernel-8B-SFT/` and `models/daVinci-kernel-8B-RL/`.

The separately prepared thesis result bundle stores the main H100 three-turn
evaluations under:

```text
Results/h100/3_turn/evaluation_results/<result-label>/l1, l2, l3/
Results/h100/3_turn/tts_runs/<result-label>/l1, l2, l3/
```

Each evaluation level contains readable and JSON metrics
(`kernelbench_metrics.md` / `.json`), raw evaluations (`eval_results.json`),
run and export metadata, and selected-turn/kernel records. Keep the selection
and policy-audit metadata with the metrics: the No Filter collection includes
a promoted Level-2 repeat, and guided DaVinci includes audited Level-2
replacements. `Results/h100/README.md` in that bundle explains these details.
The primary model's hardware comparisons are also retained under
`Results/gh200/` and `Results/A100/`.

Original HPI campaigns and newly generated pipeline outputs live under
`Results/Kernelbench/evaluation_results/` and `Results/Kernelbench/tts_runs/`.
Their historical campaign names are retained in the result metadata. Model
weights and the curated result bundle are outside this Git repository;
current model access is through the HPI workspace, with no public download
configured here.

## Upstream projects

This work builds on [KernelGYM/DrKernel](https://github.com/hkust-nlp/KernelGYM),
[VERL](https://github.com/verl-project/verl), and
[KernelBench](https://github.com/ScalingIntelligence/KernelBench).
The [DrKernel guide](rl-training/README.md#citation) retains the upstream paper
citation. [KernelBench snapshot notes](kernelbench/SNAPSHOT.md) record the
evaluator revision and local changes; its [license](kernelbench/LICENSE) is
included.
