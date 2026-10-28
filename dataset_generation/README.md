# Dataset generation

Generates synthetic PyTorch programs for training GPU kernel generation models.
Each sample contains a reference program and a prompt asking the model to
implement an optimized FP32 version with Triton. Samples are exported as
Parquet records in DrKernel's RL training format.

## Datasets

| Stage | Operations per program | Default samples | Seed |
|---|---:|---:|---:|
| Stage 1 | 2–5 | 10,000 | 42 |
| Stage 2 | 6–15 | 25,000 | 43 |

**Stage 1 is the dataset used in the thesis.** Its short programs focus on local
fusion opportunities across pointwise operations, reductions, scans,
normalization, softmax, pooling, matrix operations, and convolutions. It combines
60% base Stage-1 motifs with 40% compact motifs from the Stage-2 recipe library.

**Stage 2 contains longer operation chains.** It combines matrix, convolution,
pooling, normalization, softmax, and scan motifs into larger programs with more
operation boundaries and dependencies. Program lengths from 6 to 15 are balanced.

Both stages target 20% small, 40% medium, and 40% large workloads. They can be
generated independently. The existing datasets and their statistics are in
[`../dataset/`](../dataset/README.md).

## How generation works

1. Schedule a program length, workload size, and operator family.
2. Build a structured computation graph from compatible operations.
3. Solve tensor dimensions with OR-Tools CP-SAT, respecting operator shape
   constraints and tensor-size limits.
4. Emit a PyTorch `Model` with `get_inputs()` and `get_init_inputs()`, validate
   it, and convert it into an RL training record.

Every operation must contribute to the returned output. Prompts include the
reference program without an explicit fusion plan, leaving implementation and
fusion choices to the model.

## Setup

Activate a Python environment with CUDA-enabled PyTorch. From the repository
root, install the dependencies:

```bash
python -m pip install -r dataset_generation/requirements.txt
```

The generator uses PyTorch for execution, OR-Tools for shape solving, and
pandas/PyArrow for Parquet export. The shell entry point uses `python3` by
default; set `DATASET_PYTHON` to use a specific environment's executable.

## Generate

Run from the repository root on a CUDA machine:

```bash
./dataset_generation/generate.sh stage1
./dataset_generation/generate.sh stage2
```

These commands use the sample counts and seeds in the table above. The shell
script enables CUDA runtime validation, Parquet output, checkpointing every
100 accepted samples, and progress reporting every 10 samples.

For a small test:

```bash
./dataset_generation/generate.sh stage1 --samples 4 --checkpoint-every 2 \
  --progress-every 1 --output-dir dataset/test_stage1
```

Replace `stage1` with `stage2` and choose a different output folder to test the
longer programs. Additional options, including `--samples`, `--seed`, and
`--output-dir`, are forwarded to `generate_data.py`. List them with:

```bash
./dataset_generation/generate.sh stage1 --help
```

## Outputs and resuming

By default, each run creates a fresh `dataset/stage1_<run-id>/` or
`dataset/stage2_<run-id>/` folder. The run ID is the Slurm job ID when submitted
through Slurm, otherwise a timestamp and process ID. Use `--output-dir` or
`DATASET_OUTPUT_DIR` to select a specific folder.

A Stage-1 output folder contains:

```text
stage1.parquet                 # Training records
stage1_stats.json              # Operator, family, length, and workload counts
stage1.checkpoints/
  metadata.json                # Stage, profile, seed, and target count
  part-*.parquet               # Completed checkpoint shards
```

Records contain `prompt`, `reward_model.ground_truth` (the reference code),
and `extra_info` with operators, graph structure, and solved tensor shapes.
Stage 2 uses the corresponding `stage2` filenames.

To resume an interrupted run, repeat the command with the same output folder,
stage, seed, sample count, and generator profile. The shell script enables
`--resume`; generation continues from the last completed checkpoint shard.

## Runtime validation

The shell script executes each candidate on CUDA before accepting it. Validation
checks model construction, two repeated forward passes, finite outputs and
intermediate tensors, and consistent results. Failed or timed-out candidates
are rejected and regenerated without consuming a sample slot. The default
validation timeout is 30 seconds per candidate.

Generation settings live in [`core/generator.py`](core/generator.py); tensor
domains and workload limits are in [`core/shape_solver.py`](core/shape_solver.py).

## Slurm on hpi2

```bash
mkdir -p /sc/scratch/zongxiong.chen/daniel/logs/dataset
sbatch --export=ALL,STAGE=stage1 dataset_generation/slurm/generate_dataset.slurm
sbatch --export=ALL,STAGE=stage2 dataset_generation/slurm/generate_dataset.slurm
```

For a ten-minute test generating four Stage-1 samples, override the main
launcher's time limit and sample count:

```bash
sbatch --time=00:10:00 --export=ALL,STAGE=stage1 \
  dataset_generation/slurm/generate_dataset.slurm \
  --samples 4 --checkpoint-every 2 --progress-every 1
```

Use `STAGE=stage2` to test the longer programs.
Each job requests one GPU. Logs are written under
`/sc/scratch/zongxiong.chen/daniel/logs/dataset/`.
The Slurm launcher contains hpi2-specific resource settings and Python paths;
adapt these when running on another cluster.
