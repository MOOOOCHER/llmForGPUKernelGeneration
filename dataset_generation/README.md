# Balanced CSP-DAG PyTorch Program Generator

## KernelGYM integration

This pipeline lives in `KernelGYM/dataset_generation`. The generator CLI and
`core/*.py` are copied unchanged from the existing
`DefragmentationFramework/Dataset_Generator` implementation. Generated datasets,
checkpoint shards, shape-analysis statistics, backups, and analysis-only helpers
are intentionally not included.

The current profile identifiers are:

- Stage 1: `stage1-60base-40compact-stage2-recipes-v1-no-fusion-prompt`
- Stage 2: `stage2-recipes-v10-softmax-10-no-reduction-primary`

These source-defined profiles are authoritative; the design notes below describe
the underlying generator and also retain historical Stage-1/prompt behavior.
The current profiles do not inject a reference fusion plan into the prompt.

The copied `slurm/generate_dataset.slurm` supports both stages and preserves
the existing counts, seeds, checkpoint/resume settings, CUDA validation, and
Stage-2 structural-program mixing:

```bash
# Run from the KernelGYM repository root on hpi2.
mkdir -p /sc/scratch/zongxiong.chen/daniel/logs/dataset
sbatch --export=ALL,STAGE=stage1 dataset_generation/slurm/generate_dataset.slurm
sbatch --export=ALL,STAGE=stage2 dataset_generation/slurm/generate_dataset.slurm
```

Stage 1 generates 10,000 synthetic samples (seed 42). Stage 2 generates 25,000
synthetic samples (seed 43), then appends unique original DrKernel structural
programs using the included `analyze_stage2_structural_candidates.py`.
Its source corpus is the existing
`drkernel/data/hkust-nlp/drkernel-rl-data.parquet`; this data file is not copied.
Published datasets remain under
`/sc/scratch/zongxiong.chen/daniel/datasets` by default. Resumable generation
state is organized under `generation_state/stage1` and `generation_state/stage2`
in that output directory. Previous profiles are kept in the datasets archive;
their checkpoint settings are not edited or reused for a different profile.

Optional environment overrides: `DATASET_WORKSPACE_ROOT`, `KERNELGYM_REPO`,
`DATASET_PYTHON`, and `DATASET_OUTPUT_DIR`. The launcher's Slurm log paths and
account/partition settings are hpi2-specific. For a different cluster, adapt
those directives or override them at submission.

Only the copied launcher is adapted to the new generator/mixer locations. Its
final Stage-2 publication check also converts the input filename to `Path`
before resolving the report filename. Original scripts and datasets are left
untouched.

This package reproduces the core CSP-DAG idea and conditions it on balanced fusion targets.

## Design

1. Sample a balanced cell: structured/random, anchor family, fusion pattern, program size, fusion size.
2. In Stage 1, structured fusion covers the complete 2–5-op program. Stage 2 may add non-fusion context around a connected fusion block.
3. Represent the program as an edge-owned tensor DAG.
4. Solve all tensor dimensions with OR-Tools CP-SAT.
5. Emit KernelBench-style PyTorch code (`Model`, `get_inputs`, `get_init_inputs`).
6. Convert to Dr. Kernel RL format.

`chain` is only used for pointwise and reduction chains. `operators.py` is the single source of truth for operator arity, rank contracts, families, and legal patterns.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Package layout

```text
dataset_generation/
├── generate_data.py          # Public API and CLI
├── core/                     # Generator implementation
│   ├── dataset.py
│   ├── generator.py
│   ├── shape_solver.py
│   ├── operators.py
│   ├── graph.py
│   ├── emitter.py
│   └── drkernel_format_converter.py
├── analyze_stage2_structural_candidates.py  # Stage-2 structural-program mixer
├── slurm/generate_dataset.slurm            # Shared Stage-1/Stage-2 launcher
├── requirements.txt
└── README.md
```

## Generate

CLI:

```bash
python generate_data.py stage1 --samples 1000 --output-dir data
python generate_data.py stage2 --samples 8000 --output-dir data
python generate_data.py all --stage1-samples 1000 --stage2-samples 8000 --output-dir data
```

Long-running CUDA-validated generation can write Parquet checkpoints and resume
from the last complete shard:

```bash
python generate_data.py stage2 --samples 15000 --output-dir /path/to/datasets \
  --output-format parquet --checkpoint-every 100 --progress-every 100 --resume \
  --runtime-validate --validation-device cuda --validation-timeout 30
```

Every checkpoint interval creates an immutable Parquet shard under
`stage2.checkpoints/` and prints the accepted count, run rate, and ETA. A retry
with the same stage, seed, sample count, data source, and `--resume` loads the
contiguous shards and continues at the next scheduled cell. On completion the
shards are consolidated into `stage2.parquet`; the statistics file remains
`stage2_stats.json`.

An existing checkpoint target may also be increased with `--resume`, provided
the stage, generator profile, seed, and data source are unchanged. The existing
contiguous prefix is retained and generation continues to the larger target.
Decreasing a checkpoint target remains an error.

Python API:

```python
from generate_data import generate_stage1_data, generate_stage2_data

generate_stage1_data(1000, output="data/stage1.json")
generate_stage2_data(8000, output="data/stage2.json")
```

## Current fidelity and limitations

- Uses exact-shape binary pointwise constraints rather than full broadcasting.
- Uses an explicit common-dimension CP domain including regular and irregular sizes.
- Tensor-size bounds are enforced inside CP-SAT.
- FLOP bounds should be added per operator if exact reproduction of a specific paper configuration is required.
- Generated code is syntax validated. For full validation, execute each sample in the same CUDA environment used by KernelBench.

## Dataset-quality constraints

This revision additionally enforces:

- random DAGs have no annotated fusion region;
- every compute node contributes to the returned output;
- ordinary dimensions cannot collapse to singleton size 1;
- reduction/softmax/scan rows and widths have minimum workloads;
- pointwise, pooling, normalization, matrix and convolution targets have minimum output sizes;
- convolution channels are sampled from 4 through 256 using common channel sizes;
- input creation uses 90% random tensors, 5% ones and 5% zeros;
- division denominators use positive random values to avoid division by zero without creating identity operations.

## Distribution controls

Each generated sample is assigned one of three workload buckets (`small`, `medium`, `large`). The accepted-sample schedule balances structured/random samples, program operator count, structured anchor family, and workload bucket before CSP generation. Failed CSP attempts do not consume quota.

Division denominators use `torch.rand(...) + 0.5`. Shape variables use semantic axis domains instead of one shared dimension domain.

Stage 1 programs contain 2–5 operators. Every structured Stage 1 program marks
all of its operators as the fusion target. Two-operator anchored programs use
either `prologue` or `epilogue`; programs with three or more operators can also
use `prologue_anchor_epilogue`. Pointwise and supported reduction chains retain
the `chain` category.

For every Stage 1 anchor family, the non-chain anchored samples use a global
40% `prologue`, 40% `epilogue`, and 20% `prologue_anchor_epilogue` split.
Two-operator cells are restricted to the first two categories, while the
combined quota is spread across 3–5-op cells. Family and operator-count pairs
are balanced jointly so every family covers every legal category. Pointwise
chains and the 25% reduction-chain share are outside the 40/40/20 calculation.

## Shape domains

The solver uses KernelBench-inspired semantic domains rather than one generic dimension list. Batch, channel, spatial, feature, scan, and matrix axes have separate value sets. The workload bucket limits the largest legal values, so `small` and `medium` samples remain practical while `large` can exercise wider scans, larger matrices, higher channel counts, and standard vision resolutions.

## Shared workload distribution and safety

The warm-up schedule uses 30% small, 45% medium, and 25% large samples. Hard
per-edge and total-program element caps prevent semantic domains from producing
oversized context tensors. Direct unbounded accumulation-to-`exp2` pairs are
rejected before solving.

## Generation-time runtime validation

Runtime validation is optional and disabled by default. When enabled, every candidate is executed in a fresh subprocess before it is accepted. Failed, timed-out, non-finite, or nondeterministic candidates are rejected and regenerated without consuming their quota cell.

CPU validation:

```bash
python generate_data.py stage1 --samples 1000 --runtime-validate --validation-device cpu
```

CUDA validation:

```bash
python generate_data.py stage2 --samples 8000 --runtime-validate --validation-device cuda --validation-timeout 30
```

The validator checks model construction, `get_inputs()`, repeated forward execution, returned values, and—by default—every tensor assignment in `Model.forward`. Use `--no-intermediate-check` only when output-only checking is desired.


## Prompt-conditioned fusion regions

Structured Stage1 and Stage2 samples include the complete ordered fusion region directly in the user prompt, for example:

```text
add -> layer_norm -> clamp -> leaky_relu
```

Non-library regions receive strong guidance to implement true fusion rather than leaving the listed operations as separate launches. Convolution and matrix regions include a library-aware exception so cuDNN/cuBLAS-style operations may be retained while surrounding epilogues are fused. Random samples do not receive a reference fusion region.
