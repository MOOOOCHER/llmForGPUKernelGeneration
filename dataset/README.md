# RL datasets

Synthetic PyTorch programs for training GPU kernel generation models. Both
datasets use structured computation graphs with tensor shapes selected through
constraint solving.

## Stage 1 — thesis training dataset

**Stage 1 is the dataset used in the thesis:** 10,000 programs with **2–5
operators**, focused on local fusion opportunities. It covers pointwise
operations, reductions, scans, normalization, softmax, pooling, matrix
operations, and convolutions.

The dataset combines **60% base Stage-1 motifs** with **40% compact motifs
from the Stage-2 recipe library**, providing varied compositions while keeping
programs short.

File: [`stage1/stage1.parquet`](stage1/stage1.parquet). Generation seed: **42**.

## Stage 2 — longer operator chains

Stage 2 contains **25,000 programs with 6–15 operators**. It combines workload
motifs into longer compositions, creating more operation boundaries and
dependencies to consider when generating kernels. Its workload families include
matrix operations, convolutions, pooling, normalization, softmax, and scans.

Each program length is represented by **2,500 programs**, giving equal coverage
of all lengths from 6 to 15.

File: [`stage2/stage2.parquet`](stage2/stage2.parquet). Generation seed: **43**.

## Format and accompanying files

Each Parquet record contains a model prompt, a reference PyTorch program, and
metadata describing its operators, computation graph, and tensor shapes.
Prompts provide the reference program without explicit fusion-plan guidance.
Both datasets contain 20% small, 40% medium, and 40% large workloads.

Each stage directory includes `*_stats.json` with distribution summaries and
`*_generation_metadata.json` with the generator profile, seed, and sample count.
See [`dataset_generation/`](../dataset_generation/README.md) for the generator.
