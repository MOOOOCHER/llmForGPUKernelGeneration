# VERL source snapshot

This directory contains the VERL source included with the thesis submission.
It is tracked directly in this repository; no submodule initialization or
separate VERL checkout is required.

- Upstream: https://github.com/verl-project/verl
- Upstream commit: `c780fc34b45e01a1538d6386947585d4f7370bef`
- Snapshot date: 28 October 2026
- License: [Apache 2.0](LICENSE); upstream [Notice.txt](Notice.txt) is retained.

## Local changes

[local_changes.patch](local_changes.patch) records the changes against the
upstream commit. They are **already applied** to the included source:

- `verl/utils/dataset/multiturn_sft_dataset.py`: accept JSON-encoded list
  fields in Parquet data and handle missing optional values.
- `verl/utils/fsdp_utils.py`: select the named LoRA adapter when gathering
  parameters from FSDP modules.
- `verl/utils/vllm/utils.py`: make the LoRA hook installation idempotent and
  add optional adapter diagnostics controlled by `LORA_DIAGNOSTICS` and
  `LORA_DIAGNOSTICS_EXPECT_ZERO`.

The separate [`../verl_patch/`](../verl_patch/) directory contains the
DrKernel integration code and remains part of the training implementation.
Temporary backup files and local file-permission changes are excluded from
this snapshot. Upstream source, documentation, and license files are retained.

Install this source through [`../setup.sh`](../setup.sh), or run
`python -m pip install -e . --no-build-isolation --no-deps` from this directory
when the training dependencies are already installed.
