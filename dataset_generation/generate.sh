#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: generate.sh <stage1|stage2> [generator options]

Generates Parquet data with checkpoints and CUDA validation.
Defaults: stage1 = 10,000 samples / seed 42; stage2 = 25,000 / seed 43.
Each run writes to a fresh dataset/<stage>_<job-id-or-timestamp>/ folder.

Examples (from the repository root):
  ./dataset_generation/generate.sh stage1 --samples 4
  ./dataset_generation/generate.sh stage2 --samples 4 --output-dir dataset/test_stage2

Activate the Python environment first, or set DATASET_PYTHON.
Set DATASET_OUTPUT_DIR or pass --output-dir to select a folder (also for resuming).
All remaining arguments are passed to generate_data.py; use stage1 --help for details.
EOF
}

case "${1:-}" in
  stage1) SAMPLES=10000; SEED=42 ;;
  stage2) SAMPLES=25000; SEED=43 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
STAGE=$1
shift

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RUN_ID=${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%S)_$$}
OUTPUT_DIR=${DATASET_OUTPUT_DIR:-${REPO_DIR}/dataset/${STAGE}_${RUN_ID}}
PYTHON=${DATASET_PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

exec "${PYTHON}" -u "${SCRIPT_DIR}/generate_data.py" "${STAGE}" \
  --samples "${SAMPLES}" --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" --output-format parquet \
  --checkpoint-every 100 --progress-every 10 --resume \
  --runtime-validate --validation-device cuda --validation-timeout 30 \
  "$@"
