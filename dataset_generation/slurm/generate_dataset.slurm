#!/bin/bash
#SBATCH --job-name=dataset-generator
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --constraint=ARCH:X86
#SBATCH --exclude=gx13v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/sc/scratch/zongxiong.chen/daniel/logs/dataset/%x-%j.out
#SBATCH --error=/sc/scratch/zongxiong.chen/daniel/logs/dataset/%x-%j.err

set -euo pipefail

REPO_ROOT=${DATASET_WORKSPACE_ROOT:-/sc/scratch/zongxiong.chen/daniel}
KERNELGYM_REPO=${KERNELGYM_REPO:-${REPO_ROOT}/KernelGYM}
export DATASET_PYTHON=${DATASET_PYTHON:-${REPO_ROOT}/.venv-drkernel/bin/python}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# STAGE selects the profile; extra arguments override generation defaults.
exec bash "${KERNELGYM_REPO}/dataset_generation/generate.sh" "${STAGE:-}" "$@"
