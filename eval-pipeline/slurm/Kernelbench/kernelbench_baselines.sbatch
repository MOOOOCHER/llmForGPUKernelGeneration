#!/bin/bash
# KernelBench reference baselines: torch eager + torch.compile (inductor default),
# at bf16 and fp32, over levels 1-3, inside an enroot container via pyxis.
#
#   sbatch /.gpfs/scratch/zongxiong.chen/daniel/baselines/kernelbench_baselines.sbatch
#
# Override at submit time without editing this file:
#   sbatch --export=ALL,LEVELS=1,PRECISIONS=bf16 kernelbench_baselines.sbatch
#
# Timing is the whole point of this job, so it asks for a whole node's worth of
# CPU and the GPU is used exclusively -- a co-scheduled job sharing the GPU would
# make every number here meaningless.

#SBATCH --job-name=kb-baselines
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
# aisc-batch is mixed-arch; the container is an x86_64 image.
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One GPU: the driver times on cuda:0 only. More would sit idle.
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# 250 problems x 2 backends x 2 precisions = 1000 timings.
# Measured on the smoke test (job 2342366): 36 timings in 1.8 min, ~3s each.
# Do NOT extrapolate that to 50min -- the smoke ran only the first 3 problems of
# each level, which are the smallest. The back half of level 3 is MinGPT,
# Mamba2, UNet and NetVlad, where inductor compilation dominates and a single
# problem can take minutes. Estimate is ~2-4h; 8h leaves ~2x margin without
# asking for a 12h slot that would sit longer in the queue. If it still runs
# out, the driver saves after every problem and a resubmit resumes.
#SBATCH --time=08:00:00
# Slurm opens these before the script body runs, so logs/baselines/ must already
# exist at submit time; a mkdir here would be too late.
#SBATCH --output=/.gpfs/scratch/zongxiong.chen/daniel/logs/baselines/kb-baselines-%j.log
#SBATCH --error=/.gpfs/scratch/zongxiong.chen/daniel/logs/baselines/kb-baselines-%j.err

set -euo pipefail

REPO_ROOT=/.gpfs/scratch/zongxiong.chen/daniel
KB_ROOT="${REPO_ROOT}/KernelBench"
CONTAINER="${CONTAINER:-${REPO_ROOT}/images/kernelbench-torch2.9-cu128.sqsh}"

LEVELS="${LEVELS:-1,2,3}"
BACKENDS="${BACKENDS:-eager,compile}"
PRECISIONS="${PRECISIONS:-bf16,fp32}"
export LEVELS BACKENDS PRECISIONS
export NUM_TRIALS="${NUM_TRIALS:-100}"
export NUM_WARMUP="${NUM_WARMUP:-5}"
export HARDWARE_NAME="${HARDWARE_NAME:-H100_aisc}"
export KB_ROOT

# Inductor writes a compile cache per process; point it at scratch so it persists
# across resubmits (a warm cache makes a resumed compile pass much faster) and so
# it never lands in the container's ephemeral overlay.
export TORCHINDUCTOR_CACHE_DIR="${REPO_ROOT}/.torchinductor-cache"
export TRITON_CACHE_DIR="${REPO_ROOT}/.triton-cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "${REPO_ROOT}/logs/baselines"

srun \
  --container-image="${CONTAINER}" \
  --container-mounts="${REPO_ROOT}:${REPO_ROOT}" \
  --container-workdir="${KB_ROOT}" \
  --export=ALL \
  bash "${REPO_ROOT}/baselines/run_in_container.sh"
