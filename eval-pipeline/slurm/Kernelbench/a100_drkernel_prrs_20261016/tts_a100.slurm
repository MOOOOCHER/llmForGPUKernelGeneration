#!/bin/bash
# KernelGYM test-time scaling (TTS) inference over one KernelBench level:
# multi-turn sampling with a local vLLM rollout of the model under evaluation,
# graded live by a KernelGYM reward server started inside this job.
#
#   sbatch --job-name=kb-tts-l1 --export=ALL,KB_LEVEL=1 \
#     /.gpfs/scratch/zongxiong.chen/daniel/slurm/kernelgym/tts_kernelbench.slurm
#
# Override at submit time without editing this file, e.g.:
#   sbatch --export=ALL,KB_LEVEL=2,MODEL_PATH=/path/to/merged,MODEL_NAME=rl-step25 ...
#
# GPU usage (single node): by default this splits 2 GPUs -- GPU 0 runs the vLLM
# rollout, GPUs 1..N-1 run the KernelGYM eval workers, so kernel timing gets its
# own GPU and the measured speedup (which drives per-turn feedback and the "best
# kernel" selection) is not corrupted by generation load. Note GPU 0 is NOT
# vLLM-only: verl parks its FSDP actor/ref replica (~32 GiB for the 8B model) on
# cuda:0 as well, so vLLM's util must leave room for it (see below). If only 1
# GPU is allocated the script falls back to sharing cuda:0 between the rollout
# and the eval workers at a lower util (like the LRZ setup); acceptable because
# the authoritative timing is the later KernelBench enroot re-eval on an idle GPU.
#
# Cluster notes (same as the RL job): the aisc account/QOS only runs on aisc-*
# partitions; aisc-batch is mixed-arch so ARCH:X86 is required for the x86_64
# venv; compute nodes cannot see the login node's /tmp, so all shared paths are
# on GPFS and all scratch paths are node-local per-job /tmp dirs.

#SBATCH --job-name=kb-tts
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --constraint=ARCH:X86&GPU_SKU:A100&GPU_MEM:80GB
# gx13v1 can allocate a nominal H100 GRES without exposing a usable CUDA
# device to the job, causing vLLM initialization to fail.
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Two H100s: GPU 0 for the rollout, GPU 1 for KernelGYM eval workers so kernel
# timing has its own GPU (see the GPU-usage note above). Drop to :1 to share a
# single GPU -- the script adapts.
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
# Level 1/2: 100 problems x N_SAMPLES trajectories x MAX_USER_TURNS turns.
# Generation on a dedicated H100 dominates; 11h is the requested envelope for
# the current 3x3 evaluation jobs. Level 3 has half the problems but heavier
# reference models.
#SBATCH --time=24:00:00
# Slurm opens these before the script body runs, so logs/kernelbench-eval/
# must already exist at submit time (the submit script creates it).
#SBATCH --output=/.gpfs/scratch/zongxiong.chen/daniel/logs/kernelbench-eval/%x-%j.log
#SBATCH --error=/.gpfs/scratch/zongxiong.chen/daniel/logs/kernelbench-eval/%x-%j.err

set -x

# Some mixed-architecture nodes export AMD visibility variables alongside
# CUDA_VISIBLE_DEVICES. VERL rejects that ambiguous GPU environment.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

REPO_ROOT=/.gpfs/scratch/zongxiong.chen/daniel
KERNELGYM_ROOT="${REPO_ROOT}/KernelGYM"
SCRATCH_ROOT=/sc/scratch/zongxiong.chen/daniel
# Set persistent, job-isolated compiler caches before KernelGYM starts any GPU
# workers. This prevents Triton from falling back to ~/.triton/cache and avoids
# cross-job cache locking or partial-write collisions.
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor-cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SCRATCH_ROOT}/.triton-cache/${SLURM_JOB_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

export VLLM_CONFIG_ROOT="${SCRATCH_ROOT}/.config/vllm"
export XDG_CONFIG_HOME="${SCRATCH_ROOT}/.config"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${SCRATCH_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export VLLM_NO_USAGE_STATS=1
mkdir -p "${VLLM_CONFIG_ROOT}" "${TRANSFORMERS_CACHE}"

# ==== TUNABLES ============================================================
KB_LEVEL="${KB_LEVEL:-1}"
EVAL_DATASET="${EVAL_DATASET:-${REPO_ROOT}/Results/Kernelbench/kernelgym_datasets/kernelbench_level${KB_LEVEL}_for_kernelgym.parquet}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/drkernel-8b}"
MODEL_NAME="${MODEL_NAME:-drkernel-8b}"
MODEL_TAG="$(printf '%s' "${MODEL_NAME}" | sed -E 's#[^A-Za-z0-9._-]+#_#g')"
RUN_NAME="${RUN_NAME:-kbtts_${MODEL_TAG}_level${KB_LEVEL}_${SLURM_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Results/Kernelbench/tts_runs/${RUN_NAME}}"

N_SAMPLES="${N_SAMPLES:-3}"          # parallel trajectories per problem
MAX_USER_TURNS="${MAX_USER_TURNS:-3}" # refinement turns per trajectory
BATCH_SIZE="${BATCH_SIZE:-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-24576}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"

# Reward measurement used for turn feedback and best-kernel selection only;
# the authoritative timing happens later in the KernelBench enroot re-eval
# with 100 trials on an otherwise idle GPU.
NUM_CORRECT_TRIALS="${NUM_CORRECT_TRIALS:-5}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-20}"
REFERENCE_BACKEND="${REFERENCE_BACKEND:-pytorch}"

# vLLM's gpu_memory_utilization -- the fraction of *total* GPU memory it
# reserves. GPU 0 is never vLLM-only: verl's AsyncActorRolloutRefWorker keeps
# its FSDP actor+ref replica (~32 GiB for the 8B model) resident on cuda:0, and
# in single-GPU mode the KernelGYM eval workers share cuda:0 too. vLLM aborts at
# startup unless util*total fits in free memory ("Free memory ... less than
# desired GPU memory utilization"), so the default is chosen below once the GPU
# count is known. Set ROLLOUT_GPU_MEMORY_UTIL in the environment to override.
# ==========================================================================

if [[ ! -f "${EVAL_DATASET}" ]]; then
  echo "ERROR: EVAL_DATASET does not exist: ${EVAL_DATASET}"
  echo "Build it with Scripts/kernelbench_evaluation/python/build_kernelbench_for_kernelgym.py"
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 1
fi

# The standard env is a uv venv (the cluster disables pip, so rl-training/setup.sh
# can't be used as-is). Qwen3.8-27B-FP8 uses the newer qwen3_5 architecture and
# needs its model-specific Torch/Transformers/vLLM stack.
PYTHON_ENV="${PYTHON_ENV:-${REPO_ROOT}/.venv-drkernel}"
if [[ "${MODEL_PATH}" == *"Qwen3.8-27B-FP8"* && -x "${REPO_ROOT}/.venv-qwen38-tts-fa/bin/python" ]]; then
  PYTHON_ENV="${REPO_ROOT}/.venv-qwen38-tts-fa"
fi
if [[ ! -x "${PYTHON_ENV}/bin/python" ]]; then
  echo "ERROR: Python environment is missing: ${PYTHON_ENV}"
  exit 1
fi
source "${PYTHON_ENV}/bin/activate"
# The checkout contains the verl package at rl-training/verl/verl; put that inner source root first so import verl resolves to the real package.
export PYTHONPATH="${KERNELGYM_ROOT}/rl-training/verl:${KERNELGYM_ROOT}/rl-training:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_ENV}/bin/python"
echo "Python environment: ${PYTHON_ENV}"

# Rootless Redis: apt/sudo are unavailable, so the binaries were unpacked from
# .debs. KernelGYM's start script calls redis-server/redis-cli from PATH.
export LD_LIBRARY_PATH="${REPO_ROOT}/.local-redis/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="${REPO_ROOT}/.local-redis/root/usr/bin:${PATH}"

nvidia-smi
ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm must set CUDA_VISIBLE_DEVICES}"
TOTAL_GPUS=$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.device_count() == 2; [(print("Allocated GPU", i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory), None if "A100" in torch.cuda.get_device_name(i) and torch.cuda.get_device_properties(i).total_memory > 75*1024**3 else (_ for _ in ()).throw(RuntimeError("Expected A100 80GB"))) for i in range(2)]'
echo "Allocated CUDA_VISIBLE_DEVICES=${ALLOCATED_CUDA_VISIBLE_DEVICES}"
if [[ "${TOTAL_GPUS}" -lt 1 ]]; then
  echo "ERROR: no GPU visible to the job"
  exit 1
fi
if [[ "${TOTAL_GPUS}" -ge 2 ]]; then
  # Preferred: rollout on GPU 0, KernelGYM eval workers on GPUs 1..N-1, so kernel
  # timing never shares a GPU with generation. GPU 0 still also holds verl's
  # actor/ref replica (~32 GiB), so vLLM cannot take all of it -> 0.5.
  EVAL_GPU_LIST=$(seq -s, 1 $((TOTAL_GPUS - 1)))
  ROLLOUT_GPU_MEMORY_UTIL="${ROLLOUT_GPU_MEMORY_UTIL:-0.5}"
else
  # Fallback: one GPU shared by the rollout and the eval workers. Keep vLLM util
  # low so the actor/ref replica, vLLM weights+KV cache, and the eval workers all
  # fit on cuda:0 (matches the single-GPU LRZ setup).
  EVAL_GPU_LIST=0
  ROLLOUT_GPU_MEMORY_UTIL="${ROLLOUT_GPU_MEMORY_UTIL:-0.25}"
fi
echo "GPUs=${TOTAL_GPUS} rollout=cuda:0 kernelgym_eval=[${EVAL_GPU_LIST}] vllm_util=${ROLLOUT_GPU_MEMORY_UTIL}"

# ---- Start KernelGYM (Redis + API + GPU workers) --------------------------
# Ports are derived from the job id so two jobs on one node don't collide.
export REDIS_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
export API_PORT=$(( 30000 + SLURM_JOB_ID % 10000 ))
export REDIS_HOST=127.0.0.1
export API_HOST=0.0.0.0
export GPU_DEVICES="[${EVAL_GPU_LIST}]"
export NODE_ID="slurm-${SLURM_JOB_ID}"
export LOG_DIR="logs/kernelgym-tts-${SLURM_JOB_ID}"
export KERNELGYM_RUNTIME_DIR="/tmp/kernelgym-tts-${SLURM_JOB_ID}"
export KERNELGYM_ENV_FILE="${KERNELGYM_RUNTIME_DIR}/kernelgym.env"
export KERNELGYM_PID_FILE="${KERNELGYM_RUNTIME_DIR}/kernelgym.pids"
mkdir -p "${KERNELGYM_RUNTIME_DIR}"

cd "${KERNELGYM_ROOT}"
# Write .env explicitly rather than letting auto_configure.sh guess ports/GPUs:
# it would hand KernelGYM every GPU, including the rollout one.
cat > "${KERNELGYM_ENV_FILE}" <<EOF
REDIS_HOST=${REDIS_HOST}
REDIS_PORT=${REDIS_PORT}
REDIS_KEY_PREFIX=kernelgym-${SLURM_JOB_ID}
API_HOST=${API_HOST}
API_PORT=${API_PORT}
GPU_DEVICES=[${EVAL_GPU_LIST}]
NODE_ID=${NODE_ID}
EOF

cleanup() {
  echo "--- tearing down KernelGYM ---"
  if [[ -f "${KERNELGYM_PID_FILE}" ]]; then
    while read -r pid; do
      if [[ "${pid}" =~ ^[0-9]+$ ]] && ps -p "${pid}" -o args= 2>/dev/null | grep -q "kernelgym\."; then
        kill "${pid}" 2>/dev/null || true
      fi
    done < "${KERNELGYM_PID_FILE}"
  fi
  redis-cli -p "${REDIS_PORT}" shutdown nosave 2>/dev/null
  rm -rf "${KERNELGYM_RUNTIME_DIR}" "/tmp/kbtts-${SLURM_JOB_ID}" "/tmp/ray-kbtts-${SLURM_JOB_ID}" 2>/dev/null
}
trap cleanup EXIT

# Invoked via `bash`, not `./`: the repo's .sh files are checked out mode 600
# (no exec bit).
bash start_all_with_monitor.sh \
  --env-file "${KERNELGYM_ENV_FILE}" \
  --log-dir "${LOG_DIR}" \
  --pid-file "${KERNELGYM_PID_FILE}"

KERNELGYM_SERVER_URL="http://127.0.0.1:${API_PORT}"
for i in $(seq 1 60); do
  curl -sf --max-time 5 "${KERNELGYM_SERVER_URL}/health" >/dev/null 2>&1 && break
  sleep 5
done

if ! curl -sf --max-time 5 "${KERNELGYM_SERVER_URL}/health" >/dev/null; then
  echo "ERROR: KernelGYM health check failed: ${KERNELGYM_SERVER_URL}/health"
  exit 1
fi
curl -sf --max-time 5 "${KERNELGYM_SERVER_URL}/health" | head -c 400
echo

export KERNELGYM_SERVER_URL
export REWARD_SERVER_URL="${KERNELGYM_SERVER_URL}"
export RAY_ADDRESS=local
export RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK:-32}"
export RAY_NAMESPACE="kbtts-${SLURM_JOB_ID}"
export RAY_TMPDIR="/tmp/ray-kbtts-${SLURM_JOB_ID}"
export TMPDIR="/tmp/kbtts-${SLURM_JOB_ID}"
export VLLM_CACHE_ROOT="${TMPDIR}/vllm"
mkdir -p \
  "${RAY_TMPDIR}" \
  "${TMPDIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}"

export MERLIN_JOB_ID="slurm-${SLURM_JOB_ID}"
export ARNOLD_MONITOR_TRIAL_ID=0
export ARNOLD_WORKER_NUM=1
export ARNOLD_WORKER_GPU=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${REPO_ROOT}/logs/kernelbench-eval"
export CUDA_VISIBLE_DEVICES="${ALLOCATED_CUDA_VISIBLE_DEVICES%%,*}"

export RUN_NAME EVAL_DATASET OUTPUT_DIR MODEL_PATH MODEL_NAME
export N_SAMPLES BATCH_SIZE TEMPERATURE TOP_P MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH
export NUM_CORRECT_TRIALS NUM_PERF_TRIALS REFERENCE_BACKEND ROLLOUT_GPU_MEMORY_UTIL
export MULTI_TURN=True MAX_USER_TURNS MULTI_ITERATION=False
export DO_SAMPLE=True ROLLOUT_MODE=async_vllm BACKEND=vllm
export N_GPUS_PER_NODE=1 NNODES=1
export PROJECT_NAME=kernelgym-kernelbench-tts
export EXPERIMENT_NAME="${RUN_NAME}"

set +e
bash "${REPO_ROOT}/Scripts/kernelbench_evaluation/shell/run_kernelgym_test_time_scaling_on_kernelbench.sh"
TTS_RC=$?
set -e

if [[ -f "${OUTPUT_DIR}/metrics.json" ]]; then
  "${PYTHON_BIN}" \
    "${REPO_ROOT}/Scripts/kernelbench_evaluation/python/summarize_kernelgym_tts_run.py" \
    "${OUTPUT_DIR}" || true
fi

echo "=============== RUN VERIFICATION ==============="
echo "tts exit code   : ${TTS_RC}"
echo "output dir      : ${OUTPUT_DIR}"
N_KERNEL_DIRS="$(
  find "${OUTPUT_DIR}/eval_outputs" -maxdepth 1 -type d \
    -name 'level_*_problem_*_sample_*' 2>/dev/null | wc -l
)"
EXPECTED_TRAJECTORIES="$(
  "${PYTHON_BIN}" -c \
    'import pyarrow.parquet as pq, sys; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows * int(sys.argv[2]))' \
    "${EVAL_DATASET}" "${N_SAMPLES}"
)"
echo "trajectory dirs : ${N_KERNEL_DIRS}"
echo "expected dirs   : ${EXPECTED_TRAJECTORIES}"
if [[ "${TTS_RC}" -eq 0 && "${N_KERNEL_DIRS}" -eq "${EXPECTED_TRAJECTORIES}" ]]; then
  echo "RESULT: TTS_OK"
  echo "================================================"
  exit 0
fi

echo "RESULT: TTS_FAILED (rc=${TTS_RC}, trajectory_dirs=${N_KERNEL_DIRS}, expected=${EXPECTED_TRAJECTORIES}) - check the .err file"
echo "================================================"
exit 1
