#!/bin/bash
# Run daVinci-kernel skill-aware test-time scaling for one KernelBench level.
# GPU 0 hosts the daVinci rollout; GPU 1 hosts the cloned KernelGym evaluator.

#SBATCH --job-name=kb-davinci-skill-tts
#SBATCH --partition=aisc-shortrun,aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --constraint=ARCH:X86
# gx13v1 currently exposes no usable CUDA device to TTS allocations.
#SBATCH --exclude=gx13v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=15:00:00
#SBATCH --output=/.gpfs/scratch/zongxiong.chen/daniel/logs/kernelgym-davinci-tts/slurm/%x-%j.log
#SBATCH --error=/.gpfs/scratch/zongxiong.chen/daniel/logs/kernelgym-davinci-tts/slurm/%x-%j.err

set -euo pipefail
set -x

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

REPO_ROOT="${REPO_ROOT:-/.gpfs/scratch/zongxiong.chen/daniel}"
DAVINCI_ROOT="${DAVINCI_ROOT:-${REPO_ROOT}/daVinci-kernel}"
DAVINCI_CODE_ROOT="${DAVINCI_CODE_ROOT:-${DAVINCI_ROOT}/davinci-kernel}"
DAVINCI_EVAL_SCRIPT="${DAVINCI_EVAL_SCRIPT:-${DAVINCI_CODE_ROOT}/kernel/scripts/eval/daVinci-kernel-14b-maxturns5-maxiter10-skill.sh}"

KB_LEVEL="${KB_LEVEL:-1}"
EVAL_DATASET="${EVAL_DATASET:-${REPO_ROOT}/Results/Kernelbench/kernelgym_datasets/kernelbench_level${KB_LEVEL}_for_kernelgym.parquet}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/daVinci-kernel-8B-RL}"
# Keep the skill library model-independent so any backbone can use the same
# DaVinci optimization knowledge. Callers may still override this explicitly.
SKILL_LIBRARY_FILE="${SKILL_LIBRARY_FILE:-${DAVINCI_CODE_ROOT}/kernel/skill/skill_library.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/Results/Kernelbench/tts_runs/kbtts_davinci-repo-skill/l${KB_LEVEL}}"
RUN_NAME="${RUN_NAME:-kbtts_davinci-repo-skill-l${KB_LEVEL}}"

N_SAMPLES="${N_SAMPLES:-2}"
MAX_USER_TURNS="${MAX_USER_TURNS:-5}"
# Skill injection requires daVinci's multi-iteration engine class. A value of 1
# selects that engine without performing any additional refinement iteration.
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
SKILL_VALIDATION_ONLY="${SKILL_VALIDATION_ONLY:-true}"
export SKILL_VALIDATION_ONLY
BATCH_SIZE="${BATCH_SIZE:-5}"
NUM_CORRECT_TRIALS="${NUM_CORRECT_TRIALS:-5}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-20}"
ROLLOUT_GPU_MEMORY_UTIL="${ROLLOUT_GPU_MEMORY_UTIL:-0.5}"
KERNELGYM_CONTAINER="${KERNELGYM_CONTAINER:-${REPO_ROOT}/images/kernelbench-torch2.9-cu128.sqsh}"
ENROOT_ROOT="${ENROOT_ROOT:-/sc/scratch/zongxiong.chen/daniel/.enroot}"
ENROOT_JOB_ROOT="${ENROOT_ROOT}/jobs/kernelgym-${SLURM_JOB_ID}"
DAVINCI_LOG_ROOT="${DAVINCI_LOG_ROOT:-${REPO_ROOT}/logs/kernelgym-davinci-tts}"

for required_path in \
  "${DAVINCI_EVAL_SCRIPT}" \
  "${EVAL_DATASET}" \
  "${MODEL_PATH}" \
  "${SKILL_LIBRARY_FILE}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 1
  fi
done
if [[ ! -f "${KERNELGYM_CONTAINER}" ]]; then
  echo "ERROR: KernelGym evaluator container does not exist: ${KERNELGYM_CONTAINER}" >&2
  exit 1
fi


source "${REPO_ROOT}/.venv-drkernel/bin/activate"
export LD_LIBRARY_PATH="${REPO_ROOT}/.local-redis/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="${REPO_ROOT}/.local-redis/root/usr/bin:${PATH}"
export PYTHONPATH="${DAVINCI_CODE_ROOT}:${DAVINCI_ROOT}:${PYTHONPATH:-}"

nvidia-smi
TOTAL_GPUS="$(nvidia-smi --list-gpus | wc -l)"
if [[ "${TOTAL_GPUS}" -lt 2 ]]; then
  echo "ERROR: daVinci TTS requires two visible GPUs, found ${TOTAL_GPUS}" >&2
  exit 1
fi

export REDIS_HOST=127.0.0.1
export REDIS_PORT=$((20000 + SLURM_JOB_ID % 10000))
export REDIS_KEY_PREFIX="davinci-kernelgym-${SLURM_JOB_ID}"
export API_HOST=0.0.0.0
export API_PORT=$((30000 + SLURM_JOB_ID % 10000))
export GPU_DEVICES="[1]"
export NODE_ID="slurm-${SLURM_JOB_ID}"
export MAX_CONCURRENT_TASKS=2
WORKER_ID="worker_gpu_1_${SLURM_JOB_ID}"

JOB_TMP="/tmp/davinci-kbtts-${SLURM_JOB_ID}"
RAY_TMP="/tmp/ray-davinci-kbtts-${SLURM_JOB_ID}"
SERVICE_LOG_DIR="${DAVINCI_LOG_ROOT}/services/${SLURM_JOB_ID}"
mkdir -p \
  "${JOB_TMP}" \
  "${RAY_TMP}" \
  "${SERVICE_LOG_DIR}" \
  "${OUTPUT_DIR}" \
  "${DAVINCI_LOG_ROOT}/wandb"

export TMPDIR="${JOB_TMP}"
export RAY_ADDRESS=local
export RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK:-32}"
export RAY_NAMESPACE="davinci-kbtts-${SLURM_JOB_ID}"
export RAY_TMPDIR="${RAY_TMP}"
export ENROOT_CACHE_PATH="${ENROOT_ROOT}/cache"
export ENROOT_DATA_PATH="${ENROOT_JOB_ROOT}/data"
export ENROOT_RUNTIME_PATH="${ENROOT_JOB_ROOT}/runtime"
export ENROOT_TEMP_PATH="${ENROOT_JOB_ROOT}/tmp"
# Ray otherwise detects every CPU on a shared node and prestarts one Python
# worker per node CPU.  Keep each local Ray cluster inside its Slurm allocation
# so colocated KernelBench jobs cannot exhaust the node during startup.
export RAY_OVERRIDE_RESOURCES="{\"CPU\": ${SLURM_CPUS_PER_TASK:-32}, \"GPU\": 1}"
# The evaluator runs inside enroot, so host-only /tmp paths are not visible
# there. Keep compiler caches under the Daniel scratch tree, which is mounted
# at the same absolute path in the worker container, and pass the variables
# explicitly to enroot below.
export TORCHINDUCTOR_CACHE_DIR="${REPO_ROOT}/.torchinductor-cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${REPO_ROOT}/.triton-cache/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${REPO_ROOT}/.cache"
export CUDA_CACHE_PATH="${REPO_ROOT}/.cache/cuda/${SLURM_JOB_ID}"
mkdir -p \
  "${ENROOT_CACHE_PATH}" \
  "${ENROOT_DATA_PATH}" \
  "${ENROOT_RUNTIME_PATH}" \
  "${ENROOT_TEMP_PATH}"
export VLLM_CACHE_ROOT="${JOB_TMP}/vllm"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${DAVINCI_LOG_ROOT}/wandb"
mkdir -p \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${CUDA_CACHE_PATH}" \
  "${VLLM_CACHE_ROOT}"

service_pids=()
cleanup() {
  set +e
  for pid in "${service_pids[@]:-}"; do
    if [[ "${pid}" =~ ^[0-9]+$ ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" shutdown nosave 2>/dev/null || true
  rm -rf "${JOB_TMP}" "${RAY_TMP}"
  rm -rf "${ENROOT_JOB_ROOT}"
}
trap cleanup EXIT

redis-server --port "${REDIS_PORT}" --daemonize yes

cd "${DAVINCI_ROOT}"
python -m kernelgym.server.api.server >"${SERVICE_LOG_DIR}/api_server.log" 2>&1 &
service_pids+=("$!")
python -m kernelgym.worker.worker_monitor --persistent >"${SERVICE_LOG_DIR}/worker_monitor.log" 2>&1 &
service_pids+=("$!")
enroot start \
  --root \
  --mount "${REPO_ROOT}:${REPO_ROOT}" \
  --env REDIS_HOST \
  --env REDIS_PORT \
  --env REDIS_KEY_PREFIX \
  --env API_HOST \
  --env API_PORT \
  --env GPU_DEVICES \
  --env NODE_ID \
  --env MAX_CONCURRENT_TASKS \
  --env PYTHONPATH \
  --env TRITON_CACHE_DIR \
  --env TORCHINDUCTOR_CACHE_DIR \
  --env XDG_CACHE_HOME \
  --env CUDA_CACHE_PATH \
  "${KERNELGYM_CONTAINER}" \
  python -m kernelgym.worker.single_worker \
    --worker-id "${WORKER_ID}" \
    --device cuda:1 \
    --persistent \
  >"${SERVICE_LOG_DIR}/worker_gpu_1.log" 2>&1 &
service_pids+=("$!")

KERNELGYM_SERVER_URL="http://127.0.0.1:${API_PORT}"
for _ in $(seq 1 60); do
  if curl -sf --max-time 5 "${KERNELGYM_SERVER_URL}/health" >/dev/null; then
    break
  fi
  sleep 5
done
if ! curl -sf --max-time 5 "${KERNELGYM_SERVER_URL}/health" >/dev/null; then
  echo "ERROR: cloned KernelGym did not become healthy" >&2
  tail -100 "${SERVICE_LOG_DIR}/api_server.log" || true
  tail -100 "${SERVICE_LOG_DIR}/worker_gpu_1.log" || true
  exit 1
fi
for _ in $(seq 1 60); do
  worker_exists="$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" \
    EXISTS "${REDIS_KEY_PREFIX}:worker:${WORKER_ID}" 2>/dev/null || true)"
  if [[ "${worker_exists}" == "1" ]]; then
    break
  fi
  sleep 5
done
worker_exists="$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" \
  EXISTS "${REDIS_KEY_PREFIX}:worker:${WORKER_ID}" 2>/dev/null || true)"
if [[ "${worker_exists}" != "1" ]]; then
  echo "ERROR: Torch 2.9 KernelGym evaluator worker did not register" >&2
  tail -100 "${SERVICE_LOG_DIR}/worker_gpu_1.log" || true
  exit 1
fi

export KERNELGYM_SERVER_URL
export REWARD_SERVER_URL="${KERNELGYM_SERVER_URL}"
export CUDA_VISIBLE_DEVICES=0

OUTPUT_PARENT="$(dirname "${OUTPUT_DIR}")"
OUTPUT_LEAF="$(basename "${OUTPUT_DIR}")"
mkdir -p "${OUTPUT_PARENT}"

cd "${DAVINCI_CODE_ROOT}"
HDFS_RUNS_PATH="${OUTPUT_PARENT}" \
bash "${DAVINCI_EVAL_SCRIPT}" \
  --run_name "${OUTPUT_LEAF}" \
  --eval_dataset "${EVAL_DATASET}" \
  --model_path "${MODEL_PATH}" \
  --library_file "${SKILL_LIBRARY_FILE}" \
  --skill_enable True \
  --skill_library_root "${JOB_TMP}/skill_library" \
  --max_user_turns "${MAX_USER_TURNS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --n_samples "${N_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --max_prompt_length 24576 \
  --max_response_length 8192 \
  --rollout_mode async_vllm \
  --rollout_tp 1 \
  --rollout_gpu_memory_util "${ROLLOUT_GPU_MEMORY_UTIL}" \
  --reward_server_url "${KERNELGYM_SERVER_URL}" \
  --reward_func_name calculate_reward_speedup \
  --reward_rate_limit 2 \
  --reward_max_concurrent 2 \
  --reward_timeout 1800 \
  --reward_task_timeout 600 \
  --num_correct_trials "${NUM_CORRECT_TRIALS}" \
  --num_perf_trials "${NUM_PERF_TRIALS}" \
  --nnodes 1 \
  --n_gpus_per_node 1 \
  --gradio_visualization False \
  --gradio_share False

GRADING_DIR="${OUTPUT_DIR}/grading_results"
if [[ ! -f "${GRADING_DIR}/graded_results_skill.parquet" ]]; then
  echo "ERROR: missing skill-pass results: ${GRADING_DIR}/graded_results_skill.parquet" >&2
  exit 1
fi
if [[ ! -d "${GRADING_DIR}/eval_outputs_skill" ]]; then
  echo "ERROR: missing skill-pass outputs: ${GRADING_DIR}/eval_outputs_skill" >&2
  exit 1
fi

# A configured skill engine is not sufficient: every trajectory must have
# actually selected and injected the requested skills. Fail before export so a
# no-skill fallback can never be mislabeled as a skill evaluation again.
SELECTION_TRAJECTORIES="${GRADING_DIR}/eval_outputs_skill/selection_trajectories.jsonl"
if [[ ! -f "${SELECTION_TRAJECTORIES}" ]]; then
  echo "ERROR: missing skill selection audit: ${SELECTION_TRAJECTORIES}" >&2
  exit 1
fi
python - "${SELECTION_TRAJECTORIES}" <<'PY'
import json
import sys

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
invalid = [
    row for row in rows
    if not row.get("injected")
    or len(row.get("selected_skills") or []) != 3
    or not row.get("skill_context")
]
if not rows or invalid:
    raise SystemExit(
        f"ERROR: skill injection audit failed: rows={len(rows)} invalid={len(invalid)} path={path}"
    )
print(f"Skill injection audit passed: rows={len(rows)} selected=3 injected=all")
PY

SKILL_EXPORT_DIR="${OUTPUT_DIR}/skill_export"
mkdir -p "${SKILL_EXPORT_DIR}"
ln -sfn "${GRADING_DIR}/eval_outputs_skill" "${SKILL_EXPORT_DIR}/eval_outputs"

echo "daVinci skill-pass TTS completed: ${SKILL_EXPORT_DIR}"
