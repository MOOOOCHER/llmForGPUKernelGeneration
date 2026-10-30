#!/usr/bin/env bash
# Shared HPI runtime from slurm/RL/train_fusion_stage1_4gpu.slurm.
# Submit one of the named .slurm recipes; resource requests are in those files.

set -euo pipefail
set -x

REPO_ROOT="${REPO_ROOT:?Set by the named recipe}"
KERNELGYM_ROOT="${KERNELGYM_ROOT:?Set by the named recipe}"
DRKERNEL_ROOT="${KERNELGYM_ROOT}/rl-training"
KERNELGYM_WORKER_CONTAINER="${KERNELGYM_WORKER_CONTAINER:-${REPO_ROOT}/images/kernelbench-torch2.9-cu128.sqsh}"
KERNELGYM_ENROOT_ROOT="${KERNELGYM_ENROOT_ROOT:-/sc/scratch/zongxiong.chen/daniel/.enroot}"
export KERNELGYM_WORKER_CONTAINER KERNELGYM_ENROOT_ROOT
[[ -f "${KERNELGYM_WORKER_CONTAINER}" ]] || { echo "Missing evaluator image: ${KERNELGYM_WORKER_CONTAINER}"; exit 1; }

# ==== TUNABLES ============================================================
# Edit here, or override at submit time without touching the file:
#   sbatch --export=ALL,ALGORITHM=grpo,ROLLOUT_N=8 ...
TOTAL_GPUS_REQUIRED="${TOTAL_GPUS_REQUIRED:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"     # shared by training and KernelGYM

ALGORITHM="${ALGORITHM:-trloo}"                               # trloo | grpo | rloo
REWARD_FUNC_NAME="${REWARD_FUNC_NAME:-calculate_reward_speedup_fusion}"
FUSION_PROMPT_ENABLED="${FUSION_PROMPT_ENABLED:-False}"
# Turn-level profiling coverage rejection sampling; PR remains additive for retained correct turns.
COVERAGE_RS="${COVERAGE_RS:-null}"
COVERAGE_REWARD_ENABLE="${COVERAGE_REWARD_ENABLE:-True}"
COVERAGE_REWARD_WEIGHT="${COVERAGE_REWARD_WEIGHT:-0.25}"
REJECTION_SAMPLE="${REJECTION_SAMPLE:-True}"
# Baseline the generated kernel is timed against: pytorch (eager) | torch_compile.
# Required by kernel_reward.py but never declared in the trainer config, so the
# RL path can't score anything without it (every reward errors and every sample
# is filtered). See train_rl_common.sh.
REFERENCE_BACKEND="${REFERENCE_BACKEND:-pytorch}"
# Canonical structured Stage-1 Parquet. Reference programs are supplied without explicit fusion-plan guidance.
TRAIN_DATASET="${TRAIN_DATASET:-${REPO_ROOT}/datasets/stage1.parquet}"
EXPECTED_DATASET_ROWS="${EXPECTED_DATASET_ROWS:-10000}"
VALID_DATASET="${VALID_DATASET:-${DRKERNEL_ROOT}/data/hkust-nlp/drkernel-validation-data.parquet}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/no-filter}"
# Timed repetitions per kernel. KernelGYM is the wall-clock bottleneck (~6.0s
# per kernel at 100 trials, measured), and roughly half of that is timing, so 20
# trials cuts ~40% off timing. Fewer samples = noisier speedup reward, so this
# trades reward precision for throughput. Hardcoded twice in the 8B script until
# it was made overridable.
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-50}"
ENABLE_MULTI_TURN="${ENABLE_MULTI_TURN:-True}"
MAX_TURN="${MAX_TURN:-2}"
ROLLOUT_N="${ROLLOUT_N:-8}"                                   # samples per prompt
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"                     # divisible by 4 GPUs
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"               # must be <= TRAIN_BATCH_SIZE

# Validation. VAL_BEFORE_TRAIN runs the whole validation set before step 1.
# TEST_FREQ=0 disables validation entirely, including the one verl otherwise
# forces on the final step; the "Final validation metrics" success marker is
# still printed either way (it just reports None).
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
TEST_FREQ="${TEST_FREQ:-0}"

# Sequence budget. KV cache costs 144 KiB/token, so a max-length sequence here
# reducing this budget lowers both KV-cache and actor-update memory.
# They change results, so each setting gets its own checkpoint dir.
# 12288, not 10240: the dataset's own prompts are small (measured p50 1082, max
# 1337 tokens), but multi-turn accumulates -- by turn 2 the prompt carries the
# turn-1 response (up to MAX_RESPONSE_LENGTH) plus feedback. At 10240 that
# overflowed and was silently TRUNCATED, not rejected:
#   "exceeds max length (10725 > 10240). Truncating to 10240 tokens."
# which corrupts context on exactly the multi-turn rollouts TRLOO learns from.
# Budget at MAX_TURN=2: 1337 + 8192 + ~800 feedback ~= 10329, so 12288 leaves
# ~1.5k of margin. MAX_TURN=3 would need >= 20480.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-14336}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"

# vLLM's share of each 80GB H100 (weights ~16.4GB bf16 + KV cache).
# The ceiling is set by what FSDP already holds when vLLM initialises, and that
# depends on TRAIN_GPUS because the fp32 coldstart weights (32.8GB) are sharded:
#   TRAIN_GPUS=1 -> 32.8GB resident -> only 46GB free -> ceiling ~0.59
#   TRAIN_GPUS=4 ->  8.2GB resident -> 71GB free      -> ceiling ~0.90
# That ceiling only governs vLLM *startup*. The binding constraint is the
# update: measured evidence (jobs 2341655/2341674) shows vLLM does NOT hand its
# share back during the backward, despite FREE_CACHE_ENGINE=True. So the update
# must fit in what is left after vLLM's slice:
#   TRAIN_GPUS=4, util 0.75 -> params+grads 16.4 + logits ~2.8 + vLLM 59.4 = ~79 -> no slack
#   TRAIN_GPUS=4, util 0.55 -> ................................ + vLLM 43.6 = ~63 -> ~16GB slack
# Hence 0.55, not the upstream 0.75: upstream runs 8 GPUs, where params+grads
# are half again as small.
# KernelGYM shares each card, so reserve more headroom than the isolated-eval
# six-GPU launcher (0.55). This is intentionally conservative for the first run.
ROLLOUT_GPU_MEMORY_UTIL="${ROLLOUT_GPU_MEMORY_UTIL:-0.45}"
# True = hand the KV cache back during the update phase; False keeps it pinned.
# The 8B script hardcodes False (verl itself defaults True), which is fine on
# 8 GPUs but not here: with only 4 GPUs the FSDP param/grad shards double to
# ~4.1GB each, and False + util 0.75 leaves just 20GB for a backward pass that
# needs roughly that much. True lets rollout keep util 0.75 AND gives the
# update phase the full card, at the cost of re-init each step.
# If you set this back to False, drop ROLLOUT_GPU_MEMORY_UTIL to ~0.6.
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"

# Tokens per micro-batch in the update phase (actor.ppo_max_token_len_per_gpu,
# live because use_dynamic_bsz=True). Derived, never hand-set: verl's own config
# documents this as "n * max_prompt_length + max_response_length", so with n=1 it
# tracks the sequence budget above and one full-length sequence always fits in a
# micro-batch. Editing the lengths keeps this correct automatically.
# Leaving it unset would be worse than wrong-by-a-little: the auto-pick needs an
# uppercase "8B" and this model is "...-8b", so it silently lands on a 16384
# catch-all unrelated to the lengths actually in use.
# The enforced floor is only (sum / SP_SIZE); the sum sits well above that. To
# cut update-phase memory, lower MAX_RESPONSE_LENGTH rather than decoupling this.
PPO_MICRO_TOKEN="${PPO_MICRO_TOKEN:-22528}"

# Hard cap on optimizer steps; "null" would derive from TOTAL_EPOCHS instead.
# The upstream default is 1000 epochs / no step cap, i.e. effectively unbounded.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-2}"                                   # save every N steps
# Rolling window of checkpoints kept on disk; older ones are deleted as new
# ones are written. "null" keeps every checkpoint. Each one is roughly 100GB+
# (fp32 weights + optimizer state), so unbounded retention fills the disk fast.
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-2}"
# Consecutive batches where rejection sampling left nothing to train on before
# the run is treated as broken and aborts. A skipped batch costs rollouts but
# carries no gradient signal, so skipping is free correctness-wise; the counter
# resets on any usable batch, so a genuinely dead reward server still aborts.
# Scale with ROLLOUT_N: the smaller the rollout, the more often batches empty.
MAX_SKIP_STEPS="${MAX_SKIP_STEPS:-15}"
# ==========================================================================

# ALGORITHM and REWARD_FUNC_NAME have no CLI flag, so they pass via env (the
# training script reads them as ${VAR:-default}). The rest are passed as CLI
# flags below, which additionally folds them into RUN_NAME's hash so configs
# get their own checkpoint dir. ALGORITHM/REWARD_FUNC_NAME are not in that
# hash, so name the run after them to avoid resuming across algorithms.
export ALGORITHM REWARD_FUNC_NAME FUSION_PROMPT_ENABLED FREE_CACHE_ENGINE NUM_PERF_TRIALS TRAIN_DATASET
export COVERAGE_RS COVERAGE_REWARD_ENABLE COVERAGE_REWARD_WEIGHT REJECTION_SAMPLE
export REWARD_RATE_LIMIT=4
export REWARD_MAX_CONCURRENT=4
export RUN_NAME="${RUN_NAME:-drkernel-8b-fusion--4gpu-reconstructed-stage1-rewardfresh}"

# The env is a uv venv (the cluster disables pip, so rl-training/setup.sh can't be
# used as-is). Without this, python is the system one and torch is missing.
source "${VENV}/bin/activate"

# Rootless Redis: apt/sudo are unavailable, so the binaries were unpacked from
# .debs. KernelGYM's start script calls redis-server/redis-cli from PATH.
export LD_LIBRARY_PATH="${REPO_ROOT}/.local-redis/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="${REPO_ROOT}/.local-redis/root/usr/bin:${PATH}"

nvidia-smi
TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ "${TOTAL_GPUS}" -ne "${TOTAL_GPUS_REQUIRED}" ]]; then
  echo "ERROR: --gres gave ${TOTAL_GPUS} GPUs but this launcher requires ${TOTAL_GPUS_REQUIRED}"
  exit 1
fi
TRAIN_GPU_LIST="${GPU_LIST}"
EVAL_GPU_LIST="${GPU_LIST}"

# Configure all caches before KernelGym starts so the enroot workers inherit
# writable scratch-backed paths. The worker monitor passes and mounts these
# exact directories into every evaluator container.
export RAY_STORAGE_DIR="${REPO_ROOT}/.tmp/ray-${SLURM_JOB_ID}"
# Keep Ray's Unix-socket path short while storing all session data on scratch.
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}"
export TMPDIR="${REPO_ROOT}/.tmp/drkernel-${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/inductor"
export TRITON_CACHE_DIR="${TMPDIR}/triton"
export VLLM_CACHE_ROOT="${TMPDIR}/vllm"
export XDG_CACHE_HOME="${TMPDIR}/xdg"
export CUDA_CACHE_PATH="${TMPDIR}/cuda"
export PYTORCH_KERNEL_CACHE_PATH="${TMPDIR}/torch-kernels"
export TORCH_EXTENSIONS_DIR="${TMPDIR}/torch-extensions"
mkdir -p \
  "${RAY_STORAGE_DIR}" \
  "${TMPDIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${XDG_CACHE_HOME}" \
  "${CUDA_CACHE_PATH}" \
  "${PYTORCH_KERNEL_CACHE_PATH}" \
  "${TORCH_EXTENSIONS_DIR}"
[[ ! -e "${RAY_TMPDIR}" && ! -L "${RAY_TMPDIR}" ]]
ln -s "${RAY_STORAGE_DIR}" "${RAY_TMPDIR}"

# ---- Start KernelGYM (Redis + API + GPU workers) --------------------------
# Ask the OS for ports on the allocated compute node. Job-ID-derived ports have
# collided with unrelated services on shared nodes.
read -r REDIS_PORT API_PORT < <("${VENV}/bin/python" - <<'PY'
import socket

sockets = []
for _ in range(2):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    sockets.append(sock)
print(*(sock.getsockname()[1] for sock in sockets))
for sock in sockets:
    sock.close()
PY
)
export REDIS_PORT API_PORT
export REDIS_HOST=127.0.0.1
export API_HOST=0.0.0.0
export GPU_DEVICES="[${EVAL_GPU_LIST}]"
export NODE_ID="slurm-${SLURM_JOB_ID}"
export LOG_DIR="logs/kernelgym-${SLURM_JOB_ID}"
export KERNELGYM_ENV_FILE="${REPO_ROOT}/logs/rl-training/kernelgym-${SLURM_JOB_ID}.env"
export KERNELGYM_PID_FILE="${REPO_ROOT}/logs/rl-training/kernelgym-${SLURM_JOB_ID}.pids"
export KERNELGYM_TASK_ARCHIVE_DIR="${REPO_ROOT}/Results/RL-Training/_support/eval_artifacts/${SLURM_JOB_ID}"
mkdir -p "${KERNELGYM_TASK_ARCHIVE_DIR}"
MEMORY_LOG="${REPO_ROOT}/logs/rl-training/memory-${SLURM_JOB_ID}.log"
MEMORY_MONITOR_PID=""

cd "${KERNELGYM_ROOT}"
printf '%s\n' \
  "REDIS_HOST=${REDIS_HOST}" \
  "REDIS_PORT=${REDIS_PORT}" \
  "REDIS_KEY_PREFIX=kernelgym-${SLURM_JOB_ID}" \
  "KERNELGYM_TASK_ARCHIVE_DIR=${KERNELGYM_TASK_ARCHIVE_DIR}" \
  "API_HOST=${API_HOST}" \
  "API_PORT=${API_PORT}" \
  "GPU_DEVICES=[${EVAL_GPU_LIST}]" \
  "NODE_ID=${NODE_ID}" > "${KERNELGYM_ENV_FILE}"

start_memory_monitor() {
  local cgroup_rel cgroup_dir
  cgroup_rel=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
  cgroup_dir="/sys/fs/cgroup${cgroup_rel}"
  (
    while sleep 60; do
      printf '%s cgroup_current=%s cgroup_peak=%s events=%s\n' \
        "$(date --iso-8601=seconds)" \
        "$(if [[ -r "${cgroup_dir}/memory.current" ]]; then cat "${cgroup_dir}/memory.current"; else echo unavailable; fi)" \
        "$(if [[ -r "${cgroup_dir}/memory.peak" ]]; then cat "${cgroup_dir}/memory.peak"; else echo unavailable; fi)" \
        "$(if [[ -r "${cgroup_dir}/memory.events" ]]; then tr '\n' ',' < "${cgroup_dir}/memory.events"; else echo unavailable; fi)"
      ps -eo pid,ppid,rss,comm --sort=-rss | sed -n '1,15p'
    done
  ) >> "${MEMORY_LOG}" 2>&1 &
  MEMORY_MONITOR_PID=$!
}

cleanup() {
  echo "--- tearing down KernelGYM ---"
  if [[ -n "${MEMORY_MONITOR_PID:-}" ]]; then
    kill "${MEMORY_MONITOR_PID}" 2>/dev/null || true
    wait "${MEMORY_MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -f "${KERNELGYM_PID_FILE}" ]]; then
    while IFS= read -r pid; do
      [[ "${pid}" =~ ^[0-9]+$ ]] && kill "${pid}" 2>/dev/null || true
    done < "${KERNELGYM_PID_FILE}"
  fi
  redis-cli -p "${REDIS_PORT}" shutdown nosave 2>/dev/null || true
  rm -f "${KERNELGYM_ENV_FILE}" "${KERNELGYM_PID_FILE}" 2>/dev/null || true
  if [[ -n "${RAY_TMPDIR:-}" && -L "${RAY_TMPDIR}" && \
        "$(readlink "${RAY_TMPDIR}")" == "${RAY_STORAGE_DIR:-}" ]]; then
    rm -- "${RAY_TMPDIR}" 2>/dev/null || true
  fi
  if [[ -n "${RAY_STORAGE_DIR:-}" && \
        "${RAY_STORAGE_DIR}" == "${REPO_ROOT}/.tmp/ray-${SLURM_JOB_ID}" ]]; then
    rm -rf -- "${RAY_STORAGE_DIR}" 2>/dev/null || true
  fi
  if [[ -n "${TMPDIR:-}" && \
        "${TMPDIR}" == "${REPO_ROOT}/.tmp/drkernel-${SLURM_JOB_ID}" ]]; then
    rm -rf -- "${TMPDIR}" 2>/dev/null || true
  fi
  rm -rf "${KERNELGYM_ENROOT_ROOT}/jobs/kernelgym-${SLURM_JOB_ID}" 2>/dev/null || true
}
trap cleanup EXIT
start_memory_monitor

# Invoked via `bash`, not `./`: the repo's .sh files are checked out mode 600
# (no exec bit), so the README's ./start_all_with_monitor.sh gives
# "Permission denied".
bash start_all_with_monitor.sh \
  --env-file "${KERNELGYM_ENV_FILE}" \
  --log-dir "${LOG_DIR}" \
  --pid-file "${KERNELGYM_PID_FILE}"

# Wait for the service to actually answer before training starts.
KERNELGYM_SERVER_URL="http://127.0.0.1:${API_PORT}"
for i in $(seq 1 60); do
  curl -sf --max-time 5 "${KERNELGYM_SERVER_URL}/health" >/dev/null 2>&1 && break
  sleep 5
done
if ! curl -sf --max-time 5 "${KERNELGYM_SERVER_URL}/health" >/dev/null 2>&1; then
  echo "ERROR: KernelGYM did not become healthy at ${KERNELGYM_SERVER_URL}"
  echo "--- api_server.log ---"; tail -30 "${KERNELGYM_ROOT}/${LOG_DIR}/api_server.log" 2>/dev/null
  exit 1
fi
echo "KernelGYM healthy at ${KERNELGYM_SERVER_URL} on GPUs [${EVAL_GPU_LIST}]"
curl -s "${KERNELGYM_SERVER_URL}/workers/status" 2>/dev/null | head -c 400; echo
export KERNELGYM_SERVER_URL

# ---- Model / data / checkpoints ------------------------------------------
# MODEL_PATH overrides the script default of the HF id hkust-nlp/drkernel-8b,
# so training starts from the local coldstart weights instead.
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/sft-cold-start}"
export MODEL_NAME="drkernel-8b-coldstart"

"${VENV}/bin/python" - "${TRAIN_DATASET}" "${EXPECTED_DATASET_ROWS}" <<'PY'
import sys
import pyarrow.parquet as pq

path, expected = sys.argv[1], int(sys.argv[2])
actual = pq.read_metadata(path).num_rows
if actual != expected:
    raise SystemExit(f"dataset row mismatch: expected {expected}, got {actual}: {path}")
print(f"validated training dataset: {actual} rows in {path}")
PY

export HDFS_DATA_PATH="${KERNELGYM_ROOT}/rl-training/data"
export HDFS_CHECKPOINT_PATH="${REPO_ROOT}/checkpoints"
mkdir -p "${HDFS_CHECKPOINT_PATH}" "${CHECKPOINT_ROOT}"
LATEST_RESUME_CHECKPOINT=$(find "${CHECKPOINT_ROOT}" -maxdepth 1 \
  -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V | tail -1)
export CHECKPOINT_DIR_OVERRIDE="${CHECKPOINT_ROOT}"
echo "checkpoint root        : ${CHECKPOINT_DIR_OVERRIDE}"
if [[ -n "${LATEST_RESUME_CHECKPOINT}" ]]; then
  echo "latest resume step     : ${LATEST_RESUME_CHECKPOINT}"
else
  echo "latest resume step     : none (start from coldstart model)"
fi

# ---- Topology: all four GPUs are shared by training and evaluation --------
export N_GPUS_PER_NODE="${TOTAL_GPUS_REQUIRED}"
export NNODES=1
# Sequence parallel across the training GPUs. Must be passed as --sp_size below,
# NOT via env: the 8B script hardcodes SP_SIZE=4 with a plain assignment, so an
# exported value is silently ignored. Leaving it at 4 with fewer GPUs gives
# dp = world_size // sp = 0 and dies in init_device_mesh with
# "cannot reshape tensor of 0 elements".
SP_SIZE="${TOTAL_GPUS_REQUIRED}"

# train_rl_common.sh hardcodes trainer.logger=['console','wandb'], so wandb is
# not optional. Offline keeps it from needing an API key or network; metrics are
# written locally and can be `wandb sync`ed later. Set WANDB_API_KEY and
# WANDB_MODE=online if you want live dashboards.
# verl_patch/utils/tracking.py builds a ByteDance Merlin URL on the wandb path:
#   url = base_url + MERLIN_JOB_ID + "?..." + ARNOLD_MONITOR_TRIAL_ID
# Both are unset outside ByteDance, so it dies with
# "TypeError: can only concatenate str (not NoneType) to str" before step 1.
# Supplying any strings satisfies it; the resulting URL is inert metadata.
export MERLIN_JOB_ID="${MERLIN_JOB_ID:-slurm-${SLURM_JOB_ID}}"
export ARNOLD_MONITOR_TRIAL_ID="${ARNOLD_MONITOR_TRIAL_ID:-0}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${REPO_ROOT}/logs/rl-training"
export WANDB_PROJECT="${WANDB_PROJECT:-drkernel}"

# Training and KernelGYM intentionally share all four GPUs. Set this only after
# starting the workers so both services resolve the same Slurm-local indices.
# Some NVIDIA nodes (notably gx13v1) also inject ROCm visibility variables.
# VERL rejects ROCR/HIP visibility when CUDA visibility is present, so remove
# those irrelevant aliases before Ray snapshots and propagates the environment.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_LIST}"

# Run from rl-training/, NOT from kernel/scripts/rl as the README suggests:
# kernel_trainer.yaml's hydra searchpath is relative
# ("file://verl_patch/trainer/code/config"), so it only resolves when python's
# cwd is rl-training/. Launching from the script's own directory fails with
# "In 'kernel_trainer': Could not load 'ppo_trainer'".
cd "${DRKERNEL_ROOT}"

set +e
bash kernel/scripts/rl/8b_trloo_mrs_pr_prs.sh \
  --train_dataset "${TRAIN_DATASET}" \
  --valid_dataset "${VALID_DATASET}" \
  --enable_multi_turn "${ENABLE_MULTI_TURN}" \
  --max_turn "${MAX_TURN}" \
  --rollout_n "${ROLLOUT_N}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --ppo_mini_batch_size "${PPO_MINI_BATCH_SIZE}" \
  --sp_size "${SP_SIZE}" \
  --reference_backend "${REFERENCE_BACKEND}" \
  --rejection_sample "${REJECTION_SAMPLE}" \
  --val_before_train "${VAL_BEFORE_TRAIN}" \
  --n_val 2 \
  --test_freq "${TEST_FREQ}" \
  --rollout_gpu_memory_util "${ROLLOUT_GPU_MEMORY_UTIL}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_response_length "${MAX_RESPONSE_LENGTH}" \
  --ppo_micro_token "${PPO_MICRO_TOKEN}" \
  --total_epochs "${TOTAL_EPOCHS}" \
  --total_training_steps "${TOTAL_TRAINING_STEPS}" \
  --save_freq "${SAVE_FREQ}" \
  --max_ckpt_to_keep "${MAX_CKPT_TO_KEEP}" \
  --max_skip_steps "${MAX_SKIP_STEPS}"
TRAIN_RC=$?
set -e

# ---- Did it actually finish? ---------------------------------------------
# Exit 0 alone is not proof: the script's last act is `python -m kernel.main_kernel`
# and nothing downstream asserts the loop reached the end. verl's fit() forces a
# validation and a checkpoint save on the final step and then prints
# "Final validation metrics:" before returning (ray_trainer.py), so that line plus
# a global_step_<TOTAL_TRAINING_STEPS> directory is the real evidence.
# RUN_LOG lets a wrapper (e.g. smoke_8b_trloo.sbatch) point this at its own
# --output file, so the check below greps the log actually being written.
LOG="${RUN_LOG:-${REPO_ROOT}/logs/rl-training/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out}"
# -t (newest first), not plain ls: RUN_NAME is only a prefix -- the trainer
# appends a hash of the CLI args, so every config leaves its own directory here.
# Alphabetical order picks an arbitrary older run and reports "steps saved:"
# empty even when this run saved fine.
CKPT_ROOT="${CHECKPOINT_ROOT}"

echo "=============== RUN VERIFICATION ==============="
echo "train exit code : ${TRAIN_RC}"
grep -q "Final validation metrics" "${LOG}" 2>/dev/null \
  && FINAL_OK=yes || FINAL_OK=no
echo "reached final step (\"Final validation metrics\" logged): ${FINAL_OK}"
if [[ -n "${CKPT_ROOT}" ]]; then
  echo "checkpoint dir  : ${CKPT_ROOT}"
  echo "steps saved     : $(ls -d "${CKPT_ROOT}"/global_step_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
else
  echo "checkpoint dir  : NONE FOUND"
fi
if [[ "${TRAIN_RC}" -eq 0 && "${FINAL_OK}" == "yes" ]]; then
  echo "RESULT: RL_RUN_OK"
else
  echo "RESULT: RL_RUN_FAILED (rc=${TRAIN_RC}, final_metrics=${FINAL_OK}) - check the .err file"
fi
echo "================================================"
exit "${TRAIN_RC}"
