#!/usr/bin/env bash
set -euo pipefail

# Run KernelGYM sequential test-time scaling inference/evaluation on KernelBench problems.
# This uses the parquet produced by scripts/kernelbench_evaluation/build_kernelbench_for_kernelgym.py.

KERNELGYM_ROOT="${KERNELGYM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${KERNELGYM_ROOT}/.." && pwd)}"
DRKERNEL_ROOT="${KERNELGYM_ROOT}/rl-training"
KERNELGYM_EVAL_DIR="${DRKERNEL_ROOT}/kernel/scripts/eval"
RUN_CACHE="${RUN_CACHE:-$WORKSPACE_ROOT/cache/kernelgym/kernelbench_tts/${SLURM_JOB_ID:-manual}}"

mkdir -p "$RUN_CACHE"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_CACHE/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUN_CACHE/triton}"
export TORCH_HOME="${TORCH_HOME:-$RUN_CACHE/torch}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$RUN_CACHE/nv}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RUN_CACHE/vllm}"

RUN_NAME="${RUN_NAME:-kernelbench_tts_smoke}"
PROJECT_NAME="${PROJECT_NAME:-kernelgym-kernelbench-tts}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$RUN_NAME}"

EVAL_DATASET="${EVAL_DATASET:-${EVAL_PIPELINE_ROOT:-${KERNELGYM_ROOT}/eval-pipeline}/datasets/kernelbench_level1_for_kernelgym.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_ROOT/Results/Kernelbench/tts_runs/$RUN_NAME}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/graded_results.parquet}"
METRICS_OUTPUT_PATH="${METRICS_OUTPUT_PATH:-$OUTPUT_DIR/metrics.json}"
RAW_RESPONSE_PATH="${RAW_RESPONSE_PATH:-$OUTPUT_DIR/raw_responses.jsonl}"
DATAPROTO_PATH="${DATAPROTO_PATH:-}"

MODEL_NAME="${MODEL_NAME:-drkernel-8b-coldstart}"
if [[ -z "${MODEL_PATH:-}" ]]; then
  COLDSTART_MODEL_PATH="${COLDSTART_MODEL_PATH:-$WORKSPACE_ROOT/models/sft-cold-start}"
  if [[ -d "$COLDSTART_MODEL_PATH" ]]; then
    MODEL_PATH="$COLDSTART_MODEL_PATH"
  else
    MODEL_PATH="hkust-nlp/drkernel-8b-coldstart"
  fi
fi

MULTI_TURN="${MULTI_TURN:-True}"
MAX_USER_TURNS="${MAX_USER_TURNS:-3}"
MULTI_ITERATION="${MULTI_ITERATION:-False}"

N_SAMPLES="${N_SAMPLES:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-20480}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1000))}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
MIN_P="${MIN_P:-0.0}"
DO_SAMPLE="${DO_SAMPLE:-True}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-True}"

ROLLOUT_MODE="${ROLLOUT_MODE:-async_vllm}"
BACKEND="${BACKEND:-vllm}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-8}}"
ROLLOUT_GPU_MEMORY_UTIL="${ROLLOUT_GPU_MEMORY_UTIL:-0.5}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"

OPENAI_MODEL="${OPENAI_MODEL:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-120}"
OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-3}"
OPENAI_MAX_CONCURRENCY="${OPENAI_MAX_CONCURRENCY:-8}"

REFERENCE_BACKEND="${REFERENCE_BACKEND:-kernelbench}"
REWARD_MANAGER="${REWARD_MANAGER:-kernel_async}"
REWARD_FUNC_NAME="${REWARD_FUNC_NAME:-calculate_reward_speedup}"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-${KERNELGYM_SERVER_URL:-}}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.3_0.4_0.3}"
REWARD_RATE_LIMIT="${REWARD_RATE_LIMIT:-2}"
REWARD_MAX_CONCURRENT="${REWARD_MAX_CONCURRENT:-2}"
REWARD_TASK_TIMEOUT="${REWARD_TASK_TIMEOUT:-600}"
REWARD_TASK_TIMEOUT_CLIENT="${REWARD_TASK_TIMEOUT_CLIENT:-1800}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-1800}"
REWARD_MAX_RETRIES="${REWARD_MAX_RETRIES:-3}"
NUM_CORRECT_TRIALS="${NUM_CORRECT_TRIALS:-5}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-10}"
SPEEDUP_REWARD_UPPER_BOUND="${SPEEDUP_REWARD_UPPER_BOUND:-3.0}"
CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-kernel/rewards/kernel_reward.py}"
CUSTOM_REWARD_NAME="${CUSTOM_REWARD_NAME:-compute_kernel_reward_batch}"

SOLVE_THRESHOLD="${SOLVE_THRESHOLD:-0.99}"
PASS_AT_K="${PASS_AT_K:-1}"

FSDP_SIZE="${FSDP_SIZE:--1}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
FIX_QWEN3_CHAT_TEMPLATE="${FIX_QWEN3_CHAT_TEMPLATE:-False}"

GRADIO_VISUALIZATION="${GRADIO_VISUALIZATION:-False}"
GRADIO_SHARE="${GRADIO_SHARE:-False}"
VISUALIZE_ONLY="${VISUALIZE_ONLY:-False}"

source "$KERNELGYM_EVAL_DIR/grading_common.sh"

mkdir -p "$OUTPUT_DIR"

if [[ -z "$MODEL_PATH" && "$BACKEND" != "openai" && "$BACKEND" != "openai_sdk" ]]; then
  echo "Error: set MODEL_PATH=/path/to/model for vLLM/async_vLLM rollout." >&2
  exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME="$MODEL_PATH"
fi

if [[ -z "$REWARD_SERVER_URL" ]]; then
  echo "Error: set REWARD_SERVER_URL or KERNELGYM_SERVER_URL for kernel feedback/eval." >&2
  exit 1
fi

if [[ ! -f "$EVAL_DATASET" ]]; then
  echo "Error: EVAL_DATASET does not exist: $EVAL_DATASET" >&2
  echo "Build it with: KernelBench/.venv/bin/python scripts/kernelbench_evaluation/build_kernelbench_for_kernelgym.py --levels 1 --problem-ids 1 --output-dir results/kernelbench/kernelbench_tts_eval_inference_data/datasets_smoke_one_problem" >&2
  exit 1
fi

echo "=========================================="
echo "KernelBench -> KernelGYM TTS smoke eval"
echo "=========================================="
echo "Run: $RUN_NAME"
echo "Dataset: $EVAL_DATASET"
echo "Output: $OUTPUT_DIR"
echo "Model: $MODEL_PATH"
echo "Backend: $BACKEND"
echo "Rollout mode: $ROLLOUT_MODE"
echo "Samples: $N_SAMPLES batch_size=$BATCH_SIZE"
echo "Multi-turn: $MULTI_TURN max_user_turns=$MAX_USER_TURNS"
echo "Reward server: $REWARD_SERVER_URL"
echo "Reference backend: $REFERENCE_BACKEND"
echo "=========================================="

cd "$DRKERNEL_ROOT"
main "$@"
