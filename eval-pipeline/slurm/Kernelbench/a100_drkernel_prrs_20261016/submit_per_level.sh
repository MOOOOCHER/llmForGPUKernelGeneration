#!/usr/bin/env bash
# Merge one RL checkpoint, then submit three KernelBench evaluation chains:
#
#   merge checkpoint -> level 1 TTS -> export best kernels -> enroot evaluation
#                    -> level 2 TTS -> export best kernels -> enroot evaluation
#                    -> level 3 TTS -> export best kernels -> enroot evaluation
#
# Usage:
#   bash submit_kernelbench_tts_per_level.sh CHECKPOINT_RUN_DIR [MODEL_NAME]
#
# Example:
#   bash submit_kernelbench_tts_per_level.sh \
#     /.gpfs/scratch/zongxiong.chen/daniel/checkpoints/drkernel-8b-rl-example \
#     drkernel-8b-rl-step50
# CHECKPOINT_STEP=25 bash Scripts/kernelbench_evaluation/shell/submit_kernelbench_tts_per_level.sh  /path/to/checkpoints/run-directory   model-name
# Sampling and evaluation settings can be overridden through environment
# variables, for example:
#   N_SAMPLES=3 MAX_USER_TURNS=3 EXPORT_TURNS_PER_TRAJECTORY=1 \
#     KB_NUM_PERF_TRIALS=50 \
#     bash submit_kernelbench_tts_per_level.sh /path/to/checkpoint-run my-model

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/.gpfs/scratch/zongxiong.chen/daniel}"
TTS_SLURM="/.gpfs/scratch/zongxiong.chen/daniel/slurm/Kernelbench/a100_drkernel_prrs_20261016/tts_a100.slurm"
EVAL_SLURM="/.gpfs/scratch/zongxiong.chen/daniel/slurm/Kernelbench/a100_drkernel_prrs_20261016/eval_a100.sbatch"
MERGE_SLURM="${REPO_ROOT}/slurm/merge/merge_checkpoints.slurm"
PYBIN="${REPO_ROOT}/.venv-drkernel/bin/python"

CHECKPOINT_RUN_DIR="${1:-${RUN_DIR:-}}"
MODEL_NAME="${2:-${MODEL_NAME:-}}"

if [[ -z "${CHECKPOINT_RUN_DIR}" ]]; then
  echo "Usage: $0 CHECKPOINT_RUN_DIR [MODEL_NAME]" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT_RUN_DIR}" ]]; then
  echo "ERROR: checkpoint run directory does not exist: ${CHECKPOINT_RUN_DIR}" >&2
  exit 1
fi
if [[ -z "${MODEL_NAME}" ]]; then
  MODEL_NAME="$(basename "${CHECKPOINT_RUN_DIR%/}")"
fi

N_SAMPLES="${N_SAMPLES:-3}"
MAX_USER_TURNS="${MAX_USER_TURNS:-3}"
EXPORT_SELECT="${EXPORT_SELECT:-best_correct_or_compiled}"
EXPORT_SCOPE="${EXPORT_SCOPE:-trajectory}"
EXPORT_TURNS_PER_TRAJECTORY="${EXPORT_TURNS_PER_TRAJECTORY:-1}"
KB_NUM_CORRECT_TRIALS="${KB_NUM_CORRECT_TRIALS:-5}"
KB_NUM_PERF_TRIALS="${KB_NUM_PERF_TRIALS:-100}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/Results/Kernelbench/kernelgym_datasets}"
TTS_RUNS_ROOT="${TTS_RUNS_ROOT:-${REPO_ROOT}/Results/Kernelbench/tts_runs}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/Results/Kernelbench/evaluation_results}"
ENROOT_ROOT="${ENROOT_ROOT:-/sc/scratch/zongxiong.chen/daniel/.enroot}"
TTS_TIME_LIMIT="${TTS_TIME_LIMIT:-}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-}"
MERGE_TIME_LIMIT="${MERGE_TIME_LIMIT:-}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
MERGE_OVERWRITE="${MERGE_OVERWRITE:-0}"
# Reuse an already-submitted merge/export job instead of submitting a second
# writer for the same model directory. Downstream TTS jobs retain afterok on
# this job, so the normal merge -> TTS -> enroot ordering is unchanged.
MODEL_READY_JOB_ID="${MODEL_READY_JOB_ID:-}"
MODEL_ALREADY_READY="${MODEL_ALREADY_READY:-0}"
SBATCH_PARTITION="${SBATCH_PARTITION:-aisc-shortrun,aisc-batch}"

MODEL_TAG="$(printf '%s' "${MODEL_NAME}" | sed -E 's#[^A-Za-z0-9._-]+#_#g')"
RUN_GROUP_BASE="${RUN_GROUP_BASE:-kbtts_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)}"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-${REPO_ROOT}/models/${RUN_GROUP_BASE}-merged}"
MODEL_PATH="${MERGED_MODEL_PATH}"
TTS_OUTPUT_ROOT="${TTS_OUTPUT_ROOT:-${TTS_RUNS_ROOT}/${RUN_GROUP_BASE}}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${RESULTS_ROOT}/${RUN_GROUP_BASE}}"

for required_file in "${MERGE_SLURM}" "${TTS_SLURM}" "${EVAL_SLURM}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: missing Slurm script: ${required_file}" >&2
    exit 1
  fi
done

mkdir -p \
  "${REPO_ROOT}/logs/merge" \
  "${REPO_ROOT}/logs/kernelbench-eval" \
  "${TTS_OUTPUT_ROOT}" \
  "${EVAL_OUTPUT_ROOT}"

missing_levels=()
for level in 1 2 3; do
  if [[ ! -f "${DATASET_DIR}/kernelbench_level${level}_for_kernelgym.parquet" ]]; then
    missing_levels+=("${level}")
  fi
done
if [[ ${#missing_levels[@]} -gt 0 ]]; then
  echo "Building missing KernelGYM datasets for levels: ${missing_levels[*]}"
  "${PYBIN}" \
    "${REPO_ROOT}/Scripts/kernelbench_evaluation/python/build_kernelbench_for_kernelgym.py" \
    --levels "${missing_levels[@]}" \
    --backend triton \
    --prompt-option one_shot \
    --precision fp32 \
    --no-combined \
    --output-dir "${DATASET_DIR}"
fi

echo "=========================================="
echo "KernelBench per-level TTS evaluation"
echo "Checkpoint:  ${CHECKPOINT_RUN_DIR}"
echo "Step:        ${CHECKPOINT_STEP:-latest}"
echo "Merged model:${MODEL_PATH}"
echo "Model name:  ${MODEL_NAME}"
echo "Run base:    ${RUN_GROUP_BASE}"
echo "TTS outputs: ${TTS_OUTPUT_ROOT}/{l1,l2,l3}"
echo "Eval results:${EVAL_OUTPUT_ROOT}/{l1,l2,l3}"
echo "Sampling:    ${N_SAMPLES} trajectories, ${MAX_USER_TURNS} turns"
echo "Selection:   ${EXPORT_SELECT} (${EXPORT_SCOPE}, ${EXPORT_TURNS_PER_TRAJECTORY} best turn/trajectory)"
echo "Datasets:    ${DATASET_DIR}"
echo "Model job:   ${MODEL_READY_JOB_ID:-submit a new merge}"
echo "=========================================="

if [[ "${MODEL_ALREADY_READY}" == "1" ]]; then
  for required_model_file in config.json model.safetensors.index.json tokenizer_config.json; do
    if [[ ! -f "${MODEL_PATH}/${required_model_file}" ]]; then
      echo "ERROR: MODEL_ALREADY_READY=1 but ${MODEL_PATH}/${required_model_file} is missing" >&2
      exit 1
    fi
  done
  merge_job_id=""
  echo "Using already-exported model: ${MODEL_PATH}"
elif [[ -n "${MODEL_READY_JOB_ID}" ]]; then
  if [[ ! "${MODEL_READY_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MODEL_READY_JOB_ID must be a numeric Slurm job ID" >&2
    exit 2
  fi
  merge_job_id="${MODEL_READY_JOB_ID}"
  echo "Reusing model export job: ${merge_job_id} -> ${MODEL_PATH}"
else
  merge_flags=(
    --parsable
    --job-name="kb-merge-${MODEL_TAG}"
    --export=ALL
  )
  if [[ -n "${MERGE_TIME_LIMIT}" ]]; then
    merge_flags+=(--time="${MERGE_TIME_LIMIT}")
  fi

  merge_job_id="$(
    env \
      RUN_DIR="${CHECKPOINT_RUN_DIR}" \
      STEP="${CHECKPOINT_STEP}" \
      TARGET_DIR="${MODEL_PATH}" \
      OVERWRITE="${MERGE_OVERWRITE}" \
      sbatch "${merge_flags[@]}" "${MERGE_SLURM}"
  )"
  echo "Merge job: ${merge_job_id} -> ${MODEL_PATH}"
fi

declare -a tts_job_ids=()
declare -a eval_job_ids=()

for level in 1 2 3; do
  level_group="${RUN_GROUP_BASE}-l${level}"
  tts_output_dir="${TTS_OUTPUT_ROOT}/l${level}"
  eval_result_dir="${EVAL_OUTPUT_ROOT}/l${level}"
  dataset="${DATASET_DIR}/kernelbench_level${level}_for_kernelgym.parquet"

  tts_flags=(
    --parsable
    --job-name="a100-${MODEL_TAG}-tts-l${level}"
    --partition="${SBATCH_PARTITION}"
    --export=ALL
  )
  if [[ -n "${merge_job_id}" ]]; then
    tts_flags+=(
      --dependency="afterok:${merge_job_id}"
      --kill-on-invalid-dep=yes
    )
  fi
  if [[ -n "${TTS_TIME_LIMIT}" ]]; then
    tts_flags+=(--time="${TTS_TIME_LIMIT}")
  fi

  tts_job_id="$(
    env \
      KB_LEVEL="${level}" \
      RUN_NAME="${level_group}_level${level}" \
      EVAL_DATASET="${dataset}" \
      MODEL_PATH="${MODEL_PATH}" \
      MODEL_NAME="${MODEL_NAME}" \
      OUTPUT_DIR="${tts_output_dir}" \
      N_SAMPLES="${N_SAMPLES}" \
      MAX_USER_TURNS="${MAX_USER_TURNS}" \
      sbatch "${tts_flags[@]}" "${TTS_SLURM}"
  )"
  tts_job_ids+=("${tts_job_id}")
  echo "SUBMITTED_TTS level=${level} job=${tts_job_id}"

  eval_flags=(
    --parsable
    --job-name="a100-${MODEL_TAG}-eval-l${level}"
    --partition="${SBATCH_PARTITION}"
    --export=ALL
    --dependency="afterok:${tts_job_id}"
    --kill-on-invalid-dep=yes
  )
  if [[ -n "${EVAL_TIME_LIMIT}" ]]; then
    eval_flags+=(--time="${EVAL_TIME_LIMIT}")
  fi

  eval_job_id="$(
    env \
      RUN_GROUP="${level_group}" \
      KB_LEVELS="${level}" \
      TTS_RUN_DIR="${tts_output_dir}" \
      RESULT_DIR="${eval_result_dir}" \
      DATASET_DIR="${DATASET_DIR}" \
      EXPORT_SELECT="${EXPORT_SELECT}" \
      EXPORT_SCOPE="${EXPORT_SCOPE}" \
      EXPORT_TURNS_PER_TRAJECTORY="${EXPORT_TURNS_PER_TRAJECTORY}" \
      KB_NUM_CORRECT_TRIALS="${KB_NUM_CORRECT_TRIALS}" \
      KB_NUM_PERF_TRIALS="${KB_NUM_PERF_TRIALS}" \
      ENROOT_CACHE_PATH="${ENROOT_ROOT}/cache" \
      ENROOT_DATA_PATH="${ENROOT_ROOT}/data" \
      ENROOT_RUNTIME_PATH="${ENROOT_ROOT}/runtime" \
      ENROOT_TEMP_PATH="${ENROOT_ROOT}/tmp" \
      sbatch "${eval_flags[@]}" "${EVAL_SLURM}"
  )"
  eval_job_ids+=("${eval_job_id}")
  echo "SUBMITTED_EVAL level=${level} job=${eval_job_id} parent=${tts_job_id}"

  echo "Level ${level}: TTS ${tts_job_id} -> enroot ${eval_job_id}"
done

echo
echo "Submitted all per-level chains."
echo "Monitor: squeue -u zongxiong.chen"
echo "Merge: ${merge_job_id} -> ${MODEL_PATH}"
for index in 0 1 2; do
  level=$((index + 1))
  level_group="${RUN_GROUP_BASE}-l${level}"
  echo "Level ${level}: TTS=${tts_job_ids[$index]} EVAL=${eval_job_ids[$index]}"
  echo "  Generated: ${TTS_OUTPUT_ROOT}/l${level}/"
  echo "  Results:   ${EVAL_OUTPUT_ROOT}/l${level}/"
done
