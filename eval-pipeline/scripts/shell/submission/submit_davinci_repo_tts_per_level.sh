#!/usr/bin/env bash
# Submit daVinci-kernel skill-aware TTS and dependent KernelBench evaluations.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/.gpfs/scratch/zongxiong.chen/daniel}"
MODEL_PATH="${1:-${MODEL_PATH:-${REPO_ROOT}/models/daVinci-kernel-8B-RL}}"
MODEL_NAME="${2:-${MODEL_NAME:-daVinci-kernel-8B-RL-skill}}"

TTS_SLURM="${REPO_ROOT}/slurm/kernelgym/tts_davinci_repo_kernelbench.slurm"
EVAL_SLURM="${REPO_ROOT}/slurm/Kernelbench/eval_kernelgym_tts_enroot.sbatch"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/Results/Kernelbench/kernelgym_datasets}"
# The shared library belongs to the DaVinci framework, not to any one model.
# Individual experiments may still override SKILL_LIBRARY_FILE.
SKILL_LIBRARY_FILE="${SKILL_LIBRARY_FILE:-${REPO_ROOT}/daVinci-kernel/davinci-kernel/kernel/skill/skill_library.jsonl}"

N_SAMPLES="${N_SAMPLES:-3}"
MAX_USER_TURNS="${MAX_USER_TURNS:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
KB_NUM_PERF_TRIALS="${KB_NUM_PERF_TRIALS:-20}"
SBATCH_PARTITION="${SBATCH_PARTITION:-aisc-shortrun,aisc-batch}"
SBATCH_EXCLUDE="${SBATCH_EXCLUDE:-}"

MODEL_TAG="$(printf '%s' "${MODEL_NAME}" | sed -E 's#[^A-Za-z0-9._-]+#_#g')"
RUN_GROUP_BASE="${RUN_GROUP_BASE:-kbtts_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)}"
TTS_OUTPUT_ROOT="${TTS_OUTPUT_ROOT:-${REPO_ROOT}/Results/Kernelbench/tts_runs/${RUN_GROUP_BASE}}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${REPO_ROOT}/Results/Kernelbench/evaluation_results/${RUN_GROUP_BASE}}"

for required_path in "${MODEL_PATH}" "${SKILL_LIBRARY_FILE}" "${TTS_SLURM}" "${EVAL_SLURM}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p \
  "${REPO_ROOT}/logs/kernelgym-davinci-tts/slurm" \
  "${REPO_ROOT}/logs/kernelgym-davinci-tts/wandb" \
  "${REPO_ROOT}/logs/kernelgym-davinci-tts/services" \
  "${TTS_OUTPUT_ROOT}" \
  "${EVAL_OUTPUT_ROOT}"

echo "=========================================="
echo "daVinci skill-aware KernelBench TTS"
echo "Model:       ${MODEL_PATH}"
echo "Skills:      ${SKILL_LIBRARY_FILE}"
echo "Run:         ${RUN_GROUP_BASE}"
echo "Sampling:    ${N_SAMPLES} trajectories, ${MAX_USER_TURNS} turns"
echo "Iterations:  ${MAX_ITERATIONS} (1 means no extra refinement loop)"
echo "Final trials:${KB_NUM_PERF_TRIALS}"
echo "=========================================="

declare -a tts_jobs=()
declare -a eval_jobs=()

for level in 1 2 3; do
  dataset="${DATASET_DIR}/kernelbench_level${level}_for_kernelgym.parquet"
  output_dir="${TTS_OUTPUT_ROOT}/l${level}"
  skill_export_dir="${output_dir}/skill_export"
  result_dir="${EVAL_OUTPUT_ROOT}/l${level}"

  if [[ ! -f "${dataset}" ]]; then
    echo "ERROR: missing dataset: ${dataset}" >&2
    exit 1
  fi

  tts_flags=(
    --parsable
    --job-name="kb-dv-skill-tts-l${level}"
    --partition="${SBATCH_PARTITION}"
    --export=ALL
  )
  if [[ -n "${SBATCH_EXCLUDE}" ]]; then
    tts_flags+=(--exclude="${SBATCH_EXCLUDE}")
  fi

  tts_job="$(
    env \
      KB_LEVEL="${level}" \
      EVAL_DATASET="${dataset}" \
      MODEL_PATH="${MODEL_PATH}" \
      SKILL_LIBRARY_FILE="${SKILL_LIBRARY_FILE}" \
      OUTPUT_DIR="${output_dir}" \
      RUN_NAME="${RUN_GROUP_BASE}-l${level}" \
      N_SAMPLES="${N_SAMPLES}" \
      MAX_USER_TURNS="${MAX_USER_TURNS}" \
      MAX_ITERATIONS="${MAX_ITERATIONS}" \
      sbatch "${tts_flags[@]}" "${TTS_SLURM}"
  )"
  tts_jobs+=("${tts_job}")

  eval_flags=(
    --parsable
    --job-name="kb-dv-skill-enroot-l${level}"
    --partition="${SBATCH_PARTITION}"
    --export=ALL
    --dependency="afterok:${tts_job}"
    --kill-on-invalid-dep=yes
  )
  if [[ -n "${SBATCH_EXCLUDE}" ]]; then
    eval_flags+=(--exclude="${SBATCH_EXCLUDE}")
  fi

  eval_job="$(
    env \
      RUN_GROUP="${RUN_GROUP_BASE}-l${level}" \
      KB_LEVELS="${level}" \
      TTS_RUN_DIR="${skill_export_dir}" \
      RESULT_DIR="${result_dir}" \
      DATASET_DIR="${DATASET_DIR}" \
      EXPORT_SELECT=best_correct_or_compiled \
      EXPORT_SCOPE=trajectory \
      EXPORT_TURNS_PER_TRAJECTORY=1 \
      KB_NUM_PERF_TRIALS="${KB_NUM_PERF_TRIALS}" \
      sbatch "${eval_flags[@]}" "${EVAL_SLURM}"
  )"
  eval_jobs+=("${eval_job}")

  echo "Level ${level}: TTS ${tts_job} -> enroot ${eval_job}"
done

echo
echo "Submitted ${RUN_GROUP_BASE}"
for index in 0 1 2; do
  level=$((index + 1))
  echo "Level ${level}: TTS=${tts_jobs[$index]} EVAL=${eval_jobs[$index]}"
done
