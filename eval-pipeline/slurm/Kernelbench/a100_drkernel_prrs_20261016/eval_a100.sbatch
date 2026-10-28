#!/bin/bash
# Stage 2 of the KernelBench TTS evaluation: take finished KernelGYM TTS runs,
# export the best kernel per problem into KernelBench run directories, re-time
# them in the clean KernelBench enroot environment, and compare against the
# locally measured torch eager / torch.compile baselines.
#
# Normally submitted by Scripts/kernelbench_evaluation/shell/
# submit_kernelbench_tts_per_level.sh with --dependency=afterok on the TTS
# jobs, but can be run standalone on existing TTS output:
#
#   sbatch --export=ALL,RUN_GROUP=kbtts_drkernel-8b-coldstart_20260717,KB_LEVELS=1,2,3 \
#     /.gpfs/scratch/zongxiong.chen/daniel/slurm/Kernelbench/eval_kernelgym_tts_enroot.sbatch
#
# Expects, per level L in KB_LEVELS:
#   TTS output:  ${TTS_RUNS_ROOT}/${RUN_GROUP}_level<L>/eval_outputs/
#   dataset:     ${DATASET_DIR}/kernelbench_level<L>_for_kernelgym.parquet
# Produces:
#   KernelBench/runs/${RUN_GROUP}_level<L>/            exported kernels + eval_results.json
#   Results/Kernelbench/evaluation_results/${RUN_GROUP}/   collected results + metrics json/md
# TTS_RUN_DIR and RESULT_DIR may override those legacy locations when one level
# is evaluated per job.
#
# Timing is the whole point of this job, so the GPU is used exclusively.

#SBATCH --job-name=kb-tts-eval
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
# aisc-batch is mixed-arch; the container is an x86_64 image.
#SBATCH --constraint=ARCH:X86&GPU_SKU:A100&GPU_MEM:80GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One GPU: the evaluator times on cuda:0 only. More would sit idle.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# Up to 250 kernels x (5 correctness + 100 perf trials) plus triton JIT per
# kernel. The baseline job's measured rate (~3s/timing, heavy tail on level 3)
# suggests ~2-4h; 8h leaves margin.
#SBATCH --time=12:00:00
#SBATCH --export=ALL,ENROOT_CACHE_PATH=/sc/scratch/zongxiong.chen/daniel/.enroot/cache
# Slurm opens these before the script body runs, so logs/kernelbench-eval/
# must already exist at submit time (the submit script creates it).
#SBATCH --output=/.gpfs/scratch/zongxiong.chen/daniel/logs/kernelbench-eval/%x-%j.log
#SBATCH --error=/.gpfs/scratch/zongxiong.chen/daniel/logs/kernelbench-eval/%x-%j.err

set -euo pipefail

REPO_ROOT=/.gpfs/scratch/zongxiong.chen/daniel
KB_ROOT="${REPO_ROOT}/KernelBench"
PYBIN="${REPO_ROOT}/.venv-drkernel/bin/python"
CONTAINER="${CONTAINER:-${REPO_ROOT}/images/kernelbench-torch2.9-cu128.sqsh}"
EVAL_SCRIPTS="${REPO_ROOT}/Scripts/kernelbench_evaluation"

# ==== TUNABLES ============================================================
: "${RUN_GROUP:?set RUN_GROUP (base name of the TTS runs, without _level<L>)}"
KB_LEVELS="${KB_LEVELS:-1,2,3}"
TTS_RUNS_ROOT="${TTS_RUNS_ROOT:-${REPO_ROOT}/Results/Kernelbench/tts_runs}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/Results/Kernelbench/kernelgym_datasets}"
# best_correct = correctness first, then measured TTS speedup; the other
# choices are best_score / best_compiled / last (see the exporter).
EXPORT_SELECT="${EXPORT_SELECT:-best_correct_or_compiled}"
EXPORT_SCOPE="${EXPORT_SCOPE:-trajectory}" # problem | trajectory
EXPORT_TURNS_PER_TRAJECTORY="${EXPORT_TURNS_PER_TRAJECTORY:-1}"
KB_BACKEND="${KB_BACKEND:-triton}"
KB_PRECISION="${KB_PRECISION:-fp32}"
KB_NUM_CORRECT_TRIALS="${KB_NUM_CORRECT_TRIALS:-5}"
KB_NUM_PERF_TRIALS="${KB_NUM_PERF_TRIALS:-100}"
KB_TIMEOUT="${KB_TIMEOUT:-300}"
KB_MEASURE_COMPILE_REFERENCE="${KB_MEASURE_COMPILE_REFERENCE:-true}"
# Optional comma-separated KernelBench problem IDs, e.g. "56,85".  Empty
# preserves the full-level evaluation used by standard TTS chains.
KB_PROBLEM_IDS="${KB_PROBLEM_IDS:-}"
# Baselines measured on this cluster's H100s by kernelbench_baselines.sbatch.
BASELINE_DIR="${BASELINE_DIR:-/.gpfs/scratch/zongxiong.chen/daniel/slurm/Kernelbench/a100_drkernel_prrs_20261016/paired_only_baselines}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/Results/Kernelbench/evaluation_results}"
TTS_RUN_DIR="${TTS_RUN_DIR:-}"
RESULT_DIR="${RESULT_DIR:-}"
ENROOT_ROOT="${ENROOT_ROOT:-/sc/scratch/zongxiong.chen/daniel/.enroot}"
ENROOT_JOB_ROOT="${ENROOT_ROOT}/jobs/${SLURM_JOB_ID}"
COMPILER_CACHE_ROOT="${REPO_ROOT}/.kernelbench-cache/${SLURM_JOB_ID}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${ENROOT_ROOT}/cache}"
export ENROOT_DATA_PATH="${ENROOT_JOB_ROOT}/data"
export ENROOT_RUNTIME_PATH="${ENROOT_JOB_ROOT}/runtime"
export ENROOT_TEMP_PATH="${ENROOT_JOB_ROOT}/tmp"
# ==========================================================================

mkdir -p \
  "${ENROOT_CACHE_PATH}" \
  "${ENROOT_DATA_PATH}" \
  "${ENROOT_RUNTIME_PATH}" \
  "${ENROOT_TEMP_PATH}"

cleanup_enroot() {
  rm -rf "${ENROOT_JOB_ROOT}" "${COMPILER_CACHE_ROOT}"
}
trap cleanup_enroot EXIT

if [[ ( -n "${TTS_RUN_DIR}" || -n "${RESULT_DIR}" ) && "${KB_LEVELS}" == *,* ]]; then
  echo "ERROR: TTS_RUN_DIR/RESULT_DIR overrides require a single KB_LEVELS value"
  exit 2
fi

result_dir="${RESULT_DIR:-${RESULTS_ROOT}/${RUN_GROUP}}"
raw_dir="${result_dir}/kernelbench_eval_runs"
mkdir -p "${raw_dir}"

echo "=== Stage 2a: export best kernel per problem to KernelBench runs ==="
for level in ${KB_LEVELS//,/ }; do
  level="${level//[[:space:]]/}"
  [[ -z "$level" ]] && continue

  tts_dir="${TTS_RUN_DIR:-${TTS_RUNS_ROOT}/${RUN_GROUP}_level${level}}"
  dataset="${DATASET_DIR}/kernelbench_level${level}_for_kernelgym.parquet"
  if [[ ! -d "${tts_dir}/eval_outputs" ]]; then
    echo "ERROR: missing TTS eval outputs: ${tts_dir}/eval_outputs"
    exit 1
  fi
  if [[ ! -f "${dataset}" ]]; then
    echo "ERROR: missing dataset: ${dataset}"
    exit 1
  fi

  export_args=(
    --kernelgym-run "${tts_dir}" \
    --dataset "${dataset}" \
    --kernelbench-root "${KB_ROOT}" \
    --output-root "${KB_ROOT}/runs" \
    --output-run-name "${RUN_GROUP}_level${level}" \
    --select "${EXPORT_SELECT}" \
    --overwrite
  )
  case "${EXPORT_SCOPE}" in
    problem) export_args+=(--best-per-problem) ;;
    trajectory) export_args+=(--preserve-sample-ids --turns-per-trajectory "${EXPORT_TURNS_PER_TRAJECTORY}") ;;
    *) echo "ERROR: EXPORT_SCOPE must be problem or trajectory, got ${EXPORT_SCOPE}"; exit 1 ;;
  esac

  "${PYBIN}" "${EVAL_SCRIPTS}/python/export_kernelgym_tts_to_kernelbench.py" \
    "${export_args[@]}"
done

echo "=== Stage 2b: re-time exported kernels in the KernelBench enroot env ==="
# Inductor/triton caches on scratch so they persist across resubmits and never
# land in the container's ephemeral overlay (same as the baselines job).
export TORCHINDUCTOR_CACHE_DIR="${COMPILER_CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${COMPILER_CACHE_ROOT}/triton"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
export RUN_GROUP KB_LEVELS KB_BACKEND KB_PRECISION
export KB_NUM_CORRECT_TRIALS KB_NUM_PERF_TRIALS KB_TIMEOUT KB_MEASURE_COMPILE_REFERENCE KB_PROBLEM_IDS KB_ROOT

enroot start \
  --root \
  --mount "/.gpfs/scratch:/scratch" \
  --mount "${REPO_ROOT}:${REPO_ROOT}" \
  --env RUN_GROUP \
  --env KB_LEVELS \
  --env KB_BACKEND \
  --env KB_PRECISION \
  --env KB_NUM_CORRECT_TRIALS \
  --env KB_NUM_PERF_TRIALS \
  --env KB_TIMEOUT \
  --env KB_MEASURE_COMPILE_REFERENCE \
  --env KB_PROBLEM_IDS \
  --env KB_ROOT \
  --env TORCHINDUCTOR_CACHE_DIR \
  --env TRITON_CACHE_DIR \
  "${CONTAINER}" \
  bash -lc "cd '${KB_ROOT}' && exec bash '/.gpfs/scratch/zongxiong.chen/daniel/slurm/Kernelbench/a100_drkernel_prrs_20261016/eval_inside_a100.sh'"

echo "=== Stage 2c: collect results + compute metrics vs baselines ==="
for level in ${KB_LEVELS//,/ }; do
  level="${level//[[:space:]]/}"
  [[ -z "$level" ]] && continue
  run_name="${RUN_GROUP}_level${level}"
  tts_dir="${TTS_RUN_DIR:-${TTS_RUNS_ROOT}/${run_name}}"
  mkdir -p "${raw_dir}/${run_name}"
  cp -f "${KB_ROOT}/runs/${run_name}/eval_results.json" "${raw_dir}/${run_name}/"
  cp -f "${KB_ROOT}/runs/${run_name}/kernelgym_tts_export_metadata.json" "${raw_dir}/${run_name}/" 2>/dev/null || true
  # Keep the TTS-side summary next to the enroot numbers for cross-checking.
  cp -f "${tts_dir}/tts_summary.json" "${raw_dir}/${run_name}/" 2>/dev/null || true
done

cat > "${result_dir}/run_metadata.json" <<EOF
{
  "run_group": "${RUN_GROUP}",
  "gpu_target": "A100 80GB",
  "model_path": "${MERGED_MODEL_PATH}",
  "submission_bundle": "/.gpfs/scratch/zongxiong.chen/daniel/slurm/Kernelbench/a100_drkernel_prrs_20261016",
  "baseline_policy": "paired A100 references only; no historical fallback",
  "eval_job_id": "${SLURM_JOB_ID}",
  "levels": "${KB_LEVELS}",
  "export_selection": "${EXPORT_SELECT}",
  "export_scope": "${EXPORT_SCOPE}",
  "export_turns_per_trajectory": "${EXPORT_TURNS_PER_TRAJECTORY}",
  "backend": "${KB_BACKEND}",
  "precision": "${KB_PRECISION}",
  "num_correct_trials": "${KB_NUM_CORRECT_TRIALS}",
  "num_perf_trials": "${KB_NUM_PERF_TRIALS}",
  "problem_ids": "${KB_PROBLEM_IDS}",
  "measure_compile_reference": "${KB_MEASURE_COMPILE_REFERENCE}",
  "baseline_dir": "${BASELINE_DIR}",
  "tts_runs_root": "${TTS_RUNS_ROOT}",
  "tts_run_dir": "${TTS_RUN_DIR}",
  "result_dir": "${result_dir}",
  "dataset_dir": "${DATASET_DIR}"
}
EOF

metrics_args=(
  --run-root "${raw_dir}"
  --kernelbench-root "${KB_ROOT}"
  --baseline-name "torch.eager"
  --baseline-file "${BASELINE_DIR}/baseline_time_torch.json"
  --fast-p 1 1.05 1.2 1.5 2
  --output-json "${result_dir}/kernelbench_metrics.json"
  --output-md "${result_dir}/kernelbench_metrics.md"
)
if [[ -f "${BASELINE_DIR}/baseline_time_torch_compile_inductor_default.json" ]]; then
  metrics_args+=(--extra-baseline "torch.compile=${BASELINE_DIR}/baseline_time_torch_compile_inductor_default.json")
fi
"${PYBIN}" "${EVAL_SCRIPTS}/python/compute_kernelbench_run_metrics.py" "${metrics_args[@]}"

echo "=============== RUN VERIFICATION ==============="
echo "results dir : ${result_dir}"
metrics_json="${result_dir}/kernelbench_metrics.json"
if [[ ! -f "${metrics_json}" ]]; then
  echo "RESULT: KB_EVAL_FAILED - no metrics JSON written, check the .err file"
  exit 1
fi
if ! "${PYBIN}" -c '
import json
import sys

metrics = json.load(open(sys.argv[1]))
levels = metrics.get("levels", {})
if not levels:
    raise SystemExit("no evaluated levels in metrics")
bad = [
    level
    for level, values in levels.items()
    if int(values.get("evaluated_count", 0)) <= 0
    or int(values.get("compiled_count", 0)) <= 0
]
if bad:
    raise SystemExit(f"levels with zero evaluated or compiled kernels: {bad}")
' "${metrics_json}"; then
  echo "RESULT: KB_EVAL_FAILED - evaluation produced zero compiled kernels"
  exit 1
fi
"${PYBIN}" "/.gpfs/scratch/zongxiong.chen/daniel/slurm/Kernelbench/a100_drkernel_prrs_20261016/verify_eval.py" "${raw_dir}" "${KB_LEVELS}"
if [[ -f "${result_dir}/kernelbench_metrics.md" ]]; then
  cat "${result_dir}/kernelbench_metrics.md"
  echo "RESULT: KB_EVAL_OK"
else
  echo "RESULT: KB_EVAL_FAILED - no metrics written, check the .err file"
  exit 1
fi
echo "================================================"
