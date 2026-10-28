#!/usr/bin/env bash
#SBATCH --job-name=no-fusion
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --constraint=ARCH:X86
#SBATCH --exclude=gx13v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=450G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rl-training/%x-%j.out
#SBATCH --error=logs/rl-training/%x-%j.err

set -euo pipefail
# Submit from the repository root, or export KERNELGYM_ROOT explicitly.
export KERNELGYM_ROOT="${KERNELGYM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export REPO_ROOT="${REPO_ROOT:-$(cd "${KERNELGYM_ROOT}/.." && pwd)}"
export VENV="${VENV:-${REPO_ROOT}/.venv-drkernel}"
RECIPE_DIR="${KERNELGYM_ROOT}/rl-training/recipes"
[[ -f "${RECIPE_DIR}/_stage1.sh" ]] || { echo "Submit from the repository root or set KERNELGYM_ROOT." >&2; exit 2; }
export RUN_NAME="${RUN_NAME:-no-fusion}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/_other_runs/recipes/no-fusion}"
if [[ "${ALLOW_RESUME:-0}" != 1 && -e "${CHECKPOINT_ROOT}" ]]; then
  echo "Checkpoint root exists: ${CHECKPOINT_ROOT}. Set ALLOW_RESUME=1 for an intentional resume." >&2
  exit 2
fi
export RUN_LOG="${RUN_LOG:-${KERNELGYM_ROOT}/logs/rl-training/${SLURM_JOB_NAME:-no-fusion}-${SLURM_JOB_ID:-manual}.out}"

# Dataset, initialization, and the four shared training/evaluation GPUs.
export TRAIN_DATASET="${TRAIN_DATASET:-${KERNELGYM_ROOT}/dataset/stage1/stage1.parquet}"
export VALID_DATASET="${VALID_DATASET:-${KERNELGYM_ROOT}/rl-training/data/hkust-nlp/drkernel-validation-data.parquet}"
export EXPECTED_DATASET_ROWS=10000
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/sft-cold-start}"
export DEFRAG_FRAMEWORK_PATH="${DEFRAG_FRAMEWORK_PATH:-${REPO_ROOT}/archive/DefragmentationFramework}"
export TOTAL_GPUS_REQUIRED=4 GPU_LIST=0,1,2,3

# Scientific settings verified against the final thesis run records.
export ALGORITHM=trloo REWARD_FUNC_NAME=calculate_reward_speedup_fusion
export COVERAGE_RS=turn COVERAGE_RS_THRESHOLD=0.1 COVERAGE_RS_FACTOR=0.3
export COVERAGE_RS_KEY=time_coverage SPEEDUP_THRESHOLD=1.2
export COVERAGE_REWARD_ENABLE=True COVERAGE_REWARD_WEIGHT=0.25
export FUSION_PROMPT_ENABLED=False REFERENCE_BACKEND=pytorch
export REJECTION_SAMPLE=True ROLLOUT_RS_KWARGS='{lower:0.995,upper:1.005}'
export FREE_CACHE_ENGINE=True NUM_PERF_TRIALS=50
export ENABLE_MULTI_TURN=True MAX_TURN=2 ROLLOUT_N=8
export TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8
export MAX_PROMPT_LENGTH=14336 MAX_RESPONSE_LENGTH=8192 PPO_MICRO_TOKEN=22528
export ROLLOUT_GPU_MEMORY_UTIL=0.45 VAL_BEFORE_TRAIN=False TEST_FREQ=0
export TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
export SAVE_FREQ=2 MAX_CKPT_TO_KEEP=2 MAX_SKIP_STEPS="${MAX_SKIP_STEPS:-50}"
export RECIPE_FUSION_WEIGHT=0.0

# Match the No-Fusion launcher's existing Hydra override mechanism. A fusion
# reward function activates compression even when the nested enable default is false.
python() {
  if [[ "${1:-}" == -m && "${2:-}" == kernel.main_kernel ]]; then
    command python "$@" "++reward_model.fusion_reward.weight=${RECIPE_FUSION_WEIGHT}"
  else
    command python "$@"
  fi
}
export -f python
bash "${RECIPE_DIR}/_stage1.sh"
