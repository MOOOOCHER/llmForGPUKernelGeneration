#!/bin/bash
# Export the latest saved RL checkpoint to a HuggingFace-format model directory
# (like the ones under models/), using verl's FSDP model merger.
#
#   sbatch /.gpfs/scratch/zongxiong.chen/daniel/slurm/merge/merge_checkpoints.slurm
#
# By default it picks the most recently modified run under checkpoints/ that
# actually contains saved FSDP shards, and within it the step recorded in
# latest_checkpointed_iteration.txt. Override at submit time without editing:
#   sbatch --export=ALL,RUN_DIR=/path/to/checkpoints/<run>,STEP=25,TARGET_DIR=/path/out ...
#
# Cluster notes: the merge is pure CPU work (no --gres), but it still has to
# run on aisc-batch: the aisc account's only QOS is "aisc", cpu-batch allows
# only normal/high (job 2344448 sat forever with "QOS not permitted"), the
# "default" account has a submit limit of 0, and aisc-shortrun/-interactive
# reject sbatch (interactive-only). A CPU-only job here does not consume a GPU;
# nodes have ~2TB RAM and mostly-idle CPUs, so it backfills quickly (ignore
# sbatch --test-only's estimate -- it disregards backfill). ARCH:X86 is
# required because the x86_64 venv would not run on the partition's ARM node
# (ga03). Compute nodes cannot see the login node's /tmp, so all paths are on
# shared GPFS.

#SBATCH --job-name=drkernel-merge-ckpt
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Shards are fp32: 4 x 8.2GB on disk. The merger holds all rank shards plus the
# merged full state dict in RAM at once (~70GB peak); nodes have ~186GB.
#SBATCH --mem=128G
#SBATCH --time=02:00:00
# Slurm opens these before the script body runs, so logs/merge/ must already
# exist at submit time; a mkdir below would be too late.
#SBATCH --output=/.gpfs/scratch/zongxiong.chen/daniel/logs/merge/merge-%j.log
#SBATCH --error=/.gpfs/scratch/zongxiong.chen/daniel/logs/merge/merge-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/.gpfs/scratch/zongxiong.chen/daniel}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints}"

# ---- Locate the checkpoint to export -------------------------------------
# Newest run dir is not enough by itself: a freshly started run creates its
# directory (wandb_run_id.txt) long before the first save, so require actual
# FSDP shards before accepting a run.
if [[ -z "${RUN_DIR:-}" ]]; then
  for d in $(ls -dt "${CKPT_ROOT}"/*/ 2>/dev/null); do
    if compgen -G "${d}global_step_*/actor/model_world_size_*_rank_0.pt" > /dev/null; then
      RUN_DIR="${d%/}"
      break
    fi
  done
fi
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "ERROR: no run with saved checkpoints found under ${CKPT_ROOT}" >&2
  exit 1
fi

# verl writes latest_checkpointed_iteration.txt on every save; fall back to the
# highest global_step_* dir if it is missing (numeric sort -- alphabetical
# would put step 100 before step 25).
if [[ -z "${STEP:-}" ]]; then
  if [[ -f "${RUN_DIR}/latest_checkpointed_iteration.txt" ]]; then
    STEP=$(< "${RUN_DIR}/latest_checkpointed_iteration.txt")
  else
    STEP=$(ls -d "${RUN_DIR}"/global_step_* 2>/dev/null \
           | sed 's/.*global_step_//' | sort -n | tail -1)
  fi
fi

ACTOR_DIR="${RUN_DIR}/global_step_${STEP}/actor"

# RUN_NAME dirs look like <name>_<hash>_<data>_<basemodel>; keep name+hash so
# exports from different configs cannot collide, drop the rest for readability.
RUN_TAG=$(basename "${RUN_DIR}" | cut -d_ -f1-2)
TARGET_DIR="${TARGET_DIR:-${REPO_ROOT}/models/${RUN_TAG}-step${STEP}}"

echo "=== Merge verl FSDP checkpoint to HuggingFace format ==="
date
hostname
echo "Run dir   : ${RUN_DIR}"
echo "Step      : ${STEP}"
echo "Actor dir : ${ACTOR_DIR}"
echo "Target dir: ${TARGET_DIR}"

if [[ ! -f "${ACTOR_DIR}/fsdp_config.json" ]]; then
  echo "ERROR: missing ${ACTOR_DIR}/fsdp_config.json -- not a saved FSDP checkpoint" >&2
  exit 1
fi
if [[ ! -f "${ACTOR_DIR}/huggingface/config.json" ]]; then
  echo "ERROR: missing ${ACTOR_DIR}/huggingface/config.json -- merger needs it for model/tokenizer" >&2
  exit 1
fi
# Refuse to clobber an existing export unless explicitly asked: models/ also
# holds the hand-placed base models, and a typo'd TARGET_DIR must not rm one.
if [[ -e "${TARGET_DIR}" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "ERROR: ${TARGET_DIR} already exists; rerun with OVERWRITE=1 to replace it" >&2
  exit 1
fi

# The env is a uv venv (the cluster disables pip). verl 0.5.0.dev0 is installed
# editable from KernelGYM/rl-training/verl, so activation alone provides
# `python -m verl.model_merger`; no PYTHONPATH needed.
export PYTHONPATH="${KERNELGYM_ROOT:-${REPO_ROOT}/KernelGYM}/rl-training/verl:${PYTHONPATH:-}"
source "${REPO_ROOT}/.venv-drkernel/bin/activate"

# CPU-only node: make sure torch never tries to touch CUDA, and keep temp files
# off the node's shared /tmp and per-job.
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="/tmp/drkernel-merge-${SLURM_JOB_ID:-manual}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}"' EXIT
export TOKENIZERS_PARALLELISM=false
# Everything (config, tokenizer, weights) is read from local disk; forbid any
# accidental Hub lookup since compute nodes may lack internet access.
export HF_HUB_OFFLINE=1

[[ "${OVERWRITE:-0}" == "1" ]] && rm -rf "${TARGET_DIR}"

python --version
python -m verl.model_merger merge \
  --backend fsdp \
  --use_cpu_initialization \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${TARGET_DIR}"

# ---- Did it actually produce a loadable HF model? -------------------------
echo "=============== MERGE VERIFICATION ==============="
MERGE_OK=yes
for f in config.json model.safetensors.index.json tokenizer_config.json; do
  if [[ ! -f "${TARGET_DIR}/${f}" ]]; then
    echo "missing: ${TARGET_DIR}/${f}"
    MERGE_OK=no
  fi
done
if ! compgen -G "${TARGET_DIR}/model-*.safetensors" > /dev/null; then
  echo "missing: ${TARGET_DIR}/model-*.safetensors"
  MERGE_OK=no
fi
du -sh "${TARGET_DIR}" 2>/dev/null || true
find "${TARGET_DIR}" -maxdepth 1 -type f -printf "%s %f\n" | sort -n
if [[ "${MERGE_OK}" == "yes" ]]; then
  echo "RESULT: MERGE_OK -> ${TARGET_DIR}"
else
  echo "RESULT: MERGE_FAILED -- check the .err file"
  exit 1
fi
echo "=================================================="
date
