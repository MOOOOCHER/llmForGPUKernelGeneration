#!/bin/bash
# Container-side entrypoint for the KernelBench eval of exported KernelGYM TTS
# kernels. Runs inside the kernelbench enroot image; the sbatch script passes
# config via the environment:
#   RUN_GROUP        base run name; per-level dirs are runs/${RUN_GROUP}_level<L>
#   KB_LEVELS        comma-separated levels, e.g. "1,2,3"
#   KB_BACKEND       kernel backend of the generated code (triton)
#   KB_PRECISION     fp32 to match the dataset prompts and the baseline timings
#   KB_NUM_CORRECT_TRIALS / KB_NUM_PERF_TRIALS / KB_TIMEOUT
# It lives in its own file rather than inline in the sbatch so the srun
# invocation doesn't need nested quoting.

set -euo pipefail

KERNELGYM_ROOT="${KERNELGYM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
KB_ROOT="${KB_ROOT:-${KERNELGYM_ROOT}/kernelbench}"

: "${RUN_GROUP:?set RUN_GROUP}"
KB_LEVELS="${KB_LEVELS:-1,2,3}"
KB_BACKEND="${KB_BACKEND:-triton}"
KB_PRECISION="${KB_PRECISION:-fp32}"
KB_NUM_CORRECT_TRIALS="${KB_NUM_CORRECT_TRIALS:-5}"
KB_NUM_PERF_TRIALS="${KB_NUM_PERF_TRIALS:-100}"
KB_TIMEOUT="${KB_TIMEOUT:-300}"
KB_MEASURE_COMPILE_REFERENCE="${KB_MEASURE_COMPILE_REFERENCE:-true}"
KB_PROBLEM_IDS="${KB_PROBLEM_IDS:-}"

nvidia-smi

# Fail loudly here rather than half a level in: if pyxis did not expose the
# GPU, every timing would silently be garbage.
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in container'; x = torch.ones(1024, device='cuda'); assert (x + 1).sum().item() == 2048; torch.cuda.synchronize(); print('torch', torch.__version__, torch.cuda.get_device_name(0), 'allocation smoke OK')"

# KernelBench's deps are baked into the image; only the checkout itself is
# bind-mounted. PYTHONPATH rather than pip install -e: the image is Python
# 3.11 and KernelBench's pyproject pins 3.10.*.
export PYTHONPATH="${KB_ROOT}/src:${PYTHONPATH:-}"
python -c "import kernelbench, pydra; print('kernelbench import OK')"

cd "${KB_ROOT}"

overall_rc=0
for level in ${KB_LEVELS//,/ }; do
  level="${level//[[:space:]]/}"
  [[ -z "$level" ]] && continue

  run_name="${RUN_GROUP}_level${level}"
  run_dir="${KB_ROOT}/runs/${run_name}"
  if [[ ! -d "$run_dir" ]]; then
    echo "ERROR: missing exported run dir: $run_dir"
    overall_rc=1
    continue
  fi
  n_kernels=$(find "$run_dir" -maxdepth 1 -name "level_*_sample_*_kernel.py" | wc -l)
  max_sample_id=$(find "$run_dir" -maxdepth 1 -name "level_*_sample_*_kernel.py" -printf "%f\n" \
    | sed -n 's/.*_sample_\([0-9][0-9]*\)_kernel\.py/\1/p' \
    | sort -n | tail -1)
  if [[ -z "$max_sample_id" ]]; then
    echo "ERROR: no exported kernels found for ${run_name}"
    overall_rc=1
    continue
  fi
  num_samples_per_problem=$((max_sample_id + 1))
  echo "=== Evaluating ${run_name}: ${n_kernels} kernels, ${num_samples_per_problem} trajectories/problem ==="

  problem_filter=()
  if [[ -n "${KB_PROBLEM_IDS}" ]]; then
    problem_filter+=("problem_ids=[${KB_PROBLEM_IDS}]")
  fi

  python scripts/eval_from_generations.py \
    run_name="${run_name}" \
    dataset_src=local \
    level="${level}" \
    backend="${KB_BACKEND}" \
    precision="${KB_PRECISION}" \
    num_samples_per_problem="${num_samples_per_problem}" \
    num_correct_trials="${KB_NUM_CORRECT_TRIALS}" \
    num_perf_trials="${KB_NUM_PERF_TRIALS}" \
    measure_compile_reference="${KB_MEASURE_COMPILE_REFERENCE}" \
    compile_reference_backend=inductor \
    compile_reference_mode=default \
    timeout="${KB_TIMEOUT}" \
    num_gpu_devices=1 \
    gpu_arch='["Hopper"]' \
    "${problem_filter[@]}" \
    || { echo "ERROR: eval failed for ${run_name}"; overall_rc=1; }

  if [[ ! -f "${run_dir}/eval_results.json" ]]; then
    echo "ERROR: no eval_results.json produced for ${run_name}"
    overall_rc=1
  fi
done

exit "${overall_rc}"
