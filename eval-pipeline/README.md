# Evaluation pipeline

Evaluate a model on KernelBench Levels 1–3 using multi-turn generation followed
by a separate GPU correctness and timing pass.

```text
Checkpoint merge (optional) → TTS generation → candidate export → Enroot evaluation → metrics
```

TTS supplies execution feedback while the model refines its kernels. Enroot
independently evaluates the exported candidates for the final results.

Generation here uses KernelGym/DrKernel **without skill retrieval**.
**DaVinci skill-aware evaluation is not supported by this pipeline.** It
requires the separate DaVinci codebase. Copied DaVinci-specific helpers are
outside the workflow documented here.

## Entry points

- **Full pipeline:** [`submit_kernelbench_tts_per_level.sh`](scripts/shell/submission/submit_kernelbench_tts_per_level.sh) submits an optional checkpoint merge, then a TTS job and dependent Enroot job for each level. Slurm `afterok` dependencies enforce the order.
- **Container evaluation worker:** [`evaluate.sh`](scripts/shell/evaluation/evaluate.sh) runs the final KernelBench evaluation inside the container; it does not submit the full pipeline.

## Run on HPI

Requires Slurm, Enroot, the evaluation image, Python environments, and model
files used by the original HPI setup. From the KernelGYM repository root:

```bash
CHECKPOINT_STEP=400 N_SAMPLES=3 MAX_USER_TURNS=3 \
bash eval-pipeline/scripts/shell/submission/submit_kernelbench_tts_per_level.sh \
  /path/to/checkpoint-run model-step400
```

The main entry point defaults to three trajectories with up to three turns,
one exported turn per trajectory (`best_correct_or_compiled`), FP32, five
correctness trials, and 100 final timing trials. Eager and compiled PyTorch
references are reported separately, using median latency.

The active workflow uses this repository's `eval-pipeline/` scripts and bundled
`kernelbench/` source. Model files, Python environments, the Enroot image, and
historical H100 baseline timing files remain in the HPI workspace. Override
`BASELINE_DIR` to select another set of reference timings; paired eager and
compiled timings from the current evaluation take precedence.

For a small test, build an original-problem subset with
`python eval-pipeline/scripts/python/datasets/build_kernelbench_for_kernelgym.py --levels 1 --problem-ids 88 --no-combined --output-dir /path/to/test-dataset`.
Then set `DATASET_DIR` to that folder, `KB_LEVELS=1`, `KB_PROBLEM_IDS=88`,
`N_SAMPLES=1`, and `MAX_USER_TURNS=1` when submitting. For an existing model
export, set `MODEL_ALREADY_READY=1` and `MERGED_MODEL_PATH=/path/to/model`.
The same entry point runs generation, export, evaluation, and subset metrics.

## Layout and outputs

```text
scripts/python/  datasets/, export/, analysis/, validation/
scripts/shell/   submission/, generation/, evaluation/
slurm/          Cluster launchers and campaign-specific helpers
```

Default artifacts live below the original HPI workspace:

- `models/`: merged model exports.
- `Results/Kernelbench/tts_runs/`: generated kernels, conversations, and feedback.
- `Results/Kernelbench/evaluation_results/`: raw evaluations, provenance, and JSON/Markdown metrics.
- `logs/merge/` and `logs/kernelbench-eval/`: job logs.

Completion requires successful Slurm jobs, `TTS_OK` and `KB_EVAL_OK` log markers,
and the expected result files. Submitted jobs or partial output alone do not
indicate a completed evaluation.
