"""Resolve a post-training checkpoint once, then run canonical skill-TTS/eval.

No training jobs or training artifacts are changed. Excluded decoy-only
trajectories are explicitly recorded as failures in the fixed 3-sample protocol.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

ROOT = Path('/.gpfs/scratch/zongxiong.chen/daniel')
MERGER = ROOT / 'slurm/merge/merge_checkpoints.slurm'
TTS = ROOT / 'slurm/kernelgym/tts_davinci_repo_kernelbench.slurm'
EVAL = ROOT / 'slurm/Kernelbench/eval_kernelgym_tts_enroot.sbatch'
EXPORTER = ROOT / 'Scripts/kernelbench_evaluation/python/export_kernelgym_tts_to_kernelbench.py'
DATASETS = ROOT / 'Results/Kernelbench/kernelgym_datasets'
EXPECTED_ROWS = {1: 100, 2: 100, 3: 50}


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_checkpoint(run, library):
    run, library = Path(run), Path(library)
    step = int((run / 'latest_checkpointed_iteration.txt').read_text().strip())
    assert step > 0
    ckpt = run / f'global_step_{step}'
    assert (ckpt / 'data.pt').is_file(), 'Missing dataloader checkpoint'
    actor = ckpt / 'actor'
    for kind in ('model', 'optim', 'extra_state'):
        for rank in range(4):
            p = actor / f'{kind}_world_size_4_rank_{rank}.pt'
            assert p.is_file() and p.stat().st_size > 0, str(p)
    for name in ('fsdp_config.json', 'huggingface/config.json', 'huggingface/tokenizer_config.json'):
        assert (actor / name).is_file(), str(actor / name)
    snapshots = [(int(p.stem.split('_')[-1]), p)
                 for p in library.glob('global_step_*.jsonl')]
    skill_step, snapshot = max((n, p) for n, p in snapshots if n <= step)
    skills = [json.loads(line) for line in snapshot.read_text().splitlines() if line.strip()]
    assert len({r['name'] for r in skills}) >= 3, 'Need at least three learned skills'
    return dict(step=step, checkpoint=str(ckpt), actor=str(actor),
                skill_step=skill_step, skill_source=str(snapshot),
                skill_source_sha256=sha(snapshot), skill_count=len(skills))


def verify_export(model):
    model = Path(model)
    for name in ('config.json', 'model.safetensors.index.json', 'tokenizer_config.json'):
        assert (model / name).is_file(), str(model / name)
    index = read(model / 'model.safetensors.index.json')
    for name in set(index['weight_map'].values()):
        p = model / name
        assert p.is_file() and p.stat().st_size > 0, str(p)


def load_exporter():
    spec = importlib.util.spec_from_file_location('davinci_chain_exporter', EXPORTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def filtered_view(raw, view, dataset_map, level):
    """Preserve raw outputs; expose only non-decoy turn candidates to exporter."""
    view.mkdir(parents=True, exist_ok=False)
    excluded, sample_ids = [], set()
    dirs = sorted(p for p in raw.iterdir()
                  if p.is_dir() and re.fullmatch(r'problem_\d+_sample_\d+', p.name))
    assert len(dirs) == EXPECTED_ROWS[level] * 3, (len(dirs), level)
    identities = set()
    for source in dirs:
        index, sample = map(int, re.fullmatch(r'problem_(\d+)_sample_(\d+)', source.name).groups())
        assert index in dataset_map and sample in (0, 1, 2)
        identity = (dataset_map[index].problem_id, sample)
        assert identity not in identities
        identities.add(identity)
        eligible = []
        for kernel in sorted(source.glob('turn_*_kernel.py')):
            turn = int(re.fullmatch(r'turn_(\d+)_kernel.py', kernel.name).group(1))
            assert 1 <= turn <= 3, ('Unexpected evaluation turn', turn)
            evaluation = kernel.with_name(f'turn_{turn}_eval.json')
            result = read(evaluation) if evaluation.is_file() else {}
            if result.get('is_decoy_kernel') or result.get('decoy_kernel'):
                continue
            eligible.append((kernel, evaluation))
        if not eligible:
            excluded.append(dict(problem_id=identity[0], sample_id=sample,
                                 reason='no_nondecoy_kernel', source=str(source)))
            continue
        destination = view / source.name
        destination.mkdir()
        for kernel, evaluation in eligible:
            (destination / kernel.name).symlink_to(kernel)
            if evaluation.is_file():
                (destination / evaluation.name).symlink_to(evaluation)
        sample_ids.add(sample)
    # The existing container driver infers n_samples from exported sample IDs.
    assert sample_ids == {0, 1, 2}, 'Cannot preserve the three-sample denominator'
    return excluded


def paths(manifest, level):
    group = manifest['run_group']
    return (ROOT / 'Results/Kernelbench/tts_runs' / group / f'l{level}',
            ROOT / 'Results/Kernelbench/evaluation_results' / group / f'l{level}',
            f'{group}-l{level}')


def prepare(chain, config):
    manifest = resolve_checkpoint(config['run_dir'], config['skill_root'])
    step = manifest['step']
    group = f"{config['group_base']}_step{step}"
    model = ROOT / 'models' / f"davinci-joint-final2713752-20261022-step{step}"
    assert not model.exists(), f'Refusing to overwrite {model}'
    snapshot = chain / f"skill_library_step{manifest['skill_step']}.jsonl"
    assert not snapshot.exists()
    shutil.copy2(manifest['skill_source'], snapshot)
    assert sha(snapshot) == manifest['skill_source_sha256']
    env = dict(os.environ, RUN_DIR=config['run_dir'], STEP=str(step),
               TARGET_DIR=str(model), OVERWRITE='0')
    subprocess.run(['bash', str(MERGER)], env=env, check=True)
    verify_export(model)
    manifest.update(train_job_id=config['train_job_id'], model_path=str(model),
                    skill_library_file=str(snapshot), run_group=group,
                    trajectories=3, turns=3, max_iterations=1,
                    source_hashes={str(p): sha(p) for p in (MERGER, TTS, EVAL, EXPORTER)})
    write_new(chain / 'chain_manifest.json', manifest)
    print('RESULT: FINAL_CHECKPOINT_MERGE_OK', json.dumps(manifest), flush=True)


def tts(chain, manifest, level):
    for source in (TTS, EXPORTER):
        assert sha(source) == manifest['source_hashes'][str(source)], f'Source changed: {source}'
    model = Path(manifest['model_path'])
    verify_export(model)
    library = Path(manifest['skill_library_file'])
    assert sha(library) == manifest['skill_source_sha256']
    output, _, group = paths(manifest, level)
    assert not output.exists(), f'Refusing to overwrite {output}'
    dataset = DATASETS / f'kernelbench_level{level}_for_kernelgym.parquet'
    env = dict(os.environ, REPO_ROOT=str(ROOT), MODEL_PATH=str(model),
               SKILL_LIBRARY_FILE=str(library), KB_LEVEL=str(level),
               EVAL_DATASET=str(dataset), OUTPUT_DIR=str(output), RUN_NAME=group,
               N_SAMPLES='3', MAX_USER_TURNS='3', MAX_ITERATIONS='1',
               SKILL_VALIDATION_ONLY='true', NUM_CORRECT_TRIALS='5',
               NUM_PERF_TRIALS='20', BATCH_SIZE='5',
               ENROOT_ROOT=str(ROOT / '.enroot'))
    # Operational-only adaptation: use free ports, not job-ID modulo arithmetic.
    sockets = [socket.socket(), socket.socket()]
    try:
        for sock in sockets:
            sock.bind(('127.0.0.1', 0))
        env['DAVINCI_CHAIN_REDIS_PORT'] = str(sockets[0].getsockname()[1])
        env['DAVINCI_CHAIN_API_PORT'] = str(sockets[1].getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()
    source = TTS.read_text()
    replacements = {
        'export REDIS_PORT=$((20000 + SLURM_JOB_ID % 10000))':
            'export REDIS_PORT="${DAVINCI_CHAIN_REDIS_PORT:?}"',
        'export API_PORT=$((30000 + SLURM_JOB_ID % 10000))':
            'export API_PORT="${DAVINCI_CHAIN_API_PORT:?}"',
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, ('Canonical TTS launcher changed', old)
        source = source.replace(old, new)
    local_script = chain / f"tts_runtime_{os.environ['SLURM_JOB_ID']}.sh"
    with local_script.open('x') as f:
        f.write(source)
    subprocess.run(['bash', '-n', str(local_script)], check=True)
    subprocess.run(['bash', str(local_script)], env=env, check=True)
    exporter = load_exporter()
    mapping = exporter.load_dataset_map(dataset)
    raw = output / 'grading_results/eval_outputs_skill'
    view = output / 'skill_export/eligible_eval_outputs'
    excluded = filtered_view(raw, view, mapping, level)
    link = output / 'skill_export/eval_outputs'
    assert link.is_symlink() and link.resolve() == raw.resolve()
    link.unlink()  # Only replace our new export-view link; raw generations remain.
    link.symlink_to(view)
    write_new(output / 'chain_tts_audit.json', dict(
        step=manifest['step'], skill_step=manifest['skill_step'],
        trajectories=EXPECTED_ROWS[level] * 3, raw_outputs=str(raw),
        export_view=str(view), excluded_trajectories=excluded))
    print('RESULT: TTS_OK', level, output, 'excluded_as_failed=', len(excluded), flush=True)


def evaluate(chain, manifest, level):
    for source in (EVAL, EXPORTER):
        assert sha(source) == manifest['source_hashes'][str(source)], f'Source changed: {source}'
    output, result, group = paths(manifest, level)
    audit = read(output / 'chain_tts_audit.json')
    assert audit['step'] == manifest['step']
    run_dir = ROOT / 'KernelBench/runs' / f'{group}_level{level}'
    assert not run_dir.exists(), f'Refusing existing evaluation run {run_dir}'
    assert not result.exists(), f'Refusing existing evaluation results {result}'
    # The canonical exporter only writes kernels/metadata; it preserves this
    # explicit failed-result ledger for excluded trajectories. No decoy is run.
    excluded_results = {}
    for item in audit['excluded_trajectories']:
        excluded_results.setdefault(str(item['problem_id']), []).append(dict(
            sample_id=item['sample_id'], compiled=False, correctness=False,
            runtime=-1.0, runtime_stats={}, ref_runtime=None, ref_runtime_stats={},
            ref_compile_runtime=None, ref_compile_runtime_stats={},
            metadata=dict(error=item['reason'], excluded_from_export=True,
                          source=item['source'])))
    if excluded_results:
        run_dir.mkdir(parents=True)
        write_new(run_dir / 'eval_results.json', excluded_results)
    env = dict(os.environ, RUN_GROUP=group, KB_LEVELS=str(level),
               TTS_RUN_DIR=str(output / 'skill_export'), RESULT_DIR=str(result),
               DATASET_DIR=str(DATASETS), EXPORT_SELECT='best_correct_or_compiled',
               EXPORT_SCOPE='trajectory', EXPORT_TURNS_PER_TRAJECTORY='1',
               KB_PRECISION='fp32', KB_NUM_CORRECT_TRIALS='5',
               KB_NUM_PERF_TRIALS='100', KB_MEASURE_COMPILE_REFERENCE='true',
               ENROOT_ROOT=str(ROOT / '.enroot'))
    subprocess.run(['bash', str(EVAL)], env=env, check=True)
    metadata = read(run_dir / 'kernelgym_tts_export_metadata.json')
    assert metadata['num_selected_decoys'] == 0
    assert metadata['num_exported'] + len(audit['excluded_trajectories']) == EXPECTED_ROWS[level] * 3
    results = read(run_dir / 'eval_results.json')
    mapping = load_exporter().load_dataset_map(DATASETS / f'kernelbench_level{level}_for_kernelgym.parquet')
    for row in mapping.values():
        samples = results[str(row.problem_id)]
        assert len(samples) == 3 and {r['sample_id'] for r in samples} == {0, 1, 2}
    assert (result / 'kernelbench_metrics.json').is_file()
    write_new(result / 'chain_provenance.json', dict(manifest,
              level=level, excluded_trajectories=audit['excluded_trajectories']))
    print('RESULT: KB_EVAL_OK', level, result, flush=True)


def preflight(config):
    import pyarrow.parquet as pq
    manifest = resolve_checkpoint(config['run_dir'], config['skill_root'])
    for script in (MERGER, TTS, EVAL):
        subprocess.run(['bash', '-n', str(script)], check=True)
    for level, rows in EXPECTED_ROWS.items():
        assert pq.read_metadata(DATASETS / f'kernelbench_level{level}_for_kernelgym.parquet').num_rows == rows
    for name in ('baseline_time_torch.json', 'baseline_time_torch_compile_inductor_default.json'):
        assert (ROOT / 'KernelBench/results/timing/H100_aisc_fp32' / name).is_file()
    model_config = read(Path(manifest['actor']) / 'huggingface/config.json')
    assert model_config['max_position_embeddings'] >= 24576 + 8192
    print(json.dumps(dict(status='preflight_passed', current_checkpoint=manifest,
                         final_step_selection='deferred until training ends', trajectories=3, turns=3)))


if __name__ == '__main__':
    stage, directory = sys.argv[1:]
    chain = Path(directory)
    config = read(chain / 'chain_config.json')
    if stage == 'preflight':
        preflight(config)
    elif stage == 'prepare':
        prepare(chain, config)
    else:
        manifest = read(chain / 'chain_manifest.json')
        level = int(os.environ['KB_LEVEL'])
        assert level in EXPECTED_ROWS
        {'tts': tts, 'eval': evaluate}[stage](chain, manifest, level)
