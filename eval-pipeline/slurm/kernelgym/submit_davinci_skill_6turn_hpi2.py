"""Reuse the verified joint-training step84 export for an independent 3x6 chain."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/.gpfs/scratch/zongxiong.chen/daniel')
sys.path.insert(0, str(ROOT / 'slurm/kernelgym'))
import davinci_skill_eval_chain as chain

SOURCE = ROOT / 'Results/Kernelbench/chain_manifests/kbtts_davinci_joint_rl_final2713752_skills_3x3_20261022'
GROUP = 'kbtts_davinci_joint_step84_skills_3traj6turn_20261023'
DEST = ROOT / 'Results/Kernelbench/chain_manifests' / GROUP
ADAPTER = ROOT / 'slurm/kernelgym/davinci_skill_eval_chain.sbatch'
manifest = chain.read(SOURCE / 'chain_manifest.json')
config = chain.read(SOURCE / 'chain_config.json')
assert manifest['step'] == 84 and manifest['skill_step'] == 84
assert chain.resolve_checkpoint(config['run_dir'], config['skill_root'])['step'] == 84
chain.verify_export(manifest['model_path'])
assert chain.sha(manifest['skill_library_file']) == manifest['skill_source_sha256']
assert len([x for x in Path(manifest['skill_library_file']).read_text().splitlines() if x.strip()]) == 114
for script in (chain.TTS, chain.EVAL, chain.EXPORTER):
    assert chain.sha(script) == manifest['source_hashes'][str(script)], str(script)
for script in (chain.TTS, chain.EVAL, ADAPTER):
    subprocess.run(['bash', '-n', str(script)], check=True)
import pyarrow.parquet as pq
for level, rows in chain.EXPECTED_ROWS.items():
    dataset = chain.DATASETS / f'kernelbench_level{level}_for_kernelgym.parquet'
    assert pq.read_metadata(dataset).num_rows == rows
    for path in (ROOT / f'Results/Kernelbench/tts_runs/{GROUP}/l{level}',
                 ROOT / f'Results/Kernelbench/evaluation_results/{GROUP}/l{level}',
                 ROOT / f'KernelBench/runs/{GROUP}-l{level}_level{level}'):
        assert not path.exists(), str(path)
for baseline in ('baseline_time_torch.json', 'baseline_time_torch_compile_inductor_default.json'):
    assert (ROOT / 'KernelBench/results/timing/H100_aisc_fp32' / baseline).is_file()
assert chain.read(Path(manifest['model_path']) / 'config.json')['max_position_embeddings'] >= 32768

resources = [('--gres=gpu:h100:2', '--mem=256G', '--time=24:00:00'),
             ('--gres=gpu:h100:1', '--mem=128G', '--time=06:00:00')]
for args in resources:
    subprocess.run(['sbatch', '--test-only', *args, str(ADAPTER)], check=True)
print('RESULT: SIX_TURN_PREFLIGHT_OK step=84 skills=114 trajectories=3 turns=6', flush=True)

DEST.mkdir(exist_ok=False)
snapshot = DEST / 'skill_library_step84.jsonl'
shutil.copy2(manifest['skill_library_file'], snapshot)
assert chain.sha(snapshot) == manifest['skill_source_sha256']
manifest.update(run_group=GROUP, turns=6, trajectories=3,
    skill_library_file=str(snapshot), source_chain=str(SOURCE),
    evaluation_adapter=str(ADAPTER),
    evaluation_adapter_sha256=chain.sha(ADAPTER),
    chain_helper_sha256=chain.sha(ROOT / 'slurm/kernelgym/davinci_skill_eval_chain.py'))
config.update(group_base=GROUP, turns=6)
chain.write_new(DEST / 'chain_manifest.json', manifest)
chain.write_new(DEST / 'chain_config.json', config)

with (DEST / 'submitted_jobs.jsonl').open('x') as ledger:
    def record(value):
        ledger.write(json.dumps(value) + '\n')
        ledger.flush()
        os.fsync(ledger.fileno())
        print(json.dumps(value), flush=True)

    def submit(stage, level, args):
        env = dict(os.environ, CHAIN_STAGE=stage, CHAIN_DIR=str(DEST), KB_LEVEL=str(level))
        result = subprocess.run(['sbatch', '--parsable', '--no-requeue', '--export=ALL',
            *args, str(ADAPTER)], env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        job = result.stdout.strip().split(';')[0]
        assert job.isdigit(), result.stdout
        return job

    for level in (1, 2, 3):
        tts_time = {1: '24:00:00', 2: '09:00:00', 3: '06:00:00'}[level]
        eval_time = {1: '06:00:00', 2: '03:00:00', 3: '02:00:00'}[level]
        tts = submit('tts', level, [f'--job-name=davinci-skill-6turn-tts-l{level}',
            '--gres=gpu:h100:2', '--mem=256G', f'--time={tts_time}'])
        record(dict(stage='tts', level=level, job_id=tts, walltime=tts_time,
            model_ready=True, checkpoint_step=84, turns=6, trajectories=3,
            output=str(ROOT / f'Results/Kernelbench/tts_runs/{GROUP}/l{level}')))
        evaluation = submit('eval', level, [f'--job-name=davinci-skill-6turn-eval-l{level}',
            '--gres=gpu:h100:1', '--mem=128G', f'--time={eval_time}',
            f'--dependency=afterok:{tts}', '--kill-on-invalid-dep=yes'])
        record(dict(stage='eval', level=level, job_id=evaluation, tts_job=tts,
            walltime=eval_time, output=str(ROOT / f'Results/Kernelbench/evaluation_results/{GROUP}/l{level}')))
