from pathlib import Path
import json,os,subprocess,re,datetime
base=Path(__file__).resolve().parent
manifest=json.loads((base/'manifest.json').read_text())
journal=base/'submission.json'
if journal.exists():raise SystemExit('Submission journal exists: inspect it; do not duplicate jobs')
for model in manifest['models']:
    for key in ('tts_output_root','eval_output_root'):
        if Path(model[key]).exists():raise SystemExit('Existing destination: '+model[key])
    for level in (1,2,3):
        p=Path('/.gpfs/scratch/zongxiong.chen/daniel/KernelBench/runs')/f"{model['run_group']}-l{level}_level{level}"
        if p.exists():raise SystemExit('Existing export destination: '+str(p))
    mp=Path(model['model'])
    for name in ('config.json','model.safetensors.index.json','tokenizer_config.json'):
        assert (mp/name).is_file(),mp/name
    idx=json.loads((mp/'model.safetensors.index.json').read_text())
    assert all((mp/f).is_file() for f in set(idx['weight_map'].values()))
state={'submitted_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest':manifest,'jobs':[],'status':'submitting'}
def save():
    tmp=journal.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(journal)
save()
for model in manifest['models']:
    env=dict(os.environ)
    env.update(MODEL_ALREADY_READY='1',MERGED_MODEL_PATH=model['model'],MODEL_NAME=model['name'],RUN_GROUP_BASE=model['run_group'],TTS_OUTPUT_ROOT=model['tts_output_root'],EVAL_OUTPUT_ROOT=model['eval_output_root'],N_SAMPLES='3',MAX_USER_TURNS='3',EXPORT_SELECT='best_correct_or_compiled',EXPORT_SCOPE='trajectory',EXPORT_TURNS_PER_TRAJECTORY='1',KB_PRECISION='fp32',KB_NUM_CORRECT_TRIALS='5',KB_NUM_PERF_TRIALS='100',KB_MEASURE_COMPILE_REFERENCE='true',SBATCH_PARTITION='pot-hpi-gpu-batch',TTS_TIME_LIMIT='24:00:00',EVAL_TIME_LIMIT='12:00:00',BASELINE_DIR=str(base/'paired_only_baselines'),BATCH_SIZE='5',ROLLOUT_GPU_MEMORY_UTIL='0.5',TEMPERATURE='1.0',TOP_P='0.95',NUM_CORRECT_TRIALS='5',NUM_PERF_TRIALS='20',REFERENCE_BACKEND='pytorch',CHECKPOINT_STEP=str(model['step'] or ''))
    env.pop('MODEL_READY_JOB_ID',None)
    log=base/(model['key']+'_submission.log')
    with log.open('w') as stream:
        proc=subprocess.Popen(['bash',str(base/'submit_per_level.sh'),model['checkpoint'] or model['model'],model['name']],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:
            print(line,end='',flush=True);stream.write(line);stream.flush()
            match=re.search(r'SUBMITTED_(TTS|EVAL) level=(\d+) job=(\d+)(?: parent=(\d+))?',line)
            if match:
                kind,level,jid,parent=match.groups()
                state['jobs'].append(dict(model=model['key'],level=int(level),kind=kind,job_id=jid,parent=parent))
                save()
        rc=proc.wait()
    if rc:
        state['status']='submission_failed';state['failed_model']=model['key'];state['exit_code']=rc;save();raise SystemExit(rc)
state['status']='submitted';save()
