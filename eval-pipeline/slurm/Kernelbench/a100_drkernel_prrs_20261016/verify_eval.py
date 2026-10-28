import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for level in map(int,sys.argv[2].split(',')):
    files=list(root.glob(f'*_level{level}/eval_results.json'))
    assert len(files)==1, files
    records=json.loads(files[0].read_text())
    expected=50 if level==3 else 100
    assert set(records)==set(map(str,range(1,expected+1))), ('incomplete problems',level,len(records))
    for pid,rows in records.items():
        assert len(rows)==3 and {int(r['sample_id']) for r in rows}=={0,1,2}, (level,pid,'incomplete trajectories')
        for row in rows:
            for key in ('runtime_stats','ref_runtime_stats','ref_compile_runtime_stats'):
                stats=row.get(key) or {}
                if stats and float(row.get({'runtime_stats':'runtime','ref_runtime_stats':'ref_runtime','ref_compile_runtime_stats':'ref_compile_runtime'}[key],-1))>0:
                    assert stats.get('num_trials')==100, (level,pid,key,'trial count')
                    assert stats.get('runtime_anchor')=='median', (level,pid,key,'anchor')
                    assert 'A100' in stats.get('hardware',''), (level,pid,key,'hardware')
            if row.get('correctness'):
                assert row.get('metadata',{}).get('correctness_trials')=='(5 / 5)', (level,pid,'correctness trial count')
    print(f'A100 coverage verified: L{level}, {expected} problems, {expected*3} candidate evaluations')
