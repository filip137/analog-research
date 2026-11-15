"""Semantic completion check after the native artifact inspector (no scheduling)."""
import json,math,sys
from pathlib import Path

def verify(config, run):
    c=json.loads(Path(config).read_text()); run=Path(run)
    if json.loads((run/'result.json').read_text())['status']!='complete':
        raise ValueError('Run is not complete')
    if c['stage']=='fault_tuning':
        rows=json.loads((run/'artifacts/summary.json').read_text())['arms']
        expected=c['fault_tuning']['arms']
        if [r['label'] for r in rows]!=[a['label'] for a in expected]:
            raise ValueError('Missing, duplicate, reordered or unexpected arms')
        for row,arm in zip(rows,expected):
            if not math.isfinite(row['development']['teacher_kl']): raise ValueError('Nonfinite final KL')
            steps=0 if arm['method'] in ('naive_hold','hold') else c['fault_tuning']['steps']
            if row['updates']!=steps: raise ValueError('Incomplete step budget')
    elif not (run/'checkpoints/deployed.pt').is_file():
        raise ValueError('Missing deployed checkpoint')
    return {'schema':'fault_completion.unit.v1','config':str(Path(config).resolve()),'run':str(run.resolve()),'status':'complete'}

if __name__=='__main__':
    result=verify(*sys.argv[1:]); p=Path(sys.argv[2])/'complete.json'
    p.write_text(json.dumps(result,indent=2)+'\n')
