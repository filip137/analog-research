"""Select only the frozen development minimum; verify the unchanged sample-8 gate."""
import json
from pathlib import Path
import sys

RATES=(1e-3,3e-3,1e-2)
MODES=('random:32','importance:32','loss_only:32','context:32')

def select(root):
    rows={}; selected={}
    for mode in MODES:
        slug=mode.replace(':','-').replace('_','-')
        group=Path(root)/('tb-tune-'+slug)
        marker=json.loads((group/'complete.json').read_text())
        summaries=json.loads((group/'summaries.json').read_text())
        if len(summaries)!=1 or marker['trajectories']!=1:
            raise ValueError('Expected exactly one validated development trajectory per mode.')
        summary=summaries[0]
        if summary['realization']!=271004 or not summary['replay_exact'] or not summary['source_unchanged']:
            raise ValueError('Invalid development identity or replay.')
        arms=summary['arms']
        gate=arms['T-samp8-lr0.01']['mean_teacher_kl']
        if abs(gate-0.6914967786)>1e-9:
            raise ValueError(f'Sample-8 reference mismatch: {gate}')
        scores={str(lr):arms['TB-'+slug+'-lr'+format(lr,'g')]['mean_teacher_kl'] for lr in RATES}
        lr=min(RATES,key=lambda lr:(scores[str(lr)],lr))
        selected[mode]=lr;rows[mode]={'scores':scores,'selected':lr,'edge':lr in (RATES[0],RATES[-1])}
    return {'selected':selected,'development':rows,'selection_realization':271004,'evaluation_start':64}

if __name__=='__main__':
    output=select(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))
