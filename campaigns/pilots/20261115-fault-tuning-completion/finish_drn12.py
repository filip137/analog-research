"""Materialize only the two dependent DRN12 controls after LR selection.
Usage: finish_drn12.py --noisy-lr VALUE --child-lr VALUE --output-dir DIRECTORY
Selection must use final dev KL and the recorded 1% / fewer-writes tie rule.
"""
import argparse,copy,json,math
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--noisy-lr',type=float,required=True);p.add_argument('--child-lr',type=float,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
if not all(math.isfinite(r) and r>0 for r in (a.noisy_lr,a.child_lr)): p.error('Rates must be positive and finite')
root=Path(__file__).resolve().parents[3]
x=json.loads((root/'examples/opt_cmo_drn/fault_paper_completion/drn12-a-holds-noisy.json').read_text())
a.output_dir.mkdir(parents=True,exist_ok=True)
for name,method,lr in [('matched-noiseless','matched_noiseless',a.noisy_lr),('child30','child_d',a.child_lr)]:
    y=copy.deepcopy(x); y['fault_tuning']['arms']=[{'label':f'{name}-lr{lr:g}','method':method,'learning_rate':lr}]
    if name=='child30':y['fault_tuning']['child_fraction']=.3
    target=a.output_dir/f'drn12-{name}.json'
    with target.open('x') as f:json.dump(y,f,indent=2);f.write('\n')
    print(target)
