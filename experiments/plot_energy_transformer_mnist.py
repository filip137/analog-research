"""Rebuild quality/cost figures and comparisons from explicit native bundles."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from networks.energy_transformer.mnist import unpatchify


def validation_points(history,summary):
    """Only the fixed probe cohort; never mix final full-validation MSE into it."""
    points={}
    for row in history:
        if row.get('stage') in ('initial_validation','validation_probe') and row.get('complete'):
            points[row['step']]={'step':row['step'],'seconds':row.get('train_seconds',0.),
                                 'mse':row['mse']['model'],'count':row['count']}
    final=summary.get('evaluation',{}).get('ablation_normal',{})
    if final.get('complete'):
        points[summary['updates']]={'step':summary['updates'],'seconds':summary['train_seconds'],
                                   'mse':final['mse']['model'],'count':final['count']}
    return [points[key] for key in sorted(points)]


def summarize(root):
    rows=json.loads((root/'coverage.json').read_text());out=root/'analysis';out.mkdir(exist_ok=True)
    comparisons=[]
    cases=[case for case in ('small','reference','wider','more_tokens') if any(r['case']==case for r in rows)]
    fig,axes=plt.subplots(len(cases),2,figsize=(11,3.5*len(cases)),layout='constrained',squeeze=False)
    vf,va=plt.subplots(len(cases),2,figsize=(11,3.5*len(cases)),layout='constrained',squeeze=False)
    for i,case in enumerate(cases):
        byarm={}
        for row in rows:
            if row['case']!=case or not row.get('run_dir'):continue
            run=Path(row['run_dir']);path=run/'metrics.jsonl'
            history=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            train=[r for r in history if r['stage']=='train']
            probes=validation_points(history,row.get('summary',{}))
            byarm[row['arm']]={'summary':row.get('summary',{}),'history':train,'probes':probes}
            if probes:
                va[i,0].plot([p['step'] for p in probes],[p['mse'] for p in probes],'.-',label=row['arm'])
                va[i,1].plot([p['seconds'] for p in probes],[p['mse'] for p in probes],'.-',label=row['arm'])
            if train:
                losses=np.array([r['mse'] for r in train]);window=min(25,len(losses))
                smooth=np.convolve(losses,np.ones(window)/window,mode='valid')
                axes[i,0].plot([r['step'] for r in train][window-1:],smooth,label=row['arm'])
                axes[i,1].plot([r['train_seconds'] for r in train][window-1:],smooth,label=row['arm'])
            examples=run/'artifacts/reconstructions.pt'
            if examples.exists():
                ex=torch.load(examples,map_location='cpu',weights_only=False)
                if ex:
                    p=4 if case=='more_tokens' else 7
                    target=(unpatchify(ex['target'],p)+.4)/.8
                    mask=unpatchify(ex['mask'],p)
                    pred=((unpatchify(ex['model'],p)+.4)/.8)*mask+target*(1-mask)
                    picture,grid=plt.subplots(3,min(8,len(target)),figsize=(12,4),squeeze=False)
                    for j in range(grid.shape[1]):
                        for k,value in enumerate((target[j]*(1-mask[j])+.5*mask[j],pred[j],target[j])):
                            grid[k,j].imshow(value,vmin=0,vmax=1,cmap='gray');grid[k,j].axis('off')
                    picture.suptitle(f"{case}: {row['arm']} — masked input / prediction / target")
                    picture.savefig(out/f'{case}-{row["arm"]}-reconstructions.png',dpi=150,bbox_inches='tight');plt.close(picture)
        pair={'case':case,'arms':{name:value['summary'] for name,value in byarm.items()}}
        pair['validation_probes']={name:value['probes'] for name,value in byarm.items()}
        if len(byarm)==2:
            available=[{p['step'] for p in v['probes']} for v in byarm.values()]
            shared=set.intersection(*available)
            pair['shared_validation_steps']=sorted(shared)
            if shared:
                common=max(shared)
                matched={name:next(p for p in value['probes'] if p['step']==common) for name,value in byarm.items()}
                if len({p['count'] for p in matched.values()})==1:
                    pair['matched_validation_at_update']=matched
            pair['wall_time_validation_note']='Only measured fixed-cohort probe times are shown; no interpolation or claim of identical wall budgets between sparse checkpoints.'
        if len(byarm)==2 and all(v['history'] for v in byarm.values()):
            step=min(v['history'][-1]['step'] for v in byarm.values())
            seconds=min(v['history'][-1]['train_seconds'] for v in byarm.values())
            pair['shared_update']=step;pair['shared_seconds']=seconds
            pair['matched_training_mse']={}
            for name,value in byarm.items():
                at_step=next(r for r in value['history'] if r['step']==step)
                available=[r for r in value['history'] if r['train_seconds']<=seconds]
                pair['matched_training_mse'][name]={'at_update':at_step['mse'],
                    'at_seconds':available[-1]['mse'] if available else None,
                    'seconds_measurement_step':available[-1]['step'] if available else None}
        comparisons.append(pair)
        for j in range(2):
            axes[i,j].set(title=case,xlabel='Updates' if j==0 else 'Training seconds',ylabel='Masked training MSE (25-batch mean)')
            axes[i,j].set_yscale('log');axes[i,j].grid(alpha=.2)
            if axes[i,j].lines:axes[i,j].legend()
            va[i,j].set(title=case,xlabel='Updates' if j==0 else 'Training seconds',ylabel='Fixed-cohort validation MSE')
            va[i,j].grid(alpha=.2)
            if va[i,j].lines:va[i,j].legend()
    fig.savefig(out/'learning.png',dpi=160);fig.savefig(out/'learning.svg');plt.close(fig)
    vf.savefig(out/'validation.png',dpi=160);vf.savefig(out/'validation.svg');plt.close(vf)
    (out/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    lines=['# FP32 MNIST comparison','',
           'Exploratory, one seed. Test predictions are not clipped. Partial evaluations and scientific stops remain visible.',
           '', '| Case | Arm | Updates | Stop | Test count | Test MSE | Initial MSE | Mean-image MSE | Peak MiB |',
           '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        s=row.get('summary',{});test=s.get('evaluation',{}).get('test',{});m=test.get('mse',{})
        fmt=lambda value:'—' if value is None else f'{value:.6g}'
        lines.append(f"| {row['case']} | {row['arm']} | {s.get('updates','—')} | {s.get('stop_reason',row['outcome'])} | {test.get('count',0)} | {fmt(m.get('model'))} | {fmt(m.get('initial'))} | {fmt(m.get('mean_image'))} | {fmt(s.get('peak_memory_bytes',0)/2**20)} |")
    lines+=['','`learning.png` compares training error at matched updates and elapsed times; these curves are not held-out scores.',
            '`validation.png` shows the identical fixed probe cohort at recorded updates/times; comparison.json lists the largest shared validation checkpoint. Sparse timestamps are not exact matched wall budgets.',
            'Final test scores compare final checkpoints; convergence failure or unequal update counts prevents treating them as a matched-training quality comparison.',
            'GPU time and memory describe this simulator, not analog silicon speed or energy.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);summarize(parser.parse_args().root)
