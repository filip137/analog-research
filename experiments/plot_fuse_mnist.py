"""Derived comparison and scientific figures; never mutates run artifacts."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from networks.energy_transformer.mnist import unpatchify
from ebl.runtime.store import atomic_write_json,sha256_file

BASELINE=Path('/home/filip/analog-research/drn/results/energy-transformer-repair/training-003/analysis/comparison.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    source=root/'result.json' if (root/'result.json').exists() else root/'comparison.json'
    data=json.loads(source.read_text());historical={r['case']:r for r in json.loads(BASELINE.read_text())}
    out=root/'analysis';out.mkdir(exist_ok=True)
    cases=('small','reference','more_tokens');names={'small':'16 tokens × 32','reference':'16 tokens × 64','more_tokens':'49 tokens × 64'}
    colors={'gaussian':'#b5462c','linear':'#2367a2','energy_ep':'#775fa6','digital_bp':'#25846c'}
    rows=[];fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True,constrained_layout=True)
    for ax,case in zip(axes,cases):
        arms={r['kernel']:r for r in data['cases'] if r['case']==case and 'summary' in r}
        base=historical[case]['arms']
        mean_probe=None
        for kernel,r in arms.items():
            s=r['summary'];test=s['evaluation']['test'];mse=test['mse'].get('model')
            rows.append({'case':case,'kernel':kernel,'updates':s['updates'],'test_count':test['count'],'mse':mse,
                         'mean_image_mse':test['mse'].get('mean_image'),'initial_mse':test['mse'].get('initial'),
                         'train_seconds':s['train_seconds'],'accepted':r.get('accepted',False),
                         'trainable_conductances':s['trainable_conductances'],'physical_resistors':s['physical_resistors'],
                         'physical_fuses':s['physical_fuses'],'gradient':r.get('gradients',{}).get('global'),
                         'ablations':{k:v['mse'].get('model') for k,v in s['evaluation'].items() if k.startswith('ablation_')},
                         'final_fuse_mean':s['last_training_diagnostics'].get('fuse_mean'),'final_fuse_min':s['last_training_diagnostics'].get('fuse_min')})
            metrics=[json.loads(l) for l in (Path(r['run_dir'])/'metrics.jsonl').read_text().splitlines()]
            probe=[m for m in metrics if m['stage'] in ('initial_validation','validation_probe') and m.get('complete')]
            mean_probe=probe[0]['mse'].get('mean_image')
            ax.plot([m['step'] for m in probe],[m['mse']['model'] for m in probe],label=kernel,color=colors[kernel],marker='o',ms=3)
        for name in ('energy_ep','digital_bp'):
            probe=historical[case]['validation_probes'][name]
            ax.plot([r['step'] for r in probe],[r['mse'] for r in probe],color=colors[name],linestyle='--',alpha=.8,label='prior '+name)
        if mean_probe is not None:
            ax.axhline(mean_probe,color='#555555',linestyle=':',label='mean image')
        ax.set_title(names[case]);ax.set_xlabel('Training updates');ax.grid(alpha=.2)
        if len(arms)==2:
            rec={k:torch.load(Path(r['run_dir'])/'artifacts/reconstructions.pt',map_location='cpu',weights_only=False) for k,r in arms.items()}
            patch=4 if case=='more_tokens' else 7;gr=rec['gaussian'];lr=rec['linear']
            for key in ('target','mask'):assert torch.equal(gr[key],lr[key])
            target=unpatchify(gr['target'],patch);mask=unpatchify(gr['mask'],patch)
            observed=target*(1-mask)+(-.4)*mask
            fg=unpatchify(gr['model'],patch)*mask+target*(1-mask)
            fl=unpatchify(lr['model'],patch)*mask+target*(1-mask)
            fm=unpatchify(gr['mean_image'],patch)*mask+target*(1-mask)
            fig2,axs=plt.subplots(4,5,figsize=(8,6.5),constrained_layout=True)
            for j,(title,im) in enumerate((('Observed',observed),('Target',target),('Fuse EP',fg),('Linear EP',fl),('Mean image',fm))):
                for i in range(4):
                    axs[i,j].imshow((im[i].numpy()+.4)/.8,cmap='gray',vmin=0,vmax=1);axs[i,j].axis('off')
                    if i==0:axs[i,j].set_title(title)
            fig2.suptitle(names[case]+' — same first four test images; observed pixels retained')
            fig2.savefig(out/f'reconstructions-{case}.png',dpi=180);fig2.savefig(out/f'reconstructions-{case}.pdf');plt.close(fig2)
    axes[0].set_ylabel('Masked MSE in [0,1] image units')
    axes[-1].legend(fontsize=8);fig.suptitle('Identical 256-image validation cohort — fresh fuse controls and historical ET/BP')
    fig.savefig(out/'learning.png',dpi=180);fig.savefig(out/'learning.pdf');plt.close(fig)
    atomic_write_json(out/'comparison.json',{'rows':rows,'source':str(source),'source_sha256':sha256_file(source),'baseline':str(BASELINE),'baseline_sha256':sha256_file(BASELINE)})
    text=['# Fuse MNIST comparison','','Single seed; same masked reconstruction task; architectures are not parameter matched.','',
          '| Tokens × width | Kernel | Updates | Test n | Test MSE | Mean image | EP training s | Numerical acceptance |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        text.append(f"| {names[r['case']]} | {r['kernel']} | {r['updates']} | {r['test_count']} | {r['mse']:.7f} | {r['mean_image_mse']:.7f} | {r['train_seconds']:.2f} | {r['accepted']} |")
    (out/'comparison.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
