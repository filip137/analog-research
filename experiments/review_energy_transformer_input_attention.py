"""Independent collection of the explicit input-attention comparison bundles."""
import argparse
import json
from pathlib import Path
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ebl.runtime.store import atomic_write_json, sha256_file
from experiments.plot_energy_transformer_mnist import validation_points
from networks.energy_transformer.mnist import unpatchify


def read(path):
    return json.loads(path.read_text())


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):
        return isinstance(b,type(a)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b



def weight_ranks(runs):
    result={}
    torch.set_num_threads(1)
    for name,run in runs.items():
        records=[]
        for file in ('initial.pt',*(f'step-{step:04d}.pt' for step in range(250,2001,250))):
            path=run/'checkpoints'/file
            if not path.exists():continue
            payload=load(path);groups={}
            for group in ('query','key','value','output','memory','ff1','ff2','decoder'):
                if group not in payload['model']:continue
                values=torch.linalg.svdvals(payload['model'][group].double())
                stable=values.square().sum(-1)/values[...,0].square()
                groups[group]={'stable_rank':stable.tolist(),'singular_values':values.tolist()}
            records.append({'step':payload['step'],'path':str(path),'sha256':sha256_file(path),'groups':groups})
        result[name]=records
    return result


def collect(root,old_root):
    terminal=read(root/'result.json')
    new={r['name']:Path(r['run_dir']) for r in read(root/'coverage.json') if r.get('run_dir')}
    old={r['arm']:Path(r['run_dir']) for r in read(old_root/'coverage.json') if r['case']=='more_tokens'}
    required=('input-energy_ep','input-energy_bp','recurrent-energy_bp','gradient-selection','gradient-final')
    if not all(name in new for name in required):
        raise RuntimeError('Full comparison not collected; inspect the retained partial/negative result')
    runs={'new_ep':new['input-energy_ep'],'new_bp':new['input-energy_bp'],
          'old_bp':new['recurrent-energy_bp'],'old_ep':old['energy_ep'],'digital':old['digital_bp']}
    summaries={k:read(p/'artifacts/summary.json') for k,p in runs.items()}
    histories={k:[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()] for k,p in runs.items()}
    checks={}
    manifest={k:read(p/'manifest.json') for k,p in runs.items()}
    expected_source=read(root/'launch.json')['source_commit']
    checks['new_sources_clean']=all(manifest[k]['source']['commit']==expected_source and not manifest[k]['source']['dirty']
                                   for k in ('new_ep','new_bp','old_bp'))
    dataset=lambda m:{v['path']:v['sha256'] for v in m['inputs'] if '/MNIST/raw/' in v['path']}
    checks['identical_mnist_inputs']=all(dataset(m)==dataset(manifest['old_ep']) for m in manifest.values())
    initial={k:load(p/'checkpoints/initial.pt') for k,p in runs.items() if k!='digital'}
    final={k:load(p/'checkpoints/final.pt') for k,p in runs.items() if k!='digital'}
    checks['new_pair_initial_model_equal']=equal(initial['new_ep']['model'],initial['new_bp']['model'])
    checks['old_pair_initial_model_equal']=equal(initial['old_ep']['model'],initial['old_bp']['model'])
    paired_config=lambda c:{k:v for k,v in c.items() if k!='arm'}
    checks['new_pair_config_equal']=paired_config(final['new_ep']['config'])==paired_config(final['new_bp']['config'])
    for a,b in (('new_ep','new_bp'),('old_ep','old_bp'),('new_ep','old_ep')):
        checks[f'{a}_{b}_rng_equal']=all(torch.equal(final[a][key],final[b][key]) for key in ('order_rng','mask_rng'))
        checks[f'{a}_{b}_cohorts_equal']=equal(load(runs[a]/'artifacts/cohorts.pt'),load(runs[b]/'artifacts/cohorts.pt'))
    checks['all_weights_fp32_finite']=all(v.dtype==torch.float32 and bool(torch.isfinite(v).all())
        for payload in final.values() for v in payload['model'].values())
    checks['all_full_coverage']=all(s.get('coverage_complete') and s['updates']==2000 and
        s['evaluation']['test']['count']==10000 and s['evaluation']['validation']['count']==5000 and
        all(s['evaluation']['ablation_'+a]['count']==256 for a in ('normal','uniform','zero')) for s in summaries.values())
    training_checks={}
    for name in ('new_ep','new_bp','old_bp'):
        train=[v for v in histories[name] if v.get('stage')=='train']
        training_checks[name]=dict(count=len(train),
            max_residual=max(v['residual'] for v in train),
            max_normalized_bound=max(v['recurrent_bound'] for v in train),
            max_contraction_radius=max(v['contraction_radius'] for v in train),
            minimum_triode_margin=min(v['minimum_triode_margin'] for v in train),
            minimum_drain_margin=min(v['minimum_drain_margin'] for v in train),
            minimum_weak_inversion_margin=min(v['minimum_weak_inversion_margin'] for v in train))
        t=training_checks[name]
        checks[name+'_training_valid']=(len(train)==2000 and [v['step'] for v in train]==list(range(1,2001))
            and t['max_normalized_bound']<=.7 and t['max_contraction_radius']<1
            and all(t[k]>0 for k in ('minimum_triode_margin','minimum_drain_margin','minimum_weak_inversion_margin')))
    diagnostic={}
    for name in ('gradient-selection','gradient-final'):
        path=new[name];s=read(path/'artifacts/summary.json');m=read(path/'manifest.json')
        rows=s['rows'];groups=('query','key','value','output')
        selected=[r for r in rows if r['iterations']==terminal['selected_cap']]
        details={}
        for metric in ('ep_vs_bptt','ep_vs_reference','bptt_vs_reference'):
            details[metric]=dict(global_min=min(r[metric]['global']['cosine'] for r in selected),
                required_group_min=min(r[metric]['groups'][g]['cosine'] for r in selected for g in groups),
                all_group_min=min(v['cosine'] for r in selected for v in r[metric]['groups'].values() if v['cosine'] is not None),
                max_group_relative_l2=max(r[metric]['groups'][g]['relative_l2'] for r in selected for g in groups))
        checkpoint=next(v for v in m['inputs'] if v['path'].endswith('/checkpoints/final.pt'))
        checks[name+'_checkpoint_unchanged']=sha256_file(Path(checkpoint['path']))==checkpoint['sha256']
        checks[name+'_complete_valid']=(s['coverage_complete'] and s['reference_valid']
            and s['selected_cap']==terminal['selected_cap'] and len(selected)==8
            and {r['batch'] for r in selected}==set(range(8)) and set(s['required_groups'])==set(groups)
            and all(v['global_min']>=.9 and v['required_group_min']>=.9 for v in details.values()))
        diagnostic[name]=dict(run_dir=str(path),checkpoint=checkpoint,selected_cap=s['selected_cap'],metrics=details)
    checks['same_gradient_cohort']=equal(load(new['gradient-selection']/'artifacts/cohort.pt'),
                                         load(new['gradient-final']/'artifacts/cohort.pt'))
    inspection_paths=list(root.glob('*/attempt-*/inspection.json'))
    # Inspector command exit status is retained by each launcher row; reread
    # native summaries/checkpoints above rather than trusting cached summaries.
    checks['native_inspections']=all(r.get('integrity_passed') for r in read(root/'coverage.json'))
    rows=[]
    for name,s in summaries.items():
        e=s['evaluation'];normal=e['ablation_normal']['mse']['model']
        rows.append(dict(name=name,run_dir=str(runs[name]),parameters=s['total_parameters'],
            updates=s['updates'],steps=s['max_steps'],test_mse=e['test']['mse']['model'],
            validation_mse=e['validation']['mse']['model'],probe_mse=normal,
            uniform_penalty_percent=100*(e['ablation_uniform']['mse']['model']/normal-1),
            zero_penalty_percent=100*(e['ablation_zero']['mse']['model']/normal-1),
            attention_mean_max=e['ablation_normal'].get('attention_mean_max'),
            attention_normalized_entropy=e['ablation_normal'].get('attention_normalized_entropy'),
            training_seconds=s['train_seconds'],peak_mib=s['peak_memory_bytes']/2**20))
    byname={r['name']:r for r in rows}
    improvement=100*(1-byname['new_ep']['test_mse']/byname['old_ep']['test_mse'])
    bp_gap=100*(byname['new_ep']['test_mse']/byname['new_bp']['test_mse']-1)
    verdict=dict(numerical_and_coverage_pass=all(checks.values()),
        ep_test_improvement_percent=improvement,new_ep_vs_bp_gap_percent=bp_gap,
        meaningful_quality_gain=improvement>=5,
        meaningful_routing=byname['new_ep']['uniform_penalty_percent']>=1)
    verdict['promote']=all(verdict[k] for k in ('numerical_and_coverage_pass','meaningful_quality_gain','meaningful_routing'))
    out=root/'analysis';out.mkdir(exist_ok=True)
    review=dict(checks=checks,training_checks=training_checks,diagnostics=diagnostic,
        comparison=rows,verdict=verdict,source_commit=expected_source,
        seconds=terminal['seconds'],storage_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        inspection_paths=[str(p) for p in inspection_paths],
        limits='One seed, MNIST completion, ideal circuit equations. Parameter count and fixed input features change between architectures; only within-architecture EP/BPTT is a matched learning-rule comparison. GPU costs are simulator costs.')
    ranks=weight_ranks(runs)
    atomic_write_json(out/'weight-ranks.json',ranks)
    review['weight_rank_artifact']=str(out/'weight-ranks.json')
    atomic_write_json(out/'review.json',review)
    labels={'new_ep':'Input attention / EP','new_bp':'Input attention / BPTT',
            'old_ep':'Recurrent attention / EP','old_bp':'Recurrent attention / BPTT','digital':'Digital reference'}
    styles={'new_ep':('tab:orange','-'),'new_bp':('tab:blue','-'),
            'old_ep':('tab:orange','--'),'old_bp':('tab:blue','--'),'digital':('0.4',':')}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name in runs:
        points=validation_points(histories[name],summaries[name]);color,style=styles[name]
        for ax,key in zip(axes,('step','seconds')):
            ax.plot([p[key] for p in points],[p['mse'] for p in points],linestyle=style,
                    marker='.',color=color,label=labels[name])
    for ax in axes:
        ax.set_ylabel('Fixed 256-image probe MSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_xlabel('Training updates');axes[1].set_xlabel('Measured training seconds')
    fig.suptitle('49-token FP32 comparison — one seed, ideal circuit simulation')
    for suffix in ('png','svg'):fig.savefig(out/f'validation.{suffix}',dpi=160)
    plt.close(fig)
    detail,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    rank_fig,rank_ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name in runs:
        points=[p for p in validation_points(histories[name],summaries[name]) if p['step']>=250]
        color,style=styles[name]
        for axis,key in zip(ax,('step','seconds')):
            axis.plot([p[key] for p in points],[p['mse'] for p in points],linestyle=style,
                      marker='.',color=color,label=labels[name])
        for axis,group in zip(rank_ax,('ff1' if name=='digital' else 'memory','query')):
            records=ranks[name]
            axis.plot([r['step'] for r in records],
                      [np.mean(r['groups'][group]['stable_rank']) for r in records],
                      linestyle=style,marker='.',color=color,label=labels[name])
    for axis in (*ax,*rank_ax):axis.grid(alpha=.2);axis.legend(fontsize=8)
    ax[0].set(xlabel='Training updates',ylabel='Fixed 256-image probe MSE')
    ax[1].set(xlabel='Measured training seconds',ylabel='Fixed 256-image probe MSE')
    detail.suptitle('Validation detail — measured points from update250 onward')
    rank_ax[0].set(xlabel='Training updates',ylabel='Stable rank',title='Memory / digital FF1')
    rank_ax[1].set(xlabel='Training updates',ylabel='Mean head stable rank',title='Query maps')
    rank_fig.suptitle('Strong learned directions: squared Frobenius norm / squared spectral norm')
    for suffix in ('png','svg'):
        detail.savefig(out/f'validation-detail.{suffix}',dpi=160)
        rank_fig.savefig(out/f'weight-ranks.{suffix}',dpi=160)
    plt.close(detail);plt.close(rank_fig)
    # Identical examples and masks, with no outcome-dependent image selection.
    examples={name:load(runs[name]/'artifacts/reconstructions.pt') for name in ('old_ep','new_ep','new_bp')}
    checks['reconstruction_cohort_equal']=all(torch.equal(v[k],examples['old_ep'][k])
        for v in examples.values() for k in ('target','mask','inputs'))
    count=6;fig,axes=plt.subplots(5,count,figsize=(10,8),layout='constrained')
    target=(unpatchify(examples['old_ep']['target'],4)+.4)/.8
    mask=unpatchify(examples['old_ep']['mask'],4)
    pictures=[target*(1-mask)+.5*mask]
    for name in ('old_ep','new_ep','new_bp'):
        prediction=(unpatchify(examples[name]['model'],4)+.4)/.8
        pictures.append(prediction*mask+target*(1-mask))
    pictures.append(target)
    for i,label in enumerate(('Observed input','Old EP','New EP','New BPTT','Target')):
        for j in range(count):
            axes[i,j].imshow(pictures[i][j],cmap='gray',vmin=0,vmax=1)
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
        axes[i,0].set_ylabel(label)
    fig.savefig(out/'reconstructions.png',dpi=160);plt.close(fig)
    # Include the final image-cohort verification in the saved conclusion.
    verdict['numerical_and_coverage_pass']=all(checks.values())
    verdict['promote']=all(verdict[k] for k in ('numerical_and_coverage_pass','meaningful_quality_gain','meaningful_routing'))
    atomic_write_json(out/'review.json',review)
    lines=['# Input-attention comparison','',review['limits'],'',
        '| Model | Parameters | Test MSE | Uniform penalty | Training seconds |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {labels[r['name']]} | {r['parameters']} | {r['test_mse']:.7f} | {r['uniform_penalty_percent']:.3f}% | {r['training_seconds']:.1f} |")
    lines+=['',f"New EP improvement over old EP: {improvement:.3f}%; new EP/BPTT gap: {bp_gap:.3f}%.",
            f"All numerical/coverage checks pass: {verdict['numerical_and_coverage_pass']}. Predeclared promotion rule passes: {verdict['promote']}.",
            '', 'Full native identities, gradient gates and physical minima are in review.json.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verdict=verdict,comparison=rows,failed_checks=[k for k,v in checks.items() if not v]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--old-root',type=Path,required=True)
    args=parser.parse_args();collect(args.root,args.old_root)
