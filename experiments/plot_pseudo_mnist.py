"""Collect explicit one-run-per-arm results and draw learning/gradient checks."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained');results={}
    for ti,task in enumerate(['completion','permuted_classification']):
        for method in ['ep','bptt']:
            name=f'{task}-{method}';bundles=list((a.root/name).glob('*/artifacts/summary.json'))
            if len(bundles)!=1: raise ValueError(f'Expected one explicit complete result for {name}')
            f=bundles[0];bundle=f.parent.parent
            rows=[json.loads(l) for l in (bundle/'metrics.jsonl').read_text().splitlines()]
            summary=json.loads(f.read_text());grad=json.loads((f.parent/'gradients-final.json').read_text())
            probes=[r for r in rows if r['stage']=='validation_probe']
            key='mse' if task=='completion' else 'accuracy'
            axes[ti].plot([0]+[r['step'] for r in probes],[summary['initial_validation_probe'][key]]+[r[key] for r in probes],label=method.upper())
            caps=[r['cap'] for r in grad[0]['caps']]
            stats=[]
            for cap in caps:
                groups=[r['groups'] for b in grad for r in b['caps'] if r['cap']==cap]
                stats.append({'cap':cap,'min_global_cosine':min(g['global']['ep_bptt']['cosine'] for g in groups),
                    'max_global_relative_l2':max(g['global']['ep_bptt']['relative_l2'] for g in groups),
                    'min_group_cosine':min(v['ep_bptt']['cosine'] for g in groups for v in g.values()),
                    'max_group_relative_l2':max(v['ep_bptt']['relative_l2'] for g in groups for v in g.values()),
                    'max_all_comparisons_relative_l2':max(v[k]['relative_l2'] for g in groups for v in g.values() for k in ['ep_bptt','ep_exact','bptt_exact'])})
            axes[2].loglog(caps,[s['max_all_comparisons_relative_l2'] for s in stats],marker='o',label=f'{task.split("_")[0]} / {method}')
            results[name]={'bundle':str(bundle),'summary':summary,'cap_comparison':stats}
    axes[0].set(title='Masked completion',xlabel='Training update',ylabel='Fixed 256-image validation MSE')
    axes[1].set(title='Fixed-permutation classification',xlabel='Training update',ylabel='Fixed 256-image validation accuracy')
    axes[2].axhline(.05,color='gray',ls='--');axes[2].set(title='Final EP / BPTT / exact equilibrium',xlabel='Sweeps per phase',ylabel='Worst group relative gradient L2 error')
    for ax in axes: ax.legend(fontsize=8);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Ideal pseudoconductance attention · 32 sweeps per phase during training · one seed')
    for suffix in ['png','svg']:fig.savefig(a.output/f'results.{suffix}',dpi=180)
    (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:v['summary'] for k,v in results.items()},indent=2))

if __name__=='__main__':main()
