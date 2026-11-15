import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);results={};fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    names=['drn-dot-ep','drn-dot-bptt','drn-l1-ep','digital-dot-bptt','drn-l1_raw-ep']
    colors=dict(zip(names,['#287eb3','#73a9d0','#d98235','#888888','#479657']))
    for name in names:
        files=list((a.root/name).glob('*/artifacts/summary.json'))
        if len(files)!=1:raise ValueError(f'Expected one final result: {name}')
        path=files[0];b=path.parent.parent;s=json.loads(path.read_text());g=json.loads((path.parent/'gradients-final.json').read_text())
        rows=[json.loads(line) for line in (b/'metrics.jsonl').read_text().splitlines()];v=[r for r in rows if r['stage']=='validation_probe']
        ax[0].plot([0]+[r['step'] for r in v],[s['initial']['accuracy']]+[r['accuracy'] for r in v],label=name,color=colors[name])
        caps=[]
        for cap in [8,16,32,64]:
            vals=[z for batch in g for row in batch['caps'] if row['cap']==cap for group in row['groups'].values() for z in group.values() if z['reference_norm']>1e-9]
            caps.append({'cap':cap,'min_cosine':min(z['cosine'] for z in vals),'max_relative_l2':max(z['relative_l2'] for z in vals)})
        ax[2].loglog([r['cap'] for r in caps],[r['max_relative_l2'] for r in caps],marker='o',label=name,color=colors[name])
        results[name]={'bundle':str(b),'summary':s,'cap_comparison':caps}
    ax[0].set(title='Fixed 256-image validation probe',xlabel='Update',ylabel='Accuracy')
    bars=ax[1].bar(range(len(names)),[results[n]['summary']['test']['accuracy'] for n in names],color=[colors[n] for n in names])
    ax[1].bar_label(bars,labels=[f"{results[n]['summary']['test']['accuracy']:.1%}" for n in names],fontsize=8)
    ax[1].set_xticks(range(len(names)),['DRN dot\nEP','DRN dot\nBPTT','DRN L1\nEP','Digital MLP\nBPTT','DRN raw L1\nEP'],fontsize=8);ax[1].set(title='Full 10,000-image test',ylabel='Accuracy',ylim=(0,1))
    ax[2].axhline(.05,color='gray',ls='--');ax[2].set(title='EP / BPTT / long reference',xlabel='Sweeps per phase',ylabel='Worst parameter-group relative L2')
    ax[2].set_xticks([8,16,32,64],['8','16','32','64']);ax[2].minorticks_off()
    for i in [0,2]:ax[i].legend(fontsize=7)
    for axis in ax:axis.grid(alpha=.2);axis.spines[['top','right']].set_visible(False)
    fig.suptitle('CIFAR-10 · 64 tokens · 2,000 updates · 32 sweeps per phase · one seed')
    for ext in ['png','svg']:fig.savefig(a.output/f'results.{ext}',dpi=180)
    svg=a.output/'results.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    for n,r in results.items():print(n,json.dumps(r['summary']))

if __name__=='__main__':main()
