"""Coverage-checked fault-rate/depth comparisons, without endpoint selection."""
import argparse,csv,json,statistics,math
from collections import defaultdict
from pathlib import Path
from experiments.artifacts import atomic_write_json,sha256_file
from experiments.cifar_crossbar.sweep_config import SweepSpec,SOURCES,cases
from experiments.cifar_crossbar.hwa_fault_runtime import case_name
from experiments.cifar_crossbar.fault_config import METHODS


def validate(result, *, smoke=False):
    spec=SweepSpec()
    if smoke:
        from dataclasses import replace
        spec=replace(spec,fault_rates_ppm=(0,50000))
    wanted={(s,case_name(k,p),m) for s in SOURCES for k,p in cases(spec,s) for m in METHODS}
    rows=result['measurements']
    for row in rows:
        evaluations=[row['initial_test']]+[p['test'] for p in row['curve']]
        for value in evaluations:
            if any(not math.isfinite(value[k]) for k in ('teacher_kl','accuracy_percent','teacher_agreement_percent')):
                raise ValueError('Nonfinite evaluation metric.')
    if len(rows)!=len(wanted) or {(r['source'],r['case'],r['method']) for r in rows}!=wanted:raise ValueError('Missing/duplicate source/case controls.')
    if bool(result['smoke_only'])!=smoke or result['epochs']!=5 or result['recovery_images']!=(128 if smoke else 45000):raise ValueError('Wrong study budget.')
    grouped=defaultdict(list)
    for r in rows:grouped[(r['source'],r['case'])].append(r)
    for group in grouped.values():
        for key in ('p0_sha256','initial_apparent_sha256','identity','initial_test'):
            if len({json.dumps(r[key],sort_keys=True) for r in group})!=1:raise ValueError('Unpaired '+key)
        for r in group:
            if r['initial_test']['examples']!=(128 if smoke else 10000):raise ValueError('Incomplete deployment evaluation.')
            if r['method']=='none':
                if r['curve']:raise ValueError('Unexpected adaptation.')
                continue
            if [p['epoch'] for p in r['curve']]!=[1,2,3,4,5]:raise ValueError('Missing epoch.')
            for p in r['curve']:
                if p['image_presentations']!=p['epoch']*(128 if smoke else 45000) or p['test']['examples']!=(128 if smoke else 10000):raise ValueError('Wrong epoch data budget.')
                cost=p['cost']
                if result['backend']=='pcm':
                    expected=0 if r['method']=='calibration' else p['epoch']*(2 if smoke else 704)
                    if cost['array_reprogram_calls']!=expected:raise ValueError('Wrong PCM writing budget.')
                elif cost['max_recovery_cell_pulses']>min(640,p['epoch']*(2 if smoke else 704)):
                    raise ValueError('Wrong OM writing budget.')
    for case in {r['case'] for r in rows}:
        if len({json.dumps(r['identity'],sort_keys=True) for r in rows if r['case']==case})!=1:
            raise ValueError('Sources do not share physical identities.')
    return rows


def aggregate(rows,keys,metrics):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,values in sorted(groups.items()):
        if len(values)!=3 or len({v['array_seed'] for v in values})!=3:raise ValueError('Expected three independent arrays.')
        row=dict(zip(keys,key))
        for m in metrics:
            nums=[v[m] for v in values]
            if any(not math.isfinite(x) for x in nums):raise ValueError('Nonfinite aggregate metric.')
            row[m+'_mean']=statistics.mean(nums);row[m+'_sd']=statistics.stdev(nums)
            if m in ('kl_improvement','accuracy_gain_pp'):row[m+'_positive_arrays']=sum(x>0 for x in nums)
        out.append(row)
    return out


def csv_write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader();w.writerows(rows)


def report(root,output,datasets=('cifar10','cifar100'),revision=1):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    flat=[];paired=[];cross=[];costs=[];identities=[];clean=[];provenance=[];hwa=[];controls=0
    for ds in datasets:
        for backend in ('pcm','om'):
            for depth in (4,8):
                study=root/'results'/f'{ds}-{backend}-conv{depth}-fault-sweep-v{revision}'
                for arm in ('sources','screen1','screen2','screen3'):
                    candidates=[p for p in (study/'runs'/arm).glob('*/result.json') if json.loads(p.read_text())['status']=='complete']
                    if len(candidates)!=1:raise ValueError('Expected one completed native arm: '+str(study/arm))
                    path=candidates[0];r=json.loads(path.read_text())['metrics']
                    provenance.append({'path':str(path),'sha256':sha256_file(path)})
                    if arm=='sources':
                        for s,entry in r['sources'].items():hwa.append({'dataset':ds,'backend':backend,'depth':depth,'source':s,**entry})
                        continue
                    rows=validate(r);controls+=len(rows)
                    if {x['array_seed'] for x in rows}!={251000+int(arm[-1])}:raise ValueError('Unexpected array seed.')
                    by={(x['source'],x['case'],x['method']):x for x in rows}
                    for x in rows:
                        base={'dataset':ds,'backend':backend,'depth':depth,'source':x['source'],'case':x['case'],
                              'array_seed':x['array_seed'],'method':x['method']}
                        points=[{'epoch':0,'test':x['initial_test']}] if x['method']=='none' else x['curve']
                        for p in points:
                            flat.append({**base,'epoch':p['epoch'],**p['test']})
                            if x['method'] in ('onchip_weights','onchip_calibration'):
                                for control in ('calibration','rewrite'):
                                    ref=by[(x['source'],x['case'],control)]['curve'][p['epoch']-1]['test']
                                    paired.append({**base,'epoch':p['epoch'],'control':control,
                                        'kl_improvement':ref['teacher_kl']-p['test']['teacher_kl'],
                                        'accuracy_gain_pp':p['test']['accuracy_percent']-ref['accuracy_percent']})
                        if x['curve']:costs.append({**base,**x['curve'][-1]['cost']})
                        if x['method']=='none':
                            clean.append({**base,**x['clean']})
                            if x['source']=='digital':identities.append({**base,**x['identity']})
                        if x['method'] in ('onchip_weights','onchip_calibration'):
                            endpoint=x['curve'][-1]['test']
                            for control_row in rows:
                                if control_row['case']!=x['case'] or control_row['method'] not in ('none','calibration','rewrite'):continue
                                ref=control_row['initial_test'] if control_row['method']=='none' else control_row['curve'][-1]['test']
                                cross.append({**base,'control_source':control_row['source'],'control':control_row['method'],
                                    'kl_improvement':ref['teacher_kl']-endpoint['teacher_kl'],
                                    'accuracy_gain_pp':endpoint['accuracy_percent']-ref['accuracy_percent']})
    keys=['dataset','backend','depth','source','case','method','epoch']
    summary=aggregate(flat,keys,['teacher_kl','accuracy_percent','teacher_agreement_percent'])
    pairs=aggregate(paired,keys+['control'],['kl_improvement','accuracy_gain_pp'])
    cross_summary=aggregate(cross,[k for k in keys if k!='epoch']+['control_source','control'],['kl_improvement','accuracy_gain_pp'])
    out={'datasets':list(datasets),'controls':controls,'learning_trajectories':controls*4//5,
         'summary':summary,'paired':pairs,'cross_source':cross_summary,'sources':hwa,'provenance':provenance,
         'analysis_code_sha256':sha256_file(Path(__file__))}
    atomic_write_json(output/'results.json',out)
    for name,rows in [('per_array',flat),('summary',summary),('paired_per_array',paired),('paired_summary',pairs),('costs',costs),('fault_identities',identities),('clean_sources',clean),('cross_source_per_array',cross),('cross_source_summary',cross_summary)]:csv_write(output/(name+'.csv'),rows)
    lines=['# CIFAR stuck-device sweep','',f'{controls} controls; five full 45000-image epochs, three arrays per condition.','',
           'KL is mean KL(original digital teacher || student), nats at T=1. Values are means over three arrays; CSV includes sample SD and every epoch.',
           '', '## Corruption-aware HWA and joint recovery at epoch five','',
           '| Dataset | Device | Convs | Failure | Rate | CDT deployed KL / accuracy | CDT calibration | CDT rewrite + calibration | CDT joint recovery |',
           '|---|---|---:|---|---:|---|---|---|---|']
    lookup={(x['dataset'],x['backend'],x['depth'],x['source'],x['case'],x['method'],x['epoch']):x for x in summary}
    def fmt(x):return f"{x['teacher_kl_mean']:.4f} / {x['accuracy_percent_mean']:.2f}%"
    for ds in datasets:
        for b in ('pcm','om'):
            for d in (4,8):
                for k in ('open','gmax','random'):
                    for rate in (10000,20000,30000,50000):
                        prefix=(ds,b,d,'cdt_'+k,case_name(k,rate))
                        vals=[fmt(lookup[prefix+(m,e)]) for m,e in [('none',0),('calibration',5),('rewrite',5),('onchip_calibration',5)]]
                        lines.append('| '+' | '.join([ds,b,str(d),{'open':'stuck low','gmax':'stuck high','random':'random stuck'}[k],str(rate/10000)+'%',*vals])+' |')
    lines+=['','## Interpretation limits','','Exploratory model-based evidence. HWA training has one seed; array spread is not training-seed spread. Report late-improving source fits. PCM endpoint rewrites and OM pulses are different costs.','']
    (output/'report.md').write_text('\n'.join(lines))
    plots(summary,output,datasets)
    epoch_plots(summary,output,datasets)
    return out


def plots(summary,output,datasets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup={(x['dataset'],x['backend'],x['depth'],x['source'],x['case'],x['method'],x['epoch']):x for x in summary}
    for ds in datasets:
        for backend in ('pcm','om'):
            for kind in ('open','gmax','random'):
                fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
                curves=[('standard_hwa','calibration',5,'standard HWA + calibration'),('noise_hwa','calibration',5,'noise HWA + calibration'),
                        ('cdt_'+kind,'none',0,'CDT deployed'),('cdt_'+kind,'calibration',5,'CDT + calibration'),
                        ('cdt_'+kind,'rewrite',5,'CDT + rewrite/calibration'),('cdt_'+kind,'onchip_weights',5,'CDT + weight recovery'),
                        ('cdt_'+kind,'onchip_calibration',5,'CDT + joint recovery')]
                for col,depth in enumerate((4,8)):
                    for source,method,epoch,label in curves:
                        vals=[lookup[(ds,backend,depth,source,case_name(kind,p),method,epoch)] for p in (0,10000,20000,30000,50000)]
                        for row,metric in enumerate(('teacher_kl','accuracy_percent')):
                            axes[row,col].errorbar([0,1,2,3,5],[v[metric+'_mean'] for v in vals],yerr=[v[metric+'_sd'] for v in vals],label=label,marker='o',capsize=2,linewidth=1)
                    axes[0,col].set_title(f'{depth} analog convolutions + classifier')
                    axes[1,col].set_xlabel('Failed devices (%)')
                axes[0,0].set_ylabel('Teacher KL (nats)');axes[1,0].set_ylabel('Test accuracy (%)')
                for ax in axes.flat:ax.grid(alpha=.2);ax.set_xticks([0,1,2,3,5])
                fig.suptitle(f'{ds.upper()} / {backend.upper()} / '+{'open':'stuck low','gmax':'stuck high','random':'random stuck'}[kind])
                handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
                fig.tight_layout(rect=(0,.18,1,.95))
                for ext in ('png','pdf'):fig.savefig(output/f'{ds}_{backend}_{kind}.{ext}',dpi=160)
                plt.close(fig)


def epoch_plots(summary,output,datasets):
    import matplotlib.pyplot as plt
    lookup={(x['dataset'],x['backend'],x['depth'],x['source'],x['case'],x['method'],x['epoch']):x for x in summary}
    for ds in datasets:
        for backend in ('pcm','om'):
            for depth in (4,8):
                fig,axes=plt.subplots(3,4,figsize=(14,10),sharex=True)
                for row,kind in enumerate(('open','gmax','random')):
                    for col,rate in enumerate((10000,20000,30000,50000)):
                        ax=axes[row,col];prefix=(ds,backend,depth,'cdt_'+kind,case_name(kind,rate))
                        initial=lookup[prefix+('none',0)]['teacher_kl_mean']
                        for method in ('calibration','rewrite','onchip_weights','onchip_calibration'):
                            values=[initial]+[lookup[prefix+(method,e)]['teacher_kl_mean'] for e in range(1,6)]
                            ax.plot(range(6),values,marker='o',markersize=3,label=method)
                        ax.set_title(f'{kind}, {rate/10000:g}%');ax.grid(alpha=.2)
                        if col==0:ax.set_ylabel('Teacher KL (nats)')
                        if row==2:ax.set_xlabel('Full adaptation epochs')
                fig.suptitle(f'{ds.upper()} / {backend.upper()} / {depth} analog convolutions + classifier: CDT recovery')
                h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4)
                fig.tight_layout(rect=(0,.05,1,.96))
                for ext in ('png','pdf'):fig.savefig(output/f'{ds}_{backend}_conv{depth}_epochs.{ext}',dpi=160)
                plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--revision',type=int,default=1)
    p.add_argument('--datasets',nargs='+',choices=['cifar10','cifar100'],default=['cifar10','cifar100']);a=p.parse_args()
    report(a.root,a.output,a.datasets,a.revision)
if __name__=='__main__':main()
