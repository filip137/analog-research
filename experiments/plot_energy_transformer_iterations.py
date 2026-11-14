"""Read iteration curves from explicit validated run directories."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(runs,output):
    output.mkdir(parents=True,exist_ok=True)
    summaries=[json.loads((run/'artifacts/summary.json').read_text()) for run in runs]
    fig,axes=plt.subplots(1,len(summaries),figsize=(5*len(summaries),4),squeeze=False,layout='constrained')
    table=[]
    for ax,s in zip(axes[0],summaries):
        caps=sorted({r['iterations'] for r in s['rows']})
        for key,label in (('ep_vs_bptt','EP / same-length BPTT'),('ep_vs_reference','EP / long BPTT'),('bptt_vs_reference','BPTT / long BPTT')):
            low=[];mean=[]
            for cap in caps:
                values=[r[key]['global']['cosine'] for r in s['rows'] if r['iterations']==cap]
                values=[v for v in values if v is not None]
                low.append(min(values) if values else float('nan'));mean.append(sum(values)/len(values) if values else float('nan'))
            ax.plot(caps,low,'.-',label=label+' (min batch)')
            ax.plot(caps,mean,alpha=.25,linestyle='--')
        ax.axhline(.9,color='black',linestyle=':');ax.set(title=s['case'],xlabel='Iterations per phase',ylabel='Gradient cosine',ylim=(-1.05,1.02))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        for cap in caps:
            rows=[r for r in s['rows'] if r['iterations']==cap]
            table.append({'case':s['case'],'iterations':cap,
                **{key:min((r[key]['global']['cosine'] for r in rows if r[key]['global']['cosine'] is not None),default=None)
                   for key in ('ep_vs_bptt','ep_vs_reference','bptt_vs_reference')}})
    fig.savefig(output/'gradient-agreement.png',dpi=160);fig.savefig(output/'gradient-agreement.svg');plt.close(fig)
    complete={s['case'] for s in summaries}=={'small','reference','more_tokens'} and all(s['coverage_complete'] and s['reference_valid'] for s in summaries)
    common=set.intersection(*(set(s['passing_caps']) for s in summaries)) if summaries else set()
    selection={'complete':complete,'common_passing_caps':sorted(common) if complete else [],
               'selected_common_cap':min(common) if complete and common else None,
               'per_case':{s['case']:s['selected_cap'] for s in summaries},'rows':table,
               'note':'Global cosine selection; inspect all six parameter-group directions and norm ratios in the native summaries before interpreting attention fidelity.'}
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    lines=['# Trained-checkpoint iteration diagnostic','',
           'FP32; checkpoint after250updates; identical24validation examples. Entries are minimum global cosine over three minibatches. Longer512/1024unrolls only qualify the reference.','',
           '| Case | Iterations/phase | EP vs BPTT | EP vs reference | BPTT vs reference |',
           '| --- | ---: | ---: | ---: | ---: |']
    for row in table:
        vals=[('—' if row[k] is None else f'{row[k]:.5f}') for k in ('ep_vs_bptt','ep_vs_reference','bptt_vs_reference')]
        lines.append(f"| {row['case']} | {row['iterations']} | {' | '.join(vals)} |")
    lines+=['',f"Selected common cap: {selection['selected_common_cap']}; complete qualified coverage: {complete}.",
            'A cap T means T free and T in each nudged phase:3T serial iterations. Finite-time EP is approximate; no optimization occurred in this diagnostic.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(selection,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.runs,a.output)
