"""Validate collected exp-014 evidence and apply the frozen context decision rule.

CPU-only analysis: never dispatches, changes a lifecycle, or reads device states.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT/'campaigns/opt-cmo-corruption/launch')]
import exp014_context as experiment
from campaigns.recovery_collect import checked

CAPS = ('cap2', 'cap4', 'uncapped')
COLORS = {'cap2':'#3b78a1', 'cap4':'#df8c27', 'uncapped':'#27866b'}
SPLITS = {'256': ['development','transfer_generated','transfer_wikitext'],
          '1024': ['context1024_development','context1024_transfer_generated','context1024_transfer_wikitext']}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    with path.open('w', newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def review(evidence):
    output=evidence/'analysis';output.mkdir(exist_ok=True)
    hosts={h:json.loads((evidence/'hosts'/(h+'.json')).read_text()) for h in ('trex','loulou')}
    placements={(h,c['id']):c['placement'] for h,host in hosts.items() for c in host['cases']}
    raw_summary=json.loads((evidence/'reviews/summary.json').read_text())
    reports=raw_summary['screens']+raw_summary['confirmation']
    assert len(reports)==33 and len({r['label'] for r in reports})==33
    assert experiment.selection(raw_summary['screens'])==raw_summary['selected']==dict.fromkeys(CAPS,256)
    expected=set(experiment.labels('screen'))|{f'confirm-a{a}-l{l}-{cap}' for a in (271002,271003,271004) for l in (256,1024) for cap in CAPS}
    assert {r['label'] for r in reports}==expected
    data=[]; metrics=[]; hashes={}; held={}; parents=set()
    for report in reports:
        path=Path(report['collection']);receipt=json.loads(path.read_text());folder=path.parent
        for name,digest in receipt['files'].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,(report['label'],name)
        checked(json.loads((folder/'export.json').read_text()),placements[(receipt['target'],receipt['job'])]['recovery_input_contract'])
        rows=json.loads((folder/'measurements.json').read_text())
        assert experiment.checked_report(report['label'],rows,path)==report
        assert json.loads((evidence/'reviews'/(report['label']+'.json')).read_text())==report
        learn=next(r for r in rows if r['onchip_arm']=='onchip_weights')
        held.setdefault(report['array'],learn['initial_apparent_sha256'])
        assert held[report['array']]==learn['initial_apparent_sha256']
        records=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
        for rec in records:
            if rec['stage']=='onchip_cohort':
                v=rec['recovery_view'];parents.add((v['parent_indices_sha256'],v['parent_tokens_sha256'],v['parent_sequences'],v['score_stride']))
        writer=next(r['writer'] for r in reversed(records) if r['stage']=='onchip_development')
        memory=next(r for r in records if r['stage']=='onchip_memory')
        native_seconds=next(r['elapsed_seconds'] for r in records if r.get('terminal'))
        cost=report['cost'];attempts=cost['device_writes']+cost['stuck_write_attempts']
        assert cost.get('issued_write_attempts',attempts)==attempts and cost['verify_reads']==0
        row={k:report[k] for k in ('label','phase','array','length','cap')}
        row.update(native_seconds=native_seconds,training_seconds=learn['curve'][-1]['training_seconds'],
                   peak_reserved_gib=memory['peak_reserved_bytes']/1024**3,
                   attempts=attempts,successful_writes=cost['device_writes'],stuck_attempts=cost['stuck_write_attempts'],
                   touched_cells=cost['touched_devices'],max_successful_writes=cost['max_onchip_device_writes'],
                   max_attempts=cost.get('max_onchip_device_write_attempts'),capped_cells=cost.get('capped_devices',0),
                   pending_commands=writer['pending_nonzero'],pending_rms_q=writer['pending_rms_q'],
                   verify_reads=cost['verify_reads'],out_of_range_devices=cost['out_of_range_devices'])
        data.append(row)
        for eval_length,splits in SPLITS.items():
            for split in splits:
                initial=report['initial'][split];final=report['final'][split]
                metrics.append({**row,'evaluation_context':int(eval_length),'split':split.removeprefix('context1024_'),
                    'initial_kl':initial['teacher_kl'],'final_kl':final['teacher_kl'],
                    'kl_reduction_percent':100*(initial['teacher_kl']-final['teacher_kl'])/initial['teacher_kl'],
                    'initial_perplexity':initial['perplexity'],'final_perplexity':final['perplexity']})
        hashes[report['label']]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(parents)==1 and len(set(held.values()))==4
    confirmation=[]
    for cap in CAPS:
        for array in (271002,271003,271004):
            for evaluation in (256,1024):
                pair=[r for r in metrics if r['phase']=='confirm' and r['cap']==cap and r['array']==array
                      and r['evaluation_context']==evaluation and r['split']=='development']
                short=next(r for r in pair if r['length']==256);long=next(r for r in pair if r['length']==1024)
                assert short['initial_kl']==long['initial_kl']
                gs=short['initial_kl']-short['final_kl'];gl=long['initial_kl']-long['final_kl']
                confirmation.append({'cap':cap,'array':array,'evaluation_context':evaluation,
                    'initial_kl':short['initial_kl'],'kl_256':short['final_kl'],'kl_1024':long['final_kl'],
                    'difference_256_minus_1024':short['final_kl']-long['final_kl'],
                    'retained_gain_percent':100*gs/gl if gl>0 else None,
                    'reduction_256_percent':short['kl_reduction_percent'],
                    'passes':bool(gs>0 and gl>0 and gs>=.95*gl)})
    cap_comparison=[]
    for row in metrics:
        if row['cap']=='uncapped':continue
        ref=next(r for r in metrics if all(r[k]==row[k] for k in ['array','length','evaluation_context','split']) and r['cap']=='uncapped')
        gain=ref['initial_kl']-ref['final_kl']
        cap_comparison.append({k:row[k] for k in ['phase','array','length','cap','evaluation_context','split']}|
            {'gain_vs_uncapped_percent':100*(row['initial_kl']-row['final_kl'])/gain if gain>0 else None,
             'attempts_vs_uncapped_percent':100*row['attempts']/ref['attempts']})
    with sqlite3.connect('file:'+str(Path.home()/'.local/state/run-watch/queue/queue.sqlite3')+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        jobs=[dict(r) for r in db.execute("SELECT id,state,used_seconds,updated FROM jobs WHERE id LIKE '%exp014-context%'")]
    assert all(next(j for j in jobs if j['id']==experiment.PREFIX+'-'+r['label'])['state']=='complete' for r in reports)
    decision={'coverage_complete':True,'validated_fits':33,'native_endpoints':66,'selected_contexts':raw_summary['selected'],
              'confirmed':{cap:all(r['passes'] for r in confirmation if r['cap']==cap) for cap in CAPS},
              'charged_gpu_hours':sum(j['used_seconds'] for j in jobs)/3600,
              'finished_controller_epoch':max(j['updated'] for j in jobs if j['state']=='complete'),
              'selection_used_test_data':False,'held_state_hashes':held,'common_parent_receipt':list(parents)[0],
              'collection_receipt_sha256':hashes,'uncapped_maximum_attempts_recorded':False}
    write_csv(output/'runs.csv',data);write_csv(output/'metrics.csv',metrics)
    write_csv(output/'confirmation.csv',confirmation);write_csv(output/'cap-comparison.csv',cap_comparison)
    write_json(output/'decision.json',decision)
    plots(output,metrics,confirmation)
    return decision


def plots(output,metrics,confirmation):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/exp014-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    arrays=(271001,271002,271003,271004)
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(output/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    for evaluation in (256,1024):
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        for ax,array in zip(axes.flat,arrays):
            rows=[r for r in metrics if r['array']==array and r['evaluation_context']==evaluation and r['split']=='development']
            for cap in CAPS:
                group=sorted((r for r in rows if r['cap']==cap),key=lambda r:r['length'])
                ax.plot([r['length'] for r in group],[r['final_kl'] for r in group],'o-',color=COLORS[cap],label=cap)
            ax.axhline(rows[0]['initial_kl'],color='0.5',ls='--',lw=1,label='HWA deployment')
            ax.set_xscale('log',base=2);xs=sorted({r['length'] for r in rows});ax.set_xticks(xs,[str(x) for x in xs])
            ax.set(title=f"Array {array} — {'screen' if array==271001 else 'confirmation'}",xlabel='Recovery context (tokens)',ylabel='Development teacher KL')
            ax.grid(alpha=.15);ax.legend(fontsize=8)
        fig.suptitle(f'Fixed evaluation context {evaluation} · 3% permanent random faults · 256 updates')
        save(fig,f'context-kl-eval{evaluation}')
    for x,label in [('attempts','Issued attempts (millions)'),('touched_cells','Successfully touched active cells (millions)')]:
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        for ax,array in zip(axes.flat,arrays):
            rows=[r for r in metrics if r['array']==array and r['evaluation_context']==1024 and r['split']=='development']
            for cap in CAPS:
                group=sorted((r for r in rows if r['cap']==cap),key=lambda r:r['length'])
                ax.scatter([r[x]/1e6 for r in group],[r['final_kl'] for r in group],color=COLORS[cap],label=cap,s=35)
                for r in group:ax.annotate(str(r['length']),(r[x]/1e6,r['final_kl']),xytext=(3,4),textcoords='offset points',fontsize=7,color=COLORS[cap])
            ax.set(title=f'Array {array}',xlabel=label,ylabel='Development teacher KL at context 1,024');ax.legend(fontsize=8);ax.grid(alpha=.15)
        fig.suptitle('Recovery quality and write cost · labels show recovery context')
        save(fig,'kl-vs-'+x)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,evaluation in zip(axes,(256,1024)):
        for index,cap in enumerate(CAPS):
            rows=[r for r in confirmation if r['cap']==cap and r['evaluation_context']==evaluation]
            ax.plot([r['array']-271000 for r in rows],[r['retained_gain_percent'] for r in rows],'o-',color=COLORS[cap],label=cap)
        ax.axhline(95,color='0.4',ls='--',label='Frozen 95% threshold')
        ax.set_xticks([2,3,4],['271002','271003','271004']);ax.set(title=f'Evaluation context {evaluation}',xlabel='Independent array assignment',ylabel='Gain retained by context 256 (%)');ax.legend(fontsize=8);ax.grid(alpha=.15)
    save(fig,'confirmation-retention')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=ROOT/'results/lifecycles'/experiment.PREFIX)
    args=parser.parse_args();result=review(args.evidence)
    print(json.dumps({k:v for k,v in result.items() if k not in ['collection_receipt_sha256','held_state_hashes','common_parent_receipt']},indent=2))
