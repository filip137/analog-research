"""Collect declared screens and matched confirmations, including honest partials."""
import argparse,json,math
from dataclasses import asdict
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ebl.runtime.store import atomic_write_json,sha256_file
from ebl.schema.energy_transformer_mnist import parse_config
from experiments.review_energy_transformer_input_attention import equal,weight_ranks
from experiments.run_energy_transformer_explorer import VARIANTS,select_confirmations
from networks.energy_transformer.finite_time import select_cap
from networks.energy_transformer.mnist import unpatchify

GROUPS=('query','key','value','output')
def read(p):return json.loads(p.read_text())
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def history(run):return [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]

def gradient_record(run,cap,checkpoint):
    s=read(run/'artifacts/summary.json');manifest=read(run/'manifest.json')
    rows=[r for r in s['rows'] if r['iterations']==cap]
    records={}
    for metric in ('ep_vs_bptt','ep_vs_reference','bptt_vs_reference'):
        records[metric]=dict(global_min=min((r[metric]['global']['cosine'] or -1 for r in rows),default=-1),
            routing_min=min((r[metric]['groups'][g]['cosine'] or -1 for r in rows for g in GROUPS),default=-1))
    ref_valid=(len(s['references'])==8 and all(r['phases'][-1]['residual']<=1e-5 and
        all(v['cosine'] is not None and v['cosine']>=.999 and v['relative_l2']<=.01 for v in
            [r['agreement_512_vs_1024']['global'],*[r['agreement_512_vs_1024']['groups'][g] for g in GROUPS]])
        for r in s['references']))
    found=next(v for v in manifest['inputs'] if v['path']==str(checkpoint))
    valid=(s['coverage_complete'] and ref_valid and set(s['required_groups'])==set(GROUPS)
           and select_cap(s['rows'],(cap,),8,.9,GROUPS)==cap and sha256_file(checkpoint)==found['sha256'])
    return dict(valid=valid,cap=cap,metrics=records,checkpoint=dict(path=str(checkpoint),sha256=found['sha256']))

def train_check(run,updates):
    rows=[r for r in history(run) if r.get('stage')=='train']
    valid=(len(rows)==updates and [r['step'] for r in rows]==list(range(1,updates+1)) and all(
        r['recurrent_bound']<=.7 and r['contraction_radius']<1 and
        all(r[k]>0 for k in ('minimum_triode_margin','minimum_drain_margin','minimum_weak_inversion_margin')) for r in rows))
    return dict(valid=valid,count=len(rows),max_residual=max((r['residual'] for r in rows),default=None),
                max_bound=max((r['recurrent_bound'] for r in rows),default=None))

def collect(root):
    result=read(root/'result.json');launch=read(root/'launch.json');coverage=read(root/'coverage.json')
    out=root/'analysis';out.mkdir(exist_ok=True)
    baseline=Path(result['baseline']['run_dir']);baseline_root=Path(result['baseline']['root'])
    baseline_review=read(baseline_root/'analysis/review.json')
    controls={r['name']:r for r in baseline_review['comparison']}
    initial=load(baseline/'checkpoints/initial.pt');baseline_cohort=load(baseline/'artifacts/cohorts.pt')
    references={750:load(baseline/'checkpoints/step-0750.pt'),2000:load(baseline/'checkpoints/final.pt')}
    expected_data={v['path']:v['sha256'] for v in read(baseline/'manifest.json')['inputs'] if '/MNIST/raw/' in v['path']}
    checks={'selected_confirmations_accounted':{r['name'] for r in result['confirmations']}==set(result['selected']),
            'terminal_scientific_coverage':result['outcome']=='exploration_collected',
            'all_seven_cases_accounted':{s['name'] for s in result['screens']}==set(VARIANTS),
            'selection_matches_frozen_rule':result['selected']==select_confirmations(result['screens'],result['baseline']['probe_mse']),
            'native_inspections_pass':bool(coverage) and all(r['integrity_passed'] for r in coverage),
            'completed_within_allowance':result['seconds']<=launch['seconds_limit']+1}
    for row in coverage:
        if not row.get('run_dir'):checks[row['name']+'_native_present']=False;continue
        m=read(Path(row['run_dir'])/'manifest.json')
        checks[row['name']+'_source']=(m['source']['commit']==launch['source_commit'] and not m['source']['dirty'])
        checks[row['name']+'_data']=expected_data=={v['path']:v['sha256'] for v in m['inputs'] if '/MNIST/raw/' in v['path']}
    replay=Path(result['baseline']['replay']) if result['baseline'].get('replay') else None
    if replay:
        payload=load(replay/'checkpoints/final.pt')
        checks['baseline_replay_unchanged']=all(equal(payload[k],references[2000][k]) for k in ('model','optimizer','order_rng','mask_rng','step'))
        checks['baseline_replay_no_optimizer_steps']=not any(v.get('stage')=='train' for v in history(replay))
    else:checks['baseline_replay_unchanged']=False
    baseline_force={k:read(replay/'artifacts/summary.json')['evaluation']['ablation_normal'].get(k) for k in ('attention_force_rms','input_drive_rms','memory_force_rms','decoder_force_rms')} if replay else {}
    diagnostics={};training={};screens=[];full=[];rank_runs={}
    def audit(run,name,updates):
        payload=load(run/'checkpoints/final.pt');first=load(run/'checkpoints/initial.pt')
        checks[name+'_initial_equal']=equal(first['model'],initial['model'])
        checks[name+'_cohort_equal']=equal(load(run/'artifacts/cohorts.pt'),baseline_cohort)
        checks[name+'_rng_equal']=all(torch.equal(payload[k],references[updates][k]) for k in ('order_rng','mask_rng'))
        checks[name+'_finite_fp32']=all(v.dtype==torch.float32 and bool(v.isfinite().all()) for v in payload['model'].values())
        training[name]=train_check(run,updates);checks[name+'_all_updates_valid']=training[name]['valid']
        return payload
    for row in result['screens']:
        record=dict(row)
        if row['status']=='screen_complete':
            run=Path(row['screen']);s=read(run/'artifacts/summary.json');audit(run,row['name']+'_screen',750)
            checks[row['name']+'_screen_coverage']=(s['coverage_complete'] and s['updates']==750 and all(
                s['evaluation']['ablation_'+a]['complete'] and s['evaluation']['ablation_'+a]['count']==256 for a in ('normal','uniform','zero')))
            p=s['evaluation']['ablation_normal'];record['force_metrics']={k:p.get(k) for k in ('attention_force_rms','input_drive_rms','memory_force_rms','decoder_force_rms')}
            record.update(entropy=p['attention_normalized_entropy'],attention_mean_max=p['attention_mean_max'])
            checks[row['name']+'_screen_metric_identity']=record['probe_mse']==p['mse']['model']
            record['probe_mse']=p['mse']['model'];record['improvement_percent']=100*(1-record['probe_mse']/result['baseline']['probe_mse'])
            diagnostics[row['name']+'_selection']=gradient_record(Path(row['gradient_selection']),row['cap'],Path(row['warmup'])/'checkpoints/final.pt')
            checks[row['name']+'_gradient_selection']=diagnostics[row['name']+'_selection']['valid']
            rank_runs[row['name']+'_screen']=run
        screens.append(record)
    for row in result['confirmations']:
        name=row['name'];record=dict(row);paired={}
        for arm in ('energy_ep','energy_bp'):
            run=Path(row[arm]);s=read(run/'artifacts/summary.json');e=s['evaluation']
            if s['coverage_complete']:
                paired[arm]=audit(run,name+'_'+arm,2000);rank_runs[name+'_'+arm]=run
            normal=e.get('ablation_normal',{}).get('mse',{}).get('model')
            record[arm+'_metrics']=dict(test_mse=e.get('test',{}).get('mse',{}).get('model'),
                validation_mse=e.get('validation',{}).get('mse',{}).get('model'),probe_mse=normal,
                uniform_penalty_percent=100*(e['ablation_uniform']['mse']['model']/normal-1) if normal else None,
                zero_penalty_percent=100*(e['ablation_zero']['mse']['model']/normal-1) if normal else None,
                updates=s['updates'],training_seconds=s['train_seconds'],coverage_complete=s['coverage_complete'],
                force_metrics={k:e.get('ablation_normal',{}).get(k) for k in ('attention_force_rms','input_drive_rms','memory_force_rms','decoder_force_rms')})
            checks[name+'_'+arm+'_full_coverage']=(s['coverage_complete'] and s['updates']==2000 and
                e.get('test',{}).get('count')==10000 and e.get('validation',{}).get('count')==5000)
        if len(paired)==2:
            config=lambda p:{k:v for k,v in asdict(parse_config(p['config'])).items() if k!='arm'}
            checks[name+'_paired_config']=config(paired['energy_ep'])==config(paired['energy_bp'])
        if row.get('gradient_final'):
            diagnostics[name+'_final']=gradient_record(Path(row['gradient_final']),row['cap'],Path(row['energy_ep'])/'checkpoints/final.pt')
            checks[name+'_final_gradient']=diagnostics[name+'_final']['valid']
            selection=Path(next(s['gradient_selection'] for s in result['screens'] if s['name']==name))
            checks[name+'_same_gradient_cohort']=equal(load(selection/'artifacts/cohort.pt'),load(Path(row['gradient_final'])/'artifacts/cohort.pt'))
        else:checks[name+'_final_gradient']=False
        ep=record['energy_ep_metrics'];bp=record['energy_bp_metrics']
        if ep['test_mse'] is not None:
            record.update(improvement_over_singular_percent=100*(1-ep['test_mse']/controls['new_ep']['test_mse']),
                          improvement_over_old_percent=100*(1-ep['test_mse']/controls['old_ep']['test_mse']),
                          ep_bp_gap_percent=100*(ep['test_mse']/bp['test_mse']-1) if bp['test_mse'] else None)
        record['promotion_science']=(ep['test_mse'] is not None and ep['test_mse']<=.95*controls['old_ep']['test_mse'] and ep['uniform_penalty_percent'] is not None and ep['uniform_penalty_percent']>=1.)
        full.append(record)
    report=dict(outcome=result['outcome'],checks=checks,numerical_and_coverage_pass=all(checks.values()),screens=screens,
                confirmations=full,diagnostics=diagnostics,training_checks=training,controls=controls,baseline=result['baseline'],
                seconds=result['seconds'],source_commit=launch['source_commit'],baseline_force_metrics=baseline_force,
                limits='One seed; validation-selected exploratory comparisons of ideal FP32 circuit equations; no physical analog time/power or transistor-complete implementation.')
    for row in full:row['promote']=bool(all(checks.values()) and row['promotion_science'])
    atomic_write_json(out/'review.json',report);atomic_write_json(out/'weight-ranks.json',weight_ranks(rank_runs))
    valid=[r for r in screens if r['status']=='screen_complete']
    if valid:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        axes[0].barh([r['name'] for r in valid],[r['improvement_percent'] for r in valid]);axes[0].axvline(.5,color='black',ls=':',label='Confirmation threshold')
        axes[0].set_xlabel('750-update validation improvement (%)');axes[0].legend(fontsize=8)
        ratios=[r['force_metrics']['attention_force_rms']/max(r['force_metrics']['input_drive_rms'],1e-12) for r in valid]
        axes[1].barh([r['name'] for r in valid],ratios);axes[1].set_xlabel('Attention / input-drive RMS at750updates')
        if baseline_force.get('input_drive_rms'):
            axes[1].axvline(baseline_force['attention_force_rms']/baseline_force['input_drive_rms'],color='black',ls=':',label='Baseline at2000 (context only)');axes[1].legend(fontsize=8)
        fig.suptitle('Declared attention hypotheses — validation screens, not final test ranking')
        for suffix in ('png','svg'):fig.savefig(out/f'screens.{suffix}',dpi=160)
        plt.close(fig)
    if full:
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        names=['Singular baseline','Digital'];values=[controls['new_ep']['test_mse'],controls['digital']['test_mse']]
        for r in full:
            for arm in ('energy_ep','energy_bp'):
                v=r[arm+'_metrics']['test_mse']
                if v is not None:names.append(r['name']+' / '+arm);values.append(v)
        ax.barh(names,values);ax.set_xlabel('Full test MSE (lower is better)');fig.suptitle('Selected full comparisons — verify coverage in review.json')
        for suffix in ('png','svg'):fig.savefig(out/f'confirmations.{suffix}',dpi=160)
        plt.close(fig)
    lines=['# Attention Explorer result','',report['limits'],'','| Case | Status | Probe improvement | Uniform penalty |','| --- | --- | ---: | ---: |']
    for r in screens:lines.append(f"| {r['name']} | {r['status']} | {r.get('improvement_percent','—')} | {r.get('uniform_penalty_percent','—')} |")
    lines+=['',f"All numerical/coverage checks: {report['numerical_and_coverage_pass']}. Selected: {result['selected']}.",'', 'Full identities, matched controls, gradient gates and promotion verdicts are in review.json.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(outcome=result['outcome'],checks_pass=report['numerical_and_coverage_pass'],failed_checks=[k for k,v in checks.items() if not v],screens=screens,confirmations=full),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();collect(args.root)
