"""Read-only completion/cohort/checkpoint audit; writes only derived evidence."""
from pathlib import Path
import argparse,json,subprocess,sys
import torch
from ebl.runtime.store import atomic_write_json,sha256_file
from networks.energy_transformer.fuse_gpu import NAMES
from networks.energy_transformer.mnist import patchify

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    result=json.loads((root/'result.json').read_text())
    baseline=json.loads(Path('/home/filip/analog-research/drn/results/energy-transformer-repair/training-003/result.json').read_text())
    base_rows={r['case']:r for r in baseline['cases'] if r['arm']=='digital_bp'}
    checks=[];evidence=[];rows={}
    def check(name,value):checks.append({'name':name,'passed':bool(value)})
    check('all_six_cases',len(result['cases'])==6)
    for row in result['cases']:
        case,kernel=row['case'],row['kernel'];key=case+'-'+kernel;rows[key]=row
        bundle=Path(row['run_dir']);s=row['summary'];g=row.get('gradients',{})
        check(key+'-coverage',s['coverage_complete'] and s['updates']==2000)
        check(key+'-full_test',s['evaluation']['test']['count']==10000 and s['evaluation']['test']['complete'])
        check(key+'-full_validation',s['evaluation']['validation']['count']==5000 and s['evaluation']['validation']['complete'])
        check(key+'-three_probes',all(s['evaluation']['ablation_'+n]['count']==256 and s['evaluation']['ablation_'+n]['complete'] for n in ('normal','linear','zero')))
        check(key+'-gradient',g.get('passed') and g.get('count')==8)
        ck=bundle/'checkpoints/final.pt'
        check(key+'-checkpoint_unchanged',sha256_file(ck)==g.get('checkpoint_sha256'))
        command=[sys.executable,'-m','ebl','runs','inspect',str(bundle),'--verify-artifacts','--require-complete']
        cp=subprocess.run(command,capture_output=True,text=True)
        check(key+'-native_artifacts',cp.returncode==0)
        cohorts=torch.load(bundle/'artifacts/cohorts.pt',map_location='cpu',weights_only=False)
        prior=Path(base_rows[case]['run_dir'])
        bc=torch.load(prior/'artifacts/cohorts.pt',map_location='cpu',weights_only=False)
        manifest=json.loads((bundle/'manifest.json').read_text());bm=json.loads((prior/'manifest.json').read_text())
        check(key+'-same_raw_inputs',manifest['inputs']==bm['inputs'])
        check(key+'-pinned_clean_source',manifest['source']['commit'].startswith('763e486') and not manifest['source']['dirty'])
        check(key+'-same_split',all(torch.equal(cohorts['ids'][n],bc['ids'][n]) for n in ('train','validation','test')))
        check(key+'-same_masks',all(torch.equal(cohorts[n],bc[n]) for n in ('validation_masks','test_masks')))
        check(key+'-unpermuted_reference',cohorts['permutation'] is None)
        m=torch.load(ck,map_location='cpu',weights_only=False)
        bi=torch.load(prior/'checkpoints/final.pt',map_location='cpu',weights_only=False)
        check(key+'-same_sampler_rng',torch.equal(m['order_rng'],bi['order_rng']) and torch.equal(m['mask_rng'],bi['mask_rng']))
        env=json.loads((bundle/'artifacts/environment.json').read_text())
        check(key+'-fp32',env['precision']=='fp32' and not env['amp'] and not env['tf32'] and all(v.dtype==torch.float32 for v in m['model'].values()))
        metrics=[json.loads(line) for line in (bundle/'metrics.jsonl').read_text().splitlines()]
        train=[r for r in metrics if r['stage']=='train']
        check(key+'-all_steps_and_residuals',len(train)==2000 and [r['step'] for r in train]==list(range(1,2001)) and max(r['residual'] for r in train)<=1e-6)
        params={}
        for name in NAMES:
            pos,neg=m['model'][name+'_p'],m['model'][name+'_n'];w=pos-neg
            params[name]={'signed_norm':w.norm().item(),'signed_max':w.abs().max().item(),
                          'exact_zero_fraction':(w==0).float().mean().item(),
                          'floor_fraction':((pos==.02).sum()+(neg==.02).sum()).item()/(pos.numel()+neg.numel())}
        evidence.append({'case':case,'kernel':kernel,'run_dir':str(bundle),'counts':{k:s[k] for k in ('trainable_conductances','physical_resistors','physical_fuses')},
                         'parameters':params,'worst_training_residual':max(r['residual'] for r in train),
                         'maximum_phase_sweeps':max(max(r['phase_iterations']) for r in train)})
    for case in ('small','reference','more_tokens'):
        a=Path(rows[case+'-gaussian']['run_dir'])/'checkpoints/initial.pt'
        b=Path(rows[case+'-linear']['run_dir'])/'checkpoints/initial.pt'
        x=torch.load(a,map_location='cpu',weights_only=False)['model'];y=torch.load(b,map_location='cpu',weights_only=False)['model']
        check(case+'-matched_fuse_linear_initialization',all(torch.equal(x[k],y[k]) for k in x))
    check('aggregate_budget',result['total_charged_seconds']<=5400)
    sizes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    check('storage_budget',sizes<3_000_000_000)
    geometry={}
    for patch in (7,4):
        hidden=patchify(cohorts['test_masks'],patch).sum(-1);mixed=(hidden>0)&(hidden<patch*patch)
        geometry[str(patch)]={'fraction_hidden_pixels_in_mixed_tokens':(hidden*mixed).sum().item()/hidden.sum().item(),
                              'fraction_tokens_fully_hidden':(hidden==patch*patch).float().mean().item()}
    report={'mask_geometry':geometry,'passed':all(c['passed'] for c in checks),'checks':checks,'evidence':evidence,'storage_bytes':sizes,
            'total_charged_seconds':result['total_charged_seconds']}
    atomic_write_json(root/'analysis/parent-verification.json',report)
    print(json.dumps({'passed':report['passed'],'failed':[c for c in checks if not c['passed']],'evidence':evidence}),flush=True)
    return 0 if report['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
