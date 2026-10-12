"""Frozen noise-threshold screens and a conditional same-array full pass."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1e_prepare_host import command, sha256, DEFAULT_BASE
from exp1f_cdt_lr import service, transition
from exp3_lora import input_contract, thresholds, ARM, ARRAY
from slice_campaign import write_frozen
from workflow.lifecycle import parse_lifecycle

PREFIX = 'opt125m-cmo-l6-exp5-threshold'
PHASE = 'opt-cmo-exp5-threshold-20261012'
NOTE = 'campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-011-thresholded-writes.md'
SETTINGS = {'discard-s1': (False, 1.), 'accum-s1': (True, 1.),
            'accum-s2': (True, 2.), 'accum-s3': (True, 3.)}
DEADLINE = '2026-10-12T17:30:00+02:00'
AUTHORIZATION = ('User proposed applying only updates larger than write noise and instructed that the free '
    '5090s should be used. Exp-011 freezes four matched threshold screens and at most one qualifying '
    'fresh 50M-token pass, within two GPU-hours, 20 GB and the 17:30 Paris cutoff.')


def labels():
    return [f'{tag}-{phase}' for phase in ('dev', '50m') for tag in SETTINGS]


def lifecycle(label):
    if label not in labels():
        raise ValueError('Unknown threshold case')
    tag, phase = label.rsplit('-', 1)
    accumulate, sigma = SETTINGS[tag]
    doc = json.loads((ROOT/'campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp2-dense-50m.json').read_text())
    doc['lifecycle_id'] = PREFIX+'-'+label
    doc['question'].update(text='Can noise-referenced sparse physical writes recover the held faulty array?',
        decision='Select minimum final development KL, ties fewer writes then label; require 2% gain before one full pass. '+NOTE)
    doc['onchip'].update(update_law='cmo_thresholded_write_adam', learning_rate=.003,
        threshold_sigma=sigma, accumulate_updates=accumulate,
        development_every_steps=64 if phase == 'dev' else 256,
        training_cohort='recovery' if phase == 'dev' else 'hwa')
    return parse_lifecycle(doc)


def generate():
    paths = []
    for label in labels():
        lc = lifecycle(label)
        path = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
        write_frozen(path,lc.to_dict()); paths.append(str(path))
    return paths


def prepare_host(host, base, reused):
    if host not in ('loulou', 'trex'):
        raise ValueError('Unprepared GPU host')
    if command(['git','status','--porcelain'],cwd=ROOT).strip():
        raise ValueError('Expected clean frozen worker')
    revision = command(['git','rev-parse','HEAD'],cwd=ROOT).strip()
    python = sys.executable
    (base/'worker-receipts'/revision).mkdir(parents=True,exist_ok=True)
    inputs = input_contract(base,reused)
    inputs_path = ROOT/'results/recovery-inputs.json'; write_frozen(inputs_path,inputs)
    env = {'EBL_CORPUS_ROOT':str(base/'inputs/corpora'),'EBL_CAMPAIGN_STREAM':'1',
           'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1'}
    cases = []
    for label in labels():
        lc = lifecycle(label)
        config = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
        if parse_lifecycle(json.loads(config.read_text())).digest() != lc.digest():
            raise ValueError('Scientific lifecycle changed')
        root = ROOT/'results/lifecycles'/lc.lifecycle_id
        command([python,'-m','workflow','plan',str(config),'--teacher-weights',inputs['artifacts']['teacher_weights']['path'],
                 '--output',str(root/'plan'),'--device','cuda:0','--cpu-threads','1','--python',python],cwd=ROOT)
        plan = root/'plan/campaign.json'; raw = json.loads(plan.read_text())
        stage = next(s for s in raw['stages'] if s['id'] == f'onchip__{ARM}__{ARRAY}')
        stage['config'] = str((plan.parent/stage['config']).resolve()); stage['depends_on'] = []
        stage['inputs'] = {k:{'path':v['path']} for k,v in inputs['artifacts'].items()}
        raw['stages'] = [stage]
        manifest = root/'campaign.json'; write_frozen(manifest,raw)
        command([python,'-m','ebl','campaign','run','--manifest',str(manifest),'--output-dir',str(root/'dry-run'),
                 '--dry-run','--fail-fast'],cwd=ROOT,env=dict(os.environ,**env),logs=root/'preparation/dry-run')
        budget, expected = (900,300) if label.endswith('-dev') else (4500,2400)
        receipt_root = base/'worker-receipts'/revision/lc.lifecycle_id
        worker = root/'tmux-worker.json'; native_root = root/'runs'/raw['campaign_id']
        write_frozen(worker,{'case_id':lc.lifecycle_id,'state_root':str(receipt_root),'cwd':str(ROOT),
            'timeout_seconds':budget-120,'lease_seconds':900,'env':env,
            'argv':[python,'-m','ebl','campaign','run','--manifest',str(manifest),'--output-dir',str(root/'runs'),'--resume','--fail-fast']})
        files = [config,manifest,worker,Path(stage['config']),inputs_path]
        cases.append({'id':lc.lifecycle_id,'label':label,'lifecycle_sha256':lc.digest(),'dry_run_passed':True,
            'budget_seconds':budget,'expected_seconds':expected,'placement':{
                'workdir':str(ROOT),'run_dir':str(receipt_root),'gpu_models':['5090'],
                'argv':[python,'-m','campaigns.tmux_worker','bridge','--spec',str(worker)],
                'validate_argv':[python,'-m','campaigns.recovery_collect','validate',str(native_root),'--inputs',str(inputs_path)],
                'campaign_python':python,'campaign_worker_spec':str(worker),'campaign_result_root':str(native_root),
                'recovery_inputs':str(inputs_path),'recovery_input_contract':inputs,
                'prepared_files':[{'path':str(p),'sha256':sha256(p)} for p in files]+list(inputs['artifacts'].values())}})
    return {'host':host,'source':str(ROOT),'source_revision':revision,'base':str(base),'cases':cases}


def queue(receipts,output,controller,evidence):
    hosts = {json.loads(p.read_text())['host']:json.loads(p.read_text()) for p in receipts}
    if set(hosts) != {'loulou','trex'} or len({h['source_revision'] for h in hosts.values()}) != 1:
        raise ValueError('Both hosts must use one prepared source revision')
    jobs, cases = [], []
    for label in labels():
        lc = lifecycle(label); dev = label.endswith('-dev')
        prepared = {host:next(c for c in h['cases'] if c['label']==label) for host,h in hosts.items()}
        if any(c['lifecycle_sha256'] != lc.digest() or not c['dry_run_passed'] for c in prepared.values()):
            raise ValueError('Preparation differs from frozen science')
        c = next(iter(prepared.values()))
        job = {'id':lc.lifecycle_id,'project_dir':str(controller),'handoff':NOTE,'authorization':AUTHORIZATION,
            'peak_memory_mib':26000,'headroom_mib':2700,'expected_seconds':c['expected_seconds'],
            'budget_seconds':c['budget_seconds'],'deadline':DEADLINE,'priority':110 if dev else 90,
            'startup_grace_seconds':600,'stale_seconds':900,'recovery':False,'retry_safe':False,
            'placements':{h:p['placement'] for h,p in prepared.items()},'completion':{
                'argv':[sys.executable,str(controller/'campaigns/opt-cmo-corruption/launch/exp5_threshold.py'),'collect',
                        '--queue-config',str(Path.home()/'.config/run-watch/queue.json'),'--evidence',str(evidence)],
                'workdir':str(controller),'timeout_seconds':600,'retry_safe':True}}
        cases.append({'id':lc.lifecycle_id,'stage':'readiness' if dev else 'training',
            'dependencies':[] if dev else [PREFIX+'-'+tag+'-dev' for tag in SETTINGS],
            'required_gates':['preflight' if dev else 'selected-'+label],
            'expected_gpu_seconds':c['expected_seconds'],'expected_storage_bytes':1800000000,
            'progress':{**thresholds(c['expected_seconds']),'phases':{
                'onchip':thresholds(c['expected_seconds']),'validation':thresholds(180),'collection':thresholds(180)}}})
        path = output/'jobs'/(lc.lifecycle_id+'.json'); write_frozen(path,job); jobs.append(str(path))
    manifest = {'version':1,'id':PHASE,'poll_seconds':300,'required_adapters':['tmux'],
        'budgets':{'gpu_hours':2,'storage_bytes':20000000000,'window_seconds':14400},'cases':cases,
        'storage_roots':[{'target':h,'path':p['source']} for h,p in hosts.items()] +
            [{'target':h,'path':str(Path(p['base'])/'worker-receipts'/p['source_revision'])} for h,p in hosts.items()] +
            [{'target':'trex','path':str(Path(hosts['trex']['base'])/'inputs/exp5-reused')},
             {'target':'local','path':str(evidence)}]}
    from campaigns.execution_service import validate_manifest
    validate_manifest(manifest); write_frozen(output/'campaign.json',manifest)
    return {'manifest':str(output/'campaign.json'),'jobs':jobs}


def select(reports):
    if {r['label'] for r in reports} != {tag+'-dev' for tag in SETTINGS} or len(reports) != len(SETTINGS):
        raise ValueError('The declared threshold grid is incomplete')
    if any(r['sequence_presentations'] != 8192 or not math.isfinite(r['development_kl'])
           or r['development_kl'] <= 0 or not math.isfinite(r['initial_development_kl'])
           or r['initial_development_kl'] <= 0 for r in reports):
        raise ValueError('Invalid development-screen evidence')
    chosen = min(reports,key=lambda r:(r['development_kl'],r['cost']['device_writes'],r['label']))
    return {'label':chosen['label'].removesuffix('-dev')+'-50m',
            'threshold_sigma':chosen['threshold_sigma'],'accumulate_updates':chosen['accumulate_updates'],
            'learning_rate':.003,'development_kl':chosen['development_kl'],
            'initial_development_kl':chosen['initial_development_kl'],
            'triggered':chosen['development_kl'] <= .98*chosen['initial_development_kl'],
            'full_seconds_with_margin':chosen['full_seconds_with_margin'],
            'rule':'Minimum final development KL, ties fewer device writes then label; require 2% gain.',
            'evidence':[r['collection'] for r in reports]}


def collect(queue_config,evidence):
    from campaigns.recovery_collect import collect as native_collect
    from campaigns.execution_service import digest
    label = os.environ['RUN_WATCH_JOB_ID'].removeprefix(PREFIX+'-')
    if label not in labels():
        raise ValueError('Unexpected threshold completion job')
    result = native_collect(queue_config,evidence/'collected')
    folder = Path(result['collected']); rows = json.loads((folder/'measurements.json').read_text())
    learned = next(r for r in rows if r['onchip_arm'] == 'onchip_weights')
    baseline = next(r for r in rows if r['onchip_arm'] == 'none')
    records = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
    memory = max(m['peak_reserved_bytes'] for m in records if m.get('stage') == 'onchip_memory')
    if memory > 26000*1024**2:
        raise ValueError('Measured CUDA memory exceeds reservation')
    dev = label.endswith('-dev')
    report = {'label':label,'learning_rate':.003,
        'threshold_sigma':lifecycle(label).onchip.threshold_sigma,
        'accumulate_updates':lifecycle(label).onchip.accumulate_updates,
        'sequence_presentations':learned['sequence_presentations'],
        'initial_development_kl':baseline['final']['development']['teacher_kl'],
        'development_kl':learned['final']['development']['teacher_kl'],
        'cost':learned['cost'],'peak_reserved_bytes':memory,'collection':str(folder/'collection.json'),
        'full_seconds_with_margin':math.ceil(learned['curve'][0]['epoch_seconds']/256*6101*1.35+300) if dev else None,
        'effects':{s:{'deployment':baseline['final'][s],'onchip':learned['final'][s],
            'relative_kl_reduction':1-learned['final'][s]['teacher_kl']/baseline['final'][s]['teacher_kl']}
            for s in ('development','transfer_generated','transfer_wikitext')}}
    with (evidence/'.completion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        write_frozen(evidence/'reviews'/(label+'.json'),report)
        if dev:
            paths = [evidence/'reviews'/(tag+'-dev.json') for tag in SETTINGS]
            if all(p.exists() for p in paths):
                decision = select([json.loads(p.read_text()) for p in paths])
                write_frozen(evidence/'readiness/selection.json',decision)
                result['selection'] = decision
                if decision['triggered']:
                    if decision['full_seconds_with_margin'] > 4200:
                        raise ValueError('Measured full fit exceeds unchanged worker allowance')
                    gate = {**decision,'passed':True,'manifest_sha256':digest(json.loads((evidence/'prepared/queue/campaign.json').read_text())),
                        'phase_progress_calibration':{'onchip':thresholds(decision['full_seconds_with_margin']),
                                                      'collection':thresholds(180),'validation':thresholds(180)}}
                    path = evidence/'readiness'/('selected-'+decision['label']+'.json'); write_frozen(path,gate)
                    service(queue_config,'gate',PHASE,'selected-'+decision['label'],path)
                    transition(queue_config,'release',PREFIX+'-'+decision['label'])
        result['review'] = report
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='mode',required=True)
    sub.add_parser('generate')
    a = sub.add_parser('prepare-host'); a.add_argument('--host',required=True); a.add_argument('--base',type=Path,default=DEFAULT_BASE); a.add_argument('--reused',type=Path,required=True)
    a = sub.add_parser('queue'); a.add_argument('receipts',nargs='+',type=Path); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--controller',type=Path,required=True); a.add_argument('--evidence',type=Path,required=True)
    a = sub.add_parser('collect'); a.add_argument('--queue-config',type=Path,required=True); a.add_argument('--evidence',type=Path,required=True)
    a = p.parse_args()
    if a.mode == 'generate': result = generate()
    elif a.mode == 'prepare-host': result = prepare_host(a.host,a.base,a.reused)
    elif a.mode == 'queue': result = queue(a.receipts,a.output,a.controller,a.evidence)
    else: result = collect(a.queue_config,a.evidence)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
