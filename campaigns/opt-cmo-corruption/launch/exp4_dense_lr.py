"""Bounded development-only LR retuning of the failed dense on-chip recipe."""
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
from exp3_lora import input_contract, thresholds, ARM, ARRAY, DEADLINE
from slice_campaign import write_frozen
from ebl.schema.lifecycle import parse_lifecycle

PREFIX = 'opt125m-cmo-l6-exp4-dense-lr'
PHASE = 'opt-cmo-exp4-dense-lr-20261012'
NOTE = 'campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-010-lr-followup.md'
RATES = {'3em4':.0003, '1em3':.001, '3em3':.003, '1em2':.01, '3em2':.03}
AUTHORIZATION = ('User: "if the loss worsens tune the lr". The dense baseline development KL worsened. '
    'Screen five declared rates from the identical deployment on 8192 sequences; select using only development KL. '
    'At most one fresh 50M-token fit if the screen improves development KL by at least 2%. '
    'Two GPU-hours from the unused dense allocation, 20 GB additional evidence, original 07:30 Paris cutoff.')


def labels():
    return [f'{tag}-{phase}' for phase in ('dev','50m') for tag in RATES]


def lifecycle(label):
    if label not in labels():
        raise ValueError('Unknown dense LR case')
    tag, phase = label.split('-')
    doc = json.loads((ROOT/'campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp2-dense-50m.json').read_text())
    doc['lifecycle_id'] = PREFIX+'-'+label
    doc['question'].update(text='Development-triggered LR tuning of dense same-array recovery.',
        decision='Select minimum final development teacher KL, ties lower LR; at least 2% gain over held P0 before one fresh full pass. '+NOTE)
    doc['onchip'].update(learning_rate=RATES[tag], training_cohort='recovery' if phase == 'dev' else 'hwa')
    return parse_lifecycle(doc)


def generate():
    paths = []
    for label in labels():
        lc = lifecycle(label)
        path = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
        write_frozen(path,lc.to_dict()); paths.append(str(path))
    return paths


def prepare_host(base, reused):
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
        budget, expected = (600,240) if label.endswith('-dev') else (3000,1800)
        receipt_root = base/'worker-receipts'/revision/lc.lifecycle_id
        worker = root/'tmux-worker.json'; native_root = root/'runs'/raw['campaign_id']
        write_frozen(worker,{'case_id':lc.lifecycle_id,'state_root':str(receipt_root),'cwd':str(ROOT),
            'timeout_seconds':budget-120,'lease_seconds':900,'env':env,
            'argv':[python,'-m','ebl','campaign','run','--manifest',str(manifest),'--output-dir',str(root/'runs'),'--resume','--fail-fast']})
        files = [config,manifest,worker,Path(stage['config']),inputs_path]
        cases.append({'id':lc.lifecycle_id,'label':label,'lifecycle_sha256':lc.digest(),'dry_run_passed':True,
            'budget_seconds':budget,'expected_seconds':expected,'placement':{
                'workdir':str(ROOT),'run_dir':str(receipt_root),'gpu_models':['5090'],
                'argv':[python,'-m','ebl.execution.tmux_worker','bridge','--spec',str(worker)],
                'validate_argv':[python,'-m','ebl.execution.recovery_collect','validate',str(native_root),'--inputs',str(inputs_path)],
                'campaign_python':python,'campaign_worker_spec':str(worker),'campaign_result_root':str(native_root),
                'recovery_inputs':str(inputs_path),'recovery_input_contract':inputs,
                'prepared_files':[{'path':str(p),'sha256':sha256(p)} for p in files]+list(inputs['artifacts'].values())}})
    return {'host':'loulou','source':str(ROOT),'source_revision':revision,'base':str(base),'cases':cases}


def queue(receipt,output,controller,evidence):
    prepared = json.loads(receipt.read_text())
    if prepared['host'] != 'loulou' or {c['label'] for c in prepared['cases']} != set(labels()):
        raise ValueError('Incomplete authorized placement')
    jobs, cases = [], []
    for c in prepared['cases']:
        label = c['label']; lc = lifecycle(label); dev = label.endswith('-dev')
        if c['lifecycle_sha256'] != lc.digest() or not c['dry_run_passed']:
            raise ValueError('Preparation differs from frozen science')
        job = {'id':lc.lifecycle_id,'project_dir':str(controller),'handoff':NOTE,'authorization':AUTHORIZATION,
            'peak_memory_mib':26000,'headroom_mib':2700,'expected_seconds':c['expected_seconds'],
            'budget_seconds':c['budget_seconds'],'deadline':DEADLINE,'priority':100 if dev else 90,
            'startup_grace_seconds':600,'stale_seconds':900,'recovery':False,'retry_safe':False,
            'placements':{'loulou':c['placement']},'completion':{
                'argv':[sys.executable,str(controller/'campaigns/opt-cmo-corruption/launch/exp4_dense_lr.py'),'collect',
                        '--queue-config',str(Path.home()/'.config/run-watch/queue.json'),'--evidence',str(evidence)],
                'workdir':str(controller),'timeout_seconds':600,'retry_safe':True}}
        cases.append({'id':lc.lifecycle_id,'stage':'readiness' if dev else 'training',
            'dependencies':[] if dev else [PREFIX+'-'+tag+'-dev' for tag in RATES],
            'required_gates':['preflight' if dev else 'selected-'+label],
            'expected_gpu_seconds':c['expected_seconds'],'expected_storage_bytes':1600000000,
            'progress':{**thresholds(c['expected_seconds']),'phases':{
                'onchip':thresholds(c['expected_seconds']),'validation':thresholds(180),'collection':thresholds(180)}}})
        path = output/'jobs'/(lc.lifecycle_id+'.json'); write_frozen(path,job); jobs.append(str(path))
    manifest = {'version':1,'id':PHASE,'poll_seconds':300,'required_adapters':['tmux'],
        'budgets':{'gpu_hours':2,'storage_bytes':20000000000,'window_seconds':18000},'cases':cases,
        'storage_roots':[{'target':'loulou','path':prepared['source']},
            {'target':'loulou','path':str(Path(prepared['base'])/'worker-receipts'/prepared['source_revision'])}]}
    from ebl.execution.service import validate_manifest
    validate_manifest(manifest); write_frozen(output/'campaign.json',manifest)
    return {'manifest':str(output/'campaign.json'),'jobs':jobs}


def select(reports):
    """Test endpoints are deliberately outside the LR decision."""
    if {r['learning_rate'] for r in reports} != set(RATES.values()) or len(reports) != len(RATES):
        raise ValueError('The declared development grid is incomplete')
    if any(r['sequence_presentations'] != 8192 or not math.isfinite(r['development_kl'])
           or r['development_kl'] <= 0 or not math.isfinite(r['initial_development_kl'])
           or r['initial_development_kl'] <= 0 for r in reports):
        raise ValueError('Invalid development-screen evidence')
    chosen = min(reports,key=lambda r:(r['development_kl'],r['learning_rate']))
    return {'learning_rate':chosen['learning_rate'],'label':chosen['label'].removesuffix('-dev')+'-50m',
            'development_kl':chosen['development_kl'],'initial_development_kl':chosen['initial_development_kl'],
            'triggered':chosen['development_kl'] <= .98*chosen['initial_development_kl'],
            'full_seconds_with_margin':chosen['full_seconds_with_margin'],
            'rule':'Minimum development KL, exact ties lower LR; at least 2% gain before one full fit.',
            'evidence':[r['collection'] for r in reports]}


def collect(queue_config,evidence):
    from ebl.execution.recovery_collect import collect as native_collect
    from ebl.execution.service import digest
    label = os.environ['RUN_WATCH_JOB_ID'].removeprefix(PREFIX+'-')
    if label not in labels():
        raise ValueError('Unexpected completion job')
    result = native_collect(queue_config,evidence/'collected')
    folder = Path(result['collected']); rows = json.loads((folder/'measurements.json').read_text())
    learned = next(r for r in rows if r['onchip_arm'] == 'onchip_weights')
    baseline = next(r for r in rows if r['onchip_arm'] == 'none')
    records = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
    memory = max(m['peak_reserved_bytes'] for m in records if m.get('stage') == 'onchip_memory')
    if memory > 26000*1024**2:
        raise ValueError('Measured CUDA memory exceeds reservation')
    dev = label.endswith('-dev')
    report = {'label':label,'learning_rate':RATES[label.split('-')[0]],'sequence_presentations':learned['sequence_presentations'],
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
            paths = [evidence/'reviews'/(tag+'-dev.json') for tag in RATES]
            if all(p.exists() for p in paths):
                decision = select([json.loads(p.read_text()) for p in paths])
                write_frozen(evidence/'readiness/selection.json',decision)
                result['selection'] = decision
                if decision['triggered']:
                    if decision['full_seconds_with_margin'] > 2820:
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
    a = sub.add_parser('prepare-host'); a.add_argument('--base',type=Path,default=DEFAULT_BASE); a.add_argument('--reused',type=Path,required=True)
    a = sub.add_parser('queue'); a.add_argument('receipt',type=Path); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--controller',type=Path,required=True); a.add_argument('--evidence',type=Path,required=True)
    a = sub.add_parser('collect'); a.add_argument('--queue-config',type=Path,required=True); a.add_argument('--evidence',type=Path,required=True)
    a = p.parse_args()
    if a.mode == 'generate': result = generate()
    elif a.mode == 'prepare-host': result = prepare_host(a.base,a.reused)
    elif a.mode == 'queue': result = queue(a.receipt,a.output,a.controller,a.evidence)
    else: result = collect(a.queue_config,a.evidence)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
