"""Frozen rank/LR development and one-pass LoRA comparison on held CMO arrays."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
from exp1e_prepare_host import command, checked_input, sha256, DEFAULT_BASE
from exp1f_cdt_lr import service, transition
from slice_campaign import write_frozen
from workflow.lifecycle import parse_lifecycle, verify_provenance
from workflow.networks.opt_mlp import PINNED
from experiments.artifacts import atomic_write_json

PHASE='opt-cmo-exp3-lora-20261012'
PREFIX='opt125m-cmo-l6-exp3-lora'
NOTE='campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-009-digital-lora.md'
DEADLINE='2026-10-12T07:30:00+02:00'
RANKS=(4,16,64)
RATES={'3em4':.0003,'1em3':.001,'3em3':.003}
ARM='standard_hwa_model'
ARRAY=271001
AUTHORIZATION='User requested one synthetic-corpus epoch of LoRA at multiple ranks in parallel with dense on-chip recovery, explicitly allowing the 3090s; confirmed only the same faulty MLPs. Exp-009 freezes ranks 4/16/64, separate development-only LR selection, exact held P0, eight GPU-hours, 40 GB and 07:30 Paris deadline.'


def labels():
    return [f'r{r}-dev' for r in RANKS]+[f'r{r}-lr{rate}-50m' for r in RANKS for rate in RATES]


def lifecycle(label):
    if label not in labels(): raise ValueError('Unknown frozen LoRA case')
    rank=int(label.split('-')[0][1:]); development=label.endswith('-dev')
    doc=json.loads((ROOT/'campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp2-dense-50m.json').read_text())
    doc['lifecycle_id']=PREFIX+'-'+label
    doc['question'].update(text='Digital LoRA recovery of the fixed exp-008 deployment at rank '+str(rank)+'.',
        decision='Develop LR on 8192 training sequences and minimum final development teacher KL (tie: smaller LR), then restart for exactly one full synthetic pass. Fixed final endpoint. '+NOTE)
    onchip=doc['onchip']; onchip.update(learning_rate=0,training_cohort='recovery' if development else 'hwa')
    rates=RATES if development else {label.split('-')[1][2:]:RATES[label.split('-')[1][2:]]}
    onchip['arms']=[{'id':'none','weights':'hold','calibration':False}]+[
        {'id':'lora_lr'+tag,'weights':'hold','calibration':False,
         'digital_adapter':{'kind':'lora','rank':rank,'alpha':rank,'learning_rate':rate}} for tag,rate in rates.items()]
    return parse_lifecycle(doc)


def generate():
    result=[]
    for label in labels():
        lc=lifecycle(label); p=ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
        write_frozen(p,lc.to_dict()); result.append(str(p))
    return result


def input_contract(base, reused):
    paths={'teacher_weights':base/'inputs/opt-125m-27dcfa74/pytorch_model.bin',
           'device_data':reused/'prepare/checkpoints/cache.pt','device_model':reused/'devices/checkpoints/devices.pt',
           'device_state':reused/'portable.pt'}
    expected={'teacher_weights':PINNED['facebook/opt-125m']['sha256'],
              'device_data':'ec0d875a5027ccc2c53b7fb50655c2c5c39670441da09127de986185b813eebf',
              'device_model':'40f624d3f0ce631b2e3d25f97910212147f317d76cd8e82663f918b2318f8f78'}
    portable=json.loads((reused/'portable.json').read_text())
    if portable['parent_sha256']!='f80a9447469697cf9e4ff657cb17545d2824e60ae4f12d4694f01f3e4ebaf873' or not portable['physical_state_preserved'] or portable['new_device_writes']:
        raise ValueError('Wrong physical deployment conversion')
    expected['device_state']=portable['sha256']
    artifacts={role:checked_input(path,expected[role]) for role,path in paths.items()}
    from experiments.cifar_crossbar.runtime import load
    p0=load(paths['device_state'])
    if p0['materialized_from_sha256']!=portable['parent_sha256'] or p0['source_sha256']!='86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028':
        raise ValueError('Wrong frozen source or P0 parent')
    lc=lifecycle('r64-dev')
    for role,kind,arm in [('device_data','feature_cache',None),('device_model','device_bundle',None),('device_state','deployment',ARM)]:
        bundle=p0 if role=='device_state' else load(paths[role])
        verify_provenance(bundle['provenance'],lc,kind,hwa_arm=arm)
    return {'artifacts':artifacts,'array_seed':ARRAY,'parent_p0_sha256':portable['parent_sha256'],
            'apparent_sha256':portable['apparent_sha256']}


def prepare_host(host,base,reused):
    if host not in ('local','nom-cool-1'): raise ValueError('Unauthorized LoRA host')
    if command(['git','status','--porcelain'],cwd=ROOT).strip(): raise ValueError('Expected clean frozen worker')
    revision=command(['git','rev-parse','HEAD'],cwd=ROOT).strip(); python=sys.executable
    inputs=input_contract(base,reused); inputs_path=ROOT/'results/recovery-inputs.json'; write_frozen(inputs_path,inputs)
    env={'EBL_CORPUS_ROOT':str(base/'inputs/corpora'),'EBL_CAMPAIGN_STREAM':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1'}
    cases=[]
    for label in labels():
        lc=lifecycle(label); cfg=ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
        root=ROOT/'results/lifecycles'/lc.lifecycle_id
        command([python,'-m','workflow','plan',str(cfg),'--teacher-weights',inputs['artifacts']['teacher_weights']['path'],
                 '--output',str(root/'plan'),'--device','cuda:0','--cpu-threads','1','--python',python],cwd=ROOT)
        plan=root/'plan/campaign.json'; raw=json.loads(plan.read_text())
        stage=next(s for s in raw['stages'] if s['id']==f'onchip__{ARM}__{ARRAY}')
        stage['config']=str((plan.parent/stage['config']).resolve()); stage['depends_on']=[]
        stage['inputs']={k:{'path':v['path']} for k,v in inputs['artifacts'].items()}
        raw['stages']=[stage]; manifest=root/'campaign.json'; write_frozen(manifest,raw)
        command([python,'-m','ebl','campaign','run','--manifest',str(manifest),'--output-dir',str(root/'dry-run'),
                 '--dry-run','--fail-fast'],cwd=ROOT,env=dict(os.environ,**env),logs=root/'preparation/dry-run')
        dev=label.endswith('-dev'); budget,expected=(1800,1200) if dev else (7200,5400)
        receipt_root=base/'worker-receipts'/revision/lc.lifecycle_id; worker=root/'tmux-worker.json'
        native_root=root/'runs'/raw['campaign_id']
        write_frozen(worker,{'case_id':lc.lifecycle_id,'state_root':str(receipt_root),'cwd':str(ROOT),
            'timeout_seconds':budget-180,'lease_seconds':900,'env':env,
            'argv':[python,'-m','ebl','campaign','run','--manifest',str(manifest),'--output-dir',str(root/'runs'),'--resume','--fail-fast']})
        files=[cfg,manifest,worker,Path(stage['config']),inputs_path]
        cases.append({'id':lc.lifecycle_id,'label':label,'lifecycle_sha256':lc.digest(),'dry_run_passed':True,
            'budget_seconds':budget,'expected_seconds':expected,'placement':{'workdir':str(ROOT),'run_dir':str(receipt_root),
            'gpu_models':['3090'],'argv':[python,'-m','campaigns.tmux_worker','bridge','--spec',str(worker)],
            'validate_argv':[python,'-m','campaigns.recovery_collect','validate',str(native_root),'--inputs',str(inputs_path)],
            'campaign_python':python,'campaign_worker_spec':str(worker),'campaign_result_root':str(native_root),
            'recovery_inputs':str(inputs_path),'recovery_input_contract':inputs,
            'prepared_files':[{'path':str(p),'sha256':sha256(p)} for p in files]+list(inputs['artifacts'].values())}})
    return {'host':host,'source':str(ROOT),'source_revision':revision,'base':str(base),'cases':cases}


def thresholds(seconds):
    return {'expected_seconds':seconds,'startup_seconds':600,'stale_seconds':900}


def queue(receipts,output,controller,evidence):
    hosts={json.loads(p.read_text())['host']:json.loads(p.read_text()) for p in receipts}
    if set(hosts)!={'local','nom-cool-1'}: raise ValueError('Both 3090 hosts must be prepared')
    jobs=[]; cases=[]
    for label in labels():
        lc=lifecycle(label); rank=int(label.split('-')[0][1:]); dev=label.endswith('-dev')
        prepared={host:next(c for c in h['cases'] if c['label']==label) for host,h in hosts.items()}
        if any(c['lifecycle_sha256']!=lc.digest() or not c['dry_run_passed'] for c in prepared.values()): raise ValueError('Unprepared lifecycle')
        c=next(iter(prepared.values()))
        job={'id':lc.lifecycle_id,'project_dir':str(controller),'handoff':NOTE,'authorization':AUTHORIZATION,
            'peak_memory_mib':20000,'headroom_mib':2000,'expected_seconds':c['expected_seconds'],'budget_seconds':c['budget_seconds'],
            'deadline':DEADLINE,'priority':110+rank if dev else 90,'startup_grace_seconds':600,'stale_seconds':900,
            'recovery':False,'retry_safe':False,'placements':{h:v['placement'] for h,v in prepared.items()},
            'completion':{'argv':[sys.executable,str(controller/'campaigns/opt-cmo-corruption/launch/exp3_lora.py'),'collect',
                    '--queue-config',str(Path.home()/'.config/run-watch/queue.json'),'--evidence',str(evidence)],
                'workdir':str(controller),'timeout_seconds':600,'retry_safe':True}}
        gate='preflight' if dev else 'selected-'+label
        cases.append({'id':lc.lifecycle_id,'stage':'readiness' if dev else 'training',
            'dependencies':[] if dev else [PREFIX+f'-r{rank}-dev'],'required_gates':[gate],
            'expected_gpu_seconds':c['expected_seconds'],'expected_storage_bytes':2300000000 if dev else 1200000000,
            'progress':{**thresholds(c['expected_seconds']),'phases':{p:thresholds(t) for p,t in
                [('onchip',c['expected_seconds']),('collection',300),('validation',300)]}}})
        p=output/'jobs'/(lc.lifecycle_id+'.json');write_frozen(p,job);jobs.append(str(p))
    manifest={'version':1,'id':PHASE,'poll_seconds':300,'required_adapters':['tmux'],
        'budgets':{'gpu_hours':7.9,'storage_bytes':38000000000,'window_seconds':24000},'cases':cases,
        'storage_roots':[{'target':h,'path':data['source']} for h,data in hosts.items()] +
            [{'target':h,'path':str(Path(data['base'])/'worker-receipts'/data['source_revision'])} for h,data in hosts.items()]}
    from campaigns.execution_service import validate_manifest
    validate_manifest(manifest);write_frozen(output/'campaign.json',manifest)
    return {'jobs':jobs,'manifest':str(output/'campaign.json')}


def gate(queue_config,evidence,name,payload):
    from campaigns.execution_service import digest
    manifest=json.loads((evidence/'prepared/queue/campaign.json').read_text())
    path=evidence/'readiness'/(name+'.json')
    write_frozen(path,{'manifest_sha256':digest(manifest),'passed':True,**payload})
    service(queue_config,'gate',PHASE,name,path)


def collect(queue_config,evidence):
    from campaigns.recovery_collect import collect as native_collect
    label=os.environ['RUN_WATCH_JOB_ID'].removeprefix(PREFIX+'-')
    if label not in labels(): raise ValueError('Unexpected completion job')
    started=time.monotonic(); result=native_collect(queue_config,evidence/'collected'); elapsed=time.monotonic()-started
    folder=Path(result['collected']); rows=json.loads((folder/'measurements.json').read_text())
    with (evidence/'.completion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if label.endswith('-dev'):
            rank=int(label.split('-')[0][1:]); candidates=[r for r in rows if 'digital_adapter' in r]
            if len(candidates)!=3 or any(r['sequence_presentations']!=8192 for r in candidates): raise ValueError('Incomplete LR screen')
            selected=min(candidates,key=lambda r:(r['final']['development']['teacher_kl'],r['digital_adapter']['learning_rate']))
            rate=next(tag for tag,value in RATES.items() if value==selected['digital_adapter']['learning_rate'])
            target=f'r{rank}-lr{rate}-50m'
            records=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
            memory=max(m['peak_reserved_bytes'] for m in records if m.get('stage')=='onchip_memory')
            if memory>20000*1024**2: raise ValueError('Measured memory exceeds reservation')
            seconds_per_update=max(r['curve'][0]['epoch_seconds']/256 for r in candidates)
            projection=math.ceil(seconds_per_update*6101*1.35+300)
            if projection>6900 or projection+300>datetime.fromisoformat(DEADLINE).timestamp()-time.time():
                raise ValueError('Measured full pass cannot fit declared budget/deadline')
            receipt={'rank':rank,'selected_learning_rate':RATES[rate],'selected_job':PREFIX+'-'+target,
                'rule':'minimum final development teacher KL; exact ties choose lower LR; fresh full-pass initialization',
                'screen':[{ 'learning_rate':r['digital_adapter']['learning_rate'],'development_kl':r['final']['development']['teacher_kl']} for r in candidates],
                'peak_reserved_bytes':memory,'full_seconds_with_margin':projection,'collection_seconds':elapsed,
                'evidence':[str(folder/'collection.json')],
                'phase_progress_calibration':{'onchip':thresholds(projection),'validation':thresholds(max(60,math.ceil(elapsed*2))),
                    'collection':thresholds(max(60,math.ceil(elapsed*2)))},'progress_calibration':{'training':thresholds(projection+300)}}
            # Freeze measured values on first delivery; repeated completion is idempotent.
            decision=evidence/'readiness'/('selection-r'+str(rank)+'.json')
            if decision.exists(): receipt=json.loads(decision.read_text())
            else: write_frozen(decision,receipt)
            gate(queue_config,evidence,'selected-'+target,receipt)
            transition(queue_config,'release',PREFIX+'-'+target)
            result['selection']=receipt
        else:
            learned=next(r for r in rows if 'digital_adapter' in r); baseline=next(r for r in rows if r['onchip_arm']=='none')
            report={'rank':learned['digital_adapter']['rank'],'learning_rate':learned['digital_adapter']['learning_rate'],
                'input_tokens':learned['sequence_presentations']*256,'adapter':learned['digital_adapter'],'cost':learned['cost'],
                'effects':{split:{'deployment':baseline['final'][split],'lora':learned['final'][split],
                    'relative_kl_reduction':1-learned['final'][split]['teacher_kl']/baseline['final'][split]['teacher_kl']}
                    for split in ('development','transfer_generated','transfer_wikitext')},
                'collection':str(folder/'collection.json'),'parent_p0_sha256':'f80a9447469697cf9e4ff657cb17545d2824e60ae4f12d4694f01f3e4ebaf873',
                'scope':'Single-array exploratory comparison; independently tuned digital LoRA and user-selected dense LR.'}
            write_frozen(evidence/'reviews'/(label+'.json'),report); result['review']=report
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('generate')
    a=sub.add_parser('prepare-host');a.add_argument('--host',required=True);a.add_argument('--base',type=Path,default=DEFAULT_BASE);a.add_argument('--reused',type=Path,required=True)
    a=sub.add_parser('queue');a.add_argument('receipts',type=Path,nargs='+');a.add_argument('--output',type=Path,required=True);a.add_argument('--controller',type=Path,required=True);a.add_argument('--evidence',type=Path,required=True)
    a=sub.add_parser('collect');a.add_argument('--queue-config',type=Path,required=True);a.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='generate': result=generate()
    elif a.mode=='prepare-host':result=prepare_host(a.host,a.base,a.reused)
    elif a.mode=='queue':result=queue(a.receipts,a.output,a.controller,a.evidence)
    else:result=collect(a.queue_config,a.evidence)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__': main()
