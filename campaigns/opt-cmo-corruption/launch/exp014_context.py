"""Prepare and collect native context-length recovery runs; never implements training."""
from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from exp1e_prepare_host import command, checked_input, sha256, DEFAULT_BASE
from exp1f_cdt_lr import service, transition
from exp3_lora import thresholds
from slice_campaign import write_frozen
from experiments.artifacts import atomic_write_json
from workflow.lifecycle import parse_lifecycle, verify_provenance

PREFIX = 'opt125m-cmo-l6-exp014-context'
PHASE = 'opt-cmo-exp014-context-20261012'
NOTE = 'campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-014-context-length.md'
CATALOG = Path(__file__).with_name('exp014_context_sources.json')
LENGTHS = (64,128,256,512,1024)
CAPS = {'cap2':2,'cap4':4,'uncapped':None}
ARM = 'standard_hwa_model'
CASE = 'random_30000ppm'
AUTHORIZATION = ('User explicitly said launch this for exp-014: five recovery contexts, cap2/cap4/uncapped, '
 'fixed 2.1M input tokens and 256 updates, std3 frozen HWA, 3% faults; 15 screens and at most 18 '
 'conditional confirmation fits. Five GPU-hours including readiness, 120 GB, six hours from first launch. '
 'At most two idle authorized RTX5090 hosts; preserve unrelated work and exp013 workers.')


def labels(phase=None):
    screens=[f'screen-l{length}-{cap}' for length in LENGTHS for cap in CAPS]
    confirm=[f'confirm-a{array}-l{length}-{cap}' for array in (271002,271003,271004)
             for length in LENGTHS for cap in CAPS]
    return screens if phase=='screen' else confirm if phase=='confirm' else screens+confirm


def parts(label):
    if label not in labels(): raise ValueError('Unknown context case')
    words=label.split('-')
    return words[0],271001 if words[0]=='screen' else int(words[1][1:]),int(words[-2][1:]),words[-1]


def lifecycle(label):
    phase,array,length,cap=parts(label)
    name='opt125m-cmo-l6-exp6-writecap-cap4-dev.json' if phase=='screen' else 'opt125m-cmo-l6-exp1g-std3.json'
    doc=json.loads((ROOT/'campaigns/opt-cmo-corruption/lifecycles'/name).read_text())
    base=json.loads((ROOT/'campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp6-writecap-cap4-dev.json').read_text())
    doc['lifecycle_id']=PREFIX+'-'+label
    doc['question'].update(text='Which recovery context retains improvement after frozen HWA at matched tokens and updates?',
        decision='Shortest context retaining 95% of best development gain at fixed evaluation contexts 256 and 1024. '+NOTE)
    doc['onchip']=base['onchip']
    doc['onchip'].update(batch_size=8192//length,microbatch_size=2048//length,
                         max_write_attempts_per_device=CAPS[cap],case_labels=[CASE])
    doc['recovery_views']={'context_length':length,'parent_length':1024,'parent_sequences':2048,
                           'score_stride':64,'evaluation_lengths':[256,1024]}
    return parse_lifecycle(doc)


def probe(length):
    doc=lifecycle(f'screen-l{length}-cap4').to_dict()
    doc['lifecycle_id']=PREFIX+f'-probe-l{length}'
    doc['question']['text']='NON-EVIDENCE readiness probe: two analog layers and new effective batch/context.'
    doc['network']['analog_decoder_layers']=2
    doc['devices']['assignment_seeds']=[271901];doc['devices']['selection_seeds']=[211901]
    doc['data'].update(hwa_sequences=8,recovery_sequences=8,development_sequences=8)
    doc.pop('transfer_tests')
    doc['recovery_views']['parent_sequences']=16
    doc['defects']['cases'].insert(0,{'kind':'none','rate_ppm':0,'die_concentration':None,'array_concentration':None})
    doc['onchip'].pop('case_labels')
    doc['hwa'].update(epochs=1,batch_size=8,selection={'every_epochs':1,'extension_epochs':0})
    doc['hwa'].pop('lr_schedule');doc['hwa'].pop('sequences_per_epoch')
    return parse_lifecycle(doc)


def canary():
    doc=lifecycle('screen-l1024-cap4').to_dict()
    doc['lifecycle_id']=PREFIX+'-canary'
    doc['question']['text']='NON-EVIDENCE eight-update six-layer frozen-P0 context canary.'
    doc['recovery_views']['parent_sequences']=64
    return parse_lifecycle(doc)


def generate():
    configs=[lifecycle(label) for label in labels()]+[probe(64),probe(1024),canary()]
    for lc in configs:
        write_frozen(ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json'),lc.to_dict())
    return {'definitions':len(configs),'maximum_learning_fits':33}


def native_plan(lc,base,identifier=None):
    identifier=identifier or lc.lifecycle_id
    root=ROOT/'results/lifecycles'/identifier
    config=ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
    command([sys.executable,'-m','workflow','plan',str(config),'--teacher-weights',
        str(base/'inputs/opt-125m-27dcfa74/pytorch_model.bin'),'--output',str(root/'plan'),
        '--device','cuda:0','--cpu-threads','1','--python',sys.executable],cwd=ROOT)
    raw=json.loads((root/'plan/campaign.json').read_text());raw['campaign_id']=identifier
    for stage in raw['stages']: stage['config']=str((root/'plan'/stage['config']).resolve())
    return root,raw


def execute(root,raw,base):
    manifest=root/'campaign.json';write_frozen(manifest,raw)
    command([sys.executable,'-m','ebl','campaign','run','--manifest',str(manifest),
        '--output-dir',str(root/'runs'),'--resume','--fail-fast'],cwd=ROOT,
        env=dict(os.environ,EBL_CORPUS_ROOT=str(base/'inputs/corpora'),EBL_CAMPAIGN_STREAM='1'),logs=root/'execution')
    run=root/'runs'/raw['campaign_id']
    aggregates=json.loads((run/'aggregate/results.json').read_text())['stages']
    if any(row['status']!='complete' for row in aggregates): raise ValueError('Incomplete readiness stage')
    return [Path(row['run_result']).parent for row in aggregates]


def make_cache(lc,base,key):
    root,raw=native_plan(lc,base,PREFIX+'-cache-'+key)
    raw['stages']=[s for s in raw['stages'] if s['id']=='prepare']
    [run]=execute(root,raw,base)
    result=json.loads((run/'result.json').read_text())
    item=next(a for a in result['artifacts'] if a['kind']=='feature_cache')
    return checked_input(run/item['path'],item['sha256'])


def inputs_for(lc,array,cache,base):
    from experiments.cifar_crossbar.runtime import load
    ref=json.loads(CATALOG.read_text())[str(array)]
    paths={k:Path(v['path']) for k,v in ref['artifacts'].items()}
    # Inputs copied to this isolated worker are immutable byte-identical files.
    for role in ('device_state','device_model'):
        local=ROOT/'results/reused'/str(array)/paths[role].name
        if local.exists(): paths[role]=local
    paths['teacher_weights']=base/'inputs/opt-125m-27dcfa74/pytorch_model.bin'
    artifacts={k:checked_input(p,ref['artifacts'][k]['sha256']) for k,p in paths.items() if k!='device_data'}
    artifacts['device_data']=cache
    for role,kind,arm in [('device_state','deployment',ARM),('device_model','device_bundle',None),('device_data','feature_cache',None)]:
        bundle=load(artifacts[role]['path']);verify_provenance(bundle['provenance'],lc,kind,hwa_arm=arm)
    return {**ref,'artifacts':artifacts}


def recovery_plan(lc,array,inputs,base):
    root,raw=native_plan(lc,base)
    stage=next(s for s in raw['stages'] if s['id']==f'onchip__{ARM}__{array}')
    stage['depends_on']=[];stage['inputs']={k:{'path':v['path']} for k,v in inputs['artifacts'].items()}
    raw['stages']=[stage]
    write_frozen(root/'inputs.json',inputs)
    return root,raw


def readiness(base):
    report={'probes':[],'caches':{}}
    for length in (64,1024):
        lc=probe(length);root,raw=native_plan(lc,base)
        runs=execute(root,raw,base)
        for run in runs:
            command([sys.executable,'-m','ebl','runs','inspect',str(run),'--verify-artifacts','--require-complete'],cwd=ROOT)
        report['probes'].append({'length':length,'runs':[str(p) for p in runs]})
        print(json.dumps({'stage':'probe','length':length,'complete':True}),flush=True)
    for length in LENGTHS:
        report['caches'][str(length)]=make_cache(lifecycle(f'screen-l{length}-cap4'),base,f'l{length}')
        print(json.dumps({'stage':'prepare','length':length,'complete':True}),flush=True)
    lc=canary();cache=make_cache(lc,base,'canary')
    inputs=inputs_for(lc,271001,cache,base);root,raw=recovery_plan(lc,271001,inputs,base)
    [run]=execute(root,raw,base)
    command([sys.executable,'-m','campaigns.recovery_collect','validate',str(root/'runs'/raw['campaign_id']),
             '--inputs',str(root/'inputs.json')],cwd=ROOT)
    rows=json.loads((run/'measurements.json').read_text());learn=next(r for r in rows if r['weights']=='learn')
    metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    memory=max(r['peak_reserved_bytes'] for r in metrics if r.get('stage')=='onchip_memory')
    seconds=next(r['elapsed_seconds'] for r in metrics if r.get('stage')=='onchip' and r.get('terminal'))
    train_seconds=learn['curve'][-1]['training_seconds']
    report.update(canary_run=str(run),peak_reserved_bytes=memory,canary_seconds=seconds,
                  projected_fit_seconds=math.ceil(1.35*(train_seconds*32+seconds-train_seconds)+30),
                  physical_state_preserved=True)
    if memory>26000*1024**2: raise ValueError('Canary exceeds GPU reservation')
    if report['projected_fit_seconds']*15>5400: raise ValueError('Stage A exceeds reserved screen budget')
    write_frozen(ROOT/'results/context-readiness.json',report)
    return report


def prepare_host(host,base):
    revision=command(['git','rev-parse','HEAD'],cwd=ROOT).strip()
    if command(['git','status','--porcelain'],cwd=ROOT).strip(): raise ValueError('Worker must be clean')
    readiness=json.loads((ROOT/'results/context-readiness.json').read_text())
    cases=[];cached_inputs={}
    for label in labels():
        phase,array,length,cap=parts(label)
        # trex has limited disk: at most ten capped screens, no confirmation states.
        if host=='trex' and (phase!='screen' or cap=='uncapped'): continue
        lc=lifecycle(label);key=(array,length)
        if key not in cached_inputs: cached_inputs[key]=inputs_for(lc,array,readiness['caches'][str(length)],base)
        inputs=cached_inputs[key];root,raw=recovery_plan(lc,array,inputs,base)
        manifest=root/'campaign.json';write_frozen(manifest,raw)
        command([sys.executable,'-m','ebl','campaign','run','--manifest',str(manifest),
            '--output-dir',str(root/'dry-run'),'--dry-run','--fail-fast'],cwd=ROOT,logs=root/'preparation/dry-run')
        receipt=base/'worker-receipts'/revision/lc.lifecycle_id;receipt.parent.mkdir(parents=True,exist_ok=True)
        worker=root/'tmux-worker.json'
        write_frozen(worker,{'case_id':lc.lifecycle_id,'state_root':str(receipt),'cwd':str(ROOT),
          'timeout_seconds':660,'lease_seconds':900,'env':{'EBL_CORPUS_ROOT':str(base/'inputs/corpora'),
          'EBL_CAMPAIGN_STREAM':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1'},
          'argv':[sys.executable,'-m','ebl','campaign','run','--manifest',str(manifest),
                  '--output-dir',str(root/'runs'),'--resume','--fail-fast']})
        native=root/'runs'/raw['campaign_id']; files=[manifest,worker,root/'inputs.json',Path(raw['stages'][0]['config']),CATALOG]
        placement={'workdir':str(ROOT),'run_dir':str(receipt),'gpu_models':['5090'],
          'argv':[sys.executable,'-m','campaigns.tmux_worker','bridge','--spec',str(worker)],
          'validate_argv':[sys.executable,'-m','campaigns.recovery_collect','validate',str(native),'--inputs',str(root/'inputs.json')],
          'campaign_python':sys.executable,'campaign_worker_spec':str(worker),'campaign_result_root':str(native),
          'recovery_inputs':str(root/'inputs.json'),'recovery_input_contract':inputs,
          'prepared_files':[{'path':str(p),'sha256':sha256(p)} for p in files]+list(inputs['artifacts'].values())}
        cases.append({'id':lc.lifecycle_id,'label':label,'lifecycle_sha256':lc.digest(),'placement':placement})
    return {'host':host,'source':str(ROOT),'source_revision':revision,'cases':cases,'readiness':readiness}


def selection(reports):
    if {r['label'] for r in reports}!=set(labels('screen')) or len(reports)!=15: raise ValueError('Incomplete context screen')
    chosen={}
    for cap in CAPS:
        group=[r for r in reports if r['cap']==cap]
        baselines={name:group[0]['development'][name]['initial'] for name in ('256','1024')}
        if any(r['development'][name]['initial']!=baselines[name] for r in group for name in baselines):
            raise ValueError('Screen held baseline changed')
        gains={name:baselines[name]-min(r['development'][name]['final'] for r in group) for name in baselines}
        eligible=[r['length'] for r in group if all(gains[name]>=.02*baselines[name] and
                  baselines[name]-r['development'][name]['final']>=.95*gains[name] for name in baselines)]
        chosen[cap]=min(eligible) if eligible else None
    return chosen


def queue(receipts,evidence,controller):
    from campaigns.execution_service import validate_manifest
    hosts={d['host']:d for p in receipts for d in [json.loads(p.read_text())]}
    if set(hosts)!={'trex','loulou'} or len({d['source_revision'] for d in hosts.values()})!=1:
        raise ValueError('Two isolated hosts at one source revision are required')
    window=json.loads((evidence/'readiness/execution-window.json').read_text())
    outputs={}
    for phase,hours in [('screen',1.5),('confirm',3.)]:
        directory=evidence/'prepared'/phase;cases=[];jobs=[]
        for label in labels(phase):
            lc=lifecycle(label)
            prepared={h:c for h,d in hosts.items() for c in d['cases'] if c['label']==label}
            if not prepared or any(c['lifecycle_sha256']!=lc.digest() for c in prepared.values()):
                raise ValueError('Unprepared case '+label)
            job={'id':lc.lifecycle_id,'project_dir':str(controller),'handoff':NOTE,'authorization':AUTHORIZATION,
                 'peak_memory_mib':26000,'headroom_mib':2700,'expected_seconds':300,'budget_seconds':720,
                 'deadline':window['deadline'],'priority':90,'startup_grace_seconds':600,'stale_seconds':900,
                 'recovery':False,'retry_safe':False,'placements':{h:c['placement'] for h,c in prepared.items()},
                 'completion':{'argv':[sys.executable,str(controller/'campaigns/opt-cmo-corruption/launch/exp014_context.py'),
                    'collect','--queue-config',str(Path.home()/'.config/run-watch/queue.json'),'--evidence',str(evidence)],
                    'workdir':str(controller),'timeout_seconds':600,'retry_safe':True}}
            path=directory/'jobs'/(lc.lifecycle_id+'.json');write_frozen(path,job);jobs.append(str(path))
            cases.append({'id':lc.lifecycle_id,'stage':'training','dependencies':[],
                 'required_gates':['preflight' if phase=='screen' else 'selected-'+label],
                 'expected_gpu_seconds':300,'expected_storage_bytes':2500000000,
                 'progress':{**thresholds(600),'phases':{'onchip':thresholds(600),
                            'validation':thresholds(180),'collection':thresholds(180)}}})
        revision=next(iter(hosts.values()))['source_revision']
        manifest={'version':1,'id':PHASE+'-'+phase,'poll_seconds':300,'required_adapters':['tmux'],
                  'budgets':{'gpu_hours':hours,'storage_bytes':120000000000,'window_seconds':21600},'cases':cases,
                  'storage_roots':[{'target':h,'path':d['source']} for h,d in hosts.items()]+
                     [{'target':h,'path':str(DEFAULT_BASE/'worker-receipts'/revision)} for h in hosts]+
                     [{'target':'local','path':str(evidence)}]}
        validate_manifest(manifest);write_frozen(directory/'campaign.json',manifest)
        outputs[phase]={'manifest':str(directory/'campaign.json'),'jobs':jobs}
    return outputs


def checked_report(label,rows,collection):
    phase,array,length,cap=parts(label);lc=lifecycle(label)
    if len(rows)!=2 or {r['onchip_arm'] for r in rows}!={'none','onchip_weights'}: raise ValueError('Wrong trajectory coverage')
    hold=next(r for r in rows if r['onchip_arm']=='none');learn=next(r for r in rows if r['onchip_arm']=='onchip_weights')
    if (any(r['case']!=CASE or r['array_seed']!=array for r in rows) or
        learn['optimizer_updates']!=256 or learn['input_token_presentations']!=2097152 or
        learn['scored_token_presentations']!=2064384 or hold['final_apparent_sha256']!=hold['initial_apparent_sha256'] or
        hold['initial_apparent_sha256']!=learn['initial_apparent_sha256']): raise ValueError('Unmatched recovery contract')
    development={str(e):{'initial':hold['final'][key]['teacher_kl'],'final':learn['final'][key]['teacher_kl']}
                 for e,key in [(256,'development'),(1024,'context1024_development')]}
    if not all(math.isfinite(v) and v>=0 for row in development.values() for v in row.values()): raise ValueError('Invalid KL')
    return {'label':label,'phase':phase,'array':array,'length':length,'cap':cap,'development':development,
            'cost':learn['cost'],'initial':hold['final'],'final':learn['final'],'collection':str(collection)}


def collect(queue_config,evidence):
    from campaigns.recovery_collect import collect as native_collect
    from campaigns.execution_service import digest
    label=os.environ['RUN_WATCH_JOB_ID'].removeprefix(PREFIX+'-')
    result=native_collect(queue_config,evidence/'collected');folder=Path(result['collected'])
    report=checked_report(label,json.loads((folder/'measurements.json').read_text()),folder/'collection.json')
    with (evidence/'.completion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);write_frozen(evidence/'reviews'/(label+'.json'),report)
        screens=[evidence/'reviews'/(l+'.json') for l in labels('screen')]
        if all(p.exists() for p in screens):
            chosen=selection([json.loads(p.read_text()) for p in screens])
            write_frozen(evidence/'reviews/selection.json',{'selected_contexts':chosen,'test_data_used':False})
            manifest=json.loads((evidence/'prepared/confirm/campaign.json').read_text())
            selected=[]
            for name in labels('confirm'):
                _,_,length,cap=parts(name)
                if chosen[cap] is not None and length in {chosen[cap],1024}:
                    selected.append(name)
                    receipt=evidence/'readiness'/('selected-'+name+'.json')
                    write_frozen(receipt,{'manifest_sha256':digest(manifest),'passed':True,'selected_context':chosen[cap],
                                         'evidence':[str(evidence/'reviews/selection.json')]})
                    service(queue_config,'gate',PHASE+'-confirm','selected-'+name,receipt)
                    transition(queue_config,'release',PREFIX+'-'+name)
            complete=[p for name in selected if (p:=evidence/'reviews'/(name+'.json')).exists()]
            if len(complete)==len(selected):
                write_frozen(evidence/'reviews/summary.json',{'coverage_complete':True,'selected':chosen,
                    'screens':[json.loads(p.read_text()) for p in screens],
                    'confirmation':[json.loads(p.read_text()) for p in complete]})
        paths=list((evidence/'reviews').glob('screen-*.json'))+list((evidence/'reviews').glob('confirm-*.json'))
        atomic_write_json(evidence/'reviews/progress.json',{'completed_fits':len(paths),'screen_total':15,
                                                         'confirmation_maximum':18})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('generate')
    a=sub.add_parser('readiness');a.add_argument('--base',type=Path,default=DEFAULT_BASE)
    a=sub.add_parser('prepare-host');a.add_argument('--host',required=True);a.add_argument('--base',type=Path,default=DEFAULT_BASE)
    a=sub.add_parser('collect');a.add_argument('--queue-config',type=Path,required=True);a.add_argument('--evidence',type=Path,required=True)
    a=sub.add_parser('queue');a.add_argument('receipts',nargs='+',type=Path);a.add_argument('--evidence',type=Path,required=True);a.add_argument('--controller',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='generate':result=generate()
    elif a.mode=='readiness':result=readiness(a.base)
    elif a.mode=='prepare-host':result=prepare_host(a.host,a.base)
    elif a.mode=='queue':result=queue(a.receipts,a.evidence,a.controller)
    else:result=collect(a.queue_config,a.evidence)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
