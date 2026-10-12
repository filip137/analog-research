"""Frozen recovery transfer across HWA sources, rates and three assignments."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1e_prepare_host import command, checked_input, sha256, DEFAULT_BASE
from exp1f_cdt_lr import service, transition
from exp3_lora import thresholds
from slice_campaign import write_frozen
from experiments.artifacts import atomic_write_json
from workflow.lifecycle import parse_lifecycle, verify_provenance

PREFIX = 'opt125m-cmo-l6-exp7-recovery-regimes'
PHASE = 'opt-cmo-exp7-recovery-regimes-20261012'
NOTE = 'campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-013-recovery-regimes.md'
CATALOG = Path(__file__).with_name('exp7_recovery_sources.json')
TAGS = ('std3', 'cdt3', 'std9')
ARRAYS = (271002, 271003, 271004)
CAPS = {'cap2': 2, 'cap4': 4, 'uncapped': None}
HOSTS = ('loulou', 'nom-cool-1', 'local')
LOCAL_GROUPS = ('std3-271002', 'std9-271003')
CANARIES = {'std9-271002-cap4': 'loulou', 'cdt3-271002-cap4': 'nom-cool-1',
            'std3-271002-cap4': 'local'}
PEAK_MIB, HEADROOM_MIB = 20500, 2048
AUTHORIZATION = ('User approved exp-013: three frozen HWA sources, six fault rates, assignments '
    '271002-271004, 2/4/uncapped accumulated recovery, 2.1M tokens, LR .003, nominal 2-sigma. '
    'Use loulou RTX5090 and local/nom-cool-1 RTX3090s. Twelve GPU-hours, 600 GB, 24h from '
    'first admission; no scientific tuning or extension.')


def labels():
    return [f'{tag}-{array}-{cap}' for tag in TAGS for array in ARRAYS for cap in CAPS]


def parts(label):
    if label not in labels():
        raise ValueError('Unknown recovery case')
    tag, array, cap = label.split('-')
    return tag, int(array), cap


def lifecycle(label):
    tag, _, cap = parts(label)
    doc = json.loads((ROOT / f'campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1g-{tag}.json').read_text())
    doc['lifecycle_id'] = PREFIX + '-' + tag + '-' + cap
    doc['question'].update(text='How much recovery transfers across HWA sources and fault rates with 2, 4 or uncapped write attempts?',
        decision='Descriptive paired KL and write-cost comparison on assignments 271002-271004. No automatic winner, retuning or follow-up. '+NOTE)
    base = json.loads((ROOT / 'campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp6-writecap-cap4-dev.json').read_text())
    doc['onchip'] = base['onchip']
    doc['onchip']['max_write_attempts_per_device'] = CAPS[cap]
    # Retain the original five-assignment upstream contract; execution slices
    # select the declared three. Never rewrite frozen deployment provenance.
    return parse_lifecycle(doc)


def generate():
    out = []
    for tag in TAGS:
        for cap in CAPS:
            lc = lifecycle(f'{tag}-{ARRAYS[0]}-{cap}')
            path = ROOT / 'campaigns/opt-cmo-corruption/lifecycles' / (lc.lifecycle_id+'.json')
            write_frozen(path, lc.to_dict()); out.append(str(path))
    return out


def groups(host):
    if host not in HOSTS:
        raise ValueError('Unassigned host')
    return list(LOCAL_GROUPS) if host == 'local' else [f'{t}-{a}' for t in TAGS for a in ARRAYS]


def eligible(label, host):
    parts(label)
    if label in CANARIES:
        return host == CANARIES[label]
    return label.rsplit('-', 1)[0] in groups(host)


def materialize_all(origins, output):
    from workflow.portable_deployment import materialize, validate
    catalog = json.loads(CATALOG.read_text()); reports = []
    for group, ref in sorted(catalog.items()):
        src = origins / group; dst = output / group; dst.mkdir(parents=True, exist_ok=True)
        for name, expected in ref['run_files'].items():
            checked_input(src / name, expected)
        checked_input(src/'mapping.json', ref['mapping_sha256'])
        checked_input(src/ref['deployment']['path'], ref['deployment']['sha256'])
        for role, name in (('device_data','cache.pt'), ('device_model','devices.pt')):
            checked_input(origins/ref['tag']/name, ref['inputs'][role]['sha256'])
            target = output/ref['tag']/name; target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): shutil.copy2(origins/ref['tag']/name, target)
            checked_input(target, ref['inputs'][role]['sha256'])
        target = dst/'portable.pt'
        receipt = validate(target) if target.exists() else materialize(src, origins/ref['tag']/'devices.pt', target, 'cuda:0')
        if (receipt['parent_sha256'] != ref['deployment']['sha256'] or
            receipt['apparent_sha256'] != {k:v['apparent_sha256'] for k,v in ref['cases'].items()}):
            raise ValueError('Materialized P0 does not match catalog')
        reports.append(receipt)
        print(json.dumps({'stage':'materialization', 'group':group, 'complete':True}), flush=True)
    write_frozen(output/'materialized.json', {'groups':reports, 'source_catalog_sha256':sha256(CATALOG)})
    return {'groups':len(reports), 'physical_state_preserved':True}


def input_contract(base, reused, label):
    from experiments.cifar_crossbar.runtime import load
    tag, array, _ = parts(label); group = f'{tag}-{array}'
    ref = json.loads(CATALOG.read_text())[group]; lc = lifecycle(label)
    receipt = json.loads((reused/group/'portable.json').read_text())
    expected_apparent = {k:v['apparent_sha256'] for k,v in ref['cases'].items()}
    if (receipt['parent_sha256'] != ref['deployment']['sha256'] or not receipt['physical_state_preserved']
            or receipt['new_device_writes'] or receipt['apparent_sha256'] != expected_apparent):
        raise ValueError('Wrong frozen deployment receipt')
    paths = {'teacher_weights':base/'inputs/opt-125m-27dcfa74/pytorch_model.bin',
             'device_data':reused/tag/'cache.pt', 'device_model':reused/tag/'devices.pt',
             'device_state':reused/group/'portable.pt'}
    digests = {k:v['sha256'] for k,v in ref['inputs'].items()}
    digests['device_state'] = receipt['sha256']
    artifacts = {k:checked_input(p, digests[k]) for k,p in paths.items()}
    p0 = load(paths['device_state'])
    if (p0['array_seed'] != array or p0['source_sha256'] != ref['source_sha256']
            or p0['materialized_from_sha256'] != ref['deployment']['sha256']):
        raise ValueError('Wrong source or array in frozen P0')
    for role, kind, arm in [('device_data','feature_cache',None), ('device_model','device_bundle',None),
                            ('device_state','deployment',ref['hwa_arm'])]:
        bundle = p0 if role == 'device_state' else load(paths[role])
        verify_provenance(bundle['provenance'], lc, kind, hwa_arm=arm)
    return {'artifacts':artifacts, 'array_seed':array, 'hwa_arm':ref['hwa_arm'],
            'source_sha256':ref['source_sha256'], 'parent_p0_sha256':ref['deployment']['sha256'],
            'apparent_sha256':expected_apparent}


def prepare_host(host, base, reused):
    if command(['git','status','--porcelain'], cwd=ROOT).strip():
        raise ValueError('Worker checkout must be clean')
    revision = command(['git','rev-parse','HEAD'], cwd=ROOT).strip()
    if host == 'local' and shutil.disk_usage(ROOT).free < 120_000_000_000:
        raise ValueError('Local preparation must leave 40 GB after the declared six-job allowance')
    env = {'EBL_CORPUS_ROOT':str(base/'inputs/corpora'), 'EBL_CAMPAIGN_STREAM':'1',
           'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1'}
    contracts = {}; cases = []
    for label in labels():
        if not eligible(label, host): continue
        tag, array, _ = parts(label); lc = lifecycle(label); group = f'{tag}-{array}'
        if group not in contracts: contracts[group] = input_contract(base, reused, label)
        inputs = contracts[group]
        cfg = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
        if parse_lifecycle(json.loads(cfg.read_text())).digest() != lc.digest():
            raise ValueError('Lifecycle changed')
        jid = PREFIX+'-'+label; root = ROOT/'results/lifecycles'/jid
        inputs_path = root/'inputs.json'; write_frozen(inputs_path, inputs)
        command([sys.executable,'-m','workflow','plan',str(cfg),'--teacher-weights',inputs['artifacts']['teacher_weights']['path'],
                 '--output',str(root/'plan'),'--device','cuda:0','--cpu-threads','1','--python',sys.executable], cwd=ROOT)
        plan = root/'plan/campaign.json'; raw = json.loads(plan.read_text())
        stage = next(s for s in raw['stages'] if s['id'] == f'onchip__{inputs["hwa_arm"]}__{array}')
        stage['config'] = str((plan.parent/stage['config']).resolve()); stage['depends_on'] = []
        stage['inputs'] = {k:{'path':v['path']} for k,v in inputs['artifacts'].items()}
        raw['stages'] = [stage]; raw['campaign_id'] = jid
        manifest = root/'campaign.json'; write_frozen(manifest, raw)
        command([sys.executable,'-m','ebl','campaign','run','--manifest',str(manifest), '--output-dir',str(root/'dry-run'),
                 '--dry-run','--fail-fast'], cwd=ROOT, env=dict(os.environ,**env), logs=root/'preparation/dry-run')
        receipt = base/'worker-receipts'/revision/jid; receipt.parent.mkdir(parents=True, exist_ok=True)
        worker = root/'tmux-worker.json'; native = root/'runs'/jid
        write_frozen(worker, {'case_id':jid, 'state_root':str(receipt), 'cwd':str(ROOT), 'timeout_seconds':3900,
            'lease_seconds':900, 'env':env, 'argv':[sys.executable,'-m','ebl','campaign','run','--manifest',str(manifest),
                                                 '--output-dir',str(root/'runs'),'--resume','--fail-fast']})
        files = [cfg, manifest, worker, inputs_path, Path(stage['config']), CATALOG]
        cases.append({'id':jid,'label':label,'lifecycle_sha256':lc.digest(),'dry_run_passed':True,
            'expected_seconds':900 if host == 'loulou' else 2100,'budget_seconds':4200,
            'placement':{'workdir':str(ROOT),'run_dir':str(receipt),'gpu_models':['5090' if host=='loulou' else '3090'],
                'argv':[sys.executable,'-m','campaigns.tmux_worker','bridge','--spec',str(worker)],
                'validate_argv':[sys.executable,'-m','campaigns.recovery_collect','validate',str(native),'--inputs',str(inputs_path)],
                'campaign_python':sys.executable,'campaign_worker_spec':str(worker),'campaign_result_root':str(native),
                'recovery_inputs':str(inputs_path),'recovery_input_contract':inputs,
                'prepared_files':[{'path':str(p),'sha256':sha256(p)} for p in files]+list(inputs['artifacts'].values())}})
    return {'host':host,'source':str(ROOT),'source_revision':revision,'base':str(base),'cases':cases}


def campaign_manifest(worker, evidence, revision):
    cases = []
    for label in labels():
        canary = label in CANARIES
        cases.append({'id':PREFIX+'-'+label, 'stage':'readiness' if canary else 'training',
            'dependencies':[] if canary else [PREFIX+'-'+x for x in CANARIES],
            'required_gates':['preflight'] if canary else ['handoff'],
            'expected_gpu_seconds':2100, 'expected_storage_bytes':15_000_000_000,
            'progress':{**thresholds(2400), 'phases':{'onchip':thresholds(2400),
                        'validation':thresholds(300),'collection':thresholds(300)}}})
    doc = {'version':1,'id':PHASE,'poll_seconds':300,'required_adapters':['tmux'],
           'budgets':{'gpu_hours':11.5,'storage_bytes':600_000_000_000,'window_seconds':86400}, 'cases':cases,
           'storage_roots':[{'target':h,'path':str(worker)} for h in HOSTS] +
             [{'target':h,'path':str(DEFAULT_BASE/'worker-receipts'/revision)} for h in HOSTS] +
             [{'target':'local','path':str(evidence)}]}
    from campaigns.execution_service import validate_manifest
    validate_manifest(doc)
    return doc


def queue(receipts, output, controller, evidence):
    hosts = {json.loads(p.read_text())['host']:json.loads(p.read_text()) for p in receipts}
    if set(hosts) != set(HOSTS) or len({h['source_revision'] for h in hosts.values()}) != 1:
        raise ValueError('Require three prepared hosts at one frozen revision')
    jobs = []
    deadline = json.loads((evidence/'readiness/execution-window.json').read_text())['deadline']
    for label in labels():
        lc = lifecycle(label)
        prepared = {host:c for host,h in hosts.items() for c in h['cases']
                    if c['label']==label and eligible(label,host)}
        if not prepared:
            raise ValueError('Scientific case has no validated placement: '+label)
        if any(not c['dry_run_passed'] or c['lifecycle_sha256'] != lc.digest() for c in prepared.values()):
            raise ValueError('Unprepared scientific case')
        job = {'id':PREFIX+'-'+label,'project_dir':str(controller),'handoff':NOTE,'authorization':AUTHORIZATION,
               'peak_memory_mib':PEAK_MIB,'headroom_mib':HEADROOM_MIB,'expected_seconds':2100,
               'budget_seconds':4200,'deadline':deadline,'priority':120 if label in CANARIES else 100,
               'startup_grace_seconds':600,'stale_seconds':900,'recovery':False,'retry_safe':False,
               'placements':{h:c['placement'] for h,c in prepared.items()},
               'completion':{'argv':[sys.executable,str(controller/'campaigns/opt-cmo-corruption/launch/exp7_recovery_regimes.py'),
                    'collect','--queue-config',str(Path.home()/'.config/run-watch/queue.json'),'--evidence',str(evidence)],
                    'workdir':str(controller),'timeout_seconds':900,'retry_safe':True}}
        path = output/'jobs'/(job['id']+'.json'); write_frozen(path, job); jobs.append(str(path))
    revision = next(iter(hosts.values()))['source_revision']
    # CPU collection can live in a newer operational controller checkout while
    # prepared GPU workers retain their exact original source and native inputs.
    workers = {h['source'] for h in hosts.values()}
    if len(workers) != 1: raise ValueError('Expected the declared common worker path')
    write_frozen(output/'campaign.json', campaign_manifest(Path(next(iter(workers))),evidence,revision))
    return {'manifest':str(output/'campaign.json'),'jobs':jobs}


def checked_report(label, rows, records, collection):
    tag, array, capname = parts(label); cap = CAPS[capname]; lc = lifecycle(label)
    expected = {(c.label,a.arm_id) for c in lc.defects.cases for a in lc.onchip.arms}
    if len(rows)!=12 or {(r['case'],r['onchip_arm']) for r in rows} != expected:
        raise ValueError('Missing or duplicate source/rate/arm coverage')
    ref = json.loads(CATALOG.read_text())[f'{tag}-{array}']; values=[]
    for case in lc.defects.cases:
        hold = next(r for r in rows if r['case']==case.label and r['onchip_arm']=='none')
        learn = next(r for r in rows if r['case']==case.label and r['onchip_arm']=='onchip_weights')
        if (any(r['array_seed'] != array or r['hwa_arm'] != ref['hwa_arm'] for r in (hold,learn))
                or learn['sequence_presentations']!=8192
                or any(r['initial_apparent_sha256'] != ref['cases'][case.label]['apparent_sha256'] for r in (hold,learn))
                or hold['final_apparent_sha256']!=hold['initial_apparent_sha256']):
            raise ValueError('Wrong source, array, P0 or training exposure')
        cost = learn['cost']
        if cap is not None and (not 0 <= cost['max_onchip_device_writes'] <= cost['max_onchip_device_write_attempts'] <= cap
                or cost['issued_write_attempts']!=cost['device_writes']+cost['stuck_write_attempts']):
            raise ValueError('Write-attempt cap or accounting failed')
        if any(hold['cost'].get(k,0) for k in ('device_writes','device_programs','stuck_write_attempts','verify_reads')):
            raise ValueError('Held control changed devices')
        effects = {}
        for split in ('development','transfer_generated','transfer_wikitext'):
            initial, final = hold['final'][split], learn['final'][split]
            if not all(math.isfinite(v['teacher_kl']) and v['teacher_kl']>=0 for v in (initial,final)):
                raise ValueError('Invalid KL')
            effects[split] = {'deployment':initial,'onchip':final,'kl_reduction':initial['teacher_kl']-final['teacher_kl'],
                'relative_kl_reduction':1-final['teacher_kl']/initial['teacher_kl'] if initial['teacher_kl']>0 else None}
        values.append({'tag':tag,'array_seed':array,'case':case.label,'rate_ppm':case.rate_ppm,
                       'cap':capname,'max_write_attempts_per_device':cap,'effects':effects,'cost':cost,
                       'initial_apparent_sha256':hold['initial_apparent_sha256']})
    memory = max(m['peak_reserved_bytes'] for m in records if m.get('stage')=='onchip_memory')
    if memory>PEAK_MIB*1024**2: raise ValueError('Measured CUDA peak exceeds reservation')
    elapsed = next(m['elapsed_seconds'] for m in records if m.get('stage')=='onchip' and m.get('terminal'))
    if not math.isfinite(elapsed) or elapsed <= 0: raise ValueError('Invalid stage timing')
    return {'label':label,'rows':values,'peak_reserved_bytes':memory,'onchip_seconds':elapsed,'collection':str(collection)}


def summarize(reports):
    if len(reports)!=len(labels()) or {r['label'] for r in reports}!=set(labels()):
        raise ValueError('Incomplete or duplicate scientific coverage')
    rows = [dict(row) for report in reports for row in report['rows']]
    index = {(r['tag'],r['array_seed'],r['case'],r['cap']):r for r in rows}
    if len(index)!=162: raise ValueError('Expected 162 distinct recovery endpoints')
    for row in rows:
        ref = index[row['tag'],row['array_seed'],row['case'],'uncapped']
        if row['initial_apparent_sha256']!=ref['initial_apparent_sha256']:
            raise ValueError('Capped and uncapped P0 differ')
        row['retained_gain']={}; row['vs_best_hwa_only']={}
        for split, effect in row['effects'].items():
            gain = ref['effects'][split]['kl_reduction']
            row['retained_gain'][split] = effect['kl_reduction']/gain if gain>0 else None
            best = min(index[tag,row['array_seed'],row['case'],'uncapped']['effects'][split]['deployment']['teacher_kl'] for tag in TAGS)
            row['vs_best_hwa_only'][split] = {'reference_kl':best,'kl_reduction':best-effect['onchip']['teacher_kl']}
    return {'coverage_complete':True,'starting_conditions':54,'recovery_trajectories':162,'rows':rows,
            'selection':'Descriptive frozen-recipe comparison; no automatic winner, retuning or further dispatch.'}


def gate(queue_config, evidence, name, payload):
    from campaigns.execution_service import digest
    manifest = json.loads((evidence/'prepared/queue/campaign.json').read_text())
    path = evidence/'readiness'/(name+'.json')
    write_frozen(path, {'manifest_sha256':digest(manifest),'passed':True,**payload})
    service(queue_config,'gate',PHASE,name,path)


def collect(queue_config, evidence):
    from campaigns.recovery_collect import collect as native_collect
    label = os.environ['RUN_WATCH_JOB_ID'].removeprefix(PREFIX+'-'); parts(label)
    result = native_collect(queue_config,evidence/'collected'); folder = Path(result['collected'])
    report = checked_report(label,json.loads((folder/'measurements.json').read_text()),
        [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()], folder/'collection.json')
    with (evidence/'.completion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        write_frozen(evidence/'reviews'/(label+'.json'),report)
        canaries = [evidence/'reviews'/(name+'.json') for name in CANARIES]
        if all(p.exists() for p in canaries):
            measured = {name:json.loads(path.read_text())['onchip_seconds']
                        for name,path in zip(CANARIES,canaries)}
            gate(queue_config,evidence,'handoff',{'evidence':[str(p) for p in canaries],
                'validated_hosts':list(HOSTS),'scientific_settings_unchanged':True,
                'measured_onchip_seconds':measured,
                'phase_progress_calibration':{'onchip':thresholds(max(600,math.ceil(1.25*max(measured.values())+120)))}})
            for name in labels(): transition(queue_config,'release',PREFIX+'-'+name)
        completed = [json.loads(p.read_text()) for name in labels() if (p:=evidence/'reviews'/(name+'.json')).exists()]
        atomic_write_json(evidence/'reviews/progress.json',{'completed_jobs':len(completed),'expected_jobs':27,
                          'completed_trajectories':6*len(completed),'reports':[r['label'] for r in completed]})
        if len(completed)==27:
            summary = summarize(completed)
            write_frozen(evidence/'reviews/summary.json', summary)
            from exp7_recovery_report import render
            render(summary, evidence/'reviews/figures')
    return {'collected':str(folder),'label':label,'validated_rates':6}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('generate')
    q=sub.add_parser('materialize');q.add_argument('--origins',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('prepare-host');q.add_argument('--host',required=True);q.add_argument('--base',type=Path,default=DEFAULT_BASE);q.add_argument('--reused',type=Path,required=True)
    q=sub.add_parser('queue');q.add_argument('receipts',type=Path,nargs='+');q.add_argument('--output',type=Path,required=True);q.add_argument('--controller',type=Path,required=True);q.add_argument('--evidence',type=Path,required=True)
    q=sub.add_parser('collect');q.add_argument('--queue-config',type=Path,required=True);q.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='generate': result=generate()
    elif a.mode=='materialize': result=materialize_all(a.origins,a.output)
    elif a.mode=='prepare-host': result=prepare_host(a.host,a.base,a.reused)
    elif a.mode=='queue': result=queue(a.receipts,a.output,a.controller,a.evidence)
    else: result=collect(a.queue_config,a.evidence)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
