"""exp-015 noise-free write oracle over the exp-013 held P0s; prepares and collects native runs only."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from exp1e_prepare_host import command, checked_input, sha256, DEFAULT_BASE
from exp1f_cdt_lr import service, transition
from exp3_lora import thresholds
from exp7_recovery_regimes import ARRAYS, CATALOG, TAGS, lifecycle as exp13_lifecycle
from slice_campaign import write_frozen
from experiments.artifacts import atomic_write_json
from workflow.lifecycle import parse_lifecycle, verify_provenance

PREFIX = 'opt125m-cmo-l6-exp15-noise-free'
PHASE = 'opt-cmo-exp15-noise-free-20261013'
NOTE = 'campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-015-noise-free-writes.md'
# Arm A evidence; collect() rebinds it beside the declared exp-015 evidence root.
EXP13 = ROOT/'results/lifecycles/opt125m-cmo-l6-exp7-recovery-regimes'
ARMS = ('thresholded', 'dense')
SPLITS = ('development', 'transfer_generated', 'transfer_wikitext')
HELD_OUT = ('transfer_generated', 'transfer_wikitext')
# The anchor reruns the unchanged exp-013 lifecycle on its original host.
ANCHOR = 'anchor-std3-271002-uncapped'
HOSTS = ('riri', 'nom-cool-1', 'local')
CANARIES = {ANCHOR: 'local', 'std3-271002-dense': 'riri', 'cdt3-271002-thresholded': 'nom-cool-1'}
EXPECTED_SECONDS = {'riri': 900, 'nom-cool-1': 2100, 'local': 2100}
PEAK_MIB, HEADROOM_MIB = 20500, 2048
# Recorded from the user's explicit launch approval; queue() refuses without it.
AUTHORIZATION = ('User approved exp-015 on 2026-10-13: write-noise-only exact-write oracle, arms B '
    '(thresholded nominal 2-sigma, accumulated, uncapped) and C (dense), LR .003, the full exp-013 grid '
    '(std3/cdt3/std9, 0-5% faults, assignments 271002-271004), exp-013 uncapped reused as A only after an '
    'exact anchor replay; thresholds 0.5/0.2 and the C-validity guard confirmed; no LR screen. riri RTX5090 '
    'with P0 staging from nom-cool-1, nom-cool-1 and nom-cool-2 RTX3090s. Eight GPU-hours, 250 GB, 24h from '
    'first admission; no scientific tuning or extension.')


def labels():
    return [ANCHOR] + [f'{tag}-{array}-{arm}' for tag in TAGS for array in ARRAYS for arm in ARMS]


def parts(label):
    if label not in labels():
        raise ValueError('Unknown exp-015 case')
    if label == ANCHOR:
        return 'std3', 271002, 'anchor'
    tag, array, arm = label.split('-')
    return tag, int(array), arm


def lifecycle(label):
    tag, array, arm = parts(label)
    if arm == 'anchor':
        return exp13_lifecycle('std3-271002-uncapped')
    doc = exp13_lifecycle(f'{tag}-{array}-uncapped').to_dict()
    doc['lifecycle_id'] = f'{PREFIX}-{tag}-{arm}'
    doc['question'].update(text='How much of the KL left after noisy recovery is write noise, write gating or fault structure?',
        decision='Frozen A/B/C decomposition with a C-validity guard on assignments 271002-271004. '
                 'Exact writes are a non-physical oracle; no automatic follow-up. '+NOTE)
    onchip = doc['onchip']
    onchip['write_noise_scale'] = 0.0
    if arm == 'dense':
        onchip['update_law'] = 'cmo_idealized_write_adam'
        del onchip['threshold_sigma'], onchip['accumulate_updates']
    return parse_lifecycle(doc)


def generate():
    out = []
    for tag in TAGS:
        for arm in ARMS:
            lc = lifecycle(f'{tag}-{ARRAYS[0]}-{arm}')
            path = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
            write_frozen(path, lc.to_dict()); out.append(str(path))
    return out


def eligible(label, host):
    parts(label)
    if host not in HOSTS:
        raise ValueError('Unassigned host')
    if label in CANARIES:
        return host == CANARIES[label]
    # Local storage admits only the anchor while keeping 40 GB free.
    return host != 'local'


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
    jobs = [label for label in labels() if eligible(label, host)]
    if shutil.disk_usage(ROOT).free < 40_000_000_000 + 12_000_000_000 * (len(jobs) if host == 'local' else 0):
        raise ValueError('Preparation must leave 40 GB free after the local allowance')
    env = {'EBL_CORPUS_ROOT':str(base/'inputs/corpora'), 'EBL_CAMPAIGN_STREAM':'1',
           'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1'}
    contracts = {}; cases = []
    for label in jobs:
        tag, array, _ = parts(label); lc = lifecycle(label); group = f'{tag}-{array}'
        if label not in contracts: contracts[label] = input_contract(base, reused, label)
        inputs = contracts[label]
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
            'expected_seconds':EXPECTED_SECONDS[host],'budget_seconds':4200,
            'placement':{'workdir':str(ROOT),'run_dir':str(receipt),'gpu_models':['5090' if host == 'riri' else '3090'],
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
            'expected_gpu_seconds':2100, 'expected_storage_bytes':12_000_000_000,
            'progress':{**thresholds(2400), 'phases':{'onchip':thresholds(2400),
                        'validation':thresholds(300),'collection':thresholds(300)}}})
    doc = {'version':1,'id':PHASE,'poll_seconds':300,'required_adapters':['tmux'],
           'budgets':{'gpu_hours':8,'storage_bytes':250_000_000_000,'window_seconds':86400}, 'cases':cases,
           'storage_roots':[{'target':h,'path':str(worker)} for h in HOSTS] +
             [{'target':h,'path':str(DEFAULT_BASE/'worker-receipts'/revision)} for h in HOSTS] +
             [{'target':'local','path':str(evidence)}]}
    from campaigns.execution_service import validate_manifest
    validate_manifest(doc)
    return doc


def queue(receipts, output, controller, evidence):
    if not AUTHORIZATION:
        raise ValueError('exp-015 has no recorded launch authorization')
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
               'completion':{'argv':[sys.executable,str(controller/'campaigns/opt-cmo-corruption/launch/exp15_noise_free.py'),
                    'collect','--queue-config',str(Path.home()/'.config/run-watch/queue.json'),'--evidence',str(evidence)],
                    'workdir':str(controller),'timeout_seconds':900,'retry_safe':True}}
        path = output/'jobs'/(job['id']+'.json'); write_frozen(path, job); jobs.append(str(path))
    revision = next(iter(hosts.values()))['source_revision']
    workers = {h['source'] for h in hosts.values()}
    if len(workers) != 1: raise ValueError('Expected the declared common worker path')
    write_frozen(output/'campaign.json', campaign_manifest(Path(next(iter(workers))),evidence,revision))
    return {'manifest':str(output/'campaign.json'),'jobs':jobs}


def development_tail(records, case):
    """Development KL of the learning arm at its last two 64-update checkpoints."""

    points = sorted((m['global_step'], m['development']['teacher_kl']) for m in records
                    if m.get('stage') == 'onchip_development' and m.get('case') == case and m.get('arm') == 'onchip_weights')
    if len(points) < 2 or points[-1][0] != 256:
        raise ValueError('Missing development checkpoints')
    return {'steps':[points[-2][0], points[-1][0]], 'teacher_kl':[points[-2][1], points[-1][1]],
            'rising':points[-1][1] > points[-2][1]}


def anchor_comparison(rows):
    """Exact equality with the collected exp-013 std3/271002/uncapped endpoints."""

    original = next(EXP13.glob('collected/opt125m-cmo-l6-exp7-recovery-regimes-std3-271002-uncapped/*/measurements.json'))
    reference = {(r['case'], r['onchip_arm']):r for r in json.loads(original.read_text())}
    mismatches = []
    for row in rows:
        ref = reference.get((row['case'], row['onchip_arm']))
        fields = {'final_apparent_sha256':(row['final_apparent_sha256'], ref and ref['final_apparent_sha256']),
                  'state_sha256':(row['state_sha256'], ref and ref['state_sha256'])}
        for split in SPLITS:
            for key in ('teacher_kl', 'prediction_sha256'):
                fields[f'{split}.{key}'] = (row['final'][split][key], ref and ref['final'][split][key])
        mismatches += [{'case':row['case'], 'arm':row['onchip_arm'], 'field':k} for k,(a,b) in fields.items() if a != b]
    return {'reference':str(original), 'exact':not mismatches and len(rows) == len(reference), 'mismatches':mismatches}


def checked_report(label, rows, records, collection):
    tag, array, arm = parts(label); lc = lifecycle(label)
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
        if any(hold['cost'].get(k,0) for k in ('device_writes','device_programs','stuck_write_attempts','verify_reads')):
            raise ValueError('Held control changed devices')
        if learn['cost'].get('verify_reads', 0):
            raise ValueError('An exact-write oracle must not read verify')
        effects = {}
        for split in SPLITS:
            initial, final = hold['final'][split], learn['final'][split]
            if not all(math.isfinite(v['teacher_kl']) and v['teacher_kl']>=0 for v in (initial,final)):
                raise ValueError('Invalid KL')
            effects[split] = {'deployment':initial,'onchip':final,'kl_reduction':initial['teacher_kl']-final['teacher_kl']}
        values.append({'tag':tag,'array_seed':array,'case':case.label,'rate_ppm':case.rate_ppm,'arm':arm,
                       'effects':effects,'cost':learn['cost'],'initial_apparent_sha256':hold['initial_apparent_sha256'],
                       'development_tail':development_tail(records, case.label)})
    memory = max(m['peak_reserved_bytes'] for m in records if m.get('stage')=='onchip_memory')
    if memory>PEAK_MIB*1024**2: raise ValueError('Measured CUDA peak exceeds reservation')
    elapsed = next(m['elapsed_seconds'] for m in records if m.get('stage')=='onchip' and m.get('terminal'))
    if not math.isfinite(elapsed) or elapsed <= 0: raise ValueError('Invalid stage timing')
    report = {'label':label,'rows':values,'peak_reserved_bytes':memory,'onchip_seconds':elapsed,'collection':str(collection)}
    if arm == 'anchor':
        report['anchor'] = anchor_comparison(rows)
    return report


def summarize(reports):
    """B/C endpoints joined with exp-013 arm A, and the frozen decomposition inputs."""

    science = [r for r in reports if r['label'] != ANCHOR]
    if len(science) != len(labels())-1 or {r['label'] for r in science} != set(labels())-{ANCHOR}:
        raise ValueError('Incomplete or duplicate scientific coverage')
    rows = {(r['tag'],r['array_seed'],r['rate_ppm'],r['arm']):r for report in science for r in report['rows']}
    if len(rows) != 108: raise ValueError('Expected 108 distinct exact-write endpoints')
    previous = json.loads((EXP13/'reviews/summary.json').read_text())
    a_rows = {(r['tag'],r['array_seed'],r['rate_ppm']):r for r in previous['rows'] if r['cap'] == 'uncapped'}
    units, guards = [], {}
    for (tag, array, rate, arm), row in sorted(rows.items()):
        if arm != 'thresholded': continue
        a, c = a_rows[tag,array,rate], rows[tag,array,rate,'dense']
        c0 = rows[tag,array,0,'dense']
        if not a['initial_apparent_sha256'] == row['initial_apparent_sha256'] == c['initial_apparent_sha256']:
            raise ValueError('Arms do not share their P0')
        for split in SPLITS:
            d = row['effects'][split]['deployment']['teacher_kl']
            values = {'A':a['effects'][split]['onchip']['teacher_kl'], 'B':row['effects'][split]['onchip']['teacher_kl'],
                      'C':c['effects'][split]['onchip']['teacher_kl'], 'C0':c0['effects'][split]['onchip']['teacher_kl']}
            unit = {'source':tag,'array_seed':array,'rate_ppm':rate,'domain':split,'D':d,**values,
                    'N':values['A']-values['B'],'T':values['B']-values['C'],'F':values['C']-values['C0']}
            span = values['A']-values['C0']
            unit['shares'] = ({k:unit[k]/span for k in ('N','T','F')} if rate > 0 and span > 0 else None)
            units.append(unit)
        guard = guards.setdefault(tag, {'C0_not_above_D':True,'development_not_rising':True,'violations':[]})
        if c['development_tail']['rising']:
            guard['development_not_rising'] = False; guard['violations'].append({'array_seed':array,'rate_ppm':rate,'kind':'rising'})
        if rate == 0:
            for split in HELD_OUT:
                if c['effects'][split]['onchip']['teacher_kl'] > c['effects'][split]['deployment']['teacher_kl']:
                    guard['C0_not_above_D'] = False
                    guard['violations'].append({'array_seed':array,'domain':split,'kind':'C0_above_D'})
    for guard in guards.values():
        guard['valid'] = guard['C0_not_above_D'] and guard['development_not_rising']
    return {'coverage_complete':True,'starting_conditions':54,'exact_write_endpoints':108,
            'arm_a_source':str(EXP13/'reviews/summary.json'),'rows':list(rows.values()),'units':units,'c_validity':guards,
            'selection':'Frozen decomposition inputs only; classification and verdict belong to the review.'}


def write_units(units, path):
    fields = ('source','array_seed','rate_ppm','domain','D','A','B','C','C0','N','T','F')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields+('share_N','share_T','share_F')); writer.writeheader()
        for u in units:
            shares = u['shares'] or {}
            writer.writerow({**{k:u[k] for k in fields}, **{f'share_{k}':shares.get(k) for k in ('N','T','F')}})


def gate(queue_config, evidence, name, payload):
    from campaigns.execution_service import digest
    manifest = json.loads((evidence/'prepared/queue/campaign.json').read_text())
    path = evidence/'readiness'/(name+'.json')
    write_frozen(path, {'manifest_sha256':digest(manifest),'passed':True,**payload})
    service(queue_config,'gate',PHASE,name,path)


def collect(queue_config, evidence):
    from campaigns.recovery_collect import collect as native_collect
    global EXP13
    EXP13 = evidence.parent/'opt125m-cmo-l6-exp7-recovery-regimes'
    label = os.environ['RUN_WATCH_JOB_ID'].removeprefix(PREFIX+'-'); parts(label)
    result = native_collect(queue_config,evidence/'collected'); folder = Path(result['collected'])
    report = checked_report(label,json.loads((folder/'measurements.json').read_text()),
        [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()], folder/'collection.json')
    with (evidence/'.completion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        write_frozen(evidence/'reviews'/(label+'.json'),report)
        canaries = [evidence/'reviews'/(name+'.json') for name in CANARIES]
        anchor = evidence/'reviews'/(ANCHOR+'.json')
        if all(p.exists() for p in canaries) and not (evidence/'readiness/handoff.json').exists():
            exact = json.loads(anchor.read_text())['anchor']['exact']
            if exact:
                measured = {name:json.loads(path.read_text())['onchip_seconds'] for name,path in zip(CANARIES,canaries)}
                gate(queue_config,evidence,'handoff',{'evidence':[str(p) for p in canaries],
                    'validated_hosts':list(HOSTS),'scientific_settings_unchanged':True,'anchor_exact':True,
                    'measured_onchip_seconds':measured,
                    'phase_progress_calibration':{'onchip':thresholds(max(600,math.ceil(1.25*max(measured.values())+120)))}})
                for name in labels(): transition(queue_config,'release',PREFIX+'-'+name)
            else:
                # The note requires escalation: held jobs stay held and arm A is not reused.
                write_frozen(evidence/'readiness/anchor-failed.json', json.loads(anchor.read_text())['anchor'])
        completed = [json.loads(p.read_text()) for name in labels() if (p:=evidence/'reviews'/(name+'.json')).exists()]
        atomic_write_json(evidence/'reviews/progress.json',{'completed_jobs':len(completed),'expected_jobs':len(labels()),
                          'completed_trajectories':6*len(completed),'reports':[r['label'] for r in completed]})
        if len(completed)==len(labels()):
            summary = summarize(completed)
            write_frozen(evidence/'reviews/summary.json', summary)
            write_units(summary['units'], evidence/'reviews/decomposition.csv')
    return {'collected':str(folder),'label':label,'validated_rates':6}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('generate')
    q=sub.add_parser('prepare-host');q.add_argument('--host',required=True);q.add_argument('--base',type=Path,default=DEFAULT_BASE);q.add_argument('--reused',type=Path,required=True)
    q=sub.add_parser('queue');q.add_argument('receipts',type=Path,nargs='+');q.add_argument('--output',type=Path,required=True);q.add_argument('--controller',type=Path,required=True);q.add_argument('--evidence',type=Path,required=True)
    q=sub.add_parser('collect');q.add_argument('--queue-config',type=Path,required=True);q.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='generate': result=generate()
    elif a.mode=='prepare-host': result=prepare_host(a.host,a.base,a.reused)
    elif a.mode=='queue': result=queue(a.receipts,a.output,a.controller,a.evidence)
    else: result=collect(a.queue_config,a.evidence)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
