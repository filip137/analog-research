"""Prepare and collect exp-017 native lifecycles; no experiment-specific training runtime."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from exp16_pair_rewrite import (reused_contract, single_stage, VERIFY, SPLITS,
                                lifecycle as previous_lifecycle)
from exp1e_prepare_host import command, sha256, DEFAULT_BASE
from slice_campaign import write_frozen
from workflow.lifecycle import parse_lifecycle

PREFIX = 'opt125m-cmo-l6-exp17-partner-training'
NOTE = 'campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-017-partner-training.md'
CAMPAIGN = 'opt-cmo-exp17-partner-training-20261014'
ARRAYS = (271002, 271003, 271004)
PARTNER = 'pair_rewrite_verified_partner_recovery'
ACTIVE = 'pair_rewrite_verify_recovery'
HOLD = 'pair_rewrite_verify'
AUTHORIZATION = ('User assigned implementation and execution of exp-017: std3, three arrays, healthy/random 3%, '
    'verified-partner training versus rewrite hold and original-active training; fixed exp-016 recipe. '
    'Two GPU-hours, 80 GB, 24 hours including checks and retries. Idle standing-authorized GPUs only; '
    'repository two-5090 daytime quota applies. No tuning or broader regimes.')


def lifecycle():
    doc = previous_lifecycle('std3', 'random3').to_dict()
    doc['lifecycle_id'] = PREFIX
    doc['question'].update(text='Does on-chip training through the verified healthy partner improve after pair rewriting?',
        decision='Per held-out domain, compare final KL with rewrite hold and active-only training on every array; '
                 'lower on all three is consistent improvement, higher on all three harm, otherwise mixed. ' + NOTE)
    doc['onchip']['arms'] = [a for a in doc['onchip']['arms'] if a['id'] in ('none', HOLD, ACTIVE)] + [
        {'id': PARTNER, 'weights': 'learn', 'calibration': False,
         'fault_rewrite': VERIFY, 'write_roles': 'verified_partner'}]
    return parse_lifecycle(doc)


def job_id(array):
    return f'{PREFIX}-std3-{array}'


def generate():
    lc = lifecycle(); path = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
    write_frozen(path, lc.to_dict())
    return {'path': str(path), 'sha256': lc.digest()}


def prepare_host(host, base, reused):
    if command(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        raise ValueError('Worker checkout must be clean')
    revision = command(['git', 'rev-parse', 'HEAD'], cwd=ROOT).strip()
    lc = lifecycle(); cfg = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
    if parse_lifecycle(json.loads(cfg.read_text())).digest() != lc.digest():
        raise ValueError('Lifecycle changed')
    env = {'EBL_CORPUS_ROOT': str(base/'inputs/corpora'), 'EBL_CAMPAIGN_STREAM': '1',
           'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1'}
    cases = []
    for array in ARRAYS:
        jid = job_id(array); root = ROOT/'results/lifecycles'/jid
        inputs = reused_contract(base, reused, 'std3', array, lc)
        write_frozen(root/'inputs.json', inputs)
        manifest = single_stage(lc, inputs, root, f'onchip__{inputs["hwa_arm"]}__{array}', env)
        native = root/'runs'/lc.lifecycle_id
        receipt = base/'worker-receipts'/revision/jid; receipt.parent.mkdir(parents=True, exist_ok=True)
        worker = root/'tmux-worker.json'
        write_frozen(worker, {'case_id': jid, 'state_root': str(receipt), 'cwd': str(ROOT),
            'timeout_seconds': 1800, 'lease_seconds': 900, 'env': env,
            'argv': [sys.executable, '-m', 'ebl', 'campaign', 'run', '--manifest', str(manifest),
                     '--output-dir', str(root/'runs'), '--resume', '--fail-fast']})
        cases.append({'id': jid, 'array_seed': array, 'lifecycle_sha256': lc.digest(), 'dry_run_passed': True,
            'placement': {'workdir': str(ROOT), 'run_dir': str(receipt), 'gpu_models': ['5090'],
                'argv': [sys.executable, '-m', 'campaigns.tmux_worker', 'bridge', '--spec', str(worker)],
                'validate_argv': [sys.executable, '-m', 'campaigns.recovery_collect', 'validate', str(native),
                                  '--inputs', str(root/'inputs.json')],
                'campaign_python': sys.executable, 'campaign_worker_spec': str(worker),
                'campaign_result_root': str(native), 'recovery_inputs': str(root/'inputs.json'),
                'recovery_input_contract': inputs,
                'prepared_files': [{'path': str(p), 'sha256': sha256(p)} for p in [cfg, worker, manifest, root/'inputs.json']]
                                  + list(inputs['artifacts'].values())}})
    return {'host': host, 'source': str(ROOT), 'source_revision': revision, 'base': str(base), 'cases': cases}


def queue(receipts, evidence, controller):
    from campaigns.execution_service import validate_manifest
    from exp3_lora import thresholds
    hosts = {h['host']: h for p in receipts for h in [json.loads(p.read_text())]}
    if len({h['source_revision'] for h in hosts.values()}) != 1 or len({h['source'] for h in hosts.values()}) != 1:
        raise ValueError('Require identical prepared worker revision and path')
    window = json.loads((evidence/'readiness/execution-window.json').read_text())
    cases, jobs = [], []
    out = evidence/'prepared/queue'
    for array in ARRAYS:
        jid = job_id(array); canary = array == ARRAYS[0]
        placements = {h: c['placement'] for h, r in hosts.items() for c in r['cases']
                      if c['id'] == jid and c['dry_run_passed'] and c['lifecycle_sha256'] == lifecycle().digest()}
        if not placements: raise ValueError('Missing validated placement')
        job = {'id': jid, 'project_dir': str(controller), 'handoff': NOTE, 'authorization': AUTHORIZATION,
            'peak_memory_mib': 21500, 'headroom_mib': 2048, 'expected_seconds': 650, 'budget_seconds': 2100,
            'deadline': window['deadline'], 'windows': window['windows'], 'priority': 120 if canary else 100,
            'startup_grace_seconds': 600, 'stale_seconds': 900, 'recovery': False, 'retry_safe': False,
            'placements': placements, 'completion': {'argv': [sys.executable,
                str(controller/'campaigns/opt-cmo-corruption/launch/exp17_partner_training.py'),
                'collect', '--queue-config', str(Path.home()/'.config/run-watch/queue.json'), '--evidence', str(evidence)],
                'workdir': str(controller), 'timeout_seconds': 900, 'retry_safe': True}}
        p = out/'jobs'/(jid+'.json'); write_frozen(p, job); jobs.append(str(p))
        cases.append({'id': jid, 'stage': 'readiness' if canary else 'training',
            'dependencies': [] if canary else [job_id(ARRAYS[0])],
            'required_gates': ['preflight'] if canary else ['handoff'],
            'expected_gpu_seconds': 650, 'expected_storage_bytes': 15_000_000_000,
            'progress': {**thresholds(650), 'phases': {'onchip': thresholds(650),
                         'validation': thresholds(300), 'collection': thresholds(300)}}})
    revision = next(iter(hosts.values()))['source_revision']
    manifest = {'version': 1, 'id': CAMPAIGN, 'poll_seconds': 300, 'required_adapters': ['tmux'],
        'budgets': {'gpu_hours': 2., 'storage_bytes': 80_000_000_000, 'window_seconds': 86400}, 'cases': cases,
        'storage_roots': [{'target': h, 'path': r['source']} for h, r in hosts.items()]
            + [{'target': h, 'path': str(DEFAULT_BASE/'worker-receipts'/revision)} for h in hosts]
            + [{'target': 'local', 'path': str(evidence)}]}
    validate_manifest(manifest); write_frozen(out/'campaign.json', manifest)
    return {'manifest': str(out/'campaign.json'), 'jobs': jobs}


def checked_report(array, rows, previous):
    lc = lifecycle(); expected = {(c.label, a.arm_id) for c in lc.onchip_cases for a in lc.onchip.arms}
    if len(rows) != 8 or {(r['case'], r['onchip_arm']) for r in rows} != expected:
        raise ValueError('Missing or duplicate endpoints')
    old = {(r['case'], r['arm']): r for r in previous['rows'] if r['tag'] == 'std3' and r['array_seed'] == array}
    for row in rows:
        case, arm = row['case'], row['onchip_arm']
        if row['array_seed'] != array or not row['replay_exact']:
            raise ValueError('Wrong array or failed replay')
        reference = old[case, arm if arm != PARTNER else HOLD]
        if row['initial_apparent_sha256'] != reference['initial_apparent_sha256']:
            raise ValueError('Changed original P0')
        if arm != 'none' and row['fault_rewrite'] != old[case, HOLD]['fault_rewrite']:
            raise ValueError('Changed rewrite state, map, seed or costs')
        if arm != PARTNER:
            if row['final_apparent_sha256'] != reference['final_apparent_sha256'] or row['cost'] != reference['cost']:
                raise ValueError('Existing control does not reproduce exp-016')
            for split in SPLITS:
                before, after = reference['final'][split]['teacher_kl'], row['final'][split]['teacher_kl']
                if abs(after-before) / max(abs(before), 1e-12) > 1e-6:
                    raise ValueError('Control KL does not reproduce exp-016')
        else:
            c, routing = row['cost'], row['write_routing']
            if (row['write_roles'] != 'verified_partner' or c['verify_reads'] != 0
                    or c['device_programs'] != 0 or c['max_onchip_device_write_attempts'] > 4
                    or c['active_device_writes'] + c['reference_device_writes'] != c['device_writes']
                    or c['active_write_attempts'] + c['reference_write_attempts'] != c['issued_write_attempts']
                    or c['issued_write_attempts'] != c['device_writes'] + c['stuck_write_attempts']
                    or routing['blocked_training_pairs'] != row['fault_rewrite']['controller']['flagged_both']
                    or routing['reference_training_devices'] != row['fault_rewrite']['controller']['programmed_reference']):
                raise ValueError('Partner routing or physical accounting violated')
    return {'array_seed': array, 'controls_reproduced': True, 'rows': rows}


def review(evidence):
    from campaigns.recovery_collect import checked
    previous = json.loads((evidence/'reference/exp016-summary.json').read_text())
    reports = []
    for array in ARRAYS:
        report = json.loads((evidence/'reviews'/f'{array}.json').read_text())
        folder = Path(report['collection']).parent; receipt = json.loads((folder/'collection.json').read_text())
        if any(sha256(folder/name) != digest for name, digest in receipt['files'].items()):
            raise ValueError('Collected evidence changed')
        host = json.loads((evidence/'prepared'/(receipt['target']+'.json')).read_text())
        placement = next(c['placement'] for c in host['cases'] if c['id'] == job_id(array))
        checked(json.loads((folder/'export.json').read_text()), placement['recovery_input_contract'])
        fresh = checked_report(array, json.loads((folder/'measurements.json').read_text()), previous)
        if fresh != {k:v for k,v in report.items() if k != 'collection'}:
            raise ValueError('Report differs from native evidence')
        reports.append(fresh)
    rows = [r for report in reports for r in report['rows']]
    lookup = {(r['array_seed'], r['case'], r['onchip_arm']): r for r in rows}
    decisions = []
    for case in ('nominal', 'random_30000ppm'):
        for split in SPLITS[1:]:
            for control in (HOLD, ACTIVE):
                changes = [lookup[a, case, PARTNER]['final'][split]['teacher_kl']
                           - lookup[a, case, control]['final'][split]['teacher_kl'] for a in ARRAYS]
                decisions.append({'case': case, 'domain': split, 'control': control, 'kl_changes': changes,
                    'verdict': 'improves' if all(d < 0 for d in changes) else 'harms' if all(d > 0 for d in changes) else 'mixed'})
    summary = {'coverage_complete': True, 'starting_conditions': 6, 'endpoints': 24, 'rows': rows, 'decisions': decisions}
    write_frozen(evidence/'reviews/summary.json', summary)
    from statistics import mean
    lines = ['# exp-017: verified-partner training after rewriting', '',
        'Complete: std3, three arrays, healthy/random 3%, 24 endpoints. Controls reproduce exp-016.', '',
        '| Case | Arm | Mean WikiText KL | Mean learning writes | Mean reference writes |',
        '| --- | --- | ---: | ---: | ---: |']
    for case in ('nominal', 'random_30000ppm'):
        for arm in ('none', HOLD, ACTIVE, PARTNER):
            selected = [lookup[a, case, arm] for a in ARRAYS]
            lines.append(f'| {case} | {arm} | {mean(r["final"]["transfer_wikitext"]["teacher_kl"] for r in selected):.6f} | '
                f'{mean(r["cost"].get("device_writes", 0) for r in selected):,.0f} | '
                f'{mean(r["cost"].get("reference_device_writes", 0) for r in selected):,.0f} |')
    lines += ['', *[f'- {d["case"]}, {d["domain"]}, versus {d["control"]}: **{d["verdict"]}**; KL changes {d["kl_changes"]}.' for d in decisions], '',
        'Rewrite cost is separate: one full 56.6M-device verify scan and about 1.54M programming attempts at 3%. '
        'Subsequent training adds no verify reads. Programs/writes are not measured pulses. '
        'Model-based, programmable references, no read noise/drift; no claims beyond this source, recipe and regime.', '']
    (evidence/'reviews/summary.md').write_text('\n'.join(lines))
    return {'complete': True, 'endpoints': 24, 'decisions': decisions}


def collect(queue_config, evidence):
    from campaigns.recovery_collect import collect as native_collect
    from experiments.artifacts import atomic_write_json
    from exp1f_cdt_lr import service, transition
    from campaigns.execution_service import digest
    array = int(os.environ['RUN_WATCH_JOB_ID'].split('-')[-1])
    if array not in ARRAYS: raise ValueError('Unexpected array')
    result = native_collect(queue_config, evidence/'collected'); folder = Path(result['collected'])
    report = checked_report(array, json.loads((folder/'measurements.json').read_text()),
                            json.loads((evidence/'reference/exp016-summary.json').read_text()))
    report['collection'] = str(folder/'collection.json')
    with (evidence/'.completion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = evidence/'reviews'/f'{array}.json'; write_frozen(path, report)
        if array == ARRAYS[0]:
            manifest = json.loads((evidence/'prepared/queue/campaign.json').read_text())
            gate = evidence/'readiness/handoff.json'
            write_frozen(gate, {'manifest_sha256': digest(manifest), 'passed': True, 'evidence': [str(path)]})
            service(queue_config, 'gate', CAMPAIGN, 'handoff', gate)
            for a in ARRAYS[1:]: transition(queue_config, 'release', job_id(a))
        count = sum((evidence/'reviews'/f'{a}.json').exists() for a in ARRAYS)
        atomic_write_json(evidence/'reviews/progress.json', {'completed_jobs': count, 'expected_jobs': 3,
                                                           'completed_endpoints': count*8})
        if count == 3: review(evidence)
    return {'collected': str(folder), 'array_seed': array}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='mode', required=True)
    sub.add_parser('generate')
    q = sub.add_parser('prepare-host'); q.add_argument('--host', required=True)
    q.add_argument('--base', type=Path, default=DEFAULT_BASE); q.add_argument('--reused', type=Path, required=True)
    q = sub.add_parser('queue'); q.add_argument('receipts', type=Path, nargs='+')
    q.add_argument('--evidence', type=Path, required=True); q.add_argument('--controller', type=Path, required=True)
    q = sub.add_parser('collect'); q.add_argument('--queue-config', type=Path, required=True); q.add_argument('--evidence', type=Path, required=True)
    q = sub.add_parser('review'); q.add_argument('--evidence', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'generate': result = generate()
    elif a.mode == 'prepare-host': result = prepare_host(a.host, a.base, a.reused)
    elif a.mode == 'queue': result = queue(a.receipts, a.evidence, a.controller)
    elif a.mode == 'collect': result = collect(a.queue_config, a.evidence)
    else: result = review(a.evidence)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__': main()
