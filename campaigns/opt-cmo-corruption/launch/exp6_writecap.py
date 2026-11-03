"""Matched short recovery runs with controller-owned per-device attempt caps."""
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
from exp1e_prepare_host import DEFAULT_BASE, sha256
from exp3_lora import thresholds
from exp5_threshold import prepare_host as prepare_recovery
from slice_campaign import write_frozen
from ebl.schema.lifecycle import parse_lifecycle

PREFIX = 'opt125m-cmo-l6-exp6-writecap'
PHASE = 'opt-cmo-exp6-writecap-20261012'
NOTE = 'campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-012-write-caps.md'
CAPS = (1, 2, 4, 8, 16)
DEADLINE = '2026-10-12T17:30:00+02:00'
REFERENCE_SHA256 = '7f0ff1dceb4a5c8cf6ca4f25e3ec786bd918d4a929b3b512bdcbbecb4ec99771'
AUTHORIZATION = ('User assigned reducing the maximum device writes after exp-011 and authorized idle '
    '5090 execution. Exp-012 freezes five 2.1M-token screens with caps 1/2/4/8/16 at unchanged '
    'LR 3e-3 and accumulated nominal 2-sigma, within 0.5 GPU-hours, 20 GB and 17:30 Paris.')


def labels():
    return [f'cap{cap}-dev' for cap in CAPS]


def lifecycle(label):
    if label not in labels():
        raise ValueError('Unknown write-cap case')
    cap = int(label.split('-')[0][3:])
    path = ROOT/'campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp5-threshold-accum-s2-dev.json'
    doc = json.loads(path.read_text())
    doc['lifecycle_id'] = PREFIX+'-'+label
    doc['question'].update(text='How much recovery is retained under a hard maximum of '+str(cap)+' write attempts per device?',
        decision='Report the complete cap/performance curve and retained uncapped-short gain; no automatic winner or follow-up. '+NOTE)
    doc['onchip']['max_write_attempts_per_device'] = cap
    return parse_lifecycle(doc)


def generate():
    paths = []
    for label in labels():
        lc = lifecycle(label)
        path = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
        write_frozen(path, lc.to_dict()); paths.append(str(path))
    return paths


def prepare_host(host, base, reused):
    return prepare_recovery(host, base, reused,
        prepared_lifecycles=[(label, lifecycle(label)) for label in labels()],
        screen_budget=600, screen_expected=180)


def queue(receipts, output, controller, evidence):
    hosts = {json.loads(p.read_text())['host']: json.loads(p.read_text()) for p in receipts}
    if set(hosts) != {'loulou', 'trex'} or len({h['source_revision'] for h in hosts.values()}) != 1:
        raise ValueError('Both hosts must use one prepared source revision')
    if sha256(evidence/'uncapped-reference.json') != REFERENCE_SHA256:
        raise ValueError('Uncapped short reference changed')
    jobs, cases = [], []
    for label in labels():
        lc = lifecycle(label)
        prepared = {host: next(c for c in h['cases'] if c['label'] == label) for host, h in hosts.items()}
        if any(c['lifecycle_sha256'] != lc.digest() or not c['dry_run_passed'] for c in prepared.values()):
            raise ValueError('Preparation differs from frozen cap comparison')
        c = next(iter(prepared.values()))
        job = {'id': lc.lifecycle_id, 'project_dir': str(controller), 'handoff': NOTE,
            'authorization': AUTHORIZATION, 'peak_memory_mib': 26000, 'headroom_mib': 2700,
            'expected_seconds': c['expected_seconds'], 'budget_seconds': c['budget_seconds'],
            'deadline': DEADLINE, 'priority': 110, 'startup_grace_seconds': 300,
            'stale_seconds': 600, 'recovery': False, 'retry_safe': False,
            'placements': {h: p['placement'] for h, p in prepared.items()},
            'completion': {'argv': [sys.executable, str(controller/'campaigns/opt-cmo-corruption/launch/exp6_writecap.py'),
                'collect', '--queue-config', str(Path.home()/'.config/run-watch/queue.json'),
                '--evidence', str(evidence)], 'workdir': str(controller),
                'timeout_seconds': 600, 'retry_safe': True}}
        cases.append({'id': lc.lifecycle_id, 'stage': 'readiness', 'dependencies': [],
            'required_gates': ['preflight'], 'expected_gpu_seconds': c['expected_seconds'],
            'expected_storage_bytes': 2000000000,
            'progress': {**thresholds(c['expected_seconds']), 'phases': {
                'onchip': thresholds(c['expected_seconds']),
                'validation': thresholds(180), 'collection': thresholds(180)}}})
        path = output/'jobs'/(lc.lifecycle_id+'.json')
        write_frozen(path, job); jobs.append(str(path))
    manifest = {'version': 1, 'id': PHASE, 'poll_seconds': 300, 'required_adapters': ['tmux'],
        'budgets': {'gpu_hours': .5, 'storage_bytes': 20000000000, 'window_seconds': 7200},
        'cases': cases, 'storage_roots':
            [{'target': h, 'path': p['source']} for h, p in hosts.items()] +
            [{'target': h, 'path': str(Path(p['base'])/'worker-receipts'/p['source_revision'])}
             for h, p in hosts.items()] +
            [{'target': h, 'path': str(Path(p['base'])/'exp6-source.bundle')} for h, p in hosts.items()] +
            [{'target': 'local', 'path': str(evidence)}]}
    from ebl.execution.service import validate_manifest
    validate_manifest(manifest); write_frozen(output/'campaign.json', manifest)
    return {'manifest': str(output/'campaign.json'), 'jobs': jobs}


def checked_report(label, rows, records, collection, reference):
    lc = lifecycle(label); cap = lc.onchip.max_write_attempts_per_device
    learned = next(r for r in rows if r['onchip_arm'] == 'onchip_weights')
    baseline = next(r for r in rows if r['onchip_arm'] == 'none')
    initial = baseline['final']['development']['teacher_kl']
    final = learned['final']['development']['teacher_kl']
    reference_gain = initial - reference['development_kl']
    cost = learned['cost']
    if (learned['sequence_presentations'] != 8192 or reference['sequence_presentations'] != 8192
            or abs(initial - reference['initial_development_kl']) > 1e-12
            or not all(math.isfinite(v) for v in [initial, final, reference_gain]) or reference_gain <= 0):
        raise ValueError('Invalid matched short-window evidence')
    if (not 0 <= cost['max_onchip_device_writes'] <= cost['max_onchip_device_write_attempts'] <= cap
            or cost['issued_write_attempts'] != cost['device_writes'] + cost['stuck_write_attempts']):
        raise ValueError('Per-device attempt cap or accounting violated')
    memory = max(m['peak_reserved_bytes'] for m in records if m.get('stage') == 'onchip_memory')
    if memory > 26000*1024**2:
        raise ValueError('Measured CUDA memory exceeds reservation')
    return {'label': label, 'max_write_attempts_per_device': cap,
        'learning_rate': .003, 'threshold_sigma': 2., 'accumulate_updates': True,
        'sequence_presentations': learned['sequence_presentations'],
        'initial_development_kl': initial, 'development_kl': final,
        'uncapped_short_development_kl': reference['development_kl'],
        'retained_development_gain': (initial-final)/reference_gain,
        'cost': cost, 'fraction_devices_capped': cost['capped_devices']/lc.network.analog_weights,
        'peak_reserved_bytes': memory, 'collection': str(collection),
        'reference_sha256': REFERENCE_SHA256,
        'effects': {s: {'deployment': baseline['final'][s], 'onchip': learned['final'][s],
            'relative_kl_reduction': 1-learned['final'][s]['teacher_kl']/baseline['final'][s]['teacher_kl']}
            for s in ('development', 'transfer_generated', 'transfer_wikitext')}}


def collect(queue_config, evidence):
    from ebl.execution.recovery_collect import collect as native_collect
    label = os.environ['RUN_WATCH_JOB_ID'].removeprefix(PREFIX+'-')
    if label not in labels():
        raise ValueError('Unexpected write-cap completion job')
    reference_path = evidence/'uncapped-reference.json'
    if sha256(reference_path) != REFERENCE_SHA256:
        raise ValueError('Uncapped short reference changed')
    reference = json.loads(reference_path.read_text())
    result = native_collect(queue_config, evidence/'collected')
    folder = Path(result['collected'])
    rows = json.loads((folder/'measurements.json').read_text())
    records = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
    report = checked_report(label, rows, records, folder/'collection.json', reference)
    with (evidence/'.completion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_frozen(evidence/'reviews'/(label+'.json'), report)
        paths = [evidence/'reviews'/(name+'.json') for name in labels()]
        if all(path.exists() for path in paths):
            summary = {'coverage_complete': True, 'reference_sha256': REFERENCE_SHA256,
                'reports': [json.loads(path.read_text()) for path in paths],
                'selection': 'Complete descriptive curve; no automatic winner or follow-up.'}
            write_frozen(evidence/'reviews/summary.json', summary)
    result['review'] = report
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('generate')
    p = sub.add_parser('prepare-host'); p.add_argument('--host', required=True)
    p.add_argument('--base', type=Path, default=DEFAULT_BASE); p.add_argument('--reused', type=Path, required=True)
    p = sub.add_parser('queue'); p.add_argument('receipts', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--controller', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p = sub.add_parser('collect'); p.add_argument('--queue-config', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'generate': result = generate()
    elif args.mode == 'prepare-host': result = prepare_host(args.host, args.base, args.reused)
    elif args.mode == 'queue': result = queue(args.receipts, args.output, args.controller, args.evidence)
    else: result = collect(args.queue_config, args.evidence)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
