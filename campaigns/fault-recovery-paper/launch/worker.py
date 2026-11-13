"""Subprocess orchestration of one fault-tuning unit on Jean Zay; no numerical code.

Adapted from the drift-regimes worker. ``--prepare`` (login node) plans the unit's
lifecycle with the frozen std3 source and dry-runs it. Without it (in the job) the worker
verifies the staged source and pinned inputs, runs the native campaign, inspects every
bundle and checks the arm coverage, then writes ``complete.json`` (or ``failed-<job>.json``).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def call(*command):
    print(json.dumps({'command': list(map(str, command))}), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('unit')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    m = read(args.manifest)
    if args.unit not in m['units']:
        raise ValueError(f'Unknown unit {args.unit!r}.')
    source = Path(m['source'])
    os.chdir(source)
    os.environ['EBL_CORPUS_ROOT'] = m['corpus_root']
    os.environ['EBL_CAMPAIGN_STREAM'] = '1'
    receipt = source / 'source_snapshot.json'
    for name, sha in read(receipt)['files'].items():
        if digest(source / name) != sha:
            raise ValueError(f'Staged code changed: {name}')
    for key in ('teacher', 'weights'):
        if digest(m[key]) != m[key + '_sha256']:
            raise ValueError(f'Pinned {key} changed.')
    for name, sha in m['pinned'].items():
        if digest(Path(m['corpus_root']) / f'{name}.pt') != sha:
            raise ValueError(f'Pinned input changed: {name}')
    os.environ['EBL_SOURCE_RECEIPT'] = str(receipt)
    root = Path(m['output']) / args.unit
    root.mkdir(parents=True, exist_ok=True)
    config = source / m['units'][args.unit]
    spec = read(config)['fault_tuning']
    plan = root / 'plan'
    if args.prepare:
        call(sys.executable, '-m', 'ebl', 'lifecycle', 'plan', config, '--teacher-weights', m['teacher'],
             '--hwa-source', 'standard_hwa_model=' + m['weights'], '--output', plan,
             '--device', 'cuda:0', '--cpu-threads', '10', '--python', sys.executable)
        call(sys.executable, '-m', 'ebl', 'campaign', 'run', '--manifest', plan / 'campaign.json',
             '--output-dir', root / 'runs', '--dry-run')
        return
    job = os.environ.get('SLURM_JOB_ID', 'local')
    with (root / f'started-{job}.json').open('x') as stream:
        json.dump({'job_id': job, 'started': time.time(), 'manifest_sha256': digest(args.manifest),
                   'source_sha256': digest(receipt)}, stream)
    try:
        call(sys.executable, '-m', 'ebl', 'campaign', 'run', '--manifest', plan / 'campaign.json',
             '--output-dir', root / 'runs', '--resume')
        paths = sorted((root / 'runs').glob('**/fault_tuning__*/attempt-*/runs/*/measurements.json'))
        complete = []
        for p in paths:
            if (p.parent / 'result.json').exists() and read(p.parent / 'result.json').get('status') in (None, 'complete', 'completed'):
                complete.append(p)
        if not complete:
            raise ValueError('No completed fault-tuning bundle.')
        run = complete[-1].parent
        call(sys.executable, '-m', 'ebl', 'runs', 'inspect', run, '--verify-artifacts', '--require-complete')
        rows = read(run / 'measurements.json')
        labels = [r['label'] for r in rows]
        if labels != [a['label'] for a in spec['arms']]:
            raise ValueError('Missing, duplicate or unexpected arms.')
        if not all(math.isfinite(r['final']['teacher_kl']) for r in rows):
            raise ValueError('Nonfinite development KL.')
        summary = read(run / 'summary.json')
        write(root / 'complete.json', {'schema': 'fault_tuning.unit.complete.v1', 'unit': args.unit, 'run': str(run),
              'arms': {r['label']: r['final']['teacher_kl'] for r in rows},
              'samples_sha256': summary['samples_sha256'], 'apparent_sha256': summary['apparent_sha256'],
              'gpu': summary['gpu'], 'source_sha256': digest(receipt), 'manifest_sha256': digest(args.manifest),
              'completed': time.time()})
    except BaseException as error:
        write(root / f'failed-{job}.json', {'error': repr(error), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
