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


def verify_source(manifest, *, executed_worker=None):
    """Bind the launch-directory worker copy and optional pinned receipt to source."""
    source = Path(manifest['source']).resolve()
    receipt = source / 'source_snapshot.json'
    expected = manifest.get('source_receipt_sha256')
    if expected is not None and digest(receipt) != expected:
        raise ValueError('Pinned source receipt changed.')
    tracked_worker = source / 'campaigns/fault-recovery-paper/launch/worker.py'
    executing = Path(__file__) if executed_worker is None else Path(executed_worker)
    if digest(executing) != digest(tracked_worker):
        raise ValueError('Executed worker copy differs from the tracked source worker.')
    for name, sha in read(receipt)['files'].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or digest(path) != sha:
            raise ValueError(f'Staged code changed: {name}')
    return source, receipt


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_drift_measurements(spec, rows):
    """Rows of the ``drift`` block: native arm order, full grid coverage, finite KL, every update ran."""
    order = sorted(spec['arms'], key=lambda a: 2 if a['method'] != 'hold_fault_aware' else int(bool(a.get('no_drift'))))
    if [row.get('label') for row in rows] != [arm['label'] for arm in order]:
        raise ValueError('Missing, duplicate or unexpected arms.')
    ages, steps = spec['drift']['evaluation_seconds'], spec['schedule']['steps']
    for arm, row in zip(order, rows):
        grid, losses = row.get('grid'), row.get('train_losses', [])
        if row.get('method') != arm['method']:
            raise ValueError(f"Wrong method for arm {arm['label']}.")
        if (not isinstance(grid, list) or [point.get('age_seconds') for point in grid] != ages
                or not all(finite(point.get('teacher_kl')) for point in grid) or not finite(row.get('mean_teacher_kl'))
                or row.get('final') != grid[-1].get('development')):
            raise ValueError(f"Incomplete or nonfinite drift grid for arm {arm['label']}.")
        if not isinstance(losses, list) or (len(losses) != (0 if arm['method'] == 'hold_fault_aware' else steps)
                                            or not all(finite(loss) for loss in losses)):
            raise ValueError(f"Incomplete training updates for arm {arm['label']}.")
    if len({row['grid'][0]['teacher_kl'] for row in rows}) != 1:
        raise ValueError('Arms do not start from one released deployment.')


def validate_measurements(spec, rows, *, sequence_length=256):
    """Fail closed before publishing completion, including endpoint-only test scores."""
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Expected native measurement rows.')
    if 'drift' in spec:
        return validate_drift_measurements(spec, rows)
    labels = [row.get('label') for row in rows]
    if (labels != [arm['label'] for arm in spec['arms']]
            or len(set(labels)) != len(labels)):
        raise ValueError('Missing, duplicate or unexpected arms.')
    cohorts = spec.get('evaluation_cohorts', [])
    names = [cohort['name'] for cohort in cohorts]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate configured evaluation cohort names.')
    for arm, row in zip(spec['arms'], rows):
        if row.get('method') != arm['method']:
            raise ValueError(f"Wrong method for arm {arm['label']}.")
        if not isinstance(row.get('final'), dict) or not finite(row['final'].get('teacher_kl')):
            raise ValueError('Nonfinite or missing development KL.')
        losses, curve = row.get('train_losses', []), row.get('curve')
        if (not isinstance(losses, list) or not isinstance(curve, list)
                or any(not isinstance(point, dict) for point in curve)):
            raise ValueError('Missing training/curve records.')
        if arm['method'] in ('hold_naive', 'hold_fault_aware'):
            if losses or curve:
                raise ValueError('A hold must have zero training updates.')
        else:
            steps = spec['schedule']['steps']
            if (len(losses) != steps or not all(finite(loss) for loss in losses)
                    or not curve or curve[-1].get('step') != steps):
                raise ValueError('Incomplete training updates or final curve endpoint.')
            if (row['final'] != curve[-1].get('development')
                    or not all(finite(point.get('development', {}).get('teacher_kl')) for point in curve)):
                raise ValueError('Invalid development curve or mismatched final endpoint.')
        held = row.get('held_out', [])
        if not isinstance(held, list) or any(not isinstance(point, dict) for point in held):
            raise ValueError('Expected named held_out measurement records.')
        observed = [point.get('name') for point in held]
        if (len(observed) != len(names) or any(not isinstance(name, str) for name in observed)
                or set(observed) != set(names) or len(set(observed)) != len(observed)):
            raise ValueError('Missing, duplicate or unexpected held-out cohorts.')
        by_name = {point['name']: point for point in held}
        for cohort in cohorts:
            point = by_name[cohort['name']]
            if any(point.get(key) != value for key, value in cohort.items()):
                raise ValueError(f"Held-out identity mismatch: {cohort['name']}.")
            scores = point.get('scores', {})
            if (not isinstance(scores, dict) or not finite(scores.get('teacher_kl'))
                    or scores.get('sequences') != cohort['sequences']
                    or scores.get('scored_tokens') != cohort['sequences'] * (sequence_length - 1)):
                raise ValueError(f"Invalid held-out scores/counts: {cohort['name']}.")
            if point.get('state_unchanged') is not True:
                raise ValueError(f"Held-out scoring changed state: {cohort['name']}.")


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
    source, receipt = verify_source(m)
    os.chdir(source)
    os.environ['EBL_CORPUS_ROOT'] = m['corpus_root']
    os.environ['EBL_CAMPAIGN_STREAM'] = '1'
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
    config_doc = read(config)
    spec = config_doc['fault_tuning']
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
        validate_measurements(spec, rows, sequence_length=config_doc['data']['sequence_length'])
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
