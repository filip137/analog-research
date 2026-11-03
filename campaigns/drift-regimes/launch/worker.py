"""Subprocess orchestration of native drift-regimes plans on Jean Zay; no numerical runner.

Adapted from the drift-mitigation exp-002 worker. Each manifest group
(``canary`` or one regime) has its own output root, so groups run as separate
dev-queue jobs. ``--prepare`` composes the native plans of a group's
lifecycles, sharing one prepare/devices pair, and dry-runs them. Without it, the worker runs the
merged campaign, collects it, inspects every bundle and checks coverage.
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


def slug(regime):
    return regime.replace('.', 'p').replace('_', '-').lower()


def combine_plans(plans, campaign_id):
    """Keep one shared prepare/devices pair; prefix each regime's stages."""
    result = {'schema_version': 1, 'campaign_id': campaign_id, 'targets': None, 'stages': []}
    for i, (regime, path) in enumerate(plans):
        doc = read(path / 'campaign.json')
        if result['targets'] is None:
            result['targets'] = doc['targets']
        if result['targets'] != doc['targets']:
            raise ValueError('Regime plans disagree on the execution target.')
        for original in doc['stages']:
            stage = dict(original)
            if stage['id'] in ('prepare', 'devices'):
                if i:
                    continue
            else:
                if not stage['id'].startswith('drift_regimes__'):
                    raise ValueError(f"Unexpected stage in a drift-regimes plan: {stage['id']}")
                stage['id'] = slug(regime) + '__' + stage['id']
                stage['case_id'] = slug(regime) + '__' + stage['case_id']
            stage['config'] = str((path / stage['config']).resolve())
            result['stages'].append(stage)
    ids = [stage['id'] for stage in result['stages']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate native stage.')
    return result


def validate_coverage(rows, specs):
    expected = {(s['regime'], seed, t) for s in specs for seed in s['assignment_seeds'] for t in s['evaluation_seconds']}
    observed = [(r['regime'], r['realization'], r['seconds']) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('Missing, duplicate or unexpected regime/realization/age coverage.')
    by_regime = {s['regime']: s for s in specs}
    for row in rows:
        s = by_regime[row['regime']]
        for field in ('source_sha256', 'cohort_sha256', 'canary', 'law'):
            if row[field] != s[field]:
                raise ValueError(f'Coverage identity mismatch: {field}.')
        if row['metrics']['sequences'] != s['evaluation_sequences']:
            raise ValueError('Wrong evaluation sequence count.')
        if not all(math.isfinite(row['metrics'][k]) for k in ('teacher_kl', 'perplexity', 'cross_entropy')):
            raise ValueError('Nonfinite metrics.')


def call(*command):
    print(json.dumps({'command': list(map(str, command))}), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('mode')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    m = read(args.manifest)
    if args.mode not in m['configs']:
        raise ValueError(f'Unknown group {args.mode!r}.')
    source = Path(m['source'])
    os.chdir(source)
    os.environ['EBL_CORPUS_ROOT'] = m['corpus_root']
    receipt = source / 'source_snapshot.json'
    for name, sha in read(receipt)['files'].items():
        if digest(source / name) != sha:
            raise ValueError(f'Staged code changed: {name}')
    for key in ('teacher', 'weights'):
        if digest(m[key]) != m[key + '_sha256']:
            raise ValueError(f'Pinned {key} changed.')
    for name, sha in m['corpora'].items():
        if digest(Path(m['corpus_root']) / f'{name}.pt') != sha:
            raise ValueError(f'Pinned corpus changed: {name}')
    os.environ['EBL_SOURCE_RECEIPT'] = str(receipt)
    root = Path(m['output']) / args.mode
    root.mkdir(parents=True, exist_ok=True)
    configs = [source / p for p in m['configs'][args.mode]]
    specs = [read(p)['drift_regimes'] for p in configs]
    manifest = root / 'plan/campaign.json'
    if args.prepare:
        plans = []
        for config, spec in zip(configs, specs):
            plan = root / 'plans' / slug(spec['regime'])
            call(sys.executable, '-m', 'workflow', 'plan', config,
                 '--teacher-weights', m['teacher'], '--hwa-source', 'standard_hwa_model=' + m['weights'],
                 '--output', plan, '--device', 'cuda:0', '--cpu-threads', '10')
            plans.append((spec['regime'], plan))
        merged = combine_plans(plans, 'opt125m-drift-regimes-qual-' + args.mode)
        manifest.parent.mkdir(exist_ok=True)
        if manifest.exists() and read(manifest) != merged:
            raise ValueError('Refusing to replace a different native campaign plan.')
        write(manifest, merged)
        call(sys.executable, '-m', 'ebl', 'campaign', 'run', '--manifest', manifest,
             '--output-dir', root / 'runs', '--dry-run')
        return
    with (root / f"started-{os.environ.get('SLURM_JOB_ID', 'local')}.json").open('x') as stream:
        json.dump({'job_id': os.environ.get('SLURM_JOB_ID'), 'started': time.time(),
                   'manifest_sha256': digest(args.manifest), 'source_sha256': digest(receipt)}, stream)
    try:
        call(sys.executable, '-m', 'ebl', 'campaign', 'run', '--manifest', manifest,
             '--output-dir', root / 'runs', '--resume')
        call(sys.executable, '-m', 'workflow', 'collect', root / 'runs', '--output', root / 'collected')
        paths = sorted((root / 'runs').glob('**/drift_regimes_measurements.json'))
        rows, summaries = [], []
        for p in paths:
            call(sys.executable, '-m', 'ebl', 'runs', 'inspect', p.parent, '--verify-artifacts', '--require-complete')
            rows.extend(read(p))
            summary = read(p.parent / 'drift_regimes_summary.json')
            if not summary['source_unchanged'] or not summary['replay_exact']:
                raise ValueError('Source preservation or release-state replay failed.')
            summary['run_dir'] = str(p.parent)
            summaries.append(summary)
        validate_coverage(rows, specs)
        expected = sum(len(s['assignment_seeds']) for s in specs)
        if len(paths) != expected:
            raise ValueError('Wrong independent native trajectory count.')
        write(root / 'summaries.json', summaries)
        write(root / 'complete.json', {'schema': 'drift_regimes.qualification.complete.v1', 'mode': args.mode,
              'rows': len(rows), 'trajectories': len(paths), 'source_sha256': digest(receipt),
              'manifest_sha256': digest(args.manifest), 'summary_sha256': digest(root / 'summaries.json'),
              'completed': time.time()})
    except BaseException as error:
        write(root / f"failed-{os.environ.get('SLURM_JOB_ID', 'local')}.json", {'error': repr(error), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
