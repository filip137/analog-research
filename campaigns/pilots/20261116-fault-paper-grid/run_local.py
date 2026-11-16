"""Guarded foreground transport into the existing native DRN CLI (no numerics).

The shared GPU queue owns detachment, timeout, process identity and exit receipts.
Use --prepare for CPU-only source/input/config validation; --verify for collection.
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
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def verify(config, run):
    from importlib.util import module_from_spec, spec_from_file_location
    root = Path(__file__).resolve().parents[3]
    spec = spec_from_file_location('verify_native', root / 'campaigns/pilots/20261115-fault-tuning-completion/verify_unit.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify(config, run)
    c = read(config)
    if c['stage'] == 'fault_tuning':
        expected = c['fault_tuning'].get('evaluation_cohorts', [])
        summary = read(Path(run) / 'artifacts/summary.json')
        for row in summary['arms']:
            actual = row.get('held_out', {})
            if set(actual) != {x['name'] for x in expected}:
                raise ValueError('Missing or unexpected held-out domains.')
            for cohort in expected:
                score = actual[cohort['name']]
                if score['cohort'] != cohort or not math.isfinite(score['metrics']['teacher_kl']):
                    raise ValueError('Held-out identity mismatch or nonfinite KL.')
                if score['metrics']['scored_tokens'] != cohort['sequences'] * 255:
                    raise ValueError('Incomplete held-out scored-token coverage.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    m = read(args.manifest)
    source = Path(m['source'])
    os.chdir(source)
    sys.path.insert(0, str(source))
    for path, sha in m['pinned_files'].items():
        if digest(path) != sha:
            raise ValueError('Changed input/config: ' + path)
    receipt = source / 'source_snapshot.json'
    if digest(receipt) != m['source_receipt_sha256']:
        raise ValueError('Changed source receipt.')
    for name, sha in read(receipt)['files'].items():
        if digest(source / name) != sha:
            raise ValueError('Changed staged source: ' + name)
    from ebl.schema.opt_cmo_drn import parse_config
    config = Path(m['config'])
    parsed = parse_config(read(config))
    if parsed.analog_layers not in (6, 12):
        raise ValueError('Unexpected analog depth.')
    if args.prepare:
        print(json.dumps({'prepared': True, 'manifest_sha256': digest(args.manifest), 'gpu_used': False}))
        return
    out = Path(m['output'])
    run = out / 'run'
    def call(*argv):
        print(json.dumps({'argv': list(map(str, argv))}), flush=True)
        subprocess.run(list(map(str, argv)), check=True)
    if args.verify:
        call(sys.executable, '-m', 'ebl', 'runs', 'inspect', run, '--verify-artifacts', '--require-complete')
        print(json.dumps(verify(config, run)))
        return
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    (out / 'started.json').write_text(json.dumps({'started': started, 'pid': os.getpid(),
                                                'manifest_sha256': digest(args.manifest)}) + '\n')
    os.environ.update(EBL_CORPUS_ROOT=m['corpus_root'], OPT_DRN_RUN_ID='run',
                      OPT_DRN_SOURCE_SNAPSHOT=str(receipt), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(parsed.cpu_threads)
    argv = [sys.executable, '-m', 'ebl', 'train', '--config', config, '--output-dir', out,
            '--weights', m['weights'], '--teacher-weights', m['teacher']]
    if m.get('device_state'):
        argv += ['--device-state', m['device_state']]
    try:
        call(*argv)
        call(sys.executable, '-m', 'ebl', 'runs', 'inspect', run, '--verify-artifacts', '--require-complete')
        result = verify(config, run)
        result.update(seconds=time.time()-started, manifest_sha256=digest(args.manifest))
        (run / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    except BaseException as error:
        (out / 'failed.json').write_text(json.dumps({'error': repr(error), 'seconds': time.time()-started})+'\n')
        raise


if __name__ == '__main__':
    main()
