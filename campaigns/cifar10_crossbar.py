"""Order native EBL subprocesses and bind their exact completed artifacts.

Run from the repository root with the CUDA Python interpreter. No numerical
training or deployment operation belongs in this orchestration module.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.artifacts import atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'results/cifar10-hwa-recovery-20260917-v1'
TEACHER = ROOT / 'results/cifar10-digital-relu-pretrain-20260917-v1/runs/digital/20260917T105249.415682Z-5fd72081-a5f3b148/checkpoints/teacher.pt'
TEACHER_SHA256 = 'ee173c2accad7ae9490613b0a629e7cebf7beb9900f2bfd44cd199ac7c7ddaa4'


def read(path):
    return json.loads(Path(path).read_text())


def completed(arm, inputs):
    output = STUDY / 'runs' / arm
    pointer = output / 'completed_run.json'
    if not pointer.exists():
        return None
    run = Path(read(pointer)['run_dir'])
    if run.parent != output:
        raise ValueError('Expected the completion pointer inside the declared study arm.')
    result, manifest = read(run / 'result.json'), read(run / 'manifest.json')
    if result['status'] != 'complete' or read(run / 'status.json')['status'] != 'complete':
        raise ValueError('Expected a semantically complete native run.')
    config_path = ROOT / f'examples/cifar10_crossbar/{arm}.json'
    if read(run / 'config.resolved.json') != read(config_path):
        raise ValueError('Expected the original declared scientific config.')
    expected_inputs = {key.replace('-', '_'): sha256_file(Path(value)) for key, value in inputs.items()}
    if {v['role']: v['sha256'] for v in manifest['inputs']} != expected_inputs:
        raise ValueError('Expected the exact input artifacts for the completed run.')
    for artifact in result['artifacts']:
        path = run / artifact['path']
        if not path.is_file() or path.stat().st_size != artifact['size_bytes'] or sha256_file(path) != artifact['sha256']:
            raise ValueError(f'Expected an intact native artifact: {path}')
    return run, result


def run_arm(arm, inputs, ledger):
    existing = completed(arm, inputs)
    if existing is not None:
        ledger['completed'][arm] = str(existing[0])
        print(f'reuse verified {arm}', flush=True)
        return existing
    output = STUDY / 'runs' / arm
    command = [sys.executable, '-u', '-m', 'ebl', 'train', '--config',
               str(ROOT / f'examples/cifar10_crossbar/{arm}.json'), '--output-dir', str(output)]
    for key, value in inputs.items():
        command += [f'--{key}', str(value)]
    log_path = output / f'launcher-{time.time_ns()}.log'
    print(f'launch {arm}: {log_path}', flush=True)
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        ledger.update(state='running', active_arm=arm, child_pid=process.pid, command=command, log=str(log_path))
        while True:
            ledger.update(time=time.time())
            atomic_write_json(STUDY / 'campaign.json', ledger)
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                pass
    if code:
        ledger.update(state='failed', exit_code=code, time=time.time())
        atomic_write_json(STUDY / 'campaign.json', ledger)
        raise RuntimeError(f'Native arm {arm} failed with exit {code}; see {log_path}')
    verified = completed(arm, inputs)
    if verified is None:
        raise RuntimeError(f'Native arm {arm} exited without terminal artifacts.')
    ledger['completed'][arm] = str(verified[0])
    print(f'complete {arm}', flush=True)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if sha256_file(TEACHER) != TEACHER_SHA256:
        raise ValueError('Expected the frozen validation-selected digital teacher hash.')
    if not (STUDY / 'study.json').is_file():
        raise ValueError('Prepare the declared native study before launching its campaign.')
    ledger = dict(schema='cifar10.campaign.v1', pid=os.getpid(), started=time.time(),
                  teacher=str(TEACHER), teacher_sha256=TEACHER_SHA256, completed={})
    candidates = []
    for arm in ('hwa_n0', 'hwa_n025', 'hwa_n05', 'hwa_n1'):
        run, result = run_arm(arm, {'teacher-weights': TEACHER}, ledger)
        metric = result['metrics']
        selected = metric['selected']
        candidates.append(dict(arm=arm, run=str(run), master=str(run / 'checkpoints/hwa.pt'),
                               master_sha256=sha256_file(run / 'checkpoints/hwa.pt'),
                               noise_multiplier=metric['noise_multiplier'], epoch=selected['epoch'],
                               validation=selected['validation']))
    winner = min(candidates, key=lambda c: (c['validation']['kl_teacher_student'],
                 -c['validation']['accuracy'], c['epoch'], c['noise_multiplier']))
    receipt = dict(schema='cifar10.hwa_selection.v1', teacher_sha256=TEACHER_SHA256,
                   criterion='healthy_development_validation_KL_then_accuracy_then_epoch_then_noise',
                   test_used=False, candidates=candidates, selected=winner)
    receipt_path = STUDY / 'hwa_selection.json'
    if receipt_path.exists() and read(receipt_path) != receipt:
        raise ValueError('A prior frozen HWA selection disagrees with the completed candidates.')
    atomic_write_json(receipt_path, receipt)
    print(f"selected HWA {winner['arm']}, epoch {winner['epoch']}", flush=True)
    for array in ('dev', 'a1', 'a2'):
        for source in ('direct', 'hwa'):
            inputs = {'teacher-weights': TEACHER}
            if source == 'hwa':
                inputs['weights'] = winner['master']
            run, _ = run_arm(f'{source}_{array}', inputs, ledger)
            if source == 'hwa':
                deployment = run / 'checkpoints/deployment.pt'
        for condition in ('healthy', 'faulted'):
            for writer in ('open_loop', 'closed_loop_pv'):
                run_arm(f'recover_{array}_{condition}_{writer}',
                        {'teacher-weights': TEACHER, 'device-state': deployment}, ledger)
    subprocess.run([sys.executable, '-m', 'ebl', 'study', 'summarize', '--study-dir', str(STUDY),
                    '--verify-artifacts'], cwd=ROOT, check=True)
    ledger.update(state='complete', active_arm=None, child_pid=None, time=time.time())
    atomic_write_json(STUDY / 'campaign.json', ledger)
    print('All 22 declared hardware arms completed and artifacts verified.', flush=True)


if __name__ == '__main__':
    main()
