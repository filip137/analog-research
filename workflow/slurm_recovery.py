"""Slurm transport for the fixed-LR DRN recovery extension; numerics enter ebl."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED'}
TIMEOUT_MESSAGE = 'Recovery wall-time budget reached; complete continuation state preserved.'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def lock(path, *, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def output(argv):
    return subprocess.check_output([str(a) for a in argv], text=True, stderr=subprocess.PIPE)


def task_index(environment, explicit=None, query=output):
    value = explicit if explicit is not None else environment.get('SLURM_ARRAY_TASK_ID')
    if value is None and environment.get('SLURM_JOB_ID'):
        control = query(['scontrol', 'show', 'job', '-o', environment['SLURM_JOB_ID']])
        match = re.search(r'\bArrayTaskId=(\d+)(?:\s|$)', control)
        if match:
            value = match.group(1)
    if value is None:
        raise ValueError('Missing task index; pass --index or SLURM_ARRAY_TASK_ID.')
    index = int(value)
    if index < 0:
        raise ValueError('Negative task index.')
    return index


def accounting(text, parent, indices):
    """Require exact array coverage, including Slurm singleton JobIDRaw aliases."""
    tasks = {}
    parent_rows = []
    for line in text.splitlines():
        fields = [value.strip() for value in line.split('|')]
        if len(fields) != 9:
            raise ValueError('Unexpected sacct field count.')
        raw, job, state, exit_code, account, partition, qos, constraint, elapsed = fields
        if '.' in job:
            continue
        if job == str(parent):
            parent_rows.append(fields)
            continue
        prefix = str(parent) + '_'
        if not job.startswith(prefix) or not job[len(prefix):].isdigit():
            raise ValueError('Unexpected allocation row: ' + job)
        index = int(job[len(prefix):])
        if index in tasks:
            raise ValueError('Duplicate Slurm task row.')
        tasks[index] = dict(job_id=job, raw_id=raw, state=state.split()[0].rstrip('+'),
                            exit_code=exit_code, account=account, partition=partition,
                            qos=qos, constraint=constraint, elapsed_seconds=int(elapsed))
    if len(parent_rows) > 1 or set(tasks) != set(indices):
        raise ValueError('Missing, duplicate or unexpected Slurm array coverage.')
    return tasks


def scheduler_rows(parent, indices):
    fields = 'JobIDRaw,JobID%40,State,ExitCode,Account,Partition,QOS,Constraints,ElapsedRaw'
    return accounting(output(['sacct', '-n', '-P', '-j', parent, '--format=' + fields]), parent, indices)


def contract_rows(rows, manifest):
    expected = {k: manifest[k] for k in ('account', 'partition', 'qos', 'constraint')}
    for row in rows.values():
        if {k: row[k] for k in expected} != expected:
            raise ValueError('Slurm allocation differs from the frozen resource contract.')


def live_job(parent, format='%T'):
    """An aged-out squeue ID is absent; other scheduler errors remain fatal."""
    try:
        return output(['squeue', '-h', '-j', parent, '-o', format]).strip()
    except subprocess.CalledProcessError as error:
        if 'Invalid job id specified' in (error.stderr or ''):
            return ''
        raise


def command(manifest_path, manifest, indices, *, canary=False, wave=0):
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Expected distinct array indices.')
    if canary and indices != [0]:
        raise ValueError('Canary is a singleton 0-0 array.')
    if not canary and not set(indices) <= set(range(len(manifest['cases']))):
        raise ValueError('Unexpected production task.')
    array = '0-0' if canary else ','.join(map(str, sorted(indices)))
    if not canary:
        array += '%' + str(manifest['concurrency'])
    root = Path(manifest['output'])
    return ['sbatch', '--parsable', '--account=' + manifest['account'],
            '--partition=' + manifest['partition'], '--qos=' + manifest['qos'],
            '--constraint=' + manifest['constraint'], '--nodes=1', '--ntasks=1',
            '--gres=gpu:1', '--cpus-per-task=' + str(manifest['cpus']),
            '--hint=nomultithread', '--time=' + manifest['walltime'],
            '--array=' + array, '--job-name=' + manifest['job_prefix'] + ('canary' if canary else f'w{wave}'),
            '--output=' + str(root / 'slurm/%x-%A_%a.out'),
            '--error=' + str(root / 'slurm/%x-%A_%a.err'),
            '--export=ALL', str(Path(manifest['source']) / 'launch/recovery.slurm'),
            str(manifest_path), 'canary' if canary else 'production']


def verify_source(manifest):
    root = Path(manifest['source'])
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError('Source/config/wrapper digest differs: ' + name)


def run_native(manifest, config, destination, inputs, *, action='train', progress=None):
    destination = Path(destination)
    run = destination / 'run'
    if run.exists():
        raise ValueError('Refusing to overwrite a native run: ' + str(run))
    destination.mkdir(parents=True)
    argv = [sys.executable, '-m', 'ebl', action, '--config', str(Path(manifest['source']) / config),
            '--output', str(destination), '--teacher-weights', manifest['teacher']]
    for role, path in inputs.items():
        argv += ['--' + role.replace('_', '-'), str(path)]
    if progress is not None:
        progress(str(run))
    with (destination / 'launcher.log').open('w') as log:
        result = subprocess.run(argv, cwd=manifest['source'], stdout=log, stderr=subprocess.STDOUT)
    return run, result.returncode


def inspect(run, *, complete=True):
    argv = [sys.executable, '-m', 'ebl', 'runs', 'inspect', str(run), '--verify-artifacts', '--json']
    if complete:
        argv.append('--require-complete')
    with (Path(run).parent / 'inspection.json').open('w') as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)


def artifact(run, kind):
    matches = [a for a in read(Path(run) / 'result.json')['artifacts'] if a['kind'] == kind]
    if len(matches) != 1:
        raise ValueError('Expected exactly one native ' + kind)
    item = matches[0]
    path = Path(run) / item['path']
    if digest(path) != item['sha256']:
        raise ValueError('Native artifact digest differs.')
    return str(path)


def checkpointed(run):
    status = read(Path(run) / 'status.json')
    if status.get('error') != {'type': 'TimeoutError', 'message': TIMEOUT_MESSAGE}:
        raise ValueError('Training failed for a reason other than the declared checkpoint boundary.')
    rows = [json.loads(line) for line in (Path(run) / 'metrics.jsonl').read_text().splitlines()]
    saves = [r for r in rows if r['event'] == 'recovery_checkpoint']
    if not saves or saves[-1]['step'] <= 0:
        raise ValueError('Continuation did not make positive progress.')
    path = Path(run) / 'checkpoints/latest.pt'
    return {'path': str(path), 'sha256': digest(path), 'step': saves[-1]['step']}


def baseline(manifest, rate, progress):
    root = Path(manifest['output']) / 'baselines' / rate
    with lock(root / 'baseline.lock'):
        receipt = root / 'baseline.json'
        if receipt.exists():
            saved = read(receipt)
            if digest(saved['device_state']) != saved['device_state_sha256']:
                raise ValueError('Baseline state changed.')
            return saved
        configs = manifest['baselines'][rate]
        run, code = run_native(manifest, configs['deploy'], root / 'deploy', {'weights': manifest['weights']}, progress=progress)
        if code:
            raise ValueError('Deployment failed: ' + str(run))
        inspect(run)
        deployed = artifact(run, 'device_state')
        hold, code = run_native(manifest, configs['hold'], root / 'hold',
                                {'weights': manifest['weights'], 'device_state': deployed}, progress=progress)
        if code:
            raise ValueError('Hold failed: ' + str(hold))
        inspect(hold)
        held = artifact(hold, 'recovery_state')
        validation, code = run_native(manifest, configs['hold'], root / 'held-out', {'weights': held},
                                      action='validate', progress=progress)
        if code:
            raise ValueError('Hold validation failed: ' + str(validation))
        inspect(validation)
        saved = {'device_state': deployed, 'device_state_sha256': digest(deployed),
                 'hold_run': str(hold), 'held_out_run': str(validation)}
        write(receipt, saved)
        return saved


def case_worker(manifest, case, *, canary=False):
    root = Path(manifest['output']) / ('canary' if canary else 'cases') / case['id']
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / 'worker.lock', nonblocking=True):
        state_path = root / 'state.json'
        state = read(state_path) if state_path.exists() else {'attempts': []}
        if state.get('state') == 'complete':
            return state
        if state.get('state') not in (None, 'checkpointed'):
            raise ValueError('Unreconciled previous case state: ' + str(state.get('state')))
        state.update(state='running', job_id=os.environ['SLURM_JOB_ID'], updated_at=now())
        write(state_path, state)

        def progress(run):
            state.update(native_run=run, updated_at=now())
            write(state_path, state)

        try:
            start = baseline(manifest, case['rate'], progress)
            state['baseline'] = start
            state['config'] = case['config']
            trained = case.get('trained')
            if trained and not state.get('training_complete'):
                # An explicitly declared completed run is evaluated, never retrained.
                inspect(trained['run'])
                final = artifact(trained['run'], 'recovery_state')
                if final != trained['path'] or digest(final) != trained['sha256']:
                    raise ValueError('Declared completed training differs from its frozen identity.')
                metrics = read(Path(trained['run']) / 'result.json')['metrics']
                if metrics['initial_device_state_sha256'] != start['device_state_sha256']:
                    raise ValueError('Declared completed training started from another device state.')
                state.update(training_complete=True, final_run=trained['run'], final=final)
                write(state_path, state)
            origin = case.get('continue_from')
            if origin and not state['attempts'] and not state.get('resume'):
                # A declared horizon extension starts from an earlier study's unfinished state.
                rows = [json.loads(line) for line in (Path(origin['run']) / 'metrics.jsonl').read_text().splitlines()]
                started = [r for r in rows if r['event'] == 'recovery_start']
                if (not started or started[0]['initial_device_state_sha256'] != start['device_state_sha256']
                        or Path(origin['path']) != Path(origin['run']) / 'checkpoints/latest.pt'):
                    raise ValueError('Declared continuation started from another device state.')
                state.update(continued_from=origin, resume={k: origin[k] for k in ('path', 'sha256', 'step')})
                write(state_path, state)
            maximum = 20 if canary else 1
            for _ in range(maximum):
                if state.get('training_complete'):
                    break
                resume = state.get('resume')
                if resume:
                    if digest(resume['path']) != resume['sha256']:
                        raise ValueError('Continuation checkpoint digest differs.')
                    inputs = {'resume': resume['path']}
                else:
                    inputs = {'weights': manifest['weights'], 'device_state': start['device_state']}
                run, code = run_native(manifest, case['config'], root / f"attempt-{len(state['attempts']):03d}",
                                        inputs, progress=progress)
                state['attempts'].append(str(run))
                if code:
                    saved = checkpointed(run)
                    if saved['step'] <= (resume or {}).get('step', 0):
                        raise ValueError('Continuation made no progress.')
                    state.update(resume=saved, state='checkpointed', updated_at=now())
                    write(state_path, state)
                    if canary:
                        continue
                    return state
                inspect(run)
                state.update(training_complete=True, final_run=str(run), final=artifact(run, 'recovery_state'))
                state.pop('resume', None)
                write(state_path, state)
                break
            if not state.get('training_complete'):
                raise ValueError('Canary exhausted its continuation allowance.')
            validation, code = run_native(manifest, case['config'], root / 'held-out', {'weights': state['final']},
                                          action='validate', progress=progress)
            if code:
                raise ValueError('Held-out validation failed: ' + str(validation))
            inspect(validation)
            state.update(state='complete', held_out_run=str(validation), updated_at=now())
            write(state_path, state)
            return state
        except BaseException as error:
            state.update(state='failed', error=str(error), updated_at=now())
            write(state_path, state)
            raise


def endpoint_worker(manifest, case):
    """Evaluate an explicitly accepted saved endpoint; never call training."""
    root = Path(manifest['output']) / 'cases' / case['id']
    with lock(root / 'worker.lock', nonblocking=True):
        state_path = root / 'state.json'
        if state_path.exists():
            state = read(state_path)
            if state.get('state') == 'complete':
                inspect(state['held_out_run'])
                return state
            raise ValueError('Unreconciled endpoint evaluation: ' + case['id'])
        endpoint = case['endpoint']
        if digest(endpoint['path']) != endpoint['sha256']:
            raise ValueError('Accepted endpoint digest differs.')
        state = {'state': 'running', 'job_id': os.environ['SLURM_JOB_ID'],
                 'endpoint': endpoint, 'baseline': case['baseline'], 'updated_at': now()}
        write(state_path, state)

        def progress(run):
            state.update(native_run=run, updated_at=now())
            write(state_path, state)

        try:
            run, code = run_native(manifest, case['config'], root / 'held-out',
                                   {'weights': endpoint['path']}, action='validate', progress=progress)
            if code:
                raise ValueError('Endpoint validation failed: ' + str(run))
            inspect(run)
            metrics = read(run / 'result.json')['metrics']
            checkpoint = metrics['checkpoint']
            hold = read(Path(case['baseline']['hold_run']) / 'result.json')['metrics']
            if (checkpoint['updates'] != endpoint['step'] or checkpoint['training_complete']
                    or checkpoint['schema'] != 'opt_cmo_drn.recovery_resume.v1'
                    or checkpoint['primary_state'] != 'held_apparent'
                    or checkpoint['initial_device_state_sha256'] != case['baseline']['device_state_sha256']
                    or checkpoint['start_held_hashes'] != hold['start_held_hashes']):
                raise ValueError('Evaluation differs from the accepted endpoint or matched hold.')
            if digest(endpoint['path']) != endpoint['sha256']:
                raise ValueError('Evaluation changed its input checkpoint.')
            state.update(state='complete', held_out_run=str(run), updated_at=now())
            write(state_path, state)
            return state
        except BaseException as error:
            state.update(state='failed', error=str(error), updated_at=now())
            write(state_path, state)
            raise


def worker(manifest_path, mode, index):
    manifest = read(manifest_path)
    verify_source(manifest)
    for role in ('teacher', 'weights'):
        if digest(manifest[role]) != manifest[role + '_sha256']:
            raise ValueError('Input digest differs: ' + role)
    if manifest.get('operation') == 'evaluate_accepted_endpoints':
        if mode == 'canary' and index != 0:
            raise ValueError('Canary task must be zero.')
        state = endpoint_worker(manifest, manifest['cases'][index])
        if mode == 'canary':
            write(Path(manifest['output']) / 'canary/complete.json',
                  {'state': 'complete', 'manifest_sha256': digest(manifest_path), 'cases': [state], 'updated_at': now()})
    elif mode == 'canary':
        if index != 0:
            raise ValueError('Canary task must be zero.')
        states = [case_worker(manifest, c, canary=True) for c in manifest['canary_cases']]
        if not all(len(s['attempts']) > 1 for s in states):
            raise ValueError('Canary must exercise real checkpoint continuation in both laws.')
        write(Path(manifest['output']) / 'canary/complete.json',
              {'state': 'complete', 'manifest_sha256': digest(manifest_path), 'cases': states, 'updated_at': now()})
    else:
        case_worker(manifest, manifest['cases'][index])


def pending_indices(manifest):
    indices = []
    for index, case in enumerate(manifest['cases']):
        path = Path(manifest['output']) / 'cases' / case['id'] / 'state.json'
        if not path.exists():
            indices.append(index)
        else:
            state = read(path)['state']
            if state == 'checkpointed':
                indices.append(index)
            elif state != 'complete':
                raise ValueError('Unreconciled case state: ' + case['id'] + ' ' + state)
    return indices


def submit_once(manifest_path, manifest, ledger, indices, *, canary=False):
    """An ambiguous sbatch outcome remains recorded and must be reconciled."""
    if ledger.get('submitting'):
        raise ValueError('Unreconciled submission; never submit a duplicate.')
    verify_source(manifest)
    argv = command(manifest_path, manifest, indices, canary=canary, wave=len(ledger['jobs']))
    ledger['submitting'] = {'command': argv, 'indices': indices, 'canary': canary, 'created_at': now()}
    path = Path(manifest['output']) / 'scheduler.json'
    write(path, ledger)
    parent = output(argv).strip().split(';')[0]
    if not parent.isdigit():
        raise ValueError('Unexpected sbatch parent ID.')
    item = ledger.pop('submitting')
    item.update(parent=parent)
    ledger['jobs'].append(item)
    write(path, ledger)
    return parent


def advance(manifest_path, *, release=False):
    manifest = read(manifest_path)
    root = Path(manifest['output'])
    with lock(root / 'scheduler.lock'):
        path = root / 'scheduler.json'
        ledger = read(path) if path.exists() else {'jobs': [], 'state': 'prepared'}
        if ledger.get('submitting'):
            raise ValueError('Submission outcome is uncertain; reconcile before continuing.')
        charge = 0
        for job in ledger['jobs']:
            if 'accounting' not in job:
                live = live_job(job['parent'])
                if live:
                    ledger.update(state='queued' if all(v == 'PENDING' for v in live.splitlines()) else 'running', updated_at=now())
                    write(path, ledger)
                    return ledger
                rows = scheduler_rows(job['parent'], job['indices'])
                contract_rows(rows, manifest)
                if any(r['state'] != 'COMPLETED' or r['exit_code'] != '0:0' for r in rows.values()):
                    raise ValueError('A Slurm task failed; no automatic scientific retry: ' + job['parent'])
                job['accounting'] = rows
            charge += sum(r['elapsed_seconds'] for r in job['accounting'].values())
        ledger['charged_gpu_seconds'] = charge
        gate = root / 'gate.json'
        if not gate.exists():
            ledger.update(state='awaiting_gate', updated_at=now())
            write(path, ledger)
            return ledger
        receipt = read(gate)
        if not receipt.get('gate_passed') or receipt.get('manifest_sha256') != digest(manifest_path):
            raise ValueError('Canary/static gate does not match this immutable launch candidate.')
        if release:
            ledger['released'] = True
        if not ledger.get('released'):
            ledger.update(state='awaiting_release', updated_at=now())
            write(path, ledger)
            return ledger
        indices = pending_indices(manifest)
        if not indices:
            states = [read(root / 'cases' / c['id'] / 'state.json') for c in manifest['cases']]
            summary = {'state': 'complete', 'manifest_sha256': digest(manifest_path), 'cases': {}}
            for case, state in zip(manifest['cases'], states):
                summary['cases'][case['id']] = {
                    'case': case,
                    'training': (read(Path(state['held_out_run']) / 'result.json')['metrics']['checkpoint']
                                 if manifest.get('operation') == 'evaluate_accepted_endpoints'
                                 else read(Path(state['final_run']) / 'result.json')['metrics']),
                    'held_out': read(Path(state['held_out_run']) / 'result.json')['metrics'],
                    'baseline': read(Path(state['baseline']['held_out_run']) / 'result.json')['metrics'],
                    'runs': state}
            write(root / 'summary.json', summary)
            ledger.update(state='complete', updated_at=now())
        elif charge + len(indices) * manifest['allocation_seconds'] > manifest['gpu_budget_seconds']:
            raise ValueError('Next allocation wave exceeds the declared GPU-time allowance.')
        elif len(ledger['jobs']) >= manifest['max_waves'] + 1:
            raise ValueError('Continuation wave allowance exhausted.')
        else:
            used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            if used + len(indices) * manifest['storage_reserve_per_case'] > manifest['storage_budget_bytes']:
                raise ValueError('Next allocation wave exceeds the declared artifact allowance.')
            submit_once(manifest_path, manifest, ledger, indices)
            ledger.update(state='queued', updated_at=now())
        write(path, ledger)
        return ledger


def supervise(manifest_path):
    manifest = read(manifest_path)
    root = Path(manifest['output'])
    with lock(root / 'supervisor.lock', nonblocking=True):
        (root / 'supervisor.pid').write_text(str(os.getpid()))
        while True:
            try:
                state = advance(manifest_path)
                write(root / 'supervisor.json', {'state': state['state'], 'updated_at': now(), 'pid': os.getpid()})
                if state['state'] == 'complete':
                    return
            except Exception as error:
                write(root / 'supervisor.json', {'state': 'failed', 'error': str(error), 'updated_at': now(), 'pid': os.getpid()})
                raise
            time.sleep(manifest['poll_seconds'])


def probe(manifest_path):
    manifest = read(manifest_path)
    root = Path(manifest['output'])
    scheduler = read(root / 'scheduler.json') if (root / 'scheduler.json').exists() else {}
    owner = read(root / 'supervisor.json') if (root / 'supervisor.json').exists() else {}
    cases = {}
    for kind, records in [('cases', manifest['cases']), ('canary', manifest.get('canary_cases', []))]:
        for case in records:
            path = root / kind / case['id'] / 'state.json'
            if not path.exists():
                continue
            state = read(path)
            native = Path(state['native_run']) if state.get('native_run') else None
            progress = []
            if native is not None and (native / 'metrics.jsonl').exists():
                metrics = native / 'metrics.jsonl'
                with metrics.open('rb') as f:
                    f.seek(max(0, metrics.stat().st_size - 65536))
                    lines = f.read().decode().splitlines()
                for line in lines[-5:]:
                    try:
                        row = json.loads(line)
                        progress.append({k: row[k] for k in ('event', 'step', 'elapsed_seconds') if k in row})
                    except ValueError:
                        pass
            cases[kind + '/' + case['id']] = {'state': state['state'], 'job_id': state.get('job_id'),
                                               'native_run': str(native), 'progress': progress, 'error': state.get('error')}
    jobs = {}
    for job in scheduler.get('jobs', []):
        if 'accounting' in job:
            jobs[job['parent']] = job['accounting']
        else:
            jobs[job['parent']] = (live_job(job['parent'], '%i|%T|%R')
                                   or scheduler_rows(job['parent'], job['indices']))
    return {'scheduler': scheduler, 'supervisor': owner, 'cases': cases, 'jobs': jobs,
            'complete_evidence': (root / 'summary.json').exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('worker', 'canary-submit', 'advance', 'release', 'supervise', 'probe', 'commands'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--mode', choices=('canary', 'production'), default='production')
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    m = read(args.manifest)
    if args.action == 'worker':
        worker(args.manifest, args.mode, task_index(os.environ, args.index))
    elif args.action == 'commands':
        print(json.dumps({'canary': command(args.manifest, m, [0], canary=True),
                          'production': command(args.manifest, m, list(range(len(m['cases']))), wave=1)}, indent=2))
    elif args.action == 'canary-submit':
        root = Path(m['output'])
        (root / 'slurm').mkdir(parents=True, exist_ok=True)
        with lock(root / 'scheduler.lock'):
            path = root / 'scheduler.json'
            ledger = read(path) if path.exists() else {'jobs': [], 'state': 'prepared'}
            if ledger['jobs'] or ledger.get('submitting'):
                raise ValueError('This manifest already has a submission.')
            print(submit_once(args.manifest, m, ledger, [0], canary=True))
    elif args.action in ('advance', 'release'):
        print(json.dumps(advance(args.manifest, release=args.action == 'release'), indent=2))
    elif args.action == 'supervise':
        supervise(args.manifest)
    else:
        print(json.dumps(probe(args.manifest)))


if __name__ == '__main__':
    main()
