"""One-hour serial comparison using only the existing public EBL commands."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ebl.runtime.store import atomic_write_json
from experiments.run_energy_transformer_mnist import early_cuda_failure

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/'examples/energy_transformer_mnist/input_attention'


def now():
    return datetime.now(timezone.utc).isoformat()


def qualified_cap(summary):
    if (not summary.get('coverage_complete') or not summary.get('reference_valid')
            or summary.get('selected_cap') not in (32,64,128)
            or set(summary.get('required_groups',())) != {'query','key','value','output'}):
        return None
    return summary['selected_cap']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    if not 60 <= args.seconds <= 3600:
        raise ValueError('This comparison has at most one additional hour')
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'worker.pid').exists():
        raise RuntimeError('Preserve existing attempts; never duplicate an identity')
    begin = time.monotonic()
    deadline = min(time.time()+args.seconds, args.deadline_unix or float('inf'))
    if deadline <= time.time():
        raise TimeoutError('Original allowance has expired')
    (root/'worker.pid').write_text(str(os.getpid()))
    atomic_write_json(root/'launch.json', dict(pid=os.getpid(), started=now(),
        deadline=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
        deadline_unix=deadline, source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), command=sys.argv))
    rows = []
    active = None

    def status(state, **extra):
        metric_paths = list((root/active).rglob('metrics.jsonl')) if active else []
        atomic_write_json(root/'status.json', dict(state=state, heartbeat_at=now(),
            stage=active, progress=dict(terminal_cases=len(rows),
                active_metric_bytes=sum(p.stat().st_size for p in metric_paths)),
            elapsed_seconds=time.monotonic()-begin, deadline_unix=deadline, **extra))

    def run_native(name, config, mode='train', weights=None):
        nonlocal active
        active = name
        attempts = []
        for attempt in range(1,3):
            if deadline-time.time() < 20:
                raise TimeoutError('Shared one-hour allowance exhausted')
            folder = root/name/f'attempt-{attempt:02d}'
            folder.mkdir(parents=True)
            command = [sys.executable,'-m','ebl',mode,'--config',str(config),
                       '--output-dir',str(folder)]
            if weights is not None:
                command += ['--weights',str(weights)]
            env = dict(os.environ, EBL_CASE_DEADLINE_UNIX=str(deadline))
            start = time.monotonic()
            with (folder/'run.log').open('w') as log:
                child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                (root/'child.pid').write_text(str(child.pid))
                while child.poll() is None:
                    status('running',child_pid=child.pid)
                    if time.time() >= deadline:
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait()
                        break
                    time.sleep(2)
            manifests = list(folder.glob('*/manifest.json'))
            run = manifests[0].parent if len(manifests)==1 else None
            row = dict(name=name,attempt=attempt,returncode=child.returncode,
                       run_dir=str(run) if run else None,command=command,
                       seconds=time.monotonic()-start,integrity_passed=False,
                       outcome='execution_failure')
            if run is not None:
                summary = run/'artifacts/summary.json'
                if summary.exists():
                    row['summary'] = json.loads(summary.read_text())
                if deadline-time.time() > 2:
                    check = subprocess.run([sys.executable,'-m','ebl','runs','inspect',str(run),
                        '--verify-artifacts','--require-complete','--json'],cwd=ROOT,
                        capture_output=True,text=True,timeout=min(60,max(1,deadline-time.time())))
                    (folder/'inspection.json').write_text(check.stdout)
                    (folder/'inspection.stderr').write_text(check.stderr)
                    row['integrity_passed'] = check.returncode==0
            if child.returncode==0 and row['integrity_passed']:
                row['outcome'] = 'terminal'
            attempts.append(dict(row))
            if not early_cuda_failure(row):
                break
        row['attempts'] = attempts
        rows.append(row)
        atomic_write_json(root/'coverage.json',rows)
        with (root/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({k:v for k,v in row.items() if k not in ('attempts','summary')})+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('attempts','summary','command')}),flush=True)
        if row['outcome'] != 'terminal':
            raise RuntimeError(f'Operational failure in {name}; preserve attempts and diagnose')
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > 2_000_000_000:
            raise RuntimeError('Declared two-GB output allowance exceeded')
        return row

    def finish(outcome, **extra):
        atomic_write_json(root/'result.json',dict(outcome=outcome,cases=rows,
            seconds=time.monotonic()-begin,deadline_unix=deadline,**extra))
        status('complete',outcome=outcome)

    try:
        status('running')
        # Retained CUDA context avoids the host's previously observed cold-child
        # initialization fault. This overhead is outside per-model tensor peaks.
        import torch
        guard = torch.empty(1,device='cuda',dtype=torch.float32)
        torch.cuda.synchronize()
        atomic_write_json(root/'supervisor-gpu.json',dict(pid=os.getpid(),
            gpu=torch.cuda.get_device_name(),context_guard_bytes=guard.numel()*4))
        for arm in ('energy_ep','energy_bp'):
            smoke = run_native(f'smoke-{arm}',CONFIGS/f'smoke-{arm}.json')
            if not smoke.get('summary',{}).get('coverage_complete'):
                finish('smoke_scientific_stop'); return
        warmup = run_native('warmup-energy_ep',CONFIGS/'warmup-energy_ep.json')
        if not warmup.get('summary',{}).get('coverage_complete'):
            finish('warmup_scientific_stop'); return
        qualification = run_native('gradient-selection',CONFIGS/'gradient-selection.json',
            'validate',Path(warmup['run_dir'])/'checkpoints/final.pt')
        cap = qualified_cap(qualification['summary'])
        if cap is None:
            finish('qualification_failed',selected_cap=None); return
        atomic_write_json(root/'selection.json',dict(selected_cap=cap,
            checkpoint=str(Path(warmup['run_dir'])/'checkpoints/final.pt'),
            diagnostic=qualification['run_dir']))
        trained = {}
        for name, config in (
                ('input-energy_ep',CONFIGS/f'caps/{cap}/input-energy_ep.json'),
                ('input-energy_bp',CONFIGS/f'caps/{cap}/input-energy_bp.json'),
                ('recurrent-energy_bp',CONFIGS/'recurrent-energy_bp.json')):
            trained[name] = run_native(name,config)
        final = None
        if trained['input-energy_ep'].get('summary',{}).get('coverage_complete'):
            final = run_native('gradient-final',CONFIGS/f'caps/{cap}/gradient-final.json',
                'validate',Path(trained['input-energy_ep']['run_dir'])/'checkpoints/final.pt')
        complete = all(r.get('summary',{}).get('coverage_complete') for r in trained.values())
        gradient_passed = final is not None and qualified_cap(final['summary'])==cap
        finish('comparison_collected' if complete and gradient_passed else 'scientific_checks_failed',
               selected_cap=cap,full_training_coverage=complete,final_gradient_passed=gradient_passed)
    except BaseException as error:
        status('failed',error=repr(error))
        raise


if __name__ == '__main__':
    main()
