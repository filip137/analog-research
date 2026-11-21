"""Run a bounded, prospectively ordered series of native development and confirmation campaigns."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PILOT = Path(__file__).resolve().parent
ROOT = PILOT.parents[2]
sys.path.insert(0,str(ROOT))
from ebl.runtime.store import atomic_write_json
from prepare import ROUTES


def main(args):
    def interrupted(signum, _frame):
        raise InterruptedError(f'Lane interrupted by signal {signum}.')
    signal.signal(signal.SIGTERM, interrupted)
    base = args.root.resolve()
    lock = (base/'lane.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (base/'lane.pid').write_text(str(os.getpid())+'\n')
    path = base/'lane-status.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    if old.get('state')=='complete':
        return
    prior_spent = max(args.debited_seconds, old.get('spent_seconds',0))
    start = time.monotonic()
    spent = lambda: prior_spent + time.monotonic()-start
    # The canary runs under its own saved receipt; the GPU allowance includes it in the starting debit.
    proof = json.loads(args.canary.read_text())
    if proof['phase']!='canary' or proof['scope']['model']!=args.model or proof['completed']!=23:
        raise ValueError('Incomplete matching operational canary.')
    plan = json.loads(args.plan.read_text())
    desired = {(t,r,m) for t in ('hellaswag','boolq','pubmed') for r in ('00','03') for m in ROUTES}
    if len(plan)!=42 or {tuple(x) for x in plan}!=desired:
        raise ValueError('Lane must schedule all 42 task, rate and method cells exactly once.')
    completed = old.get('verified_endpoints',[])
    status = {'model':args.model,'cap_seconds':args.cap_seconds,'verified_endpoints':completed,
              'state':'running','active_unit':None,'reason':None}
    def publish():
        status.update(spent_seconds=round(spent(),2), verified_endpoint_count=len(completed),
                      remaining_coverage=[x for x in plan if x not in completed])
        atomic_write_json(path,status)
    publish()
    def execute(command, executable_root, log_path):
        remain = args.cap_seconds-spent()
        if remain < 1:
            raise TimeoutError('Declared GPU-hour allowance reached.')
        with log_path.open('ab',buffering=0) as log:
            proc = subprocess.Popen(command,cwd=executable_root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while proc.poll() is None:
                    if spent()>=args.cap_seconds:
                        raise TimeoutError('Declared GPU-hour allowance reached.')
                    publish()
                    try:
                        proc.wait(timeout=min(20,max(.1,args.cap_seconds-spent())))
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait()
            if proc.returncode==124:
                raise TimeoutError('GPU campaign exceeded its remaining walltime.')
            if proc.returncode:
                raise RuntimeError(f'Native command exited {proc.returncode}. See {log_path}.')
    try:
        for item in plan:
            if item in completed:
                continue
            task,rate,route = item
            group = base/'task-runs'/f'{task}-r{rate}'
            group.mkdir(parents=True,exist_ok=True)
            selection = None
            for phase in ('development','confirm'):
                status['active_unit'] = {'task':task,'rate':rate,'route':route,'phase':phase}
                publish()
                root = group/route/phase
                root.mkdir(parents=True,exist_ok=True)
                reviewed = root/'review.json'
                cache = group/f'{phase}-reusable.json'
                if reviewed.exists():
                    selection = reviewed
                    continue
                log = root/'lane.log'
                prep = [sys.executable,str(PILOT/'prepare.py'),'--phase',phase,'--model',args.model,
                        '--task',task,'--rate',rate,'--receipt',str(base/'inputs/receipt.json'),
                        '--output',str(root/'config'),'--route',route]
                if cache.exists() and not (root/'config/campaign.json').exists():
                    prep.extend(['--reuse',str(cache)])
                if phase=='confirm':
                    prep.extend(['--selection',str(selection)])
                # Preserve the exact contract and cached handoffs on safe resume.
                if not (root/'config/campaign.json').exists():
                    execute(prep,ROOT,log)
                run_id = '-'.join((args.model,rate,task,phase))
                cmd = [str(PILOT/'run_lab.sh'),str(ROOT),str(base/'inputs'),str(root/'config/campaign.json'),
                       str(root/'runs'),str(max(1,int(args.cap_seconds-spent()))),'--resume']
                execute(cmd,ROOT,log)
                execute([sys.executable,str(PILOT/'review.py'),str(root/'runs'/run_id),'--output',str(reviewed)],ROOT,log)
                verified = json.loads(reviewed.read_text())
                reusable = json.loads(cache.read_text()) if cache.exists() else {'scope':verified['scope'],'phase':phase,'reusable':{}}
                reusable['reusable'].update(verified['reusable'])
                atomic_write_json(cache,reusable)
                selection = reviewed
            completed.append(item)
            publish()
        status.update(state='complete',reason='All 42 held-out task endpoints have verified native receipts.')
    except TimeoutError as error:
        status.update(state='budget_exhausted',reason=str(error))
    except BaseException as error:
        status.update(state='failed',reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        publish()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--canary',type=Path,required=True)
    p.add_argument('--model',choices=['drn6','drn12'],required=True)
    p.add_argument('--cap-seconds',type=int,required=True)
    p.add_argument('--debited-seconds',type=float,required=True)
    main(p.parse_args())
