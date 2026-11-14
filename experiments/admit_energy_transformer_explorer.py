"""Single local admission wait; no CUDA work and no changes to other jobs."""
import argparse
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from ebl.runtime.store import atomic_write_json

def gpu_available():
    try:
        processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        device=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        if processes.returncode or device.returncode:return False,'GPU observation failed'
        if processes.stdout.strip():return False,'GPU occupied by '+processes.stdout.strip().replace('\n',', ')
        rows=device.stdout.strip().splitlines()
        if len(rows)!=1:return False,'Expected exactly the declared local GPU'
        free,util=(float(v.strip()) for v in rows[0].split(','))
        return free>=2048 and util<=10,dict(free_mib=free,utilization=util)
    except (subprocess.SubprocessError,ValueError,OSError) as error:
        return False,repr(error)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--charge-root',type=Path,action='append',default=[])
    p.add_argument('--seconds',type=int,default=5400);p.add_argument('--wait-seconds',type=int,default=1800)
    a=p.parse_args()
    if not 60<=a.seconds<=5400 or not 1<=a.wait_seconds<=1800:raise ValueError('Declared execution/admission bounds exceeded')
    root=a.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'admission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'worker.pid').exists():raise RuntimeError('Preserve existing execution; never launch a duplicate')
    (root/'supervisor.pid').write_text(str(os.getpid()))
    wait_started=time.time()
    while True:
        available,evidence=gpu_available();timestamp=datetime.now(timezone.utc).isoformat()
        atomic_write_json(root/'status.json',dict(state='queued',heartbeat_at=timestamp,stage='GPU admission',
            progress=dict(capacity=evidence),queued_seconds=time.time()-wait_started))
        if available:
            admitted=time.time();deadline=admitted+a.seconds
            atomic_write_json(root/'admission.json',dict(admitted_at=timestamp,deadline_unix=deadline,
                queued_seconds=admitted-wait_started,evidence=evidence,allowed_seconds=a.seconds))
            cmd=[sys.executable,'-u','-m','experiments.run_energy_transformer_explorer','--output-root',str(root),
                 '--baseline-root',str(a.baseline_root),'--seconds',str(a.seconds),'--deadline-unix',str(deadline)]
            for charged in a.charge_root:cmd+=['--charge-root',str(charged)]
            print(json.dumps(dict(admitted_at=timestamp,deadline_unix=deadline,command=cmd)),flush=True)
            # Same PID and unique output-root command identity transfer to the
            # scientific supervisor; no second scheduler or detached child.
            os.execv(sys.executable,cmd)
        if time.time()-wait_started>=a.wait_seconds:
            atomic_write_json(root/'status.json',dict(state='failed',heartbeat_at=timestamp,stage='GPU admission',error='Capacity wait expired; no GPU execution occurred'))
            raise TimeoutError('No exclusive local GPU became available within admission wait')
        time.sleep(10)

if __name__=='__main__':main()
