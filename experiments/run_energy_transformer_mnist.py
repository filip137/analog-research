"""Bounded serial composition of the eight public EBL runs (one local GPU)."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from ebl.runtime.store import atomic_write_json

ROOT=Path(__file__).resolve().parents[1]


def now():return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--readiness',type=Path,required=True)
    parser.add_argument('--charged-readiness-seconds',type=float,required=True)
    args=parser.parse_args();root=args.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
    gate=json.loads(args.readiness.read_text())
    if not gate['passed']:raise RuntimeError('FP32 numerical readiness failed')
    if not 0<=args.charged_readiness_seconds<=1800:raise ValueError('Readiness allowance exhausted')
    if (root/'worker.pid').exists():raise RuntimeError('Existing sweep identity; do not duplicate it')
    (root/'worker.pid').write_text(str(os.getpid()))
    begin=time.monotonic();deadline=begin+21600-args.charged_readiness_seconds
    rows=[];active=None
    def status(stage,**extra):
        native=list((root/active).glob('*/metrics.jsonl')) if active else []
        metric_bytes=sum(p.stat().st_size for p in native)
        atomic_write_json(root/'status.json',{'state':stage,'heartbeat_at':now(),'stage':active,
                          'progress':{'terminal_cases':len(rows),'active_metric_bytes':metric_bytes},
                          'elapsed_seconds':time.monotonic()-begin,
                          'charged_readiness_seconds':args.charged_readiness_seconds,**extra})
    try:
        for case in ('small','reference','wider','more_tokens'):
            for arm in ('energy_ep','digital_bp'):
                active=f'{case}-{arm}';status('running')
                remaining=deadline-time.monotonic()
                if remaining<60:
                    rows.append({'case':case,'arm':arm,'outcome':'global_budget_exhausted'});continue
                folder=root/active;folder.mkdir()
                config=ROOT/'examples/energy_transformer_mnist'/f'{active}.json'
                command=[sys.executable,'-m','ebl','train','--config',str(config),'--output-dir',str(folder)]
                child_env=dict(os.environ)
                # Runtime evaluation leaves five seconds for terminal serialization.
                child_env['EBL_CASE_DEADLINE_UNIX']=str(time.time()+min(2100,remaining))
                started=time.monotonic()
                with (folder/'run.log').open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT,env=child_env,stdout=log,stderr=subprocess.STDOUT)
                    (root/'child.pid').write_text(str(child.pid))
                    while child.poll() is None:
                        elapsed=time.monotonic()-started
                        if elapsed>=min(2100,remaining):
                            child.terminate()
                            try:child.wait(timeout=5)
                            except subprocess.TimeoutExpired:child.kill();child.wait()
                            break
                        status('running',child_pid=child.pid,case_seconds=elapsed)
                        time.sleep(10)
                runs=list(folder.glob('*/manifest.json'))
                row={'case':case,'arm':arm,'returncode':child.returncode,'wall_seconds':time.monotonic()-started,
                     'command':command,'run_dir':str(runs[0].parent) if len(runs)==1 else None}
                if row['run_dir']:
                    summary=Path(row['run_dir'])/'artifacts/summary.json'
                    if summary.exists():row['summary']=json.loads(summary.read_text())
                    inspection=subprocess.run([sys.executable,'-m','ebl','runs','inspect',row['run_dir'],
                        '--verify-artifacts','--require-complete','--json'],capture_output=True,text=True,cwd=ROOT)
                    (folder/'inspection.json').write_text(inspection.stdout)
                    (folder/'inspection.stderr').write_text(inspection.stderr)
                    row['integrity_passed']=inspection.returncode==0
                row['outcome']='terminal' if child.returncode==0 and row.get('integrity_passed') else 'execution_failure'
                rows.append(row)
                atomic_write_json(root/'coverage.json',rows)
                with (root/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('summary','command')}),flush=True)
        active='analysis';status('running')
        analysis=subprocess.run([sys.executable,'-m','experiments.plot_energy_transformer_mnist',str(root)],
                                 cwd=ROOT,timeout=max(1,min(600,deadline-time.monotonic())))
        result={'cases':rows,'terminal_cases':sum(r['outcome']=='terminal' for r in rows),
                'expected_cases':8,'coverage_complete':len(rows)==8 and all(r.get('summary',{}).get('coverage_complete',False) for r in rows),
                'integrity_passed':len(rows)==8 and all(r.get('integrity_passed',False) for r in rows),
                'analysis_returncode':analysis.returncode,'seconds':time.monotonic()-begin,
                'charged_readiness_seconds':args.charged_readiness_seconds}
        atomic_write_json(root/'result.json',result)
        status('complete' if result['integrity_passed'] and analysis.returncode==0 else 'failed')
    except BaseException as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':main()
