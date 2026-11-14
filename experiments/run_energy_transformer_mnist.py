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


def early_cuda_failure(row):
    """Only zero-update CUDA initialization failures may be retried automatically."""
    if not row.get('run_dir') or row.get('outcome')!='execution_failure':return False
    run=Path(row['run_dir']);status=run/'status.json'
    if not status.exists():return False
    error=json.loads(status.read_text()).get('error') or {}
    message=error.get('message','')
    metrics=run/'metrics.jsonl'
    if metrics.exists() and any(json.loads(line).get('stage')=='train' for line in metrics.read_text().splitlines()):
        return False
    return ('CUDA' in message and ('requires CUDA' in message or 'initialization' in message))


def previous_coverage(root):
    """Reuse terminal science, account prior wall cost, and never duplicate a live sweep."""
    launch=json.loads((root/'launch.json').read_text())
    process=Path(f'/proc/{launch["pid"]}/cmdline')
    if process.exists() and str(root).encode() in process.read_bytes():
        raise RuntimeError('Previous sweep is still alive')
    result=json.loads((root/'result.json').read_text())
    rows=result['cases']
    if len(rows)!=8 or len({(r['case'],r['arm']) for r in rows})!=8:
        raise RuntimeError('Recovery requires all eight cases accounted for')
    for row in rows:
        if row['outcome']!='terminal' and not early_cuda_failure(row):
            raise RuntimeError('Unknown operational failure; no automatic scientific or generic retry')
    return {(r['case'],r['arm']):r for r in rows}, result['seconds']+result.get('charged_prior_seconds',0),launch['deadline']


def capped64_baselines(path):
    rows=json.loads(path.read_text())
    chosen=[]
    for case in ('small','reference','more_tokens'):
        matches=[r for r in rows if r['case']==case and r['arm']=='digital_bp']
        if len(matches)!=1 or matches[0]['outcome']!='terminal' or not matches[0].get('integrity_passed'):
            raise RuntimeError(f'Missing valid completed BP baseline: {case}')
        chosen.append({**matches[0],'reused_from':str(path)})
    return chosen


def repair_reuse(path):
    """Reuse completed, unchanged science; rerun only the failed 49-token case."""
    rows=__import__('json').loads(path.read_text())
    chosen=capped64_baselines(path)
    for case in ('small','reference'):
        matches=[r for r in rows if r['case']==case and r['arm']=='energy_ep']
        if (len(matches)!=1 or matches[0]['outcome']!='terminal'
                or not matches[0].get('integrity_passed')
                or not matches[0].get('summary',{}).get('coverage_complete')):
            raise RuntimeError(f'Missing complete EP case to preserve: {case}')
        chosen.append({**matches[0],'reused_from':str(path)})
    return chosen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--readiness',type=Path,required=True)
    parser.add_argument('--charged-readiness-seconds',type=float,required=True)
    parser.add_argument('--retry-from',type=Path)
    parser.add_argument('--capped64-from',type=Path,help='Run three capped EP configs and reuse completed BP coverage')
    parser.add_argument('--repair-from',type=Path,help='Repair 49-token training and reuse five completed arms')
    args=parser.parse_args()
    if sum(v is not None for v in (args.capped64_from,args.retry_from,args.repair_from))>1:
        parser.error('Select one composition/recovery mode')
    repaired=args.repair_from is not None
    bounded=repaired or args.capped64_from is not None
    baseline_rows=repair_reuse(args.repair_from) if repaired else capped64_baselines(args.capped64_from) if args.capped64_from else []
    expected_cases=6 if bounded else 8
    root=args.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
    gate=json.loads(args.readiness.read_text())
    if not gate['passed']:raise RuntimeError('FP32 numerical readiness failed')
    if not 0<=args.charged_readiness_seconds<=1800:raise ValueError('Readiness allowance exhausted')
    if (root/'worker.pid').exists():raise RuntimeError('Existing sweep identity; do not duplicate it')
    (root/'worker.pid').write_text(str(os.getpid()))
    previous={};prior_seconds=0.;prior_deadline=None
    if args.retry_from:
        predecessor=args.retry_from.resolve()
        prior_launch=json.loads((predecessor/'launch.json').read_text())
        prior_process=Path(f'/proc/{prior_launch["pid"]}/cmdline')
        while prior_process.exists() and str(predecessor).encode() in prior_process.read_bytes():
            if time.time()>=datetime.fromisoformat(prior_launch['deadline']).timestamp():
                raise TimeoutError('Original benchmark deadline exhausted while awaiting predecessor')
            previous_status=json.loads((predecessor/'status.json').read_text())
            atomic_write_json(root/'status.json',{'state':'queued','stage':'waiting_for_predecessor',
                              'heartbeat_at':now(),'progress':previous_status.get('progress'),
                              'predecessor_stage':previous_status.get('stage')})
            time.sleep(10)
        previous,prior_seconds,prior_deadline=previous_coverage(predecessor)
    begin=time.monotonic();deadline=begin+(7200 if bounded else 21600)-args.charged_readiness_seconds-prior_seconds
    if prior_deadline:
        deadline=min(deadline,begin+datetime.fromisoformat(prior_deadline).timestamp()-time.time())
    # Keep one CUDA context alive between workers. On this host, cold worker
    # initialization failed intermittently; six child probes with a retained
    # context passed. This is supervisor overhead, not model tensor memory.
    import torch
    cuda_context_guard=torch.empty(1,device='cuda',dtype=torch.float32)
    torch.cuda.synchronize()
    atomic_write_json(root/'supervisor-gpu.json',{'context_guard_bytes':cuda_context_guard.numel()*4,
                      'gpu':torch.cuda.get_device_name(),'pid':os.getpid(),
                      'note':'Persistent supervisor CUDA context; driver/context memory is additional to reported model tensor peaks.'})
    rows=list(baseline_rows);active=None
    atomic_write_json(root/'coverage.json',rows)
    def status(stage,**extra):
        native=list((root/active).rglob('metrics.jsonl')) if active else []
        metric_bytes=sum(p.stat().st_size for p in native)
        atomic_write_json(root/'status.json',{'state':stage,'heartbeat_at':now(),'stage':active,
                          'progress':{'terminal_cases':len(rows),'active_metric_bytes':metric_bytes},
                          'elapsed_seconds':time.monotonic()-begin,
                          'charged_readiness_seconds':args.charged_readiness_seconds,**extra})
    try:
        for case in (('more_tokens',) if repaired else ('small','reference','more_tokens') if bounded else ('small','reference','wider','more_tokens')):
            for arm in (('energy_ep',) if bounded else ('energy_ep','digital_bp')):
                active=f'{case}-{arm}';status('running')
                prior=previous.get((case,arm))
                if prior and prior['outcome']=='terminal':
                    rows.append({**prior,'reused_from':str(args.retry_from)})
                    atomic_write_json(root/'coverage.json',rows)
                    continue
                remaining=deadline-time.monotonic()
                if remaining<60:
                    rows.append({'case':case,'arm':arm,'outcome':'global_budget_exhausted'});continue
                folder=root/active;folder.mkdir()
                config_root=ROOT/'examples/energy_transformer_mnist'
                config=(config_root/'repaired128' if repaired else config_root/'capped64' if bounded else config_root)/f'{active}.json'
                case_begin=time.monotonic()
                allowance=min(2100-(prior.get('wall_seconds',0) if prior else 0),remaining)
                attempts=[]
                for attempt in range(2):
                    attempt_root=folder/f'attempt-{attempt+1:02d}';attempt_root.mkdir()
                    command=[sys.executable,'-m','ebl','train','--config',str(config),'--output-dir',str(attempt_root)]
                    child_env=dict(os.environ)
                    child_env['EBL_CASE_DEADLINE_UNIX']=str(time.time()+max(0,allowance-(time.monotonic()-case_begin)))
                    started=time.monotonic()
                    with (attempt_root/'run.log').open('w') as log:
                        child=subprocess.Popen(command,cwd=ROOT,env=child_env,stdout=log,stderr=subprocess.STDOUT)
                        (root/'child.pid').write_text(str(child.pid))
                        while child.poll() is None:
                            elapsed=time.monotonic()-case_begin
                            if elapsed>=allowance:
                                child.terminate()
                                try:child.wait(timeout=5)
                                except subprocess.TimeoutExpired:child.kill();child.wait()
                                break
                            status('running',child_pid=child.pid,case_seconds=elapsed)
                            time.sleep(3)
                    runs=list(attempt_root.glob('*/manifest.json'))
                    row={'case':case,'arm':arm,'returncode':child.returncode,'attempt_seconds':time.monotonic()-started,
                         'command':command,'run_dir':str(runs[0].parent) if len(runs)==1 else None}
                    if row['run_dir']:
                        summary=Path(row['run_dir'])/'artifacts/summary.json'
                        if summary.exists():row['summary']=json.loads(summary.read_text())
                        inspection=subprocess.run([sys.executable,'-m','ebl','runs','inspect',row['run_dir'],
                            '--verify-artifacts','--require-complete','--json'],capture_output=True,text=True,cwd=ROOT)
                        (attempt_root/'inspection.json').write_text(inspection.stdout)
                        (attempt_root/'inspection.stderr').write_text(inspection.stderr)
                        row['integrity_passed']=inspection.returncode==0
                    row['outcome']='terminal' if child.returncode==0 and row.get('integrity_passed') else 'execution_failure'
                    attempts.append(dict(row))
                    if not early_cuda_failure(row) or time.monotonic()-case_begin>=allowance:break
                row['attempts']=attempts
                row['wall_seconds']=time.monotonic()-case_begin+(prior.get('wall_seconds',0) if prior else 0)
                if prior:row['supersedes']=prior
                rows.append(row)
                atomic_write_json(root/'coverage.json',rows)
                with (root/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('summary','command')}),flush=True)
        active='analysis';status('running')
        analysis=subprocess.run([sys.executable,'-m','experiments.plot_energy_transformer_mnist',str(root)],
                                 cwd=ROOT,timeout=max(1,min(600,deadline-time.monotonic())))
        result={'cases':rows,'terminal_cases':sum(r['outcome']=='terminal' for r in rows),
                'expected_cases':expected_cases,'coverage_complete':len(rows)==expected_cases and all(r.get('summary',{}).get('coverage_complete',False) for r in rows),
                'integrity_passed':len(rows)==expected_cases and all(r.get('integrity_passed',False) for r in rows),
                'analysis_returncode':analysis.returncode,'seconds':time.monotonic()-begin,
                'charged_readiness_seconds':args.charged_readiness_seconds,'charged_prior_seconds':prior_seconds}
        atomic_write_json(root/'result.json',result)
        status('complete' if result['integrity_passed'] and analysis.returncode==0 else 'failed')
    except BaseException as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':main()
