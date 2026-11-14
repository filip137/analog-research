"""Bounded serial composition of the public fuse-MNIST CLI and final checks."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone
import torch
from ebl.runtime.store import atomic_write_json

ROOT=Path(__file__).resolve().parents[1]

def now():return datetime.now(timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--charged-readiness-seconds',type=float,required=True);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Evidence launch requires a clean pinned checkout')
    out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'worker.pid').write_text(str(os.getpid()))
    started=time.time();deadline=started+5400-a.charged_readiness_seconds
    context=torch.zeros(1,device='cuda')
    cases=[(case,kernel) for case in ('small','reference','more_tokens') for kernel in ('gaussian','linear')]
    rows=[]
    launch={'started':now(),'pid':os.getpid(),'source':str(ROOT),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'deadline_unix':deadline,'charged_readiness_seconds':a.charged_readiness_seconds,'cases':cases}
    atomic_write_json(out/'launch.json',launch)
    def status(stage,child=None,progress=None):
        atomic_write_json(out/'status.json',{'state':'running','stage':stage,'worker_pid':os.getpid(),'child_pid':child,
            'heartbeat_at':now(),'progress':progress,'cases_complete':len(rows),'deadline_unix':deadline})
    def run(command,log,budget,stage,metrics=None):
        case_deadline=min(deadline,time.time()+budget)
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','EBL_CASE_DEADLINE_UNIX':str(case_deadline)}
        with log.open('w') as f:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            (out/'child.pid').write_text(str(child.pid))
            previous_progress=None
            while child.poll() is None:
                if time.time()>=case_deadline:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    return {'returncode':child.returncode,'timeout':True}
                metric_files=[] if metrics is None else list(metrics.glob('*/metrics.jsonl'))
                progress=sum(q.stat().st_size for q in metric_files)
                status(stage,child.pid,progress)
                if progress!=previous_progress:
                    with (out/'metrics.jsonl').open('a') as mf:
                        mf.write(json.dumps({'stage':stage,'child_pid':child.pid,'metric_bytes':progress,'time':now()})+'\n')
                    previous_progress=progress
                time.sleep(2)
        return {'returncode':child.returncode,'timeout':False}
    for case,kernel in cases:
        name=case+'-'+kernel;case_root=out/name
        if deadline-time.time()<60:
            rows.append({'case':case,'kernel':kernel,'outcome':'not_launched_budget'});continue
        config=ROOT/'examples/fuse_mnist'/f'{name}.json'
        command=[sys.executable,'-m','ebl','train','--config',str(config),'--output-dir',str(case_root)]
        result=run(command,out/f'{name}.log',850,name,case_root)
        row={'case':case,'kernel':kernel,'command':command,**result}
        summaries=list(case_root.glob('*/artifacts/summary.json'))
        if len(summaries)==1:
            summary=summaries[0];bundle=summary.parent.parent;row['run_dir']=str(bundle)
            row['summary']=json.loads(summary.read_text())
            check=[sys.executable,'-m','ebl','runs','inspect',str(bundle),'--verify-artifacts','--require-complete']
            integrity=run(check,out/f'{name}-integrity.log',30,name+'-integrity')
            row['integrity_passed']=integrity['returncode']==0
            if row['summary']['updates']:
                check_path=out/f'{name}-gradients.json'
                cmd=[sys.executable,'-m','experiments.fuse_mnist_checks','--checkpoint',str(bundle/'checkpoints/final.pt'),
                     '--count','8','--output',str(check_path)]
                grad=run(cmd,out/f'{name}-gradients.log',120,name+'-gradients')
                row['gradient_returncode']=grad['returncode']
                if check_path.is_file():row['gradients']=json.loads(check_path.read_text())
        row['accepted']=bool(row.get('integrity_passed') and row.get('summary',{}).get('coverage_complete') and row.get('gradients',{}).get('passed'))
        rows.append(row);atomic_write_json(out/'comparison.json',{'cases':rows})
    result={'cases':rows,'coverage_complete':len(rows)==6 and all(r.get('summary',{}).get('coverage_complete') for r in rows),
            'accepted':len(rows)==6 and all(r.get('accepted') for r in rows),
            'charged_readiness_seconds':a.charged_readiness_seconds,'composition_seconds':time.time()-started,
            'total_charged_seconds':time.time()-started+a.charged_readiness_seconds,'ended':now()}
    atomic_write_json(out/'result.json',result)
    atomic_write_json(out/'status.json',{'state':'complete','heartbeat_at':now(),'cases_complete':len(rows),
                                      'coverage_complete':result['coverage_complete'],'accepted':result['accepted']})
    print(json.dumps({'output_root':str(out),'accepted':result['accepted'],'coverage_complete':result['coverage_complete']}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
