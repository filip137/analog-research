"""Serial composition of prepared native runs with a hard shared deadline.

Invoke only after readiness gates in the owning pilot have passed. This is
not a scheduler; run-watch owns ongoing observation and incident dispatch.
"""
import argparse,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from ebl.runtime.store import atomic_write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--readiness',type=Path,required=True)
    args=parser.parse_args();root=args.output_root;root.mkdir(parents=True,exist_ok=True)
    readiness=json.loads(args.readiness.read_text())
    required=['gpu_smoke_pass','gpu_gradient_pass','resume_replay_pass']
    if not all(readiness.get(k) is True for k in required):raise RuntimeError('Readiness gates have not passed')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if readiness.get('source') != commit:raise RuntimeError('Readiness source must match launch source')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Use a clean pinned source checkout')
    budget=json.loads((root/'budget.json').read_text());deadline=datetime.fromisoformat(budget['deadline']).timestamp()
    # Retain CUDA while child processes exit/start; does not repair a broken driver.
    guard=torch.empty(1,device='cuda')
    state={'state':'running','pid':os.getpid(),'source':commit,'cases':[],'deadline':budget['deadline']}
    (root/'worker.pid').write_text(str(os.getpid()))
    def save():
        state['heartbeat_at']=datetime.now(timezone.utc).isoformat()
        atomic_write_json(root/'supervisor.json',state)
    save()
    try:
        for case in ['drn','digital']:
            if time.time()>=deadline-900:
                state['cases'].append({'case':case,'state':'not_run','reason':'shared_budget'});break
            logpath=root/f'{case}.log';output=root/case
            with logpath.open('x') as log:
                command=[sys.executable,'-m','ebl','train','--config',f'examples/pseudo_drn_large/{case}.json','--output-dir',str(output)]
                # Stay in the queue worker's process group so its hard stop owns
                # both this supervisor and the training process.
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                row={'case':case,'pid':child.pid,'log':str(logpath),'output':str(output),'state':'running'}
                state['cases'].append(row);save();print(row,flush=True)
                config=json.loads(Path(f'examples/pseudo_drn_large/{case}.json').read_text())
                case_limit=config['wall_seconds']
                stop_at=min(deadline,time.time()+case_limit)
                while child.poll() is None:
                    if time.time()>=stop_at:
                        child.terminate()
                        try:child.wait(timeout=max(1,min(30,deadline-time.time())))
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        row['hard_limit_reached']=True;break
                    active_path=output/'active-run.json'
                    if active_path.exists():
                        current=Path(json.loads(active_path.read_text())['run_dir'])
                        progress_path=current/'progress.json'
                        if progress_path.exists():
                            progress=json.loads(progress_path.read_text())
                            state.update(stage=case+':'+str(progress.get('stage','evaluation')),
                                         step=progress.get('step'),epoch=progress.get('epoch'))
                            save()
                    time.sleep(min(5,max(0,stop_at-time.time())))
                row.update(returncode=child.returncode,state='complete' if child.returncode==0 else 'failed');save()
                if child.returncode:raise RuntimeError(f'{case} failed; preserve artifacts and recover within the original allowance')
            active=json.loads((output/'active-run.json').read_text());bundle=Path(active['run_dir'])
            with (root/f'{case}-inspection.json').open('w') as stream:
                subprocess.run([sys.executable,'-m','ebl','runs','inspect',str(bundle),'--verify-artifacts','--require-complete','--json'],stdout=stream,check=True)
            row['bundle']=str(bundle);save()
        state['state']='complete';save();atomic_write_json(root/'result.json',state)
    except BaseException as error:
        state.update(state='failed',error=str(error));save();raise


if __name__=='__main__':main()
