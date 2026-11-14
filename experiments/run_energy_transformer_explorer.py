"""Bounded, declared attention hypotheses through the public EBL CLI."""
import argparse
from datetime import datetime,timezone
import json,os,subprocess,sys,time
from pathlib import Path
from ebl.runtime.store import atomic_write_json
from experiments.run_energy_transformer_mnist import early_cuda_failure
from experiments.run_energy_transformer_input_attention import qualified_cap

ROOT=Path(__file__).resolve().parents[1]
CONFIGS=ROOT/'examples/energy_transformer_mnist/attention_explorer'
VARIANTS=('sharp8','sharp16','drive2','drive4','visible_keys','context_query','sharp8_drive2')

def read(path):return json.loads(path.read_text())
def now():return datetime.now(timezone.utc).isoformat()

def baseline_reference(root):
    result=read(root/'result.json');review=read(root/'analysis/review.json')
    if not (result.get('full_training_coverage') and result.get('final_gradient_passed') and review['verdict']['numerical_and_coverage_pass']):
        raise ValueError('Require the explicitly completed singular-clipping control')
    run=Path(next(r['run_dir'] for r in read(root/'coverage.json') if r['name']=='input-energy_ep'))
    rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    probe=next(r for r in rows if r.get('stage')=='validation_probe' and r['step']==750)
    if not probe['complete'] or probe['count']!=256 or not (run/'checkpoints/step-0750.pt').exists():
        raise ValueError('Missing exact 750-update control')
    return dict(root=str(root),run_dir=str(run),probe_mse=probe['mse']['model'],step=750)

def select_confirmations(screens,baseline_mse):
    eligible=[v for v in screens if v.get('status')=='screen_complete' and v['probe_mse']<=.995*baseline_mse]
    return [v['name'] for v in sorted(eligible,key=lambda v:(v['probe_mse'],v['name']))[:2]]

class NativeRunner:
    def __init__(self,root,deadline,charged_roots):
        self.root=root;self.deadline=deadline;self.started=time.monotonic();self.rows=[];self.active=None
        self.charged_roots=set(charged_roots)|{root}
    def status(self,state,**extra):
        paths=list((self.root/self.active).glob('attempt-*/*/metrics.jsonl')) if self.active else []
        atomic_write_json(self.root/'status.json',dict(state=state,heartbeat_at=now(),stage=self.active,
            progress=dict(terminal_cases=len(self.rows),active_metric_bytes=sum(p.stat().st_size for p in paths)),
            elapsed_seconds=time.monotonic()-self.started,deadline_unix=self.deadline,**extra))
    def native(self,name,config,mode='train',weights=None,resume=None):
        self.active=name;attempts=[]
        for attempt in (1,2):
            if self.deadline-time.time()<20:raise TimeoutError('Shared explorer allowance exhausted')
            folder=self.root/name/f'attempt-{attempt:02d}';folder.mkdir(parents=True)
            cmd=[sys.executable,'-m','ebl',mode,'--config',str(config),'--output-dir',str(folder)]
            if weights is not None:cmd+=['--weights',str(weights)]
            if resume is not None:cmd+=['--resume',str(resume)]
            start=time.monotonic()
            with (folder/'run.log').open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,EBL_CASE_DEADLINE_UNIX=str(self.deadline)),stdout=log,stderr=subprocess.STDOUT)
                (self.root/'child.pid').write_text(str(child.pid))
                while child.poll() is None:
                    self.status('running',child_pid=child.pid)
                    if time.time()>=self.deadline:
                        child.terminate()
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(2)
            manifests=list(folder.glob('*/manifest.json'));run=manifests[0].parent if len(manifests)==1 else None
            row=dict(name=name,attempt=attempt,returncode=child.returncode,run_dir=str(run) if run else None,
                     command=cmd,seconds=time.monotonic()-start,outcome='execution_failure',integrity_passed=False)
            if run is not None:
                if (run/'artifacts/summary.json').exists():row['summary']=read(run/'artifacts/summary.json')
                if self.deadline-time.time()>2:
                    inspect=subprocess.run([sys.executable,'-m','ebl','runs','inspect',str(run),'--verify-artifacts','--require-complete','--json'],cwd=ROOT,capture_output=True,text=True,timeout=min(60,self.deadline-time.time()))
                    (folder/'inspection.json').write_text(inspect.stdout);(folder/'inspection.stderr').write_text(inspect.stderr)
                    row['integrity_passed']=inspect.returncode==0
            if child.returncode==0 and row['integrity_passed']:row['outcome']='terminal'
            attempts.append(dict(row))
            if not early_cuda_failure(row):break
        row['attempts']=attempts;self.rows.append(row);atomic_write_json(self.root/'coverage.json',self.rows)
        compact={k:v for k,v in row.items() if k not in ('summary','attempts','command')}
        with (self.root/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(compact)+'\n')
        print(json.dumps(compact),flush=True)
        if time.time()>=self.deadline:raise TimeoutError('Shared explorer deadline reached')
        if row['outcome']!='terminal':raise RuntimeError(f'Operational failure in {name}; preserve evidence and repair within remaining allowance')
        sizes={str(r):sum(p.stat().st_size for p in r.rglob('*') if p.is_file()) for r in self.charged_roots}
        if sizes[str(self.root)]>1_000_000_000 or sum(sizes.values())>2_000_000_000:raise RuntimeError('Declared storage allowance exceeded')
        return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True);parser.add_argument('--charge-root',type=Path,action='append',default=[])
    parser.add_argument('--seconds',type=int,default=5400);parser.add_argument('--deadline-unix',type=float)
    args=parser.parse_args()
    if not 60<=args.seconds<=5400:raise ValueError('Explorer allowance is at most90minutes')
    root=args.output_root.resolve();baseline=baseline_reference(args.baseline_root.resolve())
    if (root/'worker.pid').exists():raise RuntimeError('Never duplicate an existing explorer identity')
    root.mkdir(parents=True,exist_ok=True)
    deadline=min(time.time()+args.seconds,args.deadline_unix or float('inf'))
    if deadline<=time.time():raise TimeoutError('Approved explorer window expired')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (root/'worker.pid').write_text(str(os.getpid()))
    charged=[dict(root=str(p.resolve()),seconds=read(p/'result.json')['seconds']) for p in args.charge_root]
    atomic_write_json(root/'launch.json',dict(pid=os.getpid(),started=now(),deadline_unix=deadline,source_commit=source,
        baseline=baseline,charged_attempts=charged,variants=list(VARIANTS),seconds_limit=args.seconds,command=sys.argv))
    runner=NativeRunner(root,deadline,[args.baseline_root.resolve(),*(p.resolve() for p in args.charge_root)])
    screens=[];selected=[];confirmations=[]
    def finish(outcome,error=None):
        atomic_write_json(root/'screens.json',screens)
        atomic_write_json(root/'result.json',dict(outcome=outcome,error=error,source_commit=source,deadline_unix=deadline,
            seconds=time.monotonic()-runner.started,baseline=baseline,charged_attempts=charged,screens=screens,
            selected=selected,confirmations=confirmations,screening_complete=len(screens)==len(VARIANTS),
            confirmation_coverage_complete=all(v['coverage_complete'] and v['gradient_passed'] for v in confirmations) and len(confirmations)==len(selected)))
        runner.status('complete' if error is None else 'failed',outcome=outcome,error=error)
    try:
        import torch
        guard=torch.empty(1,device='cuda');torch.cuda.synchronize()
        atomic_write_json(root/'supervisor-gpu.json',dict(pid=os.getpid(),gpu=torch.cuda.get_device_name(),guard_bytes=guard.numel()*4))
        # No optimizer steps: the exact completed control supplies new force
        # measurements and a forward compatibility check, not another seed.
        replay=runner.native('baseline-replay',CONFIGS/'baseline-replay.json',resume=Path(baseline['run_dir'])/'checkpoints/final.pt')
        if not replay['summary']['coverage_complete']:raise RuntimeError('Baseline measurement replay incomplete')
        from experiments.review_energy_transformer_input_attention import equal
        old=torch.load(Path(baseline['run_dir'])/'checkpoints/final.pt',map_location='cpu',weights_only=False)
        new=torch.load(Path(replay['run_dir'])/'checkpoints/final.pt',map_location='cpu',weights_only=False)
        if any(not equal(old[k],new[k]) for k in ('model','optimizer','order_rng','mask_rng','step')):
            raise RuntimeError('Read-only baseline replay mutated learned/training state')
        original_summary=read(Path(baseline['run_dir'])/'artifacts/summary.json')
        if abs(original_summary['evaluation']['test']['mse']['model']-replay['summary']['evaluation']['test']['mse']['model'])>1e-8:
            raise RuntimeError('Default forward differs from the reused baseline')
        baseline['replay']=replay['run_dir']
        for name in VARIANTS:
            directory=CONFIGS/name;record=dict(name=name,status='pending')
            for arm in ('energy_ep','energy_bp'):
                smoke=runner.native(f'{name}/smoke-{arm}',directory/f'smoke-{arm}.json')
                if not smoke['summary']['coverage_complete']:
                    record.update(status='smoke_rejected',reason=smoke['summary'].get('failure') or smoke['summary']['stop_reason']);break
            if record['status']=='pending':
                warm=runner.native(f'{name}/warmup',directory/'warmup.json');record['warmup']=warm['run_dir']
                if not warm['summary']['coverage_complete']:record.update(status='warmup_rejected',reason=warm['summary'].get('failure') or warm['summary']['stop_reason'])
            if record['status']=='pending':
                diagnostic=runner.native(f'{name}/gradient-selection',CONFIGS/'gradient-selection.json','validate',Path(warm['run_dir'])/'checkpoints/final.pt')
                cap=qualified_cap(diagnostic['summary']);record.update(cap=cap,gradient_selection=diagnostic['run_dir'])
                if cap is None:record.update(status='gradient_rejected',reason='No cap passes unchanged global and routing-group reference gates')
            if record['status']=='pending':
                screen=runner.native(f'{name}/screen',directory/f'screen-{cap}.json');record['screen']=screen['run_dir']
                summary=screen['summary']
                if summary['coverage_complete']:
                    normal=summary['evaluation']['ablation_normal']['mse']['model']
                    record.update(status='screen_complete',probe_mse=normal,improvement_percent=100*(1-normal/baseline['probe_mse']),
                                  uniform_penalty_percent=100*(summary['evaluation']['ablation_uniform']['mse']['model']/normal-1))
                else:record.update(status='screen_incomplete',reason=summary.get('failure') or summary['stop_reason'])
            screens.append(record);atomic_write_json(root/'screens.json',screens)
        selected=select_confirmations(screens,baseline['probe_mse'])
        atomic_write_json(root/'selection.json',dict(selected=selected,rule='Two lowest valid750-update validation MSEs, each at least0.5% below explicit baseline; tie-break by name',baseline=baseline))
        for name in selected:
            cap=next(v['cap'] for v in screens if v['name']==name);record=dict(name=name,cap=cap)
            trained={}
            for arm in ('energy_ep','energy_bp'):
                trained[arm]=runner.native(f'{name}/full-{arm}',CONFIGS/name/f'full-{cap}-{arm}.json')
                record[arm]=trained[arm]['run_dir']
            record['coverage_complete']=all(v['summary']['coverage_complete'] for v in trained.values())
            record['gradient_passed']=False
            if trained['energy_ep']['summary']['coverage_complete']:
                final=runner.native(f'{name}/gradient-final',CONFIGS/f'gradient-final-{cap}.json','validate',Path(trained['energy_ep']['run_dir'])/'checkpoints/final.pt')
                record.update(gradient_final=final['run_dir'],gradient_passed=qualified_cap(final['summary'])==cap)
            confirmations.append(record);atomic_write_json(root/'confirmations.json',confirmations)
        complete=all(v['coverage_complete'] and v['gradient_passed'] for v in confirmations)
        budget_limited=any(v.get('reason')=='training_time_limit' for v in screens)
        finish('budget_incomplete' if budget_limited else 'exploration_collected' if complete else 'confirmation_incomplete')
    except TimeoutError as error:
        finish('budget_incomplete',str(error))
    except BaseException as error:
        finish('operational_failure',repr(error));raise

if __name__=='__main__':main()
