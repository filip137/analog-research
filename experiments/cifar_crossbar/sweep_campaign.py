"""Freeze native sweep arms, and run one exclusively owned remote arm."""
import argparse, json, os, subprocess, time
from dataclasses import asdict
from pathlib import Path
from experiments.artifacts import atomic_write_json, sha256_file
from experiments.cifar_crossbar.sweep_config import default_config, EXPERIMENT_ID, LRS, STRENGTHS


def prepare(root,dataset,backend,suffix,smoke=False,revision=1):
    root=Path(root)
    if root.exists(): raise ValueError('Cannot overwrite frozen campaign.')
    common=dict(dataset=dataset,backend=backend,suffix=suffix,device='cuda:0',disable_cudnn=(revision==1),
                calibration_lr=.0003 if dataset=='cifar10' else .001)
    if smoke: common.update(max_examples=128,hwa_epochs=1,extension_epochs=0,validation_every=1,
                           fault_rates_ppm=(0,50000),array_seed=271001,endpoint_seed=281001)
    nodes=[]
    def add(name,stage,**extra):
        config=default_config(**{**common,'stage':stage,**extra})
        path=root/'configs'/f'{name}.json';atomic_write_json(path,config)
        nodes.append({'id':name,'stage':stage,'config':str(path.relative_to(root)),'sha256':sha256_file(path)})
    add('cache','cache')
    if backend=='om':add('kernel','kernel')
    for i,lr in enumerate(LRS):
        for j,strength in enumerate(STRENGTHS): add(f'generic_{i}_{j}','fit',candidate_learning_rate=lr,candidate_noise=strength)
    add('select_generic','select_generic')
    for kind in ('open','gmax','random'):
        for i,lr in enumerate(LRS): add(f'cdt_{kind}_{i}','fit',candidate_family=kind,candidate_learning_rate=lr)
    add('sources','sources')
    for i in range(1,2 if smoke else 4):
        add(f'screen{i}','screen',array_seed=271001 if smoke else 251000+i,endpoint_seed=281001 if smoke else 261000+i)
    depth=4 if suffix=='last_two_blocks' else 8
    study_id=f'{dataset}-{backend}-conv{depth}-fault-sweep'+('-canary' if smoke else '')+f'-v{revision}'
    plan={'schema_version':1,'study_id':study_id,'title':study_id,'evidence_class':'exploratory',
          'hypothesis':'Array-specific recovery can improve teacher KL beyond calibrated corruption-aware HWA at some failure rates and analog depths.',
          'motivation':'Approved docs/cifar_fault_rate_sweep.md; CIFAR-10 first, then CIFAR-100.',
          'arms':[{'arm_id':n['id'],'configs':[n['config']],'description':n['id'],'experiment_id':EXPERIMENT_ID,'mode':'train'} for n in nodes],
          'completion_criteria':['All declared native arms complete and artifacts verify.',
              'Canary: 90 controls on one array.' if smoke else 'Three final arrays; 270 controls each, 216 learning trajectories each; all five full epochs and full test set.',
              'Unchanged prefix, fixed layer-bound nested faults, exact P0 pairing, device-specific costs and checkpoint replay.'],
          'analysis_plan':['Original teacher KL at T1, accuracy, teacher agreement, per-array spread and paired differences.',
              'Separate all HWA-only/calibration/rewrite controls from weight and joint recovery; report convergence limits.',
              'Corruption-aware sources train equal mixed 2/3/5% rates of one kind; 1% is an unselected reference.']}
    atomic_write_json(root/'study-plan.json',plan)
    out={'schema':'cifar_crossbar.campaign.v1','phase':'fault_sweep','study_id':study_id,
         'dataset':dataset,'backend':backend,'suffix':suffix,'smoke':smoke,'plan':'study-plan.json',
         'plan_sha256':sha256_file(root/'study-plan.json'),'nodes':nodes}
    atomic_write_json(root/'campaign.json',out)
    return out


def remote_node(job_path):
    """Durable launcher wrapper. Numerical work only enters through public EBL."""
    job_path=Path(job_path);job=json.loads(job_path.read_text());root=Path(job['root'])
    state=root/'launch'/job['id'];state.mkdir(parents=True,exist_ok=True)
    lock=state/'owner.json'
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'started':time.time(),'job':job},f)
    try:
        for item in job['inputs'].values():
            if sha256_file(root/item['path'])!=item['sha256']:raise ValueError('Input changed: '+item['path'])
        config=root/job['config']
        if sha256_file(config)!=job['config_sha256']:raise ValueError('Config changed.')
        arm=root/'results'/job['study_id']/'runs'/job['node']
        before=set(arm.iterdir()) if arm.exists() else set()
        cmd=[job['python'],'-m','ebl','train','--config',str(config),'--output-dir',str(arm)]
        for role,item in job['inputs'].items():cmd+=['--'+role,str(root/item['path'])]
        env=os.environ.copy();env.update(EBL_SOURCE_RECEIPT=str(root/'source-receipt.json'),
            EBL_CIFAR_ROOT=str(root/'inputs/data'),EBL_DEFER_CURRENT_SIMULATIONS='1',
            CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (state/'native.log').open('w') as log:
            p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic_write_json(state/'handle.json',{'pid':p.pid,'wrapper_pid':os.getpid(),'command':cmd,'started':time.time()})
            code=p.wait()
        created=set(arm.iterdir())-before
        if code or len(created)!=1:raise RuntimeError(f'Native exit {code}; created {len(created)} directories.')
        run=created.pop();status=json.loads((run/'status.json').read_text())
        if status['status']!='complete':raise RuntimeError('Native did not complete.')
        atomic_write_json(state/'exit.json',{'exit_code':0,'run':str(run.relative_to(root)),'finished':time.time()})
    except BaseException as e:
        atomic_write_json(state/'exit.json',{'exit_code':1,'error':repr(e),'finished':time.time()});raise


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',type=Path,required=True)
    a.add_argument('--dataset',choices=['cifar10','cifar100'],required=True);a.add_argument('--backend',choices=['pcm','om'],required=True)
    a.add_argument('--suffix',choices=['last_two_blocks','last_four_blocks'],required=True);a.add_argument('--smoke',action='store_true');a.add_argument('--revision',type=int,default=1)
    a=sub.add_parser('remote-node');a.add_argument('--job',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare': print(json.dumps(prepare(a.output,a.dataset,a.backend,a.suffix,a.smoke,a.revision)))
    else:remote_node(a.job)

if __name__=='__main__':main()
