"""Native FP32 fuse-MNIST EP training; same cohorts and metrics as the ET pilot."""
from dataclasses import asdict
from pathlib import Path
import json
import math
import os
import time
import torch
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.schema.base import ConfigError
from ebl.schema.energy_transformer_mnist import CASES
from ebl.runtime.energy_transformer_mnist import strict_fp32
from networks.energy_transformer.mnist import load_raw, hidden_masks, batch
from networks.energy_transformer.fuse_gpu import FuseNetwork, settle, ep_gradient, Unsettled

ROOT=Path(__file__).resolve().parents[2]

def task_batch(images,hidden,patch,device,permutation=None):
    if permutation is not None:
        images=images.flatten(1)[:,permutation].reshape(-1,28,28)
        hidden=hidden.flatten(1)[:,permutation].reshape(-1,28,28)
    return batch(images,hidden,patch,device)

def save(path,model,opt,c,step,seconds,order_gen,mask_gen):
    temporary=path.with_suffix('.tmp')
    torch.save({'schema':'fuse-mnist.checkpoint.v1','config':asdict(c),'model':model.state_dict(),
                'optimizer':opt.state_dict(),'step':step,'train_seconds':seconds,
                'order_rng':order_gen.get_state(),'mask_rng':mask_gen.get_state()},temporary)
    temporary.replace(path)

def predict(model,x,t,mask,c,deadline,intervention='normal'):
    v=settle(model,x,t,mask,tolerance=c.residual_tolerance,max_steps=c.max_steps,
             deadline=deadline,intervention=intervention)
    return model.gain*v.o,v

@torch.no_grad()
def evaluate(model,images,masks,patch,c,deadline,permutation,*,initial=None,mean_image=None,intervention='normal'):
    totals={'model':0.};count=0;pixels=0.;worst=0.;iterations=0;failure=None;examples={}
    if initial is not None:totals['initial']=0.
    if mean_image is not None:totals['mean_image']=0.
    device=next(model.parameters()).device
    for start in range(0,len(images),c.batch_size):
        im=images[start:start+c.batch_size];hm=masks[start:start+c.batch_size]
        x,t,mask=task_batch(im,hm,patch,device,permutation)
        try:
            pred,v=predict(model,x,t,mask,c,deadline,intervention)
            predictions={'model':pred}
            if initial is not None:predictions['initial']=predict(initial,x,t,mask,c,deadline)[0]
        except (TimeoutError,Unsettled) as error:
            failure=str(error);break
        if mean_image is not None:
            predictions['mean_image']=task_batch(mean_image[None].expand(len(im),-1,-1),hm,patch,device,permutation)[1]
        for name,y in predictions.items():totals[name]+=((y-t).square()*mask).sum().item()/.8**2
        pixels+=mask.sum().item();count+=len(im);worst=max(worst,v.residual);iterations+=len(im)*v.iterations
        if not examples:
            examples={k:v[:16].cpu() for k,v in predictions.items()}
            examples.update(target=t[:16].cpu(),mask=mask[:16].cpu(),inputs=x[:16].cpu())
    mse={k:v/pixels for k,v in totals.items()} if pixels else {}
    return {'count':count,'requested':len(images),'complete':count==len(images),'mse':mse,
            'psnr':{k:-10*math.log10(max(v,1e-30)) for k,v in mse.items()},'worst_residual':worst,
            'mean_iterations':iterations/max(count,1),'failure':failure},examples

def train(c,store,images,ids,resume=None):
    strict_fp32()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; CPU fallback disabled')
    started=time.monotonic();deadline=started+c.wall_seconds
    if os.environ.get('EBL_CASE_DEADLINE_UNIX'):
        deadline=min(deadline,started+max(0,float(os.environ['EBL_CASE_DEADLINE_UNIX'])-time.time()-3))
    n,d,patch=CASES[c.case]
    kwargs=dict(tokens=n,width=d,patch_width=patch**2,heads=c.heads,seed=c.seed,kernel=c.kernel,
                floor=c.floor,sigma=c.fuse_sigma,fuse_g=c.fuse_conductance,gain=c.readout_gain,device='cuda')
    model=FuseNetwork(**kwargs);initial=FuseNetwork(**kwargs)
    opt=torch.optim.Adam(model.parameters(),lr=c.learning_rate)
    order_gen=torch.Generator().manual_seed(3101);mask_gen=torch.Generator().manual_seed(3102)
    vm=hidden_masks(5000,torch.Generator().manual_seed(3103));tm=hidden_masks(10000,torch.Generator().manual_seed(3104))
    permutation=None if c.permutation_seed is None else torch.randperm(784,generator=torch.Generator().manual_seed(c.permutation_seed))
    mean_image=images['train'][ids['train']].float().mean(0)
    val_images=images['train'][ids['validation']]
    artifacts=store.run_dir/'artifacts';checkpoints=store.run_dir/'checkpoints'
    torch.save({'ids':ids,'validation_masks':vm,'test_masks':tm,'permutation':permutation},artifacts/'cohorts.pt')
    atomic_write_json(artifacts/'environment.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'precision':'fp32','tf32':False,'amp':False,**model.counts(),'head_width':1,'kernel':c.kernel,
        'claim':'Ideal matched differential rails; numerical equilibrium solver, digital Adam. No transistor/latency/power validation.'})
    step=0;prior=0.
    if resume is not None:
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        if ck['schema']!='fuse-mnist.checkpoint.v1' or ck['config']!=asdict(c):
            raise ConfigError('Resume requires identical config')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        order_gen.set_state(ck['order_rng']);mask_gen.set_state(ck['mask_rng'])
        step,prior=ck['step'],ck['train_seconds']
    else:save(checkpoints/'initial.pt',model,opt,c,0,0.,order_gen,mask_gen)
    probe,_=evaluate(model,val_images[:c.probe_count],vm[:c.probe_count],patch,c,deadline,permutation,mean_image=mean_image)
    store.append_metric({'stage':'initial_validation','step':step,**probe})
    if not probe['complete']:raise Unsettled('Initial validation: '+str(probe['failure']))
    begin=time.monotonic();train_deadline=min(deadline-max(1,c.wall_seconds-c.train_seconds),begin+max(0,c.train_seconds-prior))
    stop='updates';failure=None;diag={};torch.cuda.reset_peak_memory_stats()
    while step<c.updates:
        if time.monotonic()>=train_deadline:stop='training_time_limit';break
        tick=time.monotonic()
        selection=ids['train'][torch.randint(len(ids['train']),(c.batch_size,),generator=order_gen)]
        hm=hidden_masks(c.batch_size,mask_gen)
        x,t,mask=task_batch(images['train'][selection],hm,patch,'cuda',permutation)
        opt.zero_grad(set_to_none=True)
        try:
            grad,phases=ep_gradient(model,x,t,mask,beta=c.beta,tolerance=c.residual_tolerance,
                                   max_steps=c.max_steps,deadline=train_deadline)
            if not all(bool(torch.isfinite(g).all()) for g in grad.values()):raise Unsettled('Nonfinite EP gradient')
            for name,p in model.named_parameters():p.grad=grad[name]
            opt.step();model.project()
        except TimeoutError as error:stop='training_time_limit';failure=str(error);break
        except Unsettled as error:stop='scientific_stop';failure=str(error);break
        step+=1;torch.cuda.synchronize();seconds=prior+time.monotonic()-begin
        w=model.conductance(phases[0].q,phases[0].k);active=w[...,model.offdiag.bool()]
        diag={'residual':max(v.residual for v in phases),'phase_iterations':[v.iterations for v in phases],
              'fuse_mean':active.mean().item(),'fuse_min':active.min().item(),'fuse_max':active.max().item()}
        row={'stage':'train','step':step,'mse':(((model.gain*phases[0].o-t).square()*mask).sum()/mask.sum()/.8**2).item(),
             'train_seconds':seconds,'update_seconds':time.monotonic()-tick,**diag}
        store.append_metric(row)
        if step==1 or step%25==0:print(json.dumps(row),flush=True)
        if step%250==0:
            save(checkpoints/f'step-{step:04d}.pt',model,opt,c,step,seconds,order_gen,mask_gen)
            probe,_=evaluate(model,val_images[:c.probe_count],vm[:c.probe_count],patch,c,train_deadline,permutation)
            store.append_metric({'stage':'validation_probe','step':step,'train_seconds':seconds,**probe})
            if not probe['complete']:stop='validation_stop';failure=probe['failure'];break
    seconds=prior+time.monotonic()-begin
    save(checkpoints/'final.pt',model,opt,c,step,seconds,order_gen,mask_gen)
    result={'case':c.case,'kernel':c.kernel,'updates':step,'stop_reason':stop,'failure':failure,
            'train_seconds':seconds,'peak_memory_bytes':torch.cuda.max_memory_allocated(),**model.counts(),
            'permutation_seed':c.permutation_seed,'last_training_diagnostics':diag,'evaluation':{}}
    test,examples=evaluate(model,images['test'][:c.evaluation_count],tm[:c.evaluation_count],patch,c,deadline,permutation,initial=initial,mean_image=mean_image)
    result['evaluation']['test']=test;torch.save(examples,artifacts/'reconstructions.pt')
    atomic_write_json(artifacts/'summary.json',result)
    result['evaluation']['validation']=evaluate(model,val_images[:c.validation_count],vm[:c.validation_count],patch,c,deadline,permutation,mean_image=mean_image)[0]
    for intervention in ('normal','linear','zero'):
        result['evaluation']['ablation_'+intervention]=evaluate(model,val_images[:c.probe_count],vm[:c.probe_count],patch,c,deadline,permutation,intervention=intervention)[0]
    result['coverage_complete']=step==c.updates and failure is None and all(r['complete'] for r in result['evaluation'].values())
    result['wall_seconds']=time.monotonic()-started
    atomic_write_json(artifacts/'summary.json',result)
    return result

def run_train(request):
    if any(getattr(request,n,None) is not None for n in ('weights','base_weights','teacher_weights','deployment','device_state','device_data','device_model')):
        raise ConfigError('Fuse MNIST uses declared initialization or exact resume')
    c=request.spec;images,ids,paths=load_raw(c.dataset_root)
    inputs=[{'path':str(p),'sha256':sha256_file(p)} for p in paths]
    if request.resume is not None:inputs.append({'path':str(request.resume),'sha256':sha256_file(request.resume)})
    store=RunStore.create(output_root=request.output_dir,experiment_id=c.experiment_id,resolved_config=asdict(c),
        command=request.command,repo_root=ROOT,input_artifacts=inputs,resume_capability='stateful_nondeterministic')
    print('RUN_DIR='+str(store.run_dir),flush=True)
    try:
        result=train(c,store,images,ids,request.resume)
        files=[p for folder in ('artifacts','checkpoints') for p in (store.run_dir/folder).rglob('*') if p.is_file()]
        arts=[store.artifact_record(p,kind='fuse_mnist') for p in files]
        if sum(a.size_bytes for a in arts)>500_000_000:raise RuntimeError('500 MB per-case storage cap')
        store.complete(metrics={'coverage_complete':result['coverage_complete'],'updates':result['updates'],'stop_reason':result['stop_reason']},artifacts=arts)
        return 0
    except BaseException as error:
        store.fail(error);raise
