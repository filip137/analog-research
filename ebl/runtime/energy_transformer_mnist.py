"""Public GPU-only MNIST comparison; scientific stops remain terminal evidence."""
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
from networks.energy_transformer.gpu import EnergyNetwork, DigitalNetwork, EquilibriumError, settle, ep_gradient
from networks.energy_transformer.mnist import load_raw, hidden_masks, batch
from networks.energy_transformer.finite_time import relax, finite_ep

ROOT=Path(__file__).resolve().parents[2]


def strict_fp32():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_default_dtype(torch.float32)


def save_checkpoint(path,model,optimizer,config,step,train_seconds,order_gen,mask_gen):
    payload={'schema':'energy-transformer-mnist.checkpoint.v1','config':asdict(config),
             'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
             'train_seconds':train_seconds,'order_rng':order_gen.get_state(),
             'mask_rng':mask_gen.get_state(),'torch_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def restore_checkpoint(path,model,optimizer,config,order_gen,mask_gen):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['schema']!='energy-transformer-mnist.checkpoint.v1' or {'phase_mode':'settled',**payload['config']}!=asdict(config):
        raise ConfigError('Resume requires the identical scientific configuration')
    model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer'])
    order_gen.set_state(payload['order_rng']);mask_gen.set_state(payload['mask_rng'])
    torch.set_rng_state(payload['torch_rng'])
    if payload['cuda_rng']:
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload['step'],payload['train_seconds']


@torch.no_grad()
def predict(model,x,target,mask,config,deadline,attention='normal'):
    if isinstance(model,EnergyNetwork):
        if config.phase_mode=='fixed_steps':
            phase=relax(model,x,target,mask,config.max_steps,deadline=deadline,attention=attention)
        else:
            phase=settle(model,x,target,mask,tolerance=config.residual_tolerance,
                         max_steps=config.max_steps,deadline=deadline,attention=attention)
        return phase.y,{'residual':phase.residual,'iterations':phase.iterations.float().mean().item(),
                        'maximum_operand':phase.maximum_operand,'minimum_triode_margin':phase.minimum_triode_margin}
    return model(x,attention),{}


@torch.no_grad()
def evaluate(model,images,masks,patch,config,deadline,*,initial=None,mean_image=None,attention='normal'):
    sums={'model':0.}
    if initial is not None: sums['initial']=0.
    if mean_image is not None: sums['mean_image']=0.
    pixels=0.;count=0;worst=0.;iterations=0.;examples=[];failure=None
    for start in range(0,len(images),config.batch_size):
        if time.monotonic()>=deadline:
            break
        im=images[start:start+config.batch_size];hm=masks[start:start+config.batch_size]
        x,target,mask=batch(im,hm,patch,next(model.parameters()).device)
        try:
            prediction,diag=predict(model,x,target,mask,config,deadline,attention)
            predictions={'model':prediction}
            if initial is not None:
                predictions['initial']=predict(initial,x,target,mask,config,deadline)[0]
        except TimeoutError:
            break
        except EquilibriumError as error:
            failure=str(error)
            break
        if mean_image is not None:
            predictions['mean_image']=batch(mean_image[None].expand(len(im),-1,-1),hm,patch,x.device)[1]
        for key,value in predictions.items():
            sums[key]+=((value-target).square()*mask).sum().item()/.8**2
        pixels+=mask.sum().item();count+=len(im)
        worst=max(worst,diag.get('residual',0.));iterations+=len(im)*diag.get('iterations',0.)
        if not examples:
            examples={k:v[:16].cpu() for k,v in predictions.items()}
            examples.update(target=target[:16].cpu(),mask=mask[:16].cpu(),inputs=x[:16].cpu())
    mse={key:value/pixels for key,value in sums.items()} if pixels else {}
    return {'count':count,'requested':len(images),'complete':count==len(images),'mse':mse,
            'psnr':{key:-10*math.log10(max(value,1e-30)) for key,value in mse.items()},
            'worst_residual':worst,'mean_iterations':iterations/max(count,1),'failure':failure},examples


def train(config,store,images,ids,resume=None):
    strict_fp32()
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA; CPU fallback is disabled')
    started=time.monotonic();deadline=started+config.wall_seconds
    if os.environ.get('EBL_CASE_DEADLINE_UNIX'):
        deadline=min(deadline,started+max(0,float(os.environ['EBL_CASE_DEADLINE_UNIX'])-time.time()-5))
    n,d,patch=CASES[config.case]
    cls=EnergyNetwork if config.arm=='energy_ep' else DigitalNetwork
    kwargs=dict(tokens=n,width=d,patch_width=patch**2,heads=4,seed=config.seed,epsilon=config.epsilon,device='cuda')
    model=cls(**kwargs);initial=cls(**kwargs)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    order_gen=torch.Generator().manual_seed(3101)
    mask_gen=torch.Generator().manual_seed(3102)
    validation_masks=hidden_masks(5000,torch.Generator().manual_seed(3103))
    test_masks=hidden_masks(10000,torch.Generator().manual_seed(3104))
    mean_image=images['train'][ids['train']].float().mean(0)
    artifacts=store.run_dir/'artifacts';checkpoints=store.run_dir/'checkpoints'
    torch.save({'ids':ids,'validation_masks':validation_masks,'test_masks':test_masks},artifacts/'cohorts.pt')
    atomic_write_json(artifacts/'environment.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
                      'precision':'fp32','tf32':False,'amp':False,'core_parameters':6*d*d,
                      'total_parameters':sum(p.numel() for p in model.parameters()),
                      'model':config.arm,'dtype':str(next(model.parameters()).dtype)})
    step=0;prior_train=0.
    if resume is not None:
        step,prior_train=restore_checkpoint(resume,model,optimizer,config,order_gen,mask_gen)
    else:
        save_checkpoint(checkpoints/'initial.pt',model,optimizer,config,0,0.,order_gen,mask_gen)
    val_images=images['train'][ids['validation']]
    probe,unused=evaluate(model,val_images[:config.probe_count],validation_masks[:config.probe_count],
                          patch,config,deadline,mean_image=mean_image)
    store.append_metric({'stage':'initial_validation','step':step,**probe})
    training_started=time.monotonic()
    training_deadline=min(deadline-600 if config.wall_seconds>600 else deadline-1,
                          training_started+max(0,config.train_seconds-prior_train))
    stop='updates';failure=None;last_diag={};train_seconds=prior_train
    torch.cuda.reset_peak_memory_stats()
    while step<config.updates:
        if time.monotonic()>=training_deadline:
            stop='training_time_limit';break
        begin=time.monotonic()
        # Sampling with replacement gives identical replayable batches in both arms.
        selection=ids['train'][torch.randint(len(ids['train']),(config.batch_size,),generator=order_gen)]
        hidden=hidden_masks(config.batch_size,mask_gen)
        x,target,mask=batch(images['train'][selection],hidden,patch)
        optimizer.zero_grad(set_to_none=True)
        try:
            if config.arm=='energy_ep':
                if config.phase_mode=='fixed_steps':
                    gradients,phases=finite_ep(model,x,target,mask,config.max_steps,beta=config.beta,deadline=training_deadline)
                else:
                    gradients,phases=ep_gradient(model,x,target,mask,beta=config.beta,
                        tolerance=config.residual_tolerance,max_steps=config.max_steps,deadline=training_deadline)
                for name,p in model.named_parameters(): p.grad=gradients[name]
                prediction=phases[0].y
                last_diag={'phase_mode':config.phase_mode,'all_phases_settled':all(p.residual<=config.residual_tolerance for p in phases),
                           'phase_iterations':[p.iterations.float().mean().item() for p in phases],
                           'phase_max_iterations':[p.iterations.max().item() for p in phases],
                           'residual':max(p.residual for p in phases),
                           'maximum_operand':max(p.maximum_operand for p in phases),
                           'minimum_triode_margin':min(p.minimum_triode_margin for p in phases),
                           'maximum_energy_increase':max(p.energy_increase for p in phases)}
            else:
                prediction=model(x)
                (.5*((prediction-target).square()*mask).sum()/mask.sum()).backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise EquilibriumError('Nonfinite parameter gradient')
            optimizer.step()
        except TimeoutError:
            stop='training_time_limit';break
        except EquilibriumError as error:
            stop='scientific_stop';failure=str(error);break
        step+=1
        torch.cuda.synchronize()
        train_seconds=prior_train+time.monotonic()-training_started
        elapsed=time.monotonic()-begin
        row={'stage':'train','step':step,'mse':(((prediction.detach()-target).square()*mask).sum()/mask.sum()/.8**2).item(),
             'train_seconds':train_seconds,'update_seconds':elapsed,'images_per_second':len(x)/elapsed,
             'peak_memory_bytes':torch.cuda.max_memory_allocated(),**last_diag}
        store.append_metric(row)
        if step==1 or step%25==0:
            print(json.dumps(row),flush=True)
        if step%250==0:
            save_checkpoint(checkpoints/f'step-{step:04d}.pt',model,optimizer,config,step,train_seconds,order_gen,mask_gen)
            probe,_=evaluate(model,val_images[:config.probe_count],validation_masks[:config.probe_count],patch,config,training_deadline)
            store.append_metric({'stage':'validation_probe','step':step,'train_seconds':train_seconds,**probe})
    train_seconds=prior_train+time.monotonic()-training_started
    save_checkpoint(checkpoints/'final.pt',model,optimizer,config,step,train_seconds,order_gen,mask_gen)
    result={'phase_mode':config.phase_mode,'max_steps':config.max_steps,'arm':config.arm,'case':config.case,'stop_reason':stop,'failure':failure,'updates':step,
            'train_seconds':train_seconds,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
            'total_parameters':sum(p.numel() for p in model.parameters()),'core_parameters':6*d*d,
            'last_training_diagnostics':last_diag,'evaluation':{},
            'claim_boundary':'FP32 GPU simulation of ideal circuit equations and digital Adam; no SPICE, PDK, silicon, analog time or power measurements.'}
    # Final test and initialization are evaluated on exactly the same accepted prefix.
    try:
        test,examples=evaluate(model,images['test'][:config.evaluation_count],test_masks[:config.evaluation_count],
                               patch,config,deadline,initial=initial,mean_image=mean_image)
        result['evaluation']['test']=test
        torch.save(examples,artifacts/'reconstructions.pt')
        atomic_write_json(artifacts/'summary.json',result)
        validation,_=evaluate(model,val_images[:config.validation_count],validation_masks[:config.validation_count],
                               patch,config,deadline,mean_image=mean_image)
        result['evaluation']['validation']=validation
        for attention in ('normal','zero','uniform'):
            ablation,_=evaluate(model,val_images[:config.probe_count],validation_masks[:config.probe_count],
                                 patch,config,deadline,attention=attention)
            result['evaluation'][f'ablation_{attention}']=ablation
    except EquilibriumError as error:
        result['evaluation_failure']=str(error)
    result['coverage_complete']=(step==config.updates and result['evaluation'].get('test',{}).get('complete',False)
                                and result['evaluation'].get('validation',{}).get('complete',False)
                                and len(result['evaluation'])==5 and failure is None
                                and all(v['complete'] for v in result['evaluation'].values()))
    result['wall_seconds']=time.monotonic()-started
    atomic_write_json(artifacts/'summary.json',result)
    return result


def run_train(request):
    if any(getattr(request,name,None) is not None for name in ('weights','base_weights','teacher_weights',
           'deployment','device_state','device_data','device_model')):
        raise ConfigError('This benchmark starts from its declared initialization or an exact resume checkpoint')
    config=request.spec
    images,ids,paths=load_raw(config.dataset_root)
    inputs=[{'path':str(path),'sha256':sha256_file(path)} for path in paths]
    if request.resume is not None:
        inputs.append({'path':str(request.resume),'sha256':sha256_file(request.resume)})
    store=RunStore.create(output_root=request.output_dir,experiment_id=config.experiment_id,
        resolved_config=asdict(config),command=request.command,repo_root=ROOT,input_artifacts=inputs,
        resume_capability='stateful_nondeterministic')
    print(f'RUN_DIR={store.run_dir}',flush=True)
    try:
        result=train(config,store,images,ids,request.resume)
        paths=[p for folder in ('artifacts','checkpoints') for p in (store.run_dir/folder).rglob('*') if p.is_file()]
        artifacts=[store.artifact_record(p,kind='energy_transformer_mnist') for p in paths]
        if sum(a.size_bytes for a in artifacts)>500_000_000:
            raise RuntimeError('Per-case 500 MB storage budget exceeded')
        store.complete(metrics={'stop_reason':result['stop_reason'],'coverage_complete':result['coverage_complete'],
                                'updates':result['updates']},artifacts=artifacts)
        return 0
    except BaseException as error:
        store.fail(error)
        raise
