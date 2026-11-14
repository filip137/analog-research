"""Independent dense implicit derivative of the reduced physical KCL equations."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,time
import torch
from ebl.runtime.store import atomic_write_json,sha256_file
from ebl.runtime.energy_transformer_mnist import strict_fp32
from ebl.runtime.fuse_mnist import task_batch
from ebl.schema.fuse_mnist import Config
from ebl.schema.energy_transformer_mnist import CASES
from networks.energy_transformer.mnist import load_raw,hidden_masks
from networks.energy_transformer.fuse_gpu import FuseNetwork,Phase,settle,ep_gradient


def exact_gradient(model,x,t,mask):
    """One example, FP64 dense KCL Jacobian; independent of EP statistics."""
    free=settle(model,x,t,mask,tolerance=1e-11,max_steps=512)
    shapes=[a.shape for a in free.tensors()];sizes=[a.numel() for a in free.tensors()]
    def unpack(s):return Phase(*(a.reshape(sh) for a,sh in zip(s.split(sizes),shapes)))
    state=torch.cat([a.flatten() for a in free.tensors()]).requires_grad_()
    def current(s):return torch.cat([a.flatten() for a in model.force(unpack(s),x,t,mask)])
    # Chunking bounds the dense-derivative intermediate memory.
    jac=torch.func.jacrev(current,chunk_size=32)(state).detach()
    symmetry=(jac-jac.T).abs().max().item()
    h=-jac
    _,info=torch.linalg.cholesky_ex(h)
    if info.item()!=0:raise RuntimeError('Reference is not a stable isolated energy minimum')
    loss=.5*((model.gain*unpack(state).o-t).square()*mask).sum()/mask.sum()
    dl,=torch.autograd.grad(loss,state)
    adj=torch.linalg.solve(h,dl).detach()
    for p in model.parameters():p.requires_grad_(True)
    grads=torch.autograd.grad((adj*current(state.detach())).sum(),tuple(model.parameters()))
    for p in model.parameters():p.requires_grad_(False)
    return dict(zip(dict(model.named_parameters()),grads)),free,{'jacobian_symmetry_error':symmetry,'reference_residual':free.residual}


def agreement(a,b):
    a,b=a.double().flatten(),b.double().flatten()
    norm=b.norm().item()
    if norm==0:
        return {'cosine':1. if a.norm()==0 else 0.,'relative_l2':a.norm().item(),'reference_norm':norm}
    return {'cosine':torch.nn.functional.cosine_similarity(a,b,dim=0,eps=1e-30).item(),
            'relative_l2':((a-b).norm()/b.norm()).item(),'reference_norm':norm}


def check(config,checkpoint=None,count=1):
    strict_fp32();n,d,patch=CASES[config.case]
    kw=dict(tokens=n,width=d,patch_width=patch**2,heads=config.heads,seed=config.seed,kernel=config.kernel,
            floor=config.floor,sigma=config.fuse_sigma,fuse_g=config.fuse_conductance,gain=config.readout_gain,device='cuda')
    m=FuseNetwork(**kw);ref=FuseNetwork(**kw,dtype=torch.float64)
    if checkpoint is not None:m.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model'])
    ref.load_state_dict(m.state_dict())
    images,ids,_=load_raw(config.dataset_root)
    hm=hidden_masks(5000,torch.Generator().manual_seed(3103))[:count]
    perm=None if config.permutation_seed is None else torch.randperm(784,generator=torch.Generator().manual_seed(config.permutation_seed))
    x,t,mask=task_batch(images['train'][ids['validation'][:count]],hm,patch,'cuda',perm)
    approx,phases=ep_gradient(m,x,t,mask,beta=config.beta,tolerance=config.residual_tolerance,max_steps=config.max_steps)
    exact={k:torch.zeros_like(v,dtype=torch.float64) for k,v in approx.items()};rows=[];state_error=0.
    started=time.monotonic()
    for i in range(count):
        g,fr,di=exact_gradient(ref,x[i:i+1].double(),t[i:i+1].double(),mask[i:i+1].double())
        for name in exact:exact[name]+=g[name]/count
        state_error=max(state_error,max((a[i:i+1].double()-b).abs().max().item() for a,b in zip(phases[0].tensors(),fr.tensors())))
        rows.append(di)
    groups={name:agreement(approx[name],exact[name]) for name in exact}
    global_result=agreement(torch.cat([v.flatten() for v in approx.values()]),torch.cat([exact[k].flatten() for k in approx]))
    passed=all(r['cosine']>.99 and r['relative_l2']<.05 for r in [global_result,*groups.values()])
    return {'passed':passed,'case':config.case,'kernel':config.kernel,'count':count,'config':asdict(config),
            'checkpoint':None if checkpoint is None else str(checkpoint),
            'checkpoint_sha256':None if checkpoint is None else sha256_file(checkpoint),
            'global':global_result,'groups':groups,'max_forward_error':state_error,
            'references':rows,'phase_iterations':[p.iterations for p in phases],
            'phase_residual':max(p.residual for p in phases),'seconds':time.monotonic()-started}


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='small');p.add_argument('--kernel',default='gaussian')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--count',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    c=Config(case=a.case,kernel=a.kernel)
    if a.checkpoint is not None:c=Config(**torch.load(a.checkpoint,map_location='cpu',weights_only=False)['config'])
    result=check(c,a.checkpoint,a.count);atomic_write_json(a.output,result)
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
