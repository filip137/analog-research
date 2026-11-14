"""Independent numerical readiness, including FP32 EP against an implicit gradient."""
import argparse
import json
import time
from pathlib import Path
import torch
from networks.energy_transformer.gpu import EnergyNetwork, NAMES, settle, ep_gradient, triode_product, weak_bank, diode, diode_residual


def implicit_reference(model,x,target,mask,free):
    """Dual activation-coordinate energy, independently differentiated at a minimum."""
    b,n,d=free.z.shape
    basis=torch.linalg.qr(torch.eye(d,device=x.device,dtype=x.dtype)[:,:d-1]-
                          torch.eye(d,device=x.device,dtype=x.dtype)[:,[-1]])[0]
    count=b*n*(d-1)
    raw=torch.cat(((free.ports['g'] @ basis).flatten(),free.y.flatten())).detach().requires_grad_()
    params={name:p.detach().clone().requires_grad_() for name,p in model.named_parameters()}
    g=raw[:count].reshape(b,n,d-1) @ basis.T
    y=raw[count:].reshape_as(target)
    q=torch.einsum('bnd,had->bhna',g,params['query'])
    k=torch.einsum('bnd,had->bhna',g,params['key'])
    e=(-d*model.epsilon**.5*(1-g.square().sum(-1)/d).sqrt().sum()
       -(g*(x @ params['encoder'].T+params['position'])).sum()
       -torch.logsumexp(model.kappa*(q @ k.transpose(-1,-2)),-1).sum()/model.kappa
       -.5*torch.relu(g @ params['memory'].T).square().sum()
       +.5*y.square().sum()-(y*(g @ params['decoder'].T)).sum())
    stationarity,=torch.autograd.grad(e,raw,create_graph=True)
    hessian=torch.stack([torch.autograd.grad(v,raw,retain_graph=True)[0] for v in stationarity])
    dc,=torch.autograd.grad(.5*(mask*(y-target).square()).sum()/mask.sum(),raw,retain_graph=True)
    adjoint=torch.linalg.solve(hessian.detach(),dc.detach())
    gradients=torch.autograd.grad(stationarity,tuple(params.values()),grad_outputs=-adjoint)
    return dict(zip(params,(v.detach() for v in gradients))),{
        'stationarity':stationarity.detach().abs().max().item(),
        'minimum_eigenvalue':torch.linalg.eigvalsh(hessian.detach()).min().item()}


def gradient_check(device='cpu'):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    gen=torch.Generator(device=device).manual_seed(33)
    model=EnergyNetwork(tokens=3,width=4,patch_width=2,heads=1,device=device,dtype=torch.float64)
    x=torch.rand((2,3,4),generator=gen,device=device,dtype=torch.float64)-.5
    target=torch.rand((2,3,2),generator=gen,device=device,dtype=torch.float64)*.8-.4
    mask=torch.ones_like(target)
    free=settle(model,x,target,mask,tolerance=1e-11)
    expected,diagnostics=implicit_reference(model,x,target,mask,free)
    actual_model=EnergyNetwork(tokens=3,width=4,patch_width=2,heads=1,device=device)
    actual_model.load_state_dict(model.state_dict())
    actual,phases=ep_gradient(actual_model,x.float(),target.float(),mask.float())
    a=torch.cat([actual[name].double().flatten() for name in NAMES])
    e=torch.cat([expected[name].flatten() for name in NAMES])
    relative=((a-e).norm()/e.norm()).item()
    cosine=torch.nn.functional.cosine_similarity(a,e,dim=0).item()
    groups={name:((actual[name].double()-expected[name]).norm()/expected[name].norm()).item() for name in NAMES}
    # Independent directional FD of re-equilibrated loss, all parameter groups.
    fd={}
    for name in NAMES:
        p=getattr(model,name);original=p.detach().clone()
        direction=torch.randn(p.shape,device=device,dtype=torch.float64,generator=gen)
        direction/=direction.norm();costs=[]
        for sign in (-1,1):
            p.copy_(original+sign*1e-4*direction)
            state=settle(model,x,target,mask,tolerance=1e-11)
            costs.append((.5*(mask*(state.y-target).square()).sum()/mask.sum()).item())
        p.copy_(original)
        numerical=(costs[1]-costs[0])/2e-4
        predicted=(expected[name]*direction).sum().item()
        fd[name]=abs(numerical-predicted)/max(abs(predicted),1e-8)
    return {'passed':cosine>.99 and relative<.05 and max(groups.values())<.05
                     and max(fd.values())<.01 and diagnostics['minimum_eigenvalue']>0
                     and diagnostics['stationarity']<1e-8,
            'cosine':cosine,'relative_error':relative,'group_relative_errors':groups,
            'finite_difference_relative_errors':fd,'reference':diagnostics,
            'fp32_phase_residuals':[p.residual for p in phases],
            'fp32_phase_steps':[p.iterations.tolist() for p in phases]}


def device_checks(device='cpu'):
    x=torch.linspace(-4,4,101,device=device,dtype=torch.float64)
    products=(triode_product(x[:,None],x[None,:])-x[:,None]*x[None,:]).abs().max().item()
    logits=torch.stack((x,x.flip(0)))
    p,source,margin=weak_bank(logits)
    softmax_error=(p-logits.softmax(-1)).abs().max().item()
    drive=torch.tensor([-2.,-.001,0.,.001,2.],device=device)
    residual=diode_residual(drive,diode(drive)).item()
    return {'passed':products<1e-10 and softmax_error<1e-12 and margin.min()>0 and residual==0,
            'triode_absolute_error':products,'softmax_absolute_error':softmax_error,
            'weak_inversion_margin':margin.min().item(),'diode_residual':residual}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    begin=time.monotonic()
    result={'gradient':gradient_check(args.device),'devices':device_checks(args.device)}
    result['devices']['passed']=bool(result['devices']['passed'])
    result.update(passed=all(v['passed'] for v in result.values()),seconds=time.monotonic()-begin)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':
    main()
