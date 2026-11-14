"""Finite-time EP and BPTT through the same ideal energy dynamics.

An accepted step size is locally constant: BPTT differentiates the state map,
not the discrete backtracking decisions. Finite-time EP is an approximation
and does not assert that its terminal states are equilibria.
"""
import time
import torch
from networks.energy_transformer.gpu import Equilibrium, EquilibriumError, NAMES


def relax(model,x,target,mask,steps,beta=0.,initial=None,*,differentiate=False,
          deadline=None,attention='normal'):
    if type(steps) is not int or steps<1:
        raise ValueError('Expected positive iteration count')
    with torch.enable_grad() if differentiate else torch.no_grad():
        z=x.new_zeros((len(x),model.tokens,model.width)) if initial is None else initial.z.detach().clone()
        y=torch.zeros_like(target) if initial is None else initial.y.detach().clone()
        maximum=x.new_tensor(0.);increase=x.new_tensor(0.)
        allowance=8*torch.finfo(x.dtype).eps
        for step in range(steps):
            if step%16==0 and deadline is not None and time.monotonic()>=deadline:
                raise TimeoutError('Finite-time phase deadline reached')
            v=model.ports(z,y,attention)
            dz,dy=model.force(v,x,target,mask,beta)
            with torch.no_grad():
                maximum=torch.maximum(maximum,torch.maximum(v['q'].abs().amax(),v['k'].abs().amax()))
                if not torch.isfinite(dz).all() or not torch.isfinite(dy).all():
                    raise EquilibriumError('Nonfinite finite-time state')
                if maximum>4:
                    raise EquilibriumError('Finite-time phase left the declared triode domain')
                energy=model.energy(v,x,target,mask,beta,attention)
                dt=x.new_full((len(x),1,1),.5)
                for _ in range(20):
                    candidate=model.ports(z+dt*dz,y+dt*dy/1.3,attention)
                    new_energy=model.energy(candidate,x,target,mask,beta,attention)
                    accepted=torch.isfinite(new_energy)&(new_energy<=energy+allowance*(1+energy.abs()))
                    if accepted.all():break
                    dt=torch.where(accepted[:,None,None],dt,dt*.5)
                else:
                    raise EquilibriumError('Finite-time energy backtracking failed')
                increase=torch.maximum(increase,(new_energy-energy).max())
            z,y=z+dt*dz,y+dt*dy/1.3
        v=model.ports(z,y,attention)
        with torch.no_grad():
            dz,dy=model.force(v,x,target,mask,beta)
            residual=torch.maximum(dz.abs().amax(),dy.abs().amax()).item()
            maximum=torch.maximum(maximum,torch.maximum(v['q'].abs().amax(),v['k'].abs().amax()))
            if not torch.isfinite(z).all() or not torch.isfinite(y).all() or maximum>4:
                raise EquilibriumError('Invalid final finite-time circuit state')
            m=maximum.item()
        return Equilibrium(z,y,v,torch.full((len(x),),steps,device=x.device,dtype=torch.long),
                           residual,m,.7-.1*m,increase.item())


@torch.no_grad()
def finite_ep(model,x,target,mask,steps,*,beta=.005,deadline=None):
    if beta<=0 or beta>=1:raise ValueError('Expected 0 < beta < 1')
    free=relax(model,x,target,mask,steps,deadline=deadline)
    plus=relax(model,x,target,mask,steps,beta,initial=free,deadline=deadline)
    minus=relax(model,x,target,mask,steps,-beta,initial=free,deadline=deadline)
    high,low=model.statistics(plus.ports,x),model.statistics(minus.ports,x)
    gradient={name:(high[name]-low[name])/(2*beta*mask.sum()) for name in NAMES}
    return gradient,(free,plus,minus)


def bptt_gradient(model,x,target,mask,steps,*,deadline=None):
    flags=[p.requires_grad for p in model.parameters()]
    try:
        for p in model.parameters():p.requires_grad_(True)
        phase=relax(model,x,target,mask,steps,differentiate=True,deadline=deadline)
        cost=.5*(mask*(phase.y-target).square()).sum()/mask.sum()
        values=torch.autograd.grad(cost,tuple(model.parameters()),allow_unused=True)
        gradient={name:g.detach() if g is not None else torch.zeros_like(p)
                  for (name,p),g in zip(model.named_parameters(),values)}
        if not all(torch.isfinite(g).all() for g in gradient.values()):
            raise EquilibriumError('Nonfinite BPTT gradient')
        phase.z=phase.z.detach();phase.y=phase.y.detach()
        phase.ports={name:value.detach() for name,value in phase.ports.items()}
        return gradient,phase
    finally:
        for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)


def agreement(estimate,reference):
    def compare(a,b):
        a,b=a.double().flatten(),b.double().flatten()
        an,bn=a.norm().item(),b.norm().item()
        return {'cosine':(torch.dot(a,b)/(an*bn)).item() if min(an,bn)>1e-14 else None,
                'relative_l2':(a-b).norm().item()/bn if bn>1e-14 else None,
                'norm_ratio':an/bn if bn>1e-14 else None,'estimate_l2':an,'reference_l2':bn}
    return {'global':compare(torch.cat([estimate[n].flatten() for n in NAMES]),
                             torch.cat([reference[n].flatten() for n in NAMES])),
            'groups':{name:compare(estimate[name],reference[name]) for name in NAMES}}


def select_cap(rows,caps,batches,threshold=.9):
    """Smallest tested equal phase cap passing every held-out minibatch."""
    for cap in caps:
        selected=[row for row in rows if row.get('iterations')==cap]
        if len(selected)!=batches or {row['batch'] for row in selected}!=set(range(batches)):
            continue
        if any(not row.get('reference_valid') for row in selected):continue
        if all(row[key]['global']['cosine'] is not None and row[key]['global']['cosine']>=threshold
               for row in selected for key in ('ep_vs_bptt','ep_vs_reference','bptt_vs_reference')):
            return cap
    return None
