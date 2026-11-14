"""Preserve feasible singular directions when enforcing the same norm budgets.

Each matrix is projected onto its selected spectral-norm ball by clipping
singular values. For the coupled attention budget, target radii still come
from the existing common norm-reduction rule; this is not claimed to be the
Euclidean projection onto the entire joint nonconvex feasible set.
"""
import math
import torch
from networks.energy_transformer.contractive import BUDGETS
from networks.energy_transformer.gpu import EquilibriumError
from networks.energy_transformer.input_attention import norms, certificate


@torch.no_grad()
def clip_singular(matrix, radius):
    target=torch.as_tensor(radius,dtype=matrix.dtype,device=matrix.device)
    if not torch.isfinite(target).all() or (target<0).any():
        raise ValueError('Spectral radius must be finite and nonnegative')
    u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
    clipped=torch.minimum(s,target[...,None])
    active=s[...,0]>target
    reconstructed=(u*clipped.unsqueeze(-2))@vh
    matrix.copy_(torch.where(active[...,None,None],reconstructed,matrix))
    return (s>target[...,None]).sum(-1).tolist()


@torch.no_grad()
def project(model,beta=.005):
    if not 0<beta<1:
        raise ValueError('Expected 0 < beta < 1')
    n=norms(model);before=certificate(model,n,beta)
    cap=4/math.sqrt(model.width)
    clipped={}
    for name in ('query','key','value','output'):
        if (n[name]>cap).any():
            clipped[name]=clip_singular(getattr(model,name),cap)
    n=norms(model);current=certificate(model,n,beta)
    threshold=math.sqrt(model.epsilon)*(1-1e-4)
    scale=min(1.,(BUDGETS['attention']*threshold/max(current['attention'],1e-30))**.25)
    if scale<1:
        for name in ('value','output'):
            clipped[name]=clip_singular(getattr(model,name),n[name]*scale)
    for name,maximum in (
            ('memory',math.sqrt(BUDGETS['memory']*threshold)),
            ('decoder',math.sqrt(BUDGETS['decoder']*threshold*(1-beta)))):
        if n[name]>maximum:
            clipped[name]=clip_singular(getattr(model,name),maximum)
    after=certificate(model,norms(model),beta)
    if not math.isfinite(after['total']) or after['normalized_total']>.7:
        raise EquilibriumError('Singular clipping failed its unchanged norm certificate')
    if any(after[k]>v*math.sqrt(model.epsilon) for k,v in BUDGETS.items()):
        raise EquilibriumError('Singular clipping exceeded a block budget')
    return dict(before=before,after=after,budgets=dict(BUDGETS),operand_norm_cap=cap,
                projection_method='singular_value_clipping',clipped_counts=clipped)
