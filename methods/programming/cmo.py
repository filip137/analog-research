"""Held CMO endpoints and verify-only opposite-input-rail partner rewriting."""

import torch

from hardware.faults import MatrixSpec, FaultCase, hierarchical_faults, binding_seed, tensor_hash
from hardware.devices.cmo import G_MIN_US, G_MAX_US, MID_US, HALF_RANGE_US, programming_sigma
from networks.resistive.differentiable import coefficients
from networks.opt.mapping import programming_layout

NAMESPACE = 'opt125m-cmo-drn-last6'


def partner(value):
    """Pair opposite source rails feeding the same destination node."""
    a,b=value.chunk(2,dim=1)
    return torch.cat((b,a),dim=1)


def rewrite_plan(observed,target,threshold_sigma=5.0,noise_scale=1.0):
    """Only targets and apparent reads enter the controller; no hidden faults."""
    flags=(observed-target).abs()>threshold_sigma*noise_scale*programming_sigma(target,2.0)
    selected=~flags & partner(flags)
    wanted=partner(observed)+target-partner(target)
    bounded=wanted.clamp(G_MIN_US,G_MAX_US)
    return selected,torch.where(selected,bounded,observed),flags,int(((wanted!=bounded)&selected).sum())


@torch.no_grad()
def program_rewritten(model,assignment,kind='none',rate=0.0,noise_scale=1.0,*,capture_state=False):
    layout, masters, domains = programming_layout(model)
    faults=None if kind=='none' else hierarchical_faults(layout,256,48,FaultCase(kind,rate),assignment,NAMESPACE,1,domains=domains)
    view=[]
    receipts=[]
    states=[]
    for spec,u in zip(layout,masters):
        target=(MID_US+HALF_RANGE_US*u.T).contiguous()
        rng=torch.Generator(device=u.device).manual_seed(binding_seed(assignment+10000,NAMESPACE,spec.name))
        drawn=target + noise_scale*programming_sigma(target,2.0)*torch.randn(target.shape,device=u.device,generator=rng)
        if faults is None:
            mask=torch.zeros_like(target,dtype=torch.bool)
            stuck=torch.zeros_like(target)
        else:
            mask,stuck=(x.reshape(target.shape).to(u.device) for x in faults['matrices'][spec.name])
        observed=torch.where(mask,stuck,drawn)
        selected,wanted,flags,clipped=rewrite_plan(observed,target,noise_scale=noise_scale)
        rng.manual_seed(binding_seed(241016,NAMESPACE,f'rewrite/{assignment}/{spec.name}'))
        drawn=wanted+noise_scale*programming_sigma(wanted,2.0)*torch.randn(wanted.shape,device=u.device,generator=rng)
        held=torch.where(mask,stuck,torch.where(selected,drawn,observed))
        if not bool(torch.isfinite(held).all() and (held>=0).all()):
            raise FloatingPointError(f'Invalid physical DRN endpoint in {spec.name}; no clipping or resampling.')
        normalized=held.T.contiguous()/G_MAX_US
        view.append(normalized)
        if capture_state:
            write_rng=torch.Generator(device=u.device).manual_seed(binding_seed(assignment+20000,NAMESPACE,spec.name))
            states.append({'name':spec.name,'noise_scale':noise_scale,'rng':write_rng.get_state().cpu(),
                           **{key:value.T.contiguous().cpu() for key,value in
                              (('target',target),('raw',observed),('held',held),('mask',mask),('stuck',stuck),
                               ('verify_flags',flags),('rewrite_selected',selected))},
                           'write_attempts':0,'device_writes':0,'stuck_write_attempts':0})
        receipts.append({'matrix':spec.name,'devices':spec.size,'faults':int(mask.sum()),'flagged':int(flags.sum()),
                         'rewrite_attempts':int(selected.sum()),'stuck_rewrite_attempts':int((selected&mask).sum()),
                         'clipped_targets':clipped,'held_apparent_sha256':tensor_hash(normalized),
                         'node_loading_change_l1_us':float((held.sum(1)-observed.sum(1)).abs().sum())})
    pairs=tuple(tuple(view[i:i+2]) for i in range(0,len(view),2))
    for k,(g1,g2) in enumerate(pairs):
        dh,do=coefficients(g1,g2,model.settings)
        if not bool((dh>0).all() and (do>0).all()):
            raise FloatingPointError(f'Nonpositive physical DRN denominator in decoder layer {model.start+k}.')
    receipt={
        'assignment':assignment,'kind':kind,'rate':rate,'primary_state':'held_apparent',
        'persistent_equals_apparent':True,'matrices':receipts,
        'fault_identity':None if faults is None else faults['receipt']['identity_sha256']}
    if capture_state:
        return pairs,receipt,{'schema':'opt_cmo_drn.bank.v1','receipt':receipt,'cells':states}
    return pairs,receipt
