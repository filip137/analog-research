"""Projected EP updates with a sufficient global contraction constraint.

For normalized tokens ||g_i|| <= sqrt(width), each attention probability is
bounded by C_h/tokens. The bilinear and covariance Hessian terms are bounded
by ``bound``. Separate budgets prevent a saturated memory block from
shrinking attention or decoder weights on each programming update. Memory
adds ||U||_2**2 and the worst negative output nudge adds
||decoder||_2**2/(1-beta). Keeping their sum below sqrt(epsilon) gives a
strongly convex energy in centered normalized coordinates. The fixed-point
iteration contracts in a weighted product norm; MOS validity is still checked
separately. Projection is part of digital programming, never the EP gradient.
"""
import math
import torch
from networks.energy_transformer.gpu import EnergyNetwork, EquilibriumError

@torch.no_grad()
def recurrent_norms(model):
    return {
        'query': torch.linalg.matrix_norm(model.query, ord=2).tolist(),
        'key': torch.linalg.matrix_norm(model.key, ord=2).tolist(),
        'memory': torch.linalg.matrix_norm(model.memory, ord=2).item(),
        'decoder': torch.linalg.matrix_norm(model.decoder, ord=2).item(),
    }

def bound(model, norms, scale=1., beta=.005):
    attention=0.
    for a,b in zip(norms['query'],norms['key']):
        ab=scale*scale*a*b
        c=math.exp(min(math.log(model.tokens),2*model.kappa*model.width*ab))
        attention += 2*ab*math.sqrt(c) + 2*model.kappa*model.width*ab*ab*(1+c)
    u2=(scale*norms['memory'])**2
    d=scale*norms['decoder']
    total=attention+u2+d*d/(1-beta)
    m00=(attention+u2)/math.sqrt(model.epsilon)
    m11=1-(1-beta)/1.3
    rho=.5*(m00+m11+math.sqrt((m00-m11)**2+4*d*d/(1.3*math.sqrt(model.epsilon))))
    return dict(attention=attention,memory=u2,decoder=d*d/(1-beta),total=total,
                normalized_total=total/math.sqrt(model.epsilon),contraction_radius=rho)

BUDGETS={'attention':.2,'memory':.2,'decoder':.3}

@torch.no_grad()
def project(model, limit=.7, beta=.005):
    if limit != .7 or not 0 < beta < 1:
        raise ValueError('This block constraint fixes total budget0.7')
    norms=recurrent_norms(model);before=bound(model,norms,beta=beta)
    threshold=math.sqrt(model.epsilon)*(1-1e-4)
    low,high=0.,1.
    if before['attention']>BUDGETS['attention']*threshold:
        for _ in range(48):
            mid=(low+high)/2
            if bound(model,norms,mid,beta)['attention']<=BUDGETS['attention']*threshold:low=mid
            else:high=mid
        attention=low
    else:attention=1.
    memory=min(1.,math.sqrt(BUDGETS['memory']*threshold)/max(norms['memory'],1e-30))
    decoder=min(1.,math.sqrt(BUDGETS['decoder']*threshold*(1-beta))/max(norms['decoder'],1e-30))
    scales=dict(query=attention,key=attention,memory=memory,decoder=decoder)
    for name,scale in scales.items():getattr(model,name).mul_(scale)
    after=bound(model,recurrent_norms(model),beta=beta)
    if not math.isfinite(after['total']) or after['normalized_total']>limit:
        raise EquilibriumError('Recurrent block projection failed its norm certificate')
    for name,portion in BUDGETS.items():
        if after[name]>portion*math.sqrt(model.epsilon):raise EquilibriumError('Block budget exceeded')
    return dict(scale=min(scales.values()),scales=scales,budgets=BUDGETS,before=before,after=after)


@torch.no_grad()
def initialize_shared(model, policy='standard', beta=.005):
    """Match shared initial maps; digital BP has no training-time constraint."""
    if policy == 'standard':
        return None
    if policy != 'contractive_blocks_v1':
        raise ValueError('Unknown shared initialization policy')
    if isinstance(model, EnergyNetwork):
        return project(model, beta=beta)
    parameter = next(model.parameters())
    # Name-local seeds make encoder/position identical already. The temporary
    # energy model supplies the same projected query/key/decoder maps.
    seed_model = EnergyNetwork(model.tokens, model.width, model.patch_width,
                               heads=model.heads, seed=model.initial_seed,
                               epsilon=model.epsilon, device=parameter.device,
                               dtype=parameter.dtype)
    report = project(seed_model, beta=beta)
    for name in ('query', 'key', 'decoder'):
        getattr(model, name).copy_(getattr(seed_model, name))
    return report
