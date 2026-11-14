"""Batched ideal-circuit equations. Equilibration and EP never build a graph.

Memory is the algebraic minimum of .5*r*r-r*(U*g)+I[r>=0]. The
LayerNorm, buffers, signed projections and bias supplies are ideal macros.
"""
from dataclasses import dataclass
import math
import time
import torch
from torch import nn

NAMES = ('query', 'key', 'memory', 'encoder', 'decoder', 'position')


class EquilibriumError(RuntimeError):
    pass


def initialize(module, seed):
    # Name-local generators keep shared maps identical across architectures.
    order = NAMES + ('value', 'output', 'ff1', 'ff2')
    with torch.no_grad():
        for name, p in module.named_parameters():
            gen = torch.Generator(device=p.device).manual_seed(seed + order.index(name))
            p.normal_(0., .2 if name == 'position' else .15 / math.sqrt(p.shape[-1]), generator=gen)


def normalize(z, epsilon):
    u = z - z.mean(-1, keepdim=True)
    s = (u.square().mean(-1, keepdim=True) + epsilon).sqrt()
    return u / s, s


def diode(drive):
    """Ideal diode complementarity solution, with unit linear conductance."""
    return drive.clamp_min(0)


def diode_residual(drive, r):
    return torch.stack(((-r).clamp_min(0).amax(),
                        (drive-r).clamp_min(0).amax(), (r*(r-drive)).abs().amax())).amax()


def triode_product(left, right):
    """Independent four-device current calculation; primary solver uses x*y."""
    result = torch.zeros_like(left + right)
    for sg in (-1, 1):
        for sd in (-1, 1):
            gate = 1.5 + sg*.05*left
            drain = .3 + sd*.05*right
            result = result + sg*sd*1e-4*((gate-.5)*drain - .5*drain.square())
    return result / 1e-6


def weak_bank(logits):
    """Independent ideal MOS currents and source KCL, using fixed 3 V drains."""
    vt, n, i0, tail = .026, 1.5, 1e-8, 1e-9
    gates = .7 + n*vt*logits
    logsum = torch.logsumexp(logits, -1, keepdim=True)
    log_source_factor = torch.logaddexp(math.log(tail/i0)-logsum,
                                      torch.full_like(logsum, -3/vt))
    source = -vt*log_source_factor
    currents = i0*torch.exp(logits)*(torch.exp(-source/vt)-math.exp(-3/vt))
    margin = .7+n*torch.minimum(source, torch.full_like(source, 3.))-gates
    return currents/tail, source, margin


@torch.no_grad()
def mos_domain(v):
    """Physical conditions of every ideal attention multiplier and softmax bank.

    QK uses gate=1.5 +/- .05*q, drain=.3 +/- .05*k, Vth=.5.
    Returns multiply individual probabilities by k or q; P.T@Q is a SUM
    of product currents feeding a linear resistor projection, not a multiplier
    operand. Include q as a drain operand in the reciprocal return path.
    Each returned value has one entry per example.
    """
    qa, ka = v['q'].abs(), v['k'].abs()
    qm, km = qa.amax((1,2,3)), ka.amax((1,2,3))
    pm = v['p'].abs().amax((1,2,3))
    maximum = torch.maximum(qm, km)
    # All query/key token pairs share a head and component.
    score_sum = (qa.amax(-2)+ka.amax(-2)).amax((1,2))
    return_sum = pm+maximum  # Conservative bound on individual p*q and p*k.
    triode = .7-.05*torch.maximum(score_sum, return_sum)
    drain = .3-.05*maximum
    vt, n = .026, 1.5
    logsum = torch.logsumexp(v['logits'], -1)
    source = -vt*torch.logaddexp(math.log(.1)-logsum,
                                torch.full_like(logsum, -3/vt))
    # With fixed drain=3V, weak inversion requires positive threshold margin.
    weak = (n*(torch.minimum(source, torch.full_like(source,3.))
                -vt*v['logits'].amax(-1))).amin((1,2))
    valid = (torch.isfinite(maximum) & torch.isfinite(weak)
             & (triode > 0) & (drain > 0) & (weak > 0))
    return dict(maximum_operand=maximum, triode_margin=triode,
                drain_margin=drain, weak_inversion_margin=weak, valid=valid)


@torch.no_grad()
def check_domain(model, v, context=''):
    """Keep the historical conservative domain explicit; report real violations."""
    if model.device_domain == 'declared4':
        maximum = torch.maximum(v['q'].abs().amax(),v['k'].abs().amax())
        if not torch.isfinite(maximum) or maximum > 4:
            raise EquilibriumError(f'{context}: operand={maximum.item():.9g} outside declared [-4,4] range')
        return None
    values = mos_domain(v)
    if not values['valid'].all():
        bad = (~values['valid']).nonzero()[0,0].item()
        details = ', '.join(f'{key}={value[bad].item():.9g}'
                            for key,value in values.items() if key != 'valid')
        raise EquilibriumError(f'{context}: MOS device-domain violation at batch example {bad}: {details}')
    return values


class EnergyNetwork(nn.Module):
    def __init__(self, tokens=16, width=32, patch_width=49, heads=4, seed=20261114,
                 epsilon=.5, device='cpu', dtype=torch.float32, device_domain='declared4', solver_dt=.5):
        super().__init__()
        self.tokens, self.width, self.patch_width = tokens, width, patch_width
        self.heads, self.epsilon = heads, epsilon
        if device_domain not in ('declared4', 'physical'):
            raise ValueError('Unknown MOS domain policy')
        if not math.isfinite(solver_dt) or not 0 < solver_dt <= 1.:
            raise ValueError('Solver step must be in (0,1]')
        self.device_domain, self.solver_dt = device_domain, solver_dt
        self.kappa = 1/math.sqrt(width//heads)
        shapes = ((heads,width//heads,width),)*2 + ((4*width,width),
                  (width,2*patch_width),(patch_width,width),(tokens,width))
        for name, shape in zip(NAMES, shapes):
            self.register_parameter(name, nn.Parameter(torch.empty(shape, device=device, dtype=dtype),
                                                       requires_grad=False))
        initialize(self, seed)

    def ports(self, z, y, attention='normal'):
        g, s = normalize(z, self.epsilon)
        q = torch.einsum('bnd,had->bhna', g, self.query)
        k = torch.einsum('bnd,had->bhna', g, self.key)
        logits = self.kappa*(q @ k.transpose(-1,-2))
        p = logits.softmax(-1)
        if attention == 'uniform':
            p = torch.full_like(p, 1/self.tokens)
        elif attention == 'zero':
            p = torch.zeros_like(p)
        elif attention != 'normal':
            raise ValueError('Unknown attention intervention')
        r = diode(g @ self.memory.T)
        return dict(z=z, y=y, g=g, s=s, q=q, k=k, logits=logits, p=p, r=r)

    def force(self, v, x, target, mask, beta=0.):
        qr = v['p'] @ v['k']
        kr = v['p'].transpose(-1,-2) @ v['q']
        drive = x @ self.encoder.T + self.position
        drive = drive + torch.einsum('bhna,had->bnd', qr, self.query)
        drive = drive + torch.einsum('bhna,had->bnd', kr, self.key)
        dz = drive + v['r'] @ self.memory + v['y'] @ self.decoder - v['z']
        dy = v['g'] @ self.decoder.T - v['y'] - beta*mask*(v['y']-target)
        return dz, dy

    def energy(self, v, x, target, mask, beta=0., attention='normal'):
        total = (v['z']*v['g']).sum((1,2)) - self.width*v['s'].sum((1,2))
        total -= ((x @ self.encoder.T + self.position)*v['g']).sum((1,2))
        if attention == 'normal':
            total -= torch.logsumexp(v['logits'], -1).sum((1,2))/self.kappa
        elif attention == 'uniform':
            total -= v['logits'].mean(-1).sum((1,2))/self.kappa
        total -= .5*v['r'].square().sum((1,2))
        total += (.5*v['y'].square() - v['y']*(v['g'] @ self.decoder.T)).sum((1,2))
        return total + .5*beta*(mask*(v['y']-target).square()).sum((1,2))

    def statistics(self, v, x):
        return {
            'query': -torch.einsum('bhna,bnd->had', v['p'] @ v['k'], v['g']),
            'key': -torch.einsum('bhna,bnd->had', v['p'].transpose(-1,-2) @ v['q'], v['g']),
            'memory': -torch.einsum('bnm,bnd->md', v['r'], v['g']),
            'encoder': -torch.einsum('bnd,bnp->dp', v['g'], x),
            'decoder': -torch.einsum('bnp,bnd->pd', v['y'], v['g']),
            'position': -v['g'].sum(0),
        }


@dataclass
class Equilibrium:
    z: torch.Tensor
    y: torch.Tensor
    ports: dict
    iterations: torch.Tensor
    residual: float
    maximum_operand: float
    minimum_triode_margin: float
    energy_increase: float
    minimum_drain_margin: float | None = None
    minimum_weak_inversion_margin: float | None = None


@torch.no_grad()
def settle(model, x, target, mask, beta=0., initial=None, tolerance=1e-6,
           max_steps=4096, deadline=None, attention='normal', trace=None):
    if initial is None:
        z = x.new_zeros((len(x), model.tokens, model.width))
        y = torch.zeros_like(target)
    else:
        z, y = initial.z.clone(), initial.y.clone()
    v = model.ports(z, y, attention)
    energy = model.energy(v, x, target, mask, beta, attention)
    steps = torch.zeros(len(x), dtype=torch.long, device=x.device)
    maximum = x.new_tensor(0.)
    increase = x.new_tensor(0.)
    allowance = 8*torch.finfo(x.dtype).eps
    for step in range(max_steps+1):
        if step % 16 == 0 and deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Equilibrium deadline reached')
        dz, dy = model.force(v, x, target, mask, beta)
        residual = torch.maximum(dz.abs().amax((1,2)), dy.abs().amax((1,2)))
        maximum = torch.maximum(maximum, torch.maximum(v['q'].abs().amax(), v['k'].abs().amax()))
        if not torch.isfinite(residual).all() or not torch.isfinite(energy).all():
            raise EquilibriumError('Nonfinite circuit state')
        if maximum > 4:
            raise EquilibriumError(f'Triode operand outside declared [-4,4] domain: {maximum.item():g}')
        active = residual > tolerance
        if not active.any():
            m = maximum.item()
            return Equilibrium(z,y,v,steps,residual.max().item(),m,.7-.1*m,increase.item())
        if step == max_steps:
            raise EquilibriumError(f'Unsettled phase beta={beta:g}: residual={residual.max().item():g}')
        dt = active.to(x.dtype).reshape(-1,1,1)*.5
        for _ in range(20):
            nz, ny = z+dt*dz, y+dt*dy/1.3
            nv = model.ports(nz, ny, attention)
            ne = model.energy(nv, x, target, mask, beta, attention)
            acceptable = torch.isfinite(ne) & (ne <= energy + allowance*(1+energy.abs()))
            if acceptable.all():
                break
            dt = torch.where(acceptable[:,None,None], dt, dt*.5)
        else:
            raise EquilibriumError('Energy backtracking failed')
        increase = torch.maximum(increase, (ne-energy).max())
        z,y,v,energy = nz,ny,nv,ne
        steps += active
        if trace is not None:
            trace.append(energy.detach().clone())
    raise AssertionError('Unreachable')


@torch.no_grad()
def ep_gradient(model, x, target, mask, beta=.005, **solver):
    free = settle(model,x,target,mask,**solver)
    plus = settle(model,x,target,mask,beta,initial=free,**solver)
    minus = settle(model,x,target,mask,-beta,initial=free,**solver)
    high, low = model.statistics(plus.ports,x), model.statistics(minus.ports,x)
    denominator = 2*beta*mask.sum()
    gradients = {name:(high[name]-low[name])/denominator for name in NAMES}
    return gradients, (free,plus,minus)


class DigitalNetwork(nn.Module):
    """One pre-LN attention block, with exactly matching core/total weight counts."""
    def __init__(self, tokens=16,width=32,patch_width=49,heads=4,seed=20261114,
                 epsilon=.5,device='cpu',dtype=torch.float32):
        super().__init__()
        self.tokens,self.width,self.patch_width = tokens,width,patch_width
        self.epsilon = epsilon
        self.kappa = 1/math.sqrt(width//heads)
        shapes = {'query':(heads,width//heads,width),'key':(heads,width//heads,width),
                  'value':(heads,width//heads,width),'output':(width,width),
                  'ff1':(width,width),'ff2':(width,width),'encoder':(width,2*patch_width),
                  'decoder':(patch_width,width),'position':(tokens,width)}
        for name,shape in shapes.items():
            self.register_parameter(name,nn.Parameter(torch.empty(shape,device=device,dtype=dtype)))
        initialize(self,seed)

    def forward(self,x,attention='normal'):
        z = x @ self.encoder.T + self.position
        g,_ = normalize(z,self.epsilon)
        q,k,v = [torch.einsum('bnd,had->bhna',g,p) for p in (self.query,self.key,self.value)]
        p = (self.kappa*(q @ k.transpose(-1,-2))).softmax(-1)
        if attention == 'uniform':
            p = torch.full_like(p,1/self.tokens)
        elif attention == 'zero':
            p = torch.zeros_like(p)
        elif attention != 'normal':
            raise ValueError('Unknown attention intervention')
        value = (p @ v).transpose(1,2).reshape(len(x),self.tokens,self.width)
        z = z + value @ self.output.T
        g,_ = normalize(z,self.epsilon)
        z = z + torch.relu(g @ self.ff1.T) @ self.ff2.T
        return z @ self.decoder.T
