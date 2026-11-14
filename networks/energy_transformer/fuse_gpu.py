"""Batched reduction of fuse.py for matched differential rails.

Each head is scalar: every fuse really has two terminals. The negative rail
is exactly minus the positive rail (no device mismatch in this pilot).
Energy here is half the full two-rail co-content; the physical output nudge
therefore has beta/4 times the squared readout error. EP statistics below
sum both physical resistors sharing each logical conductance.
"""
from dataclasses import dataclass
import math
import time
import torch
from torch import nn

NAMES = ('encoder', 'position', 'query', 'key', 'decoder')


class Unsettled(RuntimeError):
    pass


@dataclass
class Phase:
    z: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    o: torch.Tensor
    iterations: int = 0
    residual: float = float('inf')

    def tensors(self):
        return self.z, self.q, self.k, self.o


class FuseNetwork(nn.Module):
    def __init__(self, tokens=16, width=32, patch_width=49, heads=4,
                 seed=20261114, kernel='gaussian', floor=.02,
                 sigma=.25, fuse_g=1., gain=4., device='cpu', dtype=torch.float32):
        super().__init__()
        self.tokens, self.width, self.patch_width, self.heads = tokens, width, patch_width, heads
        self.kernel, self.floor, self.sigma, self.fuse_g, self.gain = kernel, floor, sigma, fuse_g, gain
        if kernel not in ('gaussian', 'linear'):
            raise ValueError('Expected gaussian or linear kernel')
        shapes = ((width, 2*patch_width), (tokens, width), (heads, width),
                  (heads, width), (patch_width, width))
        # CPU name-local initialization agrees with the explicit element model
        # when its flat theta is constructed from these conductances.
        for i, (name, shape) in enumerate(zip(NAMES, shapes)):
            gen = torch.Generator().manual_seed(seed+i)
            w = torch.randn(shape, generator=gen, dtype=dtype) * (.2 if name=='position' else .15/math.sqrt(shape[-1]))
            for suffix, value in (('p', w.clamp_min(0)), ('n', (-w).clamp_min(0))):
                self.register_parameter(name+'_'+suffix, nn.Parameter((floor+value).to(device), requires_grad=False))
        self.register_buffer('offdiag', 1-torch.eye(tokens, device=device, dtype=dtype))

    def maps(self):
        return {n: (getattr(self,n+'_p')-getattr(self,n+'_n'),
                    getattr(self,n+'_p')+getattr(self,n+'_n')) for n in NAMES}

    def zero(self, x):
        b,n,_=x.shape
        return Phase(x.new_zeros(b,n,self.width), x.new_zeros(b,n,self.heads),
                     x.new_zeros(b,n,self.heads), x.new_zeros(b,n,self.patch_width))

    def conductance(self, q, k, intervention='normal'):
        d=q.transpose(1,2).unsqueeze(-1)-k.transpose(1,2).unsqueeze(-2)
        if intervention=='zero':
            return torch.zeros_like(d)
        if intervention=='linear' or self.kernel=='linear':
            return self.fuse_g*torch.ones_like(d)*self.offdiag
        if intervention!='normal':
            raise ValueError(intervention)
        return self.fuse_g*torch.exp(-d.square()/(2*self.sigma**2))*self.offdiag

    def force(self, v, x, target, mask, beta=0., intervention='normal', maps=None):
        p=self.maps() if maps is None else maps
        we,se=p['encoder']; wp,sp=p['position']; wq,sq=p['query']; wk,sk=p['key']; wo,so=p['decoder']
        z,q,k,o=v.tensors()
        dz=x@we.T+wp+q@wq+k@wk+o@wo-z*(se.sum(1)+sp+sq.sum(0)+sk.sum(0)+so.sum(0))
        w=self.conductance(q,k,intervention)
        qt,kt=q.transpose(1,2),k.transpose(1,2)
        fq=(w@kt.unsqueeze(-1)).squeeze(-1)-w.sum(-1)*qt
        fk=(w.transpose(-1,-2)@qt.unsqueeze(-1)).squeeze(-1)-w.sum(-2)*kt
        dq=z@wq.T-q*sq.sum(1)+fq.transpose(1,2)
        dk=z@wk.T-k*sk.sum(1)+fk.transpose(1,2)
        do=z@wo.T-o*so.sum(1)+beta*mask*self.gain*(target-self.gain*o)/2
        return dz,dq,dk,do

    def sweep(self, v, x, target, mask, beta=0., intervention='normal', maps=None):
        """Block Gauss-Seidel / lagged diffusivity; numerical equilibrium solver.

        Gaussian psi is concave in squared drop, so freezing each fuse weight
        majorizes its energy. Each block minimizes that quadratic majorizer.
        This is not a physical-time simulation or a hardware latency claim.
        """
        p=self.maps() if maps is None else maps
        we,se=p['encoder']; wp,sp=p['position']; wq,sq=p['query']; wk,sk=p['key']; wo,so=p['decoder']
        z,q,k,o=v.tensors()
        z=(x@we.T+wp+q@wq+k@wk+o@wo)/(se.sum(1)+sp+sq.sum(0)+sk.sum(0)+so.sum(0))
        w=self.conductance(q,k,intervention)
        q=((z@wq.T).transpose(1,2)+(w@k.transpose(1,2).unsqueeze(-1)).squeeze(-1))/(sq.sum(1)[None,:,None]+w.sum(-1))
        q=q.transpose(1,2)
        w=self.conductance(q,k,intervention)
        k=((z@wk.T).transpose(1,2)+(w.transpose(-1,-2)@q.transpose(1,2).unsqueeze(-1)).squeeze(-1))/(sk.sum(1)[None,:,None]+w.sum(-2))
        k=k.transpose(1,2)
        den=so.sum(1)+beta*mask*self.gain**2/2
        if beta<0 and bool((den<=0).any()):
            raise Unsettled('Negative nudge exceeds output anchoring')
        o=(z@wo.T+beta*mask*self.gain*target/2)/den
        return Phase(z,q,k,o)

    def energy(self, v, x, target, mask, beta=0., intervention='normal'):
        p=self.maps();z,q,k,o=v.tensors()
        def edge(y,u,name):
            w,s=p[name]
            return (.5*y.square()*s.sum(1)+.5*(u.square()@s.T)-y*(u@w.T)).sum((1,2))
        wp,sp=p['position']
        e=edge(z,x,'encoder')+(.5*sp*(z.square()+1)-wp*z).sum((1,2))
        e=e+edge(q,z,'query')+edge(k,z,'key')+edge(o,z,'decoder')
        d=q.transpose(1,2).unsqueeze(-1)-k.transpose(1,2).unsqueeze(-2)
        if intervention!='zero':
            psi=d.square()/2 if self.kernel=='linear' or intervention=='linear' else -self.sigma**2*torch.expm1(-d.square()/(2*self.sigma**2))
            e=e+self.fuse_g*(psi*self.offdiag).sum((1,2,3))
        return e+beta/4*((self.gain*o-target).square()*mask).sum((1,2))

    def gradient(self, plus, minus, x, mask, beta):
        """Difference of local squared voltage drops, avoiding square cancellation."""
        out={}
        def edge(yp,ym,up,um,name):
            dy,sy=yp-ym,yp+ym;du,su=up-um,up+um
            base=(dy*sy).sum((0,1))[:,None]+(du*su).sum((0,1))[None,:]
            cross=torch.einsum('bni,bnj->ij',dy,su)+torch.einsum('bni,bnj->ij',sy,du)
            out[name+'_p']=(base-cross)/(2*beta*mask.sum())
            out[name+'_n']=(base+cross)/(2*beta*mask.sum())
        edge(plus.z,minus.z,x,x,'encoder')
        edge(plus.q,minus.q,plus.z,minus.z,'query')
        edge(plus.k,minus.k,plus.z,minus.z,'key')
        edge(plus.o,minus.o,plus.z,minus.z,'decoder')
        dy,sy=plus.z-minus.z,plus.z+minus.z
        out['position_p']=(dy*(sy-2)).sum(0)/(2*beta*mask.sum())
        out['position_n']=(dy*(sy+2)).sum(0)/(2*beta*mask.sum())
        return out

    @torch.no_grad()
    def project(self):
        for p in self.parameters():
            p.clamp_(min=self.floor)

    def counts(self):
        logical=sum(p.numel() for p in self.parameters())
        copies=self.tokens*(self.width*2*self.patch_width+2*self.heads*self.width+self.patch_width*self.width)+self.tokens*self.width
        return {'trainable_conductances':logical, 'physical_resistors':4*copies,
                'physical_fuses':2*self.heads*self.tokens*(self.tokens-1),
                'physical_state_nodes':2*self.tokens*(self.width+2*self.heads+self.patch_width)}


@torch.no_grad()
def settle(model,x,target,mask,beta=0.,initial=None,tolerance=1e-6,max_steps=512,
           deadline=None,intervention='normal',trace=None):
    v=model.zero(x) if initial is None else Phase(*(t.clone() for t in initial.tensors()))
    maps=model.maps()
    for step in range(max_steps+1):
        if step%8==0 or step==max_steps:
            if deadline is not None and time.monotonic()>=deadline:
                raise TimeoutError('Fuse phase deadline reached')
            force=model.force(v,x,target,mask,beta,intervention,maps)
            residual=max(t.abs().max().item() for t in force)
            if not math.isfinite(residual):
                raise Unsettled('Nonfinite fuse state')
            if residual<=tolerance:
                v.iterations,v.residual=step,residual
                return v
        if step==max_steps:
            break
        if trace is not None:
            trace.append(model.energy(v,x,target,mask,beta,intervention))
        v=model.sweep(v,x,target,mask,beta,intervention,maps)
    raise Unsettled(f'Fuse phase beta={beta:g} failed at {max_steps} sweeps: residual={residual:g}')


@torch.no_grad()
def ep_gradient(model,x,target,mask,beta=.005,**solver):
    free=settle(model,x,target,mask,**solver)
    plus=settle(model,x,target,mask,beta,free,**solver)
    minus=settle(model,x,target,mask,-beta,free,**solver)
    return model.gradient(plus,minus,x,mask,beta),(free,plus,minus)
