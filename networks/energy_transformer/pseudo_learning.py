"""Ideal anchored-star attention; independent free value states per query.

Encoded KCL block Gauss-Seidel is a numerical solver, not physical RC time.
The EP backward holds an externally supplied error current fixed in both phases.
"""
import math
import torch
from torch import nn

H0 = .01


def settle(a, v, steps, state=None, injection=None):
    """a: B,H,Q,K; v: B,H,K,F. Exactly `steps` full key/output sweeps."""
    if state is None:
        o = torch.ones((*a.shape[:-1], v.shape[-1]), device=v.device, dtype=v.dtype)
    else:
        _, o = state
    j = 1 + v.unsqueeze(-3)
    g = a.unsqueeze(-1)
    drive = H0 if injection is None else H0-injection
    for _ in range(steps):
        u = (j+g*o.unsqueeze(-2))/(1+g)
        o = (drive+(g*u).sum(-2))/(H0+a.sum(-1)).unsqueeze(-1)
    return u, o


def exact(a, v):
    w = a/(1+a)
    return (w@v)/(H0+w.sum(-1)).unsqueeze(-1)


def residual(a, v, state):
    u,o=state; g=a.unsqueeze(-1)
    ru=(1+g)*u-g*o.unsqueeze(-2)-(1+v.unsqueeze(-3))
    ro=(H0+a.sum(-1)).unsqueeze(-1)*o-(g*u).sum(-2)-H0
    return torch.maximum(ru.abs().max(),ro.abs().max())


class _EP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, v, steps, beta):
        u,o=settle(a,v,steps)
        ctx.save_for_backward(a,v,u,o)
        ctx.steps,ctx.beta=steps,beta
        return o-1

    @staticmethod
    def backward(ctx, e):
        a,v,u,o=ctx.saved_tensors
        # Scale the current to avoid subtracting sub-ulp state perturbations.
        # The frozen linear error allows this scaling; it is undone in readout.
        scale=ctx.beta/e.abs().max().clamp_min(1e-20)
        up,op=settle(a,v,ctx.steps,(u,o),scale*e)
        um,om=settle(a,v,ctx.steps,(u,o),-scale*e)
        dp=up-op.unsqueeze(-2); dm=um-om.unsqueeze(-2)
        ga=((dp+dm)*(dp-dm)).sum(-1)/(4*scale)
        gv=(-(up-um)/(2*scale)).sum(-3)
        return ga,gv,None,None


def attention(a,v,steps=32,mode='ep',beta=.005):
    if mode=='ep': return _EP.apply(a,v,steps,beta)
    if mode=='bptt': return settle(a,v,steps)[1]-1
    if mode=='exact': return exact(a,v)
    raise ValueError(mode)


class Model(nn.Module):
    def __init__(self, seed=20261114, width=64, heads=4, patch=7, task='completion'):
        super().__init__()
        self.heads=heads; self.width=width; self.patch=patch; self.task=task
        self.encoder=nn.Linear(2*patch*patch,width,bias=False)
        self.position=nn.Parameter(torch.empty((28//patch)**2,width))
        for name in ('query','key','value','output','ff1','ff2'):
            setattr(self,name,nn.Linear(width,width,bias=False))
        self.decoder=nn.Linear(width,patch*patch if task=='completion' else 10,bias=False)
        for i,(name,p) in enumerate(self.named_parameters()):
            g=torch.Generator().manual_seed(seed+i)
            with torch.no_grad():
                p.normal_(0,.2 if name=='position' else .15/math.sqrt(p.shape[-1]),generator=g)

    def ports(self,x):
        z=self.encoder(x)+self.position
        n=torch.nn.functional.layer_norm(z,(self.width,),eps=.5)
        def split(t): return t.reshape(t.shape[0],t.shape[1],self.heads,-1).transpose(1,2)
        q,k=split(self.query(n)),split(self.key(n))
        a=torch.softmax(q@k.transpose(-1,-2)/math.sqrt(q.shape[-1]),dim=-1)
        v=.75*torch.tanh(split(self.value(n)))
        return z,a,v

    def forward(self,x,mode='ep',steps=32,beta=.005,ablation=None):
        z,a,v=self.ports(x)
        if ablation=='uniform': a=torch.ones_like(a)/a.shape[-1]
        if ablation=='zero': a=torch.zeros_like(a)
        y=attention(a,v,steps,mode,beta).transpose(1,2).flatten(-2)
        z=z+self.output(y)
        n=torch.nn.functional.layer_norm(z,(self.width,),eps=.5)
        z=z+self.ff2(torch.relu(self.ff1(n)))
        return self.decoder(z if self.task=='completion' else z.mean(1))


def compare_vectors(first, second):
    a=first.detach().double().flatten(); b=second.detach().double().flatten()
    return {'cosine':float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
            'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-20)),
            'norm':float(a.norm()),'reference_norm':float(b.norm())}


def gradient_probe(model,x,target,mask,caps=(4,8,16,32,64,128)):
    """Same inputs, weights and scalar loss; full parameter groups and exact ref."""
    def grads(mode,cap):
        y=model(x,mode,cap)
        loss=objective(y,target,mask)
        return y.detach(),torch.autograd.grad(loss,tuple(model.parameters()))
    ye,ge=grads('exact',1)
    rows=[]
    for cap in caps:
        yp,gp=grads('ep',cap);yb,gb=grads('bptt',cap)
        row={'cap':cap,'forward_ep_bptt_max':float((yp-yb).abs().max()),
             'forward_vs_exact_max':float((yp-ye).abs().max()),'groups':{}}
        names=[n for n,_ in model.named_parameters()]+['global']
        ep=list(gp)+[torch.cat([g.flatten() for g in gp])]
        bp=list(gb)+[torch.cat([g.flatten() for g in gb])]
        ex=list(ge)+[torch.cat([g.flatten() for g in ge])]
        for n,p,b,e in zip(names,ep,bp,ex):
            row['groups'][n]={'ep_bptt':compare_vectors(p,b),'ep_exact':compare_vectors(p,e),
                              'bptt_exact':compare_vectors(b,e)}
        with torch.no_grad():
            _,a,v=model.ports(x)
            row['kcl_residual']=float(residual(a,v,settle(a,v,cap)))
        rows.append(row)
    return rows


def objective(y,target,mask):
    if mask is None: return torch.nn.functional.cross_entropy(y,target)
    return ((y-target).square()*mask).sum()/(2*mask.sum())
