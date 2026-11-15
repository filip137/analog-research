"""Passive positive-resistor / perfect-diode MLP after conditional attention."""
import math
import torch
from torch import nn
from networks.energy_transformer.pseudo_learning import attention,compare_vectors,residual as attention_residual,settle as attention_settle


def drn_settle(u,g1,g2,steps,state=None,injection=None):
    """Energy: sum .5*g*(voltage drop)^2 + unit shunts; h constrained >=0."""
    o=u.new_zeros((u.shape[0],g2.shape[1])) if state is None else state[1]
    dh=1+g1.sum(0)+g2.sum(1);do=1+g2.sum(0)
    drive=u@g1
    for _ in range(steps):
        h=torch.relu((drive+o@g2.T)/dh)
        o=(h@g2-(0 if injection is None else injection))/do
    return h,o


def drn_residual(u,g1,g2,state):
    h,o=state
    rh=h-torch.relu((u@g1+o@g2.T)/(1+g1.sum(0)+g2.sum(1)))
    ro=o-(h@g2)/(1+g2.sum(0))
    return torch.maximum(rh.abs().max(),ro.abs().max())


class _DRNEP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,g1,g2,steps):
        h,o=drn_settle(u,g1,g2,steps)
        ctx.save_for_backward(u,g1,g2,h,o);ctx.steps=steps
        return o

    @staticmethod
    def backward(ctx,e):
        u,g1,g2,h,o=ctx.saved_tensors
        beta=1e-4/e.abs().max().clamp_min(1e-20)
        hp,op=drn_settle(u,g1,g2,ctx.steps,(h,o),beta*e)
        hm,om=drn_settle(u,g1,g2,ctx.steps,(h,o),-beta*e)
        lh=-(hp-hm)/(2*beta);lo=-(op-om)/(2*beta)
        ah=(hp+hm)/2;ao=(op+om)/2
        gu=lh@g1.T
        gg1=u.T@lh-(ah*lh).sum(0)[None,:]
        gg2=ah.T@lo+lh.T@ao-(ah*lh).sum(0)[:,None]-(ao*lo).sum(0)[None,:]
        return gu,gg1,gg2,None


class PassiveDRN(nn.Module):
    def __init__(self,width=64,hidden=128):
        super().__init__();self.width=width
        self.raw1=nn.Parameter(torch.randn(2*width,hidden)*.5)
        self.raw2=nn.Parameter(torch.randn(hidden,2*width)*.5)
    def conductances(self):
        return (torch.nn.functional.softplus(self.raw1)/(2*self.width),
                torch.nn.functional.softplus(self.raw2)/self.raw2.shape[0])
    def forward(self,x,mode,steps):
        shape=x.shape;u=torch.cat((x,-x),-1).reshape(-1,2*self.width)
        g1,g2=self.conductances()
        o=_DRNEP.apply(u,g1,g2,steps) if mode=='ep' else drn_settle(u,g1,g2,steps)[1]
        return (32*(o[:,:self.width]-o[:,self.width:])).reshape(shape)


class Model(nn.Module):
    def __init__(self,seed=20261115,score='dot',mlp='drn'):
        super().__init__();torch.manual_seed(seed);self.score=score;self.mlp=mlp
        self.encoder=nn.Linear(48,64,bias=False);self.position=nn.Parameter(torch.randn(64,64)*.2)
        for name in ['query','key','value','output']: setattr(self,name,nn.Linear(64,64,bias=False))
        self.drn=PassiveDRN()
        self.ff1=nn.Linear(64,128,bias=False);self.ff2=nn.Linear(128,64,bias=False)
        self.decoder=nn.Linear(64,10,bias=False)
        for name,p in self.named_parameters():
            if name.startswith('drn.') or name=='position':continue
            with torch.no_grad():p.normal_(0,.15/math.sqrt(p.shape[-1]))
        if mlp=='drn':del self.ff1;del self.ff2
        else:del self.drn

    def ports(self,x):
        z=self.encoder(x)+self.position
        n=torch.nn.functional.layer_norm(z,(64,),eps=.5)
        split=lambda t:t.reshape(t.shape[0],64,4,16).transpose(1,2)
        q,k=split(self.query(n)),split(self.key(n))
        s=q@k.transpose(-1,-2)/4 if self.score=='dot' else -torch.cdist(q,k,p=1)/4
        a=torch.softmax(s,-1);v=.75*torch.tanh(split(self.value(n)))
        return z,a,v

    def forward(self,x,mode='ep',steps=32,ablation=None):
        z,a,v=self.ports(x)
        if ablation=='uniform':a=torch.ones_like(a)/64
        y=attention(a,v,steps,mode).transpose(1,2).flatten(-2)
        z=z+self.output(y);n=torch.nn.functional.layer_norm(z,(64,),eps=.5)
        z=z+(self.drn(n,mode,steps) if self.mlp=='drn' else self.ff2(torch.relu(self.ff1(n))))
        return self.decoder(z.mean(1))


def probe(model,x,t):
    def grads(mode,cap):
        y=model(x,mode,cap)
        gs=torch.autograd.grad(torch.nn.functional.cross_entropy(y,t),tuple(model.parameters()))
        return y.detach(),list(gs)+[torch.cat([g.flatten() for g in gs])]
    yr,gr=grads('bptt',128);yl,gl=grads('bptt',256)
    names=[n for n,_ in model.named_parameters()]+['global']
    reference={n:compare_vectors(a,b) for n,a,b in zip(names,gr,gl)}
    rows=[]
    for cap in [8,16,32,64]:
        ye,ge=grads('ep',cap);yb,gb=grads('bptt',cap)
        rows.append({'cap':cap,'forward_error':float((ye-yb).abs().max()),'reference_forward_error':float((ye-yr).abs().max()),
            'groups':{n:{'ep_bptt':compare_vectors(e,b),'ep_reference':compare_vectors(e,r),'bptt_reference':compare_vectors(b,r)} for n,e,b,r in zip(names,ge,gb,gr)}})
    with torch.no_grad():
        z,a,v=model.ports(x);state=attention_settle(a,v,32)
        diag={'attention_residual':float(attention_residual(a,v,state)),'attention_max':float(a.max())}
        if model.mlp=='drn':
            z=z+model.output((state[1]-1).transpose(1,2).flatten(-2));n=torch.nn.functional.layer_norm(z,(64,),eps=.5)
            u=torch.cat((n,-n),-1).reshape(-1,128);g1,g2=model.drn.conductances();h,o=drn_settle(u,g1,g2,32)
            diag.update(drn_residual=float(drn_residual(u,g1,g2,(h,o))),hidden_active_fraction=float((h>0).float().mean()),
                        conductance_min=float(torch.minimum(g1.min(),g2.min())),conductance_max=float(torch.maximum(g1.max(),g2.max())))
    return {'reference_128_256':reference,'caps':rows,'diagnostics':diag}
