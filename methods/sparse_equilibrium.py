"""Bounded ideal updates, fixed physical masks, and task evaluation."""
import time
import torch
from torch.nn import functional as F

class BudgetEnd(RuntimeError):pass

def check_time(deadline):
    if time.monotonic()>deadline:raise BudgetEnd('Declared wall budget exhausted.')

def rows(pairs,original_only=False):
    return [p[k] for p in pairs for k in (('original',) if original_only else ('original','revised'))]

def example_loss(model,row,labels):
    device=model.z[0].device
    x=torch.tensor(row['tokens'],device=device)[None]
    return F.cross_entropy(model(x,labels),torch.tensor([row['label']],device=device))

@torch.no_grad()
def evaluate(model,pairs,labels,deadline):
    correct=[0,0]; both=0; loss=0
    for pair in pairs:
        check_time(deadline); flags=[]
        for j,kind in enumerate(('original','revised')):
            row=pair[kind]; tok=torch.tensor(row['tokens'],device=model.z[0].device)[None]
            logits=model(tok,labels); ok=int(logits.argmax(-1).item()==row['label'])
            flags.append(ok); correct[j]+=ok
            loss+=float(F.cross_entropy(logits,torch.tensor([row['label']],device=tok.device)))
        both+=all(flags)
    n=len(pairs)
    return {'pairs':n,'original_accuracy':correct[0]/n,'revised_accuracy':correct[1]/n,
            'both_correct':both/n,'loss':loss/(2*n)}

def choose_masks(model,train,labels,k,kind,seed,examples,deadline):
    quota=k//len(model.z); gen=torch.Generator().manual_seed(seed)
    if kind=='random':
        return [torch.randperm(z.numel(),generator=gen)[:quota].to(z.device) for z in model.z]
    model.configure('full'); scores=[torch.zeros_like(z) for z in model.z]
    order=torch.randperm(len(train),generator=gen)[:examples].tolist()
    for n in order:
        check_time(deadline); model.zero_grad(set_to_none=True)
        example_loss(model,train[n],labels).backward()
        for s,z in zip(scores,model.z):
            if z.grad is None or not torch.isfinite(z.grad).all():raise FloatingPointError('Invalid physical gradient')
            s.add_(z.grad.square()/len(order))
    model.zero_grad(set_to_none=True); model.configure('frozen')
    return [torch.topk(s.flatten(),quota,sorted=True).indices for s in scores]

def fit(model,train,labels,params,updates,batch,lr,seed,deadline,metric,analog_count=0):
    opt=torch.optim.Adam(params,lr=lr)
    gen=torch.Generator().manual_seed(seed)
    elapsed=[]
    for step in range(updates):
        check_time(deadline); t=time.monotonic(); opt.zero_grad(set_to_none=True); loss=0.
        indices=torch.randint(len(train),(batch,),generator=gen).tolist()
        for i in indices:
            value=example_loss(model,train[i],labels)/batch
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite task loss')
            value.backward(); loss+=float(value.detach())
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError('Nonfinite parameter gradient')
        torch.nn.utils.clip_grad_norm_(params,1.0)
        opt.step(); model.clamp(); elapsed.append(time.monotonic()-t)
        if step%8==0 or step+1==updates:
            metric({'step':step+1,'loss':loss,'ideal_write_commands':(step+1)*analog_count,
                    'seconds_per_update':sum(elapsed)/len(elapsed)})
    return {'updates':updates,'ideal_write_commands':updates*analog_count,
            'seconds_per_update':sum(elapsed)/len(elapsed),
            'trainable_parameters':sum(p.numel() for p in params)}
