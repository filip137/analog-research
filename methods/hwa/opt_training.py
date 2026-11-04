"""OPT HWA update policy composed over the shared per-step engine."""
import math
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from methods.core.engine import ExperimentComponents, GradientsReadyEvent
from methods.hwa.opt import hardware_view


def make_optimizer(model,config):
    t=config.training
    groups=model.parameter_groups()
    rates=(t.conductance_lr,t.calibration_lr,t.digital_lr)
    betas,eps=((0.9,0.999),1e-8) if t.recipe=='A' else ((0.9,0.98),1e-6)
    return torch.optim.Adam([{'params':p,'lr':lr,'initial_lr':lr} for p,lr in zip(groups,rates)],
                            betas=betas,eps=eps,weight_decay=0,foreach=False)

def lr_factor(step,total,config):
    t=config.training
    if step<t.warmup_updates: return (step+1)/max(1,t.warmup_updates)
    if t.schedule=='constant': return 1.0
    fraction=(step-t.warmup_updates)/max(1,total-1-t.warmup_updates)
    return t.minimum_lr_factor+(1-t.minimum_lr_factor)*0.5*(1+math.cos(math.pi*min(1,fraction)))

class CoreStep:
    """Adapt effective token batches to the repository's authoritative engine."""
    def __init__(self,model,teacher,optimizer,config,generator):
        self.model,self.teacher,self.optimizer,self.config,self.generator=model,teacher,optimizer,config,generator
        self.parameters=tuple(SimpleNamespace(state=p) for p in model.parameters())

    def set_input(self,inputs,reset=False): self.inputs=inputs
    def set_target(self,targets): pass
    def compute_equilibrium(self): pass

    def compute_gradient(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        t=self.config.training
        view,projected=hardware_view(self.model,0.03 if t.arm=='std3' else 0.0,self.generator)
        loss_sum=0.0
        for tokens in self.inputs.split(t.microbatch):
            tokens=tokens.to(self.config.device)
            with torch.no_grad(): reference=self.teacher(tokens)[:,:-1]
            out=self.model(tokens,view)
            loss=F.kl_div(out.flatten(0,1).log_softmax(-1),reference.flatten(0,1).softmax(-1),reduction='batchmean')
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite training KL.')
            (loss*(len(tokens)/len(self.inputs))).backward()
            loss_sum+=float(loss.detach())*len(tokens)/len(self.inputs)
        self.last={'teacher_kl':loss_sum,'projected_surrogate_cells':projected,'sequences':len(self.inputs)}
        return tuple(p.state.grad for p in self.parameters)

    def step(self):
        if self.config.training.recipe=='B':
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0,error_if_nonfinite=True)
        self.optimizer.step()
        self.model.clamp_masters()

    def components(self):
        return ExperimentComponents(self,self,self,self.parameters,self,self,clamp_after_update=False)

    def observe_gradients(self,event):
        if isinstance(event,GradientsReadyEvent):
            for label,parameters in zip(('conductance','calibration','digital'),self.model.parameter_groups()):
                norms=torch.stack([p.grad.detach().norm() for p in parameters])
                self.last[label+'_gradient_norm']=float(norms.norm())
