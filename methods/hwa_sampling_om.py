"""methods.hwa_sampling_om — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch
from hardware.plants import EndpointKernel


class MixedOmSampler:
    def __init__(self,tables,seed,device):
        self.kernels={k:EndpointKernel(v if k=='healthy' else v+torch.linspace(-1,1,v.shape[0])[:,None],seed+17*i,device) for i,(k,v) in enumerate(sorted(tables.items()))}
        self.generator=torch.Generator(device=device).manual_seed(seed+999)
    def perturb(self,q,kind,rate,strength):
        healthy=self.kernels['healthy'].perturb(q,strength)
        if not rate: return healthy
        endpoint=self.kernels[kind].perturb(q)-q
        failed=q+(endpoint-q).detach()
        mask=torch.rand(q.shape,device=q.device,generator=self.generator)<rate
        return torch.where(mask,failed,healthy)
    def state_dict(self):
        return {'mask_rng':self.generator.get_state(), 'kernel_rngs':{k:v.generator.get_state() for k,v in self.kernels.items()}}
