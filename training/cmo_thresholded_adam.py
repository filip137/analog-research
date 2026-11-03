"""CMO thresholded write-Adam, transcribed from HWA b0c1f06 workflow/onchip.py.

The controller sees an apparent-read/write port and its own command history.
It never reads hidden fault identities or successful-write counters.
"""

import math
import torch

from training.reram_cmo import HALF_RANGE_US, MID_US, programming_sigma


class ThresholdedWriteAdam:
    def __init__(self, port, *, learning_rate, threshold_sigma=2.0, cap=4, noise_scale=1.0):
        if (not math.isfinite(learning_rate) or learning_rate<0 or threshold_sigma<=0
                or (cap is not None and (type(cap) is not int or cap<1))):
            raise ValueError('Invalid CMO write-Adam contract.')
        self.port=port
        self.learning_rate=float(learning_rate)
        self.threshold_sigma=float(threshold_sigma)
        self.cap=cap
        self.noise_scale=float(noise_scale)
        self.threshold_q=threshold_sigma*noise_scale*programming_sigma(MID_US,2.0)/HALF_RANGE_US
        template=port.read()
        self.first=torch.zeros_like(template)
        self.second=torch.zeros_like(template)
        self.pending=torch.zeros_like(template)
        self.attempt_counts=torch.zeros_like(template,dtype=torch.int32)
        self.step_index=0

    @torch.no_grad()
    def step(self, gradient):
        if gradient.shape!=self.first.shape or not bool(torch.isfinite(gradient).all()):
            raise ValueError('Expected a finite normalized gradient for every DRN device.')
        self.step_index+=1
        self.first.mul_(0.9).add_(gradient,alpha=0.1)
        self.second.mul_(0.999).addcmul_(gradient,gradient,value=0.001)
        command=(-self.learning_rate*(self.first/(1-0.9**self.step_index))
                 /((self.second/(1-0.999**self.step_index)).sqrt()+1e-8))
        if self.learning_rate==0:
            return
        eligible=torch.ones_like(self.attempt_counts,dtype=torch.bool) if self.cap is None else self.attempt_counts<self.cap
        self.pending.add_(torch.where(eligible,command,0.0))
        if not bool(torch.isfinite(self.pending).all()):
            raise FloatingPointError('Nonfinite accumulated physical write command.')
        requested=(self.pending.abs()>self.threshold_q)&eligible
        issued=torch.where(requested,self.pending,0.0)
        self.port.write(issued)
        self.attempt_counts.add_(requested)
        # Failed/stuck attempts consume budget and pending commands too.
        self.pending.sub_(issued)

    def diagnostics(self):
        return {'threshold_q':self.threshold_q,'threshold_reference':'nominal_midrange',
                'issued_write_attempts':int(self.attempt_counts.sum()),
                'max_device_attempts':int(self.attempt_counts.max()),
                'capped_devices':0 if self.cap is None else int((self.attempt_counts>=self.cap).sum()),
                'max_write_attempts':self.cap,
                'pending_rms_q':float(self.pending.square().mean().sqrt())}

    def state_dict(self):
        return {'contract':['cmo_drn.thresholded_adam.v1',self.learning_rate,self.threshold_sigma,self.cap,self.noise_scale],
                'step_index':self.step_index,
                **{k:getattr(self,k).detach().cpu().clone() for k in ('first','second','pending','attempt_counts')}}

    def load_state_dict(self,state):
        expected=['cmo_drn.thresholded_adam.v1',self.learning_rate,self.threshold_sigma,self.cap,self.noise_scale]
        if state.get('contract')!=expected:
            raise ValueError('Physical writer continuation contract differs.')
        for key in ('first','second','pending','attempt_counts'):
            current=getattr(self,key)
            if state[key].shape!=current.shape or state[key].dtype!=current.dtype:
                raise ValueError(f'Writer state shape/type mismatch: {key}.')
            current.copy_(state[key].to(current.device))
        if not bool((self.attempt_counts>=0).all()) or (self.cap is not None and not bool((self.attempt_counts<=self.cap).all())):
            raise ValueError('Invalid writer attempt counts.')
        self.step_index=state['step_index']
