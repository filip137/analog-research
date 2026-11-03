"""hardware.layered_pcm — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch
from hardware.devices.pcm import GaussianEndpointArray, PermanentFaults
from core.seeds import binding_seed


class LayeredPcmArray:
    def __init__(self, target, layout, dataset, kind, rate, seed, endpoint):
        self.children = []
        self.layout = layout
        for m in layout:
            f = PermanentFaults.sample(m.size,kind,rate,binding_seed(seed,dataset,m.name),target.device)
            self.children.append(GaussianEndpointArray(target[m.offset:m.offset+m.size],f,binding_seed(endpoint,dataset,m.name)))
        self.faults = PermanentFaults(torch.cat([a.faults.mask for a in self.children],1),
                                     torch.cat([a.faults.conductance for a in self.children],1),kind,rate,seed)
    def read(self): return torch.cat([a.read() for a in self.children])
    def program(self,target):
        for a,m in zip(self.children,self.layout): a.program(target[m.offset:m.offset+m.size])
    def state_dict(self): return {'kind':'layered_pcm','children':[a.state_dict() for a in self.children]}
    def load_state_dict(self,s):
        if s.get('kind')!='layered_pcm' or len(s['children'])!=len(self.children): raise ValueError('PCM layout changed.')
        for a,v in zip(self.children,s['children']): a.load_state_dict(v)
    def verify_permanence(self):
        for a in self.children: a.verify_permanence()
    def cost(self):
        rows=[a.cost() for a in self.children]
        if len({r['array_reprogram_calls'] for r in rows})!=1: raise RuntimeError('Partial layer write.')
        return {k:(rows[0][k] if k=='array_reprogram_calls' else sum(r[k] for r in rows)) for k in rows[0]}
