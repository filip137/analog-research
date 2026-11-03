"""networks.opt_drift — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from core.tensor_identity import state_hash


class CrossbarAdapter:
    architecture = 'crossbar'

    def __init__(self, network, device):
        self.network = network.eval().requires_grad_(False)
        self.device = device

    def source_hash(self): return state_hash(self.network.state_dict())

    def clean(self, tokens): return self.network(tokens.to(self.device))

    def physical_layout(self):
        targets, groups, roles, names = [], [], [], []
        for k, matrix in enumerate(self.network.layout):
            q = self.network.q[matrix.offset:matrix.offset+matrix.size].detach()
            target = torch.stack((48.6+39.6*q, torch.full_like(q, 48.6)))
            role = torch.zeros_like(target, dtype=torch.int8)
            role[0] = 2*(k % 2)
            role[1] = 2*(k % 2)+1
            targets.append(target)
            roles.append(role)
            groups.append(torch.full_like(role, k//2))
            names.append(matrix.name)
        return targets, groups, roles, names

    def bind(self, physical):
        return torch.cat([(g[0]-g[1])/39.6 for g in physical])

    def block(self, k, x, bound):
        y = self.network._matrix(x, 2*k, bound)
        return self.network._matrix(F.relu(y), 2*k+1, bound)

    def forward(self, tokens, physical=None, correction=None, capture=False):
        if physical is None and correction is None and not capture: return self.clean(tokens)
        q = self.network.q if physical is None else self.bind(physical)
        digital = self.network.digital if self.network.digital is not None else self.network.teacher
        hidden = digital.embed(tokens.to(self.device))
        for i in range(self.network.start): hidden = digital.layers[i](hidden)
        return self.network.forward_suffix(hidden, q, digital=digital,
                                           correction=correction, capture=capture)
