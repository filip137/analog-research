"""DRN physical adapter for shared recovery algorithms."""
import torch
from hardware.faults import state_hash

class DrnAdapter:
    architecture = 'drn'

    def __init__(self, model, device):
        self.model = model.eval().requires_grad_(False)
        self.device = device

    def source_hash(self): return state_hash(self.model.state_dict())

    def clean(self, tokens): return self.model(tokens.to(self.device))

    def physical_layout(self):
        targets, groups, roles, names = [], [], [], []
        for k, block in enumerate(self.model.blocks):
            for j, u in enumerate((block.u1, block.u2)):
                target = (48.6+39.6*u.detach()).contiguous()
                row = torch.arange(u.shape[0], device=u.device)//(u.shape[0]//2)
                col = torch.arange(u.shape[1], device=u.device)
                col = col % 2 if j else col//(u.shape[1]//2)
                role = (4*j+row[:, None]*2+col[None, :]).to(torch.int8)
                targets.append(target)
                roles.append(role)
                groups.append(torch.full_like(role, k))
                names.append(f'decoder.layers.{self.model.start+k}.fc{j+1}')
        return targets, groups, roles, names

    def bind(self, physical):
        return [(physical[i]/88.2, physical[i+1]/88.2) for i in range(0, len(physical), 2)]

    def block(self, k, x, bound):
        return self.model.blocks[k](x, bound[k])

    def forward(self, tokens, physical=None, correction=None, capture=False):
        # The native solver uses extra truncated-BPTT sweeps only in training
        # mode. Frozen model parameters do not disable physical-view gradients.
        learning = torch.is_grad_enabled() and physical is not None and any(g.requires_grad for g in physical)
        for block in self.model.blocks: block.solver.train(learning)
        if physical is None and correction is None and not capture: return self.clean(tokens)
        bound = None if physical is None else self.bind(physical)
        return self.model(tokens.to(self.device), bound, correction=correction, capture=capture)
