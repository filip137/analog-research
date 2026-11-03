"""methods.optimizers.masked — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch


class MaskedAdam:
    """Optimize selected normalized physical values; issue only masked commands."""

    def __init__(self, cells, indices, learning_rate, budget):
        self.cells, self.indices = cells, indices
        self.parameters = [torch.nn.Parameter((c.value().flatten()[idx]-c.law.g_min)
                          /(c.law.g_max-c.law.g_min)) for c, idx in zip(cells, indices)]
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        self.budget, self.spent = int(budget), 0
        self.changed = 0
        self.per_update = sum(idx.numel() for idx in indices)
        if self.per_update == 0:
            raise ValueError('Learning requires a nonempty physical mask.')

    def view(self):
        result = []
        for c, idx, p in zip(self.cells, self.indices, self.parameters):
            physical = c.law.g_min + (c.law.g_max-c.law.g_min)*p
            result.append(c.value().flatten().index_copy(0, idx, physical).reshape(c.target.shape))
        return result

    def can_step(self):
        return self.spent + self.per_update <= self.budget

    def step(self):
        if not self.can_step():
            raise RuntimeError('Physical-write budget would be exceeded.')
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.parameters):
            raise FloatingPointError('Nonfinite physical gradient.')
        if not any(p.grad is not None for p in self.parameters):
            raise RuntimeError('No gradient reached the selected physical cells.')
        self.optimizer.step()
        with torch.no_grad():
            for c, idx, p in zip(self.cells, self.indices, self.parameters):
                p.clamp_(0, 1)
                endpoint = c.law.g_min+(c.law.g_max-c.law.g_min)*p
                self.changed += int((endpoint != c.value().flatten()[idx]).sum())
                c.write(idx, endpoint)
        self.spent += self.per_update
        self.optimizer.zero_grad(set_to_none=True)
