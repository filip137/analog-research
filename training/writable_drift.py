"""Custom additive drift with persistent identities and writable cell clocks.

Mirrored verbatim in the DRN worktree. This is a synthetic model, not a CMO
programming fit. A programming command is an exact endpoint operation, not a
pulse. Times are seconds; conductances are microSiemens.
"""
from dataclasses import asdict
import hashlib
import math

import torch
from training.drift_law import DriftLaw


def stream_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}/{name}'.encode()).digest()[:4], 'little')


def _hash32(value):
    value = value & 0xffffffff
    value = ((value ^ (value >> 16)) * 0x7feb352d) & 0xffffffff
    value = ((value ^ (value >> 15)) * 0x846ca68b) & 0xffffffff
    return value ^ (value >> 16)


def cycle_normal(indices, cycles, seed):
    """Counter-keyed draws: selection/order in another cell cannot change a draw."""
    counter = indices.to(torch.int64) ^ _hash32(cycles.to(torch.int64) + int(seed))
    u = (_hash32(counter).double() + .5) / 2**32
    v = (_hash32(counter ^ 0x9e3779b9).double() + .5) / 2**32
    return (torch.sqrt(-2 * torch.log(u)) * torch.cos(2 * math.pi * v)).float()


class WritableDrift:
    """One physical tensor; no gradients or model shadow weights live here."""

    def __init__(self, target, *, law, rho, seed, name):
        if not math.isfinite(rho) or not 0 <= rho <= 1:
            raise ValueError('Persistent variance fraction must be in [0,1].')
        if target.dtype != torch.float32 or not bool(torch.isfinite(target).all()):
            raise ValueError('Expected finite FP32 physical targets.')
        self.law, self.rho, self.name = law, float(rho), name
        self.seed = stream_seed(seed, name)
        self.target = target.detach().clamp(law.g_min, law.g_max).clone()
        self.anchor = self.target.clone()
        self.last_write = torch.zeros_like(target)
        self.cycles = torch.zeros_like(target, dtype=torch.int32)
        ids = torch.arange(target.numel(), device=target.device, dtype=torch.int64).reshape(target.shape)
        e0 = cycle_normal(ids, self.cycles, self.seed)
        independent = cycle_normal(ids, self.cycles, self.seed ^ 0x85ebca6b)
        # Conditional construction pairs the *same initial combined drift* across
        # rho, while z and the initial cycle component retain independent N(0,1) laws.
        self.persistent = math.sqrt(rho) * e0 + math.sqrt(1-rho) * independent
        self.epsilon = e0
        self.clock = 0.0
        self.write_commands = 0
        self.selection_reads = 0

    def value(self, seconds=None):
        seconds = self.clock if seconds is None else float(seconds)
        if not math.isfinite(seconds) or seconds < self.clock:
            raise ValueError('A writable trajectory cannot move backwards in time.')
        age = (seconds - self.last_write) / self.law.reference_seconds
        raw = (self.anchor + self.law.mean_us * age.pow(self.law.mean_exponent)
               + self.law.spread_us * age.pow(self.law.spread_exponent) * self.epsilon)
        return raw.clamp(self.law.g_min, self.law.g_max)

    def advance(self, seconds):
        value = self.value(seconds)
        self.clock = float(seconds)
        return value

    def read(self):
        self.selection_reads += self.target.numel()
        return self.value()

    @torch.no_grad()
    def write(self, indices, values):
        indices = indices.to(device=self.target.device, dtype=torch.long).flatten()
        values = values.detach().to(self.target).flatten()
        if indices.numel() != values.numel() or not bool(torch.isfinite(values).all()):
            raise ValueError('Invalid programming command.')
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.target.numel()
                                or indices.unique().numel() != indices.numel()):
            raise ValueError('Programming indices must be unique and in range.')
        if not indices.numel():
            return
        cycles = self.cycles.view(-1)[indices] + 1
        self.anchor.view(-1)[indices] = values.clamp(self.law.g_min, self.law.g_max)
        self.last_write.view(-1)[indices] = self.clock
        self.cycles.view(-1)[indices] = cycles
        self.epsilon.view(-1)[indices] = (
            math.sqrt(self.rho) * self.persistent.view(-1)[indices]
            + math.sqrt(1-self.rho) * cycle_normal(indices, cycles, self.seed))
        self.write_commands += indices.numel()

    def observed_stability(self):
        """Only the initial programming episode identifies the declared proxy."""
        if self.clock <= 0 or bool((self.cycles != 0).any()):
            raise ValueError('Stability selection requires the initial deployment episode.')
        g = self.read()
        eligible = (g > self.law.g_min) & (g < self.law.g_max)
        u = self.clock / self.law.reference_seconds
        scale = self.law.spread_us * u**self.law.spread_exponent
        if scale == 0:
            score = torch.zeros_like(g)
        else:
            score = ((g-self.target-self.law.mean_us*u**self.law.mean_exponent)/scale).abs()
        return score, eligible

    def state_dict(self):
        return {'schema': 'custom_writable_drift.v1', 'law': asdict(self.law),
                'rho': self.rho, 'name': self.name, 'seed': self.seed, 'clock': self.clock,
                'write_commands': self.write_commands, 'selection_reads': self.selection_reads,
                **{k: getattr(self, k).detach().cpu().clone() for k in
                   ('target', 'anchor', 'last_write', 'cycles', 'persistent', 'epsilon')}}

    @classmethod
    def restore(cls, state, device='cpu'):
        if state['schema'] != 'custom_writable_drift.v1':
            raise ValueError('Unknown writable drift state.')
        obj = cls.__new__(cls)
        obj.law = DriftLaw(**state['law'])
        for k in ('rho', 'name', 'seed', 'clock', 'write_commands', 'selection_reads'):
            setattr(obj, k, state[k])
        for k in ('target', 'anchor', 'last_write', 'cycles', 'persistent', 'epsilon'):
            setattr(obj, k, state[k].to(device).clone())
        shape = obj.target.shape
        if any(getattr(obj, k).shape != shape for k in
               ('anchor', 'last_write', 'cycles', 'persistent', 'epsilon')):
            raise ValueError('Inconsistent serialized cell shapes.')
        if bool((obj.last_write > obj.clock).any()) or not bool(torch.isfinite(obj.value()).all()):
            raise ValueError('Invalid serialized cell clock/state.')
        return obj


def _smallest(score, eligible, count):
    ids = torch.nonzero(eligible.flatten(), as_tuple=False).flatten()
    if len(ids) < count:
        raise ValueError(f'Only {len(ids)} uncensored cells for a requested mask of {count}.')
    # Stable ties preserve physical index order (including the no-drift control).
    return ids[torch.argsort(score.flatten()[ids], stable=True)[:count]]


def select_masks(cells, groups, roles, fraction, seed):
    """10% of *all* cells per MLP, with observable-pool matched controls.

    groups and roles are physical-cell tensors with integer labels supplied by
    the architecture. Random and oracle masks match the estimated mask's counts
    in every MLP x rail-role x initial-conductance-decile stratum.
    """
    scores, eligible = zip(*(c.observed_stability() for c in cells))
    sizes = [c.target.numel() for c in cells]
    device = cells[0].target.device
    score = torch.cat([v.flatten() for v in scores])
    pool = torch.cat([v.flatten() for v in eligible])
    group = torch.cat([v.flatten().to(device) for v in groups])
    role = torch.cat([v.flatten().to(device) for v in roles])
    decile = torch.cat([((c.target.flatten()-c.law.g_min)/(c.law.g_max-c.law.g_min)*10)
                        .long().clamp(0, 9) for c in cells])
    z = torch.cat([c.persistent.flatten().abs() for c in cells])
    chosen = torch.zeros_like(pool)
    for g in group.unique():
        in_group = group == g
        k = max(1, int(int(in_group.sum()) * fraction))
        chosen[_smallest(score, pool & in_group, k)] = True
    random_mask, oracle = torch.zeros_like(pool), torch.zeros_like(pool)
    strata = group.to(torch.int64)*1000 + role.to(torch.int64)*10 + decile
    rng = torch.Generator(device=device).manual_seed(seed)
    for stratum in strata[chosen].unique(sorted=True):
        available = pool & (strata == stratum)
        k = int((chosen & (strata == stratum)).sum())
        ids = torch.nonzero(available, as_tuple=False).flatten()
        random_mask[ids[torch.randperm(len(ids), generator=rng, device=device)[:k]]] = True
        oracle[_smallest(z, available, k)] = True
    def split_indices(mask):
        return [torch.nonzero(v, as_tuple=False).flatten() for v in mask.split(sizes)]
    return {'stable': split_indices(chosen), 'random': split_indices(random_mask),
            'oracle': split_indices(oracle)}, {
                'physical_devices': sum(sizes), 'selected_devices': int(chosen.sum()),
                'eligible_devices': int(pool.sum()), 'excluded_devices': int((~pool).sum()),
                'selection_reads': sum(sizes), 'matched_strata': 'MLP/physical-role/target-decile'}


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
