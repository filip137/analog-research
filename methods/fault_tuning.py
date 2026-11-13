"""Recovery arms of the fault-recovery paper LR screen on a held CMO DRN deployment.

All gradients use the held apparent conductances (``bank.view``). Physical arms change
conductances only through the plant's write port; the digital LoRA arm leaves every
device untouched and adds a low-rank map of each DRN MLP input to that MLP's output.
The training target is a fixed set of teacher samples per scored position.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from methods.cmo_direct_adam import DirectWriteAdam, ThresholdedBoundedWriteAdam
from methods.opt_recovery import RecoveryWritePort


def sample_loss(logits, samples):
    """Mean of ``-log q(s)`` over positions and the fixed teacher samples ``s`` at each position."""
    if logits.shape[:-1] != samples.shape[:-1]:
        raise ValueError('Teacher samples do not align with the scored positions.')
    return -logits.log_softmax(-1).gather(-1, samples).mean()


def batch_indices(steps, batch, population, seed):
    """Sampled-with-replacement batch order shared by every arm (drift digital-arm convention)."""
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(0, population, (batch,), generator=generator) for _ in range(steps)]


def cosine_to_zero(learning_rate, steps):
    """Rate of each 1-based update, identical to torch CosineAnnealingLR(T_max=steps)."""
    return [learning_rate * 0.5 * (1 + math.cos(math.pi * (k - 1) / steps)) for k in range(1, steps + 1)]


def backward_batch(model, view, tokens, samples, microbatch, device):
    """Accumulate the batch-mean sample loss gradient; return the loss."""
    total = 0.0
    for part, target in zip(tokens.split(microbatch), samples.split(microbatch)):
        fraction = len(part) / len(tokens)
        loss = sample_loss(model(part.to(device), view), target.to(device))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite sample-target training loss.')
        (loss * fraction).backward()
        total += float(loss.detach()) * fraction
    return total


class Adapters(nn.Module):
    """One digital LoRA per DRN MLP: ``y + (alpha / r) B A x`` with x the MLP input (after its LayerNorm).

    A is uniform in +/-1/sqrt(fan_in) (the standard LoRA/kaiming init), B is zero, alpha = r.
    """

    def __init__(self, model, rank, seed):
        super().__init__()
        self.scale = 1.0
        self.a, self.b = nn.ParameterList(), nn.ParameterList()
        for k, block in enumerate(model.blocks):
            width = block.bias2.numel()
            fan_in = block.u1.shape[0] // 2
            bound = 1 / math.sqrt(fan_in)
            generator = torch.Generator().manual_seed(seed + k)
            a = (torch.rand((rank, fan_in), generator=generator) * 2 - 1) * bound
            self.a.append(nn.Parameter(a.to(block.u1)))
            self.b.append(nn.Parameter(torch.zeros((width, rank), dtype=block.u1.dtype, device=block.u1.device)))
        self.handles = []

    def attach(self, model):
        if self.handles:
            raise RuntimeError('Adapters are already attached.')
        for k, block in enumerate(model.blocks):
            def hook(module, args, output, k=k):
                x = args[0]
                return output + self.scale * F.linear(F.linear(x, self.a[k]), self.b[k])
            self.handles.append(block.register_forward_hook(hook))

    def detach_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


@torch.no_grad()
def child_masks(fishers, fraction):
    """Top ``fraction`` of the devices of each matrix by Fisher score (exact count, fixed)."""
    masks = []
    for score in fishers:
        count = max(1, round(fraction * score.numel()))
        top = torch.topk(score.flatten(), count, sorted=False).indices
        mask = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
        mask[top] = True
        masks.append(mask.view_as(score))
    return masks


def fisher_scores(model, bank, tokens, samples, batch, microbatch, device, progress=None):
    """Child-Tuning_D Fisher: mean over the recovery batches of the squared batch gradient.

    One pass over the recovery set in cohort order, at the held deployed state; per device.
    """
    masters = model.parameter_groups()[0]
    model.requires_grad_(False)
    for u in masters:
        u.requires_grad_(True)
    scores = [torch.zeros_like(u) for u in masters]
    model.train()
    batches = math.ceil(len(tokens) / batch)
    for index in range(batches):
        model.zero_grad(set_to_none=True)
        view = bank.view(model)
        rows = slice(index * batch, (index + 1) * batch)
        backward_batch(model, view, tokens[rows], samples[rows], microbatch, device)
        for score, u in zip(scores, masters):
            score.add_(u.grad.detach().square())
        if progress is not None:
            progress(index + 1, batches)
    model.zero_grad(set_to_none=True)
    for u in masters:
        u.requires_grad_(False)
    model.eval()
    return [s / batches for s in scores]


class PhysicalArm:
    """Per-device Adam write commands from held-state gradients; noisy, noiseless or masked."""

    def __init__(self, model, bank, method, learning_rate, steps, masks=None, threshold_sigma=2.0):
        self.model, self.bank, self.method = model, bank, method
        self.masters = model.parameter_groups()[0]
        model.requires_grad_(False)
        for u in self.masters:
            u.requires_grad_(True)
        schedule = ('cosine_to_zero', 0, steps)
        noise = 0. if method == 'noiseless' else 1.
        self.ports = [RecoveryWritePort(a, noise, target_bounds='gmin_gmax') for a in bank.arrays]
        if method == 'noiseless':
            self.writers = [DirectWriteAdam(p, learning_rate=learning_rate, schedule=schedule) for p in self.ports]
        elif method in ('noisy', 'child_d'):
            self.writers = [ThresholdedBoundedWriteAdam(p, learning_rate=learning_rate, threshold_sigma=threshold_sigma,
                                                        schedule=schedule) for p in self.ports]
        else:
            raise ValueError(f'Unknown physical arm {method}.')
        if (masks is None) != (method != 'child_d'):
            raise ValueError('Child-Tuning_D, and only it, takes the fixed device masks.')
        self.masks = masks

    def step(self, tokens, samples, microbatch, device):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        loss = backward_batch(self.model, self.bank.view(self.model), tokens, samples, microbatch, device)
        norm = 0.0
        for k, (writer, u) in enumerate(zip(self.writers, self.masters)):
            gradient = u.grad
            if self.masks is not None:
                gradient = torch.where(self.masks[k], gradient, torch.zeros_like(gradient))
            norm += float(gradient.square().sum())
            writer.step(gradient)
        self.bank.view(self.model)  # Reject invalid coupled denominators immediately.
        self.model.zero_grad(set_to_none=True)
        return {'loss': loss, 'gradient_norm': math.sqrt(norm), 'rate': self.writers[0].rate(self.writers[0].step_index)}

    def close(self):
        for u in self.masters:
            u.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def write_report(self):
        attempts = [w.attempt_counts for w in self.writers]
        costs = self.bank.costs()
        stats = [a.statistics() for a in self.bank.arrays]
        controller = sum(int(c.sum()) for c in attempts)
        if controller != costs['write_attempts']:
            raise RuntimeError('Physical write-attempt accounting differs from controller history.')
        return {'costs': costs,
                'healthy_writes': costs['device_writes'], 'stuck_write_attempts': costs['stuck_write_attempts'],
                'touched_cells': sum(int((c > 0).sum()) for c in attempts),
                'max_writes_per_cell': max(int(c.max()) for c in attempts),
                'out_of_range_devices': sum(s['below_gmin'] + s['above_gmax'] for s in stats),
                'below_gmin_devices': sum(s['below_gmin'] for s in stats),
                'above_gmax_devices': sum(s['above_gmax'] for s in stats),
                'pending_nonzero': sum(int((w.pending != 0).sum()) for w in self.writers if hasattr(w, 'pending')),
                'arrays': stats, 'writers': [w.diagnostics() for w in self.writers]}


class LoraArm:
    """Digital adapters trained with torch Adam and CosineAnnealingLR; the physical state is held."""

    def __init__(self, model, bank, learning_rate, steps, rank, seed):
        self.model, self.bank = model, bank
        model.requires_grad_(False)
        self.adapters = Adapters(model, rank, seed)
        self.adapters.attach(model)
        self.optimizer = torch.optim.Adam(self.adapters.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                                          eps=1e-8, foreach=False)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, steps)

    def step(self, tokens, samples, microbatch, device):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        rate = self.optimizer.param_groups[0]['lr']
        loss = backward_batch(self.model, self.bank.view(self.model), tokens, samples, microbatch, device)
        norm = math.sqrt(sum(float(p.grad.square().sum()) for p in self.adapters.parameters()))
        self.optimizer.step()
        self.schedule.step()
        return {'loss': loss, 'gradient_norm': norm, 'rate': rate}

    def close(self):
        self.adapters.detach_hooks()
        self.model.eval()

    def state(self):
        return {k: v.detach().cpu().clone() for k, v in self.adapters.state_dict().items()}
