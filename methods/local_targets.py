"""Block-local recovery targets: no error crosses the digital attention.

DRNs 6-10 each receive only a local squared-error loss; every block input is
detached, so a DRN's gradient never passes through attention or another DRN.
The last analog MLP, which no attention follows, keeps the end-to-end teacher
KL. The local reference is the clean HWA model: the same OptDrn evaluated on
its unmodified master conductances (view=None), on the same tokens.

  cached    DRN_k(x_clean) -> MLP_k_clean(x_clean)    own faults, clean inputs
  onpolicy  DRN_k(x_dep)   -> MLP_k_clean(x_dep)      own faults, deployed inputs
  stream    h_dep + DRN_k(x_dep) -> h_clean           own faults + upstream drift

x is the block's normalized MLP input, h the residual stream after attention
(dep: deployed held state; clean: masters). The squared error sums features
and averages tokens.

The solver settles for T sweeps without a graph and adds K differentiable
sweeps in training. Targets and stream continuations are therefore solved at
the same T+K depth as the predictions (built with a graph, then detached),
so a healthy state has zero local loss.
"""

import torch
from torch.nn import functional as F

from methods.opt_recovery import RecoveryStep

MODES = ('cached', 'onpolicy', 'stream')


def _squared(prediction, target):
    return (prediction - target).square().sum(-1).mean()


def _settled(block, x, held=None):
    """Detached block output at the training solver depth."""
    with torch.enable_grad():
        return block(x, held).detach()


def local_forward(model, tokens, view, mode):
    """Per-block local losses for all but the last DRN, and deployed logits."""
    if mode not in MODES:
        raise ValueError(f'Expected local target mode in {MODES}.')
    digital = model.digital
    analog = list(zip(digital.layers[model.start:], model.blocks))
    with torch.no_grad():
        prefix = digital.embed(tokens)
        for layer in digital.layers[:model.start]:
            prefix = layer(prefix)
        clean, hidden = [], prefix
        for layer, block in analog:
            mid = layer.attention(hidden)
            x = layer.final_layer_norm(mid)
            y = _settled(block, x)
            hidden = mid + y
            clean.append((x, y, hidden))
    losses, hidden = [], prefix
    last = len(analog) - 1
    for k, (layer, block) in enumerate(analog):
        mid = layer.attention(hidden.detach())
        x = layer.final_layer_norm(mid)
        clean_x, clean_y, clean_h = clean[k]
        if k < last and mode == 'cached':
            losses.append(_squared(block(clean_x, view[k]), clean_y))
            out = _settled(block, x, view[k])
        else:
            out = block(x, view[k])
            if k < last and mode == 'onpolicy':
                target = _settled(block, x)
                losses.append(_squared(out, target))
            elif k < last:
                losses.append(_squared(mid + out, clean_h))
        hidden = mid + out
    return losses, digital.logits(hidden[:, :-1])


class LocalRecoveryStep(RecoveryStep):
    """RecoveryStep whose gradients come from block-local targets."""

    def __init__(self, model, teacher, bank, config, mode):
        if mode not in MODES:
            raise ValueError(f'Expected local target mode in {MODES}.')
        super().__init__(model, teacher, bank, config)
        if self.calibration:
            raise ValueError('Local targets apply to conductance learning only.')
        self.mode = mode

    def compute_gradient(self):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        view = self.bank.view(self.model)
        kl_sum, local_sums = 0.0, [0.0] * (len(self.model.blocks) - 1)
        for batch in self.inputs.split(self.config.training.microbatch):
            batch = batch.to(self.config.device)
            with torch.no_grad():
                reference = self.teacher(batch)[:, :-1]
            losses, prediction = local_forward(self.model, batch, view, self.mode)
            kl = F.kl_div(prediction.flatten(0, 1).log_softmax(-1), reference.flatten(0, 1).softmax(-1),
                          reduction='batchmean')
            if not all(bool(torch.isfinite(v)) for v in (kl, *losses)):
                raise FloatingPointError('Nonfinite local-target recovery loss.')
            fraction = len(batch) / len(self.inputs)
            # Block inputs are detached: each loss reaches only its own DRN.
            ((kl + sum(losses)) * fraction).backward()
            kl_sum += float(kl.detach()) * fraction
            for k, loss in enumerate(losses):
                local_sums[k] += float(loss.detach()) * fraction
        self.last = {'teacher_kl': kl_sum, 'local_losses': local_sums, 'local_mode': self.mode,
                     'sequences': len(self.inputs), 'primary_state': 'held_apparent'}
        return tuple(p.state.grad for p in self.parameters)
