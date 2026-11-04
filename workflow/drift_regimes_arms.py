"""Maintenance arms of the drift-regimes screen (exp-002), on the OPT crossbar.

Every arm starts from a copy of the same released deployment, so untouched
devices see identical future drift across arms (counter-keyed paths). Controllers
see only exact reads (no read noise), their own write times and the digital
targets; never D, C, Y or a.

Methods (``training.drift_regimes_config.ARM_KEYS``):
- ``none``: untreated.
- ``calibration``: per output column affine readout correction of every analog
  matrix, ``y' = g y + s`` after the digital bias, refitted at each event by
  least squares on probe inputs. The reference for a probe input is the output
  of the digital target weights for that same input. Matrices are fitted in
  forward order, each after the earlier ones are corrected.
- ``layernorm``: the affine of the LayerNorm in front of each analog MLP,
  trained on the held read at each event (warm start), teacher-KL objective.
- ``lora``: a digital rank-r adapter on each analog matrix, trained the same
  way (warm start from the previous event's adapter).
- ``rewrite``: P&V reset of the devices whose observed deviation from the
  nominal expectation ``requested + m u^alpha`` is largest (a fraction per
  matrix), or of every device at fraction 1, to the deployment targets.

Digital training takes no device time: it uses the read held at the event, and
the devices keep drifting from the event clock. A rewrite's P&V moves the
clock on by its verify delays.
"""
from types import SimpleNamespace
import math
import time

import torch
from torch.nn import functional as F

from training.drift_compensation import Compensation, MatrixCorrection
from training.drift_family import program_verify
from training.drift_recovery import tensor_hash
from workflow.networks import opt_mlp

GAIN_BOUNDS = (0.1, 10.0)


def adaptation_cohort(lifecycle, settings):
    """Seeded selection of training-split blocks of the adaptation corpus, pinned by hash."""
    from dataclasses import replace
    data = replace(lifecycle.data, corpus=settings['corpus'], corpus_sha256=settings['corpus_sha256'])
    corpus, _ = opt_mlp.load_corpus(data, lifecycle.network.model)
    length = lifecycle.data.sequence_length
    stream = corpus['splits'][settings['split']]
    blocks = stream[:len(stream) // length * length].reshape(-1, length)
    if len(blocks) < settings['sequences']:
        raise ValueError('Adaptation corpus is too small for the declared cohort.')
    order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(settings['seed']))
    tokens = blocks[order[:settings['sequences']]].contiguous()
    if tensor_hash(tokens) != settings['cohort_sha256']:
        raise ValueError('Drift-regimes adaptation cohort hash mismatch.')
    return tokens


@torch.no_grad()
def prefix_states(network, tokens, device, batch=16):
    if network.digital is not None:
        raise ValueError('Drift-regimes arms expect the analog-scope OPT student.')
    return torch.cat([network.features(tokens[i:i + batch].to(device)) for i in range(0, len(tokens), batch)])


def _identity(network, index, device):
    outputs, inputs = network.layout[index].shape
    return MatrixCorrection(torch.ones(inputs, device=device), torch.ones(outputs, device=device),
                            torch.zeros(outputs, device=device), torch.ones(outputs, device=device))


@torch.no_grad()
def _matrix_input(network, hidden, q, index):
    """Input of analog matrix ``index`` in ``forward_features`` order, with the active corrections."""
    for k in range(index // 2 + 1):
        hidden = network.teacher.layers[network.start + k].attention(hidden)
        x = network.mlp_norms[k](hidden)
        if 2 * k == index:
            return x
        x = F.relu(network._matrix(x, 2 * k, q))
        if 2 * k + 1 == index:
            return x
        hidden = hidden + network._matrix(x, 2 * k + 1, q)
    raise IndexError(index)


@torch.no_grad()
def fit_calibration(network, q, probe, batch=8):
    """Per-column least-squares affine correction of every analog matrix (see module doc)."""
    device = q.device
    matrices = [_identity(network, i, device) for i in range(network.base_matrices)]
    network.drift_compensation = Compensation(matrices)
    try:
        for index in range(network.base_matrices):
            sums = None
            for chunk in probe.split(batch):
                x = _matrix_input(network, chunk, q, index)
                current = network._matrix(x, index, q).flatten(0, -2).double()
                reference = network._matrix(x, index, network.q.detach()).flatten(0, -2).double()
                part = torch.stack((current.sum(0), reference.sum(0), current.square().sum(0),
                                    (current * reference).sum(0)))
                n = current.shape[0]
                sums = (n, part) if sums is None else (sums[0] + n, sums[1] + part)
            n, (sy, sr, syy, syr) = sums[0], sums[1]
            my, mr = sy / n, sr / n
            var = (syy / n - my.square()).clamp_min(0)
            cov = syr / n - my * mr
            gain = torch.where(var > 1e-12, cov / var.clamp_min(1e-12), torch.ones_like(var)).clamp(*GAIN_BOUNDS)
            matrices[index].affine_gain = gain.float()
            matrices[index].shift = (mr - gain * my).float()
    finally:
        network.drift_compensation = None
    return Compensation(matrices)


def train_digital(network, parameters, q, hidden, *, steps, batch, learning_rate, seed):
    """Adam with cosine decay on the teacher KL of the held read ``q``; returns the loss trace."""
    for p in network.parameters():
        p.requires_grad_(False)
    for p in parameters:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    generator = torch.Generator().manual_seed(seed)
    q = q.detach()
    losses = []
    network.train()
    with torch.enable_grad():
        for _ in range(steps):
            index = torch.randint(0, len(hidden), (batch,), generator=generator).to(hidden.device)
            states = hidden[index]
            with torch.no_grad():
                teacher = network.teacher_logits(states).flatten(0, 1).log_softmax(-1)
            student = network.forward_features(states, q).flatten(0, 1).log_softmax(-1)
            loss = (teacher.exp() * (teacher - student)).sum(-1).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite digital adaptation loss.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            losses.append(float(loss))
    network.eval()
    for p in parameters:
        p.requires_grad_(False)
    return {'first_loss': sum(losses[:10]) / min(10, len(losses)),
            'last_loss': sum(losses[-10:]) / min(10, len(losses))}


def nominal_deviation(cells):
    """Observed read minus the nominal expectation at each device's own age; counts a full read."""
    law = cells.law
    read = cells.read()
    u = (cells.clock - cells.last_write) / law.reference_seconds
    expected = cells.requested + (law.mean_us * u.pow(law.mean_exponent)).to(torch.float32)
    return (read - expected).flatten()


def rewrite(cells, arm, clock, verify_seconds):
    for c in cells:
        c.advance(clock)
    cost = {'reset_attempts': 0, 'verify_reads': 0, 'selection_reads': 0, 'selected_devices': 0}
    for c in cells:
        n = c.target.numel()
        if arm['fraction'] >= 1:
            ids = torch.arange(n, device=c.target.device)
        else:
            error = nominal_deviation(c).abs()
            cost['selection_reads'] += n
            count = max(1, round(n * arm['fraction']))
            ids = torch.argsort(error, descending=True, stable=True)[:count]
        stats = program_verify(c, ids, c.target.flatten()[ids], attempts=arm['attempts'],
                               threshold=arm['threshold'], verify_seconds=verify_seconds)
        cost['reset_attempts'] += stats['reset_attempts']
        cost['verify_reads'] += stats['verify_reads']
        cost['selected_devices'] += ids.numel()
    return cost


class Arm:
    """One arm's controller state on the network; ``event`` acts, ``context`` prepares evaluation."""

    def __init__(self, arm, network, adaptation, hidden, probe, realization, verify_seconds):
        self.arm, self.network, self.adaptation = arm, network, adaptation
        self.hidden, self.probe = hidden, probe
        self.realization, self.verify_seconds = realization, verify_seconds
        self.method = arm['method']
        self.correction = None
        self.norms = {k: v.detach().clone() for k, v in network.mlp_norms.state_dict().items()}
        self.cost = {'events': 0, 'reset_attempts': 0, 'verify_reads': 0, 'selection_reads': 0,
                     'selected_devices': 0, 'probe_tokens': 0, 'optimizer_steps': 0,
                     'adaptation_sequence_presentations': 0, 'adaptation_tokens': 0,
                     'digital_parameters': 0, 'event_wall_seconds': 0.}
        self.events = []
        if self.method == 'lora':
            network.enable_lora(SimpleNamespace(rank=arm['rank'], alpha=float(arm['rank'])), seed=realization)
            self.cost['digital_parameters'] = sum(p.numel() for p in network.lora.parameters())
        elif self.method == 'layernorm':
            self.cost['digital_parameters'] = sum(p.numel() for p in network.mlp_norms.parameters())
        elif self.method == 'calibration':
            self.cost['digital_parameters'] = sum(2 * network.layout[i].shape[0] for i in range(network.base_matrices))

    def close(self):
        self.network.enable_lora(None)
        self.network.mlp_norms.load_state_dict(self.norms)
        self.network.drift_compensation = None
        for p in self.network.parameters():
            p.requires_grad_(False)

    def context(self):
        self.network.drift_compensation = self.correction

    def event(self, index, age, clock, cells, bind):
        started = time.monotonic()
        record = {'event': index, 'seconds': age, 'clock_seconds': clock}
        self.network.drift_compensation = None
        if self.method == 'rewrite':
            cost = rewrite(cells, self.arm, clock, self.verify_seconds)
            record.update(cost, end_clock_seconds=max(c.clock for c in cells))
            for k, v in cost.items():
                self.cost[k] += v
        else:
            q = bind([c.advance(clock) for c in cells])
            reads = sum(c.target.numel() for c in cells)
            for c in cells:
                c.selection_reads += c.target.numel()
            self.cost['selection_reads'] += reads
            record['selection_reads'] = reads
            if self.method == 'calibration':
                self.correction = fit_calibration(self.network, q, self.probe)
                tokens = self.probe.shape[0] * self.probe.shape[1]
                self.cost['probe_tokens'] += tokens
                record['probe_tokens'] = tokens
            else:
                if self.method == 'lora':
                    parameters = list(self.network.lora.parameters())
                else:
                    parameters = list(self.network.mlp_norms.parameters())
                a = self.adaptation
                trace = train_digital(self.network, parameters, q, self.hidden, steps=a['steps'], batch=a['batch'],
                                      learning_rate=self.arm['learning_rate'], seed=a['seed'] + 1000 * index)
                presentations = a['steps'] * a['batch']
                self.cost['optimizer_steps'] += a['steps']
                self.cost['adaptation_sequence_presentations'] += presentations
                self.cost['adaptation_tokens'] += presentations * self.hidden.shape[1]
                record.update(trace, optimizer_steps=a['steps'])
        self.cost['events'] += 1
        record['wall_seconds'] = time.monotonic() - started
        self.cost['event_wall_seconds'] += record['wall_seconds']
        self.events.append(record)
        return record


def timeline(ages, events):
    """(age, evaluate, event index or None); an evaluation at an event age comes first."""
    points = []
    for t in sorted(set(ages) | set(events)):
        if t in ages:
            points.append((t, True, None))
        if t in events:
            points.append((t, False, events.index(t)))
    return points


def finite(metrics):
    return all(math.isfinite(metrics[k]) for k in ('cross_entropy', 'perplexity', 'teacher_kl'))
