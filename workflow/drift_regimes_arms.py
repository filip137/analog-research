"""Maintenance arms of the drift-regimes screen (exp-002), on the OPT crossbar.

Every arm starts from a copy of the same released deployment, so untouched
devices see identical future drift across arms (counter-keyed paths). Controllers
see only exact reads (no read noise), their own write times and the digital
targets; never D, C, Y or a.

Methods (``training.drift_regimes_config.ARM_KEYS``):
- ``none``: untreated.
- ``calibration``: per output column affine readout correction of every analog
  matrix, ``y' = g y + s`` after the digital bias, refitted at each event by
  mean/variance matching on probe inputs: ``g = sd_ref / sd``, ``s = mean_ref -
  g mean`` per column. The reference is the output of the digital target
  weights for the same probe inputs. Matrices are fitted in forward order,
  each after the earlier ones are corrected.
- ``layernorm``: the affine of the LayerNorm in front of each analog MLP,
  trained on the held read at each event (warm start), teacher-KL objective.
- ``lora``: a digital rank-r adapter on each analog matrix, trained the same
  way (warm start from the previous event's adapter).
- ``rewrite``: P&V reset of the devices whose observed deviation from the
  nominal expectation ``requested + m u^alpha`` is largest (a fraction per
  matrix), or of every device at fraction 1, to the deployment targets.
- ``precompensated`` (A10): ``rewrite``'s selection; each reset target is
  offset by ``-c e`` with c regressed from the previous event's rewrites (see
  ``rewrite``).
- ``rewrite_oracle`` (O2): selection by the hidden deviation at the log-midpoint
  of the interval to the next event (or the horizon). Labelled oracle.
- ``lora_oracle`` (O1): a fresh LoRA trained on the read at each evaluation
  age and evaluated right after it. Labelled oracle.
- ``lora_rescaled`` (A6): LoRA trained once; at each later read the controller
  estimates beta (spread growth) and kappa (decorrelation) from the deviations
  of a fixed 1/16 device subsample against its training-time read. Between
  reads the LoRA output is scaled by ``(t1/t)^kappa (t/t1)^beta``.
- ``lora_clean``: LoRA trained on the clean digital target weights, then
  deployed on the drifting devices (no-drift diagnostic).
- ``learning`` (A11): closed-loop analog learning. Each event runs ``rounds``
  rounds. A round reads the array and takes the teacher-KL gradient with
  respect to the read weights (``batch_sequences`` adaptation sequences). It
  selects the not yet written weights with the largest |gradient| (the round's
  share of ``fraction`` of the matrix's devices) and sets new targets
  ``q - step * g / rms(g_selected)``. The active device is reset to
  ``g_reference_now + 39.6 q_new`` with P&V. ``ideal`` writes land exactly and
  hold (diagnostic).
- ``twin`` (none, lora, learning): the no-drift twin holds the released state;
  a twin write holds its programmed endpoint.

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


class Prefix:
    """Cached frozen prefixes: the student's residual stream entering the first analog layer
    (its own digital copy for a whole-model student) and the teacher's prefix state."""

    def __init__(self, student, teacher):
        self.student, self.teacher = student, teacher

    def __len__(self):
        return len(self.student)

    def __getitem__(self, index):
        return Prefix(self.student[index], self.teacher[index])

    def split(self, size):
        return [Prefix(a, b) for a, b in zip(self.student.split(size), self.teacher.split(size))]

    @property
    def tokens(self):
        return self.student.shape[0] * self.student.shape[1]

    @property
    def sequence_length(self):
        return self.student.shape[1]


@torch.no_grad()
def prefix_states(network, tokens, device, batch=16):
    student, teacher = [], []
    for i in range(0, len(tokens), batch):
        chunk = tokens[i:i + batch].to(device)
        student.append(network.analog_input(chunk))
        features = network.features(chunk)
        teacher.append(features.teacher_hidden if isinstance(features, opt_mlp.StudentInput) else features)
    return Prefix(torch.cat(student), torch.cat(teacher))


def _digital(network):
    return network.teacher if network.digital is None else network.digital


def suffix_logits(network, hidden, q):
    """``forward_features`` from the cached analog input (same operation order)."""
    digital = _digital(network)
    for k, i in enumerate(range(network.start, len(network.teacher.layers))):
        hidden = digital.layers[i].attention(hidden)
        y = network._matrix(network.mlp_norms[k](hidden), 2 * k, q)
        hidden = hidden + network._matrix(F.relu(y), 2 * k + 1, q)
    return digital.logits(hidden[:, :-1])


def _identity(network, index, device):
    outputs, inputs = network.layout[index].shape
    return MatrixCorrection(torch.ones(inputs, device=device), torch.ones(outputs, device=device),
                            torch.zeros(outputs, device=device), torch.ones(outputs, device=device))


@torch.no_grad()
def _matrix_input(network, hidden, q, index):
    """Input of analog matrix ``index`` in ``forward_features`` order, with the active corrections."""
    for k in range(index // 2 + 1):
        hidden = _digital(network).layers[network.start + k].attention(hidden)
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
    """Per-column mean/variance-matching affine correction of every analog matrix (see module doc)."""
    device = q.device
    matrices = [_identity(network, i, device) for i in range(network.base_matrices)]
    network.drift_compensation = Compensation(matrices)
    try:
        for index in range(network.base_matrices):
            sums = None
            for chunk in probe.split(batch):
                x = _matrix_input(network, chunk.student, q, index)
                current = network._matrix(x, index, q).flatten(0, -2).double()
                reference = network._matrix(x, index, network.q.detach()).flatten(0, -2).double()
                part = torch.stack((current.sum(0), reference.sum(0), current.square().sum(0),
                                    reference.square().sum(0)))
                n = current.shape[0]
                sums = (n, part) if sums is None else (sums[0] + n, sums[1] + part)
            n, (sy, sr, syy, srr) = sums[0], sums[1]
            my, mr = sy / n, sr / n
            var = (syy / n - my.square()).clamp_min(0)
            var_ref = (srr / n - mr.square()).clamp_min(0)
            # Constant channels stay constant, without a large gain.
            gain = torch.where((var > 1e-12) & (var_ref > 1e-12), (var_ref / var.clamp_min(1e-12)).sqrt(),
                               torch.ones_like(var)).clamp(*GAIN_BOUNDS)
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
            index = torch.randint(0, len(hidden), (batch,), generator=generator).to(hidden.student.device)
            states = hidden[index]
            with torch.no_grad():
                teacher = network.teacher_logits(states.teacher).flatten(0, 1).log_softmax(-1)
            student = suffix_logits(network, states.student, q).flatten(0, 1).log_softmax(-1)
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


def _expected(cells, seconds, ids=None):
    """Nominal expectation requested + m u^alpha at each device's own age (controller knowledge)."""
    law = cells.law
    last = cells.last_write.flatten() if ids is None else cells.last_write.flatten()[ids]
    requested = cells.requested.flatten() if ids is None else cells.requested.flatten()[ids]
    u = (seconds - last) / law.reference_seconds
    return requested + (law.mean_at(requested) * u.pow(law.mean_exponent)).to(torch.float32)


def nominal_deviation(cells, ids=None):
    """Observed read minus the nominal expectation at each device's own age; counts the read."""
    read = cells.read(ids).flatten()
    return read - _expected(cells, cells.clock, ids)


def oracle_deviation(cells, seconds):
    """Hidden future deviation at ``seconds`` without moving the clock (labelled oracle arm only)."""
    return cells.value(seconds).flatten() - _expected(cells, seconds)


def rewrite(cells, arm, clock, verify_seconds, *, selection='observed', future=None, history=None):
    """P&V resets of a fraction of each matrix's devices (all of them at fraction 1).

    ``selection``: 'observed' |e| (A9), 'oracle' hidden |e| at the clock ``future`` (O2), or
    'precompensated' (A10): A9's selection with targets offset by -c * e. c is regressed, over the
    devices rewritten at the previous event, from their new-episode deviation now on their deviation
    before that rewrite, and clipped to [0, 1]; c = 0 at the first event. ``history`` carries the
    previous event's (ids, deviation) per matrix and is updated in place.
    """
    for c in cells:
        c.advance(clock)
    cost = {'reset_attempts': 0, 'verify_reads': 0, 'selection_reads': 0, 'selected_devices': 0}
    deviations = []
    for c in cells:
        if arm['fraction'] >= 1:
            deviations.append(None)
        elif selection == 'oracle':
            deviations.append(oracle_deviation(c, future))
        else:
            deviations.append(nominal_deviation(c))
            cost['selection_reads'] += c.target.numel()
    coefficient = 0.
    if selection == 'precompensated' and history:
        old = torch.cat([d for _, d in history]).double()
        new = torch.cat([dev[ids] for (ids, _), dev in zip(history, deviations)]).double()
        var = float(old.var())
        coefficient = min(max(float(((old - old.mean()) * (new - new.mean())).mean()) / var, 0.), 1.) if var > 0 else 0.
    cost['precompensation'] = coefficient
    if history is not None:
        history.clear()
    for c, error in zip(cells, deviations):
        n = c.target.numel()
        if error is None:
            ids = torch.arange(n, device=c.target.device)
        else:
            count = max(1, round(n * arm['fraction']))
            ids = torch.argsort(error.abs(), descending=True, stable=True)[:count]
        goal = c.target.flatten()[ids]
        if selection == 'precompensated':
            goal = (goal - coefficient * error[ids]).clamp(c.law.program_min, c.law.program_max)
            history.append((ids, error[ids].clone()))
        stats = program_verify(c, ids, goal, attempts=arm['attempts'],
                               threshold=arm['threshold'], verify_seconds=verify_seconds)
        cost['reset_attempts'] += stats['reset_attempts']
        cost['verify_reads'] += stats['verify_reads']
        cost['selected_devices'] += ids.numel()
    return cost


def kl_gradient(network, q, hidden, index, chunk=4):
    """Gradient of the teacher KL with respect to the held normalized weights ``q`` on ``hidden[index]``."""
    q = q.detach().requires_grad_(True)
    total = len(index)
    with torch.enable_grad():
        for part in index.split(chunk):
            states = hidden[part]
            with torch.no_grad():
                teacher = network.teacher_logits(states.teacher).flatten(0, 1).log_softmax(-1)
            student = suffix_logits(network, states.student, q).flatten(0, 1).log_softmax(-1)
            loss = (teacher.exp() * (teacher - student)).sum(-1).mean() * (len(part) / total)
            loss.backward()
    if not bool(torch.isfinite(q.grad).all()):
        raise FloatingPointError('Nonfinite analog-learning gradient.')
    return q.grad.detach()


def newton_terms(network, q, hidden, index, seed, gradient=True):
    """exp-003 terms on ``hidden[index]``: the per-token mean teacher-KL gradient g and the Monte
    Carlo diagonal Gauss-Newton (true Fisher) h of the per-token loss, sampled labels with
    Rademacher token weights, one sequence at a time. ``gradient=False`` skips g (Fisher score)."""
    grad = kl_gradient(network, q, hidden, index) if gradient else None   # mean over tokens and sequences
    tokens = hidden.student.shape[1] - 1
    q = q.detach().requires_grad_(True)
    curvature = torch.zeros_like(q)
    generator = torch.Generator(device=q.device).manual_seed(seed)
    with torch.enable_grad():
        for i in index.tolist():
            states = hidden[i:i + 1]
            out = suffix_logits(network, states.student, q).flatten(0, 1)
            with torch.no_grad():
                y = torch.multinomial(out.softmax(-1), 1, generator=generator).squeeze(1)
                r = torch.randint(0, 2, (out.shape[0],), generator=generator, device=q.device).float() * 2 - 1
            g, = torch.autograd.grad((r * F.cross_entropy(out, y, reduction='none')).sum(), q)
            curvature += g.square() / tokens
    curvature /= len(index)
    if not bool(torch.isfinite(curvature).all()):
        raise FloatingPointError('Nonfinite analog-learning curvature.')
    return grad, curvature


def fresh_pair_variance(cells, ids, tau):
    """Nominal squared weight error (normalized units) of a freshly reset active/reference pair at age tau:
    both devices' CMO endpoint variance plus drift spread. Under the scalar law the mean drift cancels
    within the pair; with coefficient bins (looked up at the targets) the squared mean mismatch is added."""
    law = cells.law
    active = cells.target[0].flatten()[ids]
    reference = cells.target[1].flatten()[ids]
    cmo = (law.cmo_scale * law.cmo_sigma(active)).square() + (law.cmo_scale * law.cmo_sigma(reference)).square()
    if not law.coefficient_bins:
        spread = law.spread_us * (tau / law.reference_seconds) ** law.spread_exponent
        return (cmo + 2 * spread ** 2) / 39.6 ** 2
    u = tau / law.reference_seconds
    drift = (law.spread_at(active).square() + law.spread_at(reference).square()) * u ** (2 * law.spread_exponent)
    bias = ((law.mean_at(active) - law.mean_at(reference)) * u ** law.mean_exponent).square()
    return (cmo + (drift + bias).to(cmo.dtype)) / 39.6 ** 2


LORA_METHODS = ('lora', 'lora_oracle', 'lora_rescaled', 'lora_clean')
REWRITE_SELECTION = {'rewrite': 'observed', 'precompensated': 'precompensated', 'rewrite_oracle': 'oracle'}
RESCALE_STRIDE = 16


class Arm:
    """One arm's controller state on the network.

    ``physical`` is the arm's view of the devices (the twin and ideal overlays apply), ``event``
    acts at an event or read age, and ``context`` prepares the network for an evaluation.
    """

    def __init__(self, arm, network, adaptation, hidden, probe, realization, verify_seconds, cells, horizon):
        self.arm, self.network, self.adaptation = arm, network, adaptation
        self.hidden, self.probe = hidden, probe
        self.realization, self.verify_seconds, self.horizon = realization, verify_seconds, horizon
        self.method = arm['method']
        self.correction = None
        self.norms = {k: v.detach().clone() for k, v in network.mlp_norms.state_dict().items()}
        self.cost = {'events': 0, 'reset_attempts': 0, 'verify_reads': 0, 'selection_reads': 0,
                     'selected_devices': 0, 'probe_tokens': 0, 'optimizer_steps': 0,
                     'adaptation_sequence_presentations': 0, 'adaptation_tokens': 0,
                     'gradient_sequences': 0, 'digital_parameters': 0, 'event_wall_seconds': 0.}
        self.events = []
        self.twin = [c.value().clone() for c in cells] if arm.get('twin') else None
        self.ideal = [None] * len(cells) if arm.get('ideal') else None
        self.history = []
        self.rescale = None
        self.base_multiplier = None
        if self.method in LORA_METHODS:
            self._fresh_lora()
            self.cost['digital_parameters'] = sum(p.numel() for p in network.lora.parameters())
        elif self.method == 'layernorm':
            self.cost['digital_parameters'] = sum(p.numel() for p in network.mlp_norms.parameters())
        elif self.method == 'calibration':
            self.cost['digital_parameters'] = sum(2 * network.layout[i].shape[0] for i in range(network.base_matrices))

    @property
    def schedule(self):
        """Every action age: training/fit/write events plus A6's rescaling reads."""
        return tuple(sorted(set(self.arm.get('events', ())) | set(self.arm.get('reads', ()))))

    @property
    def evaluate_after_event(self):
        return self.method == 'lora_oracle'

    def _fresh_lora(self):
        rank = self.arm['rank']
        self.network.enable_lora(SimpleNamespace(rank=rank, alpha=float(rank)), seed=self.realization)
        self.base_multiplier = self.network.lora_multiplier

    def physical(self, cells, clock=None):
        """Device values as this arm's array holds them (at ``clock``, or at each cell's own clock)."""
        live = [c.advance(clock) if clock is not None else c.value() for c in cells]
        if self.twin is not None:
            return [h.clone() for h in self.twin]
        if self.ideal is not None:
            for k, overlay in enumerate(self.ideal):
                if overlay is not None:
                    mask, value = overlay
                    live[k] = torch.where(mask, value, live[k])
        return live

    def close(self):
        self.network.enable_lora(None)
        self.network.mlp_norms.load_state_dict(self.norms)
        self.network.drift_compensation = None
        for p in self.network.parameters():
            p.requires_grad_(False)

    def context(self, age):
        self.network.drift_compensation = self.correction
        if self.method == 'lora_rescaled':
            factor = 1.
            if self.rescale is not None and self.rescale.get('beta') is not None:
                t1, beta, kappa = self.rescale['t1'], self.rescale['beta'], self.rescale['kappa']
                factor = (t1 / age) ** kappa * (age / t1) ** beta if age > t1 else 1.
            self.network.lora_multiplier = self.base_multiplier * factor

    def _count_read(self, cells):
        reads = sum(c.target.numel() for c in cells)
        for c in cells:
            c.selection_reads += c.target.numel()
        self.cost['selection_reads'] += reads
        return reads

    def _train(self, q, index):
        parameters = (list(self.network.lora.parameters()) if self.method in LORA_METHODS
                      else list(self.network.mlp_norms.parameters()))
        a = self.adaptation
        self.network.lora_multiplier = self.base_multiplier if self.base_multiplier is not None else 0.
        trace = train_digital(self.network, parameters, q, self.hidden, steps=a['steps'], batch=a['batch'],
                              learning_rate=self.arm['learning_rate'], seed=a['seed'] + 1000 * index)
        presentations = a['steps'] * a['batch']
        self.cost['optimizer_steps'] += a['steps']
        self.cost['adaptation_sequence_presentations'] += presentations
        self.cost['adaptation_tokens'] += presentations * self.hidden.sequence_length
        return {**trace, 'optimizer_steps': a['steps']}

    def _rescale_read(self, cells, age):
        """A6: deviations of a fixed device subsample; estimates against the training-time read."""
        parts = []
        for c in cells:
            ids = torch.arange(0, c.target.numel(), RESCALE_STRIDE, device=c.target.device)
            parts.append(nominal_deviation(c, ids).double())
        d = torch.cat(parts)
        self.cost['selection_reads'] += d.numel()
        if self.rescale is None:
            self.rescale = {'t1': age, 'd1': d, 'beta': None, 'kappa': None}
            return {'rescale_reads': d.numel()}
        d1, t1 = self.rescale['d1'], self.rescale['t1']
        ratio = math.log(age / t1)
        beta = math.log(float(d.std()) / float(d1.std())) / ratio
        corr = float(torch.corrcoef(torch.stack((d1, d)))[0, 1])
        kappa = -math.log(min(max(corr, 1e-3), 1.)) / ratio
        self.rescale.update(beta=beta, kappa=kappa)
        return {'rescale_reads': d.numel(), 'beta_hat': beta, 'kappa_hat': kappa, 'correlation': corr}

    def _learn(self, cells, clock, bind, index):
        arm, network = self.arm, self.network
        for c in cells:
            c.advance(clock)
        written = [torch.zeros(c.target[0].numel(), dtype=torch.bool, device=c.target.device) for c in cells]
        record = {'rounds': arm['rounds'], 'round_clocks': []}
        generator = torch.Generator().manual_seed(self.adaptation['seed'] + 7919 * (index + 1))
        for r in range(arm['rounds']):
            physical = self.physical(cells)
            q = bind(physical)
            pick = torch.randperm(len(self.hidden), generator=generator)[:arm['batch_sequences']]
            grad = kl_gradient(network, q, self.hidden, pick.to(self.hidden.student.device))
            self.cost['gradient_sequences'] += len(pick)
            self._count_read(cells)
            for k, (c, spec) in enumerate(zip(cells, network.layout)):
                g = grad[spec.offset:spec.offset + spec.size]
                count = max(1, round(c.target.numel() * arm['fraction'] / arm['rounds']))
                score = g.abs().masked_fill(written[k], -1.)
                ids = torch.argsort(score, descending=True, stable=True)[:count]
                written[k][ids] = True
                step = g[ids] / g[ids].square().mean().sqrt().clamp_min(1e-30)
                q_new = (q[spec.offset + ids] - arm['step'] * step).clamp(-1., 1.)
                reference = physical[k][1].flatten()[ids]
                goal = (reference + 39.6 * q_new).clamp(c.law.program_min, c.law.program_max)
                if self.ideal is not None:
                    overlay = self.ideal[k]
                    mask = torch.zeros_like(c.target, dtype=torch.bool) if overlay is None else overlay[0]
                    value = torch.zeros_like(c.target) if overlay is None else overlay[1]
                    mask[0].view(-1)[ids] = True
                    value[0].view(-1)[ids] = goal
                    self.ideal[k] = (mask, value)
                    self.cost['reset_attempts'] += ids.numel()
                elif self.twin is not None:
                    c.write(ids, goal)
                    self.twin[k][0].view(-1)[ids] = c.anchor[0].reshape(-1)[ids]
                    self.cost['reset_attempts'] += ids.numel()
                else:
                    stats = program_verify(c, ids, goal, attempts=arm['attempts'], threshold=arm['threshold'],
                                           verify_seconds=self.verify_seconds)
                    self.cost['reset_attempts'] += stats['reset_attempts']
                    self.cost['verify_reads'] += stats['verify_reads']
                self.cost['selected_devices'] += ids.numel()
            record['round_clocks'].append(max(c.clock for c in cells))
        return record

    def _learn_newton(self, cells, clock, bind, index, age):
        """A11, the exp-003 closed loop: Newton net-gain selection of pairs, reset to their clean targets."""
        arm, network = self.arm, self.network
        for c in cells:
            c.advance(clock)
        events = arm['events']
        later = [t for t in events if t > age]
        tau = 0.5 * ((later[0] if later else self.horizon) - age)
        clean = network.q.detach()
        written = [torch.zeros(c.target[0].numel(), dtype=torch.bool, device=c.target.device) for c in cells]
        record = {'rounds': arm['rounds'], 'round_clocks': [], 'positive_gain_pairs': [], 'tau_seconds': tau}
        generator = torch.Generator().manual_seed(self.adaptation['seed'] + 7919 * (index + 1))
        for r in range(arm['rounds']):
            physical = self.physical(cells)
            q = bind(physical)
            pick = torch.randperm(len(self.hidden), generator=generator)[:arm['batch_sequences']]
            fisher = arm.get('score', 'newton') == 'fisher'
            g, h = newton_terms(network, q, self.hidden, pick.to(self.hidden.student.device),
                                seed=self.adaptation['seed'] + 104729 * (index + 1) + r, gradient=not fisher)
            self.cost['gradient_sequences'] += len(pick) if fisher else 2 * len(pick)
            self._count_read(cells)
            positive = 0
            for k, (c, spec) in enumerate(zip(cells, network.layout)):
                sl = slice(spec.offset, spec.offset + spec.size)
                e = q[sl] - clean[sl]
                pairs = torch.arange(spec.size, device=c.target.device)
                v = fresh_pair_variance(c, pairs, tau)
                if fisher:      # exp-003 fisher_net: unsigned curvature-weighted net error, no overshoot
                    gain = h[sl] * (e.square() - v)
                else:           # exp-003 newton_net: signed first-order gain (overshoots at large budgets)
                    gain = g[sl] * e - 0.5 * h[sl] * (e.square() + v)
                gain = gain.masked_fill(written[k], -float('inf'))
                count = max(1, round(c.target.numel() * arm['fraction'] / 2 / arm['rounds']))
                order = torch.argsort(gain, descending=True, stable=True)[:count]
                ids = order[gain[order] > 0]
                positive += int((gain > 0).sum())
                if not ids.numel():
                    continue
                written[k][ids] = True
                devices = torch.cat((ids, ids + spec.size))           # active and reference of each pair
                goal = c.target.flatten()[devices]
                if self.ideal is not None:
                    overlay = self.ideal[k]
                    mask = torch.zeros_like(c.target, dtype=torch.bool) if overlay is None else overlay[0]
                    value = torch.zeros_like(c.target) if overlay is None else overlay[1]
                    mask.view(-1)[devices] = True
                    value.view(-1)[devices] = goal
                    self.ideal[k] = (mask, value)
                    self.cost['reset_attempts'] += devices.numel()
                elif self.twin is not None:
                    c.write(devices, goal)
                    self.twin[k].view(-1)[devices] = c.anchor.reshape(-1)[devices]
                    self.cost['reset_attempts'] += devices.numel()
                else:
                    stats = program_verify(c, devices, goal, attempts=arm['attempts'], threshold=arm['threshold'],
                                           verify_seconds=self.verify_seconds)
                    self.cost['reset_attempts'] += stats['reset_attempts']
                    self.cost['verify_reads'] += stats['verify_reads']
                self.cost['selected_devices'] += devices.numel()
            record['round_clocks'].append(max(c.clock for c in cells))
            record['positive_gain_pairs'].append(positive)
        return record

    def event(self, index, age, clock, cells, bind):
        started = time.monotonic()
        record = {'event': index, 'seconds': age, 'clock_seconds': clock}
        self.network.drift_compensation = None
        if self.method == 'learning_newton':
            record.update(self._learn_newton(cells, clock, bind, index, age))
            record['end_clock_seconds'] = max(c.clock for c in cells)
        elif self.method in REWRITE_SELECTION:
            events = self.arm['events']
            later = [t for t in events if t > age]
            future = math.sqrt(max(age, 1.) * (later[0] if later else self.horizon))
            cost = rewrite(cells, self.arm, clock, self.verify_seconds, selection=REWRITE_SELECTION[self.method],
                           future=clock - age + future, history=self.history)
            record.update(cost, end_clock_seconds=max(c.clock for c in cells))
            for k, v in cost.items():
                if k in self.cost:
                    self.cost[k] += v
        elif self.method == 'learning':
            record.update(self._learn(cells, clock, bind, index))
            record['end_clock_seconds'] = max(c.clock for c in cells)
        elif self.method == 'lora_rescaled' and age in self.arm['reads']:
            for c in cells:
                c.advance(clock)
            record.update(self._rescale_read(cells, age))
        else:
            q = bind(self.physical(cells, clock))
            record['selection_reads'] = self._count_read(cells)
            if self.method == 'calibration':
                self.correction = fit_calibration(self.network, q, self.probe)
                self.cost['probe_tokens'] += self.probe.tokens
                record['probe_tokens'] = self.probe.tokens
            else:
                if self.method == 'lora_oracle':
                    self._fresh_lora()
                if self.method == 'lora_clean':
                    q = self.network.q.detach()
                record.update(self._train(q, index))
                if self.method == 'lora_rescaled':
                    record.update(self._rescale_read(cells, age))
        self.cost['events'] += 1
        record['wall_seconds'] = time.monotonic() - started
        self.cost['event_wall_seconds'] += record['wall_seconds']
        self.events.append(record)
        return record


def timeline(ages, events, after=False):
    """(age, evaluate, event index or None). An evaluation at an event age comes first, unless
    ``after`` (the O1 oracle, evaluated right after its training)."""
    points = []
    for t in sorted(set(ages) | set(events)):
        steps = []
        if t in ages:
            steps.append((t, True, None))
        if t in events:
            steps.append((t, False, events.index(t)))
        points.extend(reversed(steps) if after else steps)
    return points


def finite(metrics):
    return all(math.isfinite(metrics[k]) for k in ('cross_entropy', 'perplexity', 'teacher_kl'))
