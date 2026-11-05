"""Held-state forwards of the frozen OPT DRN for the exp-003 drift-regimes screen.

Every device-facing forward consumes the held apparent conductances passed in by the
caller (physical microSiemens tensors in the ``u`` orientation of each DRN matrix).
The frozen prefix (decoder blocks before the first DRN MLP) is cached once per cohort
for both the student and the FP32 teacher; the trainable suffix is never cached.

Forward policy: no-graph forwards (evaluation, calibration fits, selection) run
``evaluation_sweeps`` settling sweeps; gradient forwards settle without a graph for
``settle_sweeps`` and differentiate through ``gradient_sweeps`` more. Both therefore
run the same number of sweeps.
"""
from dataclasses import replace
import math

import torch
from torch.nn import functional as F

from hardware.devices.cmo import G_MAX_US
from networks.resistive.differentiable import coefficients


class Prefix:
    """Cached frozen prefixes: the student's residual stream entering the first DRN block and
    the FP32 teacher's state at the same depth."""

    def __init__(self, student, teacher, tokens):
        self.student, self.teacher, self.tokens = student, teacher, tokens

    def __len__(self):
        return len(self.student)

    def __getitem__(self, index):
        return Prefix(self.student[index], self.teacher[index], self.tokens[index])

    def split(self, size):
        return [Prefix(a, b, c) for a, b, c in
                zip(self.student.split(size), self.teacher.split(size), self.tokens.split(size))]

    @property
    def token_count(self):
        return self.student.shape[0] * self.student.shape[1]

    @property
    def sequence_length(self):
        return self.student.shape[1]


@torch.no_grad()
def teacher_prefix(teacher, tokens, start):
    hidden = teacher.embed(tokens)
    for layer in teacher.layers[:start]:
        hidden = layer(hidden)
    return hidden


@torch.no_grad()
def prefix_states(model, teacher, tokens, device, batch=16):
    student, reference = [], []
    for i in range(0, len(tokens), batch):
        chunk = tokens[i:i + batch].to(device)
        student.append(model.analog_input(chunk))
        reference.append(teacher_prefix(teacher, chunk, model.start))
    return Prefix(torch.cat(student), torch.cat(reference), tokens.to(device))


def teacher_logits(teacher, start, hidden):
    for layer in teacher.layers[start:]:
        hidden = layer(hidden)
    return teacher.logits(hidden[:, :-1])


def suffix_logits(model, hidden, view, correction=None):
    """``OptDrn.forward`` from the cached analog input, in the same operation order."""
    for k, i in enumerate(range(model.start, len(model.digital.layers))):
        layer = model.digital.layers[i]
        hidden = layer.attention(hidden)
        y = model.blocks[k](layer.final_layer_norm(hidden), None if view is None else view[k])
        if correction is not None:
            a, b = correction[k]
            y = y * a + b
        hidden = hidden + y
    return model.digital.logits(hidden[:, :-1])


def layer_norms(model):
    return [model.digital.layers[i].final_layer_norm for i in range(model.start, len(model.digital.layers))]


class Policy:
    """Applies the versioned sweep policy to every DRN block's solver."""

    def __init__(self, model, policy):
        self.model, self.policy = model, policy
        base = model.blocks[0].solver.settings
        self.training_settings = replace(base, settling_iterations=policy.settle_sweeps,
                                         gradient_iterations=policy.gradient_sweeps)
        self.evaluation_settings = replace(base, settling_iterations=policy.evaluation_sweeps,
                                           gradient_iterations=policy.gradient_sweeps)

    def mode(self, training, sweeps=None):
        settings = self.training_settings if training else self.evaluation_settings
        if sweeps is not None:
            settings = replace(settings, settling_iterations=sweeps)
        for block in self.model.blocks:
            block.solver.settings = settings
            block.solver.train(training)
        self.model.train(training)


def bind(physical):
    """Held physical microSiemens tensors -> per-block normalized (g1, g2) view, checked."""
    values = [p / G_MAX_US for p in physical]
    return tuple(tuple(values[i:i + 2]) for i in range(0, len(values), 2))


@torch.no_grad()
def check_view(model, view):
    for block, (g1, g2) in zip(model.blocks, view):
        if g1.shape != block.u1.shape or g2.shape != block.u2.shape:
            raise ValueError('Physical array shape differs from the DRN model.')
        if not all(bool(torch.isfinite(g).all() and (g >= 0).all()) for g in (g1, g2)):
            raise FloatingPointError('Invalid held physical DRN view.')
        dh, do = coefficients(g1, g2, block.solver.settings)
        if not bool(torch.isfinite(dh).all() and torch.isfinite(do).all() and (dh > 0).all() and (do > 0).all()):
            raise FloatingPointError('Nonpositive or nonfinite physical DRN denominator.')


@torch.no_grad()
def evaluate(model, teacher, prefix, view, policy, *, correction=None, microbatch=4, sweeps=None):
    """Mean next-token teacher KL and language metrics on a cached cohort (no graph)."""
    policy.mode(False, sweeps)
    kl = ce = 0.
    correct = agree = count = 0
    for part in prefix.split(microbatch):
        out = suffix_logits(model, part.student, view, correction).flatten(0, 1)
        if not bool(torch.isfinite(out).all()):
            raise FloatingPointError('Nonfinite DRN language-model logits.')
        reference = teacher_logits(teacher, model.start, part.teacher).flatten(0, 1)
        target = part.tokens[:, 1:].flatten()
        kl += float(F.kl_div(out.log_softmax(-1), reference.log_softmax(-1), log_target=True, reduction='sum'))
        ce += float(F.cross_entropy(out, target, reduction='sum'))
        prediction = out.argmax(-1)
        correct += int((prediction == target).sum())
        agree += int((prediction == reference.argmax(-1)).sum())
        count += out.shape[0]
    return {'teacher_kl': kl / count, 'cross_entropy': ce / count, 'perplexity': math.exp(ce / count),
            'next_token_accuracy_percent': 100 * correct / count,
            'teacher_agreement_percent': 100 * agree / count, 'scored_tokens': count}


@torch.no_grad()
def solver_diagnostics(model, prefix, view, policy, sequences=1):
    """Voltage defect after the evaluation sweeps, denominators and voltages per block."""
    policy.mode(False)
    hidden = prefix.student[:sequences]
    rows = []
    for k, i in enumerate(range(model.start, len(model.digital.layers))):
        layer, block = model.digital.layers[i], model.blocks[k]
        hidden = layer.attention(hidden)
        x = layer.final_layer_norm(hidden)
        held = None if view is None else view[k]
        current = torch.cat((block.bias1, -block.bias1)) * block.current_scale
        rows.append({'block': k, **block.solver.diagnostics(x, *block.conductances(held), current, block.gains[0])})
        hidden = hidden + block(x, held)
    return rows


# -- A1: mean/variance matching at the six DRN MLP outputs ------------------------------
GAIN_BOUNDS = (0.1, 10.0)


@torch.no_grad()
def fit_calibration(model, view, probe, policy, batch=8):
    """Per-output-channel ``y' = g y + s`` after the fc2 bias of each DRN MLP, in forward order.

    The reference is the clean (target-conductance) MLP on the same current input; each block
    is fitted after the earlier blocks' corrections are applied. Constant channels keep gain 1.
    """
    policy.mode(False)
    correction = []
    hidden = [part.student for part in probe.split(batch)]
    for k, i in enumerate(range(model.start, len(model.digital.layers))):
        layer, block = model.digital.layers[i], model.blocks[k]
        sums, outputs = None, []
        for j, h in enumerate(hidden):
            h = layer.attention(h)
            hidden[j] = h
            x = layer.final_layer_norm(h)
            current = block(x, view[k])
            outputs.append(current)
            current, reference = current.flatten(0, -2).double(), block(x, None).flatten(0, -2).double()
            part = torch.stack((current.sum(0), reference.sum(0), current.square().sum(0), reference.square().sum(0)))
            n = current.shape[0]
            sums = (n, part) if sums is None else (sums[0] + n, sums[1] + part)
        n, (sy, sr, syy, srr) = sums
        my, mr = sy / n, sr / n
        var = (syy / n - my.square()).clamp_min(0)
        var_ref = (srr / n - mr.square()).clamp_min(0)
        gain = torch.where((var > 1e-12) & (var_ref > 1e-12), (var_ref / var.clamp_min(1e-12)).sqrt(),
                           torch.ones_like(var)).clamp(*GAIN_BOUNDS)
        a, b = gain.float(), (mr - gain * my).float()
        correction.append((a, b))
        hidden = [h + y * a + b for h, y in zip(hidden, outputs)]
    return correction


# -- A3: LayerNorm affine adaptation ----------------------------------------------------
def train_layernorm(model, teacher, hidden, view, policy, *, steps, batch, microbatch, learning_rate, seed):
    """Adam with cosine decay to zero on the teacher KL of the held view; fresh optimizer per call."""
    parameters = [p for norm in layer_norms(model) for p in norm.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in parameters:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    generator = torch.Generator().manual_seed(seed)
    view = tuple(tuple(g.detach() for g in pair) for pair in view)
    losses = []
    policy.mode(True)
    try:
        with torch.enable_grad():
            for _ in range(steps):
                index = torch.randint(0, len(hidden), (batch,), generator=generator).to(hidden.student.device)
                optimizer.zero_grad(set_to_none=True)
                total = 0.
                for part in index.split(microbatch):
                    states = hidden[part]
                    with torch.no_grad():
                        reference = teacher_logits(teacher, model.start, states.teacher).flatten(0, 1).log_softmax(-1)
                    student = suffix_logits(model, states.student, view).flatten(0, 1).log_softmax(-1)
                    loss = (reference.exp() * (reference - student)).sum(-1).mean() * (len(part) / batch)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite LayerNorm adaptation loss.')
                    loss.backward()
                    total += float(loss)
                if not all(bool(torch.isfinite(p.grad).all()) for p in parameters):
                    raise FloatingPointError('Nonfinite LayerNorm adaptation gradient.')
                optimizer.step()
                schedule.step()
                losses.append(total)
    finally:
        policy.mode(False)
        for p in parameters:
            p.requires_grad_(False)
    return {'first_loss': sum(losses[:10]) / min(10, len(losses)),
            'last_loss': sum(losses[-10:]) / min(10, len(losses)), 'optimizer_steps': steps}


# -- A11: physical-coordinate Fisher over four-device groups ------------------------------
def group_view(tensor, paired):
    """(2I, 2O) device tensor in ``u`` orientation -> (4, I, O) members of each logical weight.

    fc1 (unpaired): devices (i, o), (i, O+o), (I+i, o), (I+i, O+o).
    fc2 (paired, interleaved output rails): (i, 2o), (i, 2o+1), (I+i, 2o), (I+i, 2o+1).
    """
    rows, cols = tensor.shape
    inputs, outputs = rows // 2, cols // 2
    if paired:
        return tensor.reshape(2, inputs, outputs, 2).permute(0, 3, 1, 2).reshape(4, inputs, outputs)
    return tensor.reshape(2, inputs, 2, outputs).permute(0, 2, 1, 3).reshape(4, inputs, outputs)


def group_members(shape, paired, groups, device):
    """Flat device indices (4 per group) of flat group indices ``groups``."""
    ids = torch.arange(shape[0] * shape[1], device=device).reshape(shape)
    return group_view(ids, paired).reshape(4, -1)[:, groups].T.reshape(-1)


def fresh_moments(law, target, tau):
    """Nominal fresh-write error moments at age tau: per-device variance and common mean (uS)."""
    u = tau / law.reference_seconds
    variance = (law.cmo_scale * law.cmo_sigma(target)).square() + (law.spread_us * u ** law.spread_exponent) ** 2
    return variance, law.mean_us * u ** law.mean_exponent


def accumulate_group_fisher(gradient, error, variance, mean, paired, quadratic, trace):
    """Adds one probe's (g.e)^2 and g^T M g, M = diag(variance) + mean^2 11^T, per group."""
    g = group_view(gradient, paired)
    quadratic += (g * group_view(error, paired)).sum(0).double().square()
    trace += ((group_view(variance, paired) * g.square()).sum(0) + mean ** 2 * g.sum(0).square()).double()


def fisher_scores(model, hidden, index, physical, targets, variances, mean, policy, seed):
    """S_q = e^T F e - tr(F M_fresh) for every four-device group, F the true Fisher of the
    per-token loss in physical coordinates (sampled labels, Rademacher token weights)."""
    held = [p.detach().clone().requires_grad_(True) for p in physical]
    errors = [p.detach() - t for p, t in zip(physical, targets)]
    paired = [j % 2 == 1 for j in range(len(physical))]
    quadratic = [torch.zeros(p.shape[0] // 2, p.shape[1] // 2, dtype=torch.float64, device=p.device) for p in physical]
    trace = [torch.zeros_like(q) for q in quadratic]
    tokens = hidden.sequence_length - 1
    generator = torch.Generator(device=physical[0].device).manual_seed(seed)
    policy.mode(True)
    try:
        with torch.enable_grad():
            for i in index.tolist():
                states = hidden[i:i + 1]
                out = suffix_logits(model, states.student, bind(held)).flatten(0, 1)
                with torch.no_grad():
                    y = torch.multinomial(out.softmax(-1), 1, generator=generator).squeeze(1)
                    r = torch.randint(0, 2, (out.shape[0],), generator=generator, device=out.device).float() * 2 - 1
                grads = torch.autograd.grad((r * F.cross_entropy(out, y, reduction='none')).sum(), held)
                for m, g in enumerate(grads):
                    if not bool(torch.isfinite(g).all()):
                        raise FloatingPointError('Nonfinite physical Fisher gradient.')
                    accumulate_group_fisher(g, errors[m], variances[m], mean, paired[m], quadratic[m], trace[m])
    finally:
        policy.mode(False)
    scale = 1. / (tokens * len(index))
    return [(q - t) * scale for q, t in zip(quadratic, trace)]
