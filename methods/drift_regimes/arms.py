"""Maintenance arms of the exp-003 drift-regimes screen on the frozen OPT CMO DRN.

Port of the crossbar exp-002 controllers (hwa ``workflow/drift_regimes_arms.py`` at
3edcaf6) restricted to the retained methods. Every arm starts from a copy of the
same released deployment, so untouched devices see identical future drift across
arms. Controllers see exact reads, their own write times, the original targets and
the nominal law; never D, C, Y or hidden draws.

- ``none``: untreated (A0); with ``twin`` the released state is held (A0-twin).
- ``calibration`` (A1): per-output-channel affine at the six DRN MLP outputs, refitted at
  each event by mean/variance matching against the clean MLP on the same current input.
- ``layernorm`` (A3): the affine of the six LayerNorms in front of the DRN MLPs, trained
  on the held read at each event (warm start; fresh Adam and cosine schedule per event).
- ``rewrite`` (A8 at fraction 1, A9 otherwise): P&V resets to original targets of the
  devices with the largest |read - (requested + m u^alpha)| per matrix.
- ``precompensated`` (A10): A9's selection with targets offset by ``-c e``, c regressed
  from the previous event's rewritten devices and clipped to [0, 1].
- ``learning_newton`` with ``score: fisher`` (A11): four-device groups ranked by the
  physical-coordinate Fisher net error ``e^T F e - tr(F M_fresh)``; selected groups have all
  four devices reset to original targets with P&V. With ``twin`` (attempts 1) writes are
  single unverified resets whose fresh endpoint is then held without drift.

Digital fits take no device time; rewrite P&V moves the matrix clock by its verify delays.
"""
import math
import time

import torch

from hardware.drift.family import program_verify
from methods.drift_regimes import forward

REWRITE_SELECTION = {'rewrite': 'observed', 'precompensated': 'precompensated'}


def _expected(cells, seconds, ids=None):
    """Nominal expectation requested + m u^alpha at each device's own age (controller knowledge)."""
    law = cells.law
    last = cells.last_write.flatten() if ids is None else cells.last_write.flatten()[ids]
    requested = cells.requested.flatten() if ids is None else cells.requested.flatten()[ids]
    u = (seconds - last) / law.reference_seconds
    return requested + (law.mean_us * u.pow(law.mean_exponent)).to(torch.float32)


def nominal_deviation(cells, ids=None):
    """Observed read minus the nominal expectation at each device's own age; counts the read."""
    read = cells.read(ids).flatten()
    return read - _expected(cells, cells.clock, ids)


def rewrite(cells, arm, clock, verify_seconds, *, selection='observed', history=None):
    """P&V resets of a fraction of each matrix's devices (all at fraction 1); see crossbar ``rewrite``."""
    for c in cells:
        c.advance(clock)
    cost = {'reset_attempts': 0, 'verify_reads': 0, 'selection_reads': 0, 'selected_devices': 0}
    deviations = []
    for c in cells:
        if arm['fraction'] >= 1:
            deviations.append(None)
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


class Arm:
    """One arm's controller state on the frozen DRN.

    ``physical`` is the arm's view of the devices (the twin overlay applies), ``event`` acts at
    a maintenance age, and ``context`` returns the output correction for an evaluation.
    """

    def __init__(self, arm, model, teacher, settings, hidden, probe, policy, cells, horizon):
        self.arm, self.model, self.teacher, self.settings = arm, model, teacher, settings
        self.hidden, self.probe, self.policy, self.horizon = hidden, probe, policy, horizon
        self.adaptation = settings.adaptation
        self.method = arm['method']
        self.correction = None
        self.norms = {id(norm): {k: v.detach().clone() for k, v in norm.state_dict().items()}
                      for norm in forward.layer_norms(model)}
        self.cost = {'events': 0, 'reset_attempts': 0, 'verify_reads': 0, 'selection_reads': 0,
                     'selected_devices': 0, 'probe_tokens': 0, 'optimizer_steps': 0,
                     'adaptation_sequence_presentations': 0, 'adaptation_tokens': 0,
                     'gradient_sequences': 0, 'fisher_probe_sequences': 0, 'digital_parameters': 0,
                     'event_wall_seconds': 0.}
        self.events = []
        self.twin = [c.value().clone() for c in cells] if arm.get('twin') else None
        self.history = []
        if self.method == 'layernorm':
            self.cost['digital_parameters'] = sum(p.numel() for n in forward.layer_norms(model) for p in n.parameters())
        elif self.method == 'calibration':
            self.cost['digital_parameters'] = sum(2 * b.bias2.numel() for b in model.blocks)

    @property
    def schedule(self):
        return tuple(self.arm.get('events', ()))

    def physical(self, cells, clock=None):
        """Device values as this arm's array holds them (at ``clock``, or at each cell's own clock)."""
        live = [c.advance(clock) if clock is not None else c.value() for c in cells]
        if self.twin is not None:
            return [h.clone() for h in self.twin]
        return live

    def close(self):
        for norm in forward.layer_norms(self.model):
            norm.load_state_dict(self.norms[id(norm)])
        self.correction = None
        for p in self.model.parameters():
            p.requires_grad_(False)

    def context(self):
        return self.correction

    def _count_read(self, cells):
        reads = sum(c.target.numel() for c in cells)
        for c in cells:
            c.selection_reads += c.target.numel()
        self.cost['selection_reads'] += reads
        return reads

    def _fisher_rewrite(self, cells, clock, index, age):
        """A11: per round, read, score groups by Fisher net error and reset the best to targets."""
        arm = self.arm
        for c in cells:
            c.advance(clock)
        events = arm['events']
        later = [t for t in events if t > age]
        tau = 0.5 * ((later[0] if later else self.horizon) - age)
        law = cells[0].law
        targets = [c.target for c in cells]
        moments = [forward.fresh_moments(law, t, tau) for t in targets]
        variances, mean = [m[0] for m in moments], moments[0][1]
        written = [torch.zeros(c.target.numel() // 4, dtype=torch.bool, device=c.target.device) for c in cells]
        record = {'rounds': arm['rounds'], 'round_clocks': [], 'positive_gain_groups': [], 'tau_seconds': tau,
                  'fresh_mean_us': mean}
        generator = torch.Generator().manual_seed(self.adaptation.seed + 7919 * (index + 1))
        for r in range(arm['rounds']):
            physical = self.physical(cells)
            pick = torch.randperm(len(self.hidden), generator=generator)[:arm['batch_sequences']]
            scores = forward.fisher_scores(self.model, self.hidden, pick.to(self.hidden.student.device),
                                           physical, targets, variances, mean, self.policy,
                                           seed=self.adaptation.seed + 104729 * (index + 1) + r)
            self.cost['gradient_sequences'] += len(pick)
            self.cost['fisher_probe_sequences'] += len(pick)
            self._count_read(cells)
            positive = 0
            for k, c in enumerate(cells):
                gain = scores[k].flatten().masked_fill(written[k], -float('inf'))
                count = max(1, round(c.target.numel() * arm['fraction'] / 4 / arm['rounds']))
                order = torch.argsort(gain, descending=True, stable=True)[:count]
                groups = order[gain[order] > 0]
                positive += int((gain > 0).sum())
                if not groups.numel():
                    continue
                written[k][groups] = True
                devices = forward.group_members(c.target.shape, k % 2 == 1, groups, c.target.device)
                goal = c.target.flatten()[devices]
                if self.twin is not None:
                    c.write(devices, goal)
                    self.twin[k].view(-1)[devices] = c.anchor.reshape(-1)[devices]
                    self.cost['reset_attempts'] += devices.numel()
                else:
                    stats = program_verify(c, devices, goal, attempts=arm['attempts'], threshold=arm['threshold'],
                                           verify_seconds=self.settings.verify_seconds)
                    self.cost['reset_attempts'] += stats['reset_attempts']
                    self.cost['verify_reads'] += stats['verify_reads']
                self.cost['selected_devices'] += devices.numel()
            record['round_clocks'].append(max(c.clock for c in cells))
            record['positive_gain_groups'].append(positive)
        return record

    def event(self, index, age, clock, cells):
        started = time.monotonic()
        record = {'event': index, 'seconds': age, 'clock_seconds': clock}
        if self.method == 'learning_newton':
            record.update(self._fisher_rewrite(cells, clock, index, age))
            record['end_clock_seconds'] = max(c.clock for c in cells)
        elif self.method in REWRITE_SELECTION:
            cost = rewrite(cells, self.arm, clock, self.settings.verify_seconds,
                           selection=REWRITE_SELECTION[self.method], history=self.history)
            record.update(cost, end_clock_seconds=max(c.clock for c in cells))
            for k, v in cost.items():
                if k in self.cost:
                    self.cost[k] += v
        else:
            view = forward.bind(self.physical(cells, clock))
            forward.check_view(self.model, view)
            record['selection_reads'] = self._count_read(cells)
            if self.method == 'calibration':
                self.correction = forward.fit_calibration(self.model, view, self.probe, self.policy)
                self.cost['probe_tokens'] += self.probe.token_count
                record['probe_tokens'] = self.probe.token_count
                gains = torch.cat([a for a, _ in self.correction])
                record['calibration_gain'] = {'min': float(gains.min()), 'max': float(gains.max()),
                                              'mean': float(gains.mean()),
                                              'at_bounds': int(((gains <= forward.GAIN_BOUNDS[0])
                                                                | (gains >= forward.GAIN_BOUNDS[1])).sum())}
            elif self.method == 'layernorm':
                a = self.adaptation
                record.update(forward.train_layernorm(
                    self.model, self.teacher, self.hidden, view, self.policy, steps=a.steps, batch=a.batch,
                    microbatch=a.microbatch, learning_rate=self.arm['learning_rate'], seed=a.seed + 1000 * index))
                presentations = a.steps * a.batch
                self.cost['optimizer_steps'] += a.steps
                self.cost['adaptation_sequence_presentations'] += presentations
                self.cost['adaptation_tokens'] += presentations * self.hidden.sequence_length
            else:
                raise ValueError(f'Method {self.method!r} has no events.')
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
