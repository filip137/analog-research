"""AIHWKit CMO log-time drift on the held fault-tuning crossbar, and its strict settings.

The law, draw keying and settings are the same as the DRN backend's methods.drift_tuning
(drn/20261116-drift-token-budget); only the device container differs.

Drift is a read transform of the held programmed conductance (uS), never a mutation:

    G(t) = clip(G_p + M*m*L + M*(a*L + b)*Y_t, drift_min, drift_max),   L = max(ln tau, 0)

with tau the seconds since the device's clock started and no drift at tau <= 0. ``Y_t`` is
a fresh standard normal for every read age, keyed on (draw_seed, array index, absolute read
time t in ms), so reads at the same age see the same draw in every arm and the draw does not
depend on evaluation order or on earlier writes.

Write clocks:

- ``continues`` (user, 16 November 2026): every clock starts at release (tau = t) and a
  recovery write shifts the held programmed value by its step (with the declared write noise)
  without restarting it. Kept bit for bit for the drift token-budget records.
- ``reset_absorb`` (user, 18 November 2026; on-chip-learning cmo-drift-recovery exp-001): a
  write at time t sets the device's held value to the drifted conductance it was read at
  (the same Y_t the learner read) plus the step plus CMO write noise sigma(G) xi, and restarts
  its clock: tau = t - t_w per device. Devices never written keep t_w = 0 and read bit for
  bit as the untreated hold (the scalar-clock expression is evaluated for them).

Mappings (``mapping``; omitted means the crossbar's standard active/reference pair):

- ``reference``: G0 = 48.6 + 39.6 q, G1 = 48.6; q_read = (G0 - G1) / 39.6.
- ``two_sided`` (unit gain, margin 0; ported from hwa exp/20261117-xbar-mapping
  ``training/xbar_mapping.py``): G0 = 9 + 79.2 max(q, 0), G1 = 9 + 79.2 max(-q, 0);
  q_read = (G0 - G1) / 79.2. q is the per-output-channel normalized weight (|q| <= 1) and the
  network keeps applying each channel's digital scale c, so the readout of exactly programmed
  targets is q (times c in the network). A learner write goes to the sign-side device of the
  current read weight (G0 if q > 0, G1 if q < 0, the update's sign at q = 0). An update that
  crosses zero sets that device to 9 uS (a noisy write) and writes the partner with the
  remainder, so the noise-free result is q + dq.
"""
from dataclasses import asdict, dataclass
import hashlib
import math

import torch

from hardware.devices.cmo import ACTIVE, G_MIN_US, HALF_RANGE_US, MID_US, REFERENCE, programming_sigma, to_weight

LAW = 'aihwkit_cmo_log_independent.v1'
# aihwkit 1.1.0 ReRamCMONoiseModel decay_dict, uS per ln(seconds).
AIHWKIT_MEAN_US = -0.08900206
AIHWKIT_SPREAD_SLOPE_US = 0.04201137
AIHWKIT_SPREAD_OFFSET_US = 0.41183342
WRITE_CLOCKS = ('continues', 'reset_absorb')
# Two-sided programming window (xbar_mapping PROGRAM_MIN, PROGRAM_MAX = 9, 88.2 uS).
TWO_SIDED_LOW_US = 9.0
TWO_SIDED_SPAN_US = 79.2
MAPPING_KEYS = ('kind', 'gain', 'margin_us')


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected a finite number for drift_tuning.{name}.')
    if positive and value <= 0:
        raise ValueError(f'Expected a positive drift_tuning.{name}.')
    return float(value)


def parse_mapping(raw):
    """The xbar_mapping settings this runner supports: reference or two_sided, unit gain, margin 0."""
    if not isinstance(raw, dict) or set(raw) != set(MAPPING_KEYS):
        raise ValueError(f'Expected exactly the drift_tuning.mapping fields {sorted(MAPPING_KEYS)}.')
    if raw['kind'] not in ('reference', 'two_sided') or raw['gain'] != {'rule': 'unit'}:
        raise ValueError('Expected a reference or two_sided mapping with the unit gain.')
    if isinstance(raw['margin_us'], bool) or raw['margin_us'] != 0:
        raise ValueError('Expected drift_tuning.mapping.margin_us 0.')
    return {'kind': raw['kind'], 'gain': {'rule': 'unit'}, 'margin_us': 0.0}


@dataclass(frozen=True)
class DriftTuning:
    law: str
    multiplier: float
    mean_us: float
    spread_slope_us: float
    spread_offset_us: float
    drift_min_us: float
    drift_max_us: float
    write_clock: str
    draw_seed: int
    mapping_scale: float
    update_seconds: float
    evaluation_seconds: tuple
    projection_steps: tuple
    projection_seconds: float
    mapping: dict = None  # omitted (the standard pair) keeps older digests unchanged

    def to_dict(self):
        out = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}
        if self.mapping is None:
            del out['mapping']
        return out

    @classmethod
    def parse(cls, raw):
        required = set(cls.__dataclass_fields__) - {'mapping'}
        if not isinstance(raw, dict) or set(raw) - {'mapping'} != required:
            raise ValueError(f'Expected exactly the drift_tuning fields {sorted(required)} (and optionally mapping).')
        if raw['law'] != LAW or raw['write_clock'] not in WRITE_CLOCKS:
            raise ValueError(f'Expected law {LAW} with write_clock in {WRITE_CLOCKS}.')
        mapping = parse_mapping(raw['mapping']) if 'mapping' in raw else None
        if raw['write_clock'] == 'continues' and mapping is not None:
            raise ValueError('write_clock continues keeps the standard pair mapping (omit mapping).')
        coefficients = tuple(_number(raw[k], k) for k in ('mean_us', 'spread_slope_us', 'spread_offset_us'))
        if coefficients != (AIHWKIT_MEAN_US, AIHWKIT_SPREAD_SLOPE_US, AIHWKIT_SPREAD_OFFSET_US):
            raise ValueError('Expected the aihwkit 1.1.0 CMO relaxation coefficients; scale them with multiplier.')
        if type(raw['draw_seed']) is not int or raw['draw_seed'] < 0:
            raise ValueError('Expected a nonnegative integer drift_tuning.draw_seed.')
        low, high = _number(raw['drift_min_us'], 'drift_min_us', True), _number(raw['drift_max_us'], 'drift_max_us')
        if not low < high:
            raise ValueError('Expected drift_min_us < drift_max_us.')
        grid = raw['evaluation_seconds']
        if (not isinstance(grid, list) or len(grid) < 2 or grid[0] != 0
                or any(_number(t, 'evaluation_seconds') < 0 for t in grid) or list(grid) != sorted(set(grid))):
            raise ValueError('Expected an increasing evaluation_seconds grid starting at release (0).')
        steps = raw['projection_steps']
        if not isinstance(steps, list) or any(type(s) is not int or s <= 0 for s in steps) or steps != sorted(set(steps)):
            raise ValueError('Expected increasing positive integer projection_steps.')
        if raw['mapping_scale'] != 1:
            # A conductance gain (the DRN ratio3 field) is not a crossbar mapping; use ``mapping``.
            raise ValueError('The crossbar declares its mapping in drift_tuning.mapping: mapping_scale must be 1.')
        return cls(raw['law'], _number(raw['multiplier'], 'multiplier', True), *coefficients, low, high,
                   raw['write_clock'], raw['draw_seed'], _number(raw['mapping_scale'], 'mapping_scale', True),
                   _number(raw['update_seconds'], 'update_seconds', True), tuple(float(t) for t in grid),
                   tuple(steps), _number(raw['projection_seconds'], 'projection_seconds', True), mapping)

    def age(self, step):
        """Read age (s) of 1-based learner update ``step``; update 0 is release."""
        return step * self.update_seconds

    def step_at(self, seconds):
        """The learner update whose read is at ``seconds``, or None when none is."""
        step = round(seconds / self.update_seconds)
        return step if step > 0 and math.isclose(self.age(step), seconds, rel_tol=0, abs_tol=1e-9) else None


class Mapping:
    """Weight-to-conductance mapping of a pair of devices per weight (see the module docstring)."""

    def __init__(self, spec=None):
        self.spec = None if spec is None else parse_mapping(spec)
        self.kind = 'reference' if spec is None else self.spec['kind']
        self.low = MID_US if self.kind == 'reference' else TWO_SIDED_LOW_US
        self.span = HALF_RANGE_US if self.kind == 'reference' else TWO_SIDED_SPAN_US

    def targets(self, q):
        """(2, size) programmed conductance targets (uS) of normalized weights q."""
        q = q.detach()
        if self.kind == 'reference':
            return torch.stack((MID_US + q * HALF_RANGE_US, torch.full_like(q, MID_US)))
        return torch.stack((self.low + self.span * q.clamp_min(0), self.low + self.span * (-q).clamp_min(0)))

    def readout(self, g):
        """Normalized weights of (2, size) conductances; the reference readout is ``to_weight``."""
        if self.kind == 'reference':
            return to_weight(g[ACTIVE], g[REFERENCE])
        return (g[0] - g[1]) / self.span

    def read_array(self, array):
        return torch.cat([self.readout(child.g) for child in array.children]).detach().clone()

    def clean(self, q):
        """The readout of exactly programmed targets."""
        return self.readout(self.targets(q))


def gate_us(threshold_sigma, noise_scale, acceptance):
    """The nominal mid-range write gate in conductance: threshold_sigma * sigma(48.6 uS)."""
    return float(threshold_sigma * noise_scale * programming_sigma(MID_US, acceptance))


def draw_key(seed, index, seconds):
    """Counter key of the standard-normal draw of array ``index`` read at ``seconds``."""
    text = f'{seed}:{index}:{round(seconds * 1000)}'.encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'little') & ((1 << 63) - 1)


@torch.no_grad()
def drifted(held, index, seconds, law, write_seconds=None, written=None):
    """Read of the held conductances (uS) at absolute time ``seconds`` after release.

    Every clock starts at release unless ``written`` flags devices whose clock restarted at
    ``write_seconds`` (reset_absorb). Unflagged devices use exactly the release-clock
    expression; flagged ones use tau = seconds - t_w with the same draw Y_t.
    """
    if seconds <= 0:
        return held
    log_t = max(math.log(seconds), 0.0)
    generator = torch.Generator(device=held.device)
    generator.manual_seed(draw_key(law.draw_seed, index, seconds))
    y = torch.randn(held.shape, generator=generator, device=held.device, dtype=held.dtype)
    m = law.multiplier
    shifted = held + m * law.mean_us * log_t + m * (law.spread_slope_us * log_t + law.spread_offset_us) * y
    shifted = shifted.clamp(law.drift_min_us, law.drift_max_us)
    if written is None or not bool(written.any()):
        return shifted
    tau = seconds - write_seconds
    log_tau = torch.log(tau.clamp_min(1.0))  # = max(ln tau, 0); tau <= 0 is replaced below
    restarted = held + m * law.mean_us * log_tau + m * (law.spread_slope_us * log_tau + law.spread_offset_us) * y
    restarted = torch.where(tau > 0, restarted.clamp(law.drift_min_us, law.drift_max_us), held)
    return torch.where(written, restarted, shifted)


@torch.no_grad()
def drifted_weights(array, law, seconds):
    """Normalized weights q of a CmoArray whose devices are read at seconds after release.

    Reads every child's held conductances directly (no verify-read accounting); matrix k uses
    draw index k for its (2, size) active/reference devices.
    """
    parts = []
    for k, child in enumerate(array.children):
        g = drifted(child.g, k, seconds, law)
        parts.append(to_weight(g[ACTIVE], g[REFERENCE]))
    return torch.cat(parts)


@torch.no_grad()
def bound_counts(array, law, seconds):
    low = high = total = 0
    for k, child in enumerate(array.children):
        values = drifted(child.g, k, seconds, law)
        low += int((values <= law.drift_min_us).sum())
        high += int((values >= law.drift_max_us).sum())
        total += values.numel()
    return {'devices': total, 'at_drift_min': low, 'at_drift_max': high,
            'at_drift_min_fraction': low / total, 'at_drift_max_fraction': high / total}


class DriftPlant:
    """The held devices of a deployed CmoArray read through the drift law and the mapping.

    ``drift=False`` is the drift-off control: every read returns the held state. Under
    ``reset_absorb`` the plant owns the write law (``write``) and per-device clocks; under
    ``continues`` writes go through the array (``ThresholdedWriteAdam``) and every clock
    stays at release.
    """

    def __init__(self, array, law, mapping, drift=True):
        self.array, self.law, self.mapping, self.drift = array, law, mapping, bool(drift)
        self.write_seconds = [None] * len(array.children)
        self.written = [None] * len(array.children)
        self.counts = [None] * len(array.children)
        self.device_writes = 0
        self.crossing_updates = 0
        self._last = None

    def conductances(self, seconds):
        children = self.array.children
        if not self.drift or seconds is None:
            return [child.g for child in children]
        return [drifted(child.g, k, seconds, self.law, self.write_seconds[k], self.written[k])
                for k, child in enumerate(children)]

    @torch.no_grad()
    def read(self, seconds, keep=False):
        """Normalized weights at ``seconds`` (None: the held state); ``keep`` stores the read for ``write``."""
        g = self.conductances(seconds)
        if keep:
            self._last = (seconds, g)
        return torch.cat([self.mapping.readout(x) for x in g]).detach().clone()

    @torch.no_grad()
    def bounds(self, seconds):
        low = high = total = 0
        for values in self.conductances(seconds):
            low += int((values <= self.law.drift_min_us).sum())
            high += int((values >= self.law.drift_max_us).sum())
            total += values.numel()
        return {'devices': total, 'at_drift_min': low, 'at_drift_max': high,
                'at_drift_min_fraction': low / total, 'at_drift_max_fraction': high / total}

    def _requests(self, gd, dq):
        """Conductance requests (uS) and written flags, both (2, size), for weight updates dq."""
        selected = dq != 0
        if self.mapping.kind == 'reference':
            requested = torch.stack((gd[ACTIVE] + dq * HALF_RANGE_US, gd[REFERENCE]))
            return requested, torch.stack((selected, torch.zeros_like(selected))), 0
        span, low = self.mapping.span, self.mapping.low
        q = (gd[0] - gd[1]) / span
        positive = (q > 0) | ((q == 0) & (dq > 0))
        new = q + dq
        cross = selected & torch.where(positive, new < 0, new > 0)
        # Crossing: the sign-side device goes to the window floor; the partner takes the rest.
        rest = dq - torch.where(positive, (low - gd[0]) / span, (gd[1] - low) / span)
        g0 = torch.where(positive, torch.where(cross, torch.full_like(q, low), gd[0] + span * dq), gd[0] + span * rest)
        g1 = torch.where(positive, gd[1] - span * rest, torch.where(cross, torch.full_like(q, low), gd[1] - span * dq))
        written = torch.stack((selected & (positive | cross), selected & (~positive | cross)))
        return torch.stack((g0, g1)), written, int(cross.sum())

    @torch.no_grad()
    def write(self, issued, seconds):
        """reset_absorb: held <- drifted read at ``seconds`` + step + sigma(G) xi; t_w <- seconds (written only)."""
        if self.law.write_clock != 'reset_absorb':
            raise RuntimeError('Plant writes implement reset_absorb only.')
        if self._last is None or self._last[0] != seconds:
            raise RuntimeError('Expected a write at the age of the learner read it follows.')
        for k, (spec, child) in enumerate(zip(self.array.layout, self.array.children)):
            dq = issued[spec.offset:spec.offset + spec.size]
            if not bool((dq != 0).any()):
                continue
            requested, written, crossings = self._requests(self._last[1][k], dq)
            noise = torch.randn(requested.shape, device=requested.device, generator=child.generator)
            drawn = requested + child.noise_scale * programming_sigma(requested, child.acceptance) * noise
            child.g = torch.where(written, drawn, child.g)
            child._deployment_replayable = False
            if self.written[k] is None:
                self.written[k] = torch.zeros_like(written)
                self.write_seconds[k] = torch.zeros_like(child.g)
                self.counts[k] = torch.zeros(written.shape, dtype=torch.int32, device=written.device)
            self.written[k] |= written
            self.write_seconds[k] = torch.where(written, torch.full_like(child.g, float(seconds)),
                                                self.write_seconds[k])
            self.counts[k] += written
            n = int(written.sum())
            child.writes += written[ACTIVE]
            if not hasattr(child, 'reference_writes'):
                child.reference_writes = torch.zeros_like(child.writes)
            child.reference_writes += written[REFERENCE]
            child.device_writes += n
            self.device_writes += n
            self.crossing_updates += crossings
        self._last = None

    def weight_writes(self):
        """Device writes per logical weight (both devices of the pair)."""
        return torch.cat([torch.zeros(spec.size, dtype=torch.int32, device=child.g.device) if c is None else c.sum(0)
                          for spec, child, c in zip(self.array.layout, self.array.children, self.counts)])

    def cost(self):
        devices = 2 * self.array.size
        touched = sum(0 if w is None else int(w.sum()) for w in self.written)
        most = max([0] + [int(c.max()) for c in self.counts if c is not None])
        return {'device_writes': self.device_writes, 'healthy_device_writes': self.device_writes,
                'stuck_write_attempts': 0, 'touched_devices': touched, 'max_writes_per_device': most,
                'devices': devices, 'writes_per_device': self.device_writes / devices,
                'fraction_devices_written': touched / devices, 'crossing_updates': self.crossing_updates,
                'out_of_range_devices': self.array.out_of_range()}


class GatedDriftAdam:
    """Digital Adam with signed accumulation and a fixed conductance gate, written through a DriftPlant.

    A weight's pending command is issued when its conductance step |span * pending| reaches
    ``threshold_us`` (>=); the issued amount is subtracted, remainders are never flushed and
    there is no write cap. ``mask`` restricts commands to fixed weights (Child-Tuning_D). A
    zero learning rate never writes.
    """

    contract = 'drift_tuning.gated_drift_adam.v1'

    def __init__(self, plant, *, learning_rate, threshold_us, mask=None):
        self.plant, self.learning_rate, self.threshold_us = plant, float(learning_rate), float(threshold_us)
        template = plant.read(None)
        self.first = torch.zeros_like(template)
        self.second = torch.zeros_like(template)
        self.pending = torch.zeros_like(template)
        self.step_index = 0
        self.mask = None if mask is None else mask.detach().clone().to(torch.bool)
        if self.mask is not None and self.mask.shape != template.shape:
            raise ValueError('Expected one write-eligibility flag per logical weight.')

    def fires(self, pending):
        return (pending * self.plant.mapping.span).abs() >= self.threshold_us

    @torch.no_grad()
    def step(self, gradient, seconds):
        from methods.onchip import _adam_command
        command = _adam_command(self, gradient)
        if self.learning_rate == 0:
            return
        if self.mask is not None:
            command = torch.where(self.mask, command, 0.0)
        self.pending.add_(command)
        if not bool(torch.isfinite(self.pending).all()):
            raise FloatingPointError('Nonfinite thresholded write command.')
        fire = self.fires(self.pending)
        if bool(fire.any()):
            issued = torch.where(fire, self.pending, 0.0)
            self.plant.write(issued, seconds)
            self.pending.sub_(issued)

    def diagnostics(self):
        return {'threshold_reference': 'nominal_midrange_conductance', 'threshold_us': self.threshold_us,
                'span_us_per_unit_q': self.plant.mapping.span,
                'threshold_q': self.threshold_us / self.plant.mapping.span, 'accumulate_updates': True,
                'gate_comparison': '>=', 'pending_nonzero': int((self.pending != 0).sum()),
                'pending_rms_q': float(self.pending.square().mean().sqrt())}


def trapezoid_mean(seconds, values):
    """Linear-time average over [seconds[0], seconds[-1]] (drift-regimes endpoint convention)."""
    total = sum((t1 - t0) * (v0 + v1) / 2 for t0, t1, v0, v1 in zip(seconds, seconds[1:], values, values[1:]))
    return total / (seconds[-1] - seconds[0])
