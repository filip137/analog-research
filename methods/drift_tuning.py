"""AIHWKit CMO log-time drift on the held fault-tuning bank, and its strict settings.

Drift is a read transform of the held programmed conductance (uS), never a mutation:

    G(t) = clip(G_p + M*m*L + M*(a*L + b)*Y_t, drift_min, drift_max),   L = max(ln t, 0)

with t the seconds since release (the end of deployment programming) and no drift at
t <= 0. ``Y_t`` is a fresh standard normal for every read age, keyed on
(draw_seed, array index, age in ms), so reads at the same age see the same draw in every
arm and the draw does not depend on evaluation order or on earlier writes.

Write semantics ``write_clock: continues`` (user, 16 November 2026): a recovery write
shifts the held programmed value by its step (with the declared write noise) and does not
restart the device's relaxation clock; every device keeps the clock started at release.

Write semantics ``write_clock: reset_absorb`` (user, 18 November 2026, cmo-drift-recovery
exp-001): each device carries the time t_w of its latest recovery write (0 = never written
since release) and reads at absolute time t with tau = t - t_w,

    G(t) = clip(G_held + M*m*L + M*(a*L + b)*Y_t, drift_min, drift_max),   L = max(ln tau, 0),

with ``Y_t`` keyed exactly as above on the absolute read time, so a device that is never
written reads bit-identically to the untreated arm. A write at time t first absorbs the
drifted read at t (the draw the learner read for that update) into the held value, then
applies the bounded noisy write from there, and sets t_w = t for the written devices only
(``AbsorbWritePort``). A device read at its own write time (tau = 0) reads its held value.

The optional ``write_gate`` / ``drift_off_arms`` fields (``CampaignDriftTuning``) select the
gate of the thresholded writer (``target_rho``: the historical 2 rho at the bounded target;
``fixed_mid``: the fixed 2 sigma(48.6 uS) = 1.12 uS of physical conductance) and name arms
that run, and are evaluated, with drift switched off. Configs without them keep the exact
historical resolved form.
"""
from dataclasses import dataclass
import hashlib
import math

import torch

from hardware.devices.cmo import G_MAX_US, G_MIN_US, HALF_RANGE_US, MID_US, programming_sigma
from methods.opt_recovery import RecoveryWritePort

LAW = 'aihwkit_cmo_log_independent.v1'
# aihwkit 1.1.0 ReRamCMONoiseModel decay_dict, uS per ln(seconds).
AIHWKIT_MEAN_US = -0.08900206
AIHWKIT_SPREAD_SLOPE_US = 0.04201137
AIHWKIT_SPREAD_OFFSET_US = 0.41183342
WRITE_CLOCKS = ('continues', 'reset_absorb')
WRITE_GATES = ('target_rho', 'fixed_mid')


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected a finite number for drift_tuning.{name}.')
    if positive and value <= 0:
        raise ValueError(f'Expected a positive drift_tuning.{name}.')
    return float(value)


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

    @classmethod
    def parse(cls, raw):
        base = set(DriftTuning.__dataclass_fields__)
        extended = base | {'write_gate', 'drift_off_arms'}
        if not isinstance(raw, dict) or set(raw) not in (base, extended, extended | {'child_mask_scope'}):
            raise ValueError(f'Expected exactly the drift_tuning fields {sorted(base)}, '
                             'optionally with both write_gate and drift_off_arms, and child_mask_scope.')
        if raw['law'] != LAW or raw['write_clock'] not in WRITE_CLOCKS:
            raise ValueError(f'Expected law {LAW} with write_clock in {WRITE_CLOCKS}.')
        if set(raw) == base and raw['write_clock'] != 'continues':
            raise ValueError('write_clock reset_absorb needs the explicit write_gate and drift_off_arms fields.')
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
        values = (raw['law'], _number(raw['multiplier'], 'multiplier', True), *coefficients, low, high,
                  raw['write_clock'], raw['draw_seed'], _number(raw['mapping_scale'], 'mapping_scale', True),
                  _number(raw['update_seconds'], 'update_seconds', True), tuple(float(t) for t in grid),
                  tuple(steps), _number(raw['projection_seconds'], 'projection_seconds', True))
        if set(raw) == base:
            return DriftTuning(*values)
        if raw['write_gate'] not in WRITE_GATES:
            raise ValueError(f'Expected drift_tuning.write_gate in {WRITE_GATES}.')
        off = raw['drift_off_arms']
        if not isinstance(off, list) or any(not isinstance(x, str) or not x for x in off) or len(set(off)) != len(off):
            raise ValueError('Expected drift_tuning.drift_off_arms as a list of distinct arm labels.')
        if 'child_mask_scope' in raw:
            if raw['child_mask_scope'] != 'global':
                raise ValueError('Expected global child_mask_scope when selecting across arrays.')
            return GlobalChildDriftTuning(*values, raw['write_gate'], tuple(off), 'global')
        return CampaignDriftTuning(*values, raw['write_gate'], tuple(off))

    @property
    def gate(self):
        return 'target_rho'

    @property
    def off_labels(self):
        return ()

    def age(self, step):
        """Read age (s) of 1-based learner update ``step``; update 0 is release."""
        return step * self.update_seconds

    def step_at(self, seconds):
        """The learner update whose read is at ``seconds``, or None when none is."""
        step = round(seconds / self.update_seconds)
        return step if step > 0 and math.isclose(self.age(step), seconds, rel_tol=0, abs_tol=1e-9) else None


@dataclass(frozen=True)
class CampaignDriftTuning(DriftTuning):
    """Drift tuning with an explicit write gate and drift-off arms (cmo-drift-recovery)."""
    write_gate: str = 'target_rho'
    drift_off_arms: tuple = ()

    @property
    def gate(self):
        return self.write_gate

    @property
    def off_labels(self):
        return self.drift_off_arms


@dataclass(frozen=True)
class GlobalChildDriftTuning(CampaignDriftTuning):
    child_mask_scope: str = 'global'


def milliseconds(seconds):
    """Integer read/write time key in ms (the draw-key resolution)."""
    return round(seconds * 1000)


def draw_key(seed, index, seconds):
    """Counter key of the standard-normal draw of array ``index`` read at ``seconds``."""
    text = f'{seed}:{index}:{round(seconds * 1000)}'.encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'little') & ((1 << 63) - 1)


@torch.no_grad()
def drifted(held, index, seconds, law, write_ms=None):
    """Read of the held programmed conductances (uS) at ``seconds`` after release.

    ``write_ms`` (``reset_absorb`` only) holds each device's latest write time in ms, 0 for a
    device never written since release; such devices take exactly the historical expression.
    """
    if seconds <= 0:
        return held
    log_t = max(math.log(seconds), 0.0)
    generator = torch.Generator(device=held.device)
    generator.manual_seed(draw_key(law.draw_seed, index, seconds))
    y = torch.randn(held.shape, generator=generator, device=held.device, dtype=held.dtype)
    m = law.multiplier
    shifted = held + m * law.mean_us * log_t + m * (law.spread_slope_us * log_t + law.spread_offset_us) * y
    read = shifted.clamp(law.drift_min_us, law.drift_max_us)
    if write_ms is None:
        return read
    if write_ms.shape != held.shape:
        raise ValueError('Write-clock shape differs from the held array.')
    written = write_ms > 0
    if not bool(written.any()):
        return read
    tau_ms = milliseconds(seconds) - write_ms.to(torch.int64)
    if bool((written & (tau_ms < 0)).any()):
        raise ValueError('A read precedes the latest write of a device.')
    log_tau = (tau_ms.to(torch.float64) / 1000).clamp_min(1.0).log().to(held.dtype)
    own = (held + m * law.mean_us * log_tau + m * (law.spread_slope_us * log_tau + law.spread_offset_us) * y
           ).clamp(law.drift_min_us, law.drift_max_us)
    own = torch.where(tau_ms > 0, own, held)  # Read at its own write time: no relaxation yet.
    return torch.where(written, own, read)


class WriteClock:
    """Per-device latest recovery write time (ms after release; 0 = not written since release)."""

    def __init__(self, bank):
        self.write_ms = [torch.zeros(a.held.shape, dtype=torch.int32, device=a.held.device) for a in bank.arrays]

    def reset_devices(self):
        return sum(int((w > 0).sum()) for w in self.write_ms)

    def summary(self):
        written = [w[w > 0].double() for w in self.write_ms]
        count = sum(int(w.numel()) for w in written)
        return {'reset_devices': count,
                'mean_last_write_seconds': (sum(float(w.sum()) for w in written) / count / 1000) if count else None,
                'max_last_write_seconds': max((float(w.max()) / 1000 for w in written if w.numel()), default=None)}


def transform(law, seconds, clock=None):
    """``CmoDrnBank.view`` transform reading every array at ``seconds``."""
    if clock is None:
        return lambda index, held: drifted(held, index, seconds, law)
    return lambda index, held: drifted(held, index, seconds, law, clock.write_ms[index])


class AbsorbWritePort(RecoveryWritePort):
    """Bounded noisy recovery write under ``write_clock: reset_absorb``.

    Before writing at read time ``seconds`` the selected devices take their drifted read at
    that time (same key, same draw as the learner's read of that update) as the held value;
    the plant then applies the existing bounded write (target clamp G_MIN..G_MAX, CMO write
    noise at the target) from there, and the written devices restart their clock. Projected
    no-ops and stuck devices keep their held value and clock.
    """

    def __init__(self, array, noise_factor, *, law, index, clock, target_bounds='gmin_gmax'):
        super().__init__(array, noise_factor, target_bounds=target_bounds)
        if target_bounds != 'gmin_gmax':
            raise ValueError('Absorbing drift writes use the bounded G_MIN..G_MAX target law.')
        self.law, self.index, self.clock = law, index, clock
        self.seconds = None

    def drifted_read(self):
        if self.seconds is None or self.seconds <= 0:
            raise ValueError('An absorbing write needs a positive read time.')
        return drifted(self.array.held, self.index, self.seconds, self.law, self.clock.write_ms[self.index])

    def write_noise_std_q(self, command):
        target = (self.drifted_read() + HALF_RANGE_US * command).clamp(G_MIN_US, G_MAX_US)
        return self.array.noise_scale * self.noise_factor * programming_sigma(target, 2.) / HALF_RANGE_US

    @torch.no_grad()
    def write(self, command):
        array = self.array
        selected = (command != 0) & ~array._mask
        original = array.held
        if bool(selected.any()):
            array.held = torch.where(selected, self.drifted_read(), original)
        attempted = super().write(command)
        noop = selected & ~attempted
        if bool(noop.any()):
            array.held = torch.where(noop, original, array.held)
        written = attempted & ~array._mask
        if bool(written.any()):
            self.clock.write_ms[self.index][written] = milliseconds(self.seconds)
        return attempted


@torch.no_grad()
def bound_counts(bank, law, seconds, clock=None):
    low = high = total = 0
    for k, array in enumerate(bank.arrays):
        values = drifted(array.held, k, seconds, law, None if clock is None else clock.write_ms[k])
        low += int((values <= law.drift_min_us).sum())
        high += int((values >= law.drift_max_us).sum())
        total += values.numel()
    return {'devices': total, 'at_drift_min': low, 'at_drift_max': high,
            'at_drift_min_fraction': low / total, 'at_drift_max_fraction': high / total}


@torch.no_grad()
def remap(model, scale):
    """Deployment remapping G' = min(scale G, G_max) of every DRN device target, in place.

    The bias current scales with ``scale``, so with no saturation the DRN equilibrium is
    unchanged (drift-mapping exp-002 ``ratio3`` at scale 3). Returns saturation statistics.
    """
    saturated = total = 0
    for block in model.blocks:
        for u in (block.u1, block.u2):
            raw = scale * (MID_US + HALF_RANGE_US * u.double())
            saturated += int((raw > G_MAX_US).sum())
            total += raw.numel()
            u.copy_(((raw.clamp(max=G_MAX_US) - MID_US) / HALF_RANGE_US).to(u.dtype))
        block.current_scale.mul_(scale)
    return {'scale': scale, 'saturated_devices': saturated, 'devices': total, 'saturated_fraction': saturated / total}


def trapezoid_mean(seconds, values):
    """Linear-time average over [seconds[0], seconds[-1]] (drift-regimes endpoint convention)."""
    total = sum((t1 - t0) * (v0 + v1) / 2 for t0, t1, v0, v1 in zip(seconds, seconds[1:], values, values[1:]))
    return total / (seconds[-1] - seconds[0])


def source_quality_law(multiplier, assignment):
    """Frozen untreated CMO source check at 24 hours. Pair draws across sources on each assignment."""
    from types import SimpleNamespace
    return SimpleNamespace(law=LAW, multiplier=multiplier, draw_seed=391103 + assignment - 271001,
                           mean_us=AIHWKIT_MEAN_US, spread_slope_us=AIHWKIT_SPREAD_SLOPE_US,
                           spread_offset_us=AIHWKIT_SPREAD_OFFSET_US, drift_min_us=5.0, drift_max_us=150.0)
