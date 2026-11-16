"""AIHWKit CMO log-time drift on the held fault-tuning crossbar, and its strict settings.

The law, draw keying and settings are the same as the DRN backend's methods.drift_tuning
(drn/20261116-drift-token-budget); only the device container differs. The crossbar keeps
its standard active/reference mapping (mapping_scale must be 1).

Drift is a read transform of the held programmed conductance (uS), never a mutation:

    G(t) = clip(G_p + M*m*L + M*(a*L + b)*Y_t, drift_min, drift_max),   L = max(ln t, 0)

with t the seconds since release (the end of deployment programming) and no drift at
t <= 0. ``Y_t`` is a fresh standard normal for every read age, keyed on
(draw_seed, array index, age in ms), so reads at the same age see the same draw in every
arm and the draw does not depend on evaluation order or on earlier writes.

Write semantics ``write_clock: continues`` (user, 16 November 2026): a recovery write
shifts the held programmed value by its step (with the declared write noise) and does not
restart the device's relaxation clock; every device keeps the clock started at release.
"""
from dataclasses import dataclass
import hashlib
import math

import torch

from hardware.devices.cmo import ACTIVE, REFERENCE, to_weight

LAW = 'aihwkit_cmo_log_independent.v1'
# aihwkit 1.1.0 ReRamCMONoiseModel decay_dict, uS per ln(seconds).
AIHWKIT_MEAN_US = -0.08900206
AIHWKIT_SPREAD_SLOPE_US = 0.04201137
AIHWKIT_SPREAD_OFFSET_US = 0.41183342


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
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError(f'Expected exactly the drift_tuning fields {sorted(cls.__dataclass_fields__)}.')
        if raw['law'] != LAW or raw['write_clock'] != 'continues':
            raise ValueError(f'Expected law {LAW} with write_clock continues.')
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
            raise ValueError('The crossbar keeps its standard pair mapping: mapping_scale must be 1.')
        return cls(raw['law'], _number(raw['multiplier'], 'multiplier', True), *coefficients, low, high,
                   raw['write_clock'], raw['draw_seed'], _number(raw['mapping_scale'], 'mapping_scale', True),
                   _number(raw['update_seconds'], 'update_seconds', True), tuple(float(t) for t in grid),
                   tuple(steps), _number(raw['projection_seconds'], 'projection_seconds', True))

    def age(self, step):
        """Read age (s) of 1-based learner update ``step``; update 0 is release."""
        return step * self.update_seconds

    def step_at(self, seconds):
        """The learner update whose read is at ``seconds``, or None when none is."""
        step = round(seconds / self.update_seconds)
        return step if step > 0 and math.isclose(self.age(step), seconds, rel_tol=0, abs_tol=1e-9) else None


def draw_key(seed, index, seconds):
    """Counter key of the standard-normal draw of array ``index`` read at ``seconds``."""
    text = f'{seed}:{index}:{round(seconds * 1000)}'.encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'little') & ((1 << 63) - 1)


@torch.no_grad()
def drifted(held, index, seconds, law):
    """Read of the held programmed conductances (uS) at ``seconds`` after release."""
    if seconds <= 0:
        return held
    log_t = max(math.log(seconds), 0.0)
    generator = torch.Generator(device=held.device)
    generator.manual_seed(draw_key(law.draw_seed, index, seconds))
    y = torch.randn(held.shape, generator=generator, device=held.device, dtype=held.dtype)
    m = law.multiplier
    shifted = held + m * law.mean_us * log_t + m * (law.spread_slope_us * log_t + law.spread_offset_us) * y
    return shifted.clamp(law.drift_min_us, law.drift_max_us)


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


def trapezoid_mean(seconds, values):
    """Linear-time average over [seconds[0], seconds[-1]] (drift-regimes endpoint convention)."""
    total = sum((t1 - t0) * (v0 + v1) / 2 for t0, t1, v0, v1 in zip(seconds, seconds[1:], values, values[1:]))
    return total / (seconds[-1] - seconds[0])
