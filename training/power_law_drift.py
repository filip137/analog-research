"""Synthetic scalar power-law drift, with explicit first-read and replay contracts.

This module is mirrored in the DRN worktree. All conductances are in microSiemens;
time is in seconds. A saved normal draw couples ages and amplitudes. This coupling
is a modelling choice, not a measured temporal covariance.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PowerLaw:
    mean_us: float
    sigmas_us: tuple
    mean_exponent: float
    spread_exponent: float
    reference_seconds: float
    development_sequences: int
    first_read_policy: str

    def to_dict(self):
        result = asdict(self)
        result['sigmas_us'] = list(self.sigmas_us)
        return result

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError('Expected exactly the scalar power-law fields.')
        values = dict(raw)
        for key in ('mean_us', 'mean_exponent', 'spread_exponent', 'reference_seconds'):
            v = values[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f'Expected finite {key}.')
            if key != 'mean_us' and v <= 0:
                raise ValueError(f'Expected positive {key}.')
            values[key] = float(v)
        sigmas = values['sigmas_us']
        if (not isinstance(sigmas, (list, tuple)) or not sigmas
                or any(isinstance(s, bool) or not isinstance(s, (int, float))
                       or not math.isfinite(s) or s < 0 for s in sigmas)
                or list(sigmas) != sorted(set(sigmas))):
            raise ValueError('Expected increasing, distinct nonnegative sigmas_us.')
        values['sigmas_us'] = tuple(float(s) for s in sigmas)
        if type(values['development_sequences']) is not int or values['development_sequences'] <= 0:
            raise ValueError('Expected positive development_sequences.')
        if values['first_read_policy'] != 'clamp_programmed':
            raise ValueError('Expected explicit clamp_programmed first-read policy.')
        return cls(**values)


def snapshot(programmed, epsilon, seconds, sigma_us, spec, g_min, g_max):
    """Always start at the saved, bounded first read; never drift recursively."""
    import torch
    if not math.isfinite(seconds) or seconds < 0 or not math.isfinite(sigma_us) or sigma_us < 0:
        raise ValueError('Expected finite nonnegative age and spread.')
    if programmed.shape != epsilon.shape or not g_min < g_max:
        raise ValueError('Invalid drift shape or conductance window.')
    if not bool(torch.isfinite(programmed).all() and torch.isfinite(epsilon).all()):
        raise ValueError('Expected finite drift state.')
    if bool(((programmed < g_min) | (programmed > g_max)).any()):
        raise ValueError('First-read state must already satisfy the device window.')
    u = seconds / spec.reference_seconds
    raw = programmed + spec.mean_us * u**spec.mean_exponent + sigma_us * u**spec.spread_exponent * epsilon
    held = raw.clamp(g_min, g_max)
    delta = held - programmed
    n = programmed.numel()
    # Raw sums aggregate across matrices without incorrectly averaging their SDs.
    stats = {'devices': n, 'clipped_low': int((raw < g_min).sum()),
             'clipped_high': int((raw > g_max).sum()),
             'at_lower_bound': int((held == g_min).sum()), 'at_upper_bound': int((held == g_max).sum()),
             'delta_sum_us': float(delta.double().sum()),
             'delta_square_sum_us2': float(delta.double().square().sum())}
    return held, stats


def aggregate_statistics(rows):
    values = {key: sum(row[key] for row in rows) for key in rows[0]}
    n = values['devices']
    mean = values['delta_sum_us'] / n
    values.update(realized_mean_us=mean,
                  realized_std_us=math.sqrt(max(0, values['delta_square_sum_us2'] / n - mean**2)),
                  clipped_fraction=(values['clipped_low'] + values['clipped_high']) / n)
    return values
