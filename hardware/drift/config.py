"""Stdlib-only parameters for the custom additive drift law."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class DriftLaw:
    mean_us: float = -2.0
    spread_us: float = 3.0
    mean_exponent: float = .23
    spread_exponent: float = .11
    reference_seconds: float = 3600.0
    g_min: float = 9.0
    g_max: float = 88.2

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in asdict(self).values()):
            raise ValueError('Drift parameters must be finite numbers.')
        if (self.spread_us < 0 or min(self.mean_exponent, self.spread_exponent,
                                    self.reference_seconds) <= 0 or self.g_min >= self.g_max):
            raise ValueError('Invalid drift exponents, spread, reference time or bounds.')
