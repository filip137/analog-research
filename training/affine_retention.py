"""Affine coordinates for a NumPy drift predictor, without changing its time law.

The base predictor follows the flowers DriftModel interface: first-read anchors,
conductance increments, marginal scales, and jointly sampled trajectories. This
wrapper imports no dataset and starts no simulation or network run. Its bounds
define coordinates, not clipping or a new validity domain. In particular, the
base model's age-zero and measured-age restrictions still apply.

Keep a checkpoint's existing conductance encoding and query the source model
through this wrapper. This transfers a normalized drift law to that encoding;
it does not establish physical equivalence between device populations or DRNs.
"""

import math
from numbers import Real

import numpy as np


def _bounds(values):
    values = tuple(values)
    if (len(values) != 2
            or any(isinstance(x, bool) or not isinstance(x, Real)
                   or not math.isfinite(x) for x in values)
            or values[0] >= values[1]):
        raise ValueError("Expected two finite increasing conductance bounds.")
    return tuple(float(x) for x in values)


class AffineDriftModel:
    """Expose a source predictor in destination conductance coordinates.

    Units of all G0, dG and g_bounds arguments are destination units. Time,
    RNG ownership, shape, and read-noise choices pass through unchanged.
    mean/total_sd/cov retain the base method's interpretation; for a non-unit
    marginal shape they are not necessarily the actual predictive moments.
    """

    def __init__(self, base, *, source_bounds, destination_bounds):
        self.base = base
        self.source_bounds = _bounds(source_bounds)
        self.destination_bounds = _bounds(destination_bounds)
        self.scale = ((self.destination_bounds[1] - self.destination_bounds[0])
                      / (self.source_bounds[1] - self.source_bounds[0]))

    def to_source(self, conductance):
        return (self.source_bounds[0]
                + (np.asarray(conductance) - self.destination_bounds[0]) / self.scale)

    def from_source(self, conductance):
        return (self.destination_bounds[0]
                + self.scale * (np.asarray(conductance) - self.source_bounds[0]))

    def mean(self, G0, t):
        return self.scale * self.base.mean(self.to_source(G0), t)

    def var_parts(self, G0, t):
        return tuple(self.scale ** 2 * part
                     for part in self.base.var_parts(self.to_source(G0), t))

    def total_sd(self, G0, t, read_noise=True):
        return self.scale * self.base.total_sd(self.to_source(G0), t, read_noise=read_noise)

    def cov(self, G0, times, read_noise=True):
        return self.scale ** 2 * self.base.cov(self.to_source(G0), times, read_noise=read_noise)

    def logpdf(self, dG, G0, t, shape="nig"):
        return (self.base.logpdf(np.asarray(dG) / self.scale, self.to_source(G0), t, shape=shape)
                - math.log(self.scale))

    def cdf(self, dG, G0, t, shape="nig"):
        return self.base.cdf(np.asarray(dG) / self.scale, self.to_source(G0), t, shape=shape)

    def sample(self, G0, times, rng, read_noise=True, shape="nig", g_bounds=None):
        # One base call preserves the joint history and consumes no extra RNG.
        # Clipping, if explicitly requested, is expressed in destination units.
        bounds = None if g_bounds is None else tuple(self.to_source(_bounds(g_bounds)))
        values = self.base.sample(self.to_source(G0), times, rng,
                                  read_noise=read_noise, shape=shape, g_bounds=bounds)
        return self.from_source(values)

