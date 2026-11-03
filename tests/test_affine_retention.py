"""Coordinate invariants for drift predictors; no network or measured-data dependency."""

import math

import numpy as np
import pytest

from training.affine_retention import AffineDriftModel


class JointReference:
    """A correlated, state-dependent process with one fresh draw per episode."""

    def sample(self, G0, times, rng, read_noise=True, shape="nig", g_bounds=None):
        G0 = np.atleast_1d(G0)
        t = np.asarray(times)
        increments = rng.normal(size=(len(G0), len(t))) * np.sqrt(np.diff(np.r_[0., t]))
        values = G0[:, None] - t[None, :] * 0.01 + np.cumsum(increments, axis=1) / G0[:, None]
        return values if g_bounds is None else values.clip(*g_bounds)

    def mean(self, G0, t):
        return -np.asarray(t) * 0.01

    def var_parts(self, G0, t):
        return np.asarray(t) / np.asarray(G0) ** 2, 0., 0.

    def total_sd(self, G0, t, read_noise=True):
        return np.sqrt(self.var_parts(G0, t)[0])

    def cov(self, G0, times, read_noise=True):
        return np.minimum.outer(times, times) / np.asarray(G0) ** 2

    def logpdf(self, dG, G0, t, shape="nig"):
        sd = self.total_sd(G0, t)
        z = (np.asarray(dG) - self.mean(G0, t)) / sd
        return -z ** 2 / 2 - np.log(sd) - math.log(2 * math.pi) / 2

    def cdf(self, dG, G0, t, shape="nig"):
        z = (np.asarray(dG) - self.mean(G0, t)) / self.total_sd(G0, t)
        return 0.5 * (1 + np.vectorize(math.erf)(z / np.sqrt(2)))


def model():
    return AffineDriftModel(JointReference(), source_bounds=(51.6, 154.8),
                            destination_bounds=(9., 88.199997))


def test_checkpoint_encoding_and_correlated_pair_paths_are_invariant():
    m = model()
    half = (88.199997 - 9.) / 2
    anchors = (88.199997 + 9.) / 2 + half * np.linspace(-1, 1, 21)
    source_anchors = m.to_source(anchors)
    np.testing.assert_allclose(m.from_source(source_anchors), anchors, atol=2e-14)
    source_rng = np.random.default_rng(31)
    mapped_rng = np.random.default_rng(31)
    times = [3., 60., 600., 3600.]
    source = m.base.sample(source_anchors, times, source_rng)
    mapped = m.sample(anchors, times, mapped_rng)
    # Decode weights independently in each coordinate system.
    np.testing.assert_allclose((mapped[1:] - mapped[:-1]) / half,
                               (source[1:] - source[:-1]) / 51.6, atol=1e-14)
    assert source_rng.bit_generator.state == mapped_rng.bit_generator.state
    # No default clipping: drift can leave a target window.
    assert mapped[0, -1] < 9.
    clipped = m.sample(anchors, times, np.random.default_rng(31), g_bounds=(9., 88.199997))
    np.testing.assert_allclose(clipped, mapped.clip(9., 88.199997), atol=2e-14)


def test_marginal_probability_mass_and_temporal_scale_are_preserved():
    m = model()
    anchor, t = 48.5999985, 60.
    mu = float(m.mean(anchor, t))
    sd = float(m.total_sd(anchor, t))
    grid = np.linspace(mu - 8 * sd, mu + 8 * sd, 20001)
    density = np.exp(m.logpdf(grid, anchor, t))
    # This fails if the density's Jacobian is omitted.
    assert np.trapezoid(density, grid) == pytest.approx(1., abs=1e-10)
    assert float(m.cdf(mu, anchor, t)) == pytest.approx(.5)
    times = np.array([3., 60., 600., 3600.])
    K = m.cov(anchor, times)
    np.testing.assert_allclose(np.diag(K), m.total_sd(anchor, times) ** 2)
    np.testing.assert_allclose(sum(m.var_parts(anchor, times)), np.diag(K))
    source_K = m.base.cov(m.to_source(anchor), times)
    np.testing.assert_allclose(K / np.sqrt(np.outer(np.diag(K), np.diag(K))),
                               source_K / np.sqrt(np.outer(np.diag(source_K), np.diag(source_K))))


@pytest.mark.parametrize("bounds", [(1., 1.), (2., 1.), (0., float("nan")),
                                    (0., float("inf")), (False, 1.), (1.,)])
def test_invalid_coordinate_windows_are_rejected(bounds):
    with pytest.raises(ValueError):
        AffineDriftModel(JointReference(), source_bounds=bounds, destination_bounds=(9., 88.2))
    with pytest.raises(ValueError):
        AffineDriftModel(JointReference(), source_bounds=(51.6, 154.8), destination_bounds=bounds)

