"""Drift-regimes exp-001 unit checks for training/drift_family.py.

Mirrored verbatim in the DRN worktree with the module; the PCM reference test
needs hwa's training/inference_retention.py and skips where it is absent.
"""
import math

import pytest
import torch

from training.drift_family import (CMO_SIGMA, PcmDriftCells, PcmFamily, ReramDriftCells,
                                   ReramFamily, kappa_from_rho_star, program_verify)
from training.reram_cmo import PROGRAMMING_SIGMA
from training.writable_drift import DriftLaw, WritableDrift, cycle_normal

# kappa = 0, exact endpoints, unit amplitudes and the 9-88.2 uS drift clip of WritableDrift.
LEGACY = dict(rho_star=1.0, cmo_scale=0.0, amplitude_sigma=0.0, program_min=9.0,
              program_max=88.2, drift_min=9.0, drift_max=88.2, time_dtype='float32')
WIDE = dict(drift_min=-1e4, drift_max=1e4)   # no drift clipping: isolates the temporal law
TOL = .01


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def reram(n=1000, target=48.6, **law):
    t = torch.full((n,), float(target)) if not isinstance(target, torch.Tensor) else target
    return ReramDriftCells(t, law=ReramFamily(**law), seed=17, name='cell')


def corr(a, b):
    return float(torch.corrcoef(torch.stack((a.flatten(), b.flatten())))[0, 1])


# -- compatibility ------------------------------------------------------------------
@pytest.mark.parametrize('rho', [0., .5, .9, 1.])
def test_kappa_zero_replays_writable_drift_bit_exactly(rho):
    g = torch.Generator().manual_seed(3)
    target = 2 + 95 * torch.rand(4000, generator=g)          # includes out-of-window targets
    old = WritableDrift(target, law=DriftLaw(), rho=rho, seed=17, name='cell')
    new = ReramDriftCells(target, law=ReramFamily(**LEGACY, persistent_share=rho), seed=17, name='cell')
    for t in (0., 60., 3600.):
        assert torch.equal(old.advance(t), new.advance(t))
    ids = torch.randperm(4000, generator=g)[:700]
    values = 2 + 95 * torch.rand(700, generator=g)
    old.write(ids, values); new.write(ids, values)
    assert torch.equal(old.value(), new.value())
    new = ReramDriftCells.restore(new.state_dict())
    for t in (7200., 86400.):
        assert torch.equal(old.advance(t), new.advance(t))
    old.write(ids[:50], values[:50]); new.write(ids[:50], values[:50])
    assert torch.equal(old.advance(9e5), new.advance(9e5))
    assert old.write_commands == new.write_commands == 750


def test_coefficient_tables_match_the_hwa_device_modules():
    assert CMO_SIGMA == PROGRAMMING_SIGMA


# -- matched marginals and knob statistics ------------------------------------------
@pytest.mark.parametrize('rho_star,f', [(1., 0.), (.335, 0.), (.18, .7), (.5, 1.)])
def test_marginal_is_the_power_law_at_every_age_and_episode(rho_star, f):
    c = reram(200_000, rho_star=rho_star, persistent_share=f, cmo_scale=0., **WIDE)
    law = c.law
    for episode in range(3):
        start = c.clock
        for age in (0.5, 60., 3600., 86400.):
            g = c.advance(start + age)
            y = c.oracle_y()
            assert abs(float(y.mean())) < TOL and float(y.std()) == pytest.approx(1, abs=TOL)
            u = age / 3600
            expected = c.anchor + law.mean_us * u**law.mean_exponent + law.spread_us * u**law.spread_exponent * y
            torch.testing.assert_close(g, expected, rtol=0, atol=2e-4)
        c.write(torch.arange(c.target.numel()), c.target)


@pytest.mark.parametrize('rho_star', [.7, .335, .18])
def test_temporal_kernel_hits_rho_star_and_the_power_law_kernel(rho_star):
    c = reram(200_000, rho_star=rho_star, cmo_scale=0., **WIDE)
    kappa = kappa_from_rho_star(rho_star)
    ys = {}
    for t in (60., 300., 3600., 90000.):
        c.advance(t)
        ys[t] = c.oracle_y().clone()
    assert corr(ys[300.], ys[90000.]) == pytest.approx(rho_star, abs=TOL)
    assert corr(ys[60.], ys[3600.]) == pytest.approx((60 / 3600) ** kappa, abs=TOL)


@pytest.mark.parametrize('f', [0., .3, .7, 1.])
def test_consecutive_write_episodes_correlate_at_f(f):
    c = reram(200_000, persistent_share=f, cmo_scale=0., **WIDE)
    ids = torch.arange(200_000)
    ys = []
    for _ in range(3):
        start = c.clock
        c.advance(start + 3600)
        ys.append(c.oracle_y().clone())
        c.write(ids, c.target)
    assert corr(ys[0], ys[1]) == pytest.approx(f, abs=TOL)
    assert corr(ys[1], ys[2]) == pytest.approx(f, abs=TOL)


@pytest.mark.parametrize('scale', [1., 3.])
def test_cmo_endpoint_spread_and_programming_clamp(scale):
    c = reram(200_000, target=48.6, cmo_scale=scale)
    sigma = scale * (CMO_SIGMA[2.0][0] + CMO_SIGMA[2.0][1] * 48.6)
    error = c.anchor - 48.6
    assert float(error.std()) == pytest.approx(sigma, rel=TOL)
    assert abs(float(error.mean())) < 3 * sigma / math.sqrt(200_000) + 1e-5
    assert torch.equal(c.value(), c.anchor)                     # tau = 0 reads the endpoint
    low = reram(10_000, target=torch.tensor([2.] * 5000 + [95.] * 5000), cmo_scale=scale)
    assert float(low.anchor.min()) >= 9 and float(low.anchor.max()) <= 88.2
    assert bool((low.anchor[:5000] == 9).any()) and bool((low.anchor[5000:] == 88.2).any())


def test_amplitude_heterogeneity_has_unit_power_and_declared_log_spread():
    c = reram(200_000, amplitude_sigma=.5)
    a = c.amplitude
    assert float(a.square().mean()) == pytest.approx(1, abs=2 * TOL)
    assert float(a.log().std()) == pytest.approx(.5, abs=TOL)


def test_drift_floor_fraction_matches_the_campaign_table():
    c = reram(200_000, target=9.)
    law = c.law
    for age, table in ((3600., .25), (86400., .51)):
        g = c.advance(age)
        u = age / 3600
        sd = math.hypot(law.spread_us * u**law.spread_exponent, law.cmo_sigma(9.))
        z = (5 - 9 - law.mean_us * u**law.mean_exponent) / sd
        expected = .5 * (1 + math.erf(z / math.sqrt(2)))
        stats = c.bound_statistics(g)
        assert stats['at_drift_min'] / stats['devices'] == pytest.approx(expected, abs=TOL)
        assert expected == pytest.approx(table, abs=.01)
        assert stats['at_drift_max'] == 0 and float(g.min()) == 5


# -- determinism ------------------------------------------------------------------
def test_values_do_not_depend_on_queried_ages_write_order_or_restores():
    a, b = reram(5000, persistent_share=.3), reram(5000, persistent_share=.3)
    for t in (1., 30., 60., 61., 900., 3600.):
        a.advance(t)
    b.advance(3600.)
    assert torch.equal(a.value(), b.value())
    a.write(torch.tensor([1, 5, 8]), torch.tensor([30., 40., 50.]))
    b.write(torch.tensor([8]), torch.tensor([50.]))
    b.write(torch.tensor([5, 1]), torch.tensor([40., 30.]))
    peek = a.value(50000.)
    for t in (4000., 20000.):
        a.advance(t)
    assert torch.equal(peek, a.value(50000.))
    clone = ReramDriftCells.restore(b.state_dict())
    for c in (a, b, clone):
        c.advance(50000.)
    assert torch.equal(a.value(), b.value()) and torch.equal(b.value(), clone.value())
    subset = torch.tensor([4, 1, 4999])
    assert torch.equal(a.value(indices=subset), a.value()[subset])
    with pytest.raises(ValueError, match='backwards'):
        a.advance(100.)


def test_reads_are_exact_and_counted():
    c = reram(100)
    c.advance(60.)
    assert torch.equal(c.read(), c.value()) and c.selection_reads == 100
    assert torch.equal(c.read(torch.tensor([3, 7])), c.value()[[3, 7]]) and c.selection_reads == 102


# -- PCM -----------------------------------------------------------------------------
def test_pcm_without_writes_is_the_pinned_aihwkit_snapshot(monkeypatch):
    retention = pytest.importorskip('training.inference_retention')
    target = torch.linspace(9, 88.2, 5000)
    law = PcmFamily(drift_min=0., drift_max=1e9)
    c = PcmDriftCells(target, law=law, seed=5, name='pcm')
    xi = cycle_normal(torch.arange(5000), torch.zeros(5000, dtype=torch.int32), c._nu_seed)
    monkeypatch.setattr(retention, '_normal', lambda value, generator: xi)
    assert torch.equal(c.oracle_nu(), retention.pcm_coefficients(target, None, g_max=88.2))
    for t in (0., 60., 86400., 3.15e7):
        expected = retention.pcm_snapshot(target, c.oracle_nu(), None, t, g_max=88.2,
                                          read_noise_scale=0.)
        torch.testing.assert_close(c.advance(t), expected, rtol=2e-6, atol=0)


def test_pcm_write_restarts_the_clock_and_redraws_the_exponent():
    c = PcmDriftCells(torch.full((1000,), 40.), law=PcmFamily(), seed=5, name='pcm')
    c.advance(3600.)
    before, nu = c.value().clone(), c.oracle_nu()
    ids = torch.arange(0, 1000, 2)
    c.write(ids, torch.full((500,), 40.))
    assert torch.equal(c.value()[ids], torch.full((500,), 40.))
    assert torch.equal(c.value()[1::2], before[1::2])
    assert bool((c.oracle_nu()[ids] != nu[ids]).all()) and torch.equal(c.oracle_nu()[1::2], nu[1::2])
    floor = PcmDriftCells(torch.full((1000,), 9.), law=PcmFamily(), seed=5, name='pcm')
    assert float(floor.advance(3.15e7).min()) == 5.


# -- program-and-verify ------------------------------------------------------------------
@pytest.mark.parametrize('rho_star,mse,writes', [(1., .2489, 1.8229), (.335, .9538, 1.8229)])
def test_program_verify_reproduces_the_analytic_optimum(rho_star, mse, writes):
    """pv_optimal.best_kappa(f=0, g=0, rho=(60 s / 24 h)^kappa, N=4): kappa* = 0.7051."""
    n = 400_000
    c = reram(n, rho_star=rho_star, cmo_scale=0., **WIDE)
    stats = program_verify(c, torch.arange(n), c.target, attempts=4, threshold=.7051477,
                           verify_seconds=60.)
    assert stats['reset_attempts'] / n == pytest.approx(writes, abs=TOL)
    # Devices that reach the last, unverified attempt: (1 - p)^(N-1), p = P(|z| <= kappa*).
    assert stats['capped'] / n == pytest.approx((1 - math.erf(.7051477 / math.sqrt(2)))**3, abs=.003)
    assert stats['verify_reads'] == stats['reset_attempts'] - stats['capped']
    c.advance(stats['start_seconds'] + 86400.)
    assert float(c.oracle_y().square().mean()) == pytest.approx(mse, abs=TOL)


def test_single_attempt_is_one_unverified_reset():
    c = reram(1000)
    stats = program_verify(c, torch.arange(1000), c.target, attempts=1, threshold=None,
                           verify_seconds=60.)
    assert stats['reset_attempts'] == 1000 and stats['verify_reads'] == 0 and c.clock == 0
    with pytest.raises(ValueError, match='threshold'):
        program_verify(c, torch.arange(10), c.target[:10], attempts=2, threshold=None,
                       verify_seconds=60.)


def test_pcm_program_verify_rejects_fast_drifters():
    c = PcmDriftCells(torch.full((100_000,), 40.), law=PcmFamily(), seed=5, name='pcm')
    stats = program_verify(c, torch.arange(100_000), c.target, attempts=4, threshold=.7051477,
                           verify_seconds=60.)
    mean, std = c.law.nu_moments(torch.tensor(40.))
    z = (c.oracle_nu() - mean) / std
    assert float(z.square().mean()) == pytest.approx(.2489, abs=.02)
    assert stats['reset_attempts'] / 100_000 == pytest.approx(1.8229, abs=.02)
