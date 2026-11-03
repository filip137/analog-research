from dataclasses import replace
import math
import pytest
import torch
from training.writable_drift import DriftLaw, WritableDrift, MaskedAdam, cycle_normal, select_masks
from training.drift_recovery_config import DriftRecovery
from training.drift_recovery import refresh


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def cell(rho=.9, n=1000, law=None):
    return WritableDrift(torch.full((n,), 48.6), law=law or DriftLaw(), rho=rho, seed=17, name='cell')


@pytest.mark.parametrize('rho', [0., .5, .9, 1.])
def test_mixture_preserves_spread_and_cross_cycle_correlation(rho):
    c = cell(rho, 150000)
    first = c.epsilon.clone()
    assert abs(float(first.mean())) < .015
    assert float(first.std()) == pytest.approx(1, abs=.015)
    c.advance(3600)
    c.write(torch.arange(first.numel()), c.target)
    assert float(c.epsilon.std()) == pytest.approx(1, abs=.015)
    correlation = float(torch.corrcoef(torch.stack((first, c.epsilon)))[0, 1])
    assert correlation == pytest.approx(rho, abs=.015)


def test_initial_snapshots_are_paired_across_mixtures_and_legacy_law():
    cells = [cell(r) for r in (0, .9, 1)]
    for t in (0, 3600, 86400):
        values = [c.advance(t) for c in cells]
        assert all(torch.equal(v, values[0]) for v in values)
        expected = (cells[0].target-2*(t/3600)**.23+3*(t/3600)**.11*cells[0].epsilon).clamp(9, 88.2)
        torch.testing.assert_close(values[0], expected, rtol=1e-6, atol=1e-5)


def test_only_touched_cells_reset_and_replay_continues_exactly():
    c = cell()
    before = c.advance(3600).clone()
    ids = torch.arange(0, 1000, 10)
    eps = c.epsilon.clone()
    c.write(ids, c.target[ids])
    untouched = torch.ones(1000, dtype=torch.bool); untouched[ids] = False
    assert torch.equal(c.value()[untouched], before[untouched])
    assert torch.equal(c.epsilon[untouched], eps[untouched])
    assert torch.equal(c.value()[ids], c.target[ids])
    clone = WritableDrift.restore(c.state_dict())
    assert torch.equal(c.advance(7200), clone.advance(7200))
    c.write(ids[:5], torch.full((5,), 30.))
    clone.write(ids[:5], torch.full((5,), 30.))
    assert torch.equal(c.advance(90000), clone.advance(90000))
    assert c.write_commands == 105
    with pytest.raises(ValueError, match='backwards'): c.advance(3600)


def test_draws_do_not_depend_on_other_cells_selection_or_write_order():
    a, b = cell(), cell()
    for c in (a, b): c.advance(3600)
    a.write(torch.tensor([1, 5, 8]), torch.tensor([30., 40., 50.]))
    b.write(torch.tensor([8]), torch.tensor([50.]))
    b.write(torch.tensor([5, 1]), torch.tensor([40., 30.]))
    assert torch.equal(a.advance(7200), b.advance(7200))


def test_masked_adam_enforces_budget_and_never_writes_outside_mask():
    c = cell(n=100)
    c.advance(3600)
    frozen = c.state_dict()
    indices = torch.arange(10)
    writer = MaskedAdam([c], [indices], .01, 20)
    for _ in range(2):
        writer.view()[0][:10].square().mean().backward()
        writer.step()
    assert not writer.can_step()
    with pytest.raises(RuntimeError, match='budget'): writer.step()
    assert c.write_commands == 20
    assert torch.equal(c.anchor[10:], frozen['anchor'][10:])
    assert torch.equal(c.last_write[10:], frozen['last_write'][10:])
    assert torch.equal(c.epsilon[10:], frozen['epsilon'][10:])
    assert bool((c.cycles[:10] == 2).all())


def test_censored_selection_and_matched_controls():
    c = cell(n=2000)
    c.target[:100] = c.anchor[:100] = 9
    c.persistent[:100] = c.epsilon[:100] = -5
    c.advance(3600)
    group = torch.arange(2000)//1000
    role = torch.arange(2000) % 2
    masks, info = select_masks([c], [group], [role], .1, 81)
    assert info['selected_devices'] == 200 and info['excluded_devices'] >= 100
    for key in masks:
        assert not bool((masks[key][0] < 100).any())
    for g in (0, 1):
        for r in (0, 1):
            counts = [int(((group[v[0]] == g) & (role[v[0]] == r)).sum()) for v in masks.values()]
            assert len(set(counts)) == 1
    with pytest.raises(ValueError, match='uncensored'):
        select_masks([c], [group], [role], 1., 81)


def test_refresh_is_one_subset_pass_and_full_refresh_restores_all_targets():
    c = cell(n=1000); c.advance(3600)
    groups, roles = [torch.zeros(1000, dtype=torch.int8)], [torch.zeros(1000, dtype=torch.int8)]
    result = refresh([c], groups, roles, .1, 'worst_refresh', 12)
    assert result['cell_writes'] == 100
    assert int((c.cycles > 0).sum()) == 100
    result = refresh([c], groups, roles, .1, 'full_refresh', 12)
    assert result['cell_writes'] == 1000
    assert torch.equal(c.value(), c.target)


def test_no_drift_is_identity_even_after_exact_writes():
    c = cell(law=DriftLaw(mean_us=0, spread_us=0))
    assert torch.equal(c.advance(1e8), c.target)
    c.write(torch.tensor([5]), torch.tensor([20.]))
    assert float(c.advance(2e8)[5]) == 20


def test_strict_settings_and_five_weekly_events():
    config = DriftRecovery('a'*64)
    assert DriftRecovery.parse(config.to_dict()) == config
    assert config.intervention_times('repeated') == (3600, 608400, 1213200, 1818000, 2422800)
    for change in ({'selection_fraction': 0}, {'persistent_fractions': [2]}, {'unknown': 1},
                   {'write_budgets': [False]}, {'law': {'mean_us': -2}}):
        doc = config.to_dict(); doc.update(change)
        with pytest.raises(ValueError): DriftRecovery.parse(doc)
