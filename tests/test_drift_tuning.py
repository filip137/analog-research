"""Fault-tuning learner under AIHWKit CMO log-time drift: law, write semantics, remap, configs."""

import json
import math
from pathlib import Path

import pytest
import torch

from ebl.runtime.opt_cmo_drn.drift_tuning import DriftPhysicalArm
from ebl.schema.base import ConfigError
from ebl.schema.opt_cmo_drn import DriftTuningConfig, FaultTuningConfig, parse_config
from methods import drift_tuning as dt
from methods.programming.cmo import program_rewritten
from networks.opt.physical import CmoDrnBank
from tests.test_opt_cmo_drn import cpu_threads, tiny_model  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'examples' / 'opt_cmo_drn' / 'drift_token_budget'
FAULT = ROOT / 'examples' / 'opt_cmo_drn' / 'fault_token_budget' / 'full' / 'drn6-n8192.json'


def law(**changes):
    raw = json.loads((CONFIGS / 'full' / 'drn6-ratio3-m4.json').read_text())['drift_tuning']
    raw.update(changes)
    return dt.DriftTuning.parse(raw)


def test_law_marginal_matches_aihwkit_times_multiplier_and_is_identity_at_release():
    d = law()
    held = torch.full((400, 500), 50.0)
    assert torch.equal(dt.drifted(held, 0, 0.0, d), held)
    read = dt.drifted(held, 0, 3600.0, d).double()
    log_t = math.log(3600.0)
    assert float(read.mean()) == pytest.approx(50 + 4 * dt.AIHWKIT_MEAN_US * log_t, abs=0.02)
    spread = 4 * (dt.AIHWKIT_SPREAD_SLOPE_US * log_t + dt.AIHWKIT_SPREAD_OFFSET_US)
    assert float(read.std()) == pytest.approx(spread, rel=0.01)
    # Below one second the log is floored at zero: only the offset spread remains.
    sub = dt.drifted(held, 0, 0.5, d).double()
    assert float(sub.mean()) == pytest.approx(50.0, abs=0.02)
    assert float(sub.std()) == pytest.approx(4 * dt.AIHWKIT_SPREAD_OFFSET_US, rel=0.01)


def test_draws_are_keyed_on_seed_array_and_age_and_reads_are_clipped():
    d = law()
    held = torch.full((64, 64), 20.0)
    a = dt.drifted(held, 3, 900.0, d)
    assert torch.equal(a, dt.drifted(held.clone(), 3, 900.0, d))
    assert not torch.equal(a, dt.drifted(held, 4, 900.0, d))
    assert not torch.equal(a, dt.drifted(held, 3, 901.5, d))
    assert not torch.equal(a, dt.drifted(held, 3, 900.0, law(draw_seed=d.draw_seed + 1)))
    low = dt.drifted(torch.full((64, 64), 5.5), 0, 86400.0, d)
    assert float(low.min()) == d.drift_min_us and float(low.max()) <= d.drift_max_us


def test_steps_map_onto_the_grid_and_the_trapezoid_is_linear_time():
    d = law()
    assert d.age(2400) == 3600.0 and d.step_at(60.0) == 40 and d.step_at(900.0) == 600
    assert d.step_at(61.0) is None and d.step_at(0.0) is None
    assert dt.trapezoid_mean([0, 1, 3], [0.0, 1.0, 1.0]) == pytest.approx((0.5 + 2.0) / 3)


def test_ratio3_remap_triples_targets_and_bias_current(cpu_threads):
    model, _ = tiny_model()
    targets = [(dt.MID_US + dt.HALF_RANGE_US * u.detach().double()).clone() for b in model.blocks for u in (b.u1, b.u2)]
    scales = [b.current_scale.clone() for b in model.blocks]
    stats = dt.remap(model, 3.0)
    after = [(dt.MID_US + dt.HALF_RANGE_US * u.detach().double()) for b in model.blocks for u in (b.u1, b.u2)]
    for t0, t1 in zip(targets, after):
        assert torch.allclose(t1, (3 * t0).clamp(max=dt.G_MAX_US), atol=1e-4)
    assert all(torch.equal(b.current_scale, 3 * s) for b, s in zip(model.blocks, scales))
    assert stats['devices'] == sum(t.numel() for t in targets)


def test_writes_shift_the_held_value_without_restarting_the_clock(cpu_threads):
    model, _ = tiny_model()
    dt.remap(model, 3.0)
    d = law()
    _, _, state = program_rewritten(model, 271001, 'none', 0.0, capture_state=True)
    tokens, samples = torch.randint(0, 17, (4, 6)), torch.randint(0, 17, (4, 5, 8))
    hold, learner_bank = CmoDrnBank(state, 'cpu'), CmoDrnBank(state, 'cpu')
    before = [a.held.clone() for a in learner_bank.arrays]
    arm = DriftPhysicalArm(model, learner_bank, 'noisy', 1.0, 4)
    for step in range(1, 5):
        arm.step_at(tokens, samples, 2, 'cpu', dt.transform(d, d.age(step)))
    assert arm.write_report()['costs']['write_attempts'] > 0
    arm.close()
    for k, (a, h, b) in enumerate(zip(learner_bank.arrays, hold.arrays, before)):
        assert torch.equal(h.held, b), 'the hold bank is untouched'
        changed = a.held != b
        # Same age -> same draw: an unwritten device reads identically in both banks, a written one
        # differs exactly by its held shift (before clipping), i.e. its clock was not reset.
        r_learn, r_hold = dt.drifted(a.held, k, 7200.0, d), dt.drifted(h.held, k, 7200.0, d)
        assert torch.equal(r_learn[~changed], r_hold[~changed])
        inside = changed & (r_learn > d.drift_min_us) & (r_hold > d.drift_min_us)
        assert torch.allclose((r_learn - r_hold)[inside], (a.held - h.held)[inside], atol=1e-4)
    assert any(bool((a.held != b).any()) for a, b in zip(learner_bank.arrays, before))


def test_transformed_view_reads_without_mutating_and_zero_rate_replays_hold(cpu_threads):
    model, _ = tiny_model()
    d = law()
    _, _, state = program_rewritten(model, 271001, 'none', 0.0, capture_state=True)
    bank = CmoDrnBank(state, 'cpu')
    hashes = bank.held_hashes()
    tokens = torch.randint(0, 17, (2, 6))
    model.eval()
    with torch.no_grad():
        plain = model(tokens, bank.view(model))
        drifted_a = model(tokens, bank.view(model, transform=dt.transform(d, 3600.0)))
        drifted_b = model(tokens, bank.view(model, transform=dt.transform(d, 3600.0)))
        assert torch.equal(model(tokens, bank.view(model, transform=dt.transform(d, 0.0))), plain)
    assert torch.equal(drifted_a, drifted_b) and not torch.equal(drifted_a, plain)
    assert bank.held_hashes() == hashes
    with pytest.raises(ValueError):
        bank.view(model, raw=True, transform=dt.transform(d, 1.0))


def test_generated_configs_parse_and_differ_from_the_fault_run_only_where_declared():
    fault = json.loads(FAULT.read_text())
    for kind in ('full', 'canary'):
        for path in sorted((CONFIGS / kind).glob('drn6-*.json')):
            raw = json.loads(path.read_text())
            c = parse_config(raw)
            assert isinstance(c, DriftTuningConfig) and c.deployment.fault_kind == 'none'
            assert c.drift_tuning.mapping_scale == 3.0 and c.drift_tuning.multiplier == 4.0
            assert [a.method for a in c.fault_tuning.arms] == ['hold', 'noisy']
            assert c.fault_tuning.arms[1].learning_rate == 3e-4
            f, g = dict(raw['fault_tuning']), dict(fault['fault_tuning'])
            for x in (f, g):
                x.pop('arms'), x.pop('steps')
            assert f == g
            assert raw['recovery'] == fault['recovery'] and raw['solver'] == fault['solver']
    assert isinstance(parse_config(fault), FaultTuningConfig)
    assert not isinstance(parse_config(fault), DriftTuningConfig)


def test_invalid_drift_configs_are_rejected():
    raw = json.loads((CONFIGS / 'full' / 'drn6-ratio3-m4.json').read_text())

    def broken(edit):
        doc = json.loads(json.dumps(raw))
        edit(doc)
        with pytest.raises(ConfigError):
            parse_config(doc)

    broken(lambda d: d['drift_tuning'].update(evaluation_seconds=[0, 61, 86400]))
    broken(lambda d: d['drift_tuning'].update(write_clock='restarts'))
    broken(lambda d: d['drift_tuning'].update(mean_us=-0.356))
    broken(lambda d: d['drift_tuning'].update(projection_seconds=3600.0))
    broken(lambda d: d.update(deployment={'assignment_seed': 271001, 'fault_kind': 'random', 'fault_rate': 0.03}))
    broken(lambda d: d['fault_tuning'].update(arms=[{'label': 'L1', 'method': 'noisy', 'learning_rate': 3e-4}]))
    broken(lambda d: d['fault_tuning'].update(arms=d['fault_tuning']['arms'] +
                                              [{'label': 'C', 'method': 'calibration', 'learning_rate': 1e-3}]))
