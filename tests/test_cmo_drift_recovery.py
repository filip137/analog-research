"""cmo-drift-recovery exp-001: reset-absorb write clock, fixed write gate, new drift arms, configs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ebl.runtime.opt_cmo_drn.drift_tuning as stage
import ebl.runtime.opt_cmo_drn.runtime as runtime
from ebl.runtime.opt_cmo_drn.drift_tuning import DriftPhysicalArm
from ebl.schema.base import ConfigError
from ebl.schema.opt_cmo_drn import DriftTuningConfig, parse_config
from hardware.devices.cmo import G_MAX_US, G_MIN_US, HALF_RANGE_US, MID_US, programming_sigma
from methods import drift_tuning as dt
from methods import fault_tuning as ft
from methods.cmo_direct_adam import ThresholdedBoundedWriteAdam
from methods.opt_recovery import RecoveryWritePort
from methods.programming.cmo import program_rewritten
from networks.opt.physical import CmoDrnBank
from tests.test_opt_cmo_drn import cpu_threads, tiny_model  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'examples' / 'opt_cmo_drn' / 'cmo_drift_recovery'
HISTORY = ROOT / 'examples' / 'opt_cmo_drn' / 'drift_token_budget'
PILOT = ROOT / 'campaigns' / 'pilots' / '20261118-cmo-drift-recovery'


def law(path=CONFIGS / 'full' / 'drn6-ratio3-m4.json', **changes):
    raw = json.loads(path.read_text())['drift_tuning']
    raw.update(changes)
    return dt.DriftTuning.parse(raw)


def released(seed=271001):
    model, teacher = tiny_model()
    dt.remap(model, 3.0)
    _, _, state = program_rewritten(model, seed, 'none', 0.0, capture_state=True)
    return model, teacher, state


def module(name):
    spec = importlib.util.spec_from_file_location(name, PILOT / f'{name}.py')
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_unwritten_devices_read_bit_identically_to_the_historical_law():
    d = law()
    held = torch.rand(40, 50) * 70 + 10
    for t in (0.0, 0.5, 60.0, 3600.0, 86400.0):
        historical = dt.drifted(held, 2, t, d)
        assert torch.equal(dt.drifted(held, 2, t, d, torch.zeros(held.shape, dtype=torch.int32)), historical)
    # A device written at t_w reads with tau = t - t_w on the same absolute-time draw.
    clock = torch.zeros(held.shape, dtype=torch.int32)
    clock[0, :5] = 60_000
    read = dt.drifted(held, 2, 3600.0, d, clock)
    assert torch.equal(read[1:], dt.drifted(held, 2, 3600.0, d)[1:])
    generator = torch.Generator().manual_seed(dt.draw_key(d.draw_seed, 2, 3600.0))
    y = torch.randn(held.shape, generator=generator)
    log_tau = torch.tensor(3540.0).log()
    own = held + 4 * d.mean_us * log_tau + 4 * (d.spread_slope_us * log_tau + d.spread_offset_us) * y
    assert torch.allclose(read[0, :5], own.clamp(5, 150)[0, :5], atol=1e-5)
    # At its own write time a device reads its held value; a read before it is rejected.
    assert torch.equal(dt.drifted(held, 2, 60.0, d, clock)[0, :5], held[0, :5])
    with pytest.raises(ValueError):
        dt.drifted(held, 2, 30.0, d, clock)


def test_forced_absorbing_write_takes_the_drifted_read_plus_step_plus_noise(cpu_threads):
    _, _, state = released()
    d = law()
    hold, bank = CmoDrnBank(state, 'cpu'), CmoDrnBank(state, 'cpu')
    clock = dt.WriteClock(bank)
    k, seconds = 1, 60.0
    array = bank.arrays[k]
    port = dt.AbsorbWritePort(array, 1.0, law=d, index=k, clock=clock)
    port.seconds = seconds
    before = array.held.clone()
    read = dt.drifted(before, k, seconds, d)  # The learner's read of this update (clock all zero).
    command = torch.zeros_like(before)
    command.view(-1)[::7] = 0.05
    command.view(-1)[3::11] = -0.08
    rng = array.generator.get_state()
    attempted = port.write(command)
    target = (read + HALF_RANGE_US * command).clamp(G_MIN_US, G_MAX_US)
    z = torch.randn(before.shape, generator=torch.Generator().manual_seed(0).set_state(rng))
    expected = target + array.noise_scale * programming_sigma(target, 2.0) * z
    written = command != 0
    assert torch.equal(attempted, written)
    assert torch.equal(array.held[written], expected[written])
    assert torch.equal(array.held[~written], before[~written])
    # Only the written devices restart their clock, at the write time.
    assert torch.equal(clock.write_ms[k] > 0, written)
    assert bool((clock.write_ms[k][written] == 60_000).all())
    assert all(int((w > 0).sum()) == 0 for j, w in enumerate(clock.write_ms) if j != k)
    # Later reads: unwritten devices equal the untreated bank exactly; written ones relax from t_w.
    for t in (61.5, 3600.0, 86400.0):
        mine = dt.drifted(array.held, k, t, d, clock.write_ms[k])
        assert torch.equal(mine[~written], dt.drifted(hold.arrays[k].held, k, t, d)[~written])
    # Written devices relax from their new held value with tau = t - t_w on the absolute-time draw.
    y = torch.randn(before.shape, generator=torch.Generator().manual_seed(dt.draw_key(d.draw_seed, k, 86400.0)))
    log_tau = torch.tensor(86400.0 - seconds).log()
    own = (array.held + 4 * d.mean_us * log_tau + 4 * (d.spread_slope_us * log_tau + d.spread_offset_us) * y).clamp(5, 150)
    mine = dt.drifted(array.held, k, 86400.0, d, clock.write_ms[k])
    assert torch.allclose(mine[written], own[written], atol=1e-5)


def test_continues_writes_are_unchanged_and_absorb_needs_a_read_time(cpu_threads):
    _, _, state = released()
    d = law(path=HISTORY / 'full' / 'drn6-ratio3-m4.json')
    assert type(d) is dt.DriftTuning and d.gate == 'target_rho' and d.off_labels == ()
    bank = CmoDrnBank(state, 'cpu')
    clock = dt.WriteClock(bank)
    port = dt.AbsorbWritePort(bank.arrays[0], 1.0, law=law(), index=0, clock=clock)
    with pytest.raises(ValueError):
        port.write(torch.full_like(bank.arrays[0].held, 0.1))


def test_fixed_gate_fires_exactly_at_two_sigma_mid_of_physical_conductance(cpu_threads):
    _, _, state = released()  # ratio3: the bank holds physical uS.
    array = CmoDrnBank(state, 'cpu').arrays[0]
    port = RecoveryWritePort(array, 1.0, target_bounds='gmin_gmax')
    writer = ThresholdedBoundedWriteAdam(port, learning_rate=0.0, gate='fixed_mid')
    gate_us = writer.threshold_q * HALF_RANGE_US
    assert gate_us == pytest.approx(2 * programming_sigma(MID_US, 2.0), rel=1e-12)
    assert gate_us == pytest.approx(1.11985, abs=1e-5)
    q = torch.tensor(writer.threshold_q, dtype=array.held.dtype)
    below = torch.nextafter(q, torch.tensor(0.0, dtype=q.dtype))
    pending = torch.zeros_like(array.held)
    flat = pending.view(-1)
    flat[0], flat[1], flat[2], flat[3], flat[4] = q, -q, below, -below, 2 * q
    writer.pending.copy_(pending)
    before = array.held.clone()
    writer.step(torch.zeros_like(array.held))
    fired = writer.attempt_counts.view(-1) > 0
    assert fired[:5].tolist() == [True, True, False, False, True] and int(fired.sum()) == 3
    assert torch.equal(array.held.view(-1)[5:], before.view(-1)[5:])
    assert float(writer.pending.view(-1)[2]) == float(below)
    # The default 2-rho gate keeps its historical contract.
    default = ThresholdedBoundedWriteAdam(RecoveryWritePort(array, 1.0, target_bounds='gmin_gmax'), learning_rate=1e-3)
    assert default.contract() == ['cmo_drn.bounded_thresholded_adam.v1', 1e-3, 2.0, 'bounded_target_write_std',
                                  'command_before_projection', 'ge']
    assert writer.contract()[3] == 'nominal_midrange_write_std'
    with pytest.raises(ValueError):
        ThresholdedBoundedWriteAdam(port, learning_rate=0.0, gate='two_rho')


def test_zero_rate_absorbing_learner_replays_a0_at_every_age(cpu_threads):
    model, _, state = released()
    d = law()
    hold, bank = CmoDrnBank(state, 'cpu'), CmoDrnBank(state, 'cpu')
    clock = dt.WriteClock(bank)
    arm = DriftPhysicalArm(model, bank, 'noisy', 0.0, 4, gate='fixed_mid', law=d, clock=clock)
    tokens, samples = torch.randint(0, 17, (4, 6)), torch.randint(0, 17, (4, 5, 8))
    for step in range(1, 5):
        arm.step_at(tokens, samples, 2, 'cpu', dt.transform(d, d.age(step), clock), d.age(step))
    assert arm.write_report()['costs']['write_attempts'] == 0 and clock.reset_devices() == 0
    arm.close()
    model.eval()
    with torch.no_grad():
        for t in d.evaluation_seconds:
            mine = model(tokens, bank.view(model, transform=dt.transform(d, t, clock)))
            assert torch.equal(mine, model(tokens, hold.view(model, transform=dt.transform(d, t))))


def test_absorbing_learner_resets_only_written_devices_and_masks_child_d(cpu_threads):
    model, _, state = released()
    d = law()
    hold, bank = CmoDrnBank(state, 'cpu'), CmoDrnBank(state, 'cpu')
    clock = dt.WriteClock(bank)
    masks = [torch.rand(a.held.shape, generator=torch.Generator().manual_seed(k)) < 0.3
             for k, a in enumerate(bank.arrays)]
    arm = DriftPhysicalArm(model, bank, 'child_d', 1.0, 4, masks, gate='fixed_mid', law=d, clock=clock)
    tokens, samples = torch.randint(0, 17, (4, 6)), torch.randint(0, 17, (4, 5, 8))
    for step in range(1, 5):
        arm.step_at(tokens, samples, 2, 'cpu', dt.transform(d, d.age(step), clock), d.age(step))
    report = arm.write_report()
    arm.close()
    assert report['costs']['write_attempts'] > 0
    assert clock.reset_devices() == report['touched_cells']
    for k, (a, h, m, w) in enumerate(zip(bank.arrays, hold.arrays, masks, arm.writers)):
        written = w.attempt_counts > 0
        assert not bool((written & ~m).any())
        assert torch.equal(clock.write_ms[k] > 0, written)
        assert torch.equal(a.held[~written], h.held[~written])
        assert torch.equal(dt.drifted(a.held, k, 86400.0, d, clock.write_ms[k])[~written],
                           dt.drifted(h.held, k, 86400.0, d)[~written])


def test_layernorm_and_calibration_arms_read_the_drifted_state_and_leave_the_model_unchanged(cpu_threads):
    model, _, state = released()
    d = law()
    bank = CmoDrnBank(state, 'cpu')
    digest = stage.model_digest(model)
    tokens, samples = torch.randint(0, 17, (4, 6)), torch.randint(0, 17, (4, 5, 8))
    hashes = bank.held_hashes()
    for make in (lambda: ft.LayerNormArm(model, bank, 1e-2, 3), lambda: ft.CalibrationArm(model, bank, 1e-2, 3)):
        learner = make()
        with torch.no_grad():
            start = model.eval()(tokens, bank.view(model))
        for step in range(1, 4):
            learner.step(tokens, samples, 2, 'cpu', transform=dt.transform(d, d.age(step)))
        with torch.no_grad():
            assert not torch.equal(model.eval()(tokens, bank.view(model)), start)
        learner.close()
        assert stage.model_digest(model) == digest
        with torch.no_grad():
            assert torch.equal(model.eval()(tokens, bank.view(model)), start)
    assert bank.held_hashes() == hashes and bank.costs()['write_attempts'] == 0
    norms = ft.analog_layer_norms(model)
    assert len(norms) == len(model.blocks)


def test_generated_configs_are_current_and_differ_from_exp005_only_where_declared():
    make = module('make_configs')
    seen = set()
    for model, multiplier, kind in make.cases():
        path = CONFIGS / kind / (make.name(model, multiplier, kind) + '.json')
        raw = json.loads(path.read_text())
        assert raw == make.build(model, multiplier, kind)
        seen.add((model, multiplier, kind))
        c = parse_config(raw)
        assert isinstance(c, DriftTuningConfig) and isinstance(c.drift_tuning, dt.CampaignDriftTuning)
        d = c.drift_tuning
        assert (d.multiplier, d.write_clock, d.gate, d.mapping_scale) == (multiplier, 'reset_absorb', 'fixed_mid', 3.0)
        labels = {a.label: (a.method, a.learning_rate) for a in c.fault_tuning.arms}
        expected = {'A0': ('hold', None), 'A0-twin': ('hold', None), 'CAL': ('calibration', 1e-2),
                    'LN': ('layernorm', 1e-2), 'CT': ('child_d', 1e-3), 'L1': ('noisy', 3e-4)}
        if multiplier == 4.0:
            expected['CT-nodrift'] = ('child_d', 1e-3)
        assert labels == expected
        assert set(d.drift_off_arms) == {'A0-twin'} | ({'CT-nodrift'} if multiplier == 4.0 else set())
        history = json.loads((HISTORY / kind / (make.name(model, 4.0, kind) + '.json')).read_text())
        for key in raw:
            if key not in ('fault_tuning', 'drift_tuning', 'training'):
                assert raw[key] == history[key], key
        f, g = dict(raw['fault_tuning']), dict(history['fault_tuning'])
        f.pop('arms'), g.pop('arms')
        assert f == g
        a, b = dict(raw['drift_tuning']), dict(history['drift_tuning'])
        for x in (a, b):
            for key in ('multiplier', 'write_clock', 'projection_steps', 'write_gate', 'drift_off_arms'):
                x.pop(key, None)
        assert a == b
        assert d.projection_steps == ((300, 600, 1200, 2400) if kind == 'full' else (16, 32))
        assert c.fault_tuning.steps == (2400 if kind == 'full' else 32)
    assert len(seen) == 8


def test_invalid_campaign_configs_are_rejected():
    raw = json.loads((CONFIGS / 'full' / 'drn6-ratio3-m4.json').read_text())

    def broken(edit):
        doc = json.loads(json.dumps(raw))
        edit(doc)
        with pytest.raises(ConfigError):
            parse_config(doc)

    broken(lambda d: d['drift_tuning'].update(write_gate='two_rho'))
    broken(lambda d: d['drift_tuning'].update(drift_off_arms=['missing']))
    broken(lambda d: d['drift_tuning'].update(drift_off_arms=['A0', 'A0-twin']))  # No drifting A0 left.
    broken(lambda d: d['drift_tuning'].update(drift_off_arms=['CAL']))
    broken(lambda d: d['drift_tuning'].pop('drift_off_arms'))
    broken(lambda d: d['fault_tuning']['arms'].append({'label': 'R', 'method': 'lora', 'learning_rate': 1e-3,
                                                       'rank': 16}))
    # reset_absorb is only accepted in the explicit campaign form.
    history = json.loads((HISTORY / 'full' / 'drn6-ratio3-m4.json').read_text())
    history['drift_tuning']['write_clock'] = 'reset_absorb'
    with pytest.raises(ConfigError):
        parse_config(history)
    # LayerNorm adaptation runs only in the drift stage.
    fault = json.loads((ROOT / 'examples/opt_cmo_drn/fault_token_budget/full/drn6-n8192.json').read_text())
    fault['fault_tuning']['arms'].append({'label': 'LN', 'method': 'layernorm', 'learning_rate': 1e-2})
    with pytest.raises(ConfigError):
        parse_config(fault)


def test_stage_runs_every_arm_end_to_end_and_passes_verify_unit(cpu_threads, tmp_path, monkeypatch):
    """Tiny CPU run of the native stage with the campaign arm set; then the bundle check."""
    raw = json.loads((CONFIGS / 'canary' / 'drn6-ratio3-m4-canary.json').read_text())
    f, d = raw['fault_tuning'], raw['drift_tuning']
    f.update(recovery_sequences=16, fisher_batch=8, batch=2, microbatch=1, steps=6, development_every=3)
    for a in f['arms']:
        if a.get('method') in ('noisy', 'child_d'):
            a['learning_rate'] = 1.0  # Large enough to write on the tiny model.
    d.update(evaluation_seconds=[0.0, 3.0, 9.0, 60.0, 3600.0], projection_seconds=3600.0, projection_steps=[2, 6])
    raw.update(device='cpu', cpu_threads=1)
    config = parse_config(raw)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(raw))
    model, teacher = tiny_model()
    generator = torch.Generator().manual_seed(3)
    recovery = torch.randint(0, 17, (16, 6), generator=generator)
    development = torch.randint(0, 17, (3, 6), generator=generator)
    samples = torch.randint(0, 17, (16, 5, 8), generator=generator)
    for name in ('teacher.bin', 'weights.pt', 'samples.pt'):
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(stage, 'cohorts', lambda s: (recovery, development))
    monkeypatch.setattr(stage, 'load_samples', lambda s, r: (samples, tmp_path / 'samples.pt'))
    monkeypatch.setattr(stage, 'sha256_file', lambda p: 'dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9')
    monkeypatch.setattr(runtime, 'teacher_model', lambda path, c: teacher)
    monkeypatch.setattr(runtime, 'from_mapped', lambda t, c, payload: model)
    monkeypatch.setattr(stage.torch, 'load', lambda *args, **kwargs: {})
    monkeypatch.delenv('OPT_DRN_SOURCE_SNAPSHOT', raising=False)
    monkeypatch.setenv('OPT_DRN_RUN_ID', 'run')
    request = SimpleNamespace(spec=config, teacher_weights=tmp_path / 'teacher.bin', weights=tmp_path / 'weights.pt',
                              output_dir=tmp_path / 'out', command=['ebl', 'train'], resume=None, base_weights=None,
                              device_data=None, device_model=None, deployment=None, device_state=None)
    assert stage.run(request) == 0
    run_dir = tmp_path / 'out' / 'run'
    summary = json.loads((run_dir / 'artifacts/summary.json').read_text())
    arms = {a['label']: a for a in summary['arms']}
    assert list(arms) == ['A0', 'A0-twin', 'CAL', 'LN', 'CT', 'L1', 'CT-nodrift']
    assert arms['CT']['costs']['write_attempts'] > 0 and arms['L1']['costs']['write_attempts'] > 0
    assert arms['CT']['write_clock']['reset_devices'] == arms['CT']['touched_cells']
    assert 'write_clock' not in arms['CT-nodrift'] and arms['CT-nodrift']['drift'] is False
    assert arms['CT-nodrift']['grid'][0]['teacher_kl'] == arms['A0-twin']['grid'][0]['teacher_kl']
    assert arms['CT']['mask_sha256'] == arms['CT-nodrift']['mask_sha256'] == summary['child_d_fisher']['mask_sha256']
    assert summary['identity']['write_gate_us'] == pytest.approx(1.11985, abs=1e-5)
    verdict = module('verify_unit').verify(config_path, run_dir)
    assert verdict['verified']
