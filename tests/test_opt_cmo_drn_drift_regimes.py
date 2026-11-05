"""exp-003 drift-regimes screen on the OPT CMO DRN: mapping, numerical contracts and a tiny end-to-end run."""
import copy
from types import SimpleNamespace

import pytest
import torch

from ebl.registry import EXPERIMENT_REGISTRY
from ebl.schema.base import ConfigError
from ebl.schema.opt_cmo_drn_drift_regimes import parse_config
from hardware.devices.cmo import G_MAX_US
from hardware.drift.family import ReramDriftCells, ReramFamily
from methods.config.drift_regimes import DriftRegimes, ForwardPolicy
from methods.drift_regimes import arms, forward
import ebl.runtime.opt_cmo_drn.drift_regimes as stage_module
from test_opt_cmo_drn import tiny_model, cpu_threads  # noqa: F401

POLICY = ForwardPolicy('settle4_grad4_eval8.v1', 4, 4, 8)
LAW = dict(rho_star=.335, persistent_share=0., cmo_scale=1., cmo_acceptance=2., amplitude_sigma=0.,
           mean_us=-2., spread_us=3., mean_exponent=.23, spread_exponent=.11, reference_seconds=3600.,
           program_min=9., program_max=88.2, drift_min=5., drift_max=150., time_dtype='float64')
EVENTS = [900., 3600., 14400., 57600.]
ARMS = [
    {'label': 'A0', 'method': 'none'},
    {'label': 'A1', 'method': 'calibration', 'events': EVENTS},
    {'label': 'A3', 'method': 'layernorm', 'events': EVENTS, 'learning_rate': .003},
    {'label': 'A8-N4', 'method': 'rewrite', 'events': EVENTS, 'fraction': 1., 'attempts': 4, 'threshold': .7431},
    {'label': 'A9-W2-N4', 'method': 'rewrite', 'events': EVENTS, 'fraction': .1, 'attempts': 4, 'threshold': .7431},
    {'label': 'A10-W2-N4', 'method': 'precompensated', 'events': EVENTS, 'fraction': .1, 'attempts': 4,
     'threshold': .7431},
    {'label': 'A11-W2-N4', 'method': 'learning_newton', 'events': EVENTS, 'fraction': .1, 'attempts': 4,
     'threshold': .7431, 'rounds': 4, 'batch_sequences': 2, 'score': 'fisher', 'ideal': False},
    {'label': 'A0-twin', 'method': 'none', 'twin': True},
    {'label': 'A11-W2-twin', 'method': 'learning_newton', 'events': EVENTS, 'fraction': .1, 'attempts': 1,
     'threshold': None, 'rounds': 4, 'batch_sequences': 2, 'score': 'fisher', 'ideal': False, 'twin': True},
]


def settings_raw(**changes):
    raw = {'regime': 'R0', 'seed': 391103, 'realization': 271001, 'source_sha256': 'a' * 64,
           'corpus': 'wikitext-2-raw-v1', 'corpus_sha256': 'b' * 64, 'sequence_length': 6, 'law': dict(LAW),
           'deployment_attempts': 4, 'deployment_threshold': .7431, 'verify_seconds': 60.,
           'evaluation_seconds': [0., 60., 900., 1800., 3600., 7200., 14400., 28800., 57600., 86400.],
           'evaluation': {'split': 'validation', 'start': 0, 'sequences': 2, 'sha256': 'c' * 64},
           'evaluation_microbatch': 1,
           'adaptation': {'split': 'train', 'seed': 20261104, 'sequences': 4, 'probe_sequences': 2, 'steps': 3,
                          'batch': 2, 'microbatch': 1, 'sha256': 'd' * 64},
           'forward_policy': {'name': 'settle4_grad4_eval8.v1', 'settle_sweeps': 4, 'gradient_sweeps': 4,
                              'evaluation_sweeps': 8},
           'arms': copy.deepcopy(ARMS), 'canary': False}
    raw.update(changes)
    return raw


def test_config_is_strict_and_registered():
    assert 'opt_cmo_drn_drift_regimes.v1' in EXPERIMENT_REGISTRY
    raw = {'experiment_id': 'opt_cmo_drn_drift_regimes.v1', 'schema_version': 1, 'stage': 'drift_regimes',
           'device': 'cpu', 'cpu_threads': 1, 'drift_regimes': settings_raw()}
    config = parse_config(raw)
    assert config.drift_regimes.arms[-1]['twin'] and config.drift_regimes.forward_policy.evaluation_sweeps == 8
    ReramFamily(**config.drift_regimes.law)
    for bad in ({'label': 'A5', 'method': 'lora', 'events': EVENTS, 'rank': 4},
                {**ARMS[6], 'score': 'newton'}, {**ARMS[6], 'ideal': True},
                {**ARMS[8], 'attempts': 4, 'threshold': .7}, {**ARMS[1], 'unknown': 1}):
        with pytest.raises(ConfigError):
            parse_config({**raw, 'drift_regimes': settings_raw(arms=[bad])})
    with pytest.raises(ConfigError):
        parse_config({**raw, 'drift_regimes': settings_raw(forward_policy={
            'name': 'x', 'settle_sweeps': 4, 'gradient_sweeps': 4, 'evaluation_sweeps': 4})})
    with pytest.raises(ConfigError):
        parse_config({**raw, 'solver': {'settling_iterations': 8}})


def test_four_device_groups_hold_one_logical_weight():
    model, _ = tiny_model()
    targets, _, roles = stage_module.physical_layout(model)
    for j, (target, role) in enumerate(zip(targets, roles)):
        paired = j % 2 == 1
        groups = forward.group_view(target, paired)
        # Each logical weight: one positive and one negative rail pair; p/n devices sum to a constant pair.
        assert torch.allclose(groups[0], groups[3]) and torch.allclose(groups[1], groups[2])
        assert torch.allclose(torch.minimum(groups[0], groups[1]), torch.full_like(groups[0], 9.), atol=1e-4)
        assert sorted(set(forward.group_view(role, paired).reshape(4, -1)[:, 0].tolist())) == [4 * (j % 2) + r
                                                                                              for r in range(4)]
        members = forward.group_members(target.shape, paired, torch.arange(target.numel() // 4), 'cpu')
        assert torch.equal(members.sort().values, torch.arange(target.numel()))


def test_fisher_score_reduces_to_the_crossbar_differential_rule():
    torch.manual_seed(3)
    inputs, outputs, probes = 3, 5, 7
    shape = (2 * inputs, 2 * outputs)
    error, variance = torch.randn(shape) * 2, torch.rand(shape) + .1
    quadratic = torch.zeros(inputs, outputs, dtype=torch.float64)
    trace = torch.zeros_like(quadratic)
    h = torch.zeros(inputs, outputs, dtype=torch.float64)
    mean = -1.3
    for _ in range(probes):
        gw = torch.randn(inputs, outputs)
        stacked = torch.zeros(4, inputs, outputs)     # a two-device differential map inside each group
        stacked[0], stacked[1] = gw / 39.6, -gw / 39.6
        g = torch.zeros(shape)
        members = forward.group_members(shape, False, torch.arange(inputs * outputs), 'cpu')
        g.view(-1)[members] = stacked.reshape(4, -1).T.reshape(-1)
        assert torch.equal(forward.group_view(g, False), stacked)
        forward.accumulate_group_fisher(g, error, variance, mean, False, quadratic, trace)
        h += gw.double().square()
    e, v = forward.group_view(error, False), forward.group_view(variance, False)
    ew = (e[0] - e[1]).double() / 39.6
    vpair = (v[0] + v[1]).double() / 39.6 ** 2
    torch.testing.assert_close(quadratic - trace, h * (ew.square() - vpair))


def drift_cells(model, law=LAW):
    targets, names, _ = stage_module.physical_layout(model)
    return [ReramDriftCells(t, law=ReramFamily(**law), seed='391103/1', name=n) for t, n in zip(targets, names)]


def test_cached_prefix_and_forward_policy_match_the_full_forward():
    model, teacher = tiny_model()
    model.requires_grad_(False)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    policy = forward.Policy(model, POLICY)
    cells = drift_cells(model)
    physical = [c.advance(3600.) for c in cells]
    view = forward.bind(physical)
    policy.mode(False)
    with torch.no_grad():
        full = model(tokens, view)
        prefix = forward.prefix_states(model, teacher, tokens, 'cpu')
        cached = forward.suffix_logits(model, prefix.student, view)
        torch.testing.assert_close(forward.teacher_logits(teacher, 6, prefix.teacher), teacher(tokens)[:, :-1])
    assert torch.equal(full, cached)
    assert all(b.solver.settings.settling_iterations == 8 and not b.solver.training for b in model.blocks)
    held = [p.clone().requires_grad_(True) for p in physical]
    policy.mode(True)
    with torch.enable_grad():
        trained = forward.suffix_logits(model, prefix.student, forward.bind(held))
        trained.square().mean().backward()
    policy.mode(False)
    torch.testing.assert_close(trained.detach(), cached, atol=1e-6, rtol=1e-6)
    assert all(bool(torch.isfinite(p.grad).all()) for p in held) and any(bool(p.grad.abs().sum() > 0) for p in held)


def test_calibration_is_identity_on_the_clean_state():
    model, teacher = tiny_model()
    model.requires_grad_(False)
    policy = forward.Policy(model, POLICY)
    tokens = torch.randint(0, 17, (3, 6), generator=torch.Generator().manual_seed(1))
    probe = forward.prefix_states(model, teacher, tokens, 'cpu')
    targets, _, _ = stage_module.physical_layout(model)
    correction = forward.fit_calibration(model, forward.bind(targets), probe, policy, batch=2)
    for a, b in correction:
        torch.testing.assert_close(a, torch.ones_like(a), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(b, torch.zeros_like(b), atol=1e-4, rtol=1e-4)


class Store:
    def __init__(self, path):
        self.run_dir, self.metrics = path, []

    def append_metric(self, row):
        self.metrics.append(row)


def tiny_stage(tmp_path, law=LAW, arms_spec=ARMS):
    model, teacher = tiny_model()
    model.requires_grad_(False)
    settings = DriftRegimes.parse(settings_raw(law=dict(law), arms=copy.deepcopy(arms_spec)))
    stage = stage_module.Stage.__new__(stage_module.Stage)
    stage.config = SimpleNamespace(stage='drift_regimes')
    stage.settings, stage.device, stage.model, stage.teacher = settings, 'cpu', model, teacher
    stage.policy = forward.Policy(model, settings.forward_policy)
    g = torch.Generator().manual_seed(5)
    stage.evaluation_tokens = torch.randint(0, 17, (2, 6), generator=g)
    stage.adaptation_tokens = torch.randint(0, 17, (4, 6), generator=g)
    stage.law = ReramFamily(**settings.law)
    stage.store = Store(tmp_path)
    stage.identity = {'regime': 'R0', 'realization': 271001}
    return stage


@pytest.mark.parametrize('law', [LAW, {**LAW, 'rho_star': 1., 'persistent_share': 1.}])
def test_tiny_screen_runs_every_arm_with_pairing_and_reuse(tmp_path, law):
    stage = tiny_stage(tmp_path, law)
    report, artifacts = stage_module.run_screen(stage)
    assert report['replay_exact'] and report['source_unchanged'] and report['snapshots'] == 10 * len(ARMS)
    assert report['evaluations_computed'] < report['snapshots']
    import json
    rows = json.loads((tmp_path / 'drift_regimes_measurements.json').read_text())
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r['arm'], []).append(r)
    # Paired release, identical untreated trajectories before the first event.
    release = {r[0]['physical_sha256'] for r in by_arm.values()}
    assert len(release) == 1
    a0 = [r['physical_sha256'] for r in by_arm['A0']]
    assert [r['physical_sha256'] for r in by_arm['A1']] == a0
    assert len({r['physical_sha256'] for r in by_arm['A0-twin']}) == 1
    assert by_arm['A0'][-1]['metrics']['teacher_kl'] != by_arm['A0-twin'][-1]['metrics']['teacher_kl']
    summary = report['arms']
    assert summary['A8-N4']['cost']['selected_devices'] == 4 * report['cost']['physical_devices']
    assert summary['A11-W2-N4']['cost']['selected_devices'] % 4 == 0
    assert summary['A11-W2-N4']['cost']['fisher_probe_sequences'] == 4 * 4 * 2
    assert summary['A3']['cost']['optimizer_steps'] == 4 * 3
    assert summary['A11-W2-twin']['cost']['verify_reads'] == 0
    events = json.loads((tmp_path / 'drift_regimes_events.json').read_text())
    a10 = [e for e in events if e['arm'] == 'A10-W2-N4']
    assert a10[0]['precompensation'] == 0 and all(0 <= e['precompensation'] <= 1 for e in a10)
    # LayerNorm state is restored after the A3 arm.
    for norm in forward.layer_norms(stage.model):
        assert not any(p.requires_grad for p in norm.parameters())


def test_tiny_readiness_reports_checks_and_timings(tmp_path):
    stage = tiny_stage(tmp_path)
    stage.config = SimpleNamespace(stage='readiness')
    report, _ = stage_module.run_readiness(stage)
    assert len(report['checks']) == 3 and all(c['replay_exact'] for c in report['checks'])
    assert all(c['finite_physical_gradients'] and c['finite_layernorm_gradients'] for c in report['checks'])
    assert {'fisher_round_16_sequences', 'rewrite_full_event', 'evaluation_full_cohort'} <= set(report['timing'])
