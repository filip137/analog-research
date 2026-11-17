"""Deployment-time crossbar mappings (training.xbar_mapping) and their drift-regimes wiring."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from training import xbar_mapping as xm
from training.drift_regimes_config import DriftRegimes
from test_workflow_drift_regimes import config, ADAPT, _stage_inputs, _run
from test_workflow_cmo import cmo_inputs  # noqa: F401  (fixture)


def toy(scaling='per_output_channel_absmax', seed=0):
    """A network-like object: two matrices, |q| <= 1 with some weights at the clip."""
    g = torch.Generator().manual_seed(seed)
    shapes = [(6, 40), (5, 30)]
    layout, parts, offset = [], [], 0
    for i, shape in enumerate(shapes):
        q = (torch.randn(shape, generator=g) * .5).clamp(-1, 1)
        q[:, 0] = 1.
        layout.append(SimpleNamespace(name=f'm{i}', shape=shape, offset=offset, size=shape[0] * shape[1]))
        parts.append(q.flatten())
        offset += q.numel()
    channel = torch.rand(sum(s[0] for s in shapes), generator=g) + .5
    return SimpleNamespace(layout=layout, base_matrices=len(layout), q=torch.cat(parts), scaling=scaling,
                           channel_scales=channel, channel_offsets=[0, 6, 11],
                           scales=torch.tensor([.7, 1.3]))


def spec(kind='reference', rule='unit', margin=0., **gain):
    return {'kind': kind, 'gain': {'rule': rule, **gain}, 'margin_us': margin}


@pytest.mark.parametrize('raw', [
    {'kind': 'reference', 'gain': {'rule': 'unit'}},
    spec(kind='ratio'), spec(rule='percentile'), spec(rule='fixed'), spec(rule='fixed', value=0.),
    spec(rule='saturation', fraction=.6), spec(rule='unit', fraction=.1), spec(margin=4.),
    spec(kind='two_sided', margin=80.), spec(kind='two_sided', margin=-1.), dict(spec(), extra=1)])
def test_settings_fail_closed(raw):
    with pytest.raises(ValueError):
        xm.parse(raw)


def test_base_is_the_series_002_pair_mapping_bit_for_bit():
    net = toy()
    m = xm.Mapping(xm.BASE, net)
    assert m.is_base and m.gains is None
    for k, spec_ in enumerate(net.layout):
        q = net.q[spec_.offset:spec_.offset + spec_.size]
        assert torch.equal(m.targets(net.q)[k], torch.stack((48.6 + 39.6 * q, torch.full_like(q, 48.6))))
    physical = [t + torch.randn(t.shape) for t in m.targets(net.q)]
    assert torch.equal(m.bind(physical), torch.cat([(g[0] - g[1]) / 39.6 for g in physical]))
    assert torch.equal(m.clean(net.q), net.q)
    report = m.report(net.q, net)
    assert report['saturated_fraction'] == 0 and report['clean_relative_weight_error'] == 0


@pytest.mark.parametrize('raw', [spec(), spec(rule='fixed', value=1.5), spec(rule='saturation', fraction=.05),
                                 spec(rule='per_layer_absmax'), spec('push_pull'),
                                 spec('push_pull', rule='fixed', value=1.25), spec('two_sided'),
                                 spec('two_sided', margin=8.), spec('two_sided', 'saturation', 8., fraction=.05)])
@pytest.mark.parametrize('scaling', ['per_output_channel_absmax', 'per_layer_absmax'])
def test_every_mapping_programs_in_window_reads_back_clean_and_cancels_common_mode(raw, scaling):
    net = toy(scaling)
    m = xm.Mapping(raw, net)
    targets = m.targets(net.q)
    assert all(bool(((t >= 9 - 1e-5) & (t <= 88.2 + 1e-5)).all()) for t in targets)
    clean = m.clean(net.q)
    assert torch.allclose(m.bind(targets), clean, atol=3e-7, rtol=1e-6)
    assert bool((clean.abs() <= net.q.abs() + 1e-7).all())                # saturation only shrinks
    shifted = [t - 3.7 for t in targets]                                      # common-mode drift
    assert torch.allclose(m.bind(shifted), m.bind(targets), atol=2e-6)
    if raw['kind'] == 'two_sided':
        zero_side = torch.cat([torch.where(t[0] <= t[1], t[0], t[1]) for t in targets])
        assert torch.allclose(zero_side, torch.full_like(zero_side, 9 + raw['margin_us']))
    # A device-level error e on device 0 reads as e / (span s): the span is the drift lever.
    s = 1. if m.gains is None else torch.cat(m.gains).repeat_interleave(
        torch.tensor([sp.shape[1] for sp in net.layout]).repeat_interleave(torch.tensor([sp.shape[0] for sp in net.layout])))
    bumped = [torch.stack((t[0] + 1., t[1])) for t in targets]
    assert torch.allclose(m.bind(bumped) - m.bind(targets), 1 / (m.span * s) * torch.ones_like(clean), atol=1e-5)


def test_gain_rules():
    net = toy()
    sat = xm.Mapping(spec(rule='saturation', fraction=.05), net)
    for k, sp in enumerate(net.layout):
        rows, cols = sp.shape
        a = net.q[sp.offset:sp.offset + sp.size].reshape(rows, cols).abs()
        s = sat.gains[k]
        assert bool((s >= 1).all())
        assert bool(((a * s[:, None] > 1 + 1e-6).sum(1) <= int(.05 * cols)).all())
    layer = xm.Mapping(spec(rule='per_layer_absmax'), net)
    scales = xm.digital_scales(net)
    for k, sp in enumerate(net.layout):
        rows, cols = sp.shape
        a = net.q[sp.offset:sp.offset + sp.size].reshape(rows, cols).abs()
        u = a * layer.gains[k][:, None]
        assert float(u.max()) == pytest.approx(1, abs=1e-6) and bool((layer.gains[k] <= 1 + 1e-6).all())
        # Effective weights keep their ratios: u is proportional to the effective weight c |q|.
        assert torch.allclose(u, a * scales[k][:, None] / float((a * scales[k][:, None]).max()), atol=1e-6)
    fixed = xm.Mapping(spec(rule='fixed', value=2.), net)
    report = fixed.report(net.q, net)
    assert report['saturated_fraction'] == pytest.approx(float((net.q.abs() > .5).float().mean()))
    assert report['clean_relative_weight_error'] > 0


def test_settings_accept_a_mapping_round_trip_and_old_files_are_unchanged():
    old = config().to_dict()
    assert 'mapping' not in old and DriftRegimes.parse(old).mapping is None
    raw = dict(old, mapping=spec('two_sided', margin=8))
    parsed = DriftRegimes.parse(json.loads(json.dumps(raw)))
    assert parsed.mapping == {'kind': 'two_sided', 'gain': {'rule': 'unit'}, 'margin_us': 8.}
    assert DriftRegimes.parse(parsed.to_dict()) == parsed
    with pytest.raises(ValueError):
        DriftRegimes.parse(dict(old, mapping=spec(kind='ratio')))
    adaptation = {**ADAPT, 'corpus_sha256': 'b' * 64, 'cohort_sha256': 'd' * 64}
    calibration = {'label': 'A1e', 'method': 'calibration', 'events': (60., 900.)}
    assert DriftRegimes.parse(dict(raw, arms=[{'label': 'A0', 'method': 'none'}, calibration], adaptation=adaptation))
    for arm in ({'label': 'A9', 'method': 'rewrite', 'events': (900.,), 'fraction': .1, 'attempts': 4, 'threshold': .7},
                {'label': 'A2', 'method': 'layernorm', 'events': (60.,), 'learning_rate': 1e-3}):
        with pytest.raises(ValueError):
            DriftRegimes.parse(dict(raw, arms=[arm], adaptation=adaptation))


def test_native_stage_explicit_base_replays_and_mappings_run(cmo_inputs, tmp_path):
    from test_drift_family import CMO_LOG
    from dataclasses import asdict
    from training.drift_family import ReramFamily
    lc, results, teacher, base, adaptation = _stage_inputs(cmo_inputs, tmp_path)
    arm_list = ({'label': 'A0', 'method': 'none'}, {'label': 'A0-twin', 'method': 'none', 'twin': True},
                {'label': 'A1e', 'method': 'calibration', 'events': (60., 900.)})
    common = replace(base, law=asdict(ReramFamily(**CMO_LOG, drift_min=5.)), arms=arm_list, adaptation=adaptation)
    _, implicit = _run(lc, DriftRegimes.parse(common.to_dict()), results, teacher, tmp_path / 'implicit')
    explicit_dir, explicit = _run(lc, DriftRegimes.parse(dict(common.to_dict(), mapping=xm.BASE)), results, teacher,
                                  tmp_path / 'explicit')
    # The explicit series-002 mapping replays the mapping-free run exactly.
    for label in ('A0', 'A0-twin', 'A1e'):
        assert explicit['arms'][label]['final_physical_sha256'] == implicit['arms'][label]['final_physical_sha256']
        assert explicit['arms'][label]['mean_teacher_kl'] == implicit['arms'][label]['mean_teacher_kl']
    assert explicit['mapping']['saturated_fraction'] == 0
    references = json.loads((explicit_dir / 'drift_regimes_references.json').read_text())
    assert references['clean_mapped']['teacher_kl'] == pytest.approx(references['clean_hwa']['teacher_kl'], abs=1e-9)
    for raw in (spec('push_pull'), spec('two_sided', margin=8.), spec(rule='fixed', value=1.5)):
        directory, metrics = _run(lc, DriftRegimes.parse(dict(common.to_dict(), mapping=raw)), results, teacher,
                                  tmp_path / f"{raw['kind']}-{raw['margin_us']}-{raw['gain']['rule']}")
        assert metrics['source_unchanged'] and metrics['replay_exact']
        assert metrics['mapping']['mapping'] == xm.parse(raw)
        rows = json.loads((directory / 'drift_regimes_measurements.json').read_text())
        assert all(r['mapping']['mapping'] == xm.parse(raw) for r in rows)
        references = json.loads((directory / 'drift_regimes_references.json').read_text())
        assert 'clean_mapped' in references
        from workflow.drift_regimes import flat_rows
        flat = flat_rows(rows, directory)
        assert flat[0]['mapping_kind'] == raw['kind'] and 'mapping' not in flat[0]
