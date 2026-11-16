"""Pinned endpoint-only scoring on live physical, LoRA and calibration learners."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from core.artifacts import sha256_file
from core.serialization import save
from ebl.runtime import fault_tuning as runtime
from ebl.schema.lifecycle import parse_lifecycle
from hardware.arrays import CmoArray
from methods import fault_tuning as ft
import networks.opt as opt
from tests.test_workflow_opt import tiny_teacher

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'campaigns/fault-recovery-paper/lifecycles/opt125m-cmo-l6-fault-paper-tune-deploy.json'


def specification(**changes):
    return {'name': 'held-test', 'corpus': 'tiny-held', 'corpus_sha256': '1' * 64,
            'split': 'test', 'start': 0, 'sequences': 2, 'cohort_sha256': '2' * 64, **changes}


def test_schema_roundtrip_and_legacy_identity():
    doc = json.loads(DOC.read_text())
    before = parse_lifecycle(doc)
    empty = copy.deepcopy(doc)
    empty['fault_tuning']['evaluation_cohorts'] = []
    assert parse_lifecycle(empty).digest() == before.digest()
    assert 'evaluation_cohorts' not in before.fault_tuning.to_dict()
    doc['fault_tuning']['evaluation_cohorts'] = [specification()]
    parsed = parse_lifecycle(doc)
    assert parsed.digest() != before.digest()
    assert parsed.fault_tuning.to_dict()['evaluation_cohorts'] == [specification()]
    assert parse_lifecycle(parsed.to_dict()).digest() == parsed.digest()


@pytest.mark.parametrize('changes', [
    {'seed': 3}, {'name': '../test'}, {'corpus': '../tokens'}, {'split': 'other'},
    {'start': -1}, {'start': True}, {'sequences': 0}, {'cohort_sha256': 'bad'},
    {'corpus_sha256': 'bad'},
])
def test_strict_cohort_fields(changes):
    with pytest.raises(ValueError):
        ft.parse_evaluation_cohorts([specification(**changes)])


def test_reject_duplicate_names_and_selection_overlap():
    with pytest.raises(ValueError, match='distinct'):
        ft.parse_evaluation_cohorts([specification(), specification()])
    doc = json.loads(DOC.read_text())
    for split in ('train', 'validation'):
        doc['fault_tuning']['evaluation_cohorts'] = [specification(corpus='wikitext-2-raw-v1', split=split)]
        with pytest.raises(Exception, match='separate recovery|overlap'):
            parse_lifecycle(doc)
    doc['fault_tuning']['evaluation_cohorts'] = [specification(
        corpus='renamed-recovery', corpus_sha256=doc['fault_tuning']['recovery']['corpus_sha256'], split='train')]
    with pytest.raises(Exception, match='separate recovery'):
        parse_lifecycle(doc)


def test_held_out_drift_combination_is_rejected_instead_of_ignored():
    doc = json.loads((ROOT / 'campaigns/pilots/20261116-drift-token-budget/lifecycles/full/drift-token-budget-20261116-xb6-m4.json').read_text())
    doc['fault_tuning']['evaluation_cohorts'] = [specification()]
    with pytest.raises(Exception, match='without drift'):
        parse_lifecycle(doc)


@pytest.mark.parametrize('defect', ['corpus', 'cohort', 'short'])
def test_preflight_rejects_bad_pins_before_any_arm(defect, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    lc = parse_lifecycle(json.loads(DOC.read_text()))
    stream = torch.arange(18, dtype=torch.int64)
    corpus = {'schema': opt.CORPUS_SCHEMA, 'model': lc.network.model,
              'splits': {key: stream for key in ('train', 'validation', 'test')}}
    path = tmp_path / 'tiny-held.pt'
    save(path, corpus)
    spec = specification(corpus_sha256=sha256_file(path), cohort_sha256=ft.raw_hash(stream[:12].reshape(2, 6)))
    if defect == 'corpus': spec['corpus_sha256'] = '0' * 64
    if defect == 'cohort': spec['cohort_sha256'] = '0' * 64
    if defect == 'short': spec['sequences'] = 4
    lc = replace(lc, data=replace(lc.data, sequence_length=6))
    events = []
    ctx = NS(lc=lc, cfg=NS(evaluation_cohorts=[spec]), stage=NS(store=NS(append_metric=events.append)))
    monkeypatch.setenv('EBL_CORPUS_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='corpus|cohort'):
        runtime.prepare_evaluation_cohorts(ctx)
    assert not events and ctx.evaluation_cohorts == []


def test_preflight_uses_contiguous_blocks_and_registers_native_input(tmp_path, monkeypatch):
    lc = parse_lifecycle(json.loads(DOC.read_text()))
    stream = torch.arange(24, dtype=torch.int64)
    path = tmp_path / 'tiny-held.pt'
    save(path, {'schema': opt.CORPUS_SCHEMA, 'model': lc.network.model,
                'splits': {key: stream for key in ('train', 'validation', 'test')}})
    expected = stream[6:18].reshape(2, 6)
    spec = specification(start=1, corpus_sha256=sha256_file(path), cohort_sha256=ft.raw_hash(expected))
    lc = replace(lc, data=replace(lc.data, sequence_length=6),
                 fault_tuning=replace(lc.fault_tuning, evaluation_cohorts=(spec,)))
    events = []
    ctx = NS(lc=lc, cfg=lc.fault_tuning, stage=NS(store=NS(append_metric=events.append)))
    monkeypatch.setenv('EBL_CORPUS_ROOT', str(tmp_path))
    runtime.prepare_evaluation_cohorts(ctx)
    torch.testing.assert_close(ctx.evaluation_cohorts[0][1], expected, rtol=0, atol=0)
    assert events == [{'stage': 'fault_tuning_evaluation_cohort_verified', **spec}]
    assert runtime.pinned_inputs(lc) == [{'role': 'evaluation_corpus.held-test', 'path': str(path),
                                          'sha256': spec['corpus_sha256']}]


def test_bad_evaluation_cohort_aborts_run_before_target_draw_deployment_or_training(monkeypatch):
    import ebl.schema.source_fit as source_fit
    lc = parse_lifecycle(json.loads(DOC.read_text()))
    lc = replace(lc, fault_tuning=replace(lc.fault_tuning, evaluation_cohorts=(specification(),)))
    net = NS(load_state_dict=lambda state: None, parameters=lambda: [], q=torch.ones(2), layout=[])
    stage = NS(lifecycle=lc, network=net, device=torch.device('cpu'), request=NS(weights='source'),
               selection=NS(array_seed=271001, hwa_arm='standard_hwa_model'),
               checked=lambda *args, **kw: {'fit': {}, 'model': {}}, cache=lambda: None,
               device_bundle=lambda: None, store=NS(append_metric=lambda row: None))
    monkeypatch.setattr(runtime, 'sha256_file', lambda path: lc.fault_tuning.source_sha256)
    monkeypatch.setattr(source_fit, 'validate_source_fit', lambda *args: None)
    def cohort(lifecycle, settings):
        if settings['corpus'] == 'tiny-held':
            raise ValueError('Token-cohort hash mismatch (test).')
        return torch.ones(2, 6, dtype=torch.int64)
    def unexpected(*args, **kw):
        raise AssertionError('Invalid held-out input must fail before this work starts')
    monkeypatch.setattr(ft, 'cohort', cohort)
    monkeypatch.setattr(ft, 'teacher_samples', unexpected)
    monkeypatch.setattr(runtime, 'deploy', unexpected)
    monkeypatch.setattr(runtime, 'run_arm', unexpected)
    with pytest.raises(ValueError, match='Token-cohort hash'):
        runtime.run(stage)


def context():
    torch.set_num_threads(1)
    net = opt.OptMlpSuffix(tiny_teacher(), 2, 8).eval().requires_grad_(False)
    layout = ft.adapter_layout(net.layout, 2)
    adapter_target, a_scales = ft.adapter_targets(net.layout, 2, 271001)
    arrays = {}
    for key, specs, target in [('naive', net.layout, net.q.detach()),
                                ('fault_aware', net.layout, net.q.detach()),
                                ('adapter', layout, adapter_target)]:
        array = CmoArray(specs, 'tiny', 11 if key != 'naive' else 12, 2., 1., 'cpu')
        array.program(target)
        arrays[key] = array
    clones = {}
    def array(name):
        original = arrays[name]
        out = CmoArray(original.layout, 'tiny', 11 if name != 'naive' else 12, 2., 1., 'cpu')
        out.load_state_dict(original.state_dict())
        clones[name] = out
        return out
    generator = torch.Generator().manual_seed(3)
    tokens = torch.randint(0, 64, (4, 6), generator=generator)
    samples = torch.randint(0, 64, (4, 5, 8), generator=generator)
    test = torch.randint(0, 64, (2, 6), generator=generator)
    events = []
    cfg = NS(schedule={'steps': 3, 'development_every_steps': 1, 'order_seed': 20261104, 'batch': 2},
             write_law={'threshold_sigma': 2., 'accumulate_updates': True}, lora_seed=271001,
             fisher={'batch': 2})
    ctx = NS(network=net, cfg=cfg, device=torch.device('cpu'), array=array, recovery=tokens,
             samples=samples, layout=net.layout, adapter_layout=layout, a_scales=a_scales,
             b_scales=[torch.full((s.shape[0],), .1) for s in net.layout], fisher=None,
             deployed_cost=arrays['fault_aware'].cost(),
             lc=NS(program_verify=NS(acceptance_percent=2.), devices=NS(noise_scale=1.)),
             evaluation_cohorts=[(specification(cohort_sha256=ft.raw_hash(test)), test)],
             stage=NS(store=NS(append_metric=events.append), family=opt))
    ctx.evaluate = lambda q: opt.evaluate(net, {'tokens': tokens[:2]}, 'cpu', q=q)
    return ctx, clones, events, test


@pytest.mark.parametrize('method,rank', [(m, 2 if m in ft.LORA else None) for m in ft.METHODS]
                         + [('digital_lora', 4)])
def test_every_arm_scores_live_final_correction_before_cleanup(method, rank):
    ctx, clones, events, tokens = context()
    net = ctx.network
    evaluations = []
    def evaluate(network, split, device, *, q):
        assert network is net
        assert torch.equal(split['tokens'], tokens)
        assert bool(network.lora is not None) == (method in ft.LORA)
        assert bool(network.fault_output_calibration is not None) == (method == 'calibration')
        if method in ft.LORA:
            assert any(bool(module['b'].detach().abs().sum()) for module in network.lora)
        if method == 'calibration':
            assert any(not torch.equal(g, torch.ones_like(g)) for g in network.fault_output_calibration.gain)
        expected = opt.evaluate(network, split, device, q=q)
        evaluations.append(expected)
        return expected
    ctx.stage.family = NS(evaluate=evaluate)
    arm = ft.Arm('live', method, .05 if method in ft.TRAINS else None, rank,
                 .1 if method == 'child_tuning_d' else None)
    row, state = runtime.run_arm(ctx, arm)
    assert len(evaluations) == 1
    score = row['held_out'][0]
    assert score['scores'] == evaluations[0] and score['state_unchanged']
    assert score['physical_state']['base']['apparent_sha256'] == row['final_apparent_sha256']
    if method == 'analog_lora':
        assert score['physical_state']['adapter']['apparent_sha256'] == row['final_adapter_apparent_sha256']
    assert [e for e in events if e['stage'] == 'fault_tuning_held_out'] == [
        {'stage': 'fault_tuning_held_out', 'arm': 'live', **score}]
    assert row['final'] != score['scores']  # Different dev/held-out tokens; selection stays on dev.
    assert net.lora is None and net.fault_output_calibration is None


@pytest.mark.parametrize('mutation', ['conductance', 'cost', 'rng', 'adapter', 'correction', 'lora'])
def test_scoring_rejects_mutated_physical_state_counters_rng_or_correction(mutation):
    ctx, clones, events, tokens = context()
    method = {'adapter': 'analog_lora', 'correction': 'calibration', 'lora': 'digital_lora'}.get(mutation, 'hold_naive')
    def corrupt(network, split, device, *, q):
        target = clones['adapter' if mutation == 'adapter' else ('naive' if method == 'hold_naive' else 'fault_aware')]
        child = target.children[0]
        if mutation in ('conductance', 'adapter'): child.g.add_(1.)
        elif mutation == 'cost': child.device_writes += 1
        elif mutation == 'rng': torch.rand(1, generator=child.generator)
        elif mutation == 'correction': network.fault_output_calibration.shift[0].add_(1.)
        else: network.lora[0]['b'].add_(1.)
        return {'teacher_kl': 0.}
    ctx.stage.family = NS(evaluate=corrupt)
    arm = ft.Arm('mutated', method, .05 if method in ft.TRAINS else None, 2 if method in ft.LORA else None)
    with pytest.raises(RuntimeError, match='Held-out evaluation changed'):
        runtime.run_arm(ctx, arm)
    assert not [e for e in events if e['stage'] == 'fault_tuning_held_out']


def test_nonfinite_held_out_scores_are_rejected():
    ctx, _, _, _ = context()
    ctx.stage.family = NS(evaluate=lambda *args, **kw: {'teacher_kl': float('nan')})
    with pytest.raises(FloatingPointError, match='Nonfinite held-out'):
        runtime.run_arm(ctx, ft.Arm('bad-score', 'hold_fault_aware'))


def test_empty_evaluation_does_not_change_legacy_arm_outputs():
    ctx, clones, events, tokens = context()
    ctx.evaluation_cohorts = []
    def unexpected(*args, **kwargs):
        raise AssertionError('legacy arm must not score a held-out cohort')
    ctx.stage.family = NS(evaluate=unexpected)
    row, _ = runtime.run_arm(ctx, ft.Arm('legacy', 'hold_fault_aware'))
    assert 'held_out' not in row
    assert not [e for e in events if e['stage'].startswith('fault_tuning_held_out')]
