import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.artifacts import sha256_file
from training.drift_family import ReramFamily
from training.drift_recovery import tensor_hash
from training.drift_regimes_config import DriftRegimes
from workflow.drift_regimes import endpoint_weights
from workflow.lifecycle import parse_lifecycle, stage_selections, StageSelection, provenance, verify_provenance
from test_workflow_cmo import cmo_inputs, small, case_doc
from workflow_helpers import run_stage


def config(**changes):
    base = DriftRegimes('a' * 64, 'synthetic', 'b' * 64, 'c' * 64, 'R0', asdict(ReramFamily()))
    return replace(base, **changes)


@pytest.mark.parametrize('key,value', [('arm', 'A8'), ('deployment_attempts', 1), ('deployment_threshold', -1.),
                                      ('evaluation_seconds', [60, 0]), ('canary', 1), ('extra', 1),
                                      ('law', {'rho_star': .3}), ('evaluation_split', 'test')])
def test_settings_fail_closed(key, value):
    raw = config().to_dict()
    raw[key] = value
    with pytest.raises(ValueError):
        DriftRegimes.parse(raw)


def test_settings_round_trip_and_endpoint_weights():
    raw = config().to_dict()
    assert DriftRegimes.parse(json.loads(json.dumps(raw))) == DriftRegimes.parse(raw)
    unverified = config(deployment_attempts=1, deployment_threshold=None).to_dict()
    assert DriftRegimes.parse(unverified).deployment_threshold is None
    w = endpoint_weights([0., 60., 3600., 86400.])
    assert w.sum() == pytest.approx(1)
    assert float(w @ np.array([1., 1., 1., 1.])) == pytest.approx(1)
    assert float(w @ np.array([0., 60., 3600., 86400.])) == pytest.approx(86400 / 2)


@pytest.mark.parametrize('law', [dict(), dict(rho_star=1., persistent_share=.7)])
def test_native_stage_deploys_by_pv_releases_after_it_and_replays(cmo_inputs, tmp_path, law):
    teacher, digest = cmo_inputs
    lc = small(**{'onchip': None, 'defects.cases': [case_doc()], 'data.corpus': {'name': 'synthetic', 'sha256': digest},
                  'data.sequence_length': 9, 'data.max_examples': 4, 'hwa.arms': [{'id': 'digital', 'method': 'none'}]})
    results = {}
    for selection in stage_selections(lc):
        if selection.kind != 'deploy':
            results[selection.stage_id] = run_stage(lc, selection, results, teacher, tmp_path / 'parents')
    directory, result = results['hwa__digital']
    source = directory / next(a['path'] for a in result['artifacts'] if a['kind'] == 'hwa_source')
    corpus = torch.load(Path(os.environ['EBL_CORPUS_ROOT']) / 'synthetic.pt', weights_only=False)
    tokens = corpus['splits']['validation'][:18].reshape(2, 9).contiguous()
    cfg = config(source_sha256=sha256_file(source), evaluation_corpus_sha256=digest,
                 cohort_sha256=tensor_hash(tokens), law=asdict(ReramFamily(**law)),
                 assignment_seeds=(271001,), evaluation_sequences=2,
                 evaluation_seconds=(0., 60., 3600.), canary=True)
    doc = lc.to_dict()
    doc['drift_regimes'] = cfg.to_dict()
    custom = parse_lifecycle(doc)
    assert [s.kind for s in stage_selections(custom)][-1] == 'drift_regimes'
    verify_provenance(provenance(lc, 'hwa_source', hwa_arm='digital'), custom, 'hwa_source', hwa_arm='digital')
    directory, result = run_stage(custom, StageSelection('drift_regimes', 'digital', 271001), results, teacher,
                                  tmp_path / 'run')
    metrics = result['metrics']
    assert metrics['source_unchanged'] and metrics['replay_exact'] and metrics['snapshots'] == 3
    cost = metrics['cost']
    n = cost['physical_devices']
    assert n > 0 and n % 2 == 0          # active + reference device per weight (tiny test network)
    assert n <= cost['deployment_reset_attempts'] <= 4 * n
    assert sum(cost['accepted_at_attempt']) == n
    assert cost['deployment_verify_reads'] == cost['deployment_reset_attempts'] - cost['deployment_capped_devices']
    # Release is the end of deployment P&V: at most three 60-s verify rounds.
    assert 0 < cost['release_seconds'] <= 180
    rows = json.loads((directory / 'drift_regimes_measurements.json').read_text())
    assert [r['seconds'] for r in rows] == [0., 60., 3600.]
    assert all(r['clock_seconds'] == r['release_seconds'] + r['seconds'] for r in rows)
    assert all(r['canary'] and r['metrics']['scored_tokens'] == 16 for r in rows)
    assert all(r['bounds']['active']['devices'] == n // 2 for r in rows)
    assert rows[-1]['metrics']['teacher_kl'] > 0
    assert metrics['mean_teacher_kl'] == pytest.approx(
        float(endpoint_weights([0., 60., 3600.]) @ np.array([r['metrics']['teacher_kl'] for r in rows])))
    again_dir, again = run_stage(custom, StageSelection('drift_regimes', 'digital', 271001), results, teacher,
                                 tmp_path / 'again')
    assert again['metrics']['final_physical_sha256'] == metrics['final_physical_sha256']


def test_lifecycle_rejects_other_branches_and_windows(cmo_inputs):
    teacher, digest = cmo_inputs
    lc = small(**{'onchip': None, 'defects.cases': [case_doc()], 'hwa.arms': [{'id': 'digital', 'method': 'none'}]})
    doc = lc.to_dict()
    doc['drift_regimes'] = config(assignment_seeds=(999,)).to_dict()
    with pytest.raises(ValueError, match='assignment'):
        parse_lifecycle(doc)
    doc['drift_regimes'] = config(law=asdict(ReramFamily(program_min=10.)),
                                  assignment_seeds=lc.devices.assignment_seeds[:1]).to_dict()
    with pytest.raises(ValueError, match='window'):
        parse_lifecycle(doc)


def test_legacy_single_arm_files_parse_to_the_untreated_arm():
    raw = config().to_dict()
    del raw['arms'], raw['adaptation']
    raw['arm'] = 'A0'
    parsed = DriftRegimes.parse(raw)
    assert parsed.arms == ({'label': 'A0', 'method': 'none'},) and parsed.adaptation is None
    assert parsed == config()


ADAPT = {'corpus': 'synthetic', 'corpus_sha256': 'b' * 64, 'split': 'train', 'sequences': 4,
         'cohort_sha256': 'd' * 64, 'seed': 5, 'probe_sequences': 2, 'steps': 3, 'batch': 2}


@pytest.mark.parametrize('arm', [
    {'label': 'A1', 'method': 'calibration'},                                          # missing events
    {'label': 'A1', 'method': 'calibration', 'events': [900, 900]},                    # repeated event
    {'label': 'A1', 'method': 'calibration', 'events': [86400]},                       # not inside the horizon
    {'label': 'A4', 'method': 'lora', 'events': [900], 'rank': 0, 'learning_rate': 1e-3},
    {'label': 'A9', 'method': 'rewrite', 'events': [900], 'fraction': 1.5, 'attempts': 4, 'threshold': .7},
    {'label': 'A9', 'method': 'rewrite', 'events': [900], 'fraction': .1, 'attempts': 1, 'threshold': .7},
    {'label': 'A9', 'method': 'rewrite', 'events': [900], 'fraction': .1, 'attempts': 4, 'threshold': None},
    {'label': 'A0', 'method': 'none', 'events': [900]},
    {'label': 'A12', 'method': 'magic'},
])
def test_arm_settings_fail_closed(arm):
    raw = config().to_dict()
    raw['arms'] = [arm]
    raw['adaptation'] = ADAPT
    with pytest.raises(ValueError):
        DriftRegimes.parse(raw)


def test_digital_arms_require_adaptation_and_labels_are_unique():
    raw = config().to_dict()
    raw['arms'] = [{'label': 'A2', 'method': 'layernorm', 'events': [900], 'learning_rate': 1e-3}]
    with pytest.raises(ValueError, match='adaptation'):
        DriftRegimes.parse(raw)
    raw['adaptation'] = ADAPT
    assert DriftRegimes.parse(raw).event_ages == (900.,)
    raw['arms'] = [{'label': 'A0', 'method': 'none'}, {'label': 'A0', 'method': 'none'}]
    with pytest.raises(ValueError, match='unique'):
        DriftRegimes.parse(raw)


def test_timeline_evaluates_before_a_coincident_event():
    from workflow.drift_regimes_arms import timeline
    assert timeline((0., 60., 900.), (60., 300.)) == [(0., True, None), (60., True, None), (60., False, 0),
                                                       (300., False, 1), (900., True, None)]


def _stage_inputs(cmo_inputs, tmp_path):
    teacher, digest = cmo_inputs
    lc = small(**{'onchip': None, 'defects.cases': [case_doc()], 'data.corpus': {'name': 'synthetic', 'sha256': digest},
                  'data.sequence_length': 9, 'data.max_examples': 4, 'hwa.arms': [{'id': 'digital', 'method': 'none'}]})
    results = {}
    for selection in stage_selections(lc):
        if selection.kind != 'deploy':
            results[selection.stage_id] = run_stage(lc, selection, results, teacher, tmp_path / 'parents')
    directory, result = results['hwa__digital']
    source = directory / next(a['path'] for a in result['artifacts'] if a['kind'] == 'hwa_source')
    corpus = torch.load(Path(os.environ['EBL_CORPUS_ROOT']) / 'synthetic.pt', weights_only=False)
    tokens = corpus['splits']['validation'][:18].reshape(2, 9).contiguous()
    train = corpus['splits']['train']
    blocks = train[:len(train) // 9 * 9].reshape(-1, 9)
    order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(5))
    adaptation = {**ADAPT, 'corpus_sha256': digest, 'cohort_sha256': tensor_hash(blocks[order[:4]].contiguous())}
    base = config(source_sha256=sha256_file(source), evaluation_corpus_sha256=digest,
                  cohort_sha256=tensor_hash(tokens), assignment_seeds=(271001,), evaluation_sequences=2,
                  evaluation_seconds=(0., 60., 900., 3600.), canary=True)
    return lc, results, teacher, base, adaptation


def _run(lc, cfg, results, teacher, path):
    doc = lc.to_dict()
    doc['drift_regimes'] = cfg.to_dict()
    custom = parse_lifecycle(doc)
    directory, result = run_stage(custom, StageSelection('drift_regimes', 'digital', 271001), results, teacher, path)
    return directory, result['metrics']


def test_native_stage_runs_paired_maintenance_arms(cmo_inputs, tmp_path):
    lc, results, teacher, base, adaptation = _stage_inputs(cmo_inputs, tmp_path)
    arm_list = (
        {'label': 'A0', 'method': 'none'},
        {'label': 'A1', 'method': 'calibration', 'events': (60., 900.)},
        {'label': 'A2', 'method': 'layernorm', 'events': (60.,), 'learning_rate': 1e-3},
        {'label': 'A5', 'method': 'lora', 'events': (60., 900.), 'rank': 2, 'learning_rate': 1e-3},
        {'label': 'A8', 'method': 'rewrite', 'events': (900.,), 'fraction': 1., 'attempts': 1, 'threshold': None},
        {'label': 'A9', 'method': 'rewrite', 'events': (60., 900.), 'fraction': .1, 'attempts': 4,
         'threshold': .7051477},
    )
    cfg = DriftRegimes.parse(replace(base, arms=arm_list, adaptation=adaptation).to_dict())
    directory, metrics = _run(lc, cfg, results, teacher, tmp_path / 'arms')
    _, single = _run(lc, base, results, teacher, tmp_path / 'single')
    arms = metrics['arms']
    assert list(arms) == [a['label'] for a in arm_list]
    assert metrics['source_unchanged'] and metrics['replay_exact']
    # The untreated arm is the qualification trajectory, bit for bit.
    assert arms['A0']['final_physical_sha256'] == single['final_physical_sha256']
    assert arms['A0']['mean_teacher_kl'] == single['mean_teacher_kl']
    rows = json.loads((directory / 'drift_regimes_measurements.json').read_text())
    assert len(rows) == len(arm_list) * 4
    by = {(r['arm'], r['seconds']): r for r in rows}
    # Every arm starts from the same released state; ages up to the first event are shared.
    for label in arms:
        assert by[(label, 0.)]['physical_sha256'] == by[('A0', 0.)]['physical_sha256']
        assert by[(label, 60.)]['metrics']['teacher_kl'] == by[('A0', 60.)]['metrics']['teacher_kl']
    # Digital and calibration arms never write; rewrite arms do.
    for label in ('A1', 'A2', 'A5'):
        assert arms[label]['cost']['reset_attempts'] == 0
        assert by[(label, 3600.)]['physical_sha256'] == by[('A0', 3600.)]['physical_sha256']
    n = metrics['cost']['physical_devices']
    assert arms['A8']['cost']['reset_attempts'] == n and arms['A8']['cost']['verify_reads'] == 0
    a9 = arms['A9']['cost']
    assert 0 < a9['selected_devices'] < n / 2 and a9['selection_reads'] == 2 * n
    assert a9['selected_devices'] <= a9['reset_attempts'] <= 4 * a9['selected_devices']
    assert by[('A9', 3600.)]['physical_sha256'] != by[('A0', 3600.)]['physical_sha256']
    assert arms['A5']['cost']['optimizer_steps'] == 6 and arms['A5']['cost']['digital_parameters'] > 0
    assert arms['A1']['cost']['probe_tokens'] == 2 * 2 * 9
    # Calibration acts on the evaluation without changing devices.
    assert by[('A1', 3600.)]['metrics']['teacher_kl'] != by[('A0', 3600.)]['metrics']['teacher_kl']
    events = json.loads((directory / 'drift_regimes_events.json').read_text())
    assert [(e['arm'], e['event']) for e in events] == [
        ('A1', 0), ('A1', 1), ('A2', 0), ('A5', 0), ('A5', 1), ('A8', 0), ('A9', 0), ('A9', 1)]
    _, again = _run(lc, cfg, results, teacher, tmp_path / 'again')
    assert {k: v['final_physical_sha256'] for k, v in again['arms'].items()} == \
           {k: v['final_physical_sha256'] for k, v in arms.items()}
    assert {k: v['mean_teacher_kl'] for k, v in again['arms'].items()} == {k: v['mean_teacher_kl'] for k, v in arms.items()}


@pytest.mark.parametrize('scope', ['analog', 'model'])
def test_cached_suffix_matches_the_network_forward(scope):
    from workflow.networks.opt_mlp import OptConfig, OptDecoder, OptMlpSuffix
    from workflow.drift_regimes_arms import prefix_states, suffix_logits, fit_calibration
    torch.manual_seed(0)
    teacher = OptDecoder(OptConfig(50, 16, 3, 2, 32, 16)).eval()
    network = OptMlpSuffix(teacher, 2, scope=scope).eval()
    tokens = torch.randint(0, 50, (3, 8))
    q = network.q.detach() * (1 + .05 * torch.randn_like(network.q))
    cache = prefix_states(network, tokens, 'cpu', batch=2)
    with torch.no_grad():
        expected = network.forward_features(network.features(tokens), q)
        assert torch.allclose(suffix_logits(network, cache.student, q), expected, atol=1e-5)
        assert torch.allclose(network.teacher_logits(cache.teacher), network.teacher_logits(network.features(tokens)))
        # Calibration of the clean weights is the identity.
        correction = fit_calibration(network, network.q.detach(), cache)
        for m in correction.matrices:
            assert torch.allclose(m.affine_gain, torch.ones_like(m.affine_gain), atol=1e-4)
            assert torch.allclose(m.shift, torch.zeros_like(m.shift), atol=1e-4)


def test_evaluation_start_is_optional_and_validated():
    raw = config().to_dict()
    del raw['evaluation_start']
    assert DriftRegimes.parse(raw).evaluation_start == 0
    raw['evaluation_start'] = 64
    assert DriftRegimes.parse(raw).evaluation_start == 64
    raw['evaluation_start'] = -1
    with pytest.raises(ValueError):
        DriftRegimes.parse(raw)


def test_calibration_matches_column_moments_of_the_reference():
    from workflow.networks.opt_mlp import OptConfig, OptDecoder, OptMlpSuffix
    from workflow.drift_regimes_arms import prefix_states, fit_calibration, _matrix_input
    torch.manual_seed(1)
    teacher = OptDecoder(OptConfig(50, 16, 2, 2, 32, 16)).eval()
    network = OptMlpSuffix(teacher, 1).eval()
    cache = prefix_states(network, torch.randint(0, 50, (4, 8)), 'cpu')
    q = network.q.detach() * 0.8 + 0.02 * torch.randn_like(network.q)
    correction = fit_calibration(network, q, cache)
    with torch.no_grad():
        x = _matrix_input(network, cache.student, q, 0)
        reference = network._matrix(x, 0, network.q.detach()).flatten(0, -2)
        network.drift_compensation = correction
        corrected = network._matrix(x, 0, q).flatten(0, -2)
        network.drift_compensation = None
    assert torch.allclose(corrected.mean(0), reference.mean(0), atol=1e-4)
    assert torch.allclose(corrected.std(0), reference.std(0), rtol=1e-3, atol=1e-4)


def test_native_stage_runs_the_oracle_precompensation_twin_and_learning_arms(cmo_inputs, tmp_path):
    lc, results, teacher, base, adaptation = _stage_inputs(cmo_inputs, tmp_path)
    learn = {'method': 'learning', 'events': (60., 900.), 'fraction': .1, 'step': .05, 'rounds': 2,
             'batch_sequences': 2}
    arm_list = (
        {'label': 'A0', 'method': 'none'},
        {'label': 'A0-twin', 'method': 'none', 'twin': True},
        {'label': 'A9', 'method': 'rewrite', 'events': (60., 900.), 'fraction': .1, 'attempts': 4, 'threshold': .7},
        {'label': 'A10', 'method': 'precompensated', 'events': (60., 900.), 'fraction': .1, 'attempts': 4,
         'threshold': .7},
        {'label': 'O2', 'method': 'rewrite_oracle', 'events': (60.,), 'fraction': .1, 'attempts': 1, 'threshold': None},
        {'label': 'O1', 'method': 'lora_oracle', 'events': (0., 60., 900., 3600.), 'rank': 2, 'learning_rate': 1e-3},
        {'label': 'A6', 'method': 'lora_rescaled', 'events': (60.,), 'reads': (900.,), 'rank': 2,
         'learning_rate': 1e-3},
        {'label': 'A4c', 'method': 'lora_clean', 'events': (60.,), 'rank': 2, 'learning_rate': 1e-3},
        {'label': 'A5-twin', 'method': 'lora', 'events': (60.,), 'rank': 2, 'learning_rate': 1e-3, 'twin': True},
        {'label': 'A11', **learn, 'attempts': 4, 'threshold': .7, 'ideal': False},
        {'label': 'A11-ideal', **learn, 'attempts': 1, 'threshold': None, 'ideal': True},
        {'label': 'A11-twin', **learn, 'attempts': 1, 'threshold': None, 'ideal': False, 'twin': True},
        *[{'label': f'N11{suffix}', 'method': 'learning_newton', 'events': (60., 900.), 'fraction': .1,
           'rounds': 2, 'batch_sequences': 2, **extra}
          for suffix, extra in (('', {'attempts': 4, 'threshold': .7, 'ideal': False}),
                                ('-ideal', {'attempts': 1, 'threshold': None, 'ideal': True}),
                                ('-twin', {'attempts': 1, 'threshold': None, 'ideal': False, 'twin': True}),
                                ('-fisher', {'attempts': 4, 'threshold': .7, 'ideal': False, 'score': 'fisher'}))],
    )
    cfg = DriftRegimes.parse(replace(base, arms=arm_list, adaptation=adaptation).to_dict())
    directory, metrics = _run(lc, cfg, results, teacher, tmp_path / 'arms')
    arms = metrics['arms']
    rows = json.loads((directory / 'drift_regimes_measurements.json').read_text())
    by = {(r['arm'], r['seconds']): r for r in rows}
    events = json.loads((directory / 'drift_regimes_events.json').read_text())
    n = metrics['cost']['physical_devices']
    # No-drift twins hold the released state.
    release = by[('A0', 0.)]
    for t in (60., 900., 3600.):
        assert by[('A0-twin', t)]['physical_sha256'] == release['physical_sha256']
        assert by[('A0-twin', t)]['metrics']['teacher_kl'] == release['metrics']['teacher_kl']
        assert by[('A5-twin', t)]['physical_sha256'] == release['physical_sha256']
    assert by[('A11-twin', 3600.)]['physical_sha256'] != release['physical_sha256']
    # A10 has no history at its first event, so it equals A9 until its second event.
    assert by[('A10', 900.)]['physical_sha256'] == by[('A9', 900.)]['physical_sha256']
    pre = [e for e in events if e['arm'] == 'A10']
    assert pre[0]['precompensation'] == 0. and 0. <= pre[1]['precompensation'] <= 1.
    assert by[('O2', 900.)]['physical_sha256'] != by[('A9', 900.)]['physical_sha256']
    # O1 is evaluated after its training at every age, including release.
    assert arms['O1']['cost']['events'] == 4
    assert by[('O1', 0.)]['metrics']['teacher_kl'] != release['metrics']['teacher_kl']
    a6 = [e for e in events if e['arm'] == 'A6']
    assert [e['event'] for e in a6] == [0, 1] and 'kappa_hat' in a6[1]
    assert arms['A4c']['cost']['reset_attempts'] == 0
    assert by[('A4c', 3600.)]['physical_sha256'] == by[('A0', 3600.)]['physical_sha256']
    # Analog learning resets at most its budget of active devices per round.
    for label in ('A11', 'A11-ideal', 'A11-twin'):
        c = arms[label]['cost']
        assert c['gradient_sequences'] == 2 * 2 * 2
        assert 0 < c['selected_devices'] <= 2 * 2 * (n * .1 / 2 + len(metrics['arms']))
        assert by[(label, 3600.)]['physical_sha256'] != by[('A0', 3600.)]['physical_sha256']
    assert arms['A11']['cost']['reset_attempts'] >= arms['A11']['cost']['selected_devices']
    assert arms['A11-ideal']['cost']['verify_reads'] == 0
    # The exp-003 loop resets whole pairs, only where the predicted net gain is positive, within budget.
    newton = [e for e in events if e['arm'] == 'N11']
    assert all(len(e['positive_gain_pairs']) == 2 for e in newton)
    for label in ('N11', 'N11-ideal', 'N11-twin'):
        c = arms[label]['cost']
        assert c['selected_devices'] % 2 == 0 and 0 < c['selected_devices'] <= 2 * (n * .1 + 4 * 12)
        assert c['gradient_sequences'] == 2 * 2 * 2 * 2
    assert by[('N11-ideal', 3600.)]['q_rmse'] < by[('A0', 3600.)]['q_rmse']
    fisher = arms['N11-fisher']['cost']
    assert fisher['gradient_sequences'] == 2 * 2 * 2 and fisher['selected_devices'] % 2 == 0
    assert by[('N11-fisher', 3600.)]['physical_sha256'] != by[('N11', 3600.)]['physical_sha256']
    assert by[('N11-twin', 900.)]['physical_sha256'] != release['physical_sha256']
    _, again = _run(lc, cfg, results, teacher, tmp_path / 'again')
    assert {k: v['mean_teacher_kl'] for k, v in again['arms'].items()} == \
           {k: v['mean_teacher_kl'] for k, v in arms.items()}
