"""Launch-directory worker source pins and fail-closed native endpoint coverage."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'campaigns/fault-recovery-paper/launch/worker.py'
module = importlib.util.spec_from_file_location('fault_paper_worker', WORKER)
worker = importlib.util.module_from_spec(module)
module.loader.exec_module(worker)


def contract():
    cohort = {'name': 'wiki', 'corpus': 'wikitext', 'corpus_sha256': '1' * 64,
              'split': 'test', 'start': 0, 'sequences': 2, 'cohort_sha256': '2' * 64}
    spec = {'arms': [{'label': 'hold', 'method': 'hold_naive'},
                     {'label': 'learn', 'method': 'onchip_noisy', 'learning_rate': .001}],
            'schedule': {'steps': 4}, 'evaluation_cohorts': [cohort]}
    score = {'teacher_kl': .1, 'sequences': 2, 'scored_tokens': 510}
    final = {'teacher_kl': .2}
    held = [{**cohort, 'scores': score, 'state_unchanged': True}]
    rows = [{'label': 'hold', 'method': 'hold_naive', 'curve': [], 'final': final, 'held_out': copy.deepcopy(held)},
            {'label': 'learn', 'method': 'onchip_noisy', 'train_losses': [1.] * 4,
             'curve': [{'step': 4, 'development': final}], 'final': final, 'held_out': copy.deepcopy(held)}]
    return copy.deepcopy((spec, rows))


def test_complete_live_endpoints_and_legacy_no_held_out_are_accepted():
    spec, rows = contract()
    worker.validate_measurements(spec, rows)
    spec.pop('evaluation_cohorts')
    for row in rows:
        row.pop('held_out')
    worker.validate_measurements(spec, rows)


def test_cohorts_match_by_name_without_relying_on_order():
    spec, rows = contract()
    extra = {**spec['evaluation_cohorts'][0], 'name': 'generated', 'corpus': 'generated'}
    spec['evaluation_cohorts'].append(extra)
    for row in rows:
        row['held_out'].insert(0, {**extra, 'scores': row['held_out'][0]['scores'], 'state_unchanged': True})
    worker.validate_measurements(spec, rows)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unexpected', 'malformed', 'not-list'])
def test_reject_missing_duplicate_unexpected_or_malformed_held_out(mutation):
    spec, rows = contract()
    for row in rows:
        if mutation == 'missing': row.pop('held_out')
        elif mutation == 'duplicate': row['held_out'] *= 2
        elif mutation == 'unexpected': row['held_out'][0]['name'] = 'other'
        elif mutation == 'malformed': row['held_out'] = [None]
        else: row['held_out'] = {'wiki': row['held_out'][0]}
    with pytest.raises(ValueError, match='held_out|held-out'):
        worker.validate_measurements(spec, rows)


@pytest.mark.parametrize('field', ['name', 'corpus', 'corpus_sha256', 'split', 'start', 'sequences', 'cohort_sha256'])
def test_every_held_out_identity_field_is_checked(field):
    spec, rows = contract()
    rows[1]['held_out'][0][field] = 'changed'
    with pytest.raises(ValueError, match='held-out|Held-out'):
        worker.validate_measurements(spec, rows)


@pytest.mark.parametrize('field,value', [('teacher_kl', float('nan')), ('teacher_kl', float('inf')),
                                         ('teacher_kl', '0.1'), ('teacher_kl', True),
                                         ('sequences', 3), ('scored_tokens', 509)])
def test_finite_kl_and_exact_prediction_counts(field, value):
    spec, rows = contract()
    rows[0]['held_out'][0]['scores'][field] = value
    with pytest.raises(ValueError, match='scores/counts'):
        worker.validate_measurements(spec, rows)


@pytest.mark.parametrize('value', [False, None, 1])
def test_state_unchanged_requires_explicit_true(value):
    spec, rows = contract()
    rows[0]['held_out'][0]['state_unchanged'] = value
    with pytest.raises(ValueError, match='changed state'):
        worker.validate_measurements(spec, rows)


@pytest.mark.parametrize('mutation', ['missing-arm', 'duplicate-arm', 'wrong-method', 'short-losses',
                                     'short-endpoint', 'mismatched-final', 'hold-losses', 'hold-curve'])
def test_reject_incomplete_duplicate_or_wrong_arm_cases(mutation):
    spec, rows = contract()
    if mutation == 'missing-arm': rows.pop()
    elif mutation == 'duplicate-arm': rows.append(copy.deepcopy(rows[-1]))
    elif mutation == 'wrong-method': rows[-1]['method'] = 'onchip_noiseless'
    elif mutation == 'short-losses': rows[-1]['train_losses'].pop()
    elif mutation == 'short-endpoint': rows[-1]['curve'][-1]['step'] = 3
    elif mutation == 'mismatched-final': rows[-1]['final'] = {'teacher_kl': .3}
    elif mutation == 'hold-losses': rows[0]['train_losses'] = [1.]
    else: rows[0]['curve'] = [{'step': 1, 'development': rows[0]['final']}]
    with pytest.raises(ValueError):
        worker.validate_measurements(spec, rows)


def source_fixture(tmp_path):
    source = tmp_path / 'source'
    tracked = source / 'campaigns/fault-recovery-paper/launch/worker.py'
    tracked.parent.mkdir(parents=True)
    tracked.write_bytes(WORKER.read_bytes())
    code = source / 'kernel.py'
    code.write_text('unchanged numerical code\n')
    receipt = source / 'source_snapshot.json'
    receipt.write_text(json.dumps({'files': {str(path.relative_to(source)): worker.digest(path)
                                           for path in (tracked, code)}}))
    executed = tmp_path / 'worker.py'
    executed.write_bytes(tracked.read_bytes())
    return {'source': str(source), 'source_receipt_sha256': worker.digest(receipt)}, executed


def test_source_receipt_pin_and_identical_launch_worker(tmp_path):
    manifest, executed = source_fixture(tmp_path)
    source, receipt = worker.verify_source(manifest, executed_worker=executed)
    assert source == Path(manifest['source']) and receipt.name == 'source_snapshot.json'
    manifest.pop('source_receipt_sha256')
    worker.verify_source(manifest, executed_worker=executed)  # Legacy manifests have no optional pin.


@pytest.mark.parametrize('mutation', ['receipt', 'worker-copy', 'tracked-code', 'escape'])
def test_reject_changed_receipt_stale_launch_worker_changed_source_or_escape(mutation, tmp_path):
    manifest, executed = source_fixture(tmp_path)
    source = Path(manifest['source'])
    if mutation == 'receipt': manifest['source_receipt_sha256'] = '0' * 64
    elif mutation == 'worker-copy': executed.write_text('stale external worker\n')
    elif mutation == 'tracked-code': (source / 'kernel.py').write_text('modified numerical code\n')
    else:
        outside = tmp_path / 'outside.py'
        outside.write_text('unbound outside source\n')
        (source / 'kernel.py').unlink()
        (source / 'kernel.py').symlink_to(outside)
        receipt = source / 'source_snapshot.json'
        doc = worker.read(receipt)
        doc['files']['kernel.py'] = worker.digest(outside)
        receipt.write_text(json.dumps(doc))
        manifest['source_receipt_sha256'] = worker.digest(receipt)
    with pytest.raises(ValueError, match='source receipt|worker copy|Staged code'):
        worker.verify_source(manifest, executed_worker=executed)


def test_main_does_not_publish_completion_for_missing_held_out(tmp_path, monkeypatch):
    manifest, _ = source_fixture(tmp_path)
    source = Path(manifest['source'])
    spec, rows = contract()
    rows[0].pop('held_out')
    (source / 'config.json').write_text(json.dumps({'fault_tuning': spec, 'data': {'sequence_length': 256}}))
    for role in ('teacher', 'weights'):
        path = tmp_path / role
        path.write_text(role)
        manifest.update({role: str(path), role + '_sha256': worker.digest(path)})
    output = tmp_path / 'output'
    manifest.update(output=str(output), corpus_root=str(tmp_path), pinned={}, units={'unit': 'config.json'})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    run = output / 'unit/runs/study/fault_tuning__test/attempt-001/runs/native'
    run.mkdir(parents=True)
    (run / 'measurements.json').write_text(json.dumps(rows))
    (run / 'result.json').write_text(json.dumps({'status': 'complete'}))
    monkeypatch.setattr(sys, 'argv', ['worker.py', str(path), 'unit'])
    monkeypatch.setattr(worker, 'call', lambda *args: None)  # Successful native execution/inspection.
    monkeypatch.setenv('SLURM_JOB_ID', 'test-job')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='held-out'):
        worker.main()
    assert not (output / 'unit/complete.json').exists()
    assert 'held-out' in worker.read(output / 'unit/failed-test-job.json')['error']


def drift_contract():
    ages = [0.0, 60.0, 86400.0]
    spec = {'arms': [{'label': 'A0', 'method': 'hold_fault_aware'},
                     {'label': 'CT-nodrift', 'method': 'child_tuning_d', 'learning_rate': .001, 'reserve_p': .1,
                      'no_drift': True},
                     {'label': 'A0-twin', 'method': 'hold_fault_aware', 'no_drift': True},
                     {'label': 'LN', 'method': 'layernorm', 'learning_rate': .003}],
            'schedule': {'steps': 4}, 'drift': {'evaluation_seconds': ages}}

    def row(label, method, kl, losses):
        grid = [{'age_seconds': t, 'teacher_kl': .1 if t == 0 else kl, 'development': {'teacher_kl': kl, 't': t}}
                for t in ages]
        return {'label': label, 'method': method, 'grid': grid, 'final': grid[-1]['development'],
                'mean_teacher_kl': kl, 'train_losses': losses}
    rows = [row('A0', 'hold_fault_aware', .3, []), row('A0-twin', 'hold_fault_aware', .1, []),
            row('CT-nodrift', 'child_tuning_d', .09, [1.] * 4), row('LN', 'layernorm', .2, [1.] * 4)]
    return copy.deepcopy((spec, rows))


def test_drift_rows_are_validated_in_native_order():
    spec, rows = drift_contract()
    worker.validate_measurements(spec, rows)


@pytest.mark.parametrize('mutation', [None, 'mask-hash', 'size', 'out-of-mask'])
def test_global_drift_child_requires_exact_masks_and_no_writes_outside(mutation):
    spec, rows = drift_contract()
    arm = spec['arms'][1]
    arm['child_mask_scope'] = 'global'
    spec['arms'].append({**arm, 'label': 'CT'})
    spec['arms'][-1].pop('no_drift')
    rows[2].update(reserve_p=.1, cost={
        'child_mask_scope': 'global', 'mask_sha256': 'a' * 64, 'mask_weights': 3,
        'mask_arrays': [{'name': 'layer1', 'weights': 15, 'selected': 0},
                        {'name': 'layer2', 'weights': 15, 'selected': 3}],
        'devices': 60, 'writes_outside_mask': 0,
    })
    rows.append({**copy.deepcopy(rows[2]), 'label': 'CT'})
    if mutation is None:
        worker.validate_measurements(spec, rows)
        return
    field, value = {'mask-hash': ('mask_sha256', 'b' * 64), 'size': ('mask_weights', 4),
                    'out-of-mask': ('writes_outside_mask', 1)}[mutation]
    rows[-1]['cost'][field] = value
    with pytest.raises(ValueError, match='Global Child-Tuning mask mismatch'):
        worker.validate_measurements(spec, rows)


def test_drift_control_reconciles_clipping_diagnostics_without_weakening_identity():
    summary = dict.fromkeys(('assignment_seed', 'case', 'source_sha256', 'recovery_cohort_sha256',
                            'development_cohort_sha256', 'samples_sha256', 'apparent_sha256',
                            'deployment_gpu', 'deployment_torch'))
    summary['drift'] = {'records': dict.fromkeys(('mapping', 'source_ideal_teacher_kl',
                                               'clean_mapped_readout_teacher_kl', 'deployed_release_teacher_kl'))}
    controller = {'clipped_targets': 9, 'flagged_active': 12}
    summary['deployment_reports'] = {'fault_aware': {'controller': controller}}
    _, rows = drift_contract()
    for row in rows:
        row['initial_apparent_sha256'] = '0' * 64
    reference = copy.deepcopy(worker.drift_control_reference(summary, rows))
    controller.update(clipped_targets_low=8, clipped_targets_high=1)
    assert worker.drift_control_reference(summary, rows) == reference
    assert 'clipped_targets_low' in controller
    controller['flagged_active'] += 1
    assert worker.drift_control_reference(summary, rows) != reference
    for high in (None, True, -1, 2):
        controller['clipped_targets_high'] = high
        with pytest.raises(ValueError, match='do not reconcile'):
            worker.drift_control_reference(summary, rows)


@pytest.mark.parametrize('mutation', ['order', 'grid', 'nan', 'final', 'updates', 'hold_updates', 'release'])
def test_drift_rows_fail_closed(mutation):
    spec, rows = drift_contract()
    if mutation == 'order':
        rows[2], rows[3] = rows[3], rows[2]
    elif mutation == 'grid':
        rows[3]['grid'].pop(1)
    elif mutation == 'nan':
        rows[3]['grid'][1]['teacher_kl'] = float('nan')
    elif mutation == 'final':
        rows[3]['final'] = rows[3]['grid'][1]['development']
    elif mutation == 'updates':
        rows[2]['train_losses'] = [1.] * 3
    elif mutation == 'hold_updates':
        rows[0]['train_losses'] = [1.]
    else:
        rows[3]['grid'][0]['teacher_kl'] = .11
    with pytest.raises(ValueError):
        worker.validate_measurements(spec, rows)
