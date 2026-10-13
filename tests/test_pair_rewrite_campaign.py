"""Reduced exp-016 coverage, frozen inputs, canary gating and collection integrity."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'campaigns/opt-cmo-corruption/launch/exp16_pair_rewrite.py'
spec = importlib.util.spec_from_file_location('pair_rewrite_campaign', PATH)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def evidence(label):
    tag, array, family = pilot.parts(label)
    lc = pilot.lifecycle(tag, family)
    ref = json.loads(pilot.CATALOG.read_text())[f'{tag}-{array}']
    rows = []
    for case in lc.onchip_cases:
        initial = ref['cases'][case.label]['initial']
        p0 = ref['cases'][case.label]['apparent_sha256']
        for arm in lc.onchip.arms:
            final = copy.deepcopy(initial)
            if arm.arm_id != 'none':
                for metrics in final.values():
                    metrics['teacher_kl'] *= .5
            rewrite = None
            if arm.fault_rewrite:
                rewrite = {**arm.fault_rewrite.to_dict(), 'rewritten_apparent_sha256': 'rewritten',
                           'cost': {'verify_reads': 2 * lc.network.analog_weights
                                    if arm.fault_rewrite.detection == 'verify' else 0}}
            rows.append({'case': case.label, 'onchip_arm': arm.arm_id, 'array_seed': array,
                         'hwa_arm': ref['hwa_arm'], 'initial': copy.deepcopy(initial), 'final': final,
                         'initial_apparent_sha256': p0,
                         'final_apparent_sha256': 'learned' if arm.trains else 'rewritten' if rewrite else p0,
                         'fault_rewrite': rewrite, 'cost': {}})
    return rows, [{'stage': 'onchip_memory', 'peak_reserved_bytes': 1000},
                  {'stage': 'onchip', 'terminal': True, 'elapsed_seconds': 100.0}]


def reports():
    return [pilot.checked_report(label, *evidence(label), 'receipt.json') for label in pilot.labels('random3')]


def previous(tmp_path, reports):
    rows = []
    for report in reports:
        for row in report['rows']:
            if row['arm'] == 'none':
                for cap in ('cap4', 'uncapped'):
                    rows.append({**row, 'cap': cap, 'effects': {
                        split: {'deployment': row['final'][split], 'onchip': row['final'][split]}
                        for split in pilot.SPLITS}})
    (tmp_path / 'reviews').mkdir(exist_ok=True)
    (tmp_path / 'reviews/summary.json').write_text(json.dumps({'rows': rows}))
    return tmp_path


def test_random3_has_nine_jobs_and_preserves_all_upstream_contracts():
    assert len(pilot.labels('random3')) == 9
    assert not any(label.endswith('-deploy') for label in pilot.labels('random3'))
    for label in pilot.labels('random3'):
        tag, array, family = pilot.parts(label)
        assert array in (271002, 271003, 271004)
        lc = pilot.lifecycle(tag, family)
        original = pilot.exp13_lifecycle(f'{tag}-{array}-cap4').to_dict()
        for key, value in original.items():
            if key not in ('lifecycle_id', 'question', 'onchip'):
                assert lc.to_dict()[key] == value
        assert [c.label for c in lc.onchip_cases] == ['nominal', 'random_30000ppm']
        assert len(lc.onchip.arms) == 5
        after = lc.onchip.to_dict()
        assert {k: v for k, v in after.items() if k not in ('arms', 'case_labels')} == {
            k: v for k, v in original['onchip'].items() if k != 'arms'}


def test_full_lifecycles_remain_byte_equivalent_to_prepared_definitions():
    for tag in pilot.TAGS:
        for family in ('random', 'kinds'):
            lc = pilot.lifecycle(tag, family)
            recorded = json.loads((PATH.parents[1] / 'lifecycles' / (lc.lifecycle_id + '.json')).read_text())
            assert lc.to_dict() == recorded


def test_only_std3_first_array_is_a_canary():
    doc = pilot.campaign_manifest(pilot.PHASE + '-random3', pilot.labels('random3'),
        Path('/worker'), Path('/evidence'), 'frozen', ['fifi'], pilot.BUDGETS['random3'])
    canary = pilot.PREFIX + '-std3-271002-random3'
    assert len(doc['cases']) == 9
    for case in doc['cases']:
        assert case['dependencies'] == ([] if case['id'] == canary else [canary])
        assert case['required_gates'] == (['preflight'] if case['id'] == canary else ['handoff'])


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unexpected'])
def test_collector_rejects_wrong_endpoint_coverage(change):
    label = 'std3-271002-random3'
    rows, records = evidence(label)
    if change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[-1]['case'] = 'random_10000ppm'
    with pytest.raises(ValueError, match='coverage'):
        pilot.checked_report(label, rows, records, 'receipt.json')


@pytest.mark.parametrize('field', ['hwa_arm', 'array_seed', 'initial_apparent_sha256'])
def test_collector_rejects_wrong_binding(field):
    label = 'std3-271002-random3'
    rows, records = evidence(label)
    rows[1][field] = 'wrong'
    with pytest.raises(ValueError, match='P0'):
        pilot.checked_report(label, rows, records, 'receipt.json')


def test_summary_reports_only_phase_complete_and_rejects_duplicates(tmp_path):
    data = reports()
    old = previous(tmp_path, data)
    summary = pilot.summarize(data, old, 'random3')
    assert summary['coverage_complete'] and not summary['full_study_complete']
    assert summary['phase'] == 'random3'
    assert summary['starting_conditions'] == 18 and summary['endpoints'] == 90
    assert summary['held_reproduction']['reproduced']
    with pytest.raises(ValueError, match='coverage'):
        pilot.summarize(data + [data[0]], old, 'random3')
    duplicated = copy.deepcopy(data)
    duplicated[0]['rows'].append(duplicated[0]['rows'][0])
    with pytest.raises(ValueError, match='coverage'):
        pilot.summarize(duplicated, old, 'random3')
    with pytest.raises(ValueError, match='coverage'):
        pilot.summarize(data, old, 'full')


def test_summary_rejects_changed_held_metrics(tmp_path):
    data = reports()
    old = previous(tmp_path, data)
    data[0]['rows'][0]['final']['transfer_wikitext']['teacher_kl'] *= 1.001
    with pytest.raises(ValueError, match='reproduce'):
        pilot.summarize(data, old, 'random3')


def test_queue_prepares_only_authorized_phase_with_isolated_outputs(tmp_path):
    evidence_root = tmp_path / 'evidence/random3'
    (evidence_root / 'readiness').mkdir(parents=True)
    (evidence_root / 'readiness/execution-window.json').write_text(json.dumps({
        'deadline': '2026-10-14T20:00:00+00:00'}))
    cases = []
    for label in pilot.labels('random3'):
        tag, _, family = pilot.parts(label)
        cases.append({'label': label, 'dry_run_passed': True,
                      'lifecycle_sha256': pilot.lifecycle(tag, family).digest(),
                      'placement': {'workdir': '/worker', 'run_dir': '/receipts/' + label}})
    receipt = tmp_path / 'host.json'
    receipt.write_text(json.dumps({'host': 'fifi', 'source': '/worker',
                                  'source_revision': 'frozen', 'phase': 'random3', 'cases': cases}))
    out = pilot.queue([receipt], evidence_root / 'prepared/queue', Path('/controller'), evidence_root, 'random3')
    assert len(out['jobs']) == 9
    for path in out['jobs']:
        job = json.loads(Path(path).read_text())
        assert job['id'].endswith('-random3')
        assert job['authorization'] == pilot.RANDOM3_AUTHORIZATION
        assert job['budget_seconds'] == 2400
        assert str(evidence_root) in job['completion']['argv']
    manifest = json.loads(Path(out['manifest']).read_text())
    assert manifest['budgets']['storage_bytes'] == 200_000_000_000
    assert any(r['path'].endswith('/exp16-reused') for r in manifest['storage_roots'])
    with pytest.raises(ValueError, match='authorization'):
        pilot.queue([receipt], tmp_path / 'unapproved', Path('/controller'), evidence_root, 'main')
