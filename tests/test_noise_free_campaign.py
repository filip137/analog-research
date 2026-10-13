"""exp-015 identity, anchor equality, C-validity guard and A/B/C decomposition inputs."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/'campaigns/opt-cmo-corruption/launch/exp15_noise_free.py'
spec = importlib.util.spec_from_file_location('noise_free', PATH)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
LIFECYCLES = PATH.parents[1]/'lifecycles'


def development(case, values=(.3, .2, .1, .09)):
    return [{'stage':'onchip_development','case':case,'arm':'onchip_weights','global_step':64*(i+1),
             'development':{'teacher_kl':v}} for i, v in enumerate(values)]


def evidence(label, factor=.5, tail=(.3, .2, .1, .09)):
    tag, array, _ = oracle.parts(label)
    ref = json.loads(oracle.CATALOG.read_text())[f'{tag}-{array}']
    rows, records = [], [{'stage':'onchip_memory','peak_reserved_bytes':1000},
                         {'stage':'onchip','terminal':True,'elapsed_seconds':100.0}]
    for case in oracle.lifecycle(label).defects.cases:
        records += development(case.label, tail)
        for arm in ('none','onchip_weights'):
            final = copy.deepcopy(ref['cases'][case.label]['initial'])
            for value in final.values():
                value['prediction_sha256'] = f'{case.label}-{arm}'
                if arm != 'none': value['teacher_kl'] *= factor
            rows.append({'case':case.label,'onchip_arm':arm,'array_seed':array,'hwa_arm':ref['hwa_arm'],
                'sequence_presentations':8192 if arm != 'none' else 0,
                'initial_apparent_sha256':ref['cases'][case.label]['apparent_sha256'],
                'final_apparent_sha256':ref['cases'][case.label]['apparent_sha256'] if arm == 'none' else 'learned',
                'state_sha256':f'state-{case.label}-{arm}','final':final,
                'cost':{} if arm == 'none' else {'device_writes':4,'stuck_write_attempts':1,'verify_reads':0}})
    return rows, records


def test_arms_change_only_onchip_and_anchor_is_the_exp013_lifecycle():
    assert len(oracle.labels()) == 19
    exp13 = json.loads((LIFECYCLES/'opt125m-cmo-l6-exp7-recovery-regimes-std3-uncapped.json').read_text())
    assert oracle.lifecycle(oracle.ANCHOR).to_dict() == exp13
    for label in oracle.labels()[1:]:
        tag, _, arm = oracle.parts(label)
        before = json.loads((LIFECYCLES/f'opt125m-cmo-l6-exp7-recovery-regimes-{tag}-uncapped.json').read_text())
        after = oracle.lifecycle(label).to_dict()
        assert after == json.loads((LIFECYCLES/f'{after["lifecycle_id"]}.json').read_text())
        for key in before:
            if key not in ('lifecycle_id', 'question', 'onchip'): assert after[key] == before[key]
        expected = dict(before['onchip'], write_noise_scale=0.)
        if arm == 'dense':
            expected['update_law'] = 'cmo_idealized_write_adam'
            del expected['threshold_sigma'], expected['accumulate_updates']
        assert after['onchip'] == expected


def test_local_runs_only_the_anchor_and_canaries_are_exclusive():
    assert [label for label in oracle.labels() if oracle.eligible(label, 'local')] == [oracle.ANCHOR]
    for label, host in oracle.CANARIES.items():
        assert [h for h in oracle.HOSTS if oracle.eligible(label, h)] == [host]


def test_queue_refuses_without_recorded_authorization(tmp_path, monkeypatch):
    monkeypatch.setattr(oracle, "AUTHORIZATION", None)
    with pytest.raises(ValueError, match='authorization'):
        oracle.queue([], tmp_path, tmp_path, tmp_path)


def test_development_tail_flags_a_final_rise():
    rows, records = evidence('std3-271002-dense', tail=(.3, .2, .1, .11))
    report = oracle.checked_report('std3-271002-dense', rows, records, 'receipt.json')
    assert all(r['development_tail']['rising'] and r['development_tail']['steps'] == [192, 256] for r in report['rows'])


def test_anchor_comparison_requires_exact_hashes_and_metrics(tmp_path, monkeypatch):
    rows, records = evidence(oracle.ANCHOR)
    folder = tmp_path/'collected/opt125m-cmo-l6-exp7-recovery-regimes-std3-271002-uncapped/x'
    folder.mkdir(parents=True); (folder/'measurements.json').write_text(json.dumps(rows))
    monkeypatch.setattr(oracle, 'EXP13', tmp_path)
    assert oracle.checked_report(oracle.ANCHOR, rows, records, 'receipt.json')['anchor']['exact']
    changed = copy.deepcopy(rows); changed[3]['final']['transfer_wikitext']['prediction_sha256'] = 'other'
    result = oracle.checked_report(oracle.ANCHOR, changed, records, 'receipt.json')['anchor']
    assert not result['exact'] and result['mismatches'] == [
        {'case':changed[3]['case'], 'arm':'onchip_weights', 'field':'transfer_wikitext.prediction_sha256'}]


def test_summary_decomposes_against_exp013_and_applies_the_c_guard(tmp_path, monkeypatch):
    reports = []
    for label in oracle.labels()[1:]:
        tag, array, arm = oracle.parts(label)
        # C0 above D for std9 on 271004 violates the guard for that source only.
        factor = .4 if arm == 'dense' else .5
        if (tag, array, arm) == ('std9', 271004, 'dense'): factor = 1.1
        reports.append(oracle.checked_report(label, *evidence(label, factor), 'receipt.json'))
    a_rows = [{'tag':r['tag'],'array_seed':r['array_seed'],'rate_ppm':r['rate_ppm'],'cap':'uncapped',
               'initial_apparent_sha256':r['initial_apparent_sha256'],
               'effects':{s:{'onchip':{'teacher_kl':.7*e['deployment']['teacher_kl']}} for s,e in r['effects'].items()}}
              for report in reports if report['label'].endswith('thresholded') for r in report['rows']]
    (tmp_path/'reviews').mkdir(); (tmp_path/'reviews/summary.json').write_text(json.dumps({'rows':a_rows}))
    monkeypatch.setattr(oracle, 'EXP13', tmp_path)
    summary = oracle.summarize(reports)
    assert summary['exact_write_endpoints'] == 108 and len(summary['units']) == 162
    for unit in summary['units']:
        assert unit['N'] + unit['T'] + unit['F'] == pytest.approx(unit['A'] - unit['C0'])
        assert (unit['shares'] is None) == (unit['rate_ppm'] == 0 or unit['A'] <= unit['C0'])
    assert {k:v['valid'] for k,v in summary['c_validity'].items()} == {'std3':True, 'cdt3':True, 'std9':False}
    assert {v['kind'] for v in summary['c_validity']['std9']['violations']} == {'C0_above_D'}
