"""Scientific identity, negative outcomes and matched recovery aggregation."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/'campaigns/opt-cmo-corruption/launch/exp7_recovery_regimes.py'
spec = importlib.util.spec_from_file_location('recovery_regimes', PATH)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def evidence(label):
    tag, array, cap = pilot.parts(label)
    ref = json.loads(pilot.CATALOG.read_text())[f'{tag}-{array}']
    rows=[]
    for case in pilot.lifecycle(label).defects.cases:
        for arm in ('none','onchip_weights'):
            initial=copy.deepcopy(ref['cases'][case.label]['initial'])
            if arm!='none':
                for value in initial.values(): value['teacher_kl']*=.5
            rows.append({'case':case.label,'onchip_arm':arm,'array_seed':array,'hwa_arm':ref['hwa_arm'],
                'sequence_presentations':8192 if arm!='none' else 0,
                'initial_apparent_sha256':ref['cases'][case.label]['apparent_sha256'],
                'final_apparent_sha256':ref['cases'][case.label]['apparent_sha256'], 'final':initial,
                'cost':{} if arm=='none' else {'max_onchip_device_writes':1,'max_onchip_device_write_attempts':1,
                    'device_writes':4,'stuck_write_attempts':1,'issued_write_attempts':5}})
    return rows,[{'stage':'onchip_memory','peak_reserved_bytes':1000},
                 {'stage':'onchip','terminal':True,'elapsed_seconds':100.0}]


def reports():
    return [pilot.checked_report(label,*evidence(label),'receipt.json') for label in pilot.labels()]


def test_upstream_contract_preserved_and_only_three_assignments_execute():
    assert len(pilot.labels())==27
    for label in pilot.labels():
        tag,array,cap=pilot.parts(label)
        assert array in (271002,271003,271004)
        before=json.loads((PATH.parents[1]/f'lifecycles/opt125m-cmo-l6-exp1g-{tag}.json').read_text())
        after=pilot.lifecycle(label).to_dict()
        for key in before:
            if key not in ('lifecycle_id','question','onchip'): assert after[key]==before[key]
        assert after['onchip'].get('max_write_attempts_per_device')==pilot.CAPS[cap]


def test_local_placement_bounded_and_canaries_exclusive():
    assert sum(pilot.eligible(label,'local') for label in pilot.labels())==6
    for label,host in pilot.CANARIES.items():
        assert [h for h in pilot.HOSTS if pilot.eligible(label,h)]==[host]


@pytest.mark.parametrize('field,value', [('array_seed',271001),('hwa_arm','wrong'),('initial_apparent_sha256','wrong'),('sequence_presentations',8193)])
def test_mismatched_binding_rejected(field,value):
    rows,records=evidence('std3-271002-cap2');rows[1][field]=value
    with pytest.raises(ValueError,match='Wrong source'):
        pilot.checked_report('std3-271002-cap2',rows,records,'receipt.json')


def test_missing_rate_and_cap_violation_rejected():
    rows,records=evidence('std3-271002-cap2')
    with pytest.raises(ValueError,match='coverage'):
        pilot.checked_report('std3-271002-cap2',rows[:-1],records,'receipt.json')
    rows[1]['cost']['max_onchip_device_write_attempts']=3
    with pytest.raises(ValueError,match='cap'):
        pilot.checked_report('std3-271002-cap2',rows,records,'receipt.json')


def test_complete_summary_and_negative_recovery_are_valid():
    data=reports(); out=pilot.summarize(data)
    assert len(out['rows'])==162 and out['coverage_complete']
    assert all(r['retained_gain']['development']==1 for r in out['rows'])
    uncapped=next(r for r in data if r['label']=='std3-271002-uncapped')
    uncapped['rows'][0]['effects']['development']['kl_reduction']=-.1
    out=pilot.summarize(data)
    matching=[r for r in out['rows'] if r['tag']=='std3' and r['array_seed']==271002 and r['case']=='nominal']
    assert len(matching)==3
    assert all(r['retained_gain']['development'] is None for r in matching)


def test_retained_gain_is_not_clipped_and_duplicate_report_fails():
    data=reports()
    r=next(r for r in data if r['label']=='std3-271002-cap2')['rows'][0]
    r['effects']['development']['kl_reduction']*=2
    out=pilot.summarize(data)
    assert next(r for r in out['rows'] if r['tag']=='std3' and r['array_seed']==271002 and r['cap']=='cap2')['retained_gain']['development']==2
    with pytest.raises(ValueError,match='coverage'):
        pilot.summarize(data[:-1]+[data[0]])


def test_preparation_dependency_and_handoff_gate():
    doc=pilot.campaign_manifest(Path('/worker'),Path('/evidence'),'frozen')
    assert len(doc['cases'])==27
    for case in doc['cases']:
        label=case['id'].removeprefix(pilot.PREFIX+'-')
        if label not in pilot.CANARIES:
            assert case['required_gates']==['handoff']
            assert all(pilot.PREFIX+'-'+c in case['dependencies'] for c in pilot.CANARIES)


def test_uncapped_native_write_statistics_preserve_unrecorded_maximum():
    from exp7_recovery_report import write_statistics, LOGICAL_WEIGHTS
    # Actual uncapped writer schema: no issued total or per-device attempt max.
    cost={'device_writes':45917749,'stuck_write_attempts':1724555,'max_onchip_device_writes':11}
    out=write_statistics(cost)
    assert out['attempts_per_active_cell']==47642304/LOGICAL_WEIGHTS
    assert out['maximum_attempts_per_active_cell'] is None
    assert out['maximum_writes_per_active_cell']==11
    assert out['attempt_total_source']=='successful_plus_stuck'
    with pytest.raises(ValueError,match='Issued attempts'):
        write_statistics({**cost,'issued_write_attempts':45917749})
