import json
import pytest
from experiments.schema import ConfigError
from workflow.__main__ import main as workflow_main
from workflow.lifecycle import parse_lifecycle, provenance, verify_provenance
from test_workflow_cmo import cmo_inputs, small, case_doc
from workflow_helpers import run_lifecycle
from test_power_law_drift import spec


def settings():
    return {'models':['reram_cmo'],'seconds':[3600,86400],'assignment_seeds':[271001],
            'case_labels':['nominal'],'seed':391028,'read_noise':False,'power_law':spec().to_dict()}


def test_native_power_law_and_collection(cmo_inputs,tmp_path):
    weights,digest=cmo_inputs
    lc=small(**{'onchip':None,'defects.cases':[case_doc()], 'retention':settings(),
        'data.corpus':{'name':'synthetic','sha256':digest},'data.sequence_length':9,'data.max_examples':4,
        'hwa.arms':[{'id':'digital','method':'none'}]})
    doc=lc.to_dict();doc.pop('retention');parent=parse_lifecycle(doc)
    verify_provenance(provenance(parent,'hwa_source',hwa_arm='digital'),lc,'hwa_source',hwa_arm='digital')
    result=run_lifecycle(lc,weights,tmp_path/'runs')
    (path,bundle),= [v for k,v in result.items() if k.startswith('retention__')]
    assert bundle['metrics']['source_unchanged']
    rows=json.loads((path/'retention_measurements.json').read_text())
    assert len(rows)==9 and all(r['replay_exact'] for r in rows)
    assert rows[0]['perplexity_ratio']==1
    assert all(r['metrics']['development']['sequences']==2 for r in rows)
    assert workflow_main(['collect',str(tmp_path/'runs'),'--output',str(tmp_path/'collected')])==0
    collected=json.loads((tmp_path/'collected/retention.json').read_text())
    assert len(collected)==9 and {r['spread_us'] for r in collected}=={0,1,3,12}


@pytest.mark.parametrize('key,value',[('models',['pcm']),('read_noise',True),('case_labels',['random_50000ppm'])])
def test_rejects_unsupported_scalar_regimes(key,value):
    s=settings();s[key]=value
    with pytest.raises(ConfigError):small(**{'onchip':None,'retention':s})


def calibration(**overrides):
    value = {'methods': ['global', 'global_moments', 'mlp_gain', 'mlp_shift', 'mlp_moments'],
             'calibration_sequences': 4, 'row_column_iterations': 10, 'gain_bounds': [.1, 10.],
             'seconds': [86400], 'sigmas_us': [1, 12]}
    value.update(overrides)
    return value


def test_power_law_calibration_replays_parent_and_collects(cmo_inputs, tmp_path):
    from workflow.lifecycle import StageSelection
    from workflow_helpers import run_stage
    weights, digest = cmo_inputs
    lc = small(**{'onchip': None, 'defects.cases': [case_doc()], 'retention': settings(),
                  'data.corpus': {'name': 'synthetic', 'sha256': digest}, 'data.sequence_length': 9,
                  'data.max_examples': 4, 'hwa.arms': [{'id': 'digital', 'method': 'none'}]})
    results = run_lifecycle(lc, weights, tmp_path / 'parent')
    doc = lc.to_dict(); doc['drift_compensation'] = calibration(); doc['lifecycle_id'] += '-calibration'
    comp = parse_lifecycle(doc)
    run_dir, result = run_stage(comp, StageSelection('compensate', 'digital', 271001), results, weights, tmp_path / 'comp')
    assert result['metrics']['control_exact'] and result['metrics']['source_unchanged']
    rows = json.loads((run_dir / 'compensation_measurements.json').read_text())
    assert len(rows) == 10 and {(r['seconds'], r['spread_us']) for r in rows} == {(86400, 1), (86400, 12)}
    parent = json.loads((results['retention__digital__271001'][0] / 'retention_measurements.json').read_text())
    prior = {(r['seconds'], r['spread_us']): r for r in parent}
    for row in rows:
        assert row['uncompensated'] == prior[row['seconds'], row['spread_us']]['metrics']
        assert row['cost']['device_writes'] == 0 and row['state_unchanged']
    worst = {r['method']: r['compensated']['development']['teacher_kl'] for r in rows if r['spread_us'] == 12}
    assert worst['mlp_moments'] < next(r for r in rows if r['spread_us'] == 12)['uncompensated']['development']['teacher_kl']
    assert workflow_main(['collect', str(tmp_path / 'comp'), '--output', str(tmp_path / 'collected')]) == 0
    flat = json.loads((tmp_path / 'collected/compensation.json').read_text())
    assert len(flat) == 10 and {r['spread_us'] for r in flat} == {1, 12}


@pytest.mark.parametrize('overrides', [{'seconds': [7200]}, {'sigmas_us': [2]}, {'methods': ['mlp_bad']}])
def test_power_law_calibration_rejects_points_outside_parent_grid(overrides):
    doc = small(**{'onchip': None, 'retention': settings()}).to_dict()
    doc['drift_compensation'] = calibration(**overrides)
    with pytest.raises(ConfigError):
        parse_lifecycle(doc)


def test_output_methods_require_power_law():
    from test_workflow_retention import lifecycle
    spec = calibration(); spec.pop('seconds'); spec.pop('sigmas_us')
    with pytest.raises(ConfigError):
        lifecycle(**{'drift_compensation': spec})
    with pytest.raises(ConfigError):
        lifecycle(**{'drift_compensation': calibration(methods=['global'])})
