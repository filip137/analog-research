import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
import pytest
import torch
from experiments.artifacts import sha256_file
from experiments.opt_cmo_drn import runtime, retention
from experiments.opt_cmo_drn.config import Config, Solver, Data, Deployment, parse_config
from experiments.opt_cmo_drn.retention_config import Retention
from test_opt_cmo_drn import tiny_model, cpu_threads
from test_power_law_drift import spec


def test_native_drn_power_law(tmp_path,monkeypatch):
    model,teacher=tiny_model()
    teacher_path=tmp_path/'teacher.pt';torch.save({},teacher_path)
    c=replace(Config(),stage='retention',teacher_sha256=sha256_file(teacher_path),
              solver=replace(Solver(),input_gain=5.,token_chunk_size=3,activation_checkpointing=False),
              data=replace(Data(),sequence_length=4,development_sequences=2),
              deployment=Deployment(271002,'none',0))
    path=tmp_path/'source.pt';runtime.save(path,runtime.weights_payload(model,c))
    c=replace(c,retention=Retention((3600.,86400.),391028,sha256_file(path),spec()))
    assert parse_config(asdict(c))==c
    tokens=torch.tensor([[1,2,3,4],[4,3,2,1]])
    monkeypatch.setattr(runtime,'teacher_model',lambda path,c:copy.deepcopy(teacher))
    monkeypatch.setattr(retention,'load_corpus',lambda *a:{'validation':tokens.flatten()})
    request=SimpleNamespace(spec=c,teacher_weights=teacher_path,weights=path,resume=None,
                            device_state=None,output_dir=tmp_path/'runs',command=('test',))
    assert runtime.run_train(request)==0
    run=next(request.output_dir.iterdir())
    bundle=json.loads((run/'result.json').read_text())
    assert bundle['metrics']['source_unchanged']
    rows=json.loads((run/'retention_measurements.json').read_text())
    assert len(rows)==9 and rows[0]['perplexity_ratio']==1
    assert all(r['replay_exact'] and r['cost']['retention_device_writes']==0 for r in rows)
    assert all(r['physical_devices']==4*r['logical_weights'] for r in rows)
    from workflow.__main__ import collect
    assert collect([run])['complete']


def test_programming_is_unrewritten_and_held():
    model,_=tiny_model()
    a,view,n=retention.program(model,271002,391028)
    b,_,m=retention.program(model,271002,391028)
    assert n==m
    for x,y in zip(a,b):
        assert torch.equal(x['programmed'],y['programmed'])
        assert torch.equal(x['epsilon'],y['epsilon'])
    assert all(float(x['programmed'].min())>=9 for x in a)


def calibrated_parent(tmp_path,monkeypatch):
    from experiments.opt_cmo_drn import calibration
    from experiments.opt_cmo_drn.retention_config import Calibration
    model,teacher=tiny_model()
    teacher_path=tmp_path/'teacher.pt';torch.save({},teacher_path)
    c=replace(Config(),stage='retention',teacher_sha256=sha256_file(teacher_path),
              solver=replace(Solver(),input_gain=5.,token_chunk_size=3,activation_checkpointing=False),
              data=replace(Data(),sequence_length=4,development_sequences=2),
              deployment=Deployment(271002,'none',0))
    path=tmp_path/'source.pt';runtime.save(path,runtime.weights_payload(model,c))
    c=replace(c,retention=Retention((3600.,86400.),391028,sha256_file(path),spec()))
    tokens=torch.tensor([[1,2,3,4],[4,3,2,1]])
    monkeypatch.setattr(runtime,'teacher_model',lambda path,c:copy.deepcopy(teacher))
    monkeypatch.setattr(retention,'load_corpus',lambda *a:{'validation':tokens.flatten()})
    monkeypatch.setattr(calibration,'cohorts',lambda c:(torch.tensor([[2,4,1,3],[3,1,4,2],[1,1,2,2]]),tokens,{}))
    request=SimpleNamespace(spec=c,teacher_weights=teacher_path,weights=path,resume=None,
                            device_state=None,output_dir=tmp_path/'runs',command=('test',))
    assert runtime.run_train(request)==0
    run=next(request.output_dir.iterdir())
    settings=Calibration(('mlp_gain','mlp_shift','mlp_moments'),3,(86400.,),(1.,12.),(.1,10.))
    child=replace(c,retention=replace(c.retention,calibration=settings))
    return child,SimpleNamespace(**{**vars(request),'spec':child,'device_state':run/'checkpoints/power-law.pt',
                                    'output_dir':tmp_path/'calibration'}),run


def test_native_drn_calibration_replays_parent(tmp_path,monkeypatch):
    child,request,parent=calibrated_parent(tmp_path,monkeypatch)
    assert parse_config(json.loads(json.dumps(asdict(child))))==child
    assert runtime.run_train(request)==0
    run=next(request.output_dir.iterdir())
    bundle=json.loads((run/'result.json').read_text())
    assert bundle['metrics']['control_exact'] and bundle['metrics']['snapshots']==6
    rows=json.loads((run/'calibration_measurements.json').read_text())
    prior={(r['seconds'],r['spread_us']):r for r in json.loads((parent/'retention_measurements.json').read_text())}
    assert {(r['seconds'],r['spread_us'],r['method']) for r in rows}=={(86400.,s,m) for s in (1.,12.)
                                                                      for m in ('mlp_gain','mlp_shift','mlp_moments')}
    for row in rows:
        assert row['uncompensated']==prior[row['seconds'],row['spread_us']]['metrics']
        assert row['cost']['device_writes']==0 and row['cost']['digital_parameters']>0
    assert all(torch.isfinite(torch.tensor(r['compensated']['teacher_kl'])) for r in rows)


def test_drn_calibration_rejects_changed_parent_settings(tmp_path,monkeypatch):
    child,request,_=calibrated_parent(tmp_path,monkeypatch)
    changed=replace(child,retention=replace(child.retention,seed=child.retention.seed+1))
    with pytest.raises(ValueError,match='parent retention setting'):
        runtime.run_train(SimpleNamespace(**{**vars(request),'spec':changed}))


@pytest.mark.parametrize('field,value',[('seconds',[7200]),('sigmas_us',[2]),('methods',['global']),('gain_bounds',[2,3])])
def test_drn_calibration_config_is_strict(field,value):
    c=replace(Config(),stage='retention',deployment=Deployment(271002,'none',0),
              retention=Retention((3600.,86400.),391028,'0'*64,spec()))
    raw=json.loads(json.dumps(asdict(c)))
    raw['retention']['calibration']={'methods':['mlp_gain'],'calibration_sequences':4,'seconds':[86400],
                                     'sigmas_us':[1],'gain_bounds':[.1,10.]}
    parse_config(raw)
    raw['retention']['calibration'][field]=value
    with pytest.raises(Exception):
        parse_config(raw)


def test_drn_decomposed_forward_is_native():
    from training import output_calibration as oc
    model,_=tiny_model()
    tokens=torch.tensor([[1,2,3,4],[4,3,2,1]])
    with torch.no_grad():
        native=model(tokens)
        split=model.head(oc.forward(model.analog_input(tokens),model.mlp_branch,len(model.blocks),None))
    assert torch.equal(native,split)
