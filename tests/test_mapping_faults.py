"""Deployment mapping_scale: remap -> program -> faults -> verify, recorded and reapplied on restore."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ebl.schema.base import ConfigError
from ebl.schema.opt_cmo_drn import Data, Solver, Training, parse_config
from hardware.deployment_config import Deployment, MappedDeployment
from methods.drift_tuning import remap
from methods.programming.cmo import apply_mapping, deployment_report, program_rewritten
from networks.opt.physical import CmoDrnBank
from tests.test_opt_cmo_drn import cpu_threads, tiny_model

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'examples/opt_cmo_drn/fault_paper_grid/paper/drn6-r03-271002.json'
DEPLOY = ROOT / 'examples/opt_cmo_drn/fault_paper_grid/paper/drn6-r03-271002-deploy.json'


def raw(path=TEMPLATE):
    return json.loads(path.read_text())


def test_legacy_identity_unchanged_and_explicit_scale_recorded():
    old = parse_config(raw())
    assert type(old.deployment) is Deployment and old.deployment.mapping_scale == 1.0
    assert 'mapping_scale' not in asdict(old)['deployment'] and not old.deployment.mapped
    r = raw(); r['deployment']['mapping_scale'] = 3
    new = parse_config(r)
    assert isinstance(new.deployment, MappedDeployment) and new.deployment.mapped
    assert asdict(new)['deployment'] == {**asdict(old)['deployment'], 'mapping_scale': 3.0}
    assert type(asdict(new)['deployment']['mapping_scale']) is float


@pytest.mark.parametrize('scale', [0.5, 10.0, True, 'x', float('inf')])
def test_invalid_scale_rejected(scale):
    r = raw(); r['deployment']['mapping_scale'] = scale
    with pytest.raises(ConfigError):
        parse_config(r)


def test_scale_only_on_deploy_and_fault_tuning_stages():
    r = raw(DEPLOY); r['deployment']['mapping_scale'] = 2.0
    assert parse_config(r).deployment.mapping_scale == 2.0
    r['stage'] = 'recover'
    with pytest.raises(ConfigError, match='mapping_scale'):
        parse_config(r)
    drift = json.loads((ROOT / 'examples/opt_cmo_drn/cmo_drift_recovery/full/drn6-ratio3-m4.json').read_text())
    drift['deployment']['mapping_scale'] = 1.0
    with pytest.raises(ConfigError, match='mapping_scale'):
        parse_config(drift)


def test_scale_one_leaves_model_bit_identical(cpu_threads):
    model, _ = tiny_model()
    before = copy.deepcopy(model.state_dict())
    stats = apply_mapping(model, 1.0)
    # Same saturation statistic as the drift-tuning remap (round-off above G_max counts).
    assert stats == remap(copy.deepcopy(model), 1.0)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_scale_three_is_the_drift_tuning_remap(cpu_threads):
    a, _ = tiny_model(); b, _ = tiny_model()
    assert apply_mapping(a, 3.0) == remap(b, 3.0)
    for (k, v), w in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.equal(v, w), k


def test_report_denominators_and_device_counts(cpu_threads):
    model, _ = tiny_model()
    apply_mapping(model, 3.0)
    _, receipt, state = program_rewritten(model, 271001, 'random', .1, capture_state=True)
    report = deployment_report(state, model.settings, model.start)
    assert report['receipt']['faults'] == sum(m['faults'] for m in receipt['matrices']) > 0
    assert report['devices'] == sum(c['held'].numel() for c in state['cells'])
    assert report['target_at_gmax'] > 0
    assert len(report['denominators']['layers']) == len(model.blocks)
    row = report['denominators']['layers'][0]
    for key in ('fault_aware_dh', 'fault_aware_do', 'naive_dh', 'naive_do'):
        assert 0 <= row[key]['median'] <= row[key]['p95'] <= row[key]['max']
    # With an exact copy of the targets as the held state the change is zero.
    exact = copy.deepcopy(state)
    for cell in exact['cells']:
        cell['held'] = cell['target'].clone(); cell['raw'] = cell['target'].clone()
    zero = deployment_report(exact, model.settings, model.start)['denominators']['layers'][0]
    assert zero['fault_aware_dh']['max'] == 0 and zero['naive_do']['max'] == 0


class Store:
    def __init__(self, path):
        self.run_dir = path; path.mkdir(); self.events = []; self.result = None
    def append_metric(self, row): self.events.append(row)
    def artifact_record(self, path, kind): return {'path': str(path), 'kind': kind}
    def complete(self, **kwargs): self.result = kwargs
    def fail(self, error): self.error = error


def configs(scale):
    """Tiny CPU deploy and hold-only fault_tuning configs sharing one deployment."""
    r = raw(); r['device'] = 'cpu'
    if scale is not None:
        r['deployment']['mapping_scale'] = scale
    r['fault_tuning']['arms'] = [{'label': 'naive_hold', 'method': 'naive_hold'},
                                 {'label': 'fault_aware_hold', 'method': 'hold'}]
    r['fault_tuning']['evaluation_cohorts'] = []
    tune = parse_config(r)
    tune = replace(tune, solver=replace(Solver(), input_gain=5., token_chunk_size=3, activation_checkpointing=False),
                   training=replace(tune.training, effective_batch=2, microbatch=1),
                   data=replace(Data(), sequences=4, development_sequences=2, sequence_length=4),
                   fault_tuning=replace(tune.fault_tuning, sequence_length=4, recovery_sequences=2, fisher_batch=1))
    d = raw(DEPLOY); d['device'] = 'cpu'
    if scale is not None:
        d['deployment']['mapping_scale'] = scale
    deploy = replace(parse_config(d), solver=tune.solver, training=tune.training, data=tune.data)
    return deploy, tune


def deploy_and_tune(tmp_path, monkeypatch, scale, tamper=None):
    import ebl.runtime.opt_cmo_drn.runtime as runtime
    import ebl.runtime.opt_cmo_drn.recovery as recovery
    from ebl.runtime.opt_cmo_drn import fault_tuning as runner
    from ebl.runtime.store import sha256_file
    model, teacher = tiny_model()
    deploy, tune = configs(scale)
    teacher_path = tmp_path / 'teacher.pt'; torch.save({}, teacher_path)
    deploy = replace(deploy, teacher_sha256=sha256_file(teacher_path))
    tune = replace(tune, teacher_sha256=deploy.teacher_sha256)
    source = tmp_path / 'source.pt'; runtime.save(source, runtime.weights_payload(model, deploy))
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [3, 4, 1, 2], [2, 1, 4, 3]])
    monkeypatch.setattr(runtime, 'teacher_model', lambda path, c: copy.deepcopy(teacher))
    monkeypatch.setattr(recovery, 'cohorts', lambda c: (tokens, tokens[:2], {'training': 'tiny', 'development': 'tiny'}))
    out = tmp_path / f'deploy-{scale}'
    request = SimpleNamespace(spec=deploy, teacher_weights=teacher_path, weights=source, resume=None,
                              device_state=None, output_dir=out, command=('test',))
    assert runtime.run_train(request) == 0
    run_dir = next(out.iterdir())
    deployed = run_dir / 'checkpoints/deployed.pt'
    if tamper:
        state = torch.load(deployed, weights_only=True); tamper(state); torch.save(state, deployed)
    store = Store(tmp_path / f'tune-{scale}')
    monkeypatch.setattr(runner, 'cohorts', lambda settings: (tokens[:2], tokens[2:]))
    monkeypatch.setattr(runner, 'load_samples', lambda *args: (torch.randint(0, 17, (2, 3, 8)), source))
    monkeypatch.setattr(runner.runtime, 'configure_runtime', lambda *args: None)
    monkeypatch.setattr(runner, '_store', lambda *args: store)
    tune_request = SimpleNamespace(spec=tune, weights=source, teacher_weights=teacher_path, device_state=deployed)
    assert runner.run(tune_request) == 0
    return run_dir, deployed, store, source, tokens


def test_scale_one_deploy_and_holds_equal_legacy(tmp_path, monkeypatch, cpu_threads):
    (tmp_path / 'a').mkdir(); (tmp_path / 'b').mkdir()
    legacy_dir, legacy, legacy_store, _, _ = deploy_and_tune(tmp_path / 'a', monkeypatch, None)
    mapped_dir, mapped, mapped_store, _, _ = deploy_and_tune(tmp_path / 'b', monkeypatch, 1.0)
    a = torch.load(legacy, weights_only=True); b = torch.load(mapped, weights_only=True)
    assert 'mapping' not in a and b['mapping']['scale'] == 1.0
    assert a['start_held_hashes'] == b['start_held_hashes'] and a['programming'] == b['programming']
    assert a['development'] == b['development'] and a['raw_development'] == b['raw_development']
    for x, y in zip(a['bank']['cells'], b['bank']['cells']):
        for key in ('held', 'target', 'raw'):
            assert torch.equal(x[key], y[key])
    assert b['deployment'] == {**a['deployment'], 'mapping_scale': 1.0}
    report = json.loads((mapped_dir / 'artifacts/deployment_report.json').read_text())
    assert report['report']['receipt']['faults'] > 0
    ra, rb = legacy_store.result['metrics']['arms'], mapped_store.result['metrics']['arms']
    strip = lambda rows: [{k: v for k, v in r['development'].items() if k != 'seconds'} for r in rows]
    assert strip(ra) == strip(rb) and len(ra) == 2
    identity = mapped_store.result['metrics']
    assert identity['clean_reference']['source_ideal'] == identity['clean_reference']['mapped_targets']
    assert 'clean_reference' not in legacy_store.result['metrics']


def test_scale_three_restore_keeps_targets_and_current_scale(tmp_path, monkeypatch, cpu_threads):
    _, deployed, store, _, tokens = deploy_and_tune(tmp_path, monkeypatch, 3.0)
    saved = torch.load(deployed, weights_only=True)
    assert saved['mapping']['scale'] == 3.0
    # The deployment is exactly the drift-tuning ratio remap followed by today's programming.
    model, teacher = tiny_model(); remap(model, 3.0)
    _, receipt, state = program_rewritten(model, 271002, 'random', .03, capture_state=True)
    assert receipt == saved['programming']
    assert saved['current_scales'] == [float(b.current_scale) for b in model.blocks]
    reference = tiny_model()[0]
    assert all(s == float(b.current_scale * 3) for s, b in zip(saved['current_scales'], reference.blocks))
    for x, y in zip(state['cells'], saved['bank']['cells']):
        assert torch.equal(x['held'], y['held'])
    # Restored holds score the remapped model on the saved bank (bias current x3).
    from ebl.runtime.opt_cmo_drn import runtime
    bank = CmoDrnBank(saved['bank'], 'cpu')
    expected = runtime.evaluate(model, teacher, tokens[2:], 'cpu', bank.view(model), 1)['teacher_kl']
    arms = store.result['metrics']['arms']
    assert arms[1]['development']['teacher_kl'] == expected
    assert store.result['metrics']['current_scales'] == saved['current_scales']


@pytest.mark.parametrize('field', ['current_scales', 'mapping', 'target'])
def test_restore_rejects_inconsistent_mapping(tmp_path, monkeypatch, cpu_threads, field):
    def tamper(state):
        if field == 'current_scales':
            state['current_scales'][0] *= 1.5
        elif field == 'mapping':
            state['mapping']['scale'] = 2.0
        else:
            state['bank']['cells'][0]['target'] = state['bank']['cells'][0]['target'] * 2
    with pytest.raises(ValueError, match='mapping'):
        deploy_and_tune(tmp_path, monkeypatch, 3.0, tamper)
