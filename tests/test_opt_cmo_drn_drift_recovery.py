import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
import torch
from experiments.artifacts import sha256_file
from experiments.opt_cmo_drn import runtime, drift_recovery
from experiments.opt_cmo_drn.config import Config, Solver, Data, Deployment, parse_config
from methods.maintenance.config import DriftRecovery
from test_opt_cmo_drn import tiny_model, cpu_threads


def test_drn_adapter_parity_and_physical_gradient():
    model, _ = tiny_model()
    adapter = drift_recovery.DrnAdapter(model, 'cpu')
    physical, _, _, _ = adapter.physical_layout()
    tokens = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(adapter.clean(tokens), adapter.forward(tokens, physical), atol=2e-5, rtol=2e-5)
    physical[0].requires_grad_(True)
    adapter.forward(tokens, physical).square().mean().backward()
    assert physical[0].grad is not None and float(physical[0].grad.abs().sum()) > 0
    assert all(p.grad is None for p in model.parameters())


def test_native_drn_repeated_recovery(tmp_path, monkeypatch):
    model, teacher = tiny_model()
    teacher_path = tmp_path/'teacher.pt'
    torch.save({}, teacher_path)
    config = replace(Config(), teacher_sha256=sha256_file(teacher_path),
        solver=replace(Solver(), input_gain=5., token_chunk_size=3, activation_checkpointing=False),
        data=replace(Data(), sequence_length=4, development_sequences=2),
        deployment=Deployment(271002, 'none', 0))
    source = tmp_path/'source.pt'
    runtime.save(source, runtime.weights_payload(model, config))
    settings = replace(DriftRecovery(sha256_file(source)), persistent_fractions=(.9,),
        protocols=('repeated',), arms=('full_refresh', 'stable_learning', 'random_learning'),
        write_budgets=(1.,), horizon_seconds=9000., maintenance_interval_seconds=3600.,
        followup_seconds=(60.,), evaluation_seconds=(0., 3600., 9000.),
        training_sequences=4, validation_sequences=2, calibration_sequences=2,
        batch_size=2, microbatch_size=1, max_updates=2, learning_rate=.001,
        no_drift_control=False, canary=True)
    config = replace(config, stage='drift_recovery', drift_recovery=settings)
    assert parse_config(json.loads(json.dumps(asdict(config)))) == config
    streams = {'train': torch.tensor([1,2,3,4,4,3,2,1,2,3,4,5,5,4,3,2]),
               'validation': torch.tensor([2,4,1,3,3,1,4,2]),
               'test': torch.tensor([3,4,5,6,6,5,4,3])}
    monkeypatch.setattr(runtime, 'teacher_model', lambda *a: copy.deepcopy(teacher))
    monkeypatch.setattr(drift_recovery, 'load_corpus', lambda *a: streams)
    request = SimpleNamespace(spec=config, teacher_weights=teacher_path, weights=source, resume=None,
        device_state=None, output_dir=tmp_path/'runs', command=('test',))
    assert runtime.run_train(request) == 0
    directory = next(request.output_dir.iterdir())
    result = json.loads((directory/'result.json').read_text())
    assert result['metrics']['source_unchanged'] and result['metrics']['trajectories'] == 3
    trajectories = json.loads((directory/'drift_recovery_trajectories.json').read_text())
    assert all(len(t['events']) == 2 and t['replay_exact'] for t in trajectories)
    assert all(e['changed_cell_commands'] > 0 for t in trajectories if 'learning' in t['trajectory'] for e in t['events'])
    from workflow.__main__ import collect
    assert collect([directory])['complete']
