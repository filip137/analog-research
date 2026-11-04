"""Naive P&V hold: the pre-compensation held state, scored through the public validation path."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ebl.schema.opt_cmo_drn import Config, Data, Recovery, Solver, Training, parse_config
from tests.test_opt_cmo_drn import tiny_model


def test_naive_hold_scores_raw_programming_without_writes(tmp_path, monkeypatch):
    import ebl.runtime.opt_cmo_drn.runtime as runtime
    import ebl.runtime.opt_cmo_drn.recovery as recovery
    from ebl.runtime.store import sha256_file
    model, teacher = tiny_model()
    teacher_path = tmp_path / 'teacher.pt';torch.save({}, teacher_path)
    config = replace(Config(), stage='deploy', teacher_sha256=sha256_file(teacher_path),
                     solver=replace(Solver(), input_gain=5., token_chunk_size=3, activation_checkpointing=False),
                     training=replace(Training(), effective_batch=2, microbatch=1),
                     data=replace(Data(), sequences=4, development_sequences=2, sequence_length=4))
    source = tmp_path / 'source.pt';runtime.save(source, runtime.weights_payload(model, config))
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [3, 4, 1, 2], [2, 1, 4, 3]])
    monkeypatch.setattr(runtime, 'teacher_model', lambda path, c: copy.deepcopy(teacher))
    monkeypatch.setattr(recovery, 'cohorts', lambda c: (tokens, tokens[:2], {'training': 'tiny', 'development': 'tiny'}))
    monkeypatch.setattr(recovery, 'TRANSFER_TESTS', (('test', 'tiny', 'pinned', 2),))
    monkeypatch.setattr(recovery, 'load_corpus', lambda name, digest: {'test': tokens.flatten()})
    request = SimpleNamespace(spec=config, teacher_weights=teacher_path, weights=source, resume=None, device_state=None,
                              output_dir=tmp_path / 'deploy', command=('test',))
    assert runtime.run_train(request) == 0
    canonical = next(request.output_dir.iterdir()) / 'checkpoints/deployed.pt'
    deployed = torch.load(canonical, weights_only=True)
    assert any(not torch.equal(c['raw'], c['held']) for c in deployed['bank']['cells'])

    request.spec = replace(config, stage='recover', recovery=replace(Recovery(), arm='naive_hold'))
    request.device_state, request.output_dir = canonical, tmp_path / 'naive'
    assert runtime.run_train(request) == 0
    final = torch.load(next(request.output_dir.iterdir()) / 'checkpoints/final.pt', weights_only=True)
    assert final['arm'] == 'naive_hold' and final['step'] == 0 and final['costs']['write_attempts'] == 0
    assert final['initial_device_state_sha256'] == sha256_file(canonical)
    for saved, original in zip(final['bank']['cells'], deployed['bank']['cells']):
        assert torch.equal(saved['held'], original['raw']) and torch.equal(saved['raw'], original['raw'])
    assert final['development'] == deployed['raw_development']

    validation = SimpleNamespace(spec=request.spec, teacher_weights=teacher_path,
                                 weights=next(request.output_dir.iterdir()) / 'checkpoints/final.pt',
                                 output_dir=tmp_path / 'validate', command=('test',))
    assert runtime.run_validate(validation) == 0

    # A recovery output is not a deployment; the naive hold refuses it.
    request.device_state = next((tmp_path / 'naive').iterdir()) / 'checkpoints/final.pt'
    request.output_dir = tmp_path / 'again'
    with pytest.raises(ValueError):runtime.run_train(request)


def test_naive_hold_is_a_declared_recovery_arm():
    assert parse_config({'recovery': {'arm': 'naive_hold'}}).recovery.arm == 'naive_hold'
    with pytest.raises(ValueError):parse_config({'recovery': {'arm': 'naive'}})
