"""Warm-up cosine dense recovery: Adam parity, legacy contracts and config admission."""

from dataclasses import asdict, replace
import math

import pytest
import torch

from experiments.artifacts import content_hash
from experiments.opt_cmo_drn.config import Config, Recovery, compatible_config_hashes, parse_config
from experiments.opt_cmo_drn.physical import ConductanceArray
from experiments.opt_cmo_drn.recovery import RecoveryStep, RecoveryWritePort
from training.cmo_direct_adam import DirectWriteAdam
from tests.test_opt_cmo_drn import tiny_model
from tests.test_opt_cmo_drn_bounded_recovery import array_at

DENSE = {'update_rule': 'direct_adam', 'target_bounds': 'gmin_gmax', 'max_write_attempts': None,
         'write_noise_factor': 0., 'learning_rate': 3e-4}


class Port:
    def read(self):return torch.zeros(3)
    def write(self, command):
        self.command = command.clone()
        return command != 0


def test_warmup_cosine_rate_rises_linearly_then_decays_to_zero():
    writer = DirectWriteAdam(Port(), learning_rate=3e-4, schedule=('warmup_cosine', 4, 20))
    assert [writer.rate(t) for t in (1, 2, 4)] == pytest.approx([7.5e-5, 1.5e-4, 3e-4])
    assert writer.rate(12) == pytest.approx(1.5e-4) and writer.rate(20) == pytest.approx(0, abs=1e-20)
    assert all(writer.rate(t) >= writer.rate(t + 1) for t in range(4, 20))
    assert DirectWriteAdam(Port(), learning_rate=3e-4).rate(7) == 3e-4
    for bad in (('cosine', 4, 20), ('warmup_cosine', 20, 20), ('warmup_cosine', -1, 20)):
        with pytest.raises(ValueError):DirectWriteAdam(Port(), learning_rate=3e-4, schedule=bad)


def test_scheduled_commands_match_torch_adam_with_the_same_rates():
    port = Port()
    writer = DirectWriteAdam(port, learning_rate=3e-3, schedule=('warmup_cosine', 2, 6))
    parameter = torch.zeros(3, requires_grad=True)
    reference = torch.optim.Adam([parameter], lr=1., betas=(.9, .999), eps=1e-8, foreach=False)
    for step in range(1, 7):
        gradient = torch.tensor([1., -.5, .2]) * step
        reference.param_groups[0]['lr'] = writer.rate(step)
        previous = parameter.detach().clone()
        parameter.grad = gradient.clone();reference.step();writer.step(gradient)
        torch.testing.assert_close(port.command, parameter.detach() - previous, atol=1e-10, rtol=1e-5)


def test_constant_contract_is_unchanged_and_schedules_resume_exactly():
    assert DirectWriteAdam(Port(), learning_rate=3e-4).contract() == ['cmo_drn.direct_adam.v1', 3e-4]
    source = array_at([9., 14., 40.]).state_dict()
    gradients = [torch.tensor([1., -.4, .3]) * (1 + k % 2) for k in range(12)]
    array = ConductanceArray(source, 'cpu')
    writer = DirectWriteAdam(RecoveryWritePort(array, 0., target_bounds='gmin_gmax'), learning_rate=3e-3,
                             schedule=('warmup_cosine', 2, 12))
    for gradient in gradients[:5]:writer.step(gradient)
    saved, saved_array = writer.state_dict(), array.state_dict()
    for gradient in gradients[5:]:writer.step(gradient)
    clone = ConductanceArray(saved_array, 'cpu')
    resumed = DirectWriteAdam(RecoveryWritePort(clone, 0., target_bounds='gmin_gmax'), learning_rate=3e-3,
                              schedule=('warmup_cosine', 2, 12))
    resumed.load_state_dict(saved)
    for gradient in gradients[5:]:resumed.step(gradient)
    assert torch.equal(array.held, clone.held) and array.counters == clone.counters
    for other in (None, ('warmup_cosine', 3, 12)):
        with pytest.raises(ValueError, match='contract'):
            DirectWriteAdam(RecoveryWritePort(clone, 0., target_bounds='gmin_gmax'), learning_rate=3e-3,
                            schedule=other).load_state_dict(saved)


def test_config_admits_warmup_cosine_only_for_dense_direct_writes_and_keeps_legacy_hashes():
    c = parse_config({'recovery': {**DENSE, 'lr_schedule': 'warmup_cosine', 'warmup_updates': 102}})
    assert c.recovery.lr_schedule == 'warmup_cosine' and c.recovery.warmup_updates == 102
    for change in ({'warmup_updates': 2048}, {'warmup_updates': -1}, {'lr_schedule': 'cosine'},
                   {'update_rule': 'bounded_thresholded_adam', 'write_noise_factor': 1.}):
        with pytest.raises(ValueError):
            parse_config({'recovery': {**DENSE, 'lr_schedule': 'warmup_cosine', 'warmup_updates': 102, **change}})
    with pytest.raises(ValueError):parse_config({'recovery': {**DENSE, 'warmup_updates': 5}})
    constant = parse_config({'recovery': DENSE})
    legacy = asdict(constant)
    for key in ('lr_schedule', 'warmup_updates', 'extends_updates'):legacy['recovery'].pop(key)
    assert content_hash(legacy) in compatible_config_hashes(constant)
    assert content_hash(legacy) not in compatible_config_hashes(c)


def test_recovery_step_builds_the_scheduled_dense_writer():
    from experiments.opt_cmo_drn.devices import program_rewritten
    from experiments.opt_cmo_drn.physical import CmoDrnBank
    model, teacher = tiny_model()
    _, _, state = program_rewritten(model, 271002, 'none', 0., capture_state=True)
    config = replace(Config(), recovery=replace(Recovery(), **DENSE, lr_schedule='warmup_cosine',
                                                warmup_updates=3, updates=10))
    step = RecoveryStep(model, teacher, CmoDrnBank(state, 'cpu'), config)
    assert all(w.schedule == ['warmup_cosine', 3, 10] for w in step.writers)
    step.set_input(torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]))
    step.compute_gradient();step.step()
    assert all(w.diagnostics()['current_rate'] == pytest.approx(1e-4) for w in step.writers)
