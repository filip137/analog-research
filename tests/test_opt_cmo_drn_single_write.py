"""One-attempt nominal-gate writer: shared gate, allowance accounting and resume."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess

import pytest
import torch

from experiments.opt_cmo_drn.config import Config, Recovery, parse_config
from experiments.opt_cmo_drn.physical import ConductanceArray
from experiments.opt_cmo_drn.recovery import RecoveryStep, RecoveryWritePort
from training.cmo_direct_adam import CappedNominalWriteAdam
from training.reram_cmo import G_MAX_US, G_MIN_US, HALF_RANGE_US, MID_US, programming_sigma
from tests.test_opt_cmo_drn import tiny_model
from tests.test_opt_cmo_drn_bounded_recovery import array_at
from workflow import slurm_recovery as s

NOMINAL_Q = 2 * programming_sigma(MID_US, 2.) / HALF_RANGE_US
GOOD = {'update_rule': 'bounded_capped_nominal_adam', 'target_bounds': 'gmin_gmax', 'max_write_attempts': 1}


class Port:
    def __init__(self, size):
        self.size, self.commands = size, []

    def read(self):return torch.zeros(self.size)
    def nominal_write_std_q(self):return NOMINAL_Q / 2
    def write(self, command):
        self.commands.append(command.clone())
        return command != 0


def test_gate_is_the_crossbar_nominal_threshold_for_both_noise_arms():
    assert NOMINAL_Q == pytest.approx(1.1198516946 / HALF_RANGE_US)
    for factor in (0., 1.):
        port = RecoveryWritePort(array_at([20., 40.]), factor, target_bounds='gmin_gmax')
        assert CappedNominalWriteAdam(port, learning_rate=3e-4).threshold_q == NOMINAL_Q


def test_strict_gate_issues_each_device_once_and_freezes_spent_residuals():
    port = Port(3)
    writer = CappedNominalWriteAdam(port, learning_rate=3e-4)
    writer.pending.copy_(torch.tensor([NOMINAL_Q, -2 * NOMINAL_Q, .5 * NOMINAL_Q]))
    writer.step(torch.zeros(3))
    assert torch.equal(port.commands[-1], torch.tensor([0., -2 * NOMINAL_Q, 0.]))
    assert torch.equal(writer.issued_counts, torch.tensor([0, 1, 0], dtype=torch.int8))
    for _ in range(200):writer.step(torch.tensor([-1., 1., 1.]))
    issued = torch.stack(port.commands) != 0
    assert issued.sum(0).tolist() == [1, 1, 1]
    assert writer.pending[1] == 0 and writer.pending[0] == 0 and writer.pending[2] == 0
    diagnostics = writer.diagnostics()
    assert diagnostics['capped_devices'] == 3 and diagnostics['issued_commands'] == 3
    assert diagnostics['threshold_comparison'] == 'gt' and diagnostics['max_write_attempts'] == 1


def test_stuck_cells_and_projected_noops_consume_the_only_attempt():
    array = array_at([40., G_MAX_US, 40.], stuck=2)
    writer = CappedNominalWriteAdam(RecoveryWritePort(array, 1., target_bounds='gmin_gmax'), learning_rate=3e-4)
    writer.pending.copy_(torch.tensor([.1, .1, .1]))
    writer.step(torch.zeros(3))
    assert torch.equal(writer.issued_counts, torch.tensor([1, 1, 1], dtype=torch.int8))
    assert torch.equal(writer.attempt_counts, torch.tensor([1, 0, 1]))
    assert array.counters['write_attempts'] == 2 and array.counters['stuck_write_attempts'] == 1
    before = array.held.clone()
    for _ in range(20):writer.step(torch.tensor([-1., -1., -1.]))
    assert torch.equal(array.held, before) and array.counters['write_attempts'] == 2
    assert not bool(writer.pending.any())


def test_noisy_and_noise_free_arms_share_decisions_rng_and_resume_exactly():
    source = array_at([G_MIN_US, 14., 30., 60., 88., 20.], stuck=5).state_dict()
    gradients = [torch.tensor([-1., .4, -.3, .8, -.6, .2]) * (1 + k % 3) for k in range(40)]
    arms = []
    for factor in (0., 1.):
        array = ConductanceArray(source, 'cpu')
        writer = CappedNominalWriteAdam(RecoveryWritePort(array, factor, target_bounds='gmin_gmax'), learning_rate=3e-3)
        for gradient in gradients[:20]:writer.step(gradient)
        saved, saved_array = writer.state_dict(), array.state_dict()
        for gradient in gradients[20:]:writer.step(gradient)
        clone = ConductanceArray(saved_array, 'cpu')
        resumed = CappedNominalWriteAdam(RecoveryWritePort(clone, factor, target_bounds='gmin_gmax'), learning_rate=3e-3)
        resumed.load_state_dict(saved)
        for gradient in gradients[20:]:resumed.step(gradient)
        assert torch.equal(array.held, clone.held) and array.counters == clone.counters
        assert torch.equal(writer.issued_counts, resumed.issued_counts)
        assert int(writer.attempt_counts.sum()) == array.counters['write_attempts']
        arms.append((array, writer))
    (quiet, a), (noisy, b) = arms
    assert torch.equal(a.issued_counts, b.issued_counts) and int(a.issued_counts.sum()) >= 4
    assert torch.equal(quiet.generator.get_state(), noisy.generator.get_state())
    written = a.issued_counts.bool() & ~torch.tensor([False] * 5 + [True])
    assert not torch.equal(quiet.held[written], noisy.held[written])
    with pytest.raises(ValueError, match='contract'):
        CappedNominalWriteAdam(RecoveryWritePort(quiet, 0., target_bounds='gmin_gmax'),
                               learning_rate=1e-3).load_state_dict(a.state_dict())
    bad = a.state_dict();bad['issued_counts'] = bad['issued_counts'] + 1
    with pytest.raises(ValueError, match='allowance'):
        CappedNominalWriteAdam(RecoveryWritePort(quiet, 0., target_bounds='gmin_gmax'),
                               learning_rate=3e-3).load_state_dict(bad)


def test_four_attempt_cap_issues_each_device_at_most_four_times():
    port = Port(2)
    writer = CappedNominalWriteAdam(port, learning_rate=3e-3, cap=4)
    for k in range(400):writer.step(torch.tensor([-1., (-1.) ** k]))
    issued = (torch.stack(port.commands) != 0).sum(0)
    assert issued[0] == 4 and issued[1] == 0
    assert writer.issued_counts[0] == 4 and writer.pending[0] == 0
    saved = writer.state_dict()
    resumed = CappedNominalWriteAdam(Port(2), learning_rate=3e-3, cap=4);resumed.load_state_dict(saved)
    with pytest.raises(ValueError, match='contract'):
        CappedNominalWriteAdam(Port(2), learning_rate=3e-3, cap=1).load_state_dict(saved)
    with pytest.raises(ValueError, match='one or four'):CappedNominalWriteAdam(Port(2), learning_rate=3e-3, cap=2)


def test_config_admits_one_attempt_only_for_the_capped_bounded_writer():
    for factor in (0., 1.):
        for cap in (1, 4):
            recovery = parse_config({'recovery': {**GOOD, 'write_noise_factor': factor,
                                                  'max_write_attempts': cap}}).recovery
            assert recovery.update_rule == 'bounded_capped_nominal_adam' and recovery.max_write_attempts == cap
    for change in ({'max_write_attempts': None}, {'max_write_attempts': 2},
                   {'target_bounds': 'none'}, {'conductance_floor': 'gmin'}, {'threshold_sigma': 3.}):
        with pytest.raises(ValueError):parse_config({'recovery': {**GOOD, **change}})
    for rule, bounds in (('direct_adam', 'gmin_gmax'), ('bounded_thresholded_adam', 'gmin_gmax'),
                         ('thresholded_adam', 'none')):
        with pytest.raises(ValueError):
            parse_config({'recovery': {'update_rule': rule, 'target_bounds': bounds, 'max_write_attempts': 1}})


def test_recovery_step_builds_the_capped_writer_on_held_state():
    from experiments.opt_cmo_drn.devices import program_rewritten
    from experiments.opt_cmo_drn.physical import CmoDrnBank
    model, teacher = tiny_model()
    _, _, state = program_rewritten(model, 271002, 'random', .03, capture_state=True)
    bank = CmoDrnBank(state, 'cpu')
    config = replace(Config(), recovery=replace(Recovery(), **GOOD, learning_rate=3e-4, write_noise_factor=0.))
    step = RecoveryStep(model, teacher, bank, config)
    assert all(type(w) is CappedNominalWriteAdam and w.threshold_q == NOMINAL_Q for w in step.writers)
    step.set_input(torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]))
    step.compute_gradient();step.step()
    assert sum(int(w.attempt_counts.sum()) for w in step.writers) == bank.costs()['write_attempts']


def test_declared_completed_training_is_evaluated_not_retrained(tmp_path, monkeypatch):
    root = tmp_path / 'output'
    final = tmp_path / 'trained/run/checkpoints/final.pt'
    final.parent.mkdir(parents=True);final.write_bytes(b'final state')
    run = final.parents[1]
    s.write(run / 'result.json', {'artifacts': [{'kind': 'recovery_state', 'path': 'checkpoints/final.pt',
                                                 'sha256': s.digest(final)}],
                                  'metrics': {'initial_device_state_sha256': 'device'}})
    m = {'output': str(root)}
    case = {'id': 'random03-dense-noisefree', 'rate': '03', 'config': 'frozen.json',
            'trained': {'run': str(run), 'path': str(final), 'sha256': s.digest(final)}}
    calls = []
    def native(manifest, config, destination, inputs, *, action='train', progress=None):
        calls.append(action)
        assert action == 'validate' and inputs == {'weights': str(final)}
        result = destination / 'run';progress(str(result));return result, 0
    monkeypatch.setenv('SLURM_JOB_ID', '100')
    monkeypatch.setattr(s, 'baseline', lambda *a: {'device_state_sha256': 'device'})
    monkeypatch.setattr(s, 'run_native', native)
    monkeypatch.setattr(s, 'inspect', lambda run, complete=True: None)
    state = s.case_worker(m, case)
    assert state['state'] == 'complete' and calls == ['validate'] and state['final_run'] == str(run)
    other = {**case, 'id': 'other', 'trained': {**case['trained'], 'sha256': 'wrong'}}
    with pytest.raises(ValueError, match='frozen identity'):s.case_worker(m, other)


def test_batch_wrapper_loads_a100_architecture_module_only_on_gpu_p5(tmp_path):
    source = tmp_path / 'source';(source / 'launch').mkdir(parents=True)
    manifest_path = source / 'launch/manifest.json';manifest_path.write_text('{}')
    wrapper = tmp_path / 'slurm_script'
    wrapper.write_text((Path(__file__).parents[1] / 'workflow/slurm_recovery.slurm').read_text())
    binary = tmp_path / 'bin';binary.mkdir()
    for name in ('nvidia-smi', 'srun'):
        path = binary / name;path.write_text('#!/bin/sh\nexit 0\n');path.chmod(0o755)
    prelude = 'module() { echo "module $*"; }; export -f module; bash "$@"'
    for partition, expected in (('gpu_p5', True), ('gpu_p13', False)):
        # Lmod hosts set BASH_ENV, which would replace the recording module stub.
        env = {**{k: v for k, v in os.environ.items() if k != 'BASH_ENV'},
               'PATH': str(binary) + os.pathsep + os.environ['PATH'], 'SLURM_JOB_PARTITION': partition}
        out = subprocess.run(['bash', '-c', prelude, 'wrapper', str(wrapper), str(manifest_path), 'canary'],
                             env=env, text=True, capture_output=True, check=True).stdout.splitlines()
        loads = [line for line in out if line.startswith('module load')]
        assert loads == (['module load arch/a100'] if expected else []) + ['module load pytorch-gpu/py3/2.5.0']
