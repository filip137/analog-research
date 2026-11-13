"""Fault-recovery paper LR screen on the CMO DRN: configs, schedule, targets, Child-Tuning_D, LoRA."""

import json
import math
from pathlib import Path

import pytest
import torch

from ebl.schema.base import ConfigError
from ebl.schema.opt_cmo_drn import parse_config
from methods import fault_tuning as ft
from methods.cmo_direct_adam import DirectWriteAdam, ThresholdedBoundedWriteAdam
from methods.config.fault_tuning import Arm
from methods.programming.cmo import program_rewritten
from networks.opt.physical import CmoDrnBank
from tests.test_opt_cmo_drn import cpu_threads, tiny_model

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'examples' / 'opt_cmo_drn' / 'fault_paper_tuning'


def test_generated_configs_are_current_and_declare_the_frozen_screen():
    import subprocess, sys
    script = ROOT / 'campaigns/pilots/20261113-fault-paper-lr-tuning/make_configs.py'
    subprocess.run([sys.executable, str(script), '--check'], check=True, cwd=ROOT)
    arms = {}
    for path in sorted(CONFIGS.glob('job-*.json')):
        c = parse_config(json.loads(path.read_text()))
        f = c.fault_tuning
        assert c.stage == 'fault_tuning' and c.deployment.assignment_seed == 271001
        assert (c.deployment.fault_kind, c.deployment.fault_rate) == ('random', 0.03)
        assert (f.steps, f.batch, f.schedule, f.order_seed) == (300, 8, 'cosine_to_zero', 20261104)
        assert f.recovery_sha256.startswith('c919329d') and f.development_sha256.startswith('7411cf50')
        assert f.samples_sha256.startswith('df73ce66') and f.samples_count == 8
        assert (f.child_fraction, f.lora_alpha, f.development_sequences) == (0.1, 'rank', 64)
        for a in f.arms:
            arms[a.label] = a
    rates = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    for method in ('noisy', 'noiseless', 'child_d'):
        assert sorted(a.learning_rate for a in arms.values() if a.method == method) == list(rates)
    for rank in (16, 64):
        assert sorted(a.learning_rate for a in arms.values() if a.method == 'lora' and a.rank == rank) == [1e-3, 3e-3, 1e-2]
    assert {a.method for a in arms.values()} == {'naive_hold', 'hold', 'noisy', 'noiseless', 'child_d', 'lora'}
    assert len(arms) == 2 + 15 + 6
    deploy = parse_config(json.loads((CONFIGS / 'deploy.json').read_text()))
    assert deploy.stage == 'deploy' and deploy.deployment == c.deployment


def test_invalid_arms_and_stage_mismatch_are_rejected():
    with pytest.raises(ValueError):
        Arm.parse({'label': 'x', 'method': 'lora', 'learning_rate': 1e-3})
    with pytest.raises(ValueError):
        Arm.parse({'label': 'x', 'method': 'hold', 'learning_rate': 1e-3})
    raw = json.loads((CONFIGS / 'job-b-noiseless.json').read_text())
    raw['stage'] = 'recover'
    with pytest.raises(ConfigError):
        parse_config(raw)
    raw = json.loads((CONFIGS / 'job-b-noiseless.json').read_text())
    raw['fault_tuning']['schedule'] = 'constant'
    with pytest.raises(ConfigError):
        parse_config(raw)


class Port:
    def __init__(self, n):
        self.n = n
        self.commands = []

    def read(self):
        return torch.zeros(self.n)

    def write(self, command):
        self.commands.append(command.clone())
        return command != 0

    def write_noise_std_q(self, pending):
        return torch.full_like(pending, 1e-9)


def test_cosine_to_zero_matches_torch_cosine_annealing_and_adam():
    steps, lr = 7, 3e-3
    parameter = torch.zeros(3, requires_grad=True)
    reference = torch.optim.Adam([parameter], lr=lr, betas=(.9, .999), eps=1e-8, foreach=False)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(reference, steps)
    port = Port(3)
    writer = DirectWriteAdam(port, learning_rate=lr, schedule=('cosine_to_zero', 0, steps))
    torch.manual_seed(0)
    for k in range(1, steps + 1):
        assert math.isclose(writer.rate(k), reference.param_groups[0]['lr'], rel_tol=1e-12)
        assert math.isclose(ft.cosine_to_zero(lr, steps)[k - 1], writer.rate(k), rel_tol=1e-12)
        gradient = torch.randn(3)
        previous = parameter.detach().clone()
        parameter.grad = gradient.clone()
        reference.step()
        schedule.step()
        writer.step(gradient)
        torch.testing.assert_close(port.commands[-1], parameter.detach() - previous, atol=1e-10, rtol=1e-5)
    with pytest.raises(ValueError):
        DirectWriteAdam(port, learning_rate=lr, schedule=('cosine_to_zero', 2, steps))


def test_scheduled_threshold_writer_keeps_constant_contract_and_gates_accumulated_commands():
    constant = ThresholdedBoundedWriteAdam(Port(2), learning_rate=1e-3)
    assert constant.contract() == ['cmo_drn.bounded_thresholded_adam.v1', 1e-3, 2.0, 'bounded_target_write_std',
                                   'command_before_projection', 'ge']
    port = Port(2)
    writer = ThresholdedBoundedWriteAdam(port, learning_rate=1e-3, schedule=('cosine_to_zero', 0, 10))
    assert writer.contract()[-1] == ['cosine_to_zero', 0, 10]
    writer.step(torch.tensor([1., 0.]))
    assert torch.equal(port.commands[-1] != 0, torch.tensor([True, False]))


def test_sample_loss_is_mean_negative_log_likelihood_of_the_fixed_samples():
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 5)
    samples = torch.randint(0, 5, (2, 3, 4))
    logp = logits.log_softmax(-1)
    expected = -sum(logp[b, t, samples[b, t, j]] for b in range(2) for t in range(3) for j in range(4)) / 24
    torch.testing.assert_close(ft.sample_loss(logits, samples), expected)
    with pytest.raises(ValueError):
        ft.sample_loss(logits, samples[:, :2])


def test_batch_order_is_shared_and_seeded():
    a = ft.batch_indices(5, 8, 1024, 20261104)
    b = ft.batch_indices(5, 8, 1024, 20261104)
    assert all(torch.equal(x, y) for x, y in zip(a, b)) and all(len(x) == 8 for x in a)
    generator = torch.Generator().manual_seed(20261104)
    assert torch.equal(a[0], torch.randint(0, 1024, (8,), generator=generator))


def test_child_masks_take_exactly_the_top_fraction_of_each_matrix():
    scores = [torch.arange(20.).view(4, 5), torch.tensor([[3., 1.], [2., 0.]])]
    masks = ft.child_masks(scores, 0.1)
    assert int(masks[0].sum()) == 2 and bool(masks[0].flatten()[-2:].all())
    assert int(masks[1].sum()) == 1 and bool(masks[1][0, 0])


def bank_and_model(rate=.03):
    model, teacher = tiny_model()
    _, _, state = program_rewritten(model, 271001, 'random' if rate else 'none', rate, capture_state=True)
    return model, teacher, state


def test_lora_adapters_start_as_identity_add_bax_and_never_write(cpu_threads):
    model, _, state = bank_and_model()
    bank = CmoDrnBank(state, 'cpu')
    tokens = torch.randint(0, 17, (2, 6))
    model.eval()
    with torch.no_grad():
        before = model(tokens, bank.view(model))
    arm = ft.LoraArm(model, bank, 1e-2, 3, 2, 5)
    with torch.no_grad():
        assert torch.equal(model(tokens, bank.view(model)), before)
    samples = torch.randint(0, 17, (2, 5, 8))
    held = bank.held_hashes()
    rates = [arm.step(tokens, samples, 1, 'cpu')['rate'] for _ in range(3)]
    assert rates == pytest.approx(ft.cosine_to_zero(1e-2, 3))
    assert any(bool(b.abs().sum() > 0) for b in arm.adapters.b)
    with torch.no_grad():
        after = model(tokens, bank.view(model))
    assert not torch.equal(after, before)
    assert bank.held_hashes() == held and bank.costs()['write_attempts'] == 0
    assert all(not u.requires_grad for u in model.parameter_groups()[0])
    arm.close()
    with torch.no_grad():
        assert torch.equal(model(tokens, bank.view(model)), before)
    a = arm.adapters
    assert a.a[0].shape == (2, 4) and a.b[0].shape == (4, 2) and a.scale == 1.0


def test_child_d_writes_only_masked_devices_and_noisy_and_noiseless_share_the_start(cpu_threads):
    model, _, state = bank_and_model()
    tokens = torch.randint(0, 17, (4, 6))
    samples = torch.randint(0, 17, (4, 5, 8))
    bank = CmoDrnBank(state, 'cpu')
    held = bank.held_hashes()
    scores = ft.fisher_scores(model, bank, tokens, samples, 2, 1, 'cpu')
    assert bank.held_hashes() == held and bank.costs()['write_attempts'] == 0
    assert all(s.shape == a.held.shape and bool((s >= 0).all()) for s, a in zip(scores, bank.arrays))
    masks = ft.child_masks(scores, 0.1)
    arm = ft.PhysicalArm(model, bank, 'child_d', 1e-1, 4, masks)
    for _ in range(4):
        arm.step(tokens, samples, 2, 'cpu')
    report = arm.write_report()
    for writer, mask in zip(arm.writers, masks):
        assert int(writer.attempt_counts[~mask].sum()) == 0
    assert report['touched_cells'] > 0 and report['touched_cells'] <= sum(int(m.sum()) for m in masks)
    assert report['max_writes_per_cell'] <= 4
    arm.close()
    for method in ('noisy', 'noiseless'):
        fresh = CmoDrnBank(state, 'cpu')
        assert fresh.held_hashes() == held
        arm = ft.PhysicalArm(model, fresh, method, 1e-1, 4)
        arm.step(tokens, samples, 2, 'cpu')
        report = arm.write_report()
        assert report['costs']['write_attempts'] == sum(int(w.attempt_counts.sum()) for w in arm.writers)
        if method == 'noiseless':
            assert report['touched_cells'] > 0
        arm.close()
    with pytest.raises(ValueError):
        ft.PhysicalArm(model, CmoDrnBank(state, 'cpu'), 'noisy', 1e-3, 4, masks)


def test_microbatching_gives_the_batch_mean_gradient(cpu_threads):
    model, _, state = bank_and_model(0.)
    bank = CmoDrnBank(state, 'cpu')
    tokens = torch.randint(0, 17, (4, 6))
    samples = torch.randint(0, 17, (4, 5, 8))
    grads = []
    for micro in (4, 1):
        model.requires_grad_(False)
        u = model.blocks[0].u1
        u.requires_grad_(True)
        model.train()
        model.zero_grad(set_to_none=True)
        ft.backward_batch(model, bank.view(model), tokens, samples, micro, 'cpu')
        grads.append(u.grad.clone())
        u.requires_grad_(False)
    torch.testing.assert_close(grads[0], grads[1], atol=1e-6, rtol=1e-4)
