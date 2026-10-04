"""Numerical and selection contracts specific to the CIFAR-10 extension."""
import json
from pathlib import Path

import pytest
import torch

from experiments.cifar10_crossbar.config import parse_config
from experiments.cifar10_crossbar.hardware import validation_rank
from training.ibm_om_standard_crossbar import build_crossbar_layout, map_logical_weights, standard_crossbar_logits
from training.closed_loop_crossbar_adam import ClosedLoopAdam

ROOT = Path(__file__).resolve().parents[1]


DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device_name', DEVICES)
def test_cifar10_multiple_input_tiles_preserve_logits_and_gradients(device_name):
    torch.manual_seed(94)
    device = torch.device(device_name)
    layout = build_crossbar_layout((3072, 256, 10), maximum_input_size=512)
    assert len(layout) == 7
    assert sum(t.cells for t in layout) == 788992
    weights = (torch.randn(3072, 256, device=device) * .01, torch.randn(256, 10, device=device) * .01)
    q, scales = map_logical_weights(weights, layout, weight_scaling_omega=(1., 1.))
    q = q.to(device).requires_grad_(True)
    x = torch.randn(13, 3072, device=device, requires_grad=True)
    expected = torch.relu(x @ weights[0]) @ weights[1]
    observed = standard_crossbar_logits(x, q, layout, digital_scales=scales)
    torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-5)
    actual_grad, = torch.autograd.grad(observed.square().sum(), x, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected.square().sum(), x)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
    qgrad, = torch.autograd.grad(observed.square().sum(), q)
    assert bool(torch.isfinite(qgrad).all())
    assert bool((qgrad != 0).any())


def test_epoch_zero_survives_if_recovery_worsens_validation_kl():
    p0 = dict(kl_teacher_student=.02, accuracy=.5)
    later = dict(kl_teacher_student=.021, accuracy=.51)
    assert validation_rank(p0, 0) < validation_rank(later, 1)
    assert validation_rank(p0, 0) < validation_rank(p0, 30)


@pytest.mark.parametrize('device_name', DEVICES)
def test_closed_loop_accumulates_small_commands_and_respects_cap(device_name):
    apparent = torch.zeros(3, device=device_name)
    calls = []

    def pulse(direction):
        calls.append(direction.clone())
        # Middle cell is physically stuck. The writer is not told this.
        apparent.add_(direction * torch.tensor([.05, 0., .05], device=device_name))

    writer = ClosedLoopAdam(lambda: apparent.clone(), pulse, learning_rate=.01,
                            tolerance=.025, maximum_pulses=4, total_pulse_cap=3)
    gradient = torch.tensor([1., 1., 0.], device=device_name)
    for _ in range(10):
        writer.step(gradient)
    assert calls, 'Sub-tolerance commands must accumulate until a write occurs.'
    assert int(writer.pulse_count.max()) <= 3
    assert float(apparent[1]) == 0
    assert float(apparent[2]) == 0
    assert int(writer.pulse_count[2]) == 0


@pytest.mark.parametrize('field,value', [('dims', [784, 256, 10]), ('epochs', True), ('learning_rate', float('nan')), ('evaluation_limit', 0)])
def test_rejects_invalid_scientific_config(field, value):
    config = json.loads((ROOT / 'examples/cifar10_crossbar/pretrain.json').read_text())
    config[field] = value
    with pytest.raises(ValueError):
        parse_config(config)
