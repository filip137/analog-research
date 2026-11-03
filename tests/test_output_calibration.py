"""Closed-form MLP-output calibration: native-order replay and known distortions."""

import torch

from training import output_calibration as oc


import pytest


@pytest.mark.parametrize("scope", ["model", "analog"])
def test_decomposed_forward_equals_native_student(scope):
    from test_workflow_opt import tiny_teacher
    from workflow.networks import opt_mlp
    torch.manual_seed(0)
    network = opt_mlp.OptMlpSuffix(tiny_teacher(), 2, tile_size=8, scope=scope).eval()
    tokens = torch.randint(0, 64, (3, 9))
    q = network.q.detach() * .7
    with torch.no_grad():
        native = network(tokens, q)
        layers = len(network.mlp_norms)
        decomposed = network.head(oc.forward(network.analog_input(tokens), network.mlp_branch, layers, q))
        identity = [oc.OutputCorrection(torch.ones(()), torch.zeros(())) for _ in range(layers)]
        corrected = network.head(oc.forward(network.analog_input(tokens), network.mlp_branch, layers, q, identity))
    assert torch.equal(native, decomposed) and torch.equal(native, corrected)


def toy_branch(scales, offsets):
    def branch(k, hidden, view):
        z = hidden * view[k] * scales[k] + offsets[k]
        return hidden, z, torch.full_like(z[0, 0], .25)
    return branch


def test_moments_restore_reference_statistics_layer_by_layer():
    torch.manual_seed(1)
    prefix = [torch.randn(2, 16, 8) for _ in range(3)]
    reference_view, aged_view = (1., 1.), (.6, 1.4)
    branch = toy_branch((1., .5), (0., .1))
    reference = oc.propagate(prefix, branch, 2, reference_view)
    fitted = oc.propagate(prefix, branch, 2, aged_view, method="mlp_moments", reference=reference, bounds=(.1, 10.))
    restored = oc.propagate(prefix, branch, 2, aged_view, corrections=fitted)
    for a, b in zip(reference, restored):
        # Statistics of z are measured before each layer's own correction.
        assert a.count == b.count
    corrected_first = oc.statistics(fitted[0](branch(0, h, aged_view)[1]) for h in prefix)
    assert torch.allclose(corrected_first.mean, reference[0].mean, atol=1e-5)
    assert torch.allclose(corrected_first.variance, reference[0].variance, rtol=1e-4)
    shift = oc.propagate(prefix, branch, 2, aged_view, method="mlp_shift", reference=reference, bounds=(.1, 10.))
    assert torch.equal(shift[0].gain, torch.ones(8))


def test_uniform_attenuation_recovered_by_global_gain():
    torch.manual_seed(2)
    prefix = [torch.randn(4, 8) for _ in range(2)]
    branch = toy_branch((1.,), (0.,))
    reference = oc.propagate(prefix, branch, 1, (1.,))
    fitted = oc.propagate(prefix, branch, 1, (.5,), method="mlp_gain", reference=reference, bounds=(.1, 10.))
    assert torch.allclose(fitted[0].gain, torch.full((8,), 2.)) and not fitted[0].shift.any()
    state = fitted[0].state_dict()
    replay = oc.OutputCorrection.from_state_dict(state, "cpu")
    assert torch.equal(replay.gain, fitted[0].gain) and oc.parameters("mlp_moments", 768, 6) == 9216


def test_bounds_and_constant_channels_are_finite():
    reference = oc.Statistics(4, torch.zeros(3, dtype=torch.float64), torch.tensor([1., 0., 1.], dtype=torch.float64), 1.)
    current = oc.Statistics(4, torch.ones(3, dtype=torch.float64), torch.tensor([1e-6, 1., 0.], dtype=torch.float64), 1e-9)
    moments = oc.solve("mlp_moments", reference, current, (.1, 10.))
    assert torch.equal(moments.gain, torch.tensor([10., 1., 1.]))
    assert oc.solve("mlp_gain", reference, current, (.1, 10.)).gain.max() == 10.
