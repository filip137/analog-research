"""Calibration recovers known distortions without changing the device state."""

import pytest
import torch

from training.drift_compensation import METHODS, Compensation, fit_readouts, apply_moments


@pytest.mark.parametrize("method", METHODS)
def test_uniform_attenuation_recovered_with_held_weights(method):
    torch.manual_seed(67)
    reference = torch.randn(13, 7)
    observed = reference * 0.6
    saved = observed.clone()
    x = torch.randn(19, 7)
    correction = fit_readouts(reference, observed, method, tile_size=4)
    torch.testing.assert_close(correction.mvm(x, observed), x @ reference.T, rtol=2e-6, atol=2e-6)
    assert torch.equal(observed, saved)


@pytest.mark.parametrize("method", ["row", "column", "row_column", "tile_global"])
def test_structured_attenuation_is_recovered(method):
    torch.manual_seed(5)
    reference = torch.randn(8, 12)
    if method == "row":
        gain = torch.linspace(.6, .9, 12)[None, :]
    elif method == "column":
        gain = torch.linspace(.6, .9, 8)[:, None]
    elif method == "row_column":
        gain = torch.linspace(.7, 1, 8)[:, None] * torch.linspace(.6, .9, 12)[None, :]
    else:
        gain = torch.tensor([[.4, .6, .8], [.5, .7, .9]]).repeat_interleave(4, 0).repeat_interleave(4, 1)
    observed = reference * gain
    x = torch.randn(17, 12)
    correction = fit_readouts(reference, observed, method, tile_size=4, iterations=30)
    torch.testing.assert_close(correction.mvm(x, observed), x @ reference.T, rtol=2e-6, atol=2e-6)


def test_mean_variance_correction_and_serialized_replay():
    ref = torch.tensor([[1., 2.], [3., 4.], [5., 8.], [-1., 2.]])
    aged = ref * torch.tensor([.5, 2.]) + torch.tensor([3., -2.])
    compensation = Compensation([fit_readouts(torch.eye(2), torch.eye(2), "global", tile_size=2)], observe=True)
    compensation.after_bias(aged[:2], 0)
    compensation.after_bias(aged[2:], 0)
    moments = compensation.moments()
    target = [(ref.double().mean(0), ref.double().var(0, unbiased=False))]
    apply_moments(compensation, target, moments, shift_only=False)
    restored = Compensation.from_state_dict(compensation.state_dict(), 'cpu')
    torch.testing.assert_close(restored.after_bias(aged, 0), ref)


def test_shift_only_and_constant_channels_are_finite():
    correction = Compensation([fit_readouts(torch.zeros(3, 2), torch.zeros(3, 2), "global", tile_size=2)])
    target = [(torch.tensor([3., 2., 1.]), torch.zeros(3))]
    current = [(torch.tensor([0., 4., 2.]), torch.zeros(3))]
    apply_moments(correction, target, current, shift_only=True)
    torch.testing.assert_close(correction.after_bias(current[0][0], 0), target[0][0])
    apply_moments(correction, target, current, shift_only=False)
    assert torch.isfinite(correction.matrices[0].shift).all()


@pytest.mark.parametrize("method", ["global", "column"])
def test_matches_aihwkit_readout_compensation(method):
    pytest.importorskip("aihwkit")
    from aihwkit.inference.compensation.drift import GlobalDriftCompensation, PerColumnDriftCompensation
    torch.manual_seed(7)
    baseline = torch.randn(7, 5)
    aged = baseline * torch.rand(7, 5).add(.5)
    ibm = GlobalDriftCompensation() if method == "global" else PerColumnDriftCompensation()
    expected = ibm.readout(baseline.T) / ibm.readout(aged.T)
    actual = fit_readouts(baseline, aged, method, tile_size=4).output_gain
    torch.testing.assert_close(actual, expected.expand_as(actual))
