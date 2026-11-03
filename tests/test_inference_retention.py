"""Parity with the installed AIHWKit release and time-sweep invariants."""

import pytest
import torch

from training import inference_retention as retention


def stream(seed=723):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("seconds", [0.0, 1.0, 86400.0, 172800.0])
@pytest.mark.parametrize("read_noise_scale", [0.0, 1.0])
def test_pcm_matches_aihwkit_110(seconds, read_noise_scale):
    aihwkit = pytest.importorskip("aihwkit")
    assert aihwkit.__version__ == retention.AIHWKIT_VERSION
    from aihwkit.inference.noise.pcm import PCMLikeNoiseModel

    reference = PCMLikeNoiseModel(read_noise_scale=read_noise_scale)
    targets = torch.linspace(0, 25, 123)
    programmed = (targets + torch.linspace(-0.1, 0.7, 123)).clamp_min(0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(723)
        nu = reference.generate_drift_coefficients(targets)
        expected = reference.apply_drift_noise_to_conductance(programmed, nu, seconds)
    generator = stream()
    coefficients = retention.pcm_coefficients(targets, generator)
    assert torch.equal(coefficients, nu)
    actual = retention.pcm_snapshot(programmed, coefficients, generator, seconds,
                                    read_noise_scale=read_noise_scale)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seconds", [0.0, 1.0, 86400.0, 172800.0])
@pytest.mark.parametrize("read_noise_scale", [0.0, 1.0])
def test_cmo_matches_aihwkit_110(seconds, read_noise_scale):
    aihwkit = pytest.importorskip("aihwkit")
    assert aihwkit.__version__ == retention.AIHWKIT_VERSION
    from aihwkit.inference.noise.reram import ReRamCMONoiseModel

    reference = ReRamCMONoiseModel(read_noise_scale=read_noise_scale, acceptance_range=2)
    programmed = torch.linspace(8, 90, 123)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(723)
        expected = reference.apply_drift_noise_to_conductance(programmed, programmed, seconds)
    actual = retention.cmo_snapshot(programmed, stream(), seconds, read_noise_scale=read_noise_scale)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seconds", [1.0, 86400.0, 172800.0])
@pytest.mark.parametrize("g_max", [25.0, 40.0])
def test_wan_matches_aihwkit_110(seconds, g_max):
    aihwkit = pytest.importorskip("aihwkit")
    assert aihwkit.__version__ == retention.AIHWKIT_VERSION
    from aihwkit.inference.noise.reram import ReRamWan2022NoiseModel

    reference = ReRamWan2022NoiseModel(g_max=g_max)
    targets = torch.linspace(0, g_max, 123)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(723)
        # Upstream must ignore an unrelated programmed state.
        expected = reference.apply_drift_noise_to_conductance(torch.full_like(targets, 1000), targets, seconds)
    actual = retention.wan_snapshot(targets, stream(), seconds, g_max=g_max)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_pcm_age_sweep_keeps_baseline_and_device_coefficients():
    programmed = torch.tensor([1.0, 5.0, 20.0])
    original = programmed.clone()
    nu = retention.pcm_coefficients(programmed, stream())
    first = retention.pcm_snapshot(programmed, nu, stream(), 86400, read_noise_scale=0)
    second = retention.pcm_snapshot(programmed, nu, stream(), 172800, read_noise_scale=0)
    assert torch.equal(programmed, original)
    assert bool((second < first).all())
    torch.testing.assert_close(second / first, torch.pow(torch.tensor(172820 / 86420), -nu))
    # Visiting other ages must not compound aging or change a repeated snapshot.
    assert torch.equal(retention.pcm_snapshot(programmed, nu, stream(), 86400, read_noise_scale=0), first)


def test_cmo_zero_age_preserves_unclamped_programming_endpoints():
    programmed = torch.tensor([8.0, 50.0, 90.0])
    zero = retention.cmo_snapshot(programmed, stream(), 0)
    assert torch.equal(zero, programmed)
    zero[0] = 42
    assert programmed[0] == 8
    aged = retention.cmo_snapshot(programmed, stream(), 86400, read_noise_scale=0)
    assert bool(((aged >= 9) & (aged <= 88.199997)).all())


@pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf"), True])
def test_invalid_times_are_rejected(seconds):
    g = torch.tensor([20.0])
    with pytest.raises(ValueError):
        retention.pcm_snapshot(g, torch.ones_like(g), stream(), seconds)
    with pytest.raises(ValueError):
        retention.cmo_snapshot(g, stream(), seconds)
    with pytest.raises(ValueError):
        retention.wan_snapshot(g, stream(), seconds)


@pytest.mark.parametrize("seconds", [0, 10, 3600, 86401])
def test_wan_never_interpolates_or_invents_a_zero_time_fit(seconds):
    with pytest.raises(ValueError, match="no interpolation"):
        retention.wan_snapshot(torch.tensor([20.0]), stream(), seconds)


def test_cmo_rejects_an_undefined_read_noise_domain():
    with pytest.raises(ValueError, match="positive healthy-device"):
        retention.cmo_snapshot(torch.tensor([-10.0]), stream(), 86400)
    with pytest.raises(ValueError, match="t>=1"):
        retention.cmo_snapshot(torch.tensor([20.0]), stream(), 0.5)


@pytest.mark.parametrize("model", retention.RetentionSeries.models)
def test_series_replays_ages_in_any_order_and_preserves_stuck_devices(model):
    target = torch.tensor([[10.0, 12.0, 18.0], [15.0, 15.0, 15.0]])
    mask = torch.tensor([[True, False, False], [False, True, False]])
    stuck = torch.tensor([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    programmed = torch.where(mask, stuck, target + 0.2)
    series = retention.RetentionSeries(model, target, programmed, seed=981,
                                       read_noise_scale=0 if model == "reram_wan2022" else 1,
                                       fault_mask=mask, stuck=stuck)
    one_day = series.snapshot(86400)
    two_days = series.snapshot(172800)
    replay = retention.RetentionSeries.from_state_dict(series.state_dict())
    assert torch.equal(replay.snapshot(172800), two_days)
    assert torch.equal(replay.snapshot(86400), one_day)
    assert torch.equal(series.snapshot(86400), one_day)
    assert torch.equal(series.baseline(), programmed)
    for result in (one_day, two_days):
        assert torch.equal(result[mask], stuck[mask])
        assert bool(torch.isfinite(result).all())
    # Neither input mutation nor mutation of a returned baseline changes state.
    target.zero_()
    series.baseline().zero_()
    assert torch.equal(series.snapshot(86400), one_day)


def test_series_rejects_a_repaired_fault_and_cross_device_rng_replay():
    g = torch.tensor([20.0])
    with pytest.raises(ValueError, match="preserve finite stuck"):
        retention.RetentionSeries("reram_cmo", g, g, seed=1, read_noise_scale=0,
                                  fault_mask=torch.tensor([True]), stuck=torch.tensor([0.0]))
    series = retention.RetentionSeries("pcm", g, g, seed=1, read_noise_scale=1)
    state = series.state_dict()
    state["execution_device_type"] = "cuda"
    with pytest.raises(ValueError, match="original execution-device"):
        retention.RetentionSeries.from_state_dict(state)


def test_wan_noise_scale_reuses_the_age_stream():
    target = torch.linspace(0, 40, 257).repeat(2, 1)
    trace = retention.RetentionSeries("reram_wan2022", target, target.clone(), seed=5, read_noise_scale=0)
    nominal = trace.snapshot(172800.0)
    assert torch.equal(trace.snapshot(172800.0, 1.0), nominal)
    unclamped = nominal > 0
    for scale in (1.5, 4.0):
        scaled = trace.snapshot(172800.0, scale)
        torch.testing.assert_close(scaled[unclamped & (scaled > 0)] - target[unclamped & (scaled > 0)],
                                   scale * (nominal - target)[unclamped & (scaled > 0)], rtol=1e-5, atol=1e-5)
        assert bool((scaled >= 0).all())
    assert torch.equal(trace.baseline(), target)
    for model, scale in (("pcm", 2.0), ("reram_cmo", 2.0), ("reram_wan2022", 0.0), ("reram_wan2022", float("nan"))):
        other = retention.RetentionSeries(model, torch.full((2, 3), 20.0), torch.full((2, 3), 20.0), seed=5,
                                          read_noise_scale=0)
        with pytest.raises(ValueError):
            other.snapshot(86400.0, scale)
