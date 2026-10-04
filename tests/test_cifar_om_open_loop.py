"""Feedback isolation, unlimited pulse counts, and exact open-loop replay."""
import torch
import pytest
from experiments.cifar_crossbar.open_loop_updates import UncappedOpenLoopAdam, nominal_reset_counts, apply_pulse_counts


class CountingPort:
    nominal_step = .01

    def __init__(self, size=3):
        self.size = size
        self.state = torch.zeros(size)
        self.pulses = torch.zeros(size, dtype=torch.int64)
        self.finishes = 0

    def read(self):
        raise AssertionError("Open-loop controller used write feedback.")

    def pulse(self, direction):
        self.state.add_(direction * self.nominal_step).clamp_(-1, 1)
        self.pulses += direction.abs().to(torch.int64)

    def finish_step(self):
        self.finishes += 1


def test_open_loop_keeps_one_pulse_per_update_without_total_cap_or_verify_reads():
    port = CountingPort()
    writer = UncappedOpenLoopAdam(port, port.state, learning_rate=6.5, seed=1)
    for _ in range(700):
        before = writer.count.clone()
        writer.step(torch.tensor([1., -1., 0.]))
        assert bool(((writer.count - before) <= 1).all())
    torch.testing.assert_close(writer.count, torch.tensor([700, 700, 0]))
    assert torch.equal(writer.count, port.pulses)
    assert writer.verify_reads == 0 and port.finishes == 700
    assert writer.probability_clipped == 1400
    # Device saturation does not stop delivery or silently cap the command.
    torch.testing.assert_close(port.state, torch.tensor([-1., 1., 0.]))


def test_fractional_one_pulse_probability_is_retained():
    port = CountingPort(100000)
    writer = UncappedOpenLoopAdam(port, port.state, learning_rate=.003, seed=2026)
    writer.step(torch.ones(100000))
    assert set(writer.count.unique().tolist()) == {0, 1}
    assert abs(float(writer.count.float().mean()) - .3) < .006
    assert writer.probability_clipped == 0


def test_open_loop_adam_exact_state_and_rng_replay():
    port = CountingPort()
    writer = UncappedOpenLoopAdam(port, port.state, learning_rate=.014, seed=10)
    writer.step(torch.tensor([1., -2., .1]))
    saved, physical, pulses = writer.state_dict(), port.state.clone(), port.pulses.clone()
    grad = torch.tensor([-.7, .2, .9])
    writer.step(grad)
    other = CountingPort()
    other.state.copy_(physical); other.pulses.copy_(pulses)
    restored = UncappedOpenLoopAdam(other, physical, learning_rate=.014, seed=999)
    restored.load_state_dict(saved); restored.step(grad)
    assert torch.equal(port.state, other.state) and torch.equal(port.pulses, other.pulses)
    for key in ("first", "second", "count", "rng"):
        assert torch.equal(writer.state_dict()[key], restored.state_dict()[key])


def test_zero_learning_rate_never_reads_or_writes():
    port = CountingPort()
    writer = UncappedOpenLoopAdam(port, port.state, learning_rate=0., seed=1)
    writer.step(torch.ones(3))
    assert not port.pulses.any() and port.finishes == 0
    with pytest.raises(ValueError):
        writer.step(torch.full((3,), float("nan")))


def test_legacy_open_loop_pulses_are_identical_before_the_old_cap():
    from experiments.cifar_crossbar.devices import PulseWriter
    class LegacyPort(CountingPort):
        def read(self): return self.state.clone()
    port, legacy_port = CountingPort(20), LegacyPort(20)
    writer = UncappedOpenLoopAdam(port, port.state, learning_rate=.004, seed=19)
    legacy = PulseWriter(legacy_port, method='open_loop', learning_rate=.004, tolerance=.01,
                         cap=640, maximum_pulses=1, seed=19)
    rng = torch.Generator().manual_seed(5)
    for _ in range(100):
        gradient = torch.randn(20, generator=rng)
        writer.step(gradient); legacy.step(gradient)
        assert torch.equal(writer.count, legacy.count)
        assert torch.equal(port.state, legacy_port.state)


def test_nominal_reset_inverse_uses_no_feedback_or_pulse_budget():
    class SoftBoundsPort(CountingPort):
        def pulse(self, direction):
            self.state.add_(direction * self.nominal_step * (1 - self.state))
            self.pulses += direction.abs().to(torch.int64)
    target = torch.tensor([-1., -.7, 0., .7, 1.])
    port = SoftBoundsPort(5); port.state.fill_(-1.)
    count = nominal_reset_counts(target, port.nominal_step)
    assert int(count[-1]) > 640 and count[0] == 0
    apply_pulse_counts(port, torch.ones(5, dtype=torch.int8), count)
    torch.testing.assert_close(port.state, target, atol=.005, rtol=0)
    assert torch.equal(port.pulses, count)


def test_uncapped_contract_and_full_sweep_coverage():
    from experiments.cifar_crossbar.om_open_loop_config import default_config, parse_config
    from experiments.cifar_crossbar.sweep_config import SOURCES, cases
    for dataset in ("cifar10", "cifar100"):
        for initialization, rule in (("saved_p0", "inherited_v2_p0"), ("reset_open_loop", "nominal_soft_bounds_inverse")):
            spec = parse_config(default_config(dataset=dataset, calibration_lr=.0003 if dataset == "cifar10" else .001,
                                               initialization=initialization, programming_rule=rule))
            assert sum(len(cases(spec, source)) for source in SOURCES) * len(spec.methods) == 216
    for change in ({"recovery_pulses": 640}, {"update_pulses": 2}, {"programming_pulse_cap": 128},
                   {"initialization": "reset_open_loop"}, {"learning_rate": .001}):
        with pytest.raises(ValueError):
            default_config(**change)


def test_compressed_checkpoints_preserve_exact_tensors_and_rng(tmp_path):
    from experiments.cifar_crossbar.om_open_loop_runtime import save_compressed
    from experiments.cifar_crossbar.runtime import load
    port = CountingPort()
    writer = UncappedOpenLoopAdam(port, port.state, learning_rate=.014, seed=10)
    writer.step(torch.tensor([1., -2., .1]))
    state = writer.state_dict()
    path = tmp_path / "checkpoint.pt"
    save_compressed(path, state)
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    restored = load(path)
    for key in ("first", "second", "count", "rng"):
        assert torch.equal(state[key], restored[key])
    assert restored["contract"] == state["contract"]


def test_inherited_deployment_checks_numeric_metrics_and_prediction_hash():
    from experiments.cifar_crossbar.om_open_loop_runtime import verify_inherited_metrics
    expected = {'accuracy_percent': 80., 'teacher_kl': .3, 'examples': 10000, 'prediction_sha256': 'fixed'}
    verify_inherited_metrics(expected, expected)
    verify_inherited_metrics({**expected, 'teacher_kl': .300001}, expected)
    for change in ({'prediction_sha256': 'changed'}, {'accuracy_percent': 80.01}, {'teacher_kl': .4}):
        with pytest.raises(ValueError): verify_inherited_metrics({**expected, **change}, expected)
