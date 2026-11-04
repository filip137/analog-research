from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from methods.ibm_reram_program_verify import (
    ControllerSettings,
    HFO2_PRESET,
    IbmReramPlant,
    IbmReramPopulation,
    IbmReramRawActivePlant,
    OM_PRESET,
    PopulationStepEstimator,
    make_buffered_normal_draws,
    partition_device_identities,
    run_program_verify,
    sample_ibm_reram_population,
)


class _DeterministicPort:
    def __init__(self, values: list[float], *, up_step: float, down_step: float) -> None:
        self.values = torch.tensor(values, dtype=torch.float32)
        self.up_step = up_step
        self.down_step = down_step
        self.requests: list[tuple[torch.Tensor, torch.Tensor]] = []

    @property
    def size(self) -> int:
        return int(self.values.numel())

    @property
    def hidden_parameters(self):  # pragma: no cover - accessed only on regression
        raise AssertionError("controller read hidden parameters")

    def verify(self) -> torch.Tensor:
        return self.values.clone()

    def apply_identical_pulses(self, directions: torch.Tensor, counts: torch.Tensor) -> None:
        self.requests.append((directions.clone(), counts.clone()))
        up = directions > 0
        down = directions < 0
        self.values[up] += counts[up] * self.up_step
        self.values[down] -= counts[down] * self.down_step


def _synthetic_population() -> IbmReramPopulation:
    return IbmReramPopulation(
        preset=OM_PRESET,
        aihwkit_version="test",
        nominal_dw_min=0.1,
        dw_min_std=0.2,
        write_noise_std=0.5,
        mult_noise=False,
        construction_seeds=torch.tensor([1, 2], dtype=torch.int64),
        max_bound=torch.tensor([1.0, 1.0]),
        min_bound=torch.tensor([-1.0, -1.0]),
        dwmin_up=torch.tensor([0.1, 0.1]),
        dwmin_down=torch.tensor([0.1, 0.1]),
        reference=torch.tensor([0.0, 0.0]),
        corrupt=torch.tensor([False, False]),
        preset_parameters={},
    )


def test_one_pulse_controller_accepts_after_a_polarity_reversal() -> None:
    port = _DeterministicPort([0.0], up_step=0.4, down_step=0.15)
    result = run_program_verify(
        port,
        targets=torch.tensor([0.5]),
        tolerance=1e-6,
        maximum_pulses=10,
        settings=ControllerSettings(kind="one_pulse"),
    )

    assert result.accepted.tolist() == [True]
    assert result.total_pulses.tolist() == [4]
    assert result.set_count.tolist() == [2]
    assert result.reset_count.tolist() == [2]
    assert result.reversals.tolist() == [1]
    assert all(int(count[0]) == 1 for _, count in port.requests)


def test_adaptive_controller_batches_caps_and_returns_to_one_pulse() -> None:
    port = _DeterministicPort([0.0], up_step=0.1, down_step=0.1)
    estimator = PopulationStepEstimator(bins=5, fallback_step=0.1)
    result = run_program_verify(
        port,
        targets=torch.tensor([1.0]),
        tolerance=1e-6,
        maximum_pulses=12,
        settings=ControllerSettings(kind="adaptive", maximum_batch=4),
        estimator=estimator,
    )

    assert result.accepted.tolist() == [True]
    assert result.total_pulses.tolist() == [10]
    assert [int(count[0]) for _, count in port.requests] == [4, 4, 1, 1]
    assert result.verify_count.tolist() == [5]


def test_adaptive_controller_reverses_after_a_batched_overshoot() -> None:
    port = _DeterministicPort([0.0], up_step=0.3, down_step=0.1)
    estimator = PopulationStepEstimator(bins=5, fallback_step=0.1)
    result = run_program_verify(
        port,
        targets=torch.tensor([0.5]),
        tolerance=1e-6,
        maximum_pulses=12,
        settings=ControllerSettings(kind="adaptive", maximum_batch=4),
        estimator=estimator,
    )

    assert result.accepted.tolist() == [True]
    assert result.reversals.tolist() == [1]
    assert [int(direction[0]) for direction, _ in port.requests] == [1, -1, -1]
    assert [int(count[0]) for _, count in port.requests] == [3, 3, 1]


def test_controller_truncates_a_batch_at_the_total_pulse_budget() -> None:
    port = _DeterministicPort([0.0], up_step=0.01, down_step=0.01)
    estimator = PopulationStepEstimator(bins=5, fallback_step=0.01)
    result = run_program_verify(
        port,
        targets=torch.tensor([1.0]),
        tolerance=1e-6,
        maximum_pulses=5,
        settings=ControllerSettings(kind="adaptive", maximum_batch=4),
        estimator=estimator,
    )

    assert result.accepted.tolist() == [False]
    assert result.budget_exhausted.tolist() == [True]
    assert result.total_pulses.tolist() == [5]
    assert [int(count[0]) for _, count in port.requests] == [4, 1]


def test_controller_classifies_a_nonfinite_verify_without_pulsing() -> None:
    port = _DeterministicPort([float("nan")], up_step=0.1, down_step=0.1)
    result = run_program_verify(
        port,
        targets=torch.tensor([0.5]),
        tolerance=0.01,
        maximum_pulses=4,
        settings=ControllerSettings(kind="one_pulse"),
    )

    assert result.accepted.tolist() == [False]
    assert result.nonfinite.tolist() == [True]
    assert result.budget_exhausted.tolist() == [False]
    assert result.total_pulses.tolist() == [0]
    assert port.requests == []


def test_controller_uses_residual_sign_at_a_rounded_tolerance_boundary() -> None:
    target = torch.tensor([0.7576315999031067], dtype=torch.float32)
    apparent = torch.tensor([0.7576305866241455], dtype=torch.float32)
    tolerance = 1e-6
    assert bool(torch.abs(apparent - target) > tolerance)
    assert bool(apparent == target - tolerance)
    port = _DeterministicPort(
        apparent.tolist(),
        up_step=float((target - apparent).item()),
        down_step=0.1,
    )

    result = run_program_verify(
        port,
        targets=target,
        tolerance=tolerance,
        maximum_pulses=1,
        settings=ControllerSettings(kind="one_pulse"),
    )

    assert result.accepted.tolist() == [True]
    assert result.total_pulses.tolist() == [1]
    assert len(port.requests) == 1
    assert port.requests[0][0].tolist() == [1]


def test_partitioning_keeps_repeats_of_an_identity_together() -> None:
    first = partition_device_identities(20, seed=77)
    second = partition_device_identities(20, seed=77)
    assert first == second
    assert {label for label in first} == {"calibration", "fit", "validation"}
    expanded = [first[device_id] for device_id in range(20) for _ in range(8)]
    for device_id in range(20):
        assert len(set(expanded[device_id * 8 : (device_id + 1) * 8])) == 1


def test_plant_state_restores_exact_pulse_continuation() -> None:
    population = _synthetic_population()
    first = IbmReramPlant(population, seeds=[31, 32])
    first.pulse(torch.tensor([1, -1], dtype=torch.int8))
    checkpoint = first.state_dict()
    first.pulse(torch.tensor([1, 1], dtype=torch.int8))

    restored = IbmReramPlant(population, seeds=[999, 1000])
    restored.load_state_dict(checkpoint)
    restored.pulse(torch.tensor([1, 1], dtype=torch.int8))

    assert torch.equal(first.persistent, restored.persistent)
    assert torch.equal(first.apparent, restored.apparent)


def test_raw_active_plant_uses_one_a_state_and_ignores_reference() -> None:
    first_population = replace(
        _synthetic_population(),
        dw_min_std=0.0,
        write_noise_std=0.0,
        reference=torch.tensor([0.0, 0.0]),
    )
    second_population = replace(
        first_population,
        reference=torch.tensor([0.7, -0.6]),
    )
    kwargs = {
        "seeds": [301, 302],
        "coordinate_min": -3.0,
        "coordinate_scale": 6.0,
    }
    first = IbmReramRawActivePlant(first_population, **kwargs)
    second = IbmReramRawActivePlant(second_population, **kwargs)

    assert first.controller_port().verify().tolist() == pytest.approx([0.5, 0.5])
    first.pulse(torch.tensor([1, -1], dtype=torch.int8))
    second.pulse(torch.tensor([1, -1], dtype=torch.int8))

    assert first.persistent.tolist() == pytest.approx([0.1, -0.1])
    torch.testing.assert_close(first.persistent, second.persistent)
    torch.testing.assert_close(first.apparent, second.apparent)
    torch.testing.assert_close(
        first.controller_port().verify(), second.controller_port().verify()
    )


def test_buffered_normal_draws_keep_independent_seeded_rows() -> None:
    seeds = [31, 32]
    observed = make_buffered_normal_draws(
        seeds, maximum_random_draws=8, device="cpu"
    )
    expected = []
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        expected.append(torch.randn((8,), generator=generator))

    assert torch.equal(observed, torch.stack(expected))


def test_buffered_normal_draws_have_standard_normal_population_moments() -> None:
    observed = make_buffered_normal_draws(
        range(1000, 1256),
        maximum_random_draws=4096,
        device="cpu",
    ).double()

    assert abs(float(observed.mean())) < 0.005
    assert abs(float(observed.std(correction=1)) - 1.0) < 0.005


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_buffered_plant_replays_and_restores_exactly() -> None:
    population = _synthetic_population()
    first = IbmReramPlant(
        population,
        seeds=[131, 132],
        device="cuda",
        maximum_random_draws=16,
    )
    first.pulse(torch.tensor([1, -1], device="cuda", dtype=torch.int8))
    checkpoint = first.state_dict()
    first.pulse(torch.tensor([1, 1], device="cuda", dtype=torch.int8))

    restored = IbmReramPlant(
        population,
        seeds=[999, 1000],
        device="cuda",
        maximum_random_draws=16,
    )
    restored.load_state_dict(checkpoint)
    restored.pulse(torch.tensor([1, 1], device="cuda", dtype=torch.int8))

    assert torch.equal(first.persistent, restored.persistent)
    assert torch.equal(first.apparent, restored.apparent)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_and_cpu_noiseless_pulse_equations_agree() -> None:
    population = replace(
        _synthetic_population(), dw_min_std=0.0, write_noise_std=0.0
    )
    cpu = IbmReramPlant(population, seeds=[231, 232])
    cuda = IbmReramPlant(
        population,
        seeds=[231, 232],
        device="cuda",
        maximum_random_draws=16,
    )
    for direction in ([1, -1], [1, 1], [-1, 1]):
        cpu.pulse(torch.tensor(direction, dtype=torch.int8))
        cuda.pulse(torch.tensor(direction, dtype=torch.int8, device="cuda"))

    assert torch.equal(cpu.persistent, cuda.persistent.cpu())
    assert torch.equal(cpu.apparent, cuda.apparent.cpu())


def test_batched_pulses_are_exactly_sequential_nonlinear_transitions() -> None:
    population = _synthetic_population()
    batched = IbmReramPlant(population, seeds=[41, 42])
    sequential = IbmReramPlant(population, seeds=[41, 42])

    directions = torch.tensor([1, -1], dtype=torch.int8)
    counts = torch.tensor([3, 2], dtype=torch.int64)
    batched.controller_port().apply_identical_pulses(directions, counts)
    for pulse_index in range(3):
        sequential.pulse(
            torch.where(
                counts > pulse_index,
                directions,
                torch.zeros_like(directions),
            )
        )

    assert torch.equal(batched.persistent, sequential.persistent)
    assert torch.equal(batched.apparent, sequential.apparent)


def test_boundary_conditioning_reports_success_and_budget_failure() -> None:
    population = _synthetic_population()
    successful = IbmReramPlant(population, seeds=[51, 52]).condition_boundary(
        start_protocol="lower_to_target",
        quiet_steps=2,
        change_threshold=0.02,
        maximum_pulses=128,
    )
    failed = IbmReramPlant(population, seeds=[51, 52]).condition_boundary(
        start_protocol="lower_to_target",
        quiet_steps=2,
        change_threshold=0.02,
        maximum_pulses=1,
    )

    assert successful.success.tolist() == [True, True]
    assert failed.success.tolist() == [False, False]
    assert failed.pulse_count.tolist() == [1, 1]


def test_corrupt_stuck_device_retains_state_and_can_only_accept_its_stuck_value() -> None:
    base = _synthetic_population().select(torch.tensor([0]))
    corrupt = replace(
        base,
        max_bound=torch.zeros(1),
        min_bound=torch.zeros(1),
        dwmin_up=torch.zeros(1),
        dwmin_down=torch.zeros(1),
        corrupt=torch.ones(1, dtype=torch.bool),
    )
    plant = IbmReramPlant(corrupt, seeds=[61])
    conditioning = plant.condition_boundary(
        start_protocol="lower_to_target",
        quiet_steps=2,
        change_threshold=1e-6,
        maximum_pulses=4,
    )
    stuck_apparent = plant.controller_port().verify()
    accepted = run_program_verify(
        plant.controller_port(),
        targets=stuck_apparent,
        tolerance=1e-6,
        maximum_pulses=4,
        settings=ControllerSettings(kind="one_pulse"),
        eligible=conditioning.success,
    )

    assert conditioning.success.tolist() == [True]
    assert accepted.accepted.tolist() == [True]
    assert accepted.total_pulses.tolist() == [0]
    assert torch.equal(plant.persistent, torch.zeros(1))


def test_controller_port_exposes_only_apparent_verify_and_pulse_actions() -> None:
    plant = IbmReramPlant(_synthetic_population(), seeds=[71, 72])
    port = plant.controller_port()

    assert set(("size", "verify", "apply_identical_pulses")) <= set(dir(port))
    for hidden in ("persistent", "apparent", "population", "construction_seeds"):
        assert not hasattr(port, hidden)
    observed = port.verify()
    observed.fill_(123.0)
    assert not torch.equal(observed, port.verify())


def test_sampled_identity_is_independent_of_population_tail() -> None:
    pytest.importorskip("aihwkit")
    short = sample_ibm_reram_population(
        preset=OM_PRESET,
        num_devices=2,
        construction_seed=1234,
        enable_published_corruption=False,
    )
    long = sample_ibm_reram_population(
        preset=OM_PRESET,
        num_devices=4,
        construction_seed=1234,
        enable_published_corruption=False,
    )
    for name in (
        "construction_seeds", "max_bound", "min_bound", "dwmin_up",
        "dwmin_down", "reference", "corrupt",
    ):
        assert torch.equal(getattr(short, name), getattr(long, name)[:2])


@pytest.mark.parametrize("preset", [OM_PRESET, HFO2_PRESET])
def test_corruption_arm_preserves_every_noncorrupt_matched_identity(
    preset: str,
) -> None:
    pytest.importorskip("aihwkit")
    continuous = sample_ibm_reram_population(
        preset=preset,
        num_devices=64,
        construction_seed=72001,
        enable_published_corruption=False,
    )
    corrupt = sample_ibm_reram_population(
        preset=preset,
        num_devices=64,
        construction_seed=72001,
        enable_published_corruption=True,
    )
    noncorrupt = ~corrupt.corrupt

    assert bool(torch.any(corrupt.corrupt))
    assert torch.equal(continuous.construction_seeds, corrupt.construction_seeds)
    for field in ("max_bound", "min_bound", "dwmin_up", "dwmin_down", "reference"):
        assert torch.equal(
            getattr(continuous, field)[noncorrupt],
            getattr(corrupt, field)[noncorrupt],
        )


@pytest.mark.parametrize("preset", [OM_PRESET, HFO2_PRESET])
@pytest.mark.parametrize("direction", [1, -1])
def test_explicit_transition_has_native_aihwkit_unit_pulse_parity(
    direction: int,
    preset: str,
) -> None:
    pytest.importorskip("aihwkit")
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.utils import PulseType, UpdateParameters
    from aihwkit.simulator.presets.devices import (
        ReRamArrayHfO2PresetDevice,
        ReRamArrayOMPresetDevice,
    )
    from aihwkit.simulator.tiles import AnalogTile

    population = sample_ibm_reram_population(
        preset=preset,
        num_devices=1,
        construction_seed=987,
        enable_published_corruption=False,
    )
    noiseless = replace(population, dw_min_std=0.0, write_noise_std=0.0)
    preset_classes = {
        OM_PRESET: ReRamArrayOMPresetDevice,
        HFO2_PRESET: ReRamArrayHfO2PresetDevice,
    }
    device = preset_classes[preset]()
    device.construction_seed = int(population.construction_seeds[0])
    device.dw_min_std = 0.0
    device.write_noise_std = 0.0
    config = SingleRPUConfig(
        device=device,
        update=UpdateParameters(
            desired_bl=1,
            fixed_bl=True,
            pulse_type=PulseType.STOCHASTIC_COMPRESSED,
            res=-1,
            update_bl_management=False,
            update_management=False,
        ),
    )
    tile = AnalogTile(1, 1, config, bias=False)
    tile.set_weights(torch.zeros((1, 1)))
    tile.set_learning_rate(float(device.dw_min))
    plant = IbmReramPlant(noiseless, seeds=[11])
    plant.reset_logical_zero()

    for _ in range(3):
        plant.pulse(torch.tensor([direction], dtype=torch.int8))
        d_value = -1.0 if direction > 0 else 1.0
        tile.update(torch.ones((1, 1)), torch.tensor([[d_value]]))
        native = tile.get_weights()
        native = native[0] if isinstance(native, tuple) else native
        assert torch.equal(plant.persistent, native.reshape(-1))


def test_seeded_plant_and_controller_replay_exactly() -> None:
    population = _synthetic_population()

    def replay():
        plant = IbmReramPlant(population, seeds=[101, 102])
        plant.condition_boundary(
            start_protocol="lower_to_target",
            quiet_steps=2,
            change_threshold=1e-4,
            maximum_pulses=128,
        )
        plant.set_seeds([201, 202])
        events = []
        result = run_program_verify(
            plant.controller_port(),
            targets=torch.tensor([0.2, 0.8]),
            tolerance=population.nominal_dw_min / 4.0,
            maximum_pulses=32,
            settings=ControllerSettings(kind="one_pulse"),
            observer=lambda event: events.append(
                (
                    event.verify_index,
                    event.apparent.clone(),
                    event.direction.clone(),
                    event.pulse_count.clone(),
                )
            ),
        )
        return plant.persistent.clone(), result, events

    first = replay()
    second = replay()
    assert torch.equal(first[0], second[0])
    for field in (
        "accepted", "nonfinite", "budget_exhausted", "apparent_endpoint",
        "set_count", "reset_count", "total_pulses", "verify_count", "reversals",
    ):
        assert torch.equal(getattr(first[1], field), getattr(second[1], field))
    assert len(first[2]) == len(second[2])
    for left, right in zip(first[2], second[2]):
        assert left[0] == right[0]
        assert all(torch.equal(a, b) for a, b in zip(left[1:], right[1:]))
