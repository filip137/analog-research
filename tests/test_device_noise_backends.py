"""Focused parity and lifecycle tests for measured device-noise backends."""

from __future__ import annotations

from copy import deepcopy
import math

import pytest
import torch

from networks.resistive.builders import ParameterBinding, ParameterCatalog
from hardware.config import (
    CmoHfOxProgrammingConfig,
    IbmAfm2025PcmProgrammingConfig,
    Wan2022PhysicalProgrammingConfig,
)
from networks.variable.parameter import DenseWeight
from methods.add_normal import AddNormalConfig, AddNormalParameterModifier
from methods.device_programming import realize_programmed_tensor
from methods.program_verify import ProgramVerifyOptimizer


def _dense(
    values: torch.Tensor,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> DenseWeight:
    weight = DenseWeight(
        (values.shape[0],),
        (values.shape[1],),
        gain=1.0,
        device="cpu",
        clamp=True,
        clamp_min=minimum,
        clamp_max=maximum,
    )
    with torch.no_grad():
        weight.state.copy_(values)
    return weight


def _pcm_config(*, noise_scale: float = 1.0):
    return IbmAfm2025PcmProgrammingConfig(
        type="ibm_afm2025_pcm",
        programming_seed=17,
        noise_scale=noise_scale,
        fit_max=180.0,
        zero_threshold=0.6,
        max_input_size=3,
    )


def _cmo_config(
    *,
    noise_scale: float = 1.0,
    t_seconds: float = 1.0,
    mapping: str = "literal_conductance",
    reference: float = 1.0,
):
    return CmoHfOxProgrammingConfig(
        type="aihwkit_reram_cmo",
        programming_seed=23,
        g_min_us=9.0,
        g_max_us=88.199997,
        drn_conductance_at_g_max=reference,
        noise_scale=noise_scale,
        acceptance_range_percent=0.2,
        t_inference_seconds=t_seconds,
        read_noise_scale=noise_scale,
        drift_scale=noise_scale,
        mapping=mapping,
    )


def _wan_physical_config(
    *,
    noise_scale: float = 1.0,
    t_seconds: float = 86400.0,
    mapping: str = "affine_floor",
    reference: float = 1.0,
):
    return Wan2022PhysicalProgrammingConfig(
        type="aihwkit_reram_wan2022_physical",
        programming_seed=23,
        g_min_us=1.0,
        g_max_us=40.0,
        drn_conductance_at_g_max=reference,
        noise_scale=noise_scale,
        t_inference_seconds=t_seconds,
        mapping=mapping,
    )


def _reference_afm_pcm(
    target: torch.Tensor,
    *,
    seed: int,
    max_input_size: int,
) -> torch.Tensor:
    """Independent transcription of IBM's released ``add_PCM_noise``."""

    weights = target.transpose(0, 1).contiguous()
    generator = torch.Generator(device=weights.device)
    generator.manual_seed(seed)
    epsilon = torch.finfo(torch.float16).tiny
    if max_input_size <= 0:
        scale = 1.0 / (
            weights.abs().amax(1).view(-1, 1) + epsilon
        )
    else:
        columns = weights.shape[1]
        split_count = (
            columns + max_input_size - 1
        ) // max_input_size
        base, extra = divmod(columns, split_count)
        sizes = [
            base + (index < extra) for index in range(split_count)
        ]
        expanded = []
        for tile in torch.split(weights, sizes, dim=1):
            tile_scale = 1.0 / (
                tile.abs().amax(dim=1, keepdim=True) + epsilon
            )
            expanded.append(tile_scale.expand(-1, tile.shape[1]))
        scale = torch.cat(expanded, dim=1)

    mapped = weights * scale * 180.0

    def polyval(coefficients):
        result = torch.zeros_like(mapped)
        for coefficient in coefficients:
            result = result * mapped.abs() + coefficient
        return result.to(torch.float16)

    od = (
        2.939751381126213e-05,
        -0.00371581208142707,
        0.22419685954327148,
        2.5797410660159494,
    )
    td = (
        1.2329755395573392e-05,
        -0.003054993405157235,
        0.2453487119885723,
        2.110295720240385,
    )
    noisy = mapped + torch.randn(
        mapped.shape,
        generator=generator,
        dtype=mapped.dtype,
    ) * polyval(od)
    noisy[mapped.abs() < 0.6] = 0.0
    replacement = mapped + torch.randn(
        mapped.shape,
        generator=generator,
        dtype=mapped.dtype,
    ) * polyval(td)
    mask = mapped.abs() > 52.5
    noisy[mask] = replacement[mask]
    return (noisy / 180.0 / scale).transpose(0, 1).contiguous()


def test_add_normal_uses_drn_output_channels_and_restores_clean_state() -> None:
    clean = torch.tensor([[1.0, 10.0], [3.0, 20.0]])
    weight = _dense(clean, minimum=-100.0, maximum=100.0)
    modifier = AddNormalParameterModifier(
        [weight],
        AddNormalConfig(
            std_dev=0.25,
            seed=5,
            scale_mode="output_channel_abs_max",
        ),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(5)
    expected = clean + (
        torch.randn(clean.shape, generator=generator)
        * 0.25
        * torch.tensor([[3.0, 20.0]])
    )

    with modifier.training_context():
        torch.testing.assert_close(weight.state, expected)

    torch.testing.assert_close(weight.state, clean)


def test_add_normal_resume_restores_the_next_noise_realization() -> None:
    clean = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
    first_weight = _dense(clean)
    config = AddNormalConfig(
        std_dev=0.03,
        seed=101,
        scale_mode="output_channel_abs_max",
    )
    first = AddNormalParameterModifier([first_weight], config)
    with first.training_context():
        pass
    saved = deepcopy(first.state_dict())
    with first.training_context():
        expected = first_weight.state.detach().clone()

    resumed_weight = _dense(clean)
    resumed = AddNormalParameterModifier([resumed_weight], config)
    resumed.load_state_dict(saved)
    with resumed.training_context():
        actual = resumed_weight.state.detach().clone()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_weight.state, clean)
    torch.testing.assert_close(resumed_weight.state, clean)


def test_ibm_afm_pcm_matches_the_released_helper_equations() -> None:
    target = torch.tensor(
        [
            [0.02, 0.91, 0.25],
            [0.30, 0.10, 0.72],
            [0.65, 0.42, 0.06],
            [0.11, 0.33, 0.80],
            [0.89, 0.55, 0.40],
            [0.48, 0.77, 0.19],
            [0.23, 0.68, 0.99],
        ],
        dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)

    realized, report = realize_programmed_tensor(
        target,
        _pcm_config(),
        generator=generator,
    )

    expected = _reference_afm_pcm(
        target,
        seed=17,
        max_input_size=3,
    )
    torch.testing.assert_close(realized, expected, rtol=0.0, atol=0.0)
    assert report["mapping"] == "per_output_channel_or_tile_abs_max"
    assert report["fit_target_max"] == pytest.approx(
        180.0,
        abs=0.02,
    )
    assert not any(
        key.startswith("physical_") for key in report
    )


def test_cmo_literal_mapping_preserves_the_measured_nine_us_floor() -> None:
    target = torch.tensor([[0.0], [0.05], [0.5], [1.5]])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)

    realized, report = realize_programmed_tensor(
        target,
        _cmo_config(noise_scale=0.0, t_seconds=0.0),
        generator=generator,
    )

    floor = 9.0 / 88.199997
    torch.testing.assert_close(
        realized.flatten(),
        torch.tensor([floor, floor, 0.5, 1.0]),
    )
    assert report["physical_target_min_us"] == 9.0
    assert report["physical_target_max_us"] == pytest.approx(88.199997)
    assert report["clipped_low_fraction"] == 0.5
    assert report["clipped_high_fraction"] == 0.25


def test_wan_physical_affine_mapping_preserves_the_one_us_floor() -> None:
    target = torch.tensor([[0.0], [0.5], [1.0]])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)

    realized, report = realize_programmed_tensor(
        target,
        _wan_physical_config(noise_scale=0.0),
        generator=generator,
    )

    torch.testing.assert_close(
        realized.flatten(),
        torch.tensor([0.025, 0.5125, 1.0]),
    )
    assert report["physical_target_min_us"] == 1.0
    assert report["physical_target_max_us"] == 40.0
    assert report["mapping"] == "affine_floor"
    assert report["mapping_error_rmse"] > 0.0
    assert report["programming_error_rmse"] == pytest.approx(0.0)


def test_wan_physical_literal_mapping_matches_the_paper_device_floor() -> None:
    target = torch.tensor([[0.0], [0.01], [0.5], [1.5]])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)

    realized, report = realize_programmed_tensor(
        target,
        _wan_physical_config(
            noise_scale=0.0,
            mapping="literal_conductance",
        ),
        generator=generator,
    )

    torch.testing.assert_close(
        realized.flatten(),
        torch.tensor([0.025, 0.025, 0.5, 1.0]),
    )
    assert report["clipped_low_fraction"] == 0.5
    assert report["clipped_high_fraction"] == 0.25


def test_wan_physical_noise_matches_aihwkit_1_1_polynomial() -> None:
    target = torch.tensor([[0.1], [0.5], [0.9]])
    config = _wan_physical_config(mapping="normalized_offset")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.programming_seed)

    realized, report = realize_programmed_tensor(
        target,
        config,
        generator=generator,
    )

    reference_generator = torch.Generator(device="cpu")
    reference_generator.manual_seed(config.programming_seed)
    target_us = 1.0 + 39.0 * target
    x = target_us / 40.0
    coefficients = (0.701, 22.086, -50.773, 47.095, -16.458)
    sigma = torch.zeros_like(target_us)
    power = torch.ones_like(target_us)
    for coefficient in coefficients:
        sigma = sigma + coefficient * power
        power = power * x
    final_us = (
        target_us
        + sigma
        * torch.randn(target.shape, generator=reference_generator)
    ).clamp(min=1.0, max=40.0)
    expected = (final_us - 1.0) / 39.0

    torch.testing.assert_close(realized, expected, rtol=0.0, atol=1e-7)
    assert report["noise_sigma_min_us"] == pytest.approx(float(sigma.min()))
    assert report["noise_sigma_max_us"] == pytest.approx(float(sigma.max()))


def test_cmo_normalized_offset_maps_zero_to_the_physical_floor() -> None:
    target = torch.tensor([[0.0], [0.5], [1.0]])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)

    realized, report = realize_programmed_tensor(
        target,
        _cmo_config(
            noise_scale=0.0,
            t_seconds=0.0,
            mapping="normalized_offset",
        ),
        generator=generator,
    )

    torch.testing.assert_close(realized, target)
    assert report["physical_target_min_us"] == 9.0
    assert report["physical_target_max_us"] == pytest.approx(88.199997)
    assert report["mapping"] == "normalized_offset"


def test_cmo_affine_floor_preserves_floor_and_weight_ordering() -> None:
    target = torch.tensor([[0.0], [0.5], [1.0]])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)

    realized, report = realize_programmed_tensor(
        target,
        _cmo_config(
            noise_scale=0.0,
            t_seconds=0.0,
            mapping="affine_floor",
        ),
        generator=generator,
    )

    floor = 9.0 / 88.199997
    torch.testing.assert_close(
        realized.flatten(),
        torch.tensor(
            [
                floor,
                floor + 0.5 * (1.0 - floor),
                1.0,
            ]
        ),
    )
    assert report["physical_target_min_us"] == 9.0
    assert report["physical_target_max_us"] == pytest.approx(88.199997)
    assert report["clipped_low_fraction"] == 0.0
    assert report["clipped_high_fraction"] == 0.0
    assert report["mapping"] == "affine_floor"
    assert report["programming_error_rmse"] == pytest.approx(0.0)


def test_cmo_affine_floor_supports_a_non_unit_reference() -> None:
    target = torch.tensor([[0.0], [0.1], [0.2]])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)

    realized, report = realize_programmed_tensor(
        target,
        _cmo_config(
            noise_scale=0.0,
            t_seconds=0.0,
            mapping="affine_floor",
            reference=0.2,
        ),
        generator=generator,
    )

    floor = 9.0 / 88.199997
    torch.testing.assert_close(
        realized.flatten(),
        0.2 * floor + target.flatten() * (1.0 - floor),
    )
    assert report["clipped_low_fraction"] == 0.0
    assert report["clipped_high_fraction"] == 0.0
    assert report["programming_error_rmse"] == pytest.approx(0.0)


def test_cmo_affine_and_normalized_mappings_share_physical_noise() -> None:
    target = torch.tensor([[0.0], [0.25], [0.9], [1.0]])
    affine_generator = torch.Generator(device="cpu")
    affine_generator.manual_seed(23)
    normalized_generator = torch.Generator(device="cpu")
    normalized_generator.manual_seed(23)

    affine, affine_report = realize_programmed_tensor(
        target,
        _cmo_config(mapping="affine_floor"),
        generator=affine_generator,
    )
    normalized, normalized_report = realize_programmed_tensor(
        target,
        _cmo_config(mapping="normalized_offset"),
        generator=normalized_generator,
    )

    floor = 9.0 / 88.199997
    torch.testing.assert_close(
        affine,
        floor + normalized * (1.0 - floor),
    )
    for name in (
        "physical_target_min_us",
        "physical_target_max_us",
        "physical_realized_min_us",
        "physical_realized_max_us",
        "programming_error_rmse",
    ):
        assert affine_report[name] == pytest.approx(
            normalized_report[name]
            if name.startswith("physical_")
            else normalized_report[name] * (1.0 - floor)
        )


def test_cmo_rejects_an_unknown_mapping() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)

    with pytest.raises(
        ValueError,
        match="Expected device_programming.mapping to be "
        "'literal_conductance', 'affine_floor', or 'normalized_offset'",
    ):
        realize_programmed_tensor(
            torch.tensor([[0.5]]),
            _cmo_config(mapping="unknown"),
            generator=generator,
        )


def test_cmo_noise_matches_aihwkit_1_1_equations() -> None:
    target = torch.tensor([[0.1], [0.5], [0.9]])
    config = _cmo_config(mapping="normalized_offset")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.programming_seed)

    realized, _ = realize_programmed_tensor(
        target,
        config,
        generator=generator,
    )

    reference_generator = torch.Generator(device="cpu")
    reference_generator.manual_seed(config.programming_seed)
    g_min = 9.0
    g_max = 88.199997
    target_us = g_min + target * (g_max - g_min)
    sigma_program = 0.00081107 + 0.00106879 * target_us
    programmed = target_us + sigma_program * torch.randn(
        target.shape,
        generator=reference_generator,
    )
    drifted = programmed + 0.41183342 * torch.randn(
        target.shape,
        generator=reference_generator,
    )
    sigma_read = (
        0.0277316483
        * torch.log10(drifted)
        * math.sqrt(math.log((1.0 + 1e-6) / (2.0e-6)))
    )
    final_us = (
        drifted
        + sigma_read
        * torch.randn(
            target.shape,
            generator=reference_generator,
        )
    ).clamp(min=g_min, max=g_max)
    expected = (final_us - g_min) / (g_max - g_min)

    torch.testing.assert_close(realized, expected, rtol=0.0, atol=1e-7)


def test_program_verify_updates_the_shadow_then_performs_a_noisy_write() -> None:
    weight = _dense(torch.tensor([[0.2], [0.4]]))
    catalog = ParameterCatalog(
        [
            ParameterBinding(
                "base.dense_weight.0",
                weight,
                role="dense_weight",
            )
        ]
    )
    digital = torch.optim.SGD([weight.state], lr=0.1)
    optimizer = ProgramVerifyOptimizer(
        digital,
        catalog,
        _pcm_config(noise_scale=0.0),
    )
    optimizer.initialize_from_loaded_targets()
    weight.state.grad = torch.ones_like(weight.state)

    optimizer.step()

    torch.testing.assert_close(
        weight.state,
        torch.tensor([[0.1], [0.3]]),
    )
    assert optimizer.programming_report["step_count"] == 1
    assert optimizer.programming_report["pulse_model"] is False


def test_program_verify_separates_mapping_and_programming_error() -> None:
    weight = _dense(torch.tensor([[0.0], [0.5], [1.0]]))
    catalog = ParameterCatalog(
        [
            ParameterBinding(
                "base.dense_weight.0",
                weight,
                role="dense_weight",
            )
        ]
    )
    optimizer = ProgramVerifyOptimizer(
        torch.optim.SGD([weight.state], lr=0.0),
        catalog,
        _cmo_config(
            noise_scale=0.0,
            t_seconds=0.0,
            mapping="affine_floor",
        ),
    )

    report = optimizer.initialize_from_loaded_targets()

    assert report["error_reference"] == (
        "realized_effective_drn_minus_clean_digital_target"
    )
    assert report["mapping_error_rmse"] > 0.0
    assert report["programming_error_rmse"] == pytest.approx(0.0)
    assert report["programming_error_reference"] == (
        "realized_effective_drn_minus_ideal_mapped_target"
    )


def test_program_verify_resume_restores_shadow_and_rng_exactly() -> None:
    first_weight = _dense(torch.tensor([[0.3], [0.6]]))
    first_catalog = ParameterCatalog(
        [
            ParameterBinding(
                "base.dense_weight.0",
                first_weight,
                role="dense_weight",
            )
        ]
    )
    first = ProgramVerifyOptimizer(
        torch.optim.SGD([first_weight.state], lr=0.01),
        first_catalog,
        _cmo_config(),
    )
    first.initialize_from_loaded_targets()
    first_weight.state.grad = torch.tensor([[0.2], [-0.1]])
    first.step()
    saved_state = deepcopy(first.state_dict())
    saved_live_weight = first_weight.state.detach().clone()

    resumed_weight = _dense(saved_live_weight)
    resumed_catalog = ParameterCatalog(
        [
            ParameterBinding(
                "base.dense_weight.0",
                resumed_weight,
                role="dense_weight",
            )
        ]
    )
    resumed = ProgramVerifyOptimizer(
        torch.optim.SGD([resumed_weight.state], lr=0.01),
        resumed_catalog,
        _cmo_config(),
    )
    resumed.load_state_dict(saved_state)

    gradient = torch.tensor([[0.05], [0.07]])
    first_weight.state.grad = gradient.clone()
    resumed_weight.state.grad = gradient.clone()
    first.step()
    resumed.step()

    torch.testing.assert_close(
        resumed_weight.state,
        first_weight.state,
        rtol=0.0,
        atol=0.0,
    )
    assert resumed.programming_report == first.programming_report
