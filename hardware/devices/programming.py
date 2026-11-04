"""Endpoint laws for distinct measured programming models."""
from __future__ import annotations
import math
from typing import Any
from collections.abc import Mapping
import torch
from hardware.config import *

def realize_programmed_tensor(
    target: torch.Tensor,
    config: DeviceProgrammingConfig,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a noisy logical tensor and physical mapping statistics."""

    normalized = parse_device_programming_config(config)
    if isinstance(normalized, IbmAfm2025PcmProgrammingConfig):
        return _realize_ibm_afm_pcm(
            target,
            normalized,
            generator=generator,
        )
    if isinstance(normalized, Wan2022PhysicalProgrammingConfig):
        return _realize_wan2022_physical(
            target,
            normalized,
            generator=generator,
        )
    if isinstance(normalized, CmoHfOxProgrammingConfig):
        return _realize_cmo(
            target,
            normalized,
            generator=generator,
        )
    raise ValueError(
        "Expected repeated program-and-verify writes to use "
        f"{IBM_AFM2025_PCM!r}, {WAN2022_PHYSICAL!r}, or "
        f"{AIHWKIT_RERAM_CMO!r}. "
        f"Provided value: {normalized.type!r}."
    )

def _realize_ibm_afm_pcm(
    target: torch.Tensor,
    config: IbmAfm2025PcmProgrammingConfig,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if target.ndim != 2:
        raise ValueError(
            "Expected IBM AFM PCM programming to receive a two-dimensional "
            f"dense tensor in (pre, post) layout. Provided value: "
            f"shape={tuple(target.shape)!r}."
        )
    # Match the public AFM helper in its native PyTorch ``Linear`` layout
    # (out, in).  DRN dense tensors use (pre, post), hence the two transposes.
    weights = target.transpose(0, 1).contiguous()
    epsilon = torch.finfo(torch.float16).tiny
    if config.max_input_size <= 0:
        scale = 1.0 / (
            weights.abs().amax(dim=1).view(-1, 1) + epsilon
        )
    else:
        scales = []
        split_sizes = _balanced_split_sizes(
            weights.shape[1],
            config.max_input_size,
        )
        for tile in torch.split(weights, split_sizes, dim=1):
            tile_scale = 1.0 / (
                tile.abs().amax(dim=1, keepdim=True) + epsilon
            )
            scales.append(tile_scale.expand(-1, tile.shape[1]))
        scale = torch.cat(scales, dim=1)
    mapped = weights * scale * float(config.fit_max)
    abs_mapped = mapped.abs()
    od_sigma = _pcm_polyval(_PCM_OD_COEFFICIENTS, abs_mapped)
    td_sigma = _pcm_polyval(_PCM_TD_COEFFICIENTS, abs_mapped)
    od_noise = torch.randn(
        mapped.shape,
        dtype=mapped.dtype,
        device=mapped.device,
        generator=generator,
    )
    td_noise = torch.randn(
        mapped.shape,
        dtype=mapped.dtype,
        device=mapped.device,
        generator=generator,
    )
    realized_fit = mapped + (
        float(config.noise_scale) * od_sigma * od_noise
    )
    td_realized = mapped + (
        float(config.noise_scale) * td_sigma * td_noise
    )
    realized_fit = torch.where(
        abs_mapped > _PCM_OD_TD_DIVERGENCE,
        td_realized,
        realized_fit,
    )
    exact_zero = abs_mapped < float(config.zero_threshold)
    realized_fit = torch.where(
        exact_zero,
        torch.zeros_like(realized_fit),
        realized_fit,
    )
    realized = (
        realized_fit / float(config.fit_max) / scale
    ).transpose(0, 1).contiguous()
    return realized, {
        "fit_target_min": (
            float(mapped.min()) if mapped.numel() else None
        ),
        "fit_target_max": (
            float(mapped.max()) if mapped.numel() else None
        ),
        "fit_realized_min": (
            float(realized_fit.min()) if realized_fit.numel() else None
        ),
        "fit_realized_max": (
            float(realized_fit.max()) if realized_fit.numel() else None
        ),
        "zeroed_fraction": (
            float(exact_zero.float().mean()) if exact_zero.numel() else None
        ),
        "clipped_low_fraction": 0.0,
        "clipped_high_fraction": 0.0,
        "mapping": "per_output_channel_or_tile_abs_max",
    }

def _realize_wan2022_physical(
    target: torch.Tensor,
    config: Wan2022PhysicalProgrammingConfig,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the AIHWKit Wan fit with an explicit physical floor mapping."""

    reference = float(config.drn_conductance_at_g_max)
    g_min = float(config.g_min_us)
    g_max = float(config.g_max_us)
    if config.mapping == "literal_conductance":
        clipped_low = target < reference * (g_min / g_max)
        clipped_high = target > reference
        raw_target_us = target * (g_max / reference)
    else:
        clipped_low = target < 0.0
        clipped_high = target > reference
        raw_target_us = g_min + target * (
            (g_max - g_min) / reference
        )
    target_us = raw_target_us.clamp(min=g_min, max=g_max)

    coefficients = _WAN2022_COEFFICIENTS[
        float(config.t_inference_seconds)
    ]
    normalized_target = target_us / g_max
    power = torch.ones_like(target_us)
    sigma_us = torch.zeros_like(target_us)
    for coefficient in coefficients:
        sigma_us = sigma_us + float(coefficient) * power
        power = power * normalized_target
    sigma_us = sigma_us * (
        g_max / _WAN2022_COEFFICIENT_G_MAX_US
    )
    realized_us = target_us + (
        float(config.noise_scale)
        * sigma_us
        * torch.randn(
            target_us.shape,
            dtype=target_us.dtype,
            device=target_us.device,
            generator=generator,
        )
    )
    realized_us = realized_us.clamp(min=g_min, max=g_max)

    if config.mapping in ("literal_conductance", "affine_floor"):
        realized = realized_us * (reference / g_max)
        ideal_mapped = target_us * (reference / g_max)
    else:
        realized = (realized_us - g_min) * (
            reference / (g_max - g_min)
        )
        ideal_mapped = (target_us - g_min) * (
            reference / (g_max - g_min)
        )

    mapping_error = ideal_mapped - target
    programming_error = realized - ideal_mapped
    return realized, {
        "physical_target_min_us": (
            float(target_us.min()) if target_us.numel() else None
        ),
        "physical_target_max_us": (
            float(target_us.max()) if target_us.numel() else None
        ),
        "physical_realized_min_us": (
            float(realized_us.min()) if realized_us.numel() else None
        ),
        "physical_realized_max_us": (
            float(realized_us.max()) if realized_us.numel() else None
        ),
        "noise_sigma_min_us": (
            float(sigma_us.min()) if sigma_us.numel() else None
        ),
        "noise_sigma_max_us": (
            float(sigma_us.max()) if sigma_us.numel() else None
        ),
        "zeroed_fraction": 0.0,
        "clipped_low_fraction": (
            float(clipped_low.float().mean()) if clipped_low.numel() else None
        ),
        "clipped_high_fraction": (
            float(clipped_high.float().mean())
            if clipped_high.numel()
            else None
        ),
        "mapping_error_mean": (
            float(mapping_error.mean()) if mapping_error.numel() else None
        ),
        "mapping_error_rmse": (
            float(torch.sqrt(mapping_error.square().mean()))
            if mapping_error.numel()
            else None
        ),
        "mapping_error_abs_max": (
            float(mapping_error.abs().max())
            if mapping_error.numel()
            else None
        ),
        "programming_error_mean": (
            float(programming_error.mean())
            if programming_error.numel()
            else None
        ),
        "programming_error_rmse": (
            float(torch.sqrt(programming_error.square().mean()))
            if programming_error.numel()
            else None
        ),
        "programming_error_abs_max": (
            float(programming_error.abs().max())
            if programming_error.numel()
            else None
        ),
        "programming_error_reference": (
            "realized_effective_drn_minus_ideal_mapped_target"
        ),
        "mapping": config.mapping,
    }

def _realize_cmo(
    target: torch.Tensor,
    config: CmoHfOxProgrammingConfig,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    reference = float(config.drn_conductance_at_g_max)
    g_min = float(config.g_min_us)
    g_max = float(config.g_max_us)
    if config.mapping == "literal_conductance":
        clipped_low = target < reference * (g_min / g_max)
        clipped_high = target > reference
        raw_target_us = target * (g_max / reference)
    else:
        clipped_low = target < 0.0
        clipped_high = target > reference
        raw_target_us = g_min + target * (
            (g_max - g_min) / reference
        )
    target_us = raw_target_us.clamp(
        min=float(config.g_min_us),
        max=float(config.g_max_us),
    )
    intercept, slope = _CMO_PROGRAM_COEFFICIENTS[
        float(config.acceptance_range_percent)
    ]
    sigma_program = intercept + slope * target_us
    realized_us = target_us + (
        float(config.noise_scale)
        * sigma_program
        * torch.randn(
            target_us.shape,
            dtype=target_us.dtype,
            device=target_us.device,
            generator=generator,
        )
    )
    t_seconds = float(config.t_inference_seconds)
    if t_seconds > 0.0:
        log_t = math.log(t_seconds)
        mean = realized_us + (
            _CMO_DECAY_MEAN * log_t * float(config.drift_scale)
        )
        sigma_relaxation = (
            _CMO_DECAY_STD[0] * log_t + _CMO_DECAY_STD[1]
        )
        realized_us = mean + (
            sigma_relaxation
            * float(config.drift_scale)
            * torch.randn(
                target_us.shape,
                dtype=target_us.dtype,
                device=target_us.device,
                generator=generator,
            )
        )
        positive = realized_us.clamp_min(torch.finfo(realized_us.dtype).tiny)
        sigma_read = (
            _CMO_READ_K
            * torch.log10(positive)
            * math.sqrt(
                math.log(
                    (t_seconds + _CMO_T_READ_SECONDS)
                    / (2.0 * _CMO_T_READ_SECONDS)
                )
            )
        )
        realized_us = realized_us + (
            sigma_read
            * float(config.read_noise_scale)
            * torch.randn(
                target_us.shape,
                dtype=target_us.dtype,
                device=target_us.device,
                generator=generator,
            )
        )
    realized_us = realized_us.clamp(
        min=float(config.g_min_us),
        max=float(config.g_max_us),
    )
    if config.mapping in ("literal_conductance", "affine_floor"):
        realized = realized_us * (reference / g_max)
        ideal_mapped = target_us * (reference / g_max)
    else:
        realized = (realized_us - g_min) * (
            reference / (g_max - g_min)
        )
        ideal_mapped = (target_us - g_min) * (
            reference / (g_max - g_min)
        )
    mapping_error = ideal_mapped - target
    programming_error = realized - ideal_mapped
    return realized, {
        "physical_target_min_us": (
            float(target_us.min()) if target_us.numel() else None
        ),
        "physical_target_max_us": (
            float(target_us.max()) if target_us.numel() else None
        ),
        "physical_realized_min_us": (
            float(realized_us.min()) if realized_us.numel() else None
        ),
        "physical_realized_max_us": (
            float(realized_us.max()) if realized_us.numel() else None
        ),
        "zeroed_fraction": 0.0,
        "clipped_low_fraction": (
            float(clipped_low.float().mean()) if clipped_low.numel() else None
        ),
        "clipped_high_fraction": (
            float(clipped_high.float().mean())
            if clipped_high.numel()
            else None
        ),
        "mapping_error_mean": (
            float(mapping_error.mean()) if mapping_error.numel() else None
        ),
        "mapping_error_rmse": (
            float(torch.sqrt(mapping_error.square().mean()))
            if mapping_error.numel()
            else None
        ),
        "mapping_error_abs_max": (
            float(mapping_error.abs().max())
            if mapping_error.numel()
            else None
        ),
        "programming_error_mean": (
            float(programming_error.mean())
            if programming_error.numel()
            else None
        ),
        "programming_error_rmse": (
            float(torch.sqrt(programming_error.square().mean()))
            if programming_error.numel()
            else None
        ),
        "programming_error_abs_max": (
            float(programming_error.abs().max())
            if programming_error.numel()
            else None
        ),
        "programming_error_reference": (
            "realized_effective_drn_minus_ideal_mapped_target"
        ),
        "mapping": config.mapping,
    }

def _balanced_split_sizes(size: int, split_max_size: int) -> list[int]:
    """Return the balanced tile sizes used by the released AFM helper."""

    if split_max_size <= 0:
        return [size]
    n_splits = (size + split_max_size - 1) // split_max_size
    base, extra = divmod(size, n_splits)
    return [base + (index < extra) for index in range(n_splits)]

def _pcm_polyval(
    coefficients: tuple[float, ...],
    values: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the released polynomial and retain its FP16 quantization."""

    result = torch.zeros_like(values)
    for coefficient in coefficients:
        result = result * values + float(coefficient)
    result = result.to(torch.float16)
    if torch.any(torch.isinf(result)) or torch.any(torch.isnan(result)):
        raise RuntimeError(
            "Expected IBM AFM PCM polynomial values to remain finite after "
            "FP16 conversion. Provided value: overflow or NaN."
        )
    return result

def _load_wan2022():
    try:
        import aihwkit
        from aihwkit.inference.noise.reram import ReRamWan2022NoiseModel
    except ImportError as error:
        raise RuntimeError(
            "Expected the optional 'aihwkit' package to apply "
            "aihwkit_reram_wan2022 device noise. Provided value: "
            "aihwkit is not importable in this Python environment."
        ) from error
    return aihwkit, ReRamWan2022NoiseModel

_PCM_OD_TD_DIVERGENCE = 52.5

_PCM_OD_COEFFICIENTS = (
    2.939751381126213e-05,
    -0.00371581208142707,
    0.22419685954327148,
    2.5797410660159494,
)

_PCM_TD_COEFFICIENTS = (
    1.2329755395573392e-05,
    -0.003054993405157235,
    0.2453487119885723,
    2.110295720240385,
)

_CMO_PROGRAM_COEFFICIENTS = {
    0.2: (0.00081107, 0.00106879),
    2.0: (0.0112185391, 0.01129027418),
}

_CMO_DECAY_MEAN = -0.08900206

_CMO_DECAY_STD = (0.04201137, 0.41183342)

_CMO_READ_K = 0.0277316483

_CMO_T_READ_SECONDS = 1e-6

_WAN2022_COEFFICIENTS = {
    1.0: (0.348, 16.030, -43.853, 45.393, -16.815),
    86400.0: (0.701, 22.086, -50.773, 47.095, -16.458),
    172800.0: (0.782, 20.274, -43.507, 37.062, -11.934),
}

_WAN2022_COEFFICIENT_G_MAX_US = 40.0
