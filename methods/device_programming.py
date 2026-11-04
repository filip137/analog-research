"""One-shot physical device programming for frozen DRN conductances."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from networks.resistive.builders import ParameterCatalog
from hardware.config import (
    AIHWKIT_RERAM_CMO,
    IBM_AFM2025_PCM,
    WAN2022_PHYSICAL,
    CmoHfOxProgrammingConfig,
    DeviceProgrammingConfig,
    IbmAfm2025PcmProgrammingConfig,
    Wan2022PhysicalProgrammingConfig,
    Wan2022ProgrammingConfig,
    device_programming_to_mapping,
    parse_device_programming_config,
)


from hardware.devices.programming import _PCM_OD_TD_DIVERGENCE
from hardware.devices.programming import _PCM_OD_COEFFICIENTS
from hardware.devices.programming import _PCM_TD_COEFFICIENTS
from hardware.devices.programming import _CMO_PROGRAM_COEFFICIENTS
from hardware.devices.programming import _CMO_DECAY_MEAN
from hardware.devices.programming import _CMO_DECAY_STD
from hardware.devices.programming import _CMO_READ_K
from hardware.devices.programming import _CMO_T_READ_SECONDS
from hardware.devices.programming import _WAN2022_COEFFICIENTS
from hardware.devices.programming import _WAN2022_COEFFICIENT_G_MAX_US


def program_wan2022_base_conductance(
    catalog: ParameterCatalog,
    config: Wan2022ProgrammingConfig,
) -> dict[str, Any]:
    """Replace one base matrix by a deterministic Wan-2022 realization."""

    bindings = catalog.for_group("base", checkpointed_only=True)
    dense = tuple(binding for binding in bindings if binding.role == "dense_weight")
    if len(bindings) != 1 or len(dense) != 1:
        raise ValueError(
            "Expected Wan-2022 programming to target exactly one checkpointed "
            "base dense weight. Provided value: "
            f"base_keys={[binding.key for binding in bindings]!r}."
        )
    _validate_wan2022_binding(dense[0], config)
    aihwkit, noise_model_type = _load_wan2022()
    return _program_wan2022_binding(
        dense[0],
        config,
        aihwkit=aihwkit,
        noise_model_type=noise_model_type,
    )


def program_wan2022_base_conductances(
    catalog: ParameterCatalog,
    configs: Mapping[str, Wan2022ProgrammingConfig],
) -> dict[str, Any]:
    """Program every dense base edge using its stable-keyed device config."""

    if not isinstance(configs, Mapping):
        raise ValueError(
            "Expected Wan-2022 layer configurations to be an object keyed by "
            f"base dense parameter name. Provided value: {configs!r}."
        )
    dense = tuple(
        binding
        for binding in catalog.for_group("base", checkpointed_only=True)
        if binding.role == "dense_weight"
    )
    expected_keys = tuple(binding.key for binding in dense)
    provided_keys = tuple(configs)
    if set(provided_keys) != set(expected_keys):
        raise ValueError(
            "Expected Wan-2022 layer configurations to target every "
            f"checkpointed base dense weight {expected_keys!r}. Provided "
            f"value: {provided_keys!r}."
        )
    if not dense:
        raise ValueError(
            "Expected at least one checkpointed base dense weight for "
            "Wan-2022 programming. Provided value: empty dense base group."
        )
    for key, config in configs.items():
        if not isinstance(config, Wan2022ProgrammingConfig):
            raise ValueError(
                "Expected every Wan-2022 layer configuration to be a "
                "Wan2022ProgrammingConfig. Provided value: "
                f"key={key!r}, config={config!r}."
            )
    for binding in dense:
        _validate_wan2022_binding(binding, configs[binding.key])

    aihwkit, noise_model_type = _load_wan2022()
    reports = [
        _program_wan2022_binding(
            binding,
            configs[binding.key],
            aihwkit=aihwkit,
            noise_model_type=noise_model_type,
        )
        for binding in dense
    ]
    count = sum(report["count"] for report in reports)

    def weighted(name: str) -> float | None:
        if not count:
            return None
        return sum(
            float(report[name]) * report["count"] for report in reports
        ) / count

    aggregate_rmse = None
    if count:
        aggregate_rmse = math.sqrt(
            sum(
                float(report["error_rmse"]) ** 2 * report["count"]
                for report in reports
            )
            / count
        )
    return {
        "model": "aihwkit_reram_wan2022",
        "aihwkit_version": str(getattr(aihwkit, "__version__", "unknown")),
        "parameter_keys": list(expected_keys),
        "parameters": reports,
        "count": count,
        "error_rmse": aggregate_rmse,
        "error_abs_max": max(
            float(report["error_abs_max"]) for report in reports
        ),
        "clipped_low_fraction": weighted("clipped_low_fraction"),
        "clipped_high_fraction": weighted("clipped_high_fraction"),
        "changed_fraction": weighted("changed_fraction"),
    }


def program_device_base_conductances(
    catalog: ParameterCatalog,
    configs: Mapping[str, DeviceProgrammingConfig],
) -> dict[str, Any]:
    """Program every checkpointed dense base edge through a tagged model."""

    if not isinstance(configs, Mapping):
        raise ValueError(
            "Expected device layer configurations to be an object keyed by "
            f"base dense parameter name. Provided value: {configs!r}."
        )
    dense = tuple(
        binding
        for binding in catalog.for_group("base", checkpointed_only=True)
        if binding.role == "dense_weight"
    )
    expected_keys = tuple(binding.key for binding in dense)
    if set(configs) != set(expected_keys):
        raise ValueError(
            "Expected device layer configurations to target every "
            f"checkpointed base dense weight {expected_keys!r}. Provided "
            f"value: {tuple(configs)!r}."
        )
    if not dense:
        raise ValueError(
            "Expected at least one checkpointed base dense weight. "
            "Provided value: empty dense base group."
        )
    normalized = {
        key: parse_device_programming_config(
            value,
            path=f"device_programming.{key}",
        )
        for key, value in configs.items()
    }
    if all(
        isinstance(value, Wan2022ProgrammingConfig)
        for value in normalized.values()
    ):
        return program_wan2022_base_conductances(
            catalog,
            normalized,  # type: ignore[arg-type]
        )

    reports = []
    generators: dict[tuple[str, int], torch.Generator] = {}
    for binding in dense:
        config = normalized[binding.key]
        generator_key = (
            str(binding.state.device),
            int(config.programming_seed),
        )
        generator = generators.get(generator_key)
        if generator is None:
            generator = torch.Generator(device=binding.state.device)
            generator.manual_seed(int(config.programming_seed))
            generators[generator_key] = generator
        reports.append(
            program_device_binding(
                binding,
                config,
                generator=generator,
            )
        )
    return _aggregate_programming_reports(reports)


def program_device_binding(
    binding: Any,
    config: DeviceProgrammingConfig,
    *,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Replace one bound tensor by one program-and-verify realization."""

    normalized = parse_device_programming_config(config)
    clean = binding.state.detach().clone()
    if not torch.isfinite(clean).all() or torch.any(clean < 0.0):
        raise ValueError(
            "Expected a device-programmed DRN tensor to contain finite "
            f"non-negative conductances. Provided value: key={binding.key!r}."
        )
    raw_realized, physical = realize_programmed_tensor(
        clean,
        normalized,
        generator=generator,
    )
    lower = getattr(binding.parameter, "min_cond", None)
    upper = getattr(binding.parameter, "max_cond", None)
    realized = raw_realized.clamp(min=lower, max=upper)
    with torch.no_grad():
        binding.state.copy_(realized)
    error = realized - clean
    return {
        "model": normalized.type,
        "parameter_key": binding.key,
        "configuration": device_programming_to_mapping(normalized),
        "count": int(clean.numel()),
        "clean_min": float(clean.min()) if clean.numel() else None,
        "clean_max": float(clean.max()) if clean.numel() else None,
        "realized_min": float(realized.min()) if realized.numel() else None,
        "realized_max": float(realized.max()) if realized.numel() else None,
        "error_mean": float(error.mean()) if error.numel() else None,
        "error_rmse": (
            float(torch.sqrt(error.square().mean())) if error.numel() else None
        ),
        "error_abs_max": (
            float(error.abs().max()) if error.numel() else None
        ),
        "error_reference": "realized_effective_drn_minus_clean_digital_target",
        "changed_fraction": (
            float(torch.count_nonzero(error).item() / error.numel())
            if error.numel()
            else None
        ),
        **physical,
    }


from hardware.devices.programming import realize_programmed_tensor


from hardware.devices.programming import _realize_ibm_afm_pcm


from hardware.devices.programming import _realize_wan2022_physical


from hardware.devices.programming import _realize_cmo


from hardware.devices.programming import _balanced_split_sizes


from hardware.devices.programming import _pcm_polyval


def _aggregate_programming_reports(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    count = sum(int(report["count"]) for report in reports)

    def weighted(name: str) -> float | None:
        if not count:
            return None
        values = [
            (report.get(name), int(report["count"])) for report in reports
        ]
        if any(value is None for value, _ in values):
            return None
        return sum(float(value) * size for value, size in values) / count

    def aggregate_rmse(name: str) -> float | None:
        if not count or any(report.get(name) is None for report in reports):
            return None
        return math.sqrt(
            sum(
                float(report[name]) ** 2 * int(report["count"])
                for report in reports
            )
            / count
        )

    aggregate_error_rmse = aggregate_rmse("error_rmse")
    aggregate_mapping_rmse = aggregate_rmse("mapping_error_rmse")
    aggregate_programming_rmse = aggregate_rmse("programming_error_rmse")

    def maximum(name: str) -> float | None:
        values = [report.get(name) for report in reports]
        if any(value is None for value in values):
            return None
        return max(float(value) for value in values)

    return {
        "model": (
            reports[0]["model"]
            if len({report["model"] for report in reports}) == 1
            else "mixed"
        ),
        "parameter_keys": [report["parameter_key"] for report in reports],
        "parameters": reports,
        "count": count,
        "error_rmse": aggregate_error_rmse,
        "error_abs_max": max(
            float(report["error_abs_max"]) for report in reports
        ),
        "error_reference": "realized_effective_drn_minus_clean_digital_target",
        "mapping_error_rmse": aggregate_mapping_rmse,
        "mapping_error_abs_max": maximum("mapping_error_abs_max"),
        "programming_error_rmse": aggregate_programming_rmse,
        "programming_error_abs_max": maximum("programming_error_abs_max"),
        "programming_error_reference": (
            "realized_effective_drn_minus_ideal_mapped_target"
        ),
        "clipped_low_fraction": weighted("clipped_low_fraction"),
        "clipped_high_fraction": weighted("clipped_high_fraction"),
        "changed_fraction": weighted("changed_fraction"),
    }


from hardware.devices.programming import _load_wan2022


def _program_wan2022_binding(
    binding,
    config: Wan2022ProgrammingConfig,
    *,
    aihwkit,
    noise_model_type,
) -> dict[str, Any]:
    clean, reference, clean_max = _validate_wan2022_binding(binding, config)
    target_us = clean.to(dtype=torch.float32) * (
        float(config.g_max_us) / reference
    )
    model = noise_model_type(
        g_max=float(config.g_max_us),
        noise_scale=float(config.noise_scale),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config.programming_seed))
        realized_us = model.apply_drift_noise_to_conductance(
            target_us,
            target_us,
            float(config.t_inference_seconds),
        )
    raw_realized = realized_us.to(dtype=clean.dtype) * (
        reference / float(config.g_max_us)
    )
    lower = getattr(binding.parameter, "min_cond", None)
    upper = getattr(binding.parameter, "max_cond", None)
    realized = raw_realized.clamp(min=lower, max=upper)
    error = realized - clean
    with torch.no_grad():
        binding.state.copy_(
            realized.to(device=binding.state.device, dtype=binding.state.dtype)
        )

    abs_error = error.abs()
    return {
        "model": config.type,
        "aihwkit_version": str(getattr(aihwkit, "__version__", "unknown")),
        "parameter_key": binding.key,
        "programming_seed": int(config.programming_seed),
        "g_max_us": float(config.g_max_us),
        "drn_conductance_at_g_max": reference,
        "noise_scale": float(config.noise_scale),
        "t_inference_seconds": float(config.t_inference_seconds),
        "count": int(clean.numel()),
        "clean_min": float(clean.min()) if clean.numel() else None,
        "clean_max": clean_max if clean.numel() else None,
        "realized_min": float(realized.min()) if realized.numel() else None,
        "realized_max": float(realized.max()) if realized.numel() else None,
        "clipped_low_fraction": (
            float((raw_realized < lower).float().mean())
            if raw_realized.numel() and lower is not None
            else 0.0
        ),
        "clipped_high_fraction": (
            float((raw_realized > upper).float().mean())
            if raw_realized.numel() and upper is not None
            else 0.0
        ),
        "error_mean": float(error.mean()) if error.numel() else None,
        "error_rmse": (
            float(torch.sqrt(error.square().mean())) if error.numel() else None
        ),
        "error_abs_max": float(abs_error.max()) if error.numel() else None,
        "changed_fraction": (
            float(torch.count_nonzero(error).item() / error.numel())
            if error.numel()
            else None
        ),
    }


def _validate_wan2022_binding(
    binding,
    config: Wan2022ProgrammingConfig,
):
    clean = binding.state.detach().cpu()
    if not torch.isfinite(clean).all() or torch.any(clean < 0.0):
        raise ValueError(
            "Expected the base DRN matrix to contain finite non-negative "
            f"conductances. Provided value: key={binding.key!r}."
        )
    reference = float(config.drn_conductance_at_g_max)
    clean_max = float(clean.max()) if clean.numel() else 0.0
    tolerance = max(1e-12, abs(reference) * 1e-6)
    if clean_max > reference + tolerance:
        raise ValueError(
            "Expected every base conductance to be no greater than "
            "device_noise.drn_conductance_at_g_max before programming. "
            f"Provided value: max={clean_max!r}, reference={reference!r}."
        )
    return clean, reference, clean_max


__all__ = [
    "program_device_base_conductances",
    "program_device_binding",
    "program_wan2022_base_conductance",
    "program_wan2022_base_conductances",
    "realize_programmed_tensor",
]
