"""Strict configuration for ``ibm_reram_program_verify.v1``."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from ebl.schema.base import RunMode, config_error


EXPERIMENT_ID = "ibm_reram_program_verify.v1"
SCHEMA_VERSION = 1
SUPPORTED_PRESETS = ("reram_array_om", "reram_array_hfo2")
SHORT_PRODUCTION_PULSE_CAPS = MappingProxyType(
    {
        "production_short": 512,
        "production_short_cap128": 128,
    }
)
HWA_PRODUCTION_PROFILE = "hwa_production_cap128"
SUPPORTED_PROFILES = (
    "smoke",
    "production",
    *SHORT_PRODUCTION_PULSE_CAPS,
    HWA_PRODUCTION_PROFILE,
)


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise config_error(path, "to be a JSON object", value)
    return value


def _keys(
    value: Mapping[str, Any],
    path: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing or unknown:
        raise config_error(
            path,
            f"to contain exactly required keys {sorted(required)!r}"
            + (f" and optional keys {sorted(optional)!r}" if optional else ""),
            {"missing": missing, "unknown": unknown},
        )


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise config_error(path, f"to be an integer >= {minimum}", value)
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise config_error(path, "to be a finite number", value)
    result = float(value)
    if not math.isfinite(result):
        raise config_error(path, "to be a finite number", value)
    if minimum is not None and result < minimum:
        raise config_error(path, f"to be >= {minimum}", value)
    if maximum is not None and result > maximum:
        raise config_error(path, f"to be <= {maximum}", value)
    return result


def _string_tuple(
    value: Any,
    path: str,
    *,
    choices: set[str],
    required_values: set[str] | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise config_error(path, "to be a non-empty list", value)
    result = tuple(value)
    if any(not isinstance(item, str) or item not in choices for item in result):
        raise config_error(path, f"to contain only {sorted(choices)!r}", value)
    if len(set(result)) != len(result):
        raise config_error(path, "not to contain duplicates", value)
    if required_values is not None and set(result) != required_values:
        raise config_error(path, f"to contain exactly {sorted(required_values)!r}", value)
    return result


@dataclass(frozen=True)
class RuntimeSettings:
    device: str
    dtype: str
    required_aihwkit_version: str
    construction_seed: int
    conditioning_seed: int
    pulse_seed: int
    repeat_seed: int
    partition_seed: int
    analysis_seed: int
    wan_seed: int


@dataclass(frozen=True)
class DeviceSettings:
    preset: str
    enable_published_corruption: bool


@dataclass(frozen=True)
class PartitionSettings:
    calibration: float
    fit: float
    validation: float


@dataclass(frozen=True)
class ConditioningSettings:
    quiet_steps: int
    persistent_change_threshold_span_fraction: float
    maximum_pulses: int


@dataclass(frozen=True)
class AdaptiveSettings:
    eta: float
    maximum_batch: int
    epsilon: float
    force_one_within_steps: float


@dataclass(frozen=True)
class CharacterizeSettings:
    profile: str
    num_devices: int
    repeats_per_device: int
    target_minimum: float
    target_maximum: float
    target_points: int
    tolerance_step_ratios: tuple[float, ...]
    start_protocols: tuple[str, ...]
    controllers: tuple[str, ...]
    maximum_program_pulses: int
    calibration_bins: int
    partitions: PartitionSettings
    conditioning: ConditioningSettings
    conditioning_scope: str
    adaptive: AdaptiveSettings
    polynomial_order: int
    standard_deviation_floor: float
    wan_g_max_us: float
    wan_noise_scale: float


@dataclass(frozen=True)
class ReramProgramVerifyConfig:
    schema_version: int
    experiment_id: str
    runtime: RuntimeSettings
    device: DeviceSettings
    modes: Mapping[str, CharacterizeSettings]


@dataclass(frozen=True)
class ReramProgramVerifySpec:
    experiment_id: str
    runtime: RuntimeSettings
    device: DeviceSettings
    settings: CharacterizeSettings


def _parse_runtime(value: Any) -> RuntimeSettings:
    path = "config.runtime"
    raw = _object(value, path)
    required = {
        "device", "dtype", "required_aihwkit_version", "construction_seed",
        "conditioning_seed", "pulse_seed", "repeat_seed", "partition_seed",
        "analysis_seed", "wan_seed",
    }
    _keys(raw, path, required)
    if raw["device"] not in {"cpu", "cuda"}:
        raise config_error(
            f"{path}.device", "to equal 'cpu' or 'cuda'", raw["device"]
        )
    if raw["dtype"] != "float32":
        raise config_error(f"{path}.dtype", "to equal 'float32'", raw["dtype"])
    version = raw["required_aihwkit_version"]
    if version != "1.1.0":
        raise config_error(
            f"{path}.required_aihwkit_version",
            "to equal '1.1.0' for this versioned pulse equation",
            version,
        )
    result = RuntimeSettings(
        device=raw["device"], dtype="float32", required_aihwkit_version=version,
        construction_seed=_integer(raw["construction_seed"], f"{path}.construction_seed"),
        conditioning_seed=_integer(raw["conditioning_seed"], f"{path}.conditioning_seed"),
        pulse_seed=_integer(raw["pulse_seed"], f"{path}.pulse_seed"),
        repeat_seed=_integer(raw["repeat_seed"], f"{path}.repeat_seed"),
        partition_seed=_integer(raw["partition_seed"], f"{path}.partition_seed"),
        analysis_seed=_integer(raw["analysis_seed"], f"{path}.analysis_seed"),
        wan_seed=_integer(raw["wan_seed"], f"{path}.wan_seed"),
    )
    recorded_seeds = {
        name: getattr(result, name)
        for name in (
            "construction_seed",
            "conditioning_seed",
            "pulse_seed",
            "repeat_seed",
            "partition_seed",
            "analysis_seed",
            "wan_seed",
        )
    }
    if len(set(recorded_seeds.values())) != len(recorded_seeds):
        raise config_error(
            path,
            "to use distinct base seeds for every stochastic role",
            recorded_seeds,
        )
    return result


def _parse_device(value: Any) -> DeviceSettings:
    path = "config.device"
    raw = _object(value, path)
    _keys(raw, path, {"preset", "enable_published_corruption"})
    if raw["preset"] not in SUPPORTED_PRESETS:
        raise config_error(f"{path}.preset", f"to be one of {SUPPORTED_PRESETS!r}", raw["preset"])
    corruption = raw["enable_published_corruption"]
    if not isinstance(corruption, bool):
        raise config_error(f"{path}.enable_published_corruption", "to be a boolean", corruption)
    return DeviceSettings(preset=raw["preset"], enable_published_corruption=corruption)


def _parse_partitions(value: Any, path: str) -> PartitionSettings:
    raw = _object(value, path)
    _keys(raw, path, {"calibration", "fit", "validation"})
    values = {key: _number(raw[key], f"{path}.{key}", minimum=0.0, maximum=1.0)
              for key in ("calibration", "fit", "validation")}
    if any(item <= 0.0 for item in values.values()) or not math.isclose(
        sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise config_error(path, "to contain positive fractions summing to 1", dict(raw))
    return PartitionSettings(**values)


def _parse_conditioning(value: Any, path: str) -> ConditioningSettings:
    raw = _object(value, path)
    _keys(raw, path, {"quiet_steps", "persistent_change_threshold_span_fraction", "maximum_pulses"})
    threshold = _number(raw["persistent_change_threshold_span_fraction"],
                        f"{path}.persistent_change_threshold_span_fraction", minimum=0.0)
    if threshold <= 0.0:
        raise config_error(f"{path}.persistent_change_threshold_span_fraction", "to be positive", threshold)
    return ConditioningSettings(
        quiet_steps=_integer(raw["quiet_steps"], f"{path}.quiet_steps", minimum=1),
        persistent_change_threshold_span_fraction=threshold,
        maximum_pulses=_integer(raw["maximum_pulses"], f"{path}.maximum_pulses", minimum=1),
    )


def _parse_adaptive(value: Any, path: str) -> AdaptiveSettings:
    raw = _object(value, path)
    _keys(raw, path, {"eta", "maximum_batch", "epsilon", "force_one_within_steps"})
    eta = _number(raw["eta"], f"{path}.eta", minimum=0.0, maximum=1.0)
    epsilon = _number(raw["epsilon"], f"{path}.epsilon", minimum=0.0)
    force = _number(raw["force_one_within_steps"], f"{path}.force_one_within_steps", minimum=0.0)
    if eta <= 0.0 or epsilon <= 0.0 or force <= 0.0:
        raise config_error(path, "to use positive eta, epsilon, and force_one_within_steps", dict(raw))
    return AdaptiveSettings(
        eta=eta,
        maximum_batch=_integer(raw["maximum_batch"], f"{path}.maximum_batch", minimum=1),
        epsilon=epsilon,
        force_one_within_steps=force,
    )


def _parse_characterize(value: Any) -> CharacterizeSettings:
    path = "config.modes.characterize"
    raw = _object(value, path)
    required = {
        "profile", "num_devices", "repeats_per_device", "target_grid", "tolerance_step_ratios",
        "start_protocols", "controllers", "maximum_program_pulses", "calibration_bins",
        "partitions", "conditioning", "adaptive", "surrogate", "wan_comparison",
    }
    _keys(raw, path, required, {"conditioning_scope"})
    profile = raw["profile"]
    if profile not in SUPPORTED_PROFILES:
        raise config_error(
            f"{path}.profile",
            f"to be one of {SUPPORTED_PROFILES!r}",
            profile,
        )
    target = _object(raw["target_grid"], f"{path}.target_grid")
    _keys(target, f"{path}.target_grid", {"minimum", "maximum", "points"})
    target_min = _number(target["minimum"], f"{path}.target_grid.minimum")
    target_max = _number(target["maximum"], f"{path}.target_grid.maximum")
    if target_min != 0.0 or target_max != 1.0:
        raise config_error(f"{path}.target_grid", "to span the fixed normalized interval [0, 1]", dict(target))
    ratios_raw = raw["tolerance_step_ratios"]
    if not isinstance(ratios_raw, (list, tuple)) or not ratios_raw:
        raise config_error(f"{path}.tolerance_step_ratios", "to be a non-empty list", ratios_raw)
    ratios = tuple(_number(item, f"{path}.tolerance_step_ratios[{index}]", minimum=0.0)
                   for index, item in enumerate(ratios_raw))
    if any(item <= 0.0 for item in ratios) or len(set(ratios)) != len(ratios):
        raise config_error(f"{path}.tolerance_step_ratios", "to contain distinct positive values", ratios_raw)
    starts = _string_tuple(
        raw["start_protocols"],
        f"{path}.start_protocols",
        choices={"lower_to_target", "upper_to_target"},
        required_values=(
            {"lower_to_target"}
            if profile == HWA_PRODUCTION_PROFILE
            else {"lower_to_target", "upper_to_target"}
        ),
    )
    controllers = _string_tuple(raw["controllers"], f"{path}.controllers",
                                choices={"one_pulse", "adaptive"},
                                required_values={"one_pulse", "adaptive"})
    if controllers != ("one_pulse", "adaptive"):
        raise config_error(
            f"{path}.controllers",
            "to equal ['one_pulse', 'adaptive'] so calibration precedes adaptation",
            raw["controllers"],
        )
    surrogate = _object(raw["surrogate"], f"{path}.surrogate")
    _keys(surrogate, f"{path}.surrogate", {"polynomial_order", "standard_deviation_floor"})
    polynomial_order = _integer(surrogate["polynomial_order"], f"{path}.surrogate.polynomial_order", minimum=1)
    if polynomial_order != 4:
        raise config_error(f"{path}.surrogate.polynomial_order", "to equal the predeclared order 4", polynomial_order)
    std_floor = _number(surrogate["standard_deviation_floor"],
                        f"{path}.surrogate.standard_deviation_floor", minimum=0.0)
    if std_floor <= 0.0:
        raise config_error(f"{path}.surrogate.standard_deviation_floor", "to be positive", std_floor)
    wan = _object(raw["wan_comparison"], f"{path}.wan_comparison")
    _keys(wan, f"{path}.wan_comparison", {"g_max_us", "noise_scale"})
    result = CharacterizeSettings(
        profile=profile,
        num_devices=_integer(raw["num_devices"], f"{path}.num_devices", minimum=3),
        repeats_per_device=_integer(raw["repeats_per_device"], f"{path}.repeats_per_device", minimum=1),
        target_minimum=target_min, target_maximum=target_max,
        target_points=_integer(target["points"], f"{path}.target_grid.points", minimum=5),
        tolerance_step_ratios=ratios, start_protocols=starts, controllers=controllers,
        maximum_program_pulses=_integer(raw["maximum_program_pulses"], f"{path}.maximum_program_pulses", minimum=1),
        calibration_bins=_integer(raw["calibration_bins"], f"{path}.calibration_bins", minimum=2),
        partitions=_parse_partitions(raw["partitions"], f"{path}.partitions"),
        conditioning=_parse_conditioning(raw["conditioning"], f"{path}.conditioning"),
        conditioning_scope=str(raw.get("conditioning_scope", "per_target")),
        adaptive=_parse_adaptive(raw["adaptive"], f"{path}.adaptive"),
        polynomial_order=polynomial_order, standard_deviation_floor=std_floor,
        wan_g_max_us=_number(wan["g_max_us"], f"{path}.wan_comparison.g_max_us", minimum=1e-12),
        wan_noise_scale=_number(wan["noise_scale"], f"{path}.wan_comparison.noise_scale", minimum=0.0),
    )
    if result.conditioning_scope not in {"per_target", "per_start"}:
        raise config_error(
            f"{path}.conditioning_scope",
            "to equal 'per_target' or 'per_start'",
            result.conditioning_scope,
        )
    if result.profile == "production":
        violations = []
        if result.num_devices < 4096:
            violations.append("num_devices must be >= 4096")
        if result.repeats_per_device < 8:
            violations.append("repeats_per_device must be >= 8")
        if result.target_points != 41:
            violations.append("target_grid.points must equal 41")
        if set(result.tolerance_step_ratios) != {0.25, 0.5, 1.0}:
            violations.append("tolerance_step_ratios must be [0.25, 0.5, 1.0]")
        if result.maximum_program_pulses != 512:
            violations.append("maximum_program_pulses must equal 512")
        if result.partitions != PartitionSettings(0.2, 0.6, 0.2):
            violations.append("partitions must equal 0.2/0.6/0.2")
        if result.conditioning != ConditioningSettings(8, 1e-6, 4096):
            violations.append("conditioning must equal quiet=8, threshold=1e-6, maximum=4096")
        if result.conditioning_scope != "per_target":
            violations.append("conditioning_scope must equal 'per_target'")
        if result.adaptive != AdaptiveSettings(0.75, 32, 1e-8, 2.0):
            violations.append("adaptive settings must equal eta=0.75, maximum=32, epsilon=1e-8, force=2")
        if result.wan_g_max_us != 40.0 or result.wan_noise_scale != 1.0:
            violations.append("Wan comparison must use g_max_us=40 and noise_scale=1")
        if violations:
            raise config_error(
                path,
                "to satisfy the immutable production characterization contract",
                violations,
            )
    if result.profile in SHORT_PRODUCTION_PULSE_CAPS:
        violations = []
        if result.num_devices != 1024:
            violations.append("num_devices must equal 1024")
        if result.repeats_per_device != 4:
            violations.append("repeats_per_device must equal 4")
        if result.target_points != 41:
            violations.append("target_grid.points must equal 41")
        if result.tolerance_step_ratios != (0.5,):
            violations.append("tolerance_step_ratios must equal [0.5]")
        expected_pulse_cap = SHORT_PRODUCTION_PULSE_CAPS[result.profile]
        if result.maximum_program_pulses != expected_pulse_cap:
            violations.append(
                "maximum_program_pulses must equal "
                f"{expected_pulse_cap} for profile {result.profile!r}"
            )
        if result.partitions != PartitionSettings(0.2, 0.6, 0.2):
            violations.append("partitions must equal 0.2/0.6/0.2")
        if result.conditioning != ConditioningSettings(4, 1e-6, 4096):
            violations.append(
                "conditioning must equal quiet=4, threshold=1e-6, maximum=4096"
            )
        if result.conditioning_scope != "per_start":
            violations.append("conditioning_scope must equal 'per_start'")
        if result.adaptive != AdaptiveSettings(0.75, 32, 1e-8, 2.0):
            violations.append(
                "adaptive settings must equal eta=0.75, maximum=32, epsilon=1e-8, force=2"
            )
        if result.wan_g_max_us != 40.0 or result.wan_noise_scale != 1.0:
            violations.append("Wan comparison must use g_max_us=40 and noise_scale=1")
        if violations:
            raise config_error(
                path,
                "to satisfy the immutable short-production characterization contract",
                violations,
            )
    if result.profile == HWA_PRODUCTION_PROFILE:
        violations = []
        if result.num_devices != 1024:
            violations.append("num_devices must equal 1024")
        if result.repeats_per_device != 4:
            violations.append("repeats_per_device must equal 4")
        if result.target_points != 41:
            violations.append("target_grid.points must equal 41")
        if result.tolerance_step_ratios != (0.5,):
            violations.append("tolerance_step_ratios must equal [0.5]")
        if result.start_protocols != ("lower_to_target",):
            violations.append("start_protocols must equal ['lower_to_target']")
        if result.controllers != ("one_pulse", "adaptive"):
            violations.append(
                "controllers must equal ['one_pulse', 'adaptive']"
            )
        if result.maximum_program_pulses != 128:
            violations.append("maximum_program_pulses must equal 128")
        if result.partitions != PartitionSettings(0.2, 0.6, 0.2):
            violations.append("partitions must equal 0.2/0.6/0.2")
        if result.conditioning != ConditioningSettings(4, 1e-6, 4096):
            violations.append(
                "conditioning must equal quiet=4, threshold=1e-6, maximum=4096"
            )
        if result.conditioning_scope != "per_start":
            violations.append("conditioning_scope must equal 'per_start'")
        if result.adaptive != AdaptiveSettings(0.75, 32, 1e-8, 2.0):
            violations.append(
                "adaptive settings must equal eta=0.75, maximum=32, epsilon=1e-8, force=2"
            )
        if result.wan_g_max_us != 40.0 or result.wan_noise_scale != 1.0:
            violations.append("Wan comparison must use g_max_us=40 and noise_scale=1")
        if violations:
            raise config_error(
                path,
                "to satisfy the immutable HWA-production characterization contract",
                violations,
            )
    return result


def parse_reram_program_verify_config(payload: Mapping[str, Any]) -> ReramProgramVerifyConfig:
    raw = _object(payload, "config")
    _keys(raw, "config", {"schema_version", "experiment_id", "runtime", "device", "modes"})
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error("config.schema_version", "to equal 1", raw["schema_version"])
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error("config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"])
    modes = _object(raw["modes"], "config.modes")
    _keys(modes, "config.modes", {"characterize"})
    device = _parse_device(raw["device"])
    characterize = _parse_characterize(modes["characterize"])
    if (
        characterize.profile == HWA_PRODUCTION_PROFILE
        and device.preset != "reram_array_om"
    ):
        raise config_error(
            "config.device.preset",
            "to equal 'reram_array_om' for hwa_production_cap128",
            device.preset,
        )
    return ReramProgramVerifyConfig(
        schema_version=SCHEMA_VERSION, experiment_id=EXPERIMENT_ID,
        runtime=_parse_runtime(raw["runtime"]), device=device,
        modes=MappingProxyType({"characterize": characterize}),
    )


def resolve_reram_program_verify_spec(
    document: ReramProgramVerifyConfig,
    mode: RunMode,
) -> ReramProgramVerifySpec:
    if mode is not RunMode.CHARACTERIZE:
        raise config_error("the requested run mode", "to equal 'characterize'", mode.value)
    return ReramProgramVerifySpec(
        experiment_id=document.experiment_id,
        runtime=document.runtime,
        device=document.device,
        settings=document.modes["characterize"],
    )


__all__ = [
    "CharacterizeSettings", "DeviceSettings", "EXPERIMENT_ID",
    "HWA_PRODUCTION_PROFILE",
    "ReramProgramVerifyConfig", "ReramProgramVerifySpec", "RuntimeSettings",
    "SCHEMA_VERSION", "parse_reram_program_verify_config",
    "resolve_reram_program_verify_spec",
]
