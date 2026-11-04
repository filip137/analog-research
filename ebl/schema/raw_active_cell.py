"""Strict config for raw-active common-cell IBM OM P&V."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from ebl.schema.base import RunMode, config_error


EXPERIMENT_ID = "ibm_om_raw_active_cell_program_verify.v1"
SCHEMA_VERSION = 1
SMOKE_PROFILE = "smoke"
P90_COMMON_CELL_9_PROFILE = "p90_common_cell_9"
SUPPORTED_PROFILES = (SMOKE_PROFILE, P90_COMMON_CELL_9_PROFILE)

RAW_ACTIVE_BINDING_KEYS = (
    "base.dense_weight.0",
    "base.dense_weight.1",
)
RAW_ACTIVE_BINDING_SHAPES = ((1568, 100), (100, 20))
# The widest exact float64 p90 interval is
# [0.5506438854350477, 0.6496889454832263].  The executable plant is float32,
# so freeze the immediately inward float32 endpoints.  This preserves exactly
# 142,920/158,800 development cells instead of losing two boundary cells to
# outward rounding.
RAW_ACTIVE_CELL_DERIVED_INTERVAL_MINIMUM = 0.5506438854350477
RAW_ACTIVE_CELL_DERIVED_INTERVAL_MAXIMUM = 0.6496889454832263
RAW_ACTIVE_CELL_CODEBOOK_MINIMUM = 0.5506439208984375
RAW_ACTIVE_CELL_CODEBOOK_MAXIMUM = 0.6496888995170593
RAW_ACTIVE_CELL_CODEBOOK_LEVELS = 9
RAW_ACTIVE_CELL_CODEBOOK_STEP = (
    RAW_ACTIVE_CELL_CODEBOOK_MAXIMUM - RAW_ACTIVE_CELL_CODEBOOK_MINIMUM
) / (RAW_ACTIVE_CELL_CODEBOOK_LEVELS - 1)
RAW_ACTIVE_CELL_CODEBOOK_BASELINE = 0.5 * (
    RAW_ACTIVE_CELL_CODEBOOK_MINIMUM + RAW_ACTIVE_CELL_CODEBOOK_MAXIMUM
)
RAW_ACTIVE_CELL_TOLERANCE = 0.00791586015294392


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise config_error(path, "to be a JSON object", value)
    return value


def _keys(
    value: Mapping[str, Any],
    path: str,
    required: set[str],
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        raise config_error(
            path,
            f"to contain exactly required keys {sorted(required)!r}",
            {"missing": missing, "unknown": unknown},
        )


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise config_error(path, f"to be an integer >= {minimum}", value)
    return value


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise config_error(path, "to be a finite number", value)
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        condition = "to be a finite number"
        if minimum is not None:
            condition += f" >= {minimum}"
        raise config_error(path, condition, value)
    return result


@dataclass(frozen=True)
class RuntimeSettings:
    device: str
    dtype: str
    required_aihwkit_version: str
    assignment_seed: int
    selection_seed: int
    conditioning_seed: int
    pulse_seed: int
    repeat_seed: int
    partition_seed: int
    analysis_seed: int


@dataclass(frozen=True)
class DeviceSettings:
    preset: str
    corruption_policy: str
    binding_keys: tuple[str, ...]
    binding_shapes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PartitionSettings:
    calibration: float
    fit: float
    validation: float


@dataclass(frozen=True)
class ConditioningSettings:
    quiet_steps: int
    persistent_change_threshold_raw_a: float
    maximum_pulses: int


@dataclass(frozen=True)
class CharacterizeSettings:
    profile: str
    sampled_devices: int
    repeats_per_device: int
    codebook_minimum: float
    codebook_maximum: float
    codebook_levels: int
    tolerance: float
    controller: str
    start_protocol: str
    maximum_program_pulses: int
    partitions: PartitionSettings
    conditioning: ConditioningSettings


@dataclass(frozen=True)
class RawActiveCellProgramVerifyConfig:
    schema_version: int
    experiment_id: str
    runtime: RuntimeSettings
    device: DeviceSettings
    modes: Mapping[str, CharacterizeSettings]


@dataclass(frozen=True)
class RawActiveCellProgramVerifySpec:
    experiment_id: str
    runtime: RuntimeSettings
    device: DeviceSettings
    settings: CharacterizeSettings


def _parse_runtime(value: Any) -> RuntimeSettings:
    path = "config.runtime"
    raw = _object(value, path)
    required = {
        "device",
        "dtype",
        "required_aihwkit_version",
        "assignment_seed",
        "selection_seed",
        "conditioning_seed",
        "pulse_seed",
        "repeat_seed",
        "partition_seed",
        "analysis_seed",
    }
    _keys(raw, path, required)
    if raw["device"] != "cpu":
        raise config_error(f"{path}.device", "to equal 'cpu'", raw["device"])
    if raw["dtype"] != "float32":
        raise config_error(f"{path}.dtype", "to equal 'float32'", raw["dtype"])
    if raw["required_aihwkit_version"] != "1.1.0":
        raise config_error(
            f"{path}.required_aihwkit_version",
            "to equal '1.1.0'",
            raw["required_aihwkit_version"],
        )
    seed_names = (
        "assignment_seed",
        "selection_seed",
        "conditioning_seed",
        "pulse_seed",
        "repeat_seed",
        "partition_seed",
        "analysis_seed",
    )
    seeds = {
        name: _integer(raw[name], f"{path}.{name}", minimum=1)
        for name in seed_names
    }
    if len(set(seeds.values())) != len(seeds):
        raise config_error(path, "to use distinct seeds for every role", seeds)
    return RuntimeSettings(
        device="cpu",
        dtype="float32",
        required_aihwkit_version="1.1.0",
        **seeds,
    )


def _parse_device(value: Any) -> DeviceSettings:
    path = "config.device"
    raw = _object(value, path)
    _keys(raw, path, {"preset", "corruption_policy", "bindings"})
    if raw["preset"] != "reram_array_om":
        raise config_error(
            f"{path}.preset", "to equal 'reram_array_om'", raw["preset"]
        )
    if raw["corruption_policy"] != "counterfactual_repaired":
        raise config_error(
            f"{path}.corruption_policy",
            "to equal 'counterfactual_repaired'",
            raw["corruption_policy"],
        )
    bindings = raw["bindings"]
    if not isinstance(bindings, list):
        raise config_error(f"{path}.bindings", "to be a list", bindings)
    keys: list[str] = []
    shapes: list[tuple[int, int]] = []
    for index, item in enumerate(bindings):
        binding_path = f"{path}.bindings[{index}]"
        binding = _object(item, binding_path)
        _keys(binding, binding_path, {"key", "shape"})
        key = binding["key"]
        shape = binding["shape"]
        if not isinstance(key, str) or not key:
            raise config_error(f"{binding_path}.key", "to be non-empty", key)
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in shape
            )
        ):
            raise config_error(
                f"{binding_path}.shape",
                "to contain two positive integers",
                shape,
            )
        keys.append(key)
        shapes.append((shape[0], shape[1]))
    if tuple(keys) != RAW_ACTIVE_BINDING_KEYS or tuple(shapes) != RAW_ACTIVE_BINDING_SHAPES:
        raise config_error(
            f"{path}.bindings",
            "to equal the frozen 158,800-cell DRN binding layout",
            bindings,
        )
    return DeviceSettings(
        preset="reram_array_om",
        corruption_policy="counterfactual_repaired",
        binding_keys=tuple(keys),
        binding_shapes=tuple(shapes),
    )


def _parse_partitions(value: Any, path: str) -> PartitionSettings:
    raw = _object(value, path)
    _keys(raw, path, {"calibration", "fit", "validation"})
    values = {
        name: _number(raw[name], f"{path}.{name}", minimum=0.0)
        for name in ("calibration", "fit", "validation")
    }
    if any(value <= 0.0 for value in values.values()) or not math.isclose(
        sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise config_error(path, "to contain positive fractions summing to one", raw)
    return PartitionSettings(**values)


def _parse_conditioning(value: Any, path: str) -> ConditioningSettings:
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "quiet_steps",
            "persistent_change_threshold_raw_a",
            "maximum_pulses",
        },
    )
    result = ConditioningSettings(
        quiet_steps=_integer(raw["quiet_steps"], f"{path}.quiet_steps", minimum=1),
        persistent_change_threshold_raw_a=_number(
            raw["persistent_change_threshold_raw_a"],
            f"{path}.persistent_change_threshold_raw_a",
            minimum=0.0,
        ),
        maximum_pulses=_integer(
            raw["maximum_pulses"], f"{path}.maximum_pulses", minimum=1
        ),
    )
    if result.persistent_change_threshold_raw_a <= 0.0:
        raise config_error(
            f"{path}.persistent_change_threshold_raw_a",
            "to be positive",
            result.persistent_change_threshold_raw_a,
        )
    return result


def _parse_characterize(value: Any) -> CharacterizeSettings:
    path = "config.modes.characterize"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "profile",
            "sampled_devices",
            "repeats_per_device",
            "codebook",
            "tolerance",
            "controller",
            "start_protocol",
            "maximum_program_pulses",
            "partitions",
            "conditioning",
        },
    )
    profile = raw["profile"]
    if profile not in SUPPORTED_PROFILES:
        raise config_error(
            f"{path}.profile", f"to be one of {SUPPORTED_PROFILES!r}", profile
        )
    codebook = _object(raw["codebook"], f"{path}.codebook")
    _keys(codebook, f"{path}.codebook", {"minimum", "maximum", "levels"})
    result = CharacterizeSettings(
        profile=profile,
        sampled_devices=_integer(
            raw["sampled_devices"], f"{path}.sampled_devices", minimum=3
        ),
        repeats_per_device=_integer(
            raw["repeats_per_device"],
            f"{path}.repeats_per_device",
            minimum=1,
        ),
        codebook_minimum=_number(
            codebook["minimum"], f"{path}.codebook.minimum"
        ),
        codebook_maximum=_number(
            codebook["maximum"], f"{path}.codebook.maximum"
        ),
        codebook_levels=_integer(
            codebook["levels"], f"{path}.codebook.levels", minimum=2
        ),
        tolerance=_number(raw["tolerance"], f"{path}.tolerance", minimum=0.0),
        controller=str(raw["controller"]),
        start_protocol=str(raw["start_protocol"]),
        maximum_program_pulses=_integer(
            raw["maximum_program_pulses"],
            f"{path}.maximum_program_pulses",
            minimum=1,
        ),
        partitions=_parse_partitions(raw["partitions"], f"{path}.partitions"),
        conditioning=_parse_conditioning(
            raw["conditioning"], f"{path}.conditioning"
        ),
    )
    exact_values = {
        "codebook.minimum": (
            result.codebook_minimum,
            RAW_ACTIVE_CELL_CODEBOOK_MINIMUM,
        ),
        "codebook.maximum": (
            result.codebook_maximum,
            RAW_ACTIVE_CELL_CODEBOOK_MAXIMUM,
        ),
        "codebook.levels": (
            result.codebook_levels,
            RAW_ACTIVE_CELL_CODEBOOK_LEVELS,
        ),
        "tolerance": (result.tolerance, RAW_ACTIVE_CELL_TOLERANCE),
        "controller": (result.controller, "one_pulse"),
        "start_protocol": (result.start_protocol, "lower_to_target"),
        "maximum_program_pulses": (result.maximum_program_pulses, 128),
        "partitions": (result.partitions, PartitionSettings(0.2, 0.6, 0.2)),
        "conditioning": (
            result.conditioning,
            ConditioningSettings(4, 2e-6, 4096),
        ),
    }
    mismatches = {
        name: {"provided": provided, "expected": expected}
        for name, (provided, expected) in exact_values.items()
        if provided != expected
    }
    if mismatches:
        raise config_error(
            path,
            "to preserve the frozen raw-active nine-cell-level P&V protocol",
            mismatches,
        )
    if result.profile == P90_COMMON_CELL_9_PROFILE and (
        result.sampled_devices != 1024 or result.repeats_per_device != 4
    ):
        raise config_error(
            path,
            "to use 1,024 identities and four repeats for the production profile",
            {
                "sampled_devices": result.sampled_devices,
                "repeats_per_device": result.repeats_per_device,
            },
        )
    return result


def parse_raw_active_cell_program_verify_config(
    payload: Mapping[str, Any],
) -> RawActiveCellProgramVerifyConfig:
    raw = _object(payload, "config")
    _keys(
        raw,
        "config",
        {"schema_version", "experiment_id", "runtime", "device", "modes"},
    )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error(
            "config.schema_version", "to equal 1", raw["schema_version"]
        )
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"]
        )
    modes = _object(raw["modes"], "config.modes")
    _keys(modes, "config.modes", {"characterize"})
    return RawActiveCellProgramVerifyConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        runtime=_parse_runtime(raw["runtime"]),
        device=_parse_device(raw["device"]),
        modes=MappingProxyType(
            {"characterize": _parse_characterize(modes["characterize"])}
        ),
    )


def resolve_raw_active_cell_program_verify_spec(
    document: RawActiveCellProgramVerifyConfig,
    mode: RunMode,
) -> RawActiveCellProgramVerifySpec:
    if mode is not RunMode.CHARACTERIZE:
        raise config_error(
            "the requested run mode", "to equal 'characterize'", mode.value
        )
    return RawActiveCellProgramVerifySpec(
        experiment_id=document.experiment_id,
        runtime=document.runtime,
        device=document.device,
        settings=document.modes["characterize"],
    )


__all__ = [
    "EXPERIMENT_ID",
    "P90_COMMON_CELL_9_PROFILE",
    "RAW_ACTIVE_BINDING_KEYS",
    "RAW_ACTIVE_BINDING_SHAPES",
    "RAW_ACTIVE_CELL_CODEBOOK_BASELINE",
    "RAW_ACTIVE_CELL_CODEBOOK_LEVELS",
    "RAW_ACTIVE_CELL_CODEBOOK_MAXIMUM",
    "RAW_ACTIVE_CELL_CODEBOOK_MINIMUM",
    "RAW_ACTIVE_CELL_CODEBOOK_STEP",
    "RAW_ACTIVE_CELL_TOLERANCE",
    "RAW_ACTIVE_CELL_DERIVED_INTERVAL_MAXIMUM",
    "RAW_ACTIVE_CELL_DERIVED_INTERVAL_MINIMUM",
    "RawActiveCellProgramVerifyConfig",
    "RawActiveCellProgramVerifySpec",
    "SCHEMA_VERSION",
    "SMOKE_PROFILE",
    "parse_raw_active_cell_program_verify_config",
    "resolve_raw_active_cell_program_verify_spec",
]
