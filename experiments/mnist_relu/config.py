"""Pure, strict schemas for the bias-free MNIST ReLU teachers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_relu.v1"
V2_EXPERIMENT_ID = "mnist_relu.v2"
SCHEMA_VERSION = 1


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
    if missing:
        raise config_error(path, f"to contain required keys {sorted(required)!r}", dict(value))
    if unknown:
        raise config_error(path, f"to contain only keys {sorted(required | optional)!r}", dict(value))


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise config_error(path, f"to be an integer >= {minimum}", value)
    return value


def _number(value: Any, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise config_error(path, f"to be a finite number >= {minimum}", value)
    result = float(value)
    if result < minimum or result == float("inf") or result != result:
        raise config_error(path, f"to be a finite number >= {minimum}", value)
    return result


def _optional_batches(value: Any, path: str) -> int | None:
    return None if value is None else _integer(value, path, minimum=1)


@dataclass(frozen=True)
class RuntimeSettings:
    seed: int
    data_seed: int
    device: str
    dtype: str


@dataclass(frozen=True)
class DataSettings:
    batch_size: int
    validation_points: int
    num_points: int | None
    shuffle: bool


@dataclass(frozen=True)
class ModelSettings:
    dims: tuple[int, int, int]
    bias: bool


@dataclass(frozen=True)
class TeacherTrainSettings:
    num_epochs: int
    learning_rate: float
    weight_decay: float
    log_every: int
    max_batches: int | None
    max_validation_batches: int | None
    minimum_validation_accuracy: float


@dataclass(frozen=True)
class TeacherValidateSettings:
    split: str
    sample_limit: int | None


@dataclass(frozen=True)
class TeacherConfig:
    schema_version: int
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: ModelSettings
    modes: Mapping[str, Any]


@dataclass(frozen=True)
class TeacherTrainSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: ModelSettings
    settings: TeacherTrainSettings


@dataclass(frozen=True)
class TeacherValidateSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: ModelSettings
    settings: TeacherValidateSettings


def _parse_runtime(value: Any) -> RuntimeSettings:
    path = "config.runtime"
    raw = _object(value, path)
    _keys(raw, path, {"seed", "data_seed", "device", "dtype"})
    device = raw["device"]
    dtype = raw["dtype"]
    if device not in {"cpu", "cuda"}:
        raise config_error(f"{path}.device", "to be 'cpu' or 'cuda'", device)
    if dtype != "float32":
        raise config_error(f"{path}.dtype", "to equal 'float32'", dtype)
    return RuntimeSettings(
        seed=_integer(raw["seed"], f"{path}.seed"),
        data_seed=_integer(raw["data_seed"], f"{path}.data_seed"),
        device=device,
        dtype=dtype,
    )


def _parse_data(value: Any) -> DataSettings:
    path = "config.data"
    raw = _object(value, path)
    _keys(raw, path, {"batch_size", "validation_points", "num_points", "shuffle"})
    shuffle = raw["shuffle"]
    if not isinstance(shuffle, bool):
        raise config_error(f"{path}.shuffle", "to be a boolean", shuffle)
    num_points = raw["num_points"]
    return DataSettings(
        batch_size=_integer(raw["batch_size"], f"{path}.batch_size", minimum=1),
        validation_points=_integer(raw["validation_points"], f"{path}.validation_points", minimum=1),
        num_points=None if num_points is None else _integer(num_points, f"{path}.num_points", minimum=1),
        shuffle=shuffle,
    )


def _parse_model_v1(value: Any) -> ModelSettings:
    path = "config.model"
    raw = _object(value, path)
    _keys(raw, path, {"dims", "bias"})
    dims = raw["dims"]
    if not isinstance(dims, (list, tuple)) or tuple(dims) != (784, 50, 10):
        raise config_error(f"{path}.dims", "to equal [784, 50, 10]", dims)
    if raw["bias"] is not False:
        raise config_error(f"{path}.bias", "to be false", raw["bias"])
    return ModelSettings(dims=(784, 50, 10), bias=False)


def _parse_model_v2(value: Any) -> ModelSettings:
    path = "config.model"
    raw = _object(value, path)
    _keys(raw, path, {"dims", "bias"})
    dims = raw["dims"]
    if not isinstance(dims, (list, tuple)) or len(dims) != 3:
        raise config_error(
            f"{path}.dims",
            "to contain [784, positive_hidden_width, 10]",
            dims,
        )
    normalized = tuple(
        _integer(item, f"{path}.dims[{index}]", minimum=1)
        for index, item in enumerate(dims)
    )
    if normalized[0] != 784 or normalized[2] != 10:
        raise config_error(
            f"{path}.dims",
            "to contain [784, positive_hidden_width, 10]",
            dims,
        )
    if raw["bias"] is not False:
        raise config_error(f"{path}.bias", "to be false", raw["bias"])
    return ModelSettings(dims=normalized, bias=False)


def _parse_train(value: Any) -> TeacherTrainSettings:
    path = "config.modes.train"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "num_epochs",
            "optimizer",
            "log_every",
            "max_batches",
            "max_validation_batches",
            "minimum_validation_accuracy",
        },
    )
    optimizer = _object(raw["optimizer"], f"{path}.optimizer")
    _keys(optimizer, f"{path}.optimizer", {"type", "learning_rate", "weight_decay"})
    if optimizer["type"] != "adam":
        raise config_error(f"{path}.optimizer.type", "to equal 'adam'", optimizer["type"])
    minimum_accuracy = _number(
        raw["minimum_validation_accuracy"],
        f"{path}.minimum_validation_accuracy",
    )
    if minimum_accuracy > 1.0:
        raise config_error(f"{path}.minimum_validation_accuracy", "to be in [0, 1]", minimum_accuracy)
    return TeacherTrainSettings(
        num_epochs=_integer(raw["num_epochs"], f"{path}.num_epochs", minimum=1),
        learning_rate=_number(optimizer["learning_rate"], f"{path}.optimizer.learning_rate"),
        weight_decay=_number(optimizer["weight_decay"], f"{path}.optimizer.weight_decay"),
        log_every=_integer(raw["log_every"], f"{path}.log_every", minimum=1),
        max_batches=_optional_batches(raw["max_batches"], f"{path}.max_batches"),
        max_validation_batches=_optional_batches(raw["max_validation_batches"], f"{path}.max_validation_batches"),
        minimum_validation_accuracy=minimum_accuracy,
    )


def _parse_validate(value: Any) -> TeacherValidateSettings:
    path = "config.modes.validate"
    raw = _object(value, path)
    _keys(raw, path, {"split", "sample_limit"})
    if raw["split"] not in {"validation", "test"}:
        raise config_error(f"{path}.split", "to be 'validation' or 'test'", raw["split"])
    return TeacherValidateSettings(
        split=raw["split"],
        sample_limit=_optional_batches(raw["sample_limit"], f"{path}.sample_limit"),
    )


def _parse_teacher_config(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
    parse_model: Callable[[Any], ModelSettings],
) -> TeacherConfig:
    raw = _object(payload, "config")
    _keys(
        raw,
        "config",
        {"schema_version", "experiment_id", "runtime", "data", "model", "modes"},
    )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error("config.schema_version", "to equal 1", raw["schema_version"])
    if raw["experiment_id"] != experiment_id:
        raise config_error(
            "config.experiment_id",
            f"to equal {experiment_id!r}",
            raw["experiment_id"],
        )
    modes_raw = _object(raw["modes"], "config.modes")
    unknown = sorted(set(modes_raw) - {"train", "validate"})
    if unknown or not modes_raw:
        raise config_error("config.modes", "to contain one or both of 'train' and 'validate'", dict(modes_raw))
    modes: dict[str, Any] = {}
    if "train" in modes_raw:
        modes["train"] = _parse_train(modes_raw["train"])
    if "validate" in modes_raw:
        modes["validate"] = _parse_validate(modes_raw["validate"])
    return TeacherConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        runtime=_parse_runtime(raw["runtime"]),
        data=_parse_data(raw["data"]),
        model=parse_model(raw["model"]),
        modes=MappingProxyType(modes),
    )


def parse_teacher_config(payload: Mapping[str, Any]) -> TeacherConfig:
    """Parse the original fixed ``784-50-10`` v1 contract."""

    return _parse_teacher_config(
        payload,
        experiment_id=EXPERIMENT_ID,
        parse_model=_parse_model_v1,
    )


def parse_teacher_v2_config(payload: Mapping[str, Any]) -> TeacherConfig:
    """Parse the configurable bias-free ``784-H-10`` v2 contract."""

    return _parse_teacher_config(
        payload,
        experiment_id=V2_EXPERIMENT_ID,
        parse_model=_parse_model_v2,
    )


def resolve_teacher_spec(document: TeacherConfig, mode: RunMode) -> Any:
    key = mode.value
    if key not in document.modes:
        raise config_error("config.modes", f"to define requested mode {key!r}", tuple(document.modes))
    common = {
        "experiment_id": document.experiment_id,
        "runtime": document.runtime,
        "data": document.data,
        "model": document.model,
        "settings": document.modes[key],
    }
    if mode is RunMode.TRAIN:
        return TeacherTrainSpec(**common)
    if mode is RunMode.VALIDATE:
        return TeacherValidateSpec(**common)
    raise config_error("the requested run mode", "to be 'train' or 'validate'", key)


__all__ = [
    "EXPERIMENT_ID",
    "V2_EXPERIMENT_ID",
    "SCHEMA_VERSION",
    "TeacherConfig",
    "TeacherTrainSpec",
    "TeacherValidateSpec",
    "parse_teacher_config",
    "parse_teacher_v2_config",
    "resolve_teacher_spec",
]
