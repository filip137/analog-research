"""Neutral strict configuration and serialization primitives."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping

class ConfigError(ValueError):
    """Raised when an experiment document does not match its declared schema."""

def config_error(path: str, expected: str, provided: Any) -> ConfigError:
    """Build a consistently ordered validation error.

    Repository guidance requires the expected format to precede the provided
    value.  Keeping that construction here makes nested parsers consistent.
    """

    return ConfigError(
        f"Expected {path} {expected}. Provided value: {provided!r}."
    )

def freeze_json(value: Any, *, path: str = "value") -> Any:
    """Validate and recursively freeze a JSON-compatible value."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise config_error(path, "to contain only finite numbers", value)
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise config_error(
                    path,
                    "to be a JSON object with string keys",
                    value,
                )
            frozen[key] = freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    raise config_error(path, "to be valid JSON data", value)

def to_plain_data(value: Any) -> Any:
    """Convert frozen schema objects back to ordinary JSON-compatible data."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_plain_data(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): to_plain_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [to_plain_data(item) for item in value]
    return value
