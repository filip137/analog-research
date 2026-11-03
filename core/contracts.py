"""Shared contracts for versioned experiment definitions.

The schema layer intentionally has no imports from numerical code.  Config
validation can therefore run in a lightweight environment before a GPU,
dataset, or optional hardware package is touched.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple


class ConfigError(ValueError):
    """Raised when an experiment document does not match its declared schema."""


class RunMode(str, Enum):
    """Execution modes shared by the public CLI and experiment definitions."""

    TRAIN = "train"
    LINSPACE = "linspace"
    VALIDATE = "validate"
    CHARACTERIZE = "characterize"


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


@dataclass(frozen=True)
class ExtensionSelection:
    """The independently swappable extension axes of a training run."""

    model_adapter: str
    weight_modifier: str
    update_backend: str
    algorithm: str


@dataclass(frozen=True)
class ValidatedCombination:
    """One combination with an explicit numerical-validation status."""

    selection: ExtensionSelection
    status: str
    note: str


Parser = Callable[[Mapping[str, Any]], Any]
Resolver = Callable[[Any, RunMode], Any]


@dataclass(frozen=True)
class ExperimentDefinition:
    """A statically registered, versioned experiment composition root."""

    experiment_id: str
    schema_version: int
    description: str
    supported_modes: Tuple[RunMode, ...]
    parser: Parser
    resolver: Resolver
    combinations: Tuple[ValidatedCombination, ...] = ()
    legacy_names: Tuple[str, ...] = ()

    def parse(self, payload: Mapping[str, Any]) -> Any:
        """Parse one document using this definition's schema."""

        return self.parser(payload)

    def resolve(self, document: Any, mode: RunMode) -> Any:
        """Resolve one parsed document to an immutable mode-specific spec."""

        if mode not in self.supported_modes:
            raise config_error(
                "the requested run mode",
                "to be one of "
                + ", ".join(item.value for item in self.supported_modes),
                mode.value,
            )
        return self.resolver(document, mode)

    def combination_status(
        self,
        selection: ExtensionSelection,
    ) -> Optional[ValidatedCombination]:
        """Return explicit validation metadata for an extension combination."""

        for combination in self.combinations:
            if combination.selection == selection:
                return combination
        return None
