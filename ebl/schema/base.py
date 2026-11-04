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


from core.config import ConfigError


class RunMode(str, Enum):
    """Execution modes shared by the public CLI and experiment definitions."""

    TRAIN = "train"
    LINSPACE = "linspace"
    VALIDATE = "validate"
    CHARACTERIZE = "characterize"


from core.config import config_error


from core.config import freeze_json


from core.config import to_plain_data


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
