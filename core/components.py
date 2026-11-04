from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class ComponentSettings:
    """A named extension and its recursively immutable parameter object."""

    type: str
    parameters: Mapping[str, Any]
