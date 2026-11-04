from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class Deployment:
    assignment_seed: int=271002
    fault_kind: str='random'
    fault_rate: float=0.03
