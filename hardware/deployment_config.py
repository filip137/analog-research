from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class Deployment:
    assignment_seed: int=271002
    fault_kind: str='random'
    fault_rate: float=0.03

    @property
    def mapping_scale(self):
        return 1.0

    @property
    def mapped(self):
        return False


@dataclass(frozen=True)
class MappedDeployment(Deployment):
    """Deployment with an explicit ratio remap G' = min(k G, G_max) before programming.

    A subclass keeps the resolved identity of every legacy deployment (no ``mapping_scale``
    key) unchanged. ``mapping_scale`` 1 programs the unmapped targets bit for bit.
    """
    mapping_scale: float=1.0

    @property
    def mapped(self):
        return True


@dataclass(frozen=True)
class SourceEvaluationDeployment(Deployment):
    source_evaluation: str = 'cmo_24h_x1_x4'
