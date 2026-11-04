from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class StudentModelSettings:
    dims: tuple[int, int, int]
    input_gain: float
    conductance_min: float
    conductance_max: float
    voltage_amp: float
    current_amp: float
    encoding: str
    include_biases: bool
    non_linearity: str
    quadratic_diode_param: Mapping[str, Any]
    exponential_diode_param: Mapping[str, Any]
    hard_sigmoid_param: Mapping[str, Any]
    amplification_indexing: str = "logical"

@dataclass(frozen=True)
class SolverSettings:
    inference_iterations: int
    training_iterations: int
    mode: str
    overrelaxation_factor: float

@dataclass(frozen=True)
class TeacherSettings:
    type: str
    initialization: str
    conductance_bounds: tuple[float, float] | None
