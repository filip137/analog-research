from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from core.components import ComponentSettings


@dataclass(frozen=True)
class DataSettings:
    dataset: str
    batch_size: int
    num_points: Optional[int]
    validation_points: Optional[int]
    shuffle: bool

@dataclass(frozen=True)
class NonLinearitySettings:
    type: str
    quadratic_diode_param: Mapping[str, Union[int, float]]
    exponential_diode_param: Mapping[str, Union[int, float]]
    hard_sigmoid_param: Mapping[str, Union[int, float]]
    iv_data_path: Optional[str]

@dataclass(frozen=True)
class ModelSettings:
    dims: Tuple[int, ...]
    input_gain: float
    weight_gains: Tuple[float, ...]
    weight_min: float
    weight_max: float
    weight_init_mode: str
    include_biases: bool
    voltage_amp: float
    current_amp: float
    non_linearity: NonLinearitySettings
    adapter: ComponentSettings

@dataclass(frozen=True)
class UpdaterSettings:
    double_diode: Optional[str]
    single_diode: Optional[str]

@dataclass(frozen=True)
class ToleranceSettings:
    relative: float
    voltage: float
    residual_current: Optional[float]

@dataclass(frozen=True)
class PolishSettings:
    enabled: bool
    dynamic: bool
    max_newton_iterations: int
    z_threshold: float
    exponential_clip: float

@dataclass(frozen=True)
class AndersonSettings:
    memory: int
    omega: float
    tolerance_floor: float
    regularization: float

@dataclass(frozen=True)
class OverrelaxationSettings:
    factor: float
    reject_steps: bool
    reject_max_tries: int
    reject_shrink: float
    reject_epsilon: float

@dataclass(frozen=True)
class ExperimentalExponentialSettings:
    damping: float
    newton_max_steps: int
    progressive_tolerance: bool
    tolerance_start: float
    tolerance_end: float
    tolerance_switch_high: float
    tolerance_switch_low: float

@dataclass(frozen=True)
class SolverSettings:
    inference_iterations: int
    training_iterations: int
    minimizer_impl: str
    minimizer_mode: str
    adaptive_equilibrium: bool
    random_initialization: bool
    updaters: UpdaterSettings
    tolerances: ToleranceSettings
    polish: PolishSettings
    anderson: AndersonSettings
    overrelaxation: OverrelaxationSettings
    experimental_exponential: ExperimentalExponentialSettings
