from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class MappingSettings:
    scale_fractions: tuple[float, ...]
    scale_fraction_pairs: tuple[tuple[float, float], ...] | None
    range_placement: str
    calibration_examples: int
    calibration_batch_size: int
    logit_gain_min: float
    logit_gain_max: float
    logit_gain_steps: int

@dataclass(frozen=True)
class UpdateBackendSettings:
    type: str
    parameters: Mapping[str, Any]

@dataclass(frozen=True)
class StudentTrainSettings:
    num_epochs: int
    learning_rates: tuple[float, float]
    temperature: float
    log_every: int
    max_batches: int | None
    max_validation_batches: int | None
    minimum_relative_kl_improvement: float
    selection_evaluation: str
    selection_metric: str
    selection_noise_repeats: int
    weight_modifier: UpdateBackendSettings
    selection_weight_modifier: UpdateBackendSettings
    update_backend: UpdateBackendSettings

@dataclass(frozen=True)
class StudentValidateSettings:
    split: str
    sample_limit: int | None
    weight_modifier: UpdateBackendSettings
    noise_repeats: int
