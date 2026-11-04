from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class Training:
    arm: str='std3'
    epochs: int=5
    max_updates: int=0
    effective_batch: int=32
    microbatch: int=1
    conductance_lr: float=0.0003
    calibration_lr: float=0.0003
    digital_lr: float=0.00001
    recipe: str='A'
    warmup_updates: int=245
    schedule: str='cosine'
    minimum_lr_factor: float=0.2
    seed: int=41
    max_wall_seconds: int=864000
    checkpoint_every: int=512
    retention: str='best_latest_final'

@dataclass(frozen=True)
class Selection:
    every_updates: int=512
    assignment_seeds: tuple=(211001,211002)
    fault_kinds: tuple=('none','random')
    random_rate: float=0.03

@dataclass(frozen=True)
class Recovery:
    arm: str='learning'
    updates: int=2048
    learning_rate: float=0.003
    threshold_sigma: float=2.0
    max_write_attempts: int | None=4
    calibration_updates: int=200
    calibration_lr: float=0.0001
    checkpoint_every: int=128
    write_noise_factor: float=1.0
    development_every: int=0
    retain_final_optimizer: bool=True
    conductance_floor: str='none'
    update_rule: str='thresholded_adam'
    target_bounds: str='none'
    lr_schedule: str='constant'
    warmup_updates: int=0
    extends_updates: int=0
