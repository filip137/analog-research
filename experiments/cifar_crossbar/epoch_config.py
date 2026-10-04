"""Separate contract for extending fixed-array PCM recovery to many passes."""

from dataclasses import asdict, dataclass, fields

from experiments.cifar_crossbar.fault_config import FaultSpec, parse_config as parse_fault
from experiments.schema import RunMode

EXPERIMENT_ID = "cifar_pcm_recovery_epochs.v1"


@dataclass(frozen=True)
class EpochSpec(FaultSpec):
    experiment_id: str = EXPERIMENT_ID
    epochs: int = 20
    stage: str = "screen"
    array_seed: int = 101001
    endpoint_seed: int = 111001
    fault_rates_ppm: tuple = (0, 10000)
    learning_rate: float = 0.0001
    calibration_lr: float = 0.0003
    milestones: tuple = (1, 2, 5, 10, 20)
    schedules: tuple = ("constant", "step_decay")
    decay_after: tuple = (5, 10)
    decay_factor: float = 0.3
    tuning_arrays: int = 2
    validation_images: int = 1000


def parse_config(payload):
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(EpochSpec)}:
        raise ValueError("Expected exactly the versioned recovery-epochs fields.")
    values = dict(payload)
    for key in ("fault_kinds", "fault_rates_ppm", "methods", "milestones", "schedules", "decay_after"):
        if not isinstance(values[key], (tuple, list)):
            raise ValueError(f"Expected sequence for {key}.")
        values[key] = tuple(values[key])
    spec = EpochSpec(**values)
    inherited = {f.name: getattr(spec, f.name) for f in fields(FaultSpec)}
    inherited.update(experiment_id="cifar_pcm_fault_recovery.v1", stage="screen", epochs=1)
    parse_fault(inherited)
    if spec.experiment_id != EXPERIMENT_ID or spec.stage not in ("tune", "screen"):
        raise ValueError("Expected recovery-epochs tune/screen stage.")
    if type(spec.epochs) is not int or spec.epochs < 1:
        raise ValueError("Expected positive epochs.")
    if (not spec.milestones or tuple(sorted(set(spec.milestones))) != spec.milestones
            or spec.milestones[0] != 1 or spec.milestones[-1] != spec.epochs
            or any(type(e) is not int or e < 1 for e in spec.milestones)):
        raise ValueError("Expected increasing milestones from 1 to epochs.")
    if spec.schedules != ("constant", "step_decay"):
        raise ValueError("Expected paired constant and step-decay development candidates.")
    if (tuple(sorted(set(spec.decay_after))) != spec.decay_after
            or any(type(e) is not int or e < 1 for e in spec.decay_after)
            or type(spec.decay_factor) not in (float, int) or not 0 < spec.decay_factor < 1):
        raise ValueError("Expected positive decay epochs and factor in (0,1).")
    if type(spec.tuning_arrays) is not int or spec.tuning_arrays < 1:
        raise ValueError("Expected positive tuning arrays.")
    if spec.validation_images != 1000 or spec.recovery_images != 5000:
        raise ValueError("Expected the existing 5000/1000 recovery/development cohorts.")
    if spec.array_seed not in (101001, 101002, 101003) or spec.endpoint_seed != spec.array_seed + 10000:
        raise ValueError("Expected matched fresh confirmation-array seed namespaces.")
    if spec.noise_scale != 1 or spec.recovery_model != "gaussian_endpoint_reprogramming":
        raise ValueError("Expected unchanged nominal Gaussian endpoint recovery.")
    return spec


def resolve_spec(document, mode):
    if mode != RunMode.TRAIN:
        raise ValueError("Expected native train mode.")
    return document


def default_config(**changes):
    return asdict(parse_config({**asdict(EpochSpec()), **changes}))
