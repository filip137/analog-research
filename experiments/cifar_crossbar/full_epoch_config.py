"""Five full adaptation epochs; separate from historical 5000-image passes."""

from dataclasses import asdict, dataclass, fields

from experiments.cifar_crossbar.fault_config import FaultSpec, parse_config as parse_fault
from experiments.schema import RunMode

EXPERIMENT_ID = "cifar_crossbar_full_epochs.v1"


@dataclass(frozen=True)
class FullEpochSpec(FaultSpec):
    experiment_id: str = EXPERIMENT_ID
    stage: str = "screen"
    backend: str = "pcm"
    epochs: int = 5
    recovery_images: int = 45000
    validation_images: int = 1000
    array_seed: int = 151001
    endpoint_seed: int = 161001
    fault_rates_ppm: tuple = (0, 10000)
    learning_rate: float = 0.0001
    calibration_lr: float = 0.0003
    hwa_epochs: int = 30
    hwa_learning_rate: float = 0.0001
    program_pulses: int = 128
    recovery_pulses: int = 640
    update_pulses: int = 1
    verify_step_fraction: float = 0.5
    kernel_bins: int = 41
    kernel_samples: int = 512


def parse_config(payload):
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(FullEpochSpec)}:
        raise ValueError("Expected exactly the full-epoch v1 fields.")
    values = dict(payload)
    for key in ("fault_kinds", "fault_rates_ppm", "methods"):
        values[key] = tuple(values[key])
    spec = FullEpochSpec(**values)
    old = {f.name: getattr(spec, f.name) for f in fields(FaultSpec)}
    old.update(experiment_id="cifar_pcm_fault_recovery.v1", stage="screen", epochs=1,
               recovery_images=5000, recovery_model="gaussian_endpoint_reprogramming")
    parse_fault(old)
    if spec.experiment_id != EXPERIMENT_ID or spec.stage not in ("cache", "sources", "screen"):
        raise ValueError("Expected full-epoch cache/sources/screen.")
    if spec.backend not in ("pcm", "om"):
        raise ValueError("Expected PCM endpoint or OM pulse backend.")
    expected_model = "gaussian_endpoint_reprogramming" if spec.backend == "pcm" else "om_closed_loop_pulse_adam"
    if spec.recovery_model != expected_model:
        raise ValueError("Recovery law does not match the declared device.")
    for key in ("epochs", "hwa_epochs", "program_pulses", "recovery_pulses", "update_pulses", "kernel_bins", "kernel_samples"):
        if type(getattr(spec, key)) is not int or getattr(spec, key) < 1:
            raise ValueError(f"Expected positive {key}.")
    if spec.epochs != 5 or spec.recovery_images != 45000 or spec.validation_images != 1000:
        raise ValueError("Expected five full epochs over the 45000-image adaptation split.")
    if spec.program_pulses != 128 or spec.recovery_pulses != 640 or spec.update_pulses != 1 or spec.verify_step_fraction != .5:
        raise ValueError("Expected frozen 128-P0/1-per-update/640-total OM pulse budgets.")
    if spec.kernel_bins != 41 or spec.kernel_samples != 512 or spec.hwa_learning_rate != .0001:
        raise ValueError("Expected the predeclared OM HWA recipe.")
    if spec.hwa_epochs != (1 if spec.max_examples else 30):
        raise ValueError("Expected 30 HWA epochs (one in the labelled canary).")
    if spec.array_seed not in (151001, 151002, 151003) or spec.endpoint_seed != spec.array_seed + 10000:
        raise ValueError("Expected fresh final arrays, separate from prior studies.")
    if spec.noise_scale != 1:
        raise ValueError("Expected nominal device noise.")
    return spec


def resolve_spec(document, mode):
    if mode != RunMode.TRAIN:
        raise ValueError("Expected native train mode.")
    return document


def default_config(**changes):
    return asdict(parse_config({**asdict(FullEpochSpec()), **changes}))
