"""Versioned open-loop OM comparison, independent of the capped V2 sweep."""
from dataclasses import dataclass, asdict, fields
import math
from experiments.schema import RunMode
from experiments.cifar_crossbar.sweep_config import KINDS, RATES

EXPERIMENT_ID = "cifar_om_open_loop.v1"
METHODS = ("none", "calibration", "onchip_weights", "onchip_calibration")


@dataclass(frozen=True)
class OmOpenLoopSpec:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    stage: str = "screen"
    dataset: str = "cifar10"
    suffix: str = "last_two_blocks"
    backend: str = "om"
    initialization: str = "saved_p0"
    programming_rule: str = "inherited_v2_p0"
    recovery_model: str = "om_open_loop_uncapped_adam"
    tile_size: int = 512
    device: str = "cpu"
    disable_cudnn: bool = False
    cpu_threads: int = 1
    seed: int = 41
    data_seed: int = 20260917
    array_seed: int = 251001
    endpoint_seed: int = 261001
    fault_kinds: tuple = KINDS
    fault_rates_ppm: tuple = RATES
    methods: tuple = METHODS
    epochs: int = 5
    recovery_images: int = 45000
    validation_images: int = 1000
    batch_size: int = 64
    learning_rate: float = .0001
    calibration_lr: float = .0003
    recovery_pulses: None = None
    update_pulses: int = 1
    programming_pulse_cap: None = None
    max_examples: int = 0


def parse_config(payload):
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(OmOpenLoopSpec)}:
        raise ValueError("Expected exactly the open-loop OM v1 fields.")
    values = dict(payload)
    for key in ("fault_kinds", "fault_rates_ppm", "methods"):
        if not isinstance(values[key], (tuple, list)):
            raise ValueError(key)
        values[key] = tuple(values[key])
    s = OmOpenLoopSpec(**values)
    default = OmOpenLoopSpec()
    for field in fields(s):
        value, expected = getattr(s, field.name), getattr(default, field.name)
        if type(expected) is int and (type(value) is not int or value < 0):
            raise ValueError(field.name)
        if type(expected) is bool and type(value) is not bool:
            raise ValueError(field.name)
        if type(expected) is float and (type(value) not in (float, int) or not math.isfinite(value)):
            raise ValueError(field.name)
    frozen = ("experiment_id", "schema_version", "stage", "backend", "recovery_model", "tile_size",
              "seed", "data_seed", "fault_kinds", "methods", "epochs", "recovery_images",
              "validation_images", "batch_size", "learning_rate", "recovery_pulses", "update_pulses",
              "programming_pulse_cap")
    if any(getattr(s, key) != getattr(default, key) for key in frozen):
        raise ValueError("Frozen open-loop law, coverage, or budget changed.")
    if s.dataset not in ("cifar10", "cifar100") or s.suffix not in ("last_two_blocks", "last_four_blocks"):
        raise ValueError("Dataset/suffix.")
    rules = {"saved_p0": "inherited_v2_p0", "reset_open_loop": "nominal_soft_bounds_inverse"}
    if rules.get(s.initialization) != s.programming_rule:
        raise ValueError("Initialization/programming rule mismatch.")
    if s.calibration_lr != (.0003 if s.dataset == "cifar10" else .001):
        raise ValueError("Calibration learning rate.")
    if s.max_examples not in (0, 128) or s.fault_rates_ppm != ((0, 50000) if s.max_examples else RATES):
        raise ValueError("Canary/full coverage.")
    if s.array_seed not in (251001, 251002, 251003) or s.endpoint_seed != s.array_seed + 10000:
        raise ValueError("Expected paired V2 final-array identities.")
    if s.cpu_threads < 1 or not (s.device == "cpu" or s.device == "cuda" or s.device.startswith("cuda:")):
        raise ValueError("Execution settings.")
    return s


def default_config(**changes):
    return asdict(parse_config({**asdict(OmOpenLoopSpec()), **changes}))


def resolve_spec(document, mode):
    if mode != RunMode.TRAIN:
        raise ValueError("Expected native train mode.")
    return document
