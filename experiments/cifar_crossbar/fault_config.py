"""Declared exploratory PCM failure screen and one-epoch recovery contract."""

from dataclasses import asdict, dataclass, fields
import math
from experiments.schema import RunMode

EXPERIMENT_ID = "cifar_pcm_fault_recovery.v1"
METHODS = ("none", "calibration", "rewrite", "onchip_weights", "onchip_calibration")


@dataclass(frozen=True)
class FaultSpec:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    stage: str = "screen"
    dataset: str = "cifar10"
    suffix: str = "last_two_blocks"
    tile_size: int = 512
    device: str = "cpu"
    disable_cudnn: bool = False
    cpu_threads: int = 4
    seed: int = 41
    data_seed: int = 20260917
    array_seed: int = 51001
    endpoint_seed: int = 61001
    fault_kinds: tuple = ("open", "gmax", "random")
    fault_rates_ppm: tuple = (0, 100, 1000, 10000)
    methods: tuple = METHODS
    noise_scale: float = 1.0
    epochs: int = 1
    recovery_images: int = 5000
    batch_size: int = 64
    learning_rate: float = 0.001
    calibration_lr: float = 0.001
    recovery_model: str = "gaussian_endpoint_reprogramming"
    max_examples: int = 0


def parse_config(payload):
    expected = {f.name for f in fields(FaultSpec)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(
            f"Expected exactly the PCM failure fields {sorted(expected)}; got {payload}."
        )
    values = dict(payload)
    for key in ("fault_kinds", "fault_rates_ppm", "methods"):
        if not isinstance(values[key], (list, tuple)):
            raise ValueError(f"Expected a sequence for {key}.")
        values[key] = tuple(values[key])
    spec = FaultSpec(**values)
    if spec.experiment_id != EXPERIMENT_ID or spec.schema_version != 1:
        raise ValueError("Expected cifar_pcm_fault_recovery.v1 schema version 1.")
    enums = {
        "stage": ("cache", "screen"),
        "dataset": ("cifar10", "cifar100"),
        "suffix": ("last_two_blocks", "last_block", "head"),
        "recovery_model": (
            "gaussian_endpoint_reprogramming",
            "ideal_update_upper_bound",
        ),
    }
    for key, choices in enums.items():
        if getattr(spec, key) not in choices:
            raise ValueError(f"Expected {key} in {choices}.")
    for field in fields(FaultSpec):
        value, default = getattr(spec, field.name), getattr(FaultSpec(), field.name)
        if type(default) is int and (type(value) is not int or value < 0):
            raise ValueError(f"Expected a nonnegative integer for {field.name}.")
        if type(default) is bool and type(value) is not bool:
            raise ValueError(f"Expected a boolean for {field.name}.")
        if type(default) is float and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Expected a finite nonnegative value for {field.name}.")
    if (
        not spec.fault_kinds
        or len(set(spec.fault_kinds)) != len(spec.fault_kinds)
        or not set(spec.fault_kinds) <= {"open", "gmax", "random"}
    ):
        raise ValueError("Expected unique open/gmax/random failure kinds.")
    if (
        not spec.fault_rates_ppm
        or tuple(sorted(set(spec.fault_rates_ppm))) != spec.fault_rates_ppm
        or spec.fault_rates_ppm[0] != 0
        or any(
            type(p) is not int or not 0 <= p <= 1000000 for p in spec.fault_rates_ppm
        )
    ):
        raise ValueError(
            "Expected increasing integer ppm rates beginning at zero, at most 1000000."
        )
    if spec.methods != METHODS:
        raise ValueError(f"Expected all paired controls in order: {METHODS}.")
    if spec.epochs != 1 or spec.recovery_images not in (500, 5000):
        raise ValueError("Expected exactly one epoch on 500 or 5000 recovery images.")
    if min(spec.cpu_threads, spec.tile_size, spec.batch_size) < 1:
        raise ValueError("Expected positive thread, tile and batch sizes.")
    if spec.stage == "cache" and spec.max_examples:
        raise ValueError(
            "Expected complete cache preparation; truncate only a declared screen smoke."
        )
    if not isinstance(spec.device, str) or not (
        spec.device == "cpu"
        or spec.device == "cuda"
        or (spec.device.startswith("cuda:") and spec.device[5:].isdigit())
    ):
        raise ValueError("Expected cpu, cuda or cuda:N device.")
    return spec


def resolve_spec(document, mode):
    if mode != RunMode.TRAIN:
        raise ValueError("Expected native train mode for the PCM failure study.")
    return document


def default_config(**changes):
    return asdict(parse_config({**asdict(FaultSpec()), **changes}))
