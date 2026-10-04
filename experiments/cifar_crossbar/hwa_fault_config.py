"""Matched HWA/CDT development and fresh-array PCM recovery comparison."""

from dataclasses import asdict, dataclass, fields
import math

from experiments.cifar_crossbar.fault_config import (
    FaultSpec,
    parse_config as parse_fault,
)
from experiments.schema import RunMode

EXPERIMENT_ID = "cifar_pcm_hwa_comparison.v1"


@dataclass(frozen=True)
class ComparisonSpec(FaultSpec):
    experiment_id: str = EXPERIMENT_ID
    stage: str = "fit"
    array_seed: int = 81001
    endpoint_seed: int = 91001
    hwa_epochs: int = 30
    extension_epochs: int = 30
    hwa_batch_size: int = 256
    validation_images: int = 1000
    hwa_learning_rates: tuple = (0.0001, 0.0003, 0.001)
    hwa_sgd_learning_rates: tuple = (0.005, 0.05, 0.5)
    hwa_objectives: tuple = ("teacher_kl", "cross_entropy")
    hwa_noise_scales: tuple = (1.0, 2.0, 5.0)
    cdt_rates: tuple = (0.001, 0.003, 0.01)
    recovery_learning_rates: tuple = (0.00001, 0.0001, 0.001)
    calibration_learning_rates: tuple = (0.0001, 0.0003, 0.001)
    validation_every: int = 5
    development_arrays: int = 3
    tuning_arrays: int = 2


def parse_config(payload):
    expected = {f.name for f in fields(ComparisonSpec)}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Expected exactly the versioned HWA comparison fields.")
    values = dict(payload)
    for name in (
        "fault_kinds",
        "fault_rates_ppm",
        "methods",
        "hwa_learning_rates",
        "hwa_sgd_learning_rates",
        "hwa_objectives",
        "hwa_noise_scales",
        "cdt_rates",
        "recovery_learning_rates",
        "calibration_learning_rates",
    ):
        if not isinstance(values[name], (list, tuple)):
            raise ValueError(f"Expected a sequence for {name}.")
        values[name] = tuple(values[name])
    spec = ComparisonSpec(**values)
    if spec.experiment_id != EXPERIMENT_ID or spec.stage not in (
        "cache",
        "fit",
        "tune",
        "screen",
    ):
        raise ValueError(
            "Expected a native HWA comparison cache/fit/tune/screen stage."
        )
    inherited = {f.name: getattr(spec, f.name) for f in fields(FaultSpec)}
    inherited.update(experiment_id="cifar_pcm_fault_recovery.v1", stage="screen")
    parse_fault(inherited)
    for name in (
        "hwa_epochs",
        "extension_epochs",
        "hwa_batch_size",
        "validation_images",
        "validation_every",
        "development_arrays",
        "tuning_arrays",
    ):
        if type(getattr(spec, name)) is not int or getattr(spec, name) < 1:
            raise ValueError(f"Expected positive integer {name}.")
    classes = 10 if spec.dataset == "cifar10" else 100
    if (
        spec.validation_images > 5000
        or 5000 % spec.validation_images
        or spec.validation_images % classes
    ):
        raise ValueError("Expected a balanced development subset dividing 5000.")
    if spec.fault_kinds != ("open", "gmax", "random") or spec.fault_rates_ppm != (
        0,
        100,
        1000,
        10000,
    ):
        raise ValueError("Expected the complete declared physical-failure screen.")
    for name in (
        "hwa_learning_rates",
        "hwa_sgd_learning_rates",
        "hwa_noise_scales",
        "cdt_rates",
        "recovery_learning_rates",
        "calibration_learning_rates",
    ):
        seq = getattr(spec, name)
        if (
            not seq
            or len(set(seq)) != len(seq)
            or any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                or x <= 0
                for x in seq
            )
        ):
            raise ValueError(f"Expected unique finite positive {name}.")
    if max(spec.cdt_rates) > 1 or spec.hwa_noise_scales[0] != 1:
        raise ValueError(
            "Expected probability-valued CDT rates and nominal noise first."
        )
    if (
        not spec.hwa_objectives
        or len(set(spec.hwa_objectives)) != len(spec.hwa_objectives)
        or not set(spec.hwa_objectives)
        <= {
            "teacher_kl",
            "cross_entropy",
        }
    ):
        raise ValueError("Expected CE and/or teacher-KL HWA objectives.")
    if spec.array_seed < 81000 or spec.endpoint_seed < 91000:
        raise ValueError(
            "Expected fresh final-array seed namespaces, separate from development and the old screen."
        )
    return spec


def resolve_spec(document, mode):
    if mode != RunMode.TRAIN:
        raise ValueError("Expected native train mode.")
    return document


def default_config(**changes):
    return asdict(parse_config({**asdict(ComparisonSpec()), **changes}))
