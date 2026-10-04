"""Strict, path-free scientific settings for native CIFAR suffix runs."""

from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import math
from collections.abc import Mapping
from experiments.schema import RunMode, config_error

EXPERIMENT_ID = "cifar_resnet_suffix_recovery.v1"


@dataclass(frozen=True)
class Spec:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    stage: str = "audit"
    dataset: str = "cifar10"
    suffix: str = "last_two_blocks"
    backend: str = "om"
    policy: str = "repaired"
    device: str = "cpu"
    seed: int = 41
    data_seed: int = 20260917
    assignment_seed: int = 1001
    endpoint_seed: int = 2001
    tile_size: int = 512
    noise_scale: float = 1.0
    variation_scale: float = 1.0
    fault_rate: float = 0.0
    fault_kind: str = "low"
    read_noise: float = 0.0
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 0.001
    calibration_lr: float = 0.001
    objective: str = "teacher_kl"
    method: str = "closed_loop"
    recovery_images: int = 500
    validation_images: int = 5000
    evaluation: str = "development"
    max_examples: int = 0
    program_pulses: int = 128
    recovery_pulses: int = 640
    verify_step_fraction: float = 0.5
    kernel_bins: int = 41
    kernel_samples: int = 512
    kernel_seed: int = 3001
    ramp_epochs: int = 10
    retention_seconds: float = 1.0
    drift_compensation: bool = True
    disable_cudnn: bool = False


def parse_config(payload):
    if not isinstance(payload, Mapping):
        raise config_error("config", "to be an object", payload)
    expected = {f.name for f in fields(Spec)}
    if set(payload) != expected:
        raise config_error(
            "config fields", f"to be exactly {sorted(expected)}", sorted(payload)
        )
    value = Spec(**payload)
    if (
        value.experiment_id != EXPERIMENT_ID
        or type(value.schema_version) is not int
        or value.schema_version != 1
    ):
        raise config_error("experiment/schema", f"to be {EXPERIMENT_ID}/1", payload)
    enums = {
        "stage": ("audit", "characterize", "hwa", "deploy", "recover", "pcm_reference"),
        "dataset": ("cifar10", "cifar100"),
        "suffix": ("head", "last_block", "last_two_blocks"),
        "backend": ("om", "pcm", "pcm_inference"),
        "policy": ("native", "repaired"),
        "objective": ("teacher_kl", "cross_entropy"),
        "method": ("none", "calibration", "rewrite", "open_loop", "closed_loop"),
        "evaluation": ("development", "confirmation"),
        "fault_kind": ("low", "high", "middle"),
    }
    for key, choices in enums.items():
        if getattr(value, key) not in choices:
            raise config_error(key, f"to be one of {choices}", getattr(value, key))
    for f in fields(Spec):
        x, default = getattr(value, f.name), getattr(Spec(), f.name)
        if type(default) is bool and type(x) is not bool:
            raise config_error(f.name, "to be boolean", x)
        if type(default) is int and (type(x) is not int or x < 0):
            raise config_error(f.name, "to be a nonnegative integer", x)
        if type(default) is float and (
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x < 0
        ):
            raise config_error(f.name, "to be finite and nonnegative", x)
    for key in (
        "tile_size",
        "batch_size",
        "recovery_images",
        "validation_images",
        "program_pulses",
        "recovery_pulses",
        "kernel_samples",
    ):
        if getattr(value, key) < 1:
            raise config_error(key, "to be positive", getattr(value, key))
    if (
        value.kernel_bins < 3
        or value.verify_step_fraction <= 0
        or value.variation_scale <= 0
        or value.fault_rate > 1
    ):
        raise config_error(
            "device settings",
            "to have valid bins, scales and fault probability",
            payload,
        )
    if value.recovery_images not in (500, 5000) or value.validation_images != 5000:
        raise config_error(
            "data budgets",
            "to use 500/5000 recovery and 5000 development images",
            payload,
        )
    if value.stage == "recover" and value.objective != "teacher_kl":
        raise config_error("recovery objective", "to be teacher_kl", value.objective)
    if value.stage == "recover" and value.epochs > 30:
        raise config_error("recovery epochs", "to be at most 30", value.epochs)
    if value.program_pulses > 128 or value.recovery_pulses > 640:
        raise config_error(
            "write budgets",
            "to stay within 128 verify pulses/update and 640 recovery writes/cell",
            payload,
        )
    if not isinstance(value.device, str) or not (
        value.device == "cpu"
        or value.device == "cuda"
        or (value.device.startswith("cuda:") and value.device[5:].isdigit())
    ):
        raise config_error("device", "to be cpu, cuda, or cuda:N", value.device)
    if (value.stage == "pcm_reference" and value.backend != "pcm_inference") or (
        value.backend == "pcm_inference" and value.stage not in ("hwa", "pcm_reference")
    ):
        raise config_error(
            "PCM inference", "to use only its separate HWA/reference stages", payload
        )
    if value.stage in ("hwa", "characterize") and value.evaluation != "development":
        raise config_error(
            "training/characterization",
            "to use development evaluation only",
            value.evaluation,
        )
    if value.evaluation == "confirmation" and value.max_examples:
        raise config_error(
            "confirmation", "to evaluate the full test set", value.max_examples
        )
    return value


def resolve_spec(document, mode):
    if mode != RunMode.TRAIN:
        raise ValueError(
            "Expected the public train mode for the staged CIFAR experiment."
        )
    return document


def default_config(**changes):
    values = asdict(Spec())
    values.update(changes)
    parse_config(values)
    return values
