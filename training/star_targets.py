"""Strict, pickle-free class-conditional state targets for STAR recovery.

The target artifact is deliberately architecture-neutral.  Architecture
adapters provide settled states under one of the named representations below;
this module accumulates their class means, quantizes them, and binds the result
to the exact healthy deployment and calibration cohort that produced it.

Fault masks and post-fault states do not belong in this artifact.  A recovery
runtime must load it with an explicit :class:`StarTargetBinding`, so a target
recorded from another checkpoint, array, cohort, representation, or gain fails
before any learning update is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from experiments.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    content_hash,
    sha256_file,
)


STAR_TARGET_SCHEMA = "ebl.star.state_targets"
STAR_TARGET_SCHEMA_VERSION = 1
CROSSBAR_STATE_REPRESENTATION = (
    "crossbar.post_relu_hidden_and_output_logits.v1"
)
STAR_STORAGE_DTYPES = frozenset({"fp32", "fp16", "int8"})

_LIFECYCLE = "healthy_pre_fault_calibration"
_DEFAULT_CLASS_COUNT = 10
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_BASE_NPZ_FIELDS = {
    "schema",
    "schema_version",
    "metadata_json",
    "semantic_sha256",
    "class_labels",
    "class_counts",
}
_RECEIPT_FIELDS = {
    "schema",
    "schema_version",
    "artifact",
    "artifact_sha256",
    "metadata_sha256",
    "semantic_sha256",
    "tensor_hashes",
}
_METADATA_FIELDS = {
    "lifecycle",
    "architecture_id",
    "state_representation_id",
    "num_classes",
    "source",
    "calibration",
    "components",
    "accumulation",
    "storage",
}
_SOURCE_FIELDS = {
    "model_checkpoint_sha256",
    "healthy_deployment_artifact_sha256",
    "healthy_persistent_state_sha256",
    "healthy_forward_state_sha256",
    "hardware_instance_id",
    "config_sha256",
}
_CALIBRATION_FIELDS = {
    "dataset_id",
    "split",
    "examples",
    "ordered_sample_ids_sha256",
    "model_inputs_sha256",
    "labels_sha256",
    "cohort_sha256",
}
_ACCUMULATION_METADATA = {
    "dtype": "float64",
    "device": "cpu",
    "order": "stream_encounter_order",
}


class StarTargetError(ValueError):
    """Raised when a STAR target or its binding fails closed."""


@dataclass(frozen=True)
class StarStateComponent:
    """One named state vector in a supported STAR representation."""

    key: str
    role: str
    width: int
    units: str
    source_layer_index: int


_REPRESENTATIONS: Mapping[str, tuple[StarStateComponent, ...]] = MappingProxyType(
    {
        CROSSBAR_STATE_REPRESENTATION: (
            StarStateComponent(
                key="hidden_post_relu",
                role="hidden",
                width=50,
                units="model_native_activation",
                source_layer_index=0,
            ),
            StarStateComponent(
                key="output_logits",
                role="output",
                width=10,
                units="model_native_logit",
                source_layer_index=1,
            ),
        ),

    }
)


def state_components(
    state_representation_id: str,
) -> tuple[StarStateComponent, ...]:
    """Return the immutable component layout for a named representation."""

    if not isinstance(state_representation_id, str):
        raise StarTargetError(
            "Expected a supported STAR state representation. "
            f"Provided value: {state_representation_id!r}."
        )
    try:
        return _REPRESENTATIONS[state_representation_id]
    except KeyError as error:
        raise StarTargetError(
            "Expected a supported STAR state representation. "
            f"Provided value: {state_representation_id!r}."
        ) from error


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise StarTargetError(
            f"Expected {label} to be a lowercase SHA-256 digest. "
            f"Provided value: {value!r}."
        )
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise StarTargetError(
            f"Expected {label} to be a non-empty lowercase identifier. "
            f"Provided value: {value!r}."
        )
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StarTargetError(
            f"Expected {label} to be a non-empty string. Provided value: {value!r}."
        )
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StarTargetError(
            f"Expected {label} to be a positive integer. Provided value: {value!r}."
        )
    return value


@dataclass(frozen=True)
class StarSourceBinding:
    """Hashes that identify the exact healthy deployed model and array."""

    model_checkpoint_sha256: str
    healthy_deployment_artifact_sha256: str
    healthy_persistent_state_sha256: str
    healthy_forward_state_sha256: str
    hardware_instance_id: str
    config_sha256: str

    def __post_init__(self) -> None:
        for name in _SOURCE_FIELDS:
            _digest(getattr(self, name), label=f"source.{name}")


def _cohort_payload(
    *,
    dataset_id: str,
    split: str,
    examples: int,
    ordered_sample_ids_sha256: str,
    model_inputs_sha256: str,
    labels_sha256: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "split": split,
        "examples": examples,
        "ordered_sample_ids_sha256": ordered_sample_ids_sha256,
        "model_inputs_sha256": model_inputs_sha256,
        "labels_sha256": labels_sha256,
    }


@dataclass(frozen=True)
class StarCalibrationBinding:
    """Identity of the ordered labeled cohort used to record state means."""

    dataset_id: str
    split: str
    examples: int
    ordered_sample_ids_sha256: str
    model_inputs_sha256: str
    labels_sha256: str
    cohort_sha256: str

    def __post_init__(self) -> None:
        _nonempty_string(self.dataset_id, label="calibration.dataset_id")
        _nonempty_string(self.split, label="calibration.split")
        _positive_integer(self.examples, label="calibration.examples")
        for name in (
            "ordered_sample_ids_sha256",
            "model_inputs_sha256",
            "labels_sha256",
            "cohort_sha256",
        ):
            _digest(getattr(self, name), label=f"calibration.{name}")
        expected = content_hash(
            _cohort_payload(
                dataset_id=self.dataset_id,
                split=self.split,
                examples=self.examples,
                ordered_sample_ids_sha256=self.ordered_sample_ids_sha256,
                model_inputs_sha256=self.model_inputs_sha256,
                labels_sha256=self.labels_sha256,
            )
        )
        if self.cohort_sha256 != expected:
            raise StarTargetError(
                "Expected calibration.cohort_sha256 to match the exact ordered "
                "sample, input, and label hashes."
            )


@dataclass(frozen=True)
class StarTargetBinding:
    """Complete semantic binding required to create or load STAR targets."""

    architecture_id: str
    state_representation_id: str
    source: StarSourceBinding
    calibration: StarCalibrationBinding
    component_gains: tuple[float, ...]
    storage_dtype: str = "fp32"
    class_count: int = _DEFAULT_CLASS_COUNT

    def __post_init__(self) -> None:
        _identifier(self.architecture_id, label="architecture_id")
        components = state_components(self.state_representation_id)
        _positive_integer(self.class_count, label="class_count")
        try:
            gains = tuple(float(value) for value in self.component_gains)
        except (TypeError, ValueError) as error:
            raise StarTargetError(
                "Expected one numeric STAR gain per state component."
            ) from error
        if (
            len(gains) != len(components)
            or any(not math.isfinite(value) or value < 0.0 for value in gains)
            or not any(value > 0.0 for value in gains)
        ):
            raise StarTargetError(
                "Expected one finite non-negative STAR gain per state component "
                "and at least one positive gain. "
                f"Provided value: {self.component_gains!r}."
            )
        if (
            not isinstance(self.storage_dtype, str)
            or self.storage_dtype not in STAR_STORAGE_DTYPES
        ):
            raise StarTargetError(
                f"Expected storage_dtype in {sorted(STAR_STORAGE_DTYPES)!r}. "
                f"Provided value: {self.storage_dtype!r}."
            )
        if not isinstance(self.source, StarSourceBinding) or not isinstance(
            self.calibration, StarCalibrationBinding
        ):
            raise StarTargetError(
                "Expected typed STAR source and calibration bindings."
            )
        object.__setattr__(self, "component_gains", gains)


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU bytes."""

    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_tensor(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise StarTargetError(
            f"Expected {label} to be a non-empty tensor. Provided value: {value!r}."
        )
    tensor = value.detach().cpu().contiguous()
    if tensor.shape[0] < 1 or tensor.is_complex() or (
        tensor.is_floating_point()
        and not bool(torch.all(torch.isfinite(tensor)))
    ):
        raise StarTargetError(
            f"Expected {label} to contain finite examples."
        )
    return tensor


def build_calibration_binding(
    *,
    dataset_id: str,
    split: str,
    ordered_sample_ids: torch.Tensor,
    model_inputs: torch.Tensor,
    labels: torch.Tensor,
) -> StarCalibrationBinding:
    """Hash an exact ordered cohort independently of its minibatch partition."""

    sample_ids = _finite_tensor(
        ordered_sample_ids, label="ordered calibration sample IDs"
    )
    inputs = _finite_tensor(model_inputs, label="calibration model inputs")
    targets = _finite_tensor(labels, label="calibration labels")
    if (
        sample_ids.ndim != 1
        or sample_ids.dtype != torch.int64
        or targets.ndim != 1
        or targets.dtype != torch.int64
        or len(set(sample_ids.tolist())) != sample_ids.numel()
        or inputs.shape[0] != sample_ids.numel()
        or targets.shape[0] != sample_ids.numel()
    ):
        raise StarTargetError(
            "Expected unique int64 sample IDs, int64 labels, and model inputs "
            "with the same leading example dimension."
        )
    payload = _cohort_payload(
        dataset_id=_nonempty_string(dataset_id, label="calibration.dataset_id"),
        split=_nonempty_string(split, label="calibration.split"),
        examples=int(sample_ids.numel()),
        ordered_sample_ids_sha256=tensor_sha256(sample_ids),
        model_inputs_sha256=tensor_sha256(inputs),
        labels_sha256=tensor_sha256(targets),
    )
    return StarCalibrationBinding(
        **payload,
        cohort_sha256=content_hash(payload),
    )


@dataclass(frozen=True)
class StarTargetBundle:
    """Validated encoded targets and their decoded FP32 recovery view."""

    binding: StarTargetBinding
    class_labels: torch.Tensor
    class_counts: torch.Tensor
    stored_means: Mapping[str, torch.Tensor]
    decoded_means: Mapping[str, torch.Tensor]
    storage: Mapping[str, Any]
    semantic_sha256: str

    def means(self, component: str) -> torch.Tensor:
        """Return a detached FP32 class-by-state target matrix."""

        try:
            return self.decoded_means[component].detach().clone()
        except KeyError as error:
            raise StarTargetError(
                f"Expected a declared STAR component. Provided value: {component!r}."
            ) from error

    @property
    def stored_value_count(self) -> int:
        return sum(int(value.numel()) for value in self.stored_means.values())

    @property
    def stored_mean_nbytes(self) -> int:
        return sum(
            int(value.numel() * value.element_size())
            for value in self.stored_means.values()
        )


@dataclass(frozen=True)
class StarTargetArtifactRecord:
    artifact_path: Path
    receipt_path: Path
    artifact_sha256: str
    receipt_sha256: str
    semantic_sha256: str


def _component_metadata(binding: StarTargetBinding) -> list[dict[str, Any]]:
    return [
        {
            "key": component.key,
            "role": component.role,
            "width": component.width,
            "units": component.units,
            "source_layer_index": component.source_layer_index,
            "star_loss_gain": gain,
        }
        for component, gain in zip(
            state_components(binding.state_representation_id),
            binding.component_gains,
            strict=True,
        )
    ]


def _source_metadata(source: StarSourceBinding) -> dict[str, str]:
    return {name: getattr(source, name) for name in sorted(_SOURCE_FIELDS)}


def _calibration_metadata(
    calibration: StarCalibrationBinding,
) -> dict[str, Any]:
    return {name: getattr(calibration, name) for name in sorted(_CALIBRATION_FIELDS)}


def _metadata(
    binding: StarTargetBinding,
    storage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "lifecycle": _LIFECYCLE,
        "architecture_id": binding.architecture_id,
        "state_representation_id": binding.state_representation_id,
        "num_classes": binding.class_count,
        "source": _source_metadata(binding.source),
        "calibration": _calibration_metadata(binding.calibration),
        "components": _component_metadata(binding),
        "accumulation": dict(_ACCUMULATION_METADATA),
        "storage": dict(storage),
    }


def _tensor_descriptors(
    class_labels: torch.Tensor,
    class_counts: torch.Tensor,
    stored_means: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    tensors = {
        "class_labels": class_labels,
        "class_counts": class_counts,
        **{
            f"mean__{key}": value
            for key, value in stored_means.items()
        },
    }
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": tensor_sha256(value),
        }
        for name, value in sorted(tensors.items())
    }


def _semantic_sha256(
    metadata: Mapping[str, Any],
    class_labels: torch.Tensor,
    class_counts: torch.Tensor,
    stored_means: Mapping[str, torch.Tensor],
) -> str:
    return content_hash(
        {
            "metadata": dict(metadata),
            "tensors": _tensor_descriptors(
                class_labels, class_counts, stored_means
            ),
        }
    )


def _quantize(
    means: Mapping[str, torch.Tensor],
    *,
    storage_dtype: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    if storage_dtype == "fp32":
        stored = {key: value.to(torch.float32) for key, value in means.items()}
        storage: dict[str, Any] = {
            "format": "fp32",
            "stored_dtype": "float32",
            "decode_dtype": "float32",
        }
    elif storage_dtype == "fp16":
        stored = {key: value.to(torch.float16) for key, value in means.items()}
        storage = {
            "format": "fp16",
            "stored_dtype": "float16",
            "decode_dtype": "float32",
        }
    elif storage_dtype == "int8":
        stored = {}
        scales: dict[str, float] = {}
        for key, value in means.items():
            maximum = float(value.abs().max().item())
            scale = maximum / 127.0 if maximum > 0.0 else 1.0
            normalized = value / scale
            rounded = torch.sign(normalized) * torch.floor(
                torch.abs(normalized) + 0.5
            )
            stored[key] = rounded.clamp(-127.0, 127.0).to(torch.int8)
            scales[key] = scale
        storage = {
            "format": "int8",
            "stored_dtype": "int8",
            "decode_dtype": "float32",
            "scheme": "symmetric_per_component",
            "zero_point": 0,
            "qmin": -127,
            "qmax": 127,
            "rounding": "half_away_from_zero",
            "scales": scales,
        }
    else:  # guarded by StarTargetBinding, retained for direct internal calls
        raise StarTargetError(f"Unsupported STAR storage dtype: {storage_dtype!r}.")

    if storage_dtype == "int8":
        decoded = {
            key: value.to(torch.float32) * float(storage["scales"][key])
            for key, value in stored.items()
        }
    else:
        decoded = {key: value.to(torch.float32) for key, value in stored.items()}
    return stored, decoded, storage


class StarTargetAccumulator:
    """Stream state examples into deterministic CPU-FP64 class means."""

    def __init__(
        self,
        state_representation_id: str | None = None,
        *,
        class_count: int = _DEFAULT_CLASS_COUNT,
        representation_id: str | None = None,
        component_specs: tuple[StarStateComponent, ...] | None = None,
    ) -> None:
        _positive_integer(class_count, label="class_count")
        if state_representation_id is not None and representation_id is not None:
            raise StarTargetError(
                "Expected only one of state_representation_id and representation_id."
            )
        resolved_representation = representation_id or state_representation_id
        if resolved_representation is None:
            raise StarTargetError("Expected a named STAR state representation.")
        declared = state_components(resolved_representation)
        if component_specs is not None and tuple(component_specs) != declared:
            raise StarTargetError(
                "Expected component_specs to match the named STAR representation exactly."
            )
        self.state_representation_id = resolved_representation
        self.components = declared
        self.class_count = class_count
        self._sums = {
            component.key: torch.zeros(
                (class_count, component.width), dtype=torch.float64
            )
            for component in self.components
        }
        self._counts = torch.zeros((class_count,), dtype=torch.int64)
        self._labels: list[torch.Tensor] = []
        self._inputs: list[torch.Tensor] = []
        self._sample_ids: list[torch.Tensor] = []

    @property
    def examples(self) -> int:
        return int(self._counts.sum().item())

    def observe(
        self,
        labels: torch.Tensor,
        states: Mapping[str, torch.Tensor],
        *,
        inputs: torch.Tensor,
        sample_ids: torch.Tensor,
    ) -> None:
        """Observe one ordered batch without changing summation order."""

        if not isinstance(labels, torch.Tensor):
            raise StarTargetError("Expected STAR labels to be an int64 tensor.")
        targets = labels.detach().cpu().contiguous()
        expected_keys = tuple(component.key for component in self.components)
        if (
            targets.ndim != 1
            or targets.dtype != torch.int64
            or targets.numel() < 1
            or not isinstance(states, Mapping)
            or set(states) != set(expected_keys)
        ):
            raise StarTargetError(
                "Expected one int64 label vector and exactly the declared STAR "
                f"state components {expected_keys!r}."
            )
        if bool(torch.any((targets < 0) | (targets >= self.class_count))):
            raise StarTargetError(
                f"Expected STAR labels in [0, {self.class_count - 1}]."
            )

        observed_inputs = _finite_tensor(inputs, label="observed STAR inputs")
        if observed_inputs.shape[0] != targets.numel():
            raise StarTargetError(
                "Expected observed STAR inputs to match the label batch size."
            )
        self._inputs.append(observed_inputs.clone())
        observed_ids = _finite_tensor(
            sample_ids, label="observed STAR sample IDs"
        )
        if (
            observed_ids.ndim != 1
            or observed_ids.dtype != torch.int64
            or observed_ids.shape != targets.shape
        ):
            raise StarTargetError(
                "Expected observed STAR sample IDs to be an int64 label-sized vector."
            )
        self._sample_ids.append(observed_ids.clone())

        staged: dict[str, torch.Tensor] = {}
        for component in self.components:
            value = states[component.key]
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape != (targets.numel(), component.width)
                or not value.is_floating_point()
            ):
                raise StarTargetError(
                    f"Expected state {component.key!r} to have floating shape "
                    f"({targets.numel()}, {component.width})."
                )
            cpu = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
            if not bool(torch.all(torch.isfinite(cpu))):
                raise StarTargetError(
                    f"Expected state {component.key!r} to contain finite values."
                )
            staged[component.key] = cpu

        # Deliberately process rows in encounter order.  This makes the result
        # invariant to how an identical ordered cohort is partitioned into
        # minibatches, while retaining float64 accumulation.
        for row in range(targets.numel()):
            label = int(targets[row].item())
            self._counts[label] += 1
            for component in self.components:
                self._sums[component.key][label].add_(
                    staged[component.key][row]
                )
        self._labels.append(targets.clone())

    def finalize(
        self,
        binding: StarTargetBinding,
        quantization: str | None = None,
    ) -> StarTargetBundle:
        """Quantize complete class means under an exact source binding."""

        if quantization is not None:
            if quantization not in STAR_STORAGE_DTYPES:
                raise StarTargetError(
                    f"Expected quantization in {sorted(STAR_STORAGE_DTYPES)!r}. "
                    f"Provided value: {quantization!r}."
                )
            binding = replace(binding, storage_dtype=quantization)
        if binding.state_representation_id != self.state_representation_id:
            raise StarTargetError(
                "Expected the STAR accumulator and binding representations to match."
            )
        if binding.class_count != self.class_count:
            raise StarTargetError(
                "Expected the STAR accumulator and binding class counts to match."
            )
        if self.examples != binding.calibration.examples:
            raise StarTargetError(
                "Expected observed STAR examples to match the calibration binding. "
                f"Provided value: observed={self.examples}, "
                f"bound={binding.calibration.examples}."
            )
        if bool(torch.any(self._counts <= 0)):
            raise StarTargetError(
                "Expected at least one calibration example for every class."
            )
        observed_labels = torch.cat(self._labels).to(torch.int64)
        if tensor_sha256(observed_labels) != binding.calibration.labels_sha256:
            raise StarTargetError(
                "Expected observed STAR labels to match the bound ordered cohort."
            )
        if tensor_sha256(torch.cat(self._inputs, dim=0)) != (
            binding.calibration.model_inputs_sha256
        ):
            raise StarTargetError(
                "Expected observed STAR inputs to match the bound ordered cohort."
            )
        if tensor_sha256(torch.cat(self._sample_ids)) != (
            binding.calibration.ordered_sample_ids_sha256
        ):
            raise StarTargetError(
                "Expected observed STAR sample IDs to match the bound ordered cohort."
            )
        means = {
            key: value / self._counts.to(torch.float64).unsqueeze(1)
            for key, value in self._sums.items()
        }
        stored, decoded, storage = _quantize(
            means, storage_dtype=binding.storage_dtype
        )
        labels = torch.arange(self.class_count, dtype=torch.int64)
        counts = self._counts.clone()
        metadata = _metadata(binding, storage)
        semantic = _semantic_sha256(metadata, labels, counts, stored)
        return StarTargetBundle(
            binding=binding,
            class_labels=labels,
            class_counts=counts,
            stored_means=MappingProxyType(stored),
            decoded_means=MappingProxyType(decoded),
            storage=MappingProxyType(storage),
            semantic_sha256=semantic,
        )


def _validate_bundle(bundle: StarTargetBundle) -> dict[str, Any]:
    if not isinstance(bundle, StarTargetBundle):
        raise StarTargetError(
            f"Expected a StarTargetBundle. Provided value: {bundle!r}."
        )
    components = state_components(bundle.binding.state_representation_id)
    class_count = bundle.binding.class_count
    expected_keys = tuple(component.key for component in components)
    if (
        bundle.class_labels.dtype != torch.int64
        or bundle.class_labels.shape != (class_count,)
        or not torch.equal(
            bundle.class_labels.detach().cpu(),
            torch.arange(class_count, dtype=torch.int64),
        )
        or bundle.class_counts.dtype != torch.int64
        or bundle.class_counts.shape != (class_count,)
        or bool(torch.any(bundle.class_counts <= 0))
        or int(bundle.class_counts.sum().item())
        != bundle.binding.calibration.examples
        or set(bundle.stored_means) != set(expected_keys)
        or set(bundle.decoded_means) != set(expected_keys)
    ):
        raise StarTargetError("Expected a complete internally consistent STAR bundle.")
    expected_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "int8": torch.int8,
    }[bundle.binding.storage_dtype]
    for component in components:
        stored = bundle.stored_means[component.key]
        decoded = bundle.decoded_means[component.key]
        shape = (class_count, component.width)
        if (
            not isinstance(stored, torch.Tensor)
            or stored.dtype != expected_dtype
            or stored.shape != shape
            or not isinstance(decoded, torch.Tensor)
            or decoded.dtype != torch.float32
            or decoded.shape != shape
            or not bool(torch.all(torch.isfinite(decoded)))
        ):
            raise StarTargetError(
                f"Expected valid stored and decoded STAR means for {component.key!r}."
            )
        if stored.is_floating_point() and not bool(torch.all(torch.isfinite(stored))):
            raise StarTargetError(
                f"Expected finite stored STAR means for {component.key!r}."
            )
        if stored.dtype == torch.int8 and bool(torch.any(stored == -128)):
            raise StarTargetError("Expected symmetric int8 STAR codes in [-127, 127].")
    metadata = _metadata(bundle.binding, bundle.storage)
    expected_semantic = _semantic_sha256(
        metadata,
        bundle.class_labels,
        bundle.class_counts,
        bundle.stored_means,
    )
    if bundle.semantic_sha256 != expected_semantic:
        raise StarTargetError("Expected the STAR bundle semantic hash to match.")
    return metadata


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_star_targets(
    path: Path | str,
    bundle: StarTargetBundle,
    *,
    receipt_path: Path | str | None = None,
) -> StarTargetArtifactRecord:
    """Atomically save one target NPZ and its strict JSON receipt."""

    destination = Path(path).expanduser().resolve()
    if destination.suffix != ".npz":
        raise StarTargetError("Expected a STAR target artifact path ending in '.npz'.")
    receipt = (
        destination.with_suffix(".receipt.json")
        if receipt_path is None
        else Path(receipt_path).expanduser().resolve()
    )
    if receipt == destination or receipt.suffix != ".json":
        raise StarTargetError("Expected a distinct JSON STAR receipt path.")
    metadata = _validate_bundle(bundle)
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(STAR_TARGET_SCHEMA),
        "schema_version": np.asarray(STAR_TARGET_SCHEMA_VERSION, dtype=np.int64),
        "metadata_json": np.asarray(
            canonical_json_bytes(metadata).decode("utf-8")
        ),
        "semantic_sha256": np.asarray(bundle.semantic_sha256),
        "class_labels": bundle.class_labels.detach().cpu().numpy(),
        "class_counts": bundle.class_counts.detach().cpu().numpy(),
    }
    for key, value in bundle.stored_means.items():
        arrays[f"mean__{key}"] = value.detach().cpu().contiguous().numpy()
    _atomic_write_npz(destination, arrays)
    tensor_hashes = _tensor_descriptors(
        bundle.class_labels, bundle.class_counts, bundle.stored_means
    )
    receipt_value = {
        "schema": STAR_TARGET_SCHEMA,
        "schema_version": STAR_TARGET_SCHEMA_VERSION,
        "artifact": destination.name,
        "artifact_sha256": sha256_file(destination),
        "metadata_sha256": content_hash(metadata),
        "semantic_sha256": bundle.semantic_sha256,
        "tensor_hashes": tensor_hashes,
    }
    atomic_write_json(receipt, receipt_value)
    return StarTargetArtifactRecord(
        artifact_path=destination,
        receipt_path=receipt,
        artifact_sha256=receipt_value["artifact_sha256"],
        receipt_sha256=sha256_file(receipt),
        semantic_sha256=bundle.semantic_sha256,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    duplicates = []
    for key, value in pairs:
        if key in result:
            duplicates.append(key)
        result[key] = value
    if duplicates:
        raise StarTargetError(
            f"Expected unique JSON object keys. Duplicate keys: {duplicates!r}."
        )
    return result


def _reject_constant(value: str) -> Any:
    raise StarTargetError(
        f"Expected strict finite JSON numbers. Provided value: {value!r}."
    )


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except StarTargetError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise StarTargetError(f"Expected a readable strict {label} JSON object.") from error
    if not isinstance(value, dict):
        raise StarTargetError(f"Expected {label} to contain one JSON object.")
    return value


def _npz_scalar(payload: Mapping[str, np.ndarray], name: str) -> Any:
    value = payload[name]
    if value.shape != () or value.dtype.hasobject:
        raise StarTargetError(
            f"Expected scalar non-object STAR field {name!r}."
        )
    return value.item()


def _binding_from_metadata(metadata: Mapping[str, Any]) -> StarTargetBinding:
    if set(metadata) != _METADATA_FIELDS:
        raise StarTargetError("Expected exact STAR metadata fields.")
    if (
        metadata["lifecycle"] != _LIFECYCLE
        or isinstance(metadata["num_classes"], bool)
        or not isinstance(metadata["num_classes"], int)
        or metadata["num_classes"] < 1
        or metadata["accumulation"] != _ACCUMULATION_METADATA
    ):
        raise StarTargetError("Expected the STAR healthy pre-fault FP64 lifecycle.")
    raw_source = metadata["source"]
    raw_calibration = metadata["calibration"]
    if not isinstance(raw_source, dict) or set(raw_source) != _SOURCE_FIELDS:
        raise StarTargetError("Expected exact STAR source-binding fields.")
    if (
        not isinstance(raw_calibration, dict)
        or set(raw_calibration) != _CALIBRATION_FIELDS
    ):
        raise StarTargetError("Expected exact STAR calibration-binding fields.")
    source = StarSourceBinding(**raw_source)
    calibration = StarCalibrationBinding(**raw_calibration)
    representation = metadata["state_representation_id"]
    components = state_components(representation)
    raw_components = metadata["components"]
    if not isinstance(raw_components, list) or len(raw_components) != len(components):
        raise StarTargetError("Expected all STAR component descriptors.")
    gains = []
    for expected, raw in zip(components, raw_components, strict=True):
        expected_fields = {
            "key",
            "role",
            "width",
            "units",
            "source_layer_index",
            "star_loss_gain",
        }
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise StarTargetError("Expected exact STAR component fields.")
        structural = {
            "key": expected.key,
            "role": expected.role,
            "width": expected.width,
            "units": expected.units,
            "source_layer_index": expected.source_layer_index,
        }
        if {name: raw[name] for name in structural} != structural:
            raise StarTargetError(
                "Expected STAR component descriptors to match their representation."
            )
        gains.append(raw["star_loss_gain"])
    storage = metadata["storage"]
    if not isinstance(storage, dict):
        raise StarTargetError("Expected STAR storage metadata to be an object.")
    return StarTargetBinding(
        architecture_id=metadata["architecture_id"],
        state_representation_id=representation,
        source=source,
        calibration=calibration,
        component_gains=tuple(gains),
        storage_dtype=storage.get("format"),
        class_count=metadata["num_classes"],
    )


def _validate_storage(
    storage: Mapping[str, Any],
    *,
    binding: StarTargetBinding,
    stored: Mapping[str, torch.Tensor],
) -> None:
    if binding.storage_dtype == "fp32":
        expected = {
            "format": "fp32",
            "stored_dtype": "float32",
            "decode_dtype": "float32",
        }
        if dict(storage) != expected:
            raise StarTargetError("Expected exact FP32 STAR storage metadata.")
    elif binding.storage_dtype == "fp16":
        expected = {
            "format": "fp16",
            "stored_dtype": "float16",
            "decode_dtype": "float32",
        }
        if dict(storage) != expected:
            raise StarTargetError("Expected exact FP16 STAR storage metadata.")
    else:
        expected_keys = {
            "format",
            "stored_dtype",
            "decode_dtype",
            "scheme",
            "zero_point",
            "qmin",
            "qmax",
            "rounding",
            "scales",
        }
        scales = storage.get("scales")
        component_keys = {item.key for item in state_components(binding.state_representation_id)}
        if (
            set(storage) != expected_keys
            or storage.get("format") != "int8"
            or storage.get("stored_dtype") != "int8"
            or storage.get("decode_dtype") != "float32"
            or storage.get("scheme") != "symmetric_per_component"
            or storage.get("zero_point") != 0
            or storage.get("qmin") != -127
            or storage.get("qmax") != 127
            or storage.get("rounding") != "half_away_from_zero"
            or not isinstance(scales, dict)
            or set(scales) != component_keys
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in scales.values()
            )
            or any(bool(torch.any(value == -128)) for value in stored.values())
        ):
            raise StarTargetError("Expected exact symmetric INT8 STAR storage metadata.")


def load_star_targets(
    path: Path | str,
    receipt_path: Path | str | None = None,
    expected: StarTargetBinding | None = None,
) -> StarTargetBundle:
    """Load and independently validate targets against an exact binding."""

    if not isinstance(expected, StarTargetBinding):
        raise StarTargetError("Expected an explicit StarTargetBinding when loading.")
    source = Path(path).expanduser().resolve()
    receipt_source = (
        source.with_suffix(".receipt.json")
        if receipt_path is None
        else Path(receipt_path).expanduser().resolve()
    )
    receipt = _json_object(receipt_source, label="STAR receipt")
    try:
        artifact_sha256 = sha256_file(source)
    except OSError as error:
        raise StarTargetError("Expected a readable STAR target artifact.") from error
    if (
        set(receipt) != _RECEIPT_FIELDS
        or receipt.get("schema") != STAR_TARGET_SCHEMA
        or receipt.get("schema_version") != STAR_TARGET_SCHEMA_VERSION
        or receipt.get("artifact") != source.name
        or receipt.get("artifact_sha256") != artifact_sha256
    ):
        raise StarTargetError("Expected the STAR receipt to match its artifact exactly.")

    components = state_components(expected.state_representation_id)
    expected_mean_fields = {f"mean__{component.key}" for component in components}
    try:
        with np.load(source, allow_pickle=False) as raw:
            if len(raw.files) != len(set(raw.files)) or set(raw.files) != (
                _BASE_NPZ_FIELDS | expected_mean_fields
            ):
                raise StarTargetError("Expected exact unique STAR NPZ fields.")
            payload = {name: raw[name].copy() for name in raw.files}
    except StarTargetError:
        raise
    except (OSError, ValueError) as error:
        raise StarTargetError("Expected a readable pickle-free STAR target NPZ.") from error
    if (
        str(_npz_scalar(payload, "schema")) != STAR_TARGET_SCHEMA
        or payload["schema_version"].dtype != np.dtype(np.int64)
        or int(_npz_scalar(payload, "schema_version"))
        != STAR_TARGET_SCHEMA_VERSION
    ):
        raise StarTargetError("Expected STAR target schema version 1.")
    metadata_text = str(_npz_scalar(payload, "metadata_json"))
    try:
        metadata = json.loads(
            metadata_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except StarTargetError:
        raise
    except json.JSONDecodeError as error:
        raise StarTargetError("Expected canonical strict STAR metadata JSON.") from error
    if (
        not isinstance(metadata, dict)
        or canonical_json_bytes(metadata).decode("utf-8") != metadata_text
    ):
        raise StarTargetError("Expected canonical STAR metadata JSON encoding.")
    loaded_binding = _binding_from_metadata(metadata)
    if loaded_binding != expected:
        raise StarTargetError(
            "Expected the STAR target architecture, representation, gains, "
            "source, cohort, and storage binding to match exactly."
        )

    labels_array = payload["class_labels"]
    counts_array = payload["class_counts"]
    class_count = expected.class_count
    if (
        labels_array.dtype != np.dtype(np.int64)
        or labels_array.shape != (class_count,)
        or not np.array_equal(
            labels_array, np.arange(class_count, dtype=np.int64)
        )
        or counts_array.dtype != np.dtype(np.int64)
        or counts_array.shape != (class_count,)
        or bool(np.any(counts_array <= 0))
        or int(counts_array.sum()) != expected.calibration.examples
    ):
        raise StarTargetError("Expected complete ordered STAR labels and class counts.")
    class_labels = torch.from_numpy(labels_array.copy())
    class_counts = torch.from_numpy(counts_array.copy())
    expected_dtype = {
        "fp32": np.dtype(np.float32),
        "fp16": np.dtype(np.float16),
        "int8": np.dtype(np.int8),
    }[expected.storage_dtype]
    stored: dict[str, torch.Tensor] = {}
    for component in components:
        value = payload[f"mean__{component.key}"]
        if (
            value.dtype != expected_dtype
            or value.shape != (class_count, component.width)
            or (np.issubdtype(value.dtype, np.floating) and not bool(np.all(np.isfinite(value))))
        ):
            raise StarTargetError(
                f"Expected valid stored STAR mean {component.key!r}."
            )
        stored[component.key] = torch.from_numpy(np.ascontiguousarray(value))
    storage = metadata["storage"]
    _validate_storage(storage, binding=expected, stored=stored)
    if expected.storage_dtype == "int8":
        means = {
            key: value.to(torch.float32) * float(storage["scales"][key])
            for key, value in stored.items()
        }
    else:
        means = {key: value.to(torch.float32) for key, value in stored.items()}
    descriptors = _tensor_descriptors(class_labels, class_counts, stored)
    semantic = _semantic_sha256(metadata, class_labels, class_counts, stored)
    if (
        not isinstance(receipt.get("tensor_hashes"), dict)
        or receipt["tensor_hashes"] != descriptors
        or receipt.get("metadata_sha256") != content_hash(metadata)
        or receipt.get("semantic_sha256") != semantic
        or str(_npz_scalar(payload, "semantic_sha256")) != semantic
    ):
        raise StarTargetError("Expected all STAR receipt and semantic hashes to match.")
    bundle = StarTargetBundle(
        binding=expected,
        class_labels=class_labels,
        class_counts=class_counts,
        stored_means=MappingProxyType(stored),
        decoded_means=MappingProxyType(means),
        storage=MappingProxyType(dict(storage)),
        semantic_sha256=semantic,
    )
    _validate_bundle(bundle)
    return bundle


__all__ = [
    "CROSSBAR_STATE_REPRESENTATION",
    "STAR_STORAGE_DTYPES",
    "STAR_TARGET_SCHEMA",
    "STAR_TARGET_SCHEMA_VERSION",
    "StarCalibrationBinding",
    "StarSourceBinding",
    "StarStateComponent",
    "StarTargetAccumulator",
    "StarTargetArtifactRecord",
    "StarTargetBinding",
    "StarTargetBundle",
    "StarTargetError",
    "build_calibration_binding",
    "load_star_targets",
    "save_star_targets",
    "state_components",
    "tensor_sha256",
]
