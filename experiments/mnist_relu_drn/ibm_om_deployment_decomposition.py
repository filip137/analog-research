"""Read-only decomposition of saved IBM OM MNIST deployment endpoints.

This module never constructs an IBM modifier.  It restores the selected named
weights, validates and replays the tensors already present in the deployment
sidecar, and evaluates reversible counterfactual state compositions on the
same ordered validation loader.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn.functional as F

from experiments.artifacts import atomic_write_json, sha256_file


SCHEMA = "ebl.mnist_relu_drn.ibm_om_deployment_decomposition"
SCHEMA_VERSION = 1
ENDPOINT_APPLICATION_POLICY = (
    "aihwkit_apparent_forward_persistent_update_state"
)
_PULSE_SCHEMA = "ebl.ibm_reram.om_pulse_resolved_deployment"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TENSOR_FIELDS = (
    "global_requested_target",
    "requested_target",
    "raw_apparent_endpoint",
    "apparent_endpoint",
    "persistent_endpoint",
    "accepted",
)


@dataclass(frozen=True)
class LayerGeometry:
    """One logical layer and its flattened physical binding slices."""

    index: int
    name: str
    encoding: str
    rail_layout: str
    binding_keys: tuple[str, ...]
    binding_shapes: tuple[tuple[int, int], ...]
    binding_slices: tuple[slice, ...]

    @property
    def device_count(self) -> int:
        return sum(
            (section.stop or 0) - (section.start or 0)
            for section in self.binding_slices
        )


@dataclass(frozen=True)
class EvaluationTrace:
    """Numerical outputs retained in memory for cross-state comparisons."""

    metrics: dict[str, Any]
    student_logits: torch.Tensor
    raw_scores: torch.Tensor
    teacher_logits: torch.Tensor
    labels: torch.Tensor
    cohort_sha256: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose one saved IBM OM MNIST deployment without programming "
            "or resampling any device."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--deployment-config",
        type=Path,
        default=None,
        help=(
            "optional strict validate config for a held-out deployment; "
            "--config remains the checkpoint-producing train config"
        ),
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Expected {label} to name an existing file. Provided value: "
            f"{str(resolved)!r}."
        )
    return resolved


def _strict_torch_load(path: Path) -> Any:
    source = _require_file(path, label="--deployment")
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported PyTorch
        return torch.load(source, map_location="cpu")


def load_deployment_artifact(path: Path) -> Any:
    """Load a tensor-only deployment bundle through the strict reader."""

    return _strict_torch_load(path)


def _json_value(value: Any) -> Any:
    """Return a strict JSON-compatible copy without accepting tensors."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Expected report values not to contain NaN or infinity.")
        return value
    raise TypeError(
        "Expected deployment report metadata to be strict JSON data. "
        f"Provided value: {type(value).__name__}."
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(repr(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _commissioning_tensor_sha256(value: torch.Tensor) -> str:
    """Match the commissioning artifact's canonical tensor digest."""

    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(
        torch.tensor(tuple(tensor.shape), dtype=torch.int64).numpy().tobytes()
    )
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _summary(value: torch.Tensor) -> dict[str, float | int | None]:
    flattened = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flattened.numel() == 0:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
            "mae": None,
            "rms": None,
            "p50_absolute": None,
            "p95_absolute": None,
            "maximum_absolute": None,
        }
    absolute = flattened.abs()
    return {
        "count": int(flattened.numel()),
        "minimum": float(flattened.min().item()),
        "maximum": float(flattened.max().item()),
        "mean": float(flattened.mean().item()),
        "standard_deviation": float(
            flattened.std(unbiased=False).item()
        ),
        "mae": float(absolute.mean().item()),
        "rms": float(flattened.square().mean().sqrt().item()),
        "p50_absolute": float(torch.quantile(absolute, 0.5).item()),
        "p95_absolute": float(torch.quantile(absolute, 0.95).item()),
        "maximum_absolute": float(absolute.max().item()),
    }


def _scaled_residual_summary(
    value: torch.Tensor,
    *,
    conductance_span: float,
) -> dict[str, Any]:
    return {
        "normalized_coordinate": _summary(value),
        "conductance_s": _summary(value * conductance_span),
    }


def _validate_flat_tensor(
    value: Any,
    *,
    name: str,
    size: int,
    dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.shape != (size,)
        or (dtype is not None and value.dtype != dtype)
    ):
        raise ValueError(
            f"Expected deployment {name!r} to be a CPU "
            f"{dtype or 'tensor'} vector of length {size}. Provided value: "
            f"shape={getattr(value, 'shape', None)!r}, "
            f"dtype={getattr(value, 'dtype', None)!r}, "
            f"device={getattr(value, 'device', None)!r}."
        )
    if value.is_floating_point() and not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"Expected deployment {name!r} to be finite.")
    return value.detach().clone()


def _binding_layout(bindings: Sequence[Any]) -> tuple[
    tuple[str, ...],
    tuple[tuple[int, int], ...],
    tuple[slice, ...],
]:
    selected = tuple(bindings)
    if not selected:
        raise ValueError("Expected a non-empty selected-weight catalog.")
    keys = tuple(binding.key for binding in selected)
    if len(set(keys)) != len(keys):
        raise ValueError("Expected unique selected-weight binding keys.")
    shapes = []
    slices = []
    offset = 0
    for binding in selected:
        shape = tuple(binding.state.shape)
        if len(shape) != 2 or any(item < 1 for item in shape):
            raise ValueError(
                "Expected IBM OM decomposition only for non-empty rank-2 "
                f"bindings. Provided value: key={binding.key!r}, shape={shape!r}."
            )
        count = int(binding.state.numel())
        shapes.append((int(shape[0]), int(shape[1])))
        slices.append(slice(offset, offset + count))
        offset += count
    return keys, tuple(shapes), tuple(slices)


def canonical_layer_geometries(
    binding_keys: Sequence[str],
    binding_shapes: Sequence[tuple[int, int]],
    binding_slices: Sequence[slice],
    *,
    encoding: str,
    dual_rail_layout_by_parameter: Mapping[str, str] | None,
) -> tuple[LayerGeometry, ...]:
    """Resolve the two canonical MNIST layers without regrouping devices."""

    keys = tuple(binding_keys)
    shapes = tuple(tuple(shape) for shape in binding_shapes)
    slices = tuple(binding_slices)
    if len(keys) != len(shapes) or len(keys) != len(slices):
        raise ValueError("Expected matching binding keys, shapes, and slices.")
    if encoding == "single":
        expected = ("base.dense_weight.0", "base.dense_weight.1")
        layouts = dict(dual_rail_layout_by_parameter or {})
        if keys != expected or layouts != {
            expected[0]: "halves",
            expected[1]: "paired",
        }:
            raise ValueError(
                "Expected the canonical single/quad catalog and W1-halves, "
                f"W2-paired layouts. Provided keys={keys!r}, layouts={layouts!r}."
            )
        return tuple(
            LayerGeometry(
                index=index,
                name=f"W{index + 1}",
                encoding=encoding,
                rail_layout=layouts[key],
                binding_keys=(key,),
                binding_shapes=(shapes[index],),
                binding_slices=(slices[index],),
            )
            for index, key in enumerate(keys)
        )
    if encoding != "differential":
        raise ValueError(
            "Expected model encoding to be 'single' or 'differential'. "
            f"Provided value: {encoding!r}."
        )
    if dual_rail_layout_by_parameter is not None:
        raise ValueError(
            "Expected differential-pair mapping to use null explicit layouts."
        )
    expected = (
        "base.conductance_plus.0",
        "base.conductance_minus.0",
        "base.conductance_plus.1",
        "base.conductance_minus.1",
    )
    if keys != expected or shapes[0] != shapes[1] or shapes[2] != shapes[3]:
        raise ValueError(
            "Expected the canonical adjacent shape-matched differential-pair "
            f"catalog. Provided keys={keys!r}, shapes={shapes!r}."
        )
    return (
        LayerGeometry(
            index=0,
            name="W1",
            encoding=encoding,
            rail_layout="halves",
            binding_keys=keys[:2],
            binding_shapes=shapes[:2],
            binding_slices=slices[:2],
        ),
        LayerGeometry(
            index=1,
            name="W2",
            encoding=encoding,
            rail_layout="paired",
            binding_keys=keys[2:],
            binding_shapes=shapes[2:],
            binding_slices=slices[2:],
        ),
    )


def _layer_device_mask(
    layer: LayerGeometry,
    *,
    size: int,
) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    for section in layer.binding_slices:
        mask[section] = True
    return mask


def logical_differential_contrast(
    normalized_state: torch.Tensor,
    layer: LayerGeometry,
    *,
    conductance_min: float,
    conductance_max: float,
) -> torch.Tensor:
    """Collapse four- or eight-cell rails into one signed logical contrast."""

    span = conductance_max - conductance_min
    matrices = [
        conductance_min
        + span
        * normalized_state[section].reshape(shape).to(torch.float64)
        for section, shape in zip(
            layer.binding_slices,
            layer.binding_shapes,
        )
    ]
    effective = matrices[0] if layer.encoding == "single" else matrices[0] - matrices[1]
    rows, columns = effective.shape
    if rows % 2 or columns % 2:
        raise ValueError(
            "Expected every logical contrast layer to be even-by-even. "
            f"Provided value: layer={layer.name!r}, shape={(rows, columns)!r}."
        )
    logical_rows = rows // 2
    logical_columns = columns // 2
    plus_columns = (
        torch.arange(logical_columns)
        if layer.rail_layout == "halves"
        else torch.arange(logical_columns) * 2
    )
    minus_columns = (
        plus_columns + logical_columns
        if layer.rail_layout == "halves"
        else plus_columns + 1
    )
    plus_rows = slice(0, logical_rows)
    minus_rows = slice(logical_rows, rows)
    return (
        effective[plus_rows][:, plus_columns]
        - effective[plus_rows][:, minus_columns]
        - effective[minus_rows][:, plus_columns]
        + effective[minus_rows][:, minus_columns]
    )


def _snr(signal: torch.Tensor, error: torch.Tensor) -> dict[str, Any]:
    signal_rms = float(signal.to(torch.float64).square().mean().sqrt().item())
    error_rms = float(error.to(torch.float64).square().mean().sqrt().item())
    if error_rms == 0.0:
        ratio = None
        db = None
        status = "infinite_zero_error"
    elif signal_rms == 0.0:
        ratio = 0.0
        db = None
        status = "zero_signal"
    else:
        ratio = signal_rms / error_rms
        if math.isfinite(ratio):
            db = 20.0 * math.log10(ratio)
            status = "finite"
        else:
            ratio = None
            db = None
            status = "overflow"
    return {
        "signal_rms_s": signal_rms,
        "error_rms_s": error_rms,
        "ratio": ratio,
        "db": db,
        "status": status,
    }


def _sign_comparison(
    target: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, Any]:
    target = target.reshape(-1)
    observed = observed.reshape(-1)
    nonzero_target = target != 0.0
    flips = nonzero_target & ((target * observed) < 0.0)
    zeroed = nonzero_target & (observed == 0.0)
    eligible = int(nonzero_target.sum().item())
    return {
        "logical_values": int(target.numel()),
        "nonzero_target_values": eligible,
        "sign_flip_count": int(flips.sum().item()),
        "sign_flip_fraction_of_nonzero_target": (
            float(flips.sum().item()) / eligible if eligible else None
        ),
        "observed_zero_count_from_nonzero_target": int(zeroed.sum().item()),
        "target_zero_count": int((~nonzero_target).sum().item()),
    }


def build_decomposition_states(
    *,
    clean_selected: torch.Tensor,
    global_requested: torch.Tensor,
    mapped_target: torch.Tensor,
    apparent_endpoint: torch.Tensor,
    persistent_endpoint: torch.Tensor,
    accepted: torch.Tensor,
    layers: Sequence[LayerGeometry],
    optional_masks: Mapping[str, torch.Tensor | None],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Create exact endpoint-error partitions and correction states."""

    size = int(mapped_target.numel())
    tensors = (
        clean_selected,
        global_requested,
        mapped_target,
        apparent_endpoint,
        persistent_endpoint,
    )
    if any(value.shape != (size,) for value in tensors) or accepted.shape != (size,):
        raise ValueError("Expected all decomposition state vectors to match.")
    if accepted.dtype != torch.bool:
        raise ValueError("Expected accepted to be a boolean vector.")
    residual = apparent_endpoint - mapped_target
    states = {
        "clean_selected": clean_selected.clone(),
        "ideal_global_requested": global_requested.clone(),
        "ideal_mapped_target": mapped_target.clone(),
        "apparent_endpoint": apparent_endpoint.clone(),
        "persistent_endpoint_diagnostic": persistent_endpoint.clone(),
    }
    layer_partitions = []
    for layer in layers:
        mask = _layer_device_mask(layer, size=size)
        name = f"{layer.name.lower()}_endpoint_error_only"
        states[name] = mapped_target + residual * mask.to(residual.dtype)
        layer_partitions.append(states[name] - mapped_target)
    accepted_residual = residual * accepted.to(residual.dtype)
    failed_residual = residual * (~accepted).to(residual.dtype)
    states["accepted_endpoint_error_only"] = mapped_target + accepted_residual
    states["failure_endpoint_error_only"] = mapped_target + failed_residual

    partition_checks = {
        "layer_error_partition_max_abs": float(
            (torch.stack(layer_partitions).sum(dim=0) - residual)
            .abs()
            .max()
            .item()
        ),
        "acceptance_error_partition_max_abs": float(
            (accepted_residual + failed_residual - residual).abs().max().item()
        ),
    }
    optional = {}
    for correction, raw_mask in optional_masks.items():
        if raw_mask is None:
            optional[correction] = {
                "available": False,
                "affected_devices": None,
                "state": None,
            }
            continue
        mask = raw_mask.to(device="cpu", dtype=torch.bool)
        if mask.shape != (size,):
            raise ValueError(
                f"Expected optional {correction!r} mask to match deployment size."
            )
        count = int(mask.sum().item())
        state_name = f"apparent_{correction}_corrected"
        if count:
            corrected = apparent_endpoint.clone()
            corrected[mask] = mapped_target[mask]
            states[state_name] = corrected
        optional[correction] = {
            "available": True,
            "affected_devices": count,
            "state": state_name if count else None,
            "correction": "replace_saved_apparent_endpoint_with_mapped_target",
        }
    return states, {
        "partitions": partition_checks,
        "optional_corrections": optional,
    }


@contextmanager
def temporary_normalized_state(
    bindings: Sequence[Any],
    normalized_state: torch.Tensor,
    *,
    conductance_min: float,
    conductance_max: float,
) -> Iterator[None]:
    """Apply one flat state and restore every tensor byte-for-byte afterward."""

    selected = tuple(bindings)
    _keys, _shapes, sections = _binding_layout(selected)
    size = sum(binding.state.numel() for binding in selected)
    value = torch.as_tensor(normalized_state, device="cpu")
    if value.shape != (size,) or not bool(torch.all(torch.isfinite(value))):
        raise ValueError(
            f"Expected a finite normalized state vector of length {size}."
        )
    snapshots = tuple(binding.state.detach().clone() for binding in selected)
    span = conductance_max - conductance_min
    try:
        with torch.no_grad():
            for binding, section in zip(selected, sections):
                programmed = (
                    conductance_min + span * value[section]
                ).reshape(binding.state.shape)
                binding.state.copy_(programmed.to(binding.state))
        yield
    finally:
        with torch.no_grad():
            for binding, snapshot in zip(selected, snapshots):
                binding.state.copy_(snapshot)


def evaluate_reversible_states(
    bindings: Sequence[Any],
    states: Mapping[str, torch.Tensor],
    *,
    conductance_min: float,
    conductance_max: float,
    evaluator: Callable[[], Any],
    state_context: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate states in insertion order and prove final tensor restoration."""

    selected = tuple(bindings)
    initial = tuple(binding.state.detach().clone() for binding in selected)
    outputs = {}
    try:
        for name, state in states.items():
            with (
                temporary_normalized_state(
                    selected,
                    state,
                    conductance_min=conductance_min,
                    conductance_max=conductance_max,
                ),
                (
                    state_context(name)
                    if state_context is not None
                    else nullcontext()
                ),
            ):
                outputs[name] = evaluator()
    finally:
        restored = all(
            torch.equal(binding.state, snapshot)
            for binding, snapshot in zip(selected, initial)
        )
        if not restored:
            raise RuntimeError(
                "Expected read-only decomposition to restore every selected "
                "weight tensor exactly."
            )
    return outputs


def _update_cohort_digest(
    digest: Any,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    for name, value in (("inputs", inputs), ("labels", labels)):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())


def collect_evaluation_trace(
    stack: Any,
    teacher: Any,
    loader: Iterable,
    *,
    maximum_batches: int | None,
    sample_limit: int | None = None,
) -> EvaluationTrace:
    """Replay the production equations and retain ordered output diagnostics."""

    from experiments.mnist_shared import limited

    raw_pieces = []
    student_pieces = []
    teacher_pieces = []
    label_pieces = []
    digest = sha256()
    with torch.no_grad():
        examples_seen = 0
        for inputs, labels in limited(loader, maximum_batches):
            if sample_limit is not None:
                remaining = sample_limit - examples_seen
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
                labels = labels[:remaining]
            _update_cohort_digest(digest, inputs, labels)
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            teacher_logits = teacher.logits(inputs)
            stack.network.set_input(inputs, reset=True)
            stack.minimizer.compute_equilibrium()
            stack.cost.set_teacher(teacher_logits, labels)
            raw_scores = stack.cost.student_logits() / stack.cost.gain
            student_logits = raw_scores * stack.cost.gain
            raw_pieces.append(raw_scores.detach().cpu())
            student_pieces.append(student_logits.detach().cpu())
            teacher_pieces.append(teacher_logits.detach().cpu())
            label_pieces.append(labels.detach().cpu())
            examples_seen += int(labels.shape[0])
    if not student_pieces:
        raise ValueError("Expected decomposition evaluation to process examples.")
    raw_scores = torch.cat(raw_pieces)
    student_logits = torch.cat(student_pieces)
    teacher_logits = torch.cat(teacher_pieces)
    labels = torch.cat(label_pieces)
    examples, classes = student_logits.shape
    teacher_log_prob = F.log_softmax(teacher_logits, dim=1)
    teacher_prob = teacher_log_prob.exp()
    student_log_prob = F.log_softmax(student_logits, dim=1)
    raw_student_log_prob = F.log_softmax(raw_scores, dim=1)
    student_prediction = student_logits.argmax(dim=1)
    teacher_prediction = teacher_logits.argmax(dim=1)
    metrics = {
        "examples": examples,
        "kl_teacher_student": float(
            (
                teacher_prob
                * (teacher_log_prob - student_log_prob)
            ).sum(dim=1).mean().item()
        ),
        "raw_kl_teacher_student": float(
            (
                teacher_prob
                * (teacher_log_prob - raw_student_log_prob)
            ).sum(dim=1).mean().item()
        ),
        "student_accuracy": float(student_prediction.eq(labels).float().mean().item()),
        "teacher_accuracy": float(teacher_prediction.eq(labels).float().mean().item()),
        "teacher_agreement": float(
            student_prediction.eq(teacher_prediction).float().mean().item()
        ),
        "raw_score_rms": float(raw_scores.square().mean().sqrt().item()),
        "calibrated_score_rms": float(
            student_logits.square().mean().sqrt().item()
        ),
        "teacher_logit_rms": float(
            teacher_logits.square().mean().sqrt().item()
        ),
        "fixed_logit_gain": float(stack.cost.gain),
        "classes": classes,
    }
    return EvaluationTrace(
        metrics=metrics,
        student_logits=student_logits,
        raw_scores=raw_scores,
        teacher_logits=teacher_logits,
        labels=labels,
        cohort_sha256=digest.hexdigest(),
    )


def _class_margin(logits: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    selected = logits.gather(1, classes[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, classes[:, None], -torch.inf)
    return selected - masked.max(dim=1).values


def _output_state_summary(trace: EvaluationTrace) -> dict[str, Any]:
    prediction = trace.student_logits.argmax(dim=1)
    top_two = torch.topk(trace.student_logits, k=2, dim=1).values
    return {
        "metrics": trace.metrics,
        "top1_minus_top2_margin": _summary(top_two[:, 0] - top_two[:, 1]),
        "label_margin": _summary(
            _class_margin(trace.student_logits, trace.labels)
        ),
        "prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
        "student_logits_sha256": _tensor_sha256(trace.student_logits),
    }


def _output_comparison(
    observed: EvaluationTrace,
    reference: EvaluationTrace,
) -> dict[str, Any]:
    if (
        observed.cohort_sha256 != reference.cohort_sha256
        or not torch.equal(observed.labels, reference.labels)
        or not torch.equal(observed.teacher_logits, reference.teacher_logits)
    ):
        raise RuntimeError(
            "Expected every decomposition state to replay the identical "
            "ordered validation cohort and teacher outputs."
        )
    observed_prediction = observed.student_logits.argmax(dim=1)
    reference_prediction = reference.student_logits.argmax(dim=1)
    flips = observed_prediction != reference_prediction
    reference_correct = reference_prediction == reference.labels
    observed_correct = observed_prediction == observed.labels
    reference_margin = _class_margin(
        reference.student_logits,
        reference_prediction,
    )
    observed_reference_margin = _class_margin(
        observed.student_logits,
        reference_prediction,
    )
    count = int(flips.sum().item())
    return {
        "prediction_flip_count": count,
        "prediction_flip_fraction": count / int(flips.numel()),
        "reference_correct_to_observed_incorrect": int(
            (reference_correct & ~observed_correct).sum().item()
        ),
        "reference_incorrect_to_observed_correct": int(
            (~reference_correct & observed_correct).sum().item()
        ),
        "reference_class_margin": _summary(observed_reference_margin),
        "reference_class_margin_change": _summary(
            observed_reference_margin - reference_margin
        ),
        "reference_margin_on_flipped_examples": _summary(
            reference_margin[flips]
        ),
        "observed_reference_margin_on_flipped_examples": _summary(
            observed_reference_margin[flips]
        ),
        "student_logit_delta": _summary(
            observed.student_logits - reference.student_logits
        ),
    }


def _cost_summary(
    deployment: Mapping[str, Any],
    selection: torch.Tensor,
) -> dict[str, Any]:
    count = int(selection.sum().item())
    result: dict[str, Any] = {"devices": count}
    for name in (
        "set_count",
        "reset_count",
        "total_pulses",
        "verify_count",
        "reversals",
    ):
        value = deployment.get(name)
        if isinstance(value, torch.Tensor) and value.shape == selection.shape:
            selected = value[selection].to(torch.int64)
            result[name] = {
                "sum": int(selected.sum().item()),
                "mean": (
                    float(selected.to(torch.float64).mean().item())
                    if count
                    else None
                ),
                "median": (
                    float(selected.to(torch.float64).median().item())
                    if count
                    else None
                ),
                "maximum": int(selected.max().item()) if count else 0,
            }
        else:
            result[name] = None
    for name in (
        "accepted",
        "budget_exhausted",
        "corrupt",
        "saturated",
        "structural_quad_eligible",
        "structural_target_assignment_failure",
        "exact_target_in_support",
        "programming_eligible",
        "persistent_within_tolerance",
    ):
        value = deployment.get(name)
        result[name] = (
            int(value[selection].to(torch.bool).sum().item())
            if isinstance(value, torch.Tensor) and value.shape == selection.shape
            else None
        )
    conditioning = deployment.get("conditioning")
    conditioning_pulses = (
        conditioning.get("pulse_count")
        if isinstance(conditioning, Mapping)
        else None
    )
    conditioning_success = (
        conditioning.get("success")
        if isinstance(conditioning, Mapping)
        else None
    )
    if (
        isinstance(conditioning_pulses, torch.Tensor)
        and conditioning_pulses.shape == selection.shape
    ):
        selected = conditioning_pulses[selection].to(torch.int64)
        result["boundary_conditioning"] = {
            "devices": count,
            "pulse_sum": int(selected.sum().item()),
            "pulse_mean": (
                float(selected.to(torch.float64).mean().item())
                if count
                else None
            ),
            "pulse_median": (
                float(selected.to(torch.float64).median().item())
                if count
                else None
            ),
            "pulse_maximum": int(selected.max().item()) if count else 0,
            "success": (
                int(
                    conditioning_success[selection]
                    .to(torch.bool)
                    .sum()
                    .item()
                )
                if isinstance(conditioning_success, torch.Tensor)
                and conditioning_success.shape == selection.shape
                else None
            ),
        }
    else:
        result["boundary_conditioning"] = None
    raw = deployment.get("raw_apparent_endpoint")
    apparent = deployment.get("apparent_endpoint")
    result["endpoint_clipped"] = (
        int(((raw != apparent) & selection).sum().item())
        if isinstance(raw, torch.Tensor)
        and isinstance(apparent, torch.Tensor)
        and raw.shape == selection.shape
        and apparent.shape == selection.shape
        else None
    )
    fallback = deployment.get("pulse_resolved_fallback_mask")
    result["fallback"] = (
        int((fallback.to(torch.bool) & selection).sum().item())
        if isinstance(fallback, torch.Tensor) and fallback.shape == selection.shape
        else None
    )
    return result


def _layer_decomposition(
    *,
    layer: LayerGeometry,
    size: int,
    clean: torch.Tensor,
    global_requested: torch.Tensor,
    mapped: torch.Tensor,
    apparent: torch.Tensor,
    persistent: torch.Tensor,
    accepted: torch.Tensor,
    saturated: torch.Tensor | None,
    clipped: torch.Tensor,
    fallback: torch.Tensor | None,
    conductance_min: float,
    conductance_max: float,
    deployment: Mapping[str, Any],
    raw_active_coordinate: bool,
) -> dict[str, Any]:
    layer_mask = _layer_device_mask(layer, size=size)
    span = conductance_max - conductance_min
    cell_residuals = {
        "global_requested_minus_clean": _scaled_residual_summary(
            (global_requested - clean)[layer_mask],
            conductance_span=span,
        ),
        "mapped_minus_global_requested": _scaled_residual_summary(
            (mapped - global_requested)[layer_mask],
            conductance_span=span,
        ),
        "apparent_minus_mapped": _scaled_residual_summary(
            (apparent - mapped)[layer_mask],
            conductance_span=span,
        ),
        "persistent_minus_mapped": _scaled_residual_summary(
            (persistent - mapped)[layer_mask],
            conductance_span=span,
        ),
        "apparent_minus_mapped_accepted": _scaled_residual_summary(
            (apparent - mapped)[layer_mask & accepted],
            conductance_span=span,
        ),
        "apparent_minus_mapped_failed": _scaled_residual_summary(
            (apparent - mapped)[layer_mask & ~accepted],
            conductance_span=span,
        ),
    }
    contrasts = {
        name: logical_differential_contrast(
            value,
            layer,
            conductance_min=conductance_min,
            conductance_max=conductance_max,
        )
        for name, value in (
            ("clean_selected", clean),
            ("global_requested", global_requested),
            ("mapped_target", mapped),
            ("apparent_endpoint", apparent),
            ("persistent_endpoint", persistent),
        )
    }
    target = contrasts["mapped_target"]
    contrast_report = {
        "definition": (
            "(Gpp-Gpm-Gmp+Gmm)"
            if layer.encoding == "single"
            else "((Gplus-Gminus)pp-(Gplus-Gminus)pm-"
            "(Gplus-Gminus)mp+(Gplus-Gminus)mm)"
        ),
        "logical_shape": list(target.shape),
        "mapped_target_signal_s": _summary(target),
    }
    for name in (
        "clean_selected",
        "global_requested",
        "apparent_endpoint",
        "persistent_endpoint",
    ):
        observed = contrasts[name]
        error = observed - target
        contrast_report[name] = {
            "signal_s": _summary(observed),
            "error_from_mapped_target_s": _summary(error),
            "snr_from_mapped_target": _snr(target, error),
            "sign_comparison_to_mapped_target": _sign_comparison(
                target,
                observed,
            ),
        }
    differential_report = None
    if raw_active_coordinate:
        differentials = {
            name: value / 2.0 for name, value in contrasts.items()
        }
        differential_target = differentials["mapped_target"]
        differential_report = {
            "definition": "D=(gpp-gpm-gmp+gmm)/2",
            "coordinate": "array_wide_raw_active_g",
            "logical_shape": list(differential_target.shape),
            "mapped_target_signal": _summary(differential_target),
        }
        for name in (
            "clean_selected",
            "global_requested",
            "apparent_endpoint",
            "persistent_endpoint",
        ):
            observed = differentials[name]
            error = observed - differential_target
            differential_report[name] = {
                "signal": _summary(observed),
                "error_from_mapped_target": _summary(error),
                "snr_from_mapped_target": _snr(
                    differential_target,
                    error,
                ),
                "sign_comparison_to_mapped_target": _sign_comparison(
                    differential_target,
                    observed,
                ),
            }
    mask_counts = {
        "accepted": int((layer_mask & accepted).sum().item()),
        "failed": int((layer_mask & ~accepted).sum().item()),
        "saturated": (
            int((layer_mask & saturated).sum().item())
            if saturated is not None
            else None
        ),
        "clipped": int((layer_mask & clipped).sum().item()),
        "fallback": (
            int((layer_mask & fallback).sum().item())
            if fallback is not None
            else None
        ),
    }
    return {
        "layer_index": layer.index,
        "encoding": layer.encoding,
        "rail_layout": layer.rail_layout,
        "binding_keys": list(layer.binding_keys),
        "binding_shapes": [list(shape) for shape in layer.binding_shapes],
        "devices": layer.device_count,
        "mask_counts": mask_counts,
        "cell_residuals": cell_residuals,
        "logical_differential_contrast": contrast_report,
        "raw_active_logical_differential_D": differential_report,
        "programming_costs": _cost_summary(deployment, layer_mask),
    }


def _population_from_deployment(
    deployment: Mapping[str, Any],
    *,
    config: Any,
) -> Any:
    from training.ibm_reram_hwa import (
        IbmReramArrayPopulation,
        _population_fingerprint,
    )

    scalars = deployment.get("population_scalars")
    tensors = deployment.get("population")
    if not isinstance(scalars, Mapping) or set(scalars) != {
        "preset",
        "aihwkit_version",
        "nominal_dw_min",
        "dw_min_std",
        "write_noise_std",
    }:
        raise ValueError("Expected exact embedded IBM OM population scalars.")
    if scalars.get("preset") != "reram_array_om":
        raise ValueError("Expected the embedded IBM OM population preset.")
    expected_tensor_keys = {
        "max_bound",
        "min_bound",
        "dwmin_up",
        "dwmin_down",
        "reference",
        "corrupt",
        "published_corrupt",
    }
    if not isinstance(tensors, Mapping) or set(tensors) != expected_tensor_keys:
        raise ValueError("Expected exact embedded IBM OM population tensors.")
    keys = tuple(deployment["binding_keys"])
    shapes = tuple(tuple(shape) for shape in deployment["binding_shapes"])
    binding_seeds = tuple(deployment.get("binding_sampling_seeds", ()))
    donor_seeds = tuple(deployment.get("donor_sampling_seeds", ()))
    if (
        len(binding_seeds) != len(keys)
        or len(donor_seeds) != len(keys)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (*binding_seeds, *donor_seeds)
        )
    ):
        raise ValueError(
            "Expected one integer sampling and donor seed per binding."
        )
    fingerprint = deployment.get("population_fingerprint")
    scalar_fingerprint_fields = {
        "aihwkit_version": str(scalars["aihwkit_version"]),
        "nominal_dw_min": float(scalars["nominal_dw_min"]),
        "dw_min_std": float(scalars["dw_min_std"]),
        "write_noise_std": float(scalars["write_noise_std"]),
    }
    recomputed = _population_fingerprint(
        assignment_seed=config.assignment_seed,
        corruption_policy=config.corruption_policy,
        keys=keys,
        shapes=shapes,
        binding_sampling_seeds=binding_seeds,
        donor_sampling_seeds=donor_seeds,
        scalar_parameters=scalar_fingerprint_fields,
        tensors=tensors,
    )
    if fingerprint != recomputed:
        raise ValueError(
            "Expected embedded IBM OM population fingerprint to recompute "
            f"exactly. Provided value: saved={fingerprint!r}, "
            f"recomputed={recomputed!r}."
        )
    return IbmReramArrayPopulation(
        assignment_seed=config.assignment_seed,
        corruption_policy=config.corruption_policy,
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=binding_seeds,
        donor_sampling_seeds=donor_seeds,
        nominal_dw_min=float(scalars["nominal_dw_min"]),
        dw_min_std=float(scalars["dw_min_std"]),
        write_noise_std=float(scalars["write_noise_std"]),
        max_bound=tensors["max_bound"],
        min_bound=tensors["min_bound"],
        dwmin_up=tensors["dwmin_up"],
        dwmin_down=tensors["dwmin_down"],
        reference=tensors["reference"],
        corrupt=tensors["corrupt"],
        published_corrupt=tensors["published_corrupt"],
        fingerprint=str(fingerprint),
        aihwkit_version=str(scalars["aihwkit_version"]),
    )


def population_from_deployment(
    deployment: Mapping[str, Any],
    *,
    config: Any,
) -> Any:
    """Reconstruct and revalidate the fixed population embedded in a bundle."""

    return _population_from_deployment(deployment, config=config)


def _reset_commissioning_from_deployment(
    deployment: Mapping[str, Any],
    *,
    config: Any,
    population: Any,
) -> Any | None:
    """Restore only the observed RESET commissioning state from a sidecar."""

    from training.ibm_reram_hwa import IbmReramResetCommissioning

    raw = deployment.get("reset_commissioning")
    if config.target_mapping != "shared_reset_relative_quad":
        if raw is not None:
            raise ValueError(
                "Expected RESET commissioning only for the shared "
                "RESET-relative target mapping."
            )
        return None
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema",
        "schema_version",
        "population_fingerprint",
        "assignment_seed",
        "commissioning_seed",
        "read_samples",
        "guard_standard_errors",
        "binding_keys",
        "binding_shapes",
        "dual_rail_layout_by_parameter",
        "observed",
        "report",
    }:
        raise ValueError(
            "Expected a strict observed RESET commissioning bundle for "
            "shared RESET-relative mapper replay."
        )
    keys = tuple(raw.get("binding_keys", ()))
    shapes = tuple(tuple(shape) for shape in raw.get("binding_shapes", ()))
    layouts = raw.get("dual_rail_layout_by_parameter")
    expected_layouts = dict(config.dual_rail_layout_by_parameter or ())
    observed = raw.get("observed")
    report = raw.get("report")
    seed = raw.get("commissioning_seed")
    reads = raw.get("read_samples")
    guard = raw.get("guard_standard_errors")
    if (
        raw.get("schema") != "ebl.ibm_reram.reset_relative_commissioning"
        or raw.get("schema_version") != 1
        or raw.get("population_fingerprint") != population.fingerprint
        or raw.get("assignment_seed") != config.assignment_seed
        or keys != population.binding_keys
        or shapes != population.binding_shapes
        or layouts != expected_layouts
        or not isinstance(observed, Mapping)
        or set(observed)
        != {"reset_mean", "reset_standard_error", "baseline"}
        or not isinstance(report, Mapping)
        or report.get("population_fingerprint") != population.fingerprint
        or report.get("commissioning_seed") != seed
        or report.get("read_samples") != reads
        or report.get("guard_standard_errors") != guard
        or report.get("controller_observations_only") is not True
        or report.get("hidden_device_bounds_consumed_by_target_mapper")
        is not False
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(reads, bool)
        or not isinstance(reads, int)
        or reads != config.reset_read_samples
        or isinstance(guard, bool)
        or not isinstance(guard, (int, float))
        or float(guard) != config.reset_guard_standard_errors
    ):
        raise ValueError(
            "Expected RESET commissioning identity, observation policy, and "
            "config provenance to match the deployment exactly."
        )
    size = population.size
    tensors = {
        name: _validate_flat_tensor(
            observed.get(name),
            name=f"reset_commissioning.observed.{name}",
            size=size,
            dtype=torch.float32,
        )
        for name in ("reset_mean", "reset_standard_error", "baseline")
    }
    if bool(torch.any(tensors["reset_standard_error"] < 0.0)):
        raise ValueError(
            "Expected non-negative RESET commissioning standard errors."
        )
    # Validate that every report value is strict JSON without converting the
    # tensor-bearing commissioning bundle itself.
    normalized_report = _json_value(report)
    saved_mapping_report = deployment.get("target_mapping_report")
    if (
        not isinstance(saved_mapping_report, Mapping)
        or saved_mapping_report.get("commissioning") != normalized_report
        or normalized_report.get("devices") != size
        or not isinstance(normalized_report.get("quad_count"), int)
        or normalized_report.get("quad_count") * 4 != size
        or normalized_report.get("total_reset_pulses") != size * reads
        or normalized_report.get("reset_mean_sha256")
        != _commissioning_tensor_sha256(tensors["reset_mean"])
        or normalized_report.get("reset_standard_error_sha256")
        != _commissioning_tensor_sha256(tensors["reset_standard_error"])
        or normalized_report.get("expanded_baseline_sha256")
        != _commissioning_tensor_sha256(tensors["baseline"])
    ):
        raise ValueError(
            "Expected RESET commissioning reports, tensor digests, and "
            "mapping provenance to match the saved observations exactly."
        )
    return IbmReramResetCommissioning(
        population_fingerprint=population.fingerprint,
        assignment_seed=config.assignment_seed,
        commissioning_seed=seed,
        read_samples=reads,
        guard_standard_errors=float(guard),
        binding_keys=keys,
        binding_shapes=shapes,
        dual_rail_layout_by_parameter=tuple(
            (key, str(expected_layouts[key])) for key in keys
        ),
        reset_mean=tensors["reset_mean"],
        reset_standard_error=tensors["reset_standard_error"],
        baseline=tensors["baseline"],
        report=normalized_report,
    )


def validate_deployment_contract(
    deployment: Any,
    *,
    expected_selected_weights_sha256: str,
    expected_binding_keys: Sequence[str],
    expected_binding_shapes: Sequence[tuple[int, int]],
    configured_modifiers: Sequence[Mapping[str, Any]],
    checkpoint_metadata: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], Any, dict[str, Any]]:
    """Validate the saved sidecar and replay its pure target mapper."""

    from training.ibm_reram_hwa import (
        IbmReramHwaConfig,
        map_ibm_reram_array_targets,
        validate_ibm_reram_target_mapping_preflight,
    )

    if not isinstance(deployment, Mapping):
        raise ValueError("Expected deployment sidecar to contain an object.")
    schema = deployment.get("schema")
    if schema != _PULSE_SCHEMA or deployment.get("schema_version") != 1:
        raise ValueError(
            "Expected a current pulse-resolved IBM OM deployment sidecar "
            "with its embedded fixed population."
        )
    if deployment.get("endpoint_application_policy") != ENDPOINT_APPLICATION_POLICY:
        raise ValueError(
            "Expected the apparent-forward/persistent-update-state endpoint policy."
        )
    report = deployment.get("report")
    if (
        not isinstance(report, Mapping)
        or report.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
    ):
        raise ValueError("Expected deployment report endpoint-policy parity.")
    selected_weights_sha256 = deployment.get("selected_weights_sha256")
    source_weights_sha256 = deployment.get("source_weights_sha256")
    if selected_weights_sha256 is not None:
        if (
            source_weights_sha256 is not None
            or selected_weights_sha256 != expected_selected_weights_sha256
        ):
            raise ValueError(
                "Expected deployment selected-weight SHA-256 to match "
                "--weights exclusively."
            )
        selected_epoch = deployment.get("selected_epoch")
        if (
            isinstance(selected_epoch, bool)
            or not isinstance(selected_epoch, int)
            or selected_epoch != checkpoint_metadata.get("selection_epoch")
        ):
            raise ValueError(
                "Expected deployment and selected checkpoint epoch "
                "provenance to match."
            )
        weights_provenance_field = "selected_weights_sha256"
    else:
        if (
            source_weights_sha256 != expected_selected_weights_sha256
            or deployment.get("selected_epoch") is not None
        ):
            raise ValueError(
                "Expected validation deployment source-weight SHA-256 to "
                "match --weights without an unbound selected epoch."
            )
        selected_epoch = checkpoint_metadata.get("selection_epoch")
        if isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int):
            raise ValueError(
                "Expected the source checkpoint to bind its selected epoch."
            )
        weights_provenance_field = "source_weights_sha256"
    device_model_sha256 = deployment.get("device_model_sha256")
    if (
        not isinstance(device_model_sha256, str)
        or _SHA256.fullmatch(device_model_sha256) is None
        or checkpoint_metadata.get("device_model_sha256") != device_model_sha256
    ):
        raise ValueError(
            "Expected deployment and checkpoint device-model SHA-256 parity."
        )
    keys = tuple(deployment.get("binding_keys", ()))
    shapes = tuple(tuple(shape) for shape in deployment.get("binding_shapes", ()))
    if keys != tuple(expected_binding_keys) or shapes != tuple(expected_binding_shapes):
        raise ValueError(
            "Expected deployment binding catalog and shapes to match the "
            f"selected model. Provided keys={keys!r}, shapes={shapes!r}."
        )
    size = sum(math.prod(shape) for shape in shapes)
    raw_config = deployment.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Expected deployment to embed its IBM OM config.")
    try:
        deployment_config = IbmReramHwaConfig(**dict(raw_config))
    except (TypeError, ValueError) as error:
        raise ValueError("Expected a valid embedded IBM OM config.") from error
    normalized = asdict(deployment_config)
    normalized_candidates = []
    for parameters in configured_modifiers:
        try:
            normalized_candidates.append(
                asdict(IbmReramHwaConfig(**dict(parameters)))
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Expected configured IBM modifier parameters.") from error
    matches = [
        candidate
        for candidate in normalized_candidates
        if candidate == normalized
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected deployment config to match exactly one configured IBM "
            "OM modifier."
        )
    if deployment_config.execution != "pulse_resolved":
        raise ValueError(
            "Expected the saved completed-pilot deployment to be pulse resolved."
        )
    tensors: dict[str, torch.Tensor] = {}
    for name in _TENSOR_FIELDS:
        dtype = torch.bool if name == "accepted" else torch.float32
        tensors[name] = _validate_flat_tensor(
            deployment.get(name),
            name=name,
            size=size,
            dtype=dtype,
        )
    expected_apparent = (
        tensors["raw_apparent_endpoint"]
        if deployment_config.endpoint_policy == "preserve"
        else tensors["raw_apparent_endpoint"].clamp(0.0, 1.0)
    )
    if not torch.equal(tensors["apparent_endpoint"], expected_apparent):
        raise ValueError(
            "Expected apparent endpoint to implement the configured "
            "preserve-or-clip policy exactly."
        )
    pulse_vectors = {}
    for name in ("budget_exhausted", "corrupt", "saturated"):
        pulse_vectors[name] = _validate_flat_tensor(
            deployment.get(name),
            name=name,
            size=size,
            dtype=torch.bool,
        )
    for name in (
        "set_count",
        "reset_count",
        "total_pulses",
        "verify_count",
        "reversals",
    ):
        value = _validate_flat_tensor(
            deployment.get(name),
            name=name,
            size=size,
            dtype=torch.int64,
        )
        if bool(torch.any(value < 0)):
            raise ValueError(f"Expected non-negative deployment {name!r}.")
        pulse_vectors[name] = value
    if not torch.equal(
        pulse_vectors["set_count"] + pulse_vectors["reset_count"],
        pulse_vectors["total_pulses"],
    ):
        raise ValueError("Expected total pulses to equal SET plus RESET pulses.")
    fingerprint = deployment.get("population_fingerprint")
    mapping_report = deployment.get("target_mapping_report")
    if (
        not isinstance(fingerprint, str)
        or _SHA256.fullmatch(fingerprint) is None
        or report.get("population_fingerprint") != fingerprint
        or not isinstance(mapping_report, Mapping)
        or mapping_report.get("population_fingerprint") != fingerprint
        or report.get("target_mapping_report") != mapping_report
        or mapping_report.get("target_mapping")
        != deployment_config.target_mapping
    ):
        raise ValueError(
            "Expected deployment/report/mapping population fingerprint and "
            "target-mapping parity."
        )
    population = _population_from_deployment(
        deployment,
        config=deployment_config,
    )
    reset_commissioning = _reset_commissioning_from_deployment(
        deployment,
        config=deployment_config,
        population=population,
    )
    remapped, remapping_report = map_ibm_reram_array_targets(
        tensors["global_requested_target"],
        population,
        target_mapping=deployment_config.target_mapping,
        dual_rail_layout_by_parameter=(
            deployment_config.dual_rail_layout_by_parameter
        ),
        common_window_margin_fraction=(
            deployment_config.common_window_margin_fraction
        ),
        reset_commissioning=reset_commissioning,
        reset_relative_mode=deployment_config.reset_relative_mode,
        reset_relative_contrast_step=(
            deployment_config.reset_relative_contrast_step
        ),
        raw_active_mode=deployment_config.raw_active_mode,
        raw_active_unsupported_quad_policy=(
            deployment_config.raw_active_unsupported_quad_policy
        ),
    )
    validate_ibm_reram_target_mapping_preflight(remapping_report)
    if not torch.equal(remapped, tensors["requested_target"]):
        raise ValueError(
            "Expected saved requested targets to equal read-only mapper replay."
        )
    if deployment_config.target_mapping == "raw_active_p90_quad":
        integer_fields = (
            "devices",
            "quad_count",
            "required_p90_quad_count",
            "p90_eligible_quad_count",
            "structural_failure_quad_count",
            "structural_failure_cell_count",
            "exact_target_supported_quad_count",
            "mapped_target_below_lower_bound_empty_quad",
            "mapped_target_above_upper_bound_empty_quad",
        )
    elif deployment_config.target_mapping == "dual_rail_quad_common_window":
        integer_fields = (
            "devices",
            "quad_count",
            "common_window_empty_quad_count",
            "mapped_target_below_lower_bound_nonempty_quad",
            "mapped_target_above_upper_bound_nonempty_quad",
            "mapped_target_below_lower_bound_empty_quad",
            "mapped_target_above_upper_bound_empty_quad",
        )
    elif deployment_config.target_mapping == "differential_pair_common_window":
        integer_fields = (
            "devices",
            "pair_count",
            "common_window_empty_pair_count",
            "mapped_target_below_lower_bound_nonempty_pair",
            "mapped_target_above_upper_bound_nonempty_pair",
            "mapped_target_below_lower_bound_empty_pair",
            "mapped_target_above_upper_bound_empty_pair",
        )
    else:
        integer_fields = (
            "devices",
            "quad_count",
            "fully_supported_quad_count",
            "mapped_target_below_lower_bound_nonempty_quad",
            "mapped_target_above_upper_bound_nonempty_quad",
            "mapped_target_outside_0_1",
        )
    if any(
        mapping_report.get(name) != remapping_report.get(name)
        for name in integer_fields
    ):
        raise ValueError(
            "Expected saved and replayed mapping structural counts to match."
        )
    clipped_count = int(
        (
            tensors["raw_apparent_endpoint"]
            != tensors["apparent_endpoint"]
        ).sum().item()
    )
    reported_pulse = report.get("pulse_count")
    if (
        report.get("execution") != "pulse_resolved"
        or report.get("devices") != size
        or report.get("accepted") != int(tensors["accepted"].sum().item())
        or report.get("budget_exhausted")
        != int(pulse_vectors["budget_exhausted"].sum().item())
        or report.get("corrupt") != int(pulse_vectors["corrupt"].sum().item())
        or report.get("saturated")
        != int(pulse_vectors["saturated"].sum().item())
        or report.get("endpoint_clipped") != clipped_count
        or not isinstance(reported_pulse, Mapping)
        or reported_pulse.get("maximum")
        != int(pulse_vectors["total_pulses"].max().item())
    ):
        raise ValueError(
            "Expected deployment report counts to match exact pulse tensors."
        )
    if deployment_config.target_mapping == "raw_active_p90_quad":
        coordinate = deployment.get("coordinate")
        structural = _validate_flat_tensor(
            deployment.get("structural_quad_eligible"),
            name="structural_quad_eligible",
            size=size,
            dtype=torch.bool,
        )
        exact_support = _validate_flat_tensor(
            deployment.get("exact_target_in_support"),
            name="exact_target_in_support",
            size=size,
            dtype=torch.bool,
        )
        programming_eligible = _validate_flat_tensor(
            deployment.get("programming_eligible"),
            name="programming_eligible",
            size=size,
            dtype=torch.bool,
        )
        if (
            not isinstance(coordinate, Mapping)
            or coordinate.get("version")
            != "ibm_om_raw_active_development_assignment_84001_v1"
            or coordinate.get("reference_excluded_from_numerics") is not True
            or not torch.equal(
                programming_eligible,
                structural
                & exact_support
                & _validate_flat_tensor(
                    deployment.get("conditioning", {}).get("success")
                    if isinstance(deployment.get("conditioning"), Mapping)
                    else None,
                    name="conditioning.success",
                    size=size,
                    dtype=torch.bool,
                ),
            )
            or report.get("structural_quad_eligible")
            != int(structural.sum().item())
            or report.get("exact_target_in_support")
            != int(exact_support.sum().item())
            or report.get("programming_eligible")
            != int(programming_eligible.sum().item())
            or report.get("reference_consumed_by_plant_or_controller")
            is not False
        ):
            raise ValueError(
                "Expected strict raw-active coordinate, structural support, "
                "conditioning, and controller-capability provenance."
            )
    receipt = deployment.get("population_sampling_receipt")
    if receipt is not None:
        request = receipt.get("request") if isinstance(receipt, Mapping) else None
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("population_fingerprint") != fingerprint
            or receipt.get("num_cells") != size
            or not isinstance(request, Mapping)
            or request.get("assignment_seed") != deployment_config.assignment_seed
            or request.get("corruption_policy")
            != deployment_config.corruption_policy
            or tuple(request.get("binding_keys", ())) != keys
            or tuple(tuple(shape) for shape in request.get("binding_shapes", ()))
            != shapes
        ):
            raise ValueError(
                "Expected population receipt fingerprint, size, and layout parity."
            )
    return tensors, deployment_config, {
        "schema": schema,
        "population_fingerprint": fingerprint,
        "selected_epoch": selected_epoch,
        "weights_provenance_field": weights_provenance_field,
        "device_model_sha256": device_model_sha256,
        "mapping_report": _json_value(mapping_report),
        "programming_report": _json_value(report),
    }


def run_decomposition(
    *,
    config_path: Path,
    deployment_config_path: Path | None = None,
    weights_path: Path,
    teacher_weights_path: Path,
    deployment_path: Path,
) -> dict[str, Any]:
    """Load strict artifacts and produce the read-only decomposition report."""

    from experiments.definitions import resolve_experiment_config
    from experiments.mnist_relu_drn.components import build_student_stack
    from experiments.mnist_relu_drn.runtime import (
        _amplification_index_report,
        _load_teacher,
        _validate_checkpoint_metadata,
    )
    from experiments.mnist_shared import build_mnist_loaders
    from experiments.schema import RunMode, to_plain_data
    from training.checkpoint import load_named_weights
    from training.ibm_om.spec import backfill_weight_modifier_metadata

    config_path = _require_file(config_path, label="--config")
    deployment_config_path = (
        _require_file(deployment_config_path, label="--deployment-config")
        if deployment_config_path is not None
        else None
    )
    weights_path = _require_file(weights_path, label="--weights")
    teacher_weights_path = _require_file(
        teacher_weights_path,
        label="--teacher-weights",
    )
    deployment_path = _require_file(deployment_path, label="--deployment")
    definition, spec = resolve_experiment_config(config_path, RunMode.TRAIN)
    if definition.experiment_id != "mnist_relu_drn_kd.v1":
        raise ValueError("Expected the strict MNIST ReLU DRN KD experiment.")
    expected_target_mappings = {
        "single": {
            "dual_rail_quad_common_window",
            "shared_reset_relative_quad",
            "raw_active_p90_quad",
        },
        "differential": {"differential_pair_common_window"},
    }.get(spec.model.encoding)
    deployment_spec = None
    if deployment_config_path is None:
        modifiers = (
            spec.settings.weight_modifier,
            spec.settings.selection_weight_modifier,
        )
    else:
        deployment_definition, deployment_spec = resolve_experiment_config(
            deployment_config_path,
            RunMode.VALIDATE,
        )
        if deployment_definition.experiment_id != definition.experiment_id:
            raise ValueError(
                "Expected checkpoint and deployment configs to select one "
                "experiment."
            )
        for field in (
            "runtime",
            "data",
            "teacher",
            "model",
            "solver",
            "mapping",
        ):
            if getattr(deployment_spec, field) != getattr(spec, field):
                raise ValueError(
                    "Expected checkpoint and deployment configs to match "
                    f"exactly for {field!r}."
                )
        checkpoint_modifier = spec.settings.selection_weight_modifier
        deployment_modifier = deployment_spec.settings.weight_modifier
        checkpoint_parameters = dict(checkpoint_modifier.parameters)
        deployment_parameters = dict(deployment_modifier.parameters)
        for seed_field in ("assignment_seed", "endpoint_seed"):
            checkpoint_parameters.pop(seed_field, None)
            deployment_parameters.pop(seed_field, None)
        if (
            checkpoint_modifier.type != deployment_modifier.type
            or checkpoint_parameters != deployment_parameters
        ):
            raise ValueError(
                "Expected held-out deployment modifier to differ from the "
                "checkpoint selection modifier only in assignment and "
                "endpoint seeds."
            )
        modifiers = (deployment_modifier,)
    configured_ibm = tuple(
        dict(modifier.parameters)
        for modifier in modifiers
        if modifier.type == "ibm_reram_om_program_verify"
    )
    if not configured_ibm or any(
        parameters.get("target_mapping") not in expected_target_mappings
        for parameters in configured_ibm
    ):
        raise ValueError(
            "Expected the config to use the encoding-matched IBM OM "
            "array-specific target mapping."
        )

    teacher_sha256 = sha256_file(teacher_weights_path)
    stack = build_student_stack(spec, enable_measured=False)
    loaded = load_named_weights(weights_path, stack.bundle.catalog)
    _validate_checkpoint_metadata(
        loaded.metadata,
        spec=spec,
        teacher_sha256=teacher_sha256,
        expected_amplification_indices=_amplification_index_report(stack),
    )
    expected_weight_modifier = {
        "type": spec.settings.weight_modifier.type,
        "parameters": to_plain_data(spec.settings.weight_modifier.parameters),
    }
    expected_selection_modifier = {
        "type": spec.settings.selection_weight_modifier.type,
        "parameters": to_plain_data(
            spec.settings.selection_weight_modifier.parameters
        ),
    }
    if (
        backfill_weight_modifier_metadata(loaded.metadata.get("weight_modifier"))
        != expected_weight_modifier
        or backfill_weight_modifier_metadata(
            loaded.metadata.get("selection_weight_modifier")
        )
        != expected_selection_modifier
    ):
        raise ValueError(
            "Expected selected checkpoint modifier provenance to match config."
        )
    fixed_gain = loaded.metadata.get("fixed_logit_gain")
    if (
        isinstance(fixed_gain, bool)
        or not isinstance(fixed_gain, (int, float))
        or not math.isfinite(float(fixed_gain))
        or float(fixed_gain) <= 0.0
    ):
        raise ValueError("Expected selected checkpoint to provide a positive gain.")
    stack.cost.gain = float(fixed_gain)
    teacher, _teacher_metadata = _load_teacher(
        teacher_weights_path,
        device=stack.device,
        spec=spec,
    )

    bindings = tuple(stack.bundle.catalog.trainable)
    keys, shapes, sections = _binding_layout(bindings)
    selected_flat = torch.cat(
        tuple(binding.state.detach().reshape(-1) for binding in bindings)
    )
    conductance_min = float(spec.model.conductance_min)
    conductance_max = float(spec.model.conductance_max)
    # Reproduce the modifier's arithmetic on the model device before moving
    # to CPU.  Performing this subtraction/division after a CPU copy can
    # differ by one FP32 ULP from the saved CUDA global target.
    selected_normalized = (
        selected_flat - conductance_min
    ) / (conductance_max - conductance_min)
    selected_normalized = selected_normalized.detach().cpu()
    selected_sha256 = sha256_file(weights_path)
    deployment = _strict_torch_load(deployment_path)
    tensors, deployment_config, contract = validate_deployment_contract(
        deployment,
        expected_selected_weights_sha256=selected_sha256,
        expected_binding_keys=keys,
        expected_binding_shapes=shapes,
        configured_modifiers=configured_ibm,
        checkpoint_metadata=loaded.metadata,
    )
    if not torch.equal(
        selected_normalized.to(torch.float32),
        tensors["global_requested_target"],
    ):
        raise ValueError(
            "Expected selected checkpoint tensors to reproduce the saved "
            "global requested target exactly."
        )
    layout_mapping = (
        None
        if deployment_config.dual_rail_layout_by_parameter is None
        else dict(deployment_config.dual_rail_layout_by_parameter)
    )
    layers = canonical_layer_geometries(
        keys,
        shapes,
        sections,
        encoding=spec.model.encoding,
        dual_rail_layout_by_parameter=layout_mapping,
    )
    size = int(tensors["requested_target"].numel())
    saturated = deployment.get("saturated")
    saturated_mask = (
        _validate_flat_tensor(
            saturated,
            name="saturated",
            size=size,
            dtype=torch.bool,
        )
        if saturated is not None
        else None
    )
    clipped_mask = (
        tensors["raw_apparent_endpoint"] != tensors["apparent_endpoint"]
    )
    fallback = deployment.get("pulse_resolved_fallback_mask")
    fallback_mask = (
        _validate_flat_tensor(
            fallback,
            name="pulse_resolved_fallback_mask",
            size=size,
            dtype=torch.bool,
        )
        if fallback is not None
        else None
    )
    states, state_contract = build_decomposition_states(
        clean_selected=selected_normalized.to(torch.float32),
        global_requested=tensors["global_requested_target"],
        mapped_target=tensors["requested_target"],
        apparent_endpoint=tensors["apparent_endpoint"],
        persistent_endpoint=tensors["persistent_endpoint"],
        accepted=tensors["accepted"],
        layers=layers,
        optional_masks={
            "saturation": saturated_mask,
            "clipping": clipped_mask,
            "fallback": fallback_mask,
        },
    )
    physical_gain = (
        float(deployment_config.forward_logit_gain)
        if deployment_config.forward_logit_gain is not None
        else float(fixed_gain)
    )

    @contextmanager
    def state_gain(state_name: str) -> Iterator[None]:
        original_gain = float(stack.cost.gain)
        stack.cost.gain = (
            float(fixed_gain)
            if state_name in {"clean_selected", "ideal_global_requested"}
            else physical_gain
        )
        try:
            yield
        finally:
            stack.cost.gain = original_gain

    data = build_mnist_loaders(
        spec.data,
        data_seed=spec.runtime.data_seed,
        calibration_examples=spec.mapping.calibration_examples,
        calibration_batch_size=spec.mapping.calibration_batch_size,
    )
    if deployment_spec is None:
        evaluation_loader = data.validation
        evaluation_split = "validation"
        maximum_batches = spec.settings.max_validation_batches
        sample_limit = None
        expected_examples = int(spec.data.validation_points)
        if maximum_batches is not None:
            expected_examples = min(
                expected_examples,
                int(maximum_batches) * int(spec.data.batch_size),
            )
    else:
        evaluation_split = deployment_spec.settings.split
        evaluation_loader = (
            data.validation if evaluation_split == "validation" else data.test
        )
        maximum_batches = None
        sample_limit = deployment_spec.settings.sample_limit
        expected_examples = len(evaluation_loader.dataset)
        if sample_limit is not None:
            expected_examples = min(expected_examples, int(sample_limit))
    traces = evaluate_reversible_states(
        bindings,
        states,
        conductance_min=conductance_min,
        conductance_max=conductance_max,
        evaluator=lambda: collect_evaluation_trace(
            stack,
            teacher,
            evaluation_loader,
            maximum_batches=maximum_batches,
            sample_limit=sample_limit,
        ),
        state_context=state_gain,
    )
    cohort_digests = {trace.cohort_sha256 for trace in traces.values()}
    example_counts = {trace.metrics["examples"] for trace in traces.values()}
    if len(cohort_digests) != 1 or len(example_counts) != 1:
        raise RuntimeError(
            "Expected every state to replay one identical ordered cohort."
        )
    if example_counts != {expected_examples}:
        raise RuntimeError(
            "Expected the exact configured validation cohort size. Provided "
            f"value: expected={expected_examples}, observed={example_counts!r}."
        )
    mapped_trace = traces["ideal_mapped_target"]
    clean_trace = traces["clean_selected"]
    output_states = {
        name: {
            **_output_state_summary(trace),
            "versus_ideal_mapped_target": _output_comparison(
                trace,
                mapped_trace,
            ),
            "versus_clean_selected": _output_comparison(
                trace,
                clean_trace,
            ),
        }
        for name, trace in traces.items()
    }
    layer_reports = {
        layer.name: _layer_decomposition(
            layer=layer,
            size=size,
            clean=states["clean_selected"],
            global_requested=states["ideal_global_requested"],
            mapped=states["ideal_mapped_target"],
            apparent=states["apparent_endpoint"],
            persistent=states["persistent_endpoint_diagnostic"],
            accepted=tensors["accepted"],
            saturated=saturated_mask,
            clipped=clipped_mask,
            fallback=fallback_mask,
            conductance_min=conductance_min,
            conductance_max=conductance_max,
            deployment=deployment,
            raw_active_coordinate=(
                deployment_config.target_mapping
                == "raw_active_p90_quad"
            ),
        )
        for layer in layers
    }
    all_devices = torch.ones(size, dtype=torch.bool)
    population_receipt = deployment.get("population_sampling_receipt")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "operation": "read_only_saved_endpoint_replay",
        "inputs": {
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "deployment_config": (
                {
                    "path": str(deployment_config_path),
                    "sha256": sha256_file(deployment_config_path),
                }
                if deployment_config_path is not None
                else None
            ),
            "selected_weights": {
                "path": str(weights_path),
                "sha256": selected_sha256,
            },
            "teacher_weights": {
                "path": str(teacher_weights_path),
                "sha256": teacher_sha256,
            },
            "deployment_sidecar": {
                "path": str(deployment_path),
                "sha256": sha256_file(deployment_path),
            },
        },
        "contract": {
            **contract,
            "experiment_id": definition.experiment_id,
            "encoding": spec.model.encoding,
            "target_mapping": deployment_config.target_mapping,
            "logical_checkpoint_gain": float(fixed_gain),
            "physical_mapper_forward_gain": physical_gain,
            "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            "binding_keys": list(keys),
            "binding_shapes": [list(shape) for shape in shapes],
            "population_sampling_receipt": (
                _json_value(population_receipt)
                if population_receipt is not None
                else None
            ),
            "programming_or_resampling_invoked": False,
            "modifier_constructed": False,
        },
        "cohort": {
            "split": evaluation_split,
            "data_seed": int(spec.runtime.data_seed),
            "validation_points": int(spec.data.validation_points),
            "batch_size": int(spec.data.batch_size),
            "maximum_batches": spec.settings.max_validation_batches,
            "examples": expected_examples,
            "shuffle": False,
            "ordered_inputs_and_labels_sha256": next(iter(cohort_digests)),
            "identical_across_all_states": True,
        },
        "state_construction": state_contract,
        "state_vectors": {
            name: {
                "devices": int(value.numel()),
                "minimum": float(value.min().item()),
                "maximum": float(value.max().item()),
                "sha256": _tensor_sha256(value),
            }
            for name, value in states.items()
        },
        "validation_states": output_states,
        "output_margin_flip_decomposition": {
            "primary_reference": "ideal_mapped_target",
            "secondary_reference": "clean_selected",
            "states": {
                name: {
                    "versus_ideal_mapped_target": value[
                        "versus_ideal_mapped_target"
                    ],
                    "versus_clean_selected": value["versus_clean_selected"],
                }
                for name, value in output_states.items()
            },
        },
        "layers": layer_reports,
        "programming_costs": {
            "bundle_all_devices": _cost_summary(deployment, all_devices),
            "reported_summary": contract["programming_report"],
        },
        "state_restoration": {
            "policy": "context_managed_tensor_copy_restore",
            "all_selected_weights_torch_equal_after_every_state": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            "Expected --output not to overwrite an existing artifact. "
            f"Provided value: {str(output)!r}."
        )
    report = run_decomposition(
        config_path=args.config,
        deployment_config_path=args.deployment_config,
        weights_path=args.weights,
        teacher_weights_path=args.teacher_weights,
        deployment_path=args.deployment,
    )
    atomic_write_json(output, report)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())
