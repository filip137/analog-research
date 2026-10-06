"""Measurements, typed reports and diagnostic artifacts for ``small_drn.v1``."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from training.core.engine import FreePhaseEvent, evaluate
from training.core.modifier import ParameterModifier
from training.core.probes import MeanCostProbe, MeanErrorProbe


@dataclass(frozen=True)
class TrainingEpochReport:
    """Typed, immutable measurements available after one completed epoch.

    Error and accuracy values are fractions in ``[0, 1]``.  Selection fields
    always describe the global clean-validation best known at this boundary,
    including a best restored from a resume checkpoint.
    """

    epoch_index: int
    completed_epochs: int
    global_step: int
    train_examples: int
    train_mean_cost: float | None
    train_error_fraction: float | None
    train_accuracy: float | None
    validation_examples: int
    validation_mean_cost: float
    validation_error_fraction: float
    validation_accuracy: float
    selected_this_epoch: bool
    selected_epoch_index: int
    selected_validation_cost: float
    selected_validation_error_fraction: float | None
    selected_validation_accuracy: float | None


@dataclass(frozen=True)
class TrainingOutcome:
    """Typed completion record for programmatic training callers."""

    run_dir: Path
    result_path: Path
    weights_path: Path
    resume_path: Path
    completed_epochs: int
    global_step: int
    selected_epoch_index: int
    selected_validation_cost: float
    selected_validation_error_fraction: float | None
    selected_validation_accuracy: float | None
    last_epoch_report: TrainingEpochReport | None
    resume_capability: str


TrainingObserver = Callable[[TrainingEpochReport], None]


def epoch_report(
    *,
    epoch: int,
    completed_epoch: int,
    global_step: int,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    improved: bool,
    selected_epoch: int | None,
    selected_cost: float | None,
    selected_error: float | None,
    selected_accuracy: float | None,
) -> TrainingEpochReport:
    if selected_epoch is None or selected_cost is None:
        raise RuntimeError(
            "Expected clean validation to define selection fields before "
            "reporting an epoch. "
            f"Provided value: epoch={selected_epoch!r}, cost={selected_cost!r}."
        )
    validation_cost = validation.get("mean_cost")
    validation_error = validation.get("mean_error")
    validation_accuracy = validation.get("accuracy")
    if (
        validation_cost is None
        or validation_error is None
        or validation_accuracy is None
    ):
        raise RuntimeError(
            "Expected completed clean validation metrics to include cost, "
            "error fraction, and accuracy. "
            f"Provided value: {dict(validation)!r}."
        )
    return TrainingEpochReport(
        epoch_index=epoch,
        completed_epochs=completed_epoch,
        global_step=global_step,
        train_examples=int(train["examples"]),
        train_mean_cost=optional_float(train.get("mean_cost")),
        train_error_fraction=optional_fraction(
            train.get("mean_error"),
            name="training error fraction",
        ),
        train_accuracy=optional_fraction(
            train.get("accuracy"),
            name="training accuracy",
        ),
        validation_examples=int(validation["examples"]),
        validation_mean_cost=float(validation_cost),
        validation_error_fraction=float(validation_error),
        validation_accuracy=float(validation_accuracy),
        selected_this_epoch=improved,
        selected_epoch_index=selected_epoch,
        selected_validation_cost=float(selected_cost),
        selected_validation_error_fraction=selected_error,
        selected_validation_accuracy=selected_accuracy,
    )


def validated_observers(
    observers: Iterable[TrainingObserver],
) -> tuple[TrainingObserver, ...]:
    try:
        resolved = tuple(observers)
    except TypeError as exc:
        raise TypeError(
            "Expected observers to be an iterable of callables accepting one "
            f"TrainingEpochReport. Provided value: {observers!r}."
        ) from exc
    for index, observer in enumerate(resolved):
        if not callable(observer):
            raise TypeError(
                "Expected every observer to be callable with one "
                f"TrainingEpochReport. Provided value: observers[{index}]="
                f"{observer!r}."
            )
    return resolved


def classification_metrics(result: Any) -> dict[str, Any]:
    cost = result.probe_value("mean_cost")
    error = result.probe_value("mean_error")
    mean_error = error["mean"]
    return {
        "examples": result.example_count,
        "batches": result.batch_count,
        "mean_cost": cost["mean"],
        "mean_error": mean_error,
        "accuracy": None if mean_error is None else 1.0 - mean_error,
    }


def held_out_metrics(
    runtime: Any,
    *,
    maximum_batches: int | None,
    modifier: ParameterModifier | None,
    epoch: int,
    split: str,
) -> dict[str, Any]:
    """Evaluate the held-out loader with fresh inputs and summarize it.

    Training runs with ``reset_input=False``, so the layer states this leaves
    behind seed the next training minibatch; callers that must not influence
    training restore ``runtime.runtime_state`` themselves.
    """

    return classification_metrics(
        evaluate(
            runtime.evaluation_components,
            limit_batches(runtime.data.held_out_loader, maximum_batches),
            modifier=modifier,
            probes=(MeanCostProbe(), MeanErrorProbe()),
            epoch=epoch,
            split=split,
            reset_input=True,
        )
    )


@dataclass
class TrainingMetrics:
    """Free-phase cost and error accumulated over one training epoch."""

    cost_sum: float = 0.0
    error_sum: float = 0.0
    count: int = 0

    def __call__(self, event: Any) -> None:
        if not isinstance(event, FreePhaseEvent):
            return
        cost = event.components.cost_fn.eval()
        error = event.components.cost_fn.error_fn()
        count = int(cost.numel())
        self.cost_sum += float(cost.sum().item())
        self.error_sum += float(error.sum().item())
        self.count += count

    def result(self) -> dict[str, Any]:
        return {
            "examples": self.count,
            "mean_cost": self.cost_sum / self.count if self.count else None,
            "mean_error": self.error_sum / self.count if self.count else None,
            "accuracy": (
                1.0 - self.error_sum / self.count if self.count else None
            ),
        }


class BatchCapture:
    """Record every evaluated batch's inputs, labels and optional indices."""

    def __init__(self) -> None:
        self._inputs: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._indices: list[np.ndarray] = []

    def __call__(self, event: Any) -> None:
        self._inputs.append(_as_numpy(event.batch.inputs))
        self._labels.append(_as_numpy(event.batch.targets))
        if event.batch.indices is not None:
            self._indices.append(_as_numpy(event.batch.indices))

    def result(self) -> dict[str, np.ndarray]:
        values = {
            "inputs": _concatenate(self._inputs),
            "labels": _concatenate(self._labels),
        }
        if self._indices:
            values["indices"] = _concatenate(self._indices)
        return values


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def optional_fraction(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    valid = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )
    if not valid:
        raise ValueError(
            f"Expected {name} to be a finite fraction in [0, 1] or null. "
            f"Provided value: {value!r}."
        )
    return float(value)


def limit_batches(
    loader: Iterable[Any],
    maximum: int | None,
) -> Iterable[Any]:
    return loader if maximum is None else islice(loader, maximum)


def limit_examples(
    loader: Iterable[Any],
    maximum: int | None,
) -> Iterable[Any]:
    if maximum is None:
        return loader

    def limited():
        remaining = maximum
        for raw_batch in loader:
            if remaining <= 0:
                return
            if not isinstance(raw_batch, (tuple, list)) or len(raw_batch) not in (
                2,
                3,
            ):
                raise ValueError(
                    "Expected validation loader batches to be pairs or "
                    f"triples. Provided value: {raw_batch!r}."
                )
            batch_size = int(raw_batch[0].shape[0])
            take = min(batch_size, remaining)
            if take == batch_size:
                yield raw_batch
            else:
                yield tuple(item[:take] for item in raw_batch)
            remaining -= take

    return limited()


def residual_metrics(
    values: Mapping[str, np.ndarray],
    *,
    tolerance: float | None,
) -> dict[str, Any]:
    per_layer = {
        name: numeric_summary(array, tolerance=tolerance)
        for name, array in values.items()
    }
    # The first ordered layer is the clamped input and is not an equilibrium
    # residual.  Do not rely on legacy process-global layer-name counters.
    comparable = [
        np.asarray(array).reshape(-1)
        for index, array in enumerate(values.values())
        if index != 0 and np.asarray(array).size
    ]
    overall = (
        numeric_summary(
            np.concatenate(comparable),
            tolerance=tolerance,
        )
        if comparable
        else None
    )
    return {"layers": per_layer, "overall": overall}


def numeric_summary(
    value: Any,
    *,
    tolerance: float | None = None,
) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size and not np.isfinite(array).all():
        raise FloatingPointError(
            "Expected diagnostic values to be finite. "
            f"Provided value: {array!r}."
        )
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    result: dict[str, Any] = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
    }
    if tolerance is not None:
        result.update(
            {
                "tolerance": float(tolerance),
                "pass_rate": float(np.mean(array <= tolerance)),
                "pass_all": bool(np.all(array <= tolerance)),
            }
        )
    return result


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def numpy_mapping(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {name: _as_numpy(value) for name, value in values.items()}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _concatenate(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(values, axis=0)
