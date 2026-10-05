"""Probes consumed by the shared training and evaluation loops.

The concrete probes use only the public or long-established duck-typed
interfaces of the current network, cost, and minimizer objects.  They retain
data in memory and leave persistence decisions to experiment handlers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
from math import isfinite
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
)

import numpy as np
import torch

if TYPE_CHECKING:
    from training.core.engine import EvaluationBatchEvent


ResultT = TypeVar("ResultT")


class EvaluationProbe(Protocol[ResultT]):
    """Accumulate one named result while a loop traverses its loader.

    ``evaluate`` passes each settled ``EvaluationBatchEvent``; ``train_epoch``
    passes each ``FreePhaseEvent``. Both carry ``batch`` and ``components``.
    """

    @property
    def name(self) -> str:
        """Stable result name."""

    def reset(self) -> None:
        """Clear state before an evaluation pass."""

    def observe(self, event: "EvaluationBatchEvent") -> None:
        """Observe one already-settled minibatch."""

    def result(self) -> ResultT:
        """Return the accumulated result after the loop's contexts exit."""


@dataclass(frozen=True)
class ProbeResult(Generic[ResultT]):
    """The named, typed value produced by one evaluation probe."""

    name: str
    value: ResultT


class PerBatchProbe:
    """Collect ``measure(event)`` once per minibatch, in loader order.

    Reductions stay with the caller, so concatenation and summation order
    are exactly the caller's.
    """

    def __init__(self, name: str, measure: Callable[[Any], Any]) -> None:
        self._name = name
        self._measure = measure
        self._values: List[Any] = []

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self._values = []

    def observe(self, event: Any) -> None:
        self._values.append(self._measure(event))

    def result(self) -> List[Any]:
        return list(self._values)


class MeanValueProbe:
    """Compute an example-weighted mean from one value function per batch."""

    def __init__(
        self,
        name: str,
        value_fn: Callable[["EvaluationBatchEvent"], Any],
    ) -> None:
        _validate_name(name)
        if not callable(value_fn):
            raise ValueError(
                "Expected value_fn to be callable. "
                f"Provided value: {value_fn!r}."
            )
        self._name = name
        self._value_fn = value_fn
        self.reset()

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def observe(self, event: "EvaluationBatchEvent") -> None:
        value = self._value_fn(event)
        amount, count = _sum_and_count(value, source=self.name)
        self._sum += amount
        self._count += count

    def result(self) -> Dict[str, Any]:
        return {
            "mean": self._sum / self._count if self._count else None,
            "sum": self._sum,
            "count": self._count,
        }


class MeanCostProbe(MeanValueProbe):
    """Example-weighted mean of ``cost_fn.eval()``."""

    def __init__(self, name: str = "mean_cost") -> None:
        super().__init__(
            name,
            lambda event: event.components.cost_fn.eval(),
        )


class MeanErrorProbe(MeanValueProbe):
    """Example-weighted mean error fraction from ``cost_fn.error_fn()``."""

    def __init__(self, name: str = "mean_error") -> None:
        super().__init__(
            name,
            lambda event: event.components.cost_fn.error_fn(),
        )


class SettledLayerStatesProbe:
    """Collect detached CPU snapshots of every settled network layer."""

    def __init__(self, name: str = "settled_layer_states") -> None:
        _validate_name(name)
        self._name = name
        self.reset()

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self._states: Dict[str, List[Any]] = {}
        self._layer_order: Optional[tuple] = None

    def observe(self, event: "EvaluationBatchEvent") -> None:
        layers_fn = getattr(event.components.network, "layers", None)
        if not callable(layers_fn):
            raise ValueError(
                "Expected the evaluation network to expose layers(). "
                f"Provided value: {event.components.network!r}."
            )
        layers = tuple(layers_fn())
        names = tuple(_layer_name(layer) for layer in layers)
        _require_stable_names(
            previous=self._layer_order,
            current=names,
            source="settled network layers",
        )
        if self._layer_order is None:
            self._layer_order = names
            self._states = {name: [] for name in names}

        for name, layer in zip(names, layers):
            if not hasattr(layer, "state"):
                raise ValueError(
                    "Expected every settled network layer to expose state. "
                    f"Provided value: layer={layer!r}."
                )
            self._states[name].append(_snapshot(layer.state))

    def result(self) -> Dict[str, Any]:
        return {
            name: _concatenate_snapshots(states)
            for name, states in self._states.items()
        }

    def summary(self) -> Dict[str, Any]:
        """Return JSON-friendly shape and dtype metadata."""

        return {
            name: _value_summary(value)
            for name, value in self.result().items()
        }


class ResidualInfinityNormProbe:
    """Collect per-layer residual norms, preferring minimizer diagnostics."""

    def __init__(self, name: str = "residual_inf_norm") -> None:
        _validate_name(name)
        self._name = name
        self.reset()

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self._residuals: Dict[str, List[float]] = {}
        self._layer_order: Optional[tuple] = None

    def observe(self, event: "EvaluationBatchEvent") -> None:
        residuals = _residuals_for_event(event)
        names = tuple(residuals)
        _require_stable_names(
            previous=self._layer_order,
            current=names,
            source="residual layers",
        )
        if self._layer_order is None:
            self._layer_order = names
            self._residuals = {name: [] for name in names}
        for name, residual in residuals.items():
            self._residuals[name].append(float(residual))

    def result(self) -> Dict[str, np.ndarray]:
        return {
            name: np.asarray(values, dtype=np.float64).copy()
            for name, values in self._residuals.items()
        }

    def summary(self) -> Dict[str, Any]:
        """Return JSON-friendly per-layer residual summaries."""

        return {
            name: _array_summary(values)
            for name, values in self.result().items()
        }


class SolverIterationCountsProbe:
    """Collect the number of equilibrium iterations used for every sample."""

    def __init__(self, name: str = "solver_iteration_counts") -> None:
        _validate_name(name)
        self._name = name
        self.reset()

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self._counts: List[np.ndarray] = []

    def observe(self, event: "EvaluationBatchEvent") -> None:
        counts = _iteration_counts_for_event(event)
        self._counts.append(counts)

    def result(self) -> np.ndarray:
        if not self._counts:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(self._counts, axis=0).astype(
            np.int64,
            copy=True,
        )

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-friendly aggregate of the collected counts."""

        return _array_summary(self.result())


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(
            "Expected a probe name to be a non-empty string. "
            f"Provided value: {name!r}."
        )


def _sum_and_count(value: Any, *, source: str) -> tuple:
    if value is None:
        raise ValueError(
            f"Expected probe {source!r} to produce numeric values. "
            "Provided value: None."
        )

    numel_fn = getattr(value, "numel", None)
    if callable(numel_fn):
        count = int(numel_fn())
        sum_fn = getattr(value, "sum", None)
        if not callable(sum_fn):
            raise ValueError(
                f"Expected probe {source!r} values to expose sum(). "
                f"Provided value: {value!r}."
            )
        total = float(sum_fn().item())
    else:
        try:
            array = np.asarray(value)
            count = int(array.size)
            total = float(array.sum())
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Expected probe {source!r} to produce numeric values. "
                f"Provided value: {value!r}."
            ) from error

    if count and not isfinite(total):
        raise ValueError(
            f"Expected probe {source!r} to produce finite numeric values. "
            f"Provided value: sum={total!r}."
        )
    return total, count


def _layer_name(layer: Any) -> str:
    name = getattr(layer, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            "Expected every probed layer to have a non-empty string name. "
            f"Provided value: {name!r} on {layer!r}."
        )
    return name


def _require_stable_names(
    *,
    previous: Optional[tuple],
    current: tuple,
    source: str,
) -> None:
    if len(set(current)) != len(current):
        raise ValueError(
            f"Expected {source} to have unique names. "
            f"Provided value: {current!r}."
        )
    if previous is not None and current != previous:
        raise ValueError(
            f"Expected {source} to remain ordered and stable across batches. "
            f"Provided value: previous={previous!r}, current={current!r}."
        )


def _snapshot(value: Any) -> Any:
    detach_fn = getattr(value, "detach", None)
    if callable(detach_fn):
        value = detach_fn()
    cpu_fn = getattr(value, "cpu", None)
    if callable(cpu_fn):
        value = cpu_fn()
    clone_fn = getattr(value, "clone", None)
    if callable(clone_fn):
        return clone_fn()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _concatenate_snapshots(values: List[Any]) -> Any:
    if not values:
        return np.empty((0,), dtype=np.float64)
    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            return torch.stack(values, dim=0)
        return torch.cat(values, dim=0)
    if isinstance(first, np.ndarray):
        if first.ndim == 0:
            return np.stack(values, axis=0)
        return np.concatenate(values, axis=0)
    return tuple(copy.deepcopy(values))


def _residuals_for_event(
    event: "EvaluationBatchEvent",
) -> Dict[str, float]:
    minimizer_method = getattr(
        event.components.energy_minimizer,
        "residual_currents_inf",
        None,
    )
    if callable(minimizer_method):
        raw_residuals = minimizer_method()
        if not isinstance(raw_residuals, Mapping):
            raise ValueError(
                "Expected energy_minimizer.residual_currents_inf() to return "
                f"a mapping. Provided value: {raw_residuals!r}."
            )
        return {
            _mapping_layer_name(name): _scalar_value(
                value,
                source=f"residual for layer {name!r}",
            )
            for name, value in raw_residuals.items()
        }

    network = event.components.network
    function = getattr(network, "_function", None)
    if function is None:
        function = getattr(event.components.energy_minimizer, "_fn", None)
    layers_fn = getattr(function, "layers", None)
    grad_layer_fn = getattr(function, "grad_layer_fn", None)
    if not callable(layers_fn) or not callable(grad_layer_fn):
        raise ValueError(
            "Expected the minimizer to expose residual_currents_inf(), or "
            "the network energy function to expose layers() and "
            f"grad_layer_fn(). Provided value: network={network!r}."
        )

    residuals = {}
    for layer in tuple(layers_fn()):
        name = _layer_name(layer)
        gradient_fn = grad_layer_fn(layer)
        if not callable(gradient_fn):
            raise ValueError(
                "Expected grad_layer_fn(layer) to return a callable. "
                f"Provided value: layer={name!r}, gradient={gradient_fn!r}."
            )
        gradient = gradient_fn()
        residuals[name] = _infinity_norm(
            gradient,
            source=f"gradient for layer {name!r}",
        )
    return residuals


def _mapping_layer_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(
            "Expected residual mappings to use non-empty layer-name strings. "
            f"Provided value: {name!r}."
        )
    return name


def _scalar_value(value: Any, *, source: str) -> float:
    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        value = item_fn()
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Expected {source} to be a scalar number. "
            f"Provided value: {value!r}."
        ) from error


def _infinity_norm(value: Any, *, source: str) -> float:
    abs_fn = getattr(value, "abs", None)
    if callable(abs_fn):
        absolute = abs_fn()
        max_fn = getattr(absolute, "max", None)
        if callable(max_fn):
            return _scalar_value(max_fn(), source=source)
    try:
        array = np.asarray(value)
        if not array.size:
            raise ValueError
        return float(np.max(np.abs(array)))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Expected {source} to be a non-empty numeric tensor or array. "
            f"Provided value: {value!r}."
        ) from error


def _iteration_counts_for_event(
    event: "EvaluationBatchEvent",
) -> np.ndarray:
    minimizer = event.components.energy_minimizer
    raw_counts = None

    sample_iterations_fn = getattr(
        minimizer,
        "equilibrium_sample_iterations",
        None,
    )
    if callable(sample_iterations_fn):
        raw_counts = sample_iterations_fn()

    if raw_counts is None:
        stats_fn = getattr(minimizer, "equilibrium_iteration_stats", None)
        if callable(stats_fn):
            stats = stats_fn(reset=False)
            if not isinstance(stats, Mapping) or "last_iterations" not in stats:
                raise ValueError(
                    "Expected equilibrium_iteration_stats(reset=False) to "
                    "return a mapping containing last_iterations. "
                    f"Provided value: {stats!r}."
                )
            raw_counts = stats["last_iterations"]

    if raw_counts is None:
        raw_counts = getattr(minimizer, "num_iterations", None)
        if callable(raw_counts):
            raw_counts = raw_counts()

    if raw_counts is None:
        raise ValueError(
            "Expected the energy minimizer to expose per-sample iterations, "
            "equilibrium iteration stats, or num_iterations. "
            f"Provided value: {minimizer!r}."
        )

    counts = _one_dimensional_array(raw_counts)
    batch_size = event.batch.example_count
    if counts.size == 1 and batch_size != 1:
        counts = np.full(
            (batch_size,),
            counts.item(),
            dtype=counts.dtype,
        )
    if counts.size != batch_size:
        raise ValueError(
            "Expected solver iteration counts to contain one value per "
            f"example ({batch_size}). Provided value: {counts.size} values."
        )
    if not np.issubdtype(counts.dtype, np.number):
        raise ValueError(
            "Expected solver iteration counts to be numeric. "
            f"Provided value: dtype={counts.dtype}."
        )
    as_float = counts.astype(np.float64, copy=False)
    if (
        not np.isfinite(as_float).all()
        or (as_float < 0).any()
        or not np.equal(as_float, np.floor(as_float)).all()
    ):
        raise ValueError(
            "Expected solver iteration counts to be finite non-negative "
            f"integers. Provided value: {counts!r}."
        )
    return as_float.astype(np.int64, copy=True)


def _one_dimensional_array(value: Any) -> np.ndarray:
    detach_fn = getattr(value, "detach", None)
    if callable(detach_fn):
        value = detach_fn()
    cpu_fn = getattr(value, "cpu", None)
    if callable(cpu_fn):
        value = cpu_fn()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        value = numpy_fn()
    try:
        return np.asarray(value).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Expected solver iteration counts to be tensor- or array-like. "
            f"Provided value: {value!r}."
        ) from error


def _array_summary(values: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(values)
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _value_summary(value: Any) -> Dict[str, Any]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    return {
        "shape": [int(dimension) for dimension in shape]
        if shape is not None
        else None,
        "dtype": str(dtype) if dtype is not None else type(value).__name__,
    }
