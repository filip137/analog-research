"""methods.programming.om — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence
import torch
from hardware.devices.om import VerifyPort


@dataclass(frozen=True)
class ControllerSettings:
    kind: str
    eta: float = 0.75
    maximum_batch: int = 32
    epsilon: float = 1e-8
    force_one_within_steps: float = 2.0

    def __post_init__(self) -> None:
        if self.kind not in {"one_pulse", "adaptive"}:
            raise ValueError(
                "Expected controller kind to be 'one_pulse' or 'adaptive'. "
                f"Provided value: {self.kind!r}."
            )
        if not math.isfinite(self.eta) or self.eta <= 0.0 or self.eta > 1.0:
            raise ValueError(
                "Expected eta to be finite and in (0, 1]. "
                f"Provided value: {self.eta!r}."
            )
        if self.maximum_batch < 1:
            raise ValueError(
                "Expected maximum_batch to be positive. "
                f"Provided value: {self.maximum_batch!r}."
            )
        if self.epsilon <= 0.0 or self.force_one_within_steps <= 0.0:
            raise ValueError(
                "Expected epsilon and force_one_within_steps to be positive."
            )


class PopulationStepEstimator:
    """Direction/state-binned one-pulse response learned from calibration."""

    def __init__(self, *, bins: int, fallback_step: float) -> None:
        if bins < 2 or fallback_step <= 0.0:
            raise ValueError("Expected at least two bins and a positive fallback step.")
        self.bins = int(bins)
        self.fallback_step = float(fallback_step)
        self._sum = torch.zeros((2, bins), dtype=torch.float64)
        self._count = torch.zeros((2, bins), dtype=torch.int64)

    def _ensure_device(self, device: torch.device) -> None:
        if self._sum.device != device:
            self._sum = self._sum.to(device)
            self._count = self._count.to(device)

    def _indices(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            torch.floor(state * self.bins).to(torch.int64),
            min=0,
            max=self.bins - 1,
        )

    def observe(
        self,
        *,
        before: torch.Tensor,
        after: torch.Tensor,
        direction: torch.Tensor,
        pulse_count: torch.Tensor,
        include: torch.Tensor | None = None,
    ) -> None:
        self._ensure_device(before.device)
        valid = pulse_count > 0
        if include is not None:
            valid &= include.to(device=before.device, dtype=torch.bool)
        valid &= torch.isfinite(before) & torch.isfinite(after)
        valid &= direction != 0
        response = torch.abs(after - before) / pulse_count.clamp_min(1)
        state_bin = self._indices(before)
        row = (direction > 0).to(torch.int64)
        flat_index = row * self.bins + state_bin
        selected = flat_index[valid]
        self._sum.view(-1).scatter_add_(0, selected, response[valid].double())
        self._count.view(-1).scatter_add_(
            0,
            selected,
            torch.ones_like(selected, dtype=torch.int64),
        )

    def estimate(self, state: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        self._ensure_device(state.device)
        state_bin = self._indices(state)
        result = torch.full_like(state, self.fallback_step, dtype=torch.float32)
        for sign, row in ((-1, 0), (1, 1)):
            selected = direction == sign
            if not bool(torch.any(selected)):
                continue
            bins = state_bin[selected]
            sums = self._sum[row, bins]
            counts = self._count[row, bins]
            values = torch.where(
                counts > 0,
                sums / counts.clamp_min(1),
                torch.full_like(sums, self.fallback_step),
            )
            result[selected] = values.float()
        return result

    def to_mapping(self) -> dict[str, object]:
        values = torch.where(
            self._count > 0,
            self._sum / self._count.clamp_min(1),
            torch.full_like(self._sum, self.fallback_step),
        )
        return {
            "schema": "ebl.ibm_reram.population_step_estimator",
            "schema_version": 1,
            "bins": self.bins,
            "fallback_step": self.fallback_step,
            "state_bin_edges": [index / self.bins for index in range(self.bins + 1)],
            "directions": {
                "reset_down": values[0].detach().cpu().tolist(),
                "set_up": values[1].detach().cpu().tolist(),
            },
            "counts": {
                "reset_down": self._count[0].detach().cpu().tolist(),
                "set_up": self._count[1].detach().cpu().tolist(),
            },
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PopulationStepEstimator":
        """Restore a calibrated estimator from its strict JSON artifact form."""

        expected = {
            "schema",
            "schema_version",
            "bins",
            "fallback_step",
            "state_bin_edges",
            "directions",
            "counts",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(
                "Expected an IBM ReRAM population-step estimator mapping with "
                f"exact keys {sorted(expected)!r}. Provided value: {value!r}."
            )
        if (
            value["schema"] != "ebl.ibm_reram.population_step_estimator"
            or value["schema_version"] != 1
        ):
            raise ValueError(
                "Expected IBM ReRAM population-step estimator schema version 1."
            )
        bins = value["bins"]
        fallback = value["fallback_step"]
        if isinstance(bins, bool) or not isinstance(bins, int):
            raise ValueError("Expected estimator bins to be an integer.")
        estimator = cls(bins=bins, fallback_step=float(fallback))
        edges = value["state_bin_edges"]
        expected_edges = [index / bins for index in range(bins + 1)]
        if not isinstance(edges, Sequence) or len(edges) != bins + 1 or any(
            not math.isclose(float(provided), expected_value, abs_tol=1e-12)
            for provided, expected_value in zip(edges, expected_edges)
        ):
            raise ValueError(
                "Expected estimator state-bin edges to match its declared bin count."
            )
        directions = value["directions"]
        counts = value["counts"]
        names = ("reset_down", "set_up")
        if (
            not isinstance(directions, Mapping)
            or not isinstance(counts, Mapping)
            or set(directions) != set(names)
            or set(counts) != set(names)
        ):
            raise ValueError(
                "Expected estimator directions and counts for reset_down and set_up."
            )
        rows = []
        count_rows = []
        for name in names:
            row = torch.as_tensor(directions[name], dtype=torch.float64)
            count = torch.as_tensor(counts[name], dtype=torch.int64)
            if (
                row.shape != (bins,)
                or count.shape != (bins,)
                or not bool(torch.all(torch.isfinite(row)))
                or bool(torch.any(row <= 0.0))
                or bool(torch.any(count < 0))
            ):
                raise ValueError(
                    "Expected finite positive estimator steps and non-negative "
                    f"counts with length {bins}."
                )
            rows.append(row)
            count_rows.append(count)
        estimator._count = torch.stack(count_rows)
        estimator._sum = torch.stack(rows) * estimator._count
        return estimator


@dataclass(frozen=True)
class VerifyObservation:
    verify_index: int
    apparent: torch.Tensor
    direction: torch.Tensor
    pulse_count: torch.Tensor
    total_pulses: torch.Tensor


@dataclass(frozen=True)
class ProgramVerifyResult:
    accepted: torch.Tensor
    nonfinite: torch.Tensor
    budget_exhausted: torch.Tensor
    apparent_endpoint: torch.Tensor
    set_count: torch.Tensor
    reset_count: torch.Tensor
    total_pulses: torch.Tensor
    verify_count: torch.Tensor
    reversals: torch.Tensor


def run_program_verify(
    port: VerifyPort,
    *,
    targets: torch.Tensor,
    tolerance: float,
    maximum_pulses: int,
    settings: ControllerSettings,
    estimator: PopulationStepEstimator | None = None,
    eligible: torch.Tensor | None = None,
    observer: Callable[[VerifyObservation], None] | None = None,
) -> ProgramVerifyResult:
    """Run one-pulse or adaptive P&V using apparent-state feedback only."""

    apparent = port.verify()
    execution_device = apparent.device
    target = torch.as_tensor(
        targets, dtype=torch.float32, device=execution_device
    )
    if target.shape != (port.size,):
        raise ValueError(
            f"Expected targets to have shape {(port.size,)!r}. "
            f"Provided value: {tuple(target.shape)!r}."
        )
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError(
            "Expected tolerance to be a positive finite number. "
            f"Provided value: {tolerance!r}."
        )
    if maximum_pulses < 1:
        raise ValueError(
            "Expected maximum_pulses to be positive. "
            f"Provided value: {maximum_pulses!r}."
        )
    if settings.kind == "adaptive" and estimator is None:
        raise ValueError("Expected the adaptive controller to receive a calibrated estimator.")

    allowed = (
        torch.ones(port.size, dtype=torch.bool, device=execution_device)
        if eligible is None
        else torch.as_tensor(
            eligible, dtype=torch.bool, device=execution_device
        ).clone()
    )
    if allowed.shape != (port.size,):
        raise ValueError("Expected eligible to match the controller population.")
    set_count = torch.zeros(
        port.size, dtype=torch.int64, device=execution_device
    )
    reset_count = torch.zeros_like(set_count)
    total = torch.zeros_like(set_count)
    verifies = allowed.to(torch.int64)
    reversals = torch.zeros_like(set_count)
    previous_direction = torch.zeros(
        port.size, dtype=torch.int8, device=execution_device
    )
    nonfinite = torch.zeros_like(allowed)
    accepted = torch.zeros_like(allowed)
    adaptive_steps = torch.full(
        (2, port.size),
        float("nan"),
        dtype=torch.float32,
        device=execution_device,
    )

    initial_direction = torch.zeros_like(previous_direction)
    initial_count = torch.zeros_like(total)
    if observer is not None:
        observer(VerifyObservation(0, apparent.clone(), initial_direction, initial_count, total.clone()))

    verify_index = 0
    while True:
        finite = torch.isfinite(apparent)
        residual = apparent - target
        nonfinite |= allowed & ~finite
        accepted |= allowed & finite & (torch.abs(residual) <= tolerance)
        active = allowed & finite & ~accepted & (total < maximum_pulses)
        if not bool(torch.any(active)):
            break

        direction = torch.zeros_like(previous_direction)
        direction[active & (residual < 0.0)] = 1
        direction[active & (residual > 0.0)] = -1
        if bool(torch.any(active & (direction == 0))):  # pragma: no cover - defensive
            raise RuntimeError("Expected every active trajectory to require a pulse direction.")

        changed = active & (previous_direction != 0) & (direction != previous_direction)
        reversals[changed] += 1
        previous_direction[active] = direction[active]

        count = torch.zeros_like(total)
        if settings.kind == "one_pulse":
            count[active] = 1
        else:
            assert estimator is not None
            row = (direction > 0).to(torch.int64)
            uninitialized = active & ~torch.isfinite(
                adaptive_steps[
                    row,
                    torch.arange(port.size, device=execution_device),
                ]
            )
            if bool(torch.any(uninitialized)):
                initial = estimator.estimate(apparent, direction)
                indices = torch.nonzero(uninitialized, as_tuple=False).reshape(-1)
                adaptive_steps[row[indices], indices] = initial[indices]
            step = adaptive_steps[
                row,
                torch.arange(port.size, device=execution_device),
            ].clamp_min(settings.epsilon)
            distance = torch.abs(apparent - target)
            predicted = torch.floor(settings.eta * distance / step).to(torch.int64)
            predicted.clamp_(min=1, max=settings.maximum_batch)
            predicted = torch.where(
                distance <= settings.force_one_within_steps * step,
                torch.ones_like(predicted),
                predicted,
            )
            count[active] = predicted[active]

        remaining = maximum_pulses - total
        count = torch.minimum(count, remaining)
        before = apparent.clone()
        port.apply_identical_pulses(direction, count)
        apparent = port.verify()
        total += count
        set_count += torch.where(direction > 0, count, torch.zeros_like(count))
        reset_count += torch.where(direction < 0, count, torch.zeros_like(count))
        verifies[count > 0] += 1
        verify_index += 1

        if settings.kind == "adaptive":
            row = (direction > 0).to(torch.int64)
            observed = torch.abs(apparent - before) / count.clamp_min(1)
            updated = active & torch.isfinite(observed) & (count > 0)
            indices = torch.nonzero(updated, as_tuple=False).reshape(-1)
            adaptive_steps[row[indices], indices] = observed[indices].clamp_min(
                settings.epsilon
            )
        if observer is not None:
            observer(
                VerifyObservation(
                    verify_index,
                    apparent.clone(),
                    direction.clone(),
                    count.clone(),
                    total.clone(),
                )
            )

    budget_exhausted = allowed & ~accepted & ~nonfinite & (total >= maximum_pulses)
    return ProgramVerifyResult(
        accepted=accepted,
        nonfinite=nonfinite,
        budget_exhausted=budget_exhausted,
        apparent_endpoint=apparent.clone(),
        set_count=set_count,
        reset_count=reset_count,
        total_pulses=total,
        verify_count=verifies,
        reversals=reversals,
    )
