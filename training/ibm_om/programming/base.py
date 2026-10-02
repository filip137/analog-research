"""Inputs and outputs of the IBM OM programming layer.

Programming receives per-cell targets in a declared device coordinate plus
the plain data a mapping hands over (``ProgramRequest``).  It never sees
which mapping produced them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from training.ibm_om.coordinates import DeviceCoordinate
from training.ibm_om.population import IbmReramArrayPopulation
from training.ibm_reram_program_verify import ProgramVerifyResult


@dataclass(frozen=True)
class OutOfSupportContract:
    """How programming must treat non-corrupt targets outside cell support.

    ``mode`` is ``reject`` (no exact fallback; the compact sampler applies
    its own out-of-support policy) or ``pulse_fallback`` (exactly the
    expected out-of-support cells are programmed pulse-resolved).  The
    expected counts, messages and labels come from the mapping verbatim so
    reports and failures stay byte-identical.
    """

    mode: str
    expected_below: Any
    expected_above: Any
    expected_nonempty_below: Any
    expected_nonempty_above: Any
    require_nonempty_zero: bool
    missing_audit_message: str | None
    mismatch_message: str
    fallback_policy: str
    execution_detail: str
    endpoint_generation_policy: str

    def check(self, *, below: int, above: int) -> None:
        """Fail closed unless the observed out-of-support set is expected."""

        if self.missing_audit_message is not None:
            raise RuntimeError(self.missing_audit_message)
        if self.mode != "pulse_fallback":
            return
        if (
            self.expected_below != below
            or self.expected_above != above
            or (
                self.require_nonempty_zero
                and (
                    self.expected_nonempty_below != 0
                    or self.expected_nonempty_above != 0
                )
            )
        ):
            raise RuntimeError(self.mismatch_message)


@dataclass(frozen=True)
class ProgramRequest:
    """Per-cell targets plus the mapping's programming contract."""

    targets: torch.Tensor
    program_mask: torch.Tensor | None
    out_of_support: OutOfSupportContract


@dataclass(frozen=True)
class ProgrammingContext:
    """Everything programming needs besides the request and the RNG stream."""

    population: IbmReramArrayPopulation
    coordinate: DeviceCoordinate
    device_model: Mapping[str, Any]
    device_model_sha256: str
    calibration_branch: str
    controller: str
    tolerance_step_ratio: float
    maximum_program_pulses: int
    target_out_of_support: str
    endpoint_policy: str


class ProgrammingResult(NamedTuple):
    """Programmed endpoint plus the execution report and deployment facts."""

    endpoint: torch.Tensor
    report: dict[str, Any]
    deployment: dict[str, Any]


def _result_mapping(result: ProgramVerifyResult) -> dict[str, Any]:
    count = int(result.accepted.numel())
    return {
        "devices": count,
        "accepted": int(result.accepted.sum().item()),
        "success_fraction": float(result.accepted.float().mean().item()),
        "budget_exhausted": int(result.budget_exhausted.sum().item()),
        "nonfinite": int(result.nonfinite.sum().item()),
        "pulse_count": {
            "mean": float(result.total_pulses.float().mean().item()),
            "median": float(result.total_pulses.float().median().item()),
            "maximum": int(result.total_pulses.max().item()),
            "set_mean": float(result.set_count.float().mean().item()),
            "reset_mean": float(result.reset_count.float().mean().item()),
        },
        "verify_count_mean": float(result.verify_count.float().mean().item()),
        "reversal_count_mean": float(result.reversals.float().mean().item()),
    }
