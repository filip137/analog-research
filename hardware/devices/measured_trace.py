"""Measured device cohorts, interpolation and deterministic identity streams."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import torch

@dataclass(frozen=True)
class _TraceDataset:
    source_path: str
    source_sha256: str
    source_trace_count: int
    formed_trace_count: int
    screened_trace_names: tuple[str, ...]
    physical_cell_count: int
    cohort_a_cell_count: int
    cohort_b_cell_count: int
    cohort_a_trace_names: tuple[str, ...]
    cohort_b_trace_names: tuple[str, ...]
    active_cohort: str
    active_trace_names: tuple[str, ...]
    conductance_s: np.ndarray

@dataclass
class _ProjectionTable:
    key: str
    shape: tuple[int, ...]
    source_left: torch.Tensor
    source_right: torch.Tensor
    alpha: torch.Tensor
    assignment_sha256: str
    sorted_values: torch.Tensor | None
    sorted_pulses: torch.Tensor | None

def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _pava_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Least-squares non-increasing fit with equal observation weights."""

    source = np.asarray(values, dtype=np.float64)
    count = int(source.size)
    levels = np.empty(count, dtype=np.float64)
    weights = np.empty(count, dtype=np.int64)
    starts = np.empty(count, dtype=np.int64)
    blocks = 0
    # Negate so the standard non-decreasing PAVA inequality applies.
    for index, raw in enumerate(-source):
        levels[blocks] = raw
        weights[blocks] = 1
        starts[blocks] = index
        blocks += 1
        while blocks >= 2 and levels[blocks - 2] > levels[blocks - 1]:
            total_weight = weights[blocks - 2] + weights[blocks - 1]
            levels[blocks - 2] = (
                levels[blocks - 2] * weights[blocks - 2]
                + levels[blocks - 1] * weights[blocks - 1]
            ) / total_weight
            weights[blocks - 2] = total_weight
            blocks -= 1
    fitted = np.empty(count, dtype=np.float64)
    for block in range(blocks):
        start = int(starts[block])
        stop = count if block + 1 == blocks else int(starts[block + 1])
        fitted[start:stop] = -levels[block]
    return fitted.astype(np.float32)

def _load_trace_dataset(
    path: Path,
    config: MeasuredTraceConfig,
) -> _TraceDataset:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            "Expected --device-data to name an existing HDF5 file. "
            f"Provided value: {str(source)!r}."
        )
    records: dict[str, tuple[tuple[int, int], np.ndarray]] = {}
    screened: list[str] = []
    with h5py.File(source, "r") as handle:
        for name in sorted(handle.keys()):
            dataset = handle[name]
            resistance = np.asarray(dataset, dtype=np.float32)
            if resistance.shape != (config.expected_trace_length,):
                raise ValueError(
                    "Expected every measured resistance trace to have shape "
                    f"({config.expected_trace_length},). Provided value: "
                    f"trace={name!r}, shape={resistance.shape!r}."
                )
            if not np.isfinite(resistance).all() or np.any(resistance <= 0.0):
                raise ValueError(
                    "Expected every measured resistance to be positive and "
                    f"finite. Provided value: trace={name!r}."
                )
            if float(resistance[0]) > config.formed_resistance_max_ohm:
                screened.append(name)
                continue
            if "row" not in dataset.attrs or "col" not in dataset.attrs:
                raise ValueError(
                    "Expected every measured trace to define integer row and "
                    f"col attributes. Provided value: trace={name!r}."
                )
            cell = (int(dataset.attrs["row"]), int(dataset.attrs["col"]))
            conductance = np.reciprocal(resistance, dtype=np.float32)
            if config.curve_preprocessing == "isotonic_nonincreasing":
                conductance = _pava_nonincreasing(conductance)
            records[name] = (cell, conductance)

    groups: dict[tuple[int, int], list[str]] = {}
    for name, (cell, _) in records.items():
        groups.setdefault(cell, []).append(name)
    ordered_cells = sorted(groups)
    if len(ordered_cells) < 2:
        raise ValueError(
            "Expected at least two formed physical cells before the cohort "
            f"split. Provided value: {len(ordered_cells)}."
        )
    generator = np.random.default_rng(config.split_seed)
    permutation = generator.permutation(len(ordered_cells))
    cohort_a_count = int(len(ordered_cells) * config.cohort_fraction)
    cohort_a_cells = {
        ordered_cells[int(index)] for index in permutation[:cohort_a_count]
    }
    cohort_b_cells = set(ordered_cells) - cohort_a_cells

    def trace_names(cells: set[tuple[int, int]]) -> tuple[str, ...]:
        return tuple(
            name
            for cell in sorted(cells)
            for name in sorted(groups[cell])
        )

    cohort_a_names = trace_names(cohort_a_cells)
    cohort_b_names = trace_names(cohort_b_cells)
    active_names = (
        cohort_a_names if config.cohort == "A" else cohort_b_names
    )
    if len(active_names) < config.source_traces_per_cell:
        raise ValueError(
            f"Expected cohort {config.cohort} to contain at least two formed "
            "traces for virtual interpolation. Provided value: "
            f"{len(active_names)}."
        )
    conductance = np.stack(
        [records[name][1] for name in active_names],
        axis=0,
    ).astype(np.float32, copy=False)
    return _TraceDataset(
        source_path=str(source),
        source_sha256=_sha256_file(source),
        source_trace_count=len(records) + len(screened),
        formed_trace_count=len(records),
        screened_trace_names=tuple(screened),
        physical_cell_count=len(ordered_cells),
        cohort_a_cell_count=len(cohort_a_cells),
        cohort_b_cell_count=len(cohort_b_cells),
        cohort_a_trace_names=cohort_a_names,
        cohort_b_trace_names=cohort_b_names,
        active_cohort=config.cohort,
        active_trace_names=active_names,
        conductance_s=conductance,
    )

def _assignment_seed(seed: int, key: str, cohort: str = "A") -> int:
    digest = sha256(
        f"measured-cohort-{cohort.lower()}:{seed}:{key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)

def _programming_seed(seed: int, key: str) -> int:
    digest = sha256(
        f"measured-cohort-b-programming:{seed}:{key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63)

def _assignment_digest(
    left: np.ndarray,
    right: np.ndarray,
    alpha: np.ndarray,
) -> str:
    digest = sha256()
    digest.update(left.astype("<i4", copy=False).tobytes())
    digest.update(right.astype("<i4", copy=False).tobytes())
    digest.update(alpha.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()
