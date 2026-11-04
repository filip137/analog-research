"""Derive the zero-update DRN shadow for raw-active IBM OM studies.

The source checkpoint stores the same logical four-cell coordinates in the
historical bounded conductance interval.  This derivation changes only the
array-wide coordinate to ``[0, 1]``, recalibrates the clean output gain, and
measures the ideal continuous and seven-level mapper gains on one frozen
development population.  It performs no optimizer update and no pulse
programming.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from ebl.runtime.store import atomic_write_json, sha256_file


SCHEMA = "ebl.mnist_relu_drn.ibm_om_raw_active_initialization"
SCHEMA_VERSION = 1
_EXPECTED_KEYS = ("base.dense_weight.0", "base.dense_weight.1")
_SOURCE_BOUNDS = (0.1020408197973068, 1.0)
_TARGET_BOUNDS = (0.0, 1.0)
_TARGET_FRACTIONS = (1.0, 1.0)
_RAW_MODES = ("continuous", "quantized_7_level")
_LAYOUTS = {
    "base.dense_weight.0": "halves",
    "base.dense_weight.1": "paired",
}
_ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _atomic_deterministic_torch_save(payload: Any, path: Path) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _strict_load(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Expected a readable source checkpoint. Provided value: {str(source)!r}."
        )
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported PyTorch
        payload = torch.load(source, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {"schema", "schema_version", "catalog", "weights", "metadata"}
        or payload.get("schema") != "drn.named-weights"
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("weights"), Mapping)
        or tuple(payload["weights"]) != _EXPECTED_KEYS
        or not isinstance(payload.get("metadata"), Mapping)
    ):
        raise ValueError(
            "Expected the canonical two-tensor single-encoding named checkpoint."
        )
    return payload


def transform_to_raw_active_coordinate(
    weights: Mapping[str, torch.Tensor],
    *,
    source_lower: float = _SOURCE_BOUNDS[0],
    source_upper: float = _SOURCE_BOUNDS[1],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Apply one frozen array-wide affine transform, without cell scaling."""

    if (
        not math.isfinite(source_lower)
        or not math.isfinite(source_upper)
        or not source_lower < source_upper
        or tuple(weights) != _EXPECTED_KEYS
    ):
        raise ValueError("Expected exact named weights and increasing source bounds.")
    span = source_upper - source_lower
    transformed: dict[str, torch.Tensor] = {}
    reports = []
    for layer_index, key in enumerate(_EXPECTED_KEYS):
        value = weights[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not bool(torch.all(torch.isfinite(value)))
        ):
            raise ValueError(f"Expected source tensor {key!r} to be finite CPU FP32.")
        raw = (value - source_lower) / span
        outside = torch.maximum(
            (-raw).clamp_min(0.0),
            (raw - 1.0).clamp_min(0.0),
        )
        maximum_correction = float(outside.max().item())
        if maximum_correction > 2e-6:
            raise ValueError(
                f"Expected source tensor {key!r} to transform into [0, 1] "
                "without material clipping."
            )
        target = raw.clamp(0.0, 1.0).to(torch.float32)
        transformed[key] = target
        reports.append(
            {
                "layer": layer_index,
                "key": key,
                "source_minimum": float(value.min().item()),
                "source_maximum": float(value.max().item()),
                "raw_active_minimum": float(target.min().item()),
                "raw_active_maximum": float(target.max().item()),
                "maximum_clamp_correction": maximum_correction,
            }
        )
    return transformed, reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive and calibrate the zero-update raw-active IBM OM DRN shadow."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--expected-population-sha256", required=True)
    parser.add_argument("--expected-population-fingerprint", required=True)
    return parser


def derive_raw_active_checkpoint(
    *,
    source_path: Path,
    config_path: Path,
    teacher_weights_path: Path,
    population_path: Path,
    output_path: Path,
    receipt_path: Path,
    expected_source_sha256: str,
    expected_teacher_sha256: str,
    expected_population_sha256: str,
    expected_population_fingerprint: str,
) -> dict[str, Any]:
    """Create one no-update raw-active shadow and calibration receipt."""

    from ebl.registry import resolve_experiment_config
    from ebl.runtime.mnist_relu_drn.components import (
        apply_targets,
        build_student_stack,
        collect_calibration,
        fit_positive_logit_gain,
    )
    from ebl.runtime.mnist_relu_drn.runtime import _load_teacher
    from networks.data.mnist import build_mnist_loaders
    from ebl.schema.base import RunMode
    from hardware.ibm_om.constants import IBM_OM_RAW_ACTIVE_COORDINATE_VERSION, IBM_OM_RAW_ACTIVE_D90
    from methods.ibm_om.mappings.raw_active import _raw_active_structural_cell_mask
    from hardware.ibm_om.population import load_om_array_population
    from methods.ibm_om.mappings import map_ibm_reram_array_targets, validate_ibm_reram_target_mapping_preflight

    source = source_path.expanduser().resolve()
    config = config_path.expanduser().resolve()
    teacher_path = teacher_weights_path.expanduser().resolve()
    population_source = population_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    receipt_destination = receipt_path.expanduser().resolve()
    if output.exists() or receipt_destination.exists():
        raise FileExistsError(
            "Expected fresh raw-active output and receipt paths; overwrite is forbidden."
        )
    hashes = {
        "source": sha256_file(source),
        "teacher": sha256_file(teacher_path),
        "population": sha256_file(population_source),
        "config": sha256_file(config),
    }
    expected_hashes = {
        "source": expected_source_sha256,
        "teacher": expected_teacher_sha256,
        "population": expected_population_sha256,
    }
    mismatches = {
        name: {"expected": expected, "actual": hashes[name]}
        for name, expected in expected_hashes.items()
        if hashes[name] != expected
    }
    if mismatches:
        raise ValueError(
            "Expected frozen raw-active derivation input SHA-256 values to match. "
            f"Provided value: {mismatches!r}."
        )

    payload = _strict_load(source)
    metadata = deepcopy(dict(payload["metadata"]))
    if (
        metadata.get("teacher_sha256") != expected_teacher_sha256
        or metadata.get("encoding") != "single"
        or metadata.get("conductance_bounds_s") != list(_SOURCE_BOUNDS)
        or metadata.get("mapping_scale_fraction_pairs")
        != [list(_TARGET_FRACTIONS)]
    ):
        raise ValueError(
            "Expected the frozen full-span single-encoding source contract."
        )
    transformed, layer_reports = transform_to_raw_active_coordinate(
        payload["weights"]
    )

    definition, spec = resolve_experiment_config(config, RunMode.TRAIN)
    if (
        definition.experiment_id != "mnist_relu_drn_kd.v1"
        or spec.model.encoding != "single"
        or (spec.model.conductance_min, spec.model.conductance_max)
        != _TARGET_BOUNDS
        or spec.mapping.scale_fraction_pairs != (_TARGET_FRACTIONS,)
        or spec.settings.weight_modifier.type != "none"
        or spec.settings.selection_weight_modifier.type != "none"
        or spec.settings.num_epochs != 0
    ):
        raise ValueError(
            "Expected a zero-update, modifier-free, [0,1] calibration config."
        )

    stack = build_student_stack(spec, enable_measured=False)
    bindings = tuple(stack.bundle.catalog.trainable)
    if tuple(binding.key for binding in bindings) != _EXPECTED_KEYS:
        raise ValueError("Expected the canonical two trainable DRN tensors.")
    with torch.no_grad():
        for binding in bindings:
            binding.state.copy_(transformed[binding.key].to(binding.state))
    clean_targets = tuple(binding.state.detach().clone() for binding in bindings)

    teacher, _teacher_metadata = _load_teacher(
        teacher_path,
        device=stack.device,
        spec=spec,
    )
    data = build_mnist_loaders(
        spec.data,
        data_seed=spec.runtime.data_seed,
        calibration_examples=spec.mapping.calibration_examples,
        calibration_batch_size=spec.mapping.calibration_batch_size,
    )

    def calibrate() -> dict[str, Any]:
        raw_scores, teacher_logits = collect_calibration(
            stack,
            teacher,
            data.calibration,
        )
        return fit_positive_logit_gain(
            raw_scores,
            teacher_logits,
            gain_min=spec.mapping.logit_gain_min,
            gain_max=spec.mapping.logit_gain_max,
            steps=spec.mapping.logit_gain_steps,
        )

    clean_calibration = calibrate()
    population = load_om_array_population(population_source)
    if (
        population.fingerprint != expected_population_fingerprint
        or population.assignment_seed != 84001
        or population.binding_keys != _EXPECTED_KEYS
        or population.binding_shapes
        != tuple(tuple(binding.state.shape) for binding in bindings)
    ):
        raise ValueError(
            "Expected the frozen assignment-84001 development population and layout."
        )
    population_device = population.to(stack.device)
    global_targets = torch.cat(
        tuple(value.reshape(-1) for value in clean_targets)
    )
    structural, lower, upper = _raw_active_structural_cell_mask(
        population_device,
        _LAYOUTS,
        device=stack.device,
    )
    raw_mode_reports: dict[str, Any] = {}
    mapped_calibrations: dict[str, Any] = {}
    for mode in _RAW_MODES:
        mapped, mapping_report = map_ibm_reram_array_targets(
            global_targets,
            population_device,
            target_mapping="raw_active_p90_quad",
            dual_rail_layout_by_parameter=_LAYOUTS,
            common_window_margin_fraction=0.0,
            raw_active_mode=mode,
            raw_active_unsupported_quad_policy="structural_failure",
        )
        validate_ibm_reram_target_mapping_preflight(mapping_report)
        target_in_support = (
            torch.isfinite(mapped)
            & (mapped >= 0.0)
            & (mapped >= lower)
            & (mapped <= upper)
        )
        accepted = structural & target_in_support
        surrogate = torch.where(accepted, mapped, lower)
        split = []
        offset = 0
        for binding in bindings:
            count = binding.state.numel()
            split.append(
                surrogate[offset : offset + count].reshape(binding.state.shape)
            )
            offset += count
        apply_targets(stack.bundle.catalog, tuple(split))
        mapped_calibrations[mode] = calibrate()
        raw_mode_reports[mode] = {
            "mapping": mapping_report,
            "mapping_only_training_surrogate": {
                "accepted_cells": int(accepted.sum().item()),
                "structural_failure_cells": int((~structural).sum().item()),
                "exact_target_out_of_support_cells": int(
                    (structural & ~target_in_support).sum().item()
                ),
                "unsupported_state": "transformed_cell_reset_bound",
            },
            "calibration": mapped_calibrations[mode],
        }
        apply_targets(stack.bundle.catalog, clean_targets)

    if (
        raw_mode_reports["continuous"]["mapping"]["quad_count"] != 39700
        or raw_mode_reports["continuous"]["mapping"][
            "p90_eligible_quad_count"
        ]
        != 35730
        or raw_mode_reports["quantized_7_level"]["mapping"][
            "p90_eligible_quad_count"
        ]
        != 35730
    ):
        raise ValueError(
            "Expected the frozen development population to reproduce 35,730 "
            "eligible quads out of 39,700."
        )

    derivation = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy": "single_array_wide_affine_transform_no_cell_rescaling",
        "source_path": _portable_path(source),
        "source_sha256": hashes["source"],
        "teacher_path": _portable_path(teacher_path),
        "teacher_sha256": hashes["teacher"],
        "calibration_config_path": _portable_path(config),
        "calibration_config_sha256": hashes["config"],
        "population_path": _portable_path(population_source),
        "population_sha256": hashes["population"],
        "population_fingerprint": population.fingerprint,
        "coordinate_version": IBM_OM_RAW_ACTIVE_COORDINATE_VERSION,
        "source_bounds": list(_SOURCE_BOUNDS),
        "target_bounds": list(_TARGET_BOUNDS),
        "optimizer_updates": 0,
        "cell_specific_rescaling": False,
        "reference_consumed": False,
        "frozen_differential_budget": IBM_OM_RAW_ACTIVE_D90,
        "layers": layer_reports,
        "clean_calibration": clean_calibration,
        "raw_active_modes": raw_mode_reports,
    }
    metadata.update(
        {
            "conductance_bounds_s": list(_TARGET_BOUNDS),
            "fixed_logit_gain": float(clean_calibration["gain"]),
            "mapping_range_placement": "lower",
            "mapping_scale_fraction_pairs": [list(_TARGET_FRACTIONS)],
            "mapping": {
                "selection_domain": "zero_update_raw_active_coordinate",
                "selected_index": 0,
                "selected": {
                    "scale_fractions": list(_TARGET_FRACTIONS),
                    "calibration": clean_calibration,
                    "mapping": {
                        "encoding": "single",
                        "coordinate": IBM_OM_RAW_ACTIVE_COORDINATE_VERSION,
                        "layers": layer_reports,
                    },
                },
                "candidates": [
                    {
                        "scale_fractions": list(_TARGET_FRACTIONS),
                        "calibration": clean_calibration,
                        "mapping": {
                            "encoding": "single",
                            "coordinate": IBM_OM_RAW_ACTIVE_COORDINATE_VERSION,
                            "layers": layer_reports,
                        },
                    }
                ],
                "search_examples": spec.mapping.calibration_examples,
                "raw_active_development_population": raw_mode_reports,
            },
            "selection_metric": "raw_active_derivation.clean_calibrated_kl",
            "selection_value": float(clean_calibration["calibrated_kl"]),
            "selection_epoch": -1,
            "raw_active_derivation": derivation,
        }
    )
    output_payload = {
        "schema": payload["schema"],
        "schema_version": payload["schema_version"],
        "catalog": deepcopy(payload["catalog"]),
        "weights": transformed,
        "metadata": metadata,
    }
    _atomic_deterministic_torch_save(output_payload, output)
    receipt = {
        **derivation,
        "output_path": _portable_path(output),
        "output_sha256": sha256_file(output),
        "fixed_clean_logit_gain": float(clean_calibration["gain"]),
        "mapped_forward_logit_gain_by_mode": {
            mode: float(calibration["gain"])
            for mode, calibration in mapped_calibrations.items()
        },
        "weight_keys": list(_EXPECTED_KEYS),
    }
    atomic_write_json(receipt_destination, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = derive_raw_active_checkpoint(
        source_path=args.source,
        config_path=args.config,
        teacher_weights_path=args.teacher_weights,
        population_path=args.population,
        output_path=args.output,
        receipt_path=args.receipt,
        expected_source_sha256=args.expected_source_sha256,
        expected_teacher_sha256=args.expected_teacher_sha256,
        expected_population_sha256=args.expected_population_sha256,
        expected_population_fingerprint=(
            args.expected_population_fingerprint
        ),
    )
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
