"""Derive the frozen full-span FP32 shadow for RESET-relative IBM OM studies."""

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


SCHEMA = "ebl.mnist_relu_drn.ibm_om_reset_relative_initialization"
SCHEMA_VERSION = 1
_EXPECTED_KEYS = ("base.dense_weight.0", "base.dense_weight.1")
_TARGET_FRACTIONS = (1.0, 1.0)
_ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _atomic_deterministic_torch_save(payload: Any, path: Path) -> None:
    """Write a stable torch archive whose internal root is not a temp name."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a full-span four-cell FP32 shadow from one frozen "
            "teacher-mapped bounded DRN checkpoint."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    return parser


def _strict_load(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Expected a readable source named checkpoint: {str(source)!r}."
        )
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported PyTorch
        payload = torch.load(source, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "schema_version", "catalog", "weights", "metadata"}
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


def _selected_mapping(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[float, float]]:
    mapping = metadata.get("mapping")
    selected = mapping.get("selected") if isinstance(mapping, Mapping) else None
    fractions = selected.get("scale_fractions") if isinstance(selected, Mapping) else None
    if (
        not isinstance(mapping, Mapping)
        or not isinstance(selected, Mapping)
        or not isinstance(fractions, (list, tuple))
        or len(fractions) != 2
    ):
        raise ValueError(
            "Expected source mapping metadata with exactly two selected layer fractions."
        )
    normalized = (float(fractions[0]), float(fractions[1]))
    if any(not math.isfinite(value) or value <= 0.0 for value in normalized):
        raise ValueError("Expected positive finite source layer fractions.")
    return deepcopy(dict(mapping)), normalized


def _target_candidate(mapping: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    candidates = mapping.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Expected source mapping candidates for provenance.")
    matches = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if isinstance(candidate, Mapping)
        and tuple(float(value) for value in candidate.get("scale_fractions", ()))
        == _TARGET_FRACTIONS
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one precomputed full-span (1.0, 1.0) mapping candidate."
        )
    return matches[0][0], deepcopy(dict(matches[0][1]))


def derive_full_span_checkpoint(
    *,
    source_path: Path,
    output_path: Path,
    receipt_path: Path,
    expected_source_sha256: str,
    expected_teacher_sha256: str,
) -> dict[str, Any]:
    """Create one no-update full-span shadow and its strict derivation receipt."""

    source = source_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    receipt_destination = receipt_path.expanduser().resolve()
    if output.exists() or receipt_destination.exists():
        raise FileExistsError(
            "Expected fresh full-span output and receipt paths; overwrite is forbidden."
        )
    actual_source_sha256 = sha256_file(source)
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            "Expected frozen source checkpoint SHA-256 to match. Provided "
            f"value: expected={expected_source_sha256!r}, "
            f"actual={actual_source_sha256!r}."
        )
    payload = _strict_load(source)
    metadata = deepcopy(dict(payload["metadata"]))
    if metadata.get("teacher_sha256") != expected_teacher_sha256:
        raise ValueError(
            "Expected source checkpoint to name the frozen ReLU teacher SHA-256."
        )
    if metadata.get("encoding") != "single" or metadata.get(
        "conductance_bounds_s"
    ) != [0.1020408197973068, 1.0]:
        raise ValueError(
            "Expected the frozen single-encoding bounded conductance contract."
        )
    mapping, source_fractions = _selected_mapping(metadata)
    target_index, target_candidate = _target_candidate(mapping)
    lower, upper = (float(value) for value in metadata["conductance_bounds_s"])
    span = upper - lower
    transformed: dict[str, torch.Tensor] = {}
    layer_reports = []
    for layer_index, (key, source_fraction) in enumerate(
        zip(_EXPECTED_KEYS, source_fractions)
    ):
        value = payload["weights"][key]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not bool(torch.all(torch.isfinite(value)))
        ):
            raise ValueError(f"Expected source tensor {key!r} to be finite CPU FP32.")
        source_q = (value - lower) / span
        full_q_unclamped = source_q / source_fraction
        outside = torch.maximum(
            (-full_q_unclamped).clamp_min(0.0),
            (full_q_unclamped - 1.0).clamp_min(0.0),
        )
        if float(outside.max().item()) > 2e-6:
            raise ValueError(
                f"Expected source layer {layer_index} to expand to [0, 1] "
                "without material clipping."
            )
        full_q = full_q_unclamped.clamp(0.0, 1.0)
        transformed[key] = (lower + span * full_q).to(torch.float32)
        layer_reports.append(
            {
                "layer": layer_index,
                "key": key,
                "source_fraction": source_fraction,
                "target_fraction": 1.0,
                "source_normalized_minimum": float(source_q.min().item()),
                "source_normalized_maximum": float(source_q.max().item()),
                "full_span_normalized_minimum": float(full_q.min().item()),
                "full_span_normalized_maximum": float(full_q.max().item()),
                "maximum_clamp_correction": float(outside.max().item()),
            }
        )
    target_calibration = target_candidate.get("calibration")
    if not isinstance(target_calibration, Mapping):
        raise ValueError("Expected full-span candidate calibration metadata.")
    target_gain = float(target_calibration.get("gain"))
    if not math.isfinite(target_gain) or target_gain <= 0.0:
        raise ValueError("Expected a positive full-span candidate output gain.")
    mapping["selected_index"] = target_index
    mapping["selected"] = target_candidate
    mapping["selection_domain"] = "frozen_precomputed_nominal_candidate"
    derivation = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy": "expand_selected_lower_placed_layer_fraction_to_one",
        "source_path": _portable_path(source),
        "source_sha256": actual_source_sha256,
        "teacher_sha256": expected_teacher_sha256,
        "source_scale_fractions": list(source_fractions),
        "target_scale_fractions": list(_TARGET_FRACTIONS),
        "optimizer_updates": 0,
        "layers": layer_reports,
    }
    metadata.update(
        {
            "fixed_logit_gain": target_gain,
            "mapping": mapping,
            "mapping_scale_fraction_pairs": [list(_TARGET_FRACTIONS)],
            "selection_metric": "full_span_derivation.precomputed_calibrated_kl",
            "selection_value": float(target_calibration["calibrated_kl"]),
            "selection_epoch": -1,
            "full_span_derivation": derivation,
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
        "fixed_logit_gain": target_gain,
        "weight_keys": list(_EXPECTED_KEYS),
    }
    atomic_write_json(receipt_destination, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = derive_full_span_checkpoint(
        source_path=args.source,
        output_path=args.output,
        receipt_path=args.receipt,
        expected_source_sha256=args.expected_source_sha256,
        expected_teacher_sha256=args.expected_teacher_sha256,
    )
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
