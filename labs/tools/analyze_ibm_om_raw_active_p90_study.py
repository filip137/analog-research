#!/usr/bin/env python3
"""Aggregate the raw-active p90 MNIST deployment/decomposition study.

The analysis is read-only with respect to checkpoints and deployment bundles.
It verifies the declared coverage, reuses the saved endpoint tensors without
programming or resampling, and writes a machine-readable report plus a compact
evidence-only Markdown summary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ebl.runtime.store import atomic_write_json, sha256_file
from ebl.runtime.mnist_relu_drn.ibm_om_deployment_decomposition import (
    canonical_layer_geometries,
    logical_differential_contrast,
)


SCHEMA = "ebl.ibm_om_raw_active_p90.primary_analysis"
SCHEMA_VERSION = 1
EXPECTED_STUDY_ID = (
    "mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1"
)
EXPECTED_ENDPOINT_SEEDS = (85101, 85102, 85103, 85104, 85105)
EXPECTED_POPULATION_SHA256 = (
    "8cf02438727c4090c4721976e09eee066e2408dad9c1c8cd8328ef3268965cab"
)
EXPECTED_POPULATION_FINGERPRINT = (
    "2721e735ba6722df27a879a5373de2d1da02481cb5eec05b592f967d27e74310"
)
EXPECTED_P90_ELIGIBLE_QUADS = 35898

_ARMS = (
    {
        "arm": "deploy-zero-update-heldout",
        "pipeline": "zero_update_quantized",
        "train_arm": "train-zero-update-quantized",
        "label": "Zero-update quantized",
    },
    {
        "arm": "deploy-clean-bptt-heldout",
        "pipeline": "clean_bptt_quantized_deploy",
        "train_arm": "train-clean-bptt-quantized-deploy",
        "label": "Clean BPTT, quantized deploy",
    },
    {
        "arm": "deploy-continuous-hwa-heldout",
        "pipeline": "continuous_mapped_hwa",
        "train_arm": "train-continuous-mapped-hwa",
        "label": "Continuous mapped HWA",
    },
    {
        "arm": "deploy-quantized-qat-heldout",
        "pipeline": "quantized_qat",
        "train_arm": "train-quantized-qat",
        "label": "Seven-level QAT",
    },
)

_STATE_NAMES = (
    "clean_selected",
    "ideal_global_requested",
    "ideal_mapped_target",
    "apparent_endpoint",
    "persistent_endpoint_diagnostic",
    "accepted_endpoint_error_only",
    "failure_endpoint_error_only",
    "apparent_saturation_corrected",
    "w1_endpoint_error_only",
    "w2_endpoint_error_only",
)

_STATE_METRIC_NAMES = (
    "student_accuracy",
    "teacher_agreement",
    "kl_teacher_student",
    "raw_kl_teacher_student",
    "raw_score_rms",
    "calibrated_score_rms",
    "fixed_logit_gain",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Expected at least one value.")
    return float(statistics.fmean(values))


def _sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Expected at least one value for a quantile.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Expected a probability in [0, 1].")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def _series(values_by_seed: Mapping[int, float]) -> dict[str, Any]:
    if tuple(sorted(values_by_seed)) != EXPECTED_ENDPOINT_SEEDS:
        raise ValueError(
            "Expected exactly the five declared endpoint seeds. "
            f"Provided value: {sorted(values_by_seed)!r}."
        )
    values = [float(values_by_seed[seed]) for seed in EXPECTED_ENDPOINT_SEEDS]
    return {
        "by_endpoint_seed": {
            str(seed): float(values_by_seed[seed])
            for seed in EXPECTED_ENDPOINT_SEEDS
        },
        "mean": _mean(values),
        "sample_standard_deviation": _sample_standard_deviation(values),
        "range": [min(values), max(values)],
    }


def _assert_constant(
    values: Sequence[float],
    *,
    name: str,
    absolute_tolerance: float = 1e-12,
) -> float:
    first = float(values[0])
    if any(
        not math.isclose(
            float(value),
            first,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        )
        for value in values[1:]
    ):
        raise ValueError(f"Expected {name} to be invariant: {values!r}.")
    return first


def _sha256_tensor(value: torch.Tensor) -> str:
    payload = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _rms(value: torch.Tensor) -> float:
    selected = value.detach().to(torch.float64).reshape(-1)
    if selected.numel() == 0:
        raise ValueError("Expected a non-empty tensor for RMS.")
    return float(selected.square().mean().sqrt().item())


def _sign_flip_fraction(target: torch.Tensor, observed: torch.Tensor) -> float | None:
    target = target.reshape(-1)
    observed = observed.reshape(-1)
    nonzero = target != 0.0
    count = int(nonzero.sum().item())
    if not count:
        return None
    flips = nonzero & ((target * observed) < 0.0)
    return float(flips.sum().item()) / count


def _snr(signal_rms: float, error_rms: float) -> dict[str, Any]:
    if error_rms == 0.0:
        return {
            "signal_rms": signal_rms,
            "error_rms": error_rms,
            "ratio": None,
            "db": None,
            "status": "infinite_zero_error",
        }
    if signal_rms == 0.0:
        return {
            "signal_rms": signal_rms,
            "error_rms": error_rms,
            "ratio": 0.0,
            "db": None,
            "status": "zero_signal",
        }
    ratio = signal_rms / error_rms
    return {
        "signal_rms": signal_rms,
        "error_rms": error_rms,
        "ratio": ratio,
        "db": 20.0 * math.log10(ratio),
        "status": "finite",
    }


def _quad_mask(mask: torch.Tensor, layer: Any) -> torch.Tensor:
    if len(layer.binding_slices) != 1 or len(layer.binding_shapes) != 1:
        raise ValueError("Expected the raw-active single-encoding layer layout.")
    matrix = mask[layer.binding_slices[0]].reshape(layer.binding_shapes[0])
    rows, columns = matrix.shape
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
    cells = (
        matrix[:logical_rows][:, plus_columns],
        matrix[:logical_rows][:, minus_columns],
        matrix[logical_rows:][:, plus_columns],
        matrix[logical_rows:][:, minus_columns],
    )
    all_selected = cells[0] & cells[1] & cells[2] & cells[3]
    any_selected = cells[0] | cells[1] | cells[2] | cells[3]
    if mask.dtype == torch.bool and not torch.equal(all_selected, any_selected):
        # Structural eligibility is required to be quad-uniform. Acceptance is
        # intentionally allowed to vary cell by cell, so callers requesting an
        # all-four mask pass it through the separate helper below.
        raise ValueError("Expected a quad-uniform device mask.")
    return all_selected


def _all_four_mask(mask: torch.Tensor, layer: Any) -> torch.Tensor:
    if len(layer.binding_slices) != 1 or len(layer.binding_shapes) != 1:
        raise ValueError("Expected the raw-active single-encoding layer layout.")
    matrix = mask[layer.binding_slices[0]].reshape(layer.binding_shapes[0])
    rows, columns = matrix.shape
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
    return (
        matrix[:logical_rows][:, plus_columns]
        & matrix[:logical_rows][:, minus_columns]
        & matrix[logical_rows:][:, plus_columns]
        & matrix[logical_rows:][:, minus_columns]
    )


def _layer_geometries(bundle: Mapping[str, Any]) -> tuple[Any, ...]:
    keys = tuple(bundle["binding_keys"])
    shapes = tuple(tuple(int(item) for item in shape) for shape in bundle["binding_shapes"])
    sections = []
    offset = 0
    for shape in shapes:
        count = math.prod(shape)
        sections.append(slice(offset, offset + count))
        offset += count
    layouts = dict(bundle["config"]["dual_rail_layout_by_parameter"])
    return canonical_layer_geometries(
        keys,
        shapes,
        tuple(sections),
        encoding="single",
        dual_rail_layout_by_parameter=layouts,
    )


def _differential(value: torch.Tensor, layer: Any) -> torch.Tensor:
    return logical_differential_contrast(
        value,
        layer,
        conductance_min=0.0,
        conductance_max=1.0,
    ) / 2.0


def _subset_metrics(
    target: torch.Tensor,
    observed: torch.Tensor,
    selection: torch.Tensor,
) -> dict[str, Any]:
    target_selected = target[selection]
    observed_selected = observed[selection]
    if target_selected.numel() == 0:
        raise ValueError("Expected at least one selected logical quad.")
    error = observed_selected - target_selected
    signal_rms = _rms(target_selected)
    error_rms = _rms(error)
    absolute_nonzero = target_selected.abs()[target_selected != 0.0]
    return {
        "quad_count": int(target_selected.numel()),
        "nonzero_target_count": int(absolute_nonzero.numel()),
        "minimum_nonzero_target_magnitude": (
            float(absolute_nonzero.min().item())
            if absolute_nonzero.numel()
            else None
        ),
        "target_rms": signal_rms,
        "error_rms": error_rms,
        "snr": _snr(signal_rms, error_rms),
        "sign_flip_fraction_of_nonzero_target": _sign_flip_fraction(
            target_selected,
            observed_selected,
        ),
    }


def _load_bundle(report: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = report["inputs"]["deployment_sidecar"]
    path = Path(descriptor["path"])
    if sha256_file(path) != descriptor["sha256"]:
        raise ValueError(f"Deployment sidecar hash changed: {path}.")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected a deployment dictionary: {path}.")
    return bundle


def _layer_seed_snapshot(
    bundle: Mapping[str, Any],
    layer: Any,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    target = _differential(bundle["requested_target"], layer)
    apparent = _differential(bundle["apparent_endpoint"], layer)
    persistent = _differential(bundle["persistent_endpoint"], layer)
    structural_eligible = _quad_mask(bundle["structural_quad_eligible"], layer)
    structural_failure = ~structural_eligible
    accepted_complete_quad = _all_four_mask(bundle["accepted"], layer)

    accepted_cell_only_state = torch.where(
        bundle["accepted"],
        bundle["apparent_endpoint"],
        bundle["requested_target"],
    )
    failure_cell_only_state = torch.where(
        bundle["accepted"],
        bundle["requested_target"],
        bundle["apparent_endpoint"],
    )
    accepted_cell_only = _differential(accepted_cell_only_state, layer)
    failure_cell_only = _differential(failure_cell_only_state, layer)

    all_quads = torch.ones_like(structural_eligible, dtype=torch.bool)
    snapshot = {
        "target_sha256": _sha256_tensor(target),
        "structural_eligibility_sha256": _sha256_tensor(
            structural_eligible.to(torch.uint8)
        ),
        "all_quads": _subset_metrics(target, apparent, all_quads),
        "eligible_quads": _subset_metrics(
            target,
            apparent,
            structural_eligible,
        ),
        "structural_failure_quads": _subset_metrics(
            target,
            apparent,
            structural_failure,
        ),
        "accepted_complete_quads": _subset_metrics(
            target,
            apparent,
            accepted_complete_quad,
        ),
        "persistent_all_quads": _subset_metrics(
            target,
            persistent,
            all_quads,
        ),
        "persistent_eligible_quads": _subset_metrics(
            target,
            persistent,
            structural_eligible,
        ),
        "persistent_structural_failure_quads": _subset_metrics(
            target,
            persistent,
            structural_failure,
        ),
        "accepted_cell_only_D_error_rms": _rms(accepted_cell_only - target),
        "failure_cell_only_D_error_rms": _rms(failure_cell_only - target),
        "complete_quad_acceptance_count": int(
            accepted_complete_quad.sum().item()
        ),
    }
    return snapshot, target, structural_eligible


def _aggregate_layer_snapshots(
    snapshots: Mapping[int, Mapping[str, Any]],
    *,
    quantized_spacing: float | None,
    target: torch.Tensor,
    structural_eligible: torch.Tensor,
) -> dict[str, Any]:
    target_hashes = {value["target_sha256"] for value in snapshots.values()}
    eligibility_hashes = {
        value["structural_eligibility_sha256"] for value in snapshots.values()
    }
    if len(target_hashes) != 1 or len(eligibility_hashes) != 1:
        raise ValueError("Expected target and structural mask invariance over endpoint seeds.")

    eligible_target = target[structural_eligible]
    nonzero = eligible_target[eligible_target != 0.0]
    histogram = None
    if quantized_spacing is not None:
        codes = torch.round(eligible_target / quantized_spacing).to(torch.int64)
        reconstructed = codes.to(torch.float64) * quantized_spacing
        if not torch.allclose(
            reconstructed,
            eligible_target.to(torch.float64),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError("Eligible quantized target does not match its codebook.")
        histogram = {
            str(code): int((codes == code).sum().item())
            for code in range(-3, 4)
        }

    first = snapshots[EXPECTED_ENDPOINT_SEEDS[0]]
    result: dict[str, Any] = {
        "target_sha256": first["target_sha256"],
        "structural_eligibility_sha256": first[
            "structural_eligibility_sha256"
        ],
        "target": {
            "all_quad_count": int(target.numel()),
            "eligible_quad_count": int(structural_eligible.sum().item()),
            "structural_failure_quad_count": int(
                (~structural_eligible).sum().item()
            ),
            "eligible_nonzero_count": int(nonzero.numel()),
            "eligible_minimum_nonzero_D": (
                float(nonzero.abs().min().item()) if nonzero.numel() else None
            ),
            "eligible_D_rms": _rms(eligible_target),
            "all_quad_D_rms": _rms(target),
            "quantized_spacing_D": quantized_spacing,
            "eligible_code_histogram": histogram,
        },
        "by_endpoint_seed": {
            str(seed): snapshots[seed] for seed in EXPECTED_ENDPOINT_SEEDS
        },
    }

    paths = {
        "apparent_all_quad_D_error_rms": ("all_quads", "error_rms"),
        "apparent_eligible_D_error_rms": ("eligible_quads", "error_rms"),
        "apparent_structural_failure_D_error_rms": (
            "structural_failure_quads",
            "error_rms",
        ),
        "apparent_accepted_complete_quad_D_error_rms": (
            "accepted_complete_quads",
            "error_rms",
        ),
        "apparent_eligible_D_snr_db": (
            "eligible_quads",
            "snr",
            "db",
        ),
        "apparent_eligible_sign_flip_fraction": (
            "eligible_quads",
            "sign_flip_fraction_of_nonzero_target",
        ),
        "persistent_all_quad_D_error_rms": (
            "persistent_all_quads",
            "error_rms",
        ),
        "persistent_eligible_D_error_rms": (
            "persistent_eligible_quads",
            "error_rms",
        ),
        "persistent_structural_failure_D_error_rms": (
            "persistent_structural_failure_quads",
            "error_rms",
        ),
        "persistent_eligible_D_snr_db": (
            "persistent_eligible_quads",
            "snr",
            "db",
        ),
        "persistent_eligible_sign_flip_fraction": (
            "persistent_eligible_quads",
            "sign_flip_fraction_of_nonzero_target",
        ),
        "accepted_cell_only_D_error_rms": (
            "accepted_cell_only_D_error_rms",
        ),
        "failure_cell_only_D_error_rms": (
            "failure_cell_only_D_error_rms",
        ),
        "complete_quad_acceptance_count": (
            "complete_quad_acceptance_count",
        ),
    }

    def read_path(value: Mapping[str, Any], path: Iterable[str]) -> float:
        selected: Any = value
        for key in path:
            selected = selected[key]
        if selected is None:
            raise ValueError(f"Expected a finite aggregate value at {tuple(path)!r}.")
        return float(selected)

    result["aggregate"] = {
        name: _series(
            {
                seed: read_path(snapshots[seed], path)
                for seed in EXPECTED_ENDPOINT_SEEDS
            }
        )
        for name, path in paths.items()
    }
    return result


def _state_summary(
    reports: Mapping[int, Mapping[str, Any]],
    state_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric_name in _STATE_METRIC_NAMES:
        result[metric_name] = _series(
            {
                seed: float(
                    reports[seed]["validation_states"][state_name]["metrics"][
                        metric_name
                    ]
                )
                for seed in EXPECTED_ENDPOINT_SEEDS
            }
        )
    for summary_name, report_key in (
        ("label_margin_mean", ("label_margin", "mean")),
        ("label_margin_rms", ("label_margin", "rms")),
        ("top1_minus_top2_margin_mean", ("top1_minus_top2_margin", "mean")),
        ("top1_minus_top2_margin_rms", ("top1_minus_top2_margin", "rms")),
    ):
        result[summary_name] = _series(
            {
                seed: float(
                    reports[seed]["validation_states"][state_name][report_key[0]][
                        report_key[1]
                    ]
                )
                for seed in EXPECTED_ENDPOINT_SEEDS
            }
        )
    return result


def _programming_summary(
    reports: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    records: dict[int, dict[str, float]] = {}
    for seed in EXPECTED_ENDPOINT_SEEDS:
        costs = reports[seed]["programming_costs"]
        bundle = costs["bundle_all_devices"]
        reported = costs["reported_summary"]
        records[seed] = {
            "devices": float(bundle["devices"]),
            "conditioning_total_pulses": float(
                reported["conditioning_total_pulses"]
            ),
            "conditioning_mean_pulses_per_device": float(
                reported["conditioning_pulse_count"]["mean"]
            ),
            "conditioning_success": float(reported["conditioning_success"]),
            "conditioning_failure": float(reported["conditioning_failure"]),
            "target_set_pulses": float(bundle["set_count"]["sum"]),
            "target_reset_pulses": float(bundle["reset_count"]["sum"]),
            "target_total_pulses": float(bundle["total_pulses"]["sum"]),
            "target_mean_pulses_per_device": float(
                bundle["total_pulses"]["mean"]
            ),
            "target_verify_reads": float(bundle["verify_count"]["sum"]),
            "target_reversals": float(bundle["reversals"]["sum"]),
            "programming_eligible": float(reported["programming_eligible"]),
            "programming_eligible_accepted": float(
                reported["programming_eligible_accepted"]
            ),
            "programming_eligible_success_fraction": float(
                reported["programming_eligible_success_fraction"]
            ),
            "accepted_persistent_within_tolerance": float(
                reported["accepted_persistent_within_tolerance"]
            ),
            "accepted_persistent_outside_tolerance": float(
                reported["accepted_persistent_outside_tolerance"]
            ),
            "budget_exhausted": float(reported["budget_exhausted"]),
            "saturated": float(reported["saturated"]),
            "structural_quad_eligible_cells": float(
                reported["structural_quad_eligible"]
            ),
            "structural_target_assignment_failure_cells": float(
                reported["structural_target_assignment_failure"]
            ),
            "exact_target_in_support": float(reported["exact_target_in_support"]),
            "target_below_lower_bound": float(reported["target_below_lower_bound"]),
            "target_above_upper_bound": float(reported["target_above_upper_bound"]),
            "active_corrupt": float(reported["corrupt"]),
            "published_corrupt": float(reported["published_corrupt"]),
            "nonfinite": float(reported["nonfinite"]),
        }
    return {
        "conditioning_excluded_from_target_cap": True,
        "target_programming_cap_per_cell": 128,
        "tolerance_g": _assert_constant(
            [
                float(
                    reports[seed]["programming_costs"]["reported_summary"][
                        "tolerance"
                    ]
                )
                for seed in EXPECTED_ENDPOINT_SEEDS
            ],
            name="programming tolerance",
        ),
        "by_endpoint_seed": {
            str(seed): records[seed] for seed in EXPECTED_ENDPOINT_SEEDS
        },
        "aggregate": {
            key: _series(
                {seed: record[key] for seed, record in records.items()}
            )
            for key in next(iter(records.values()))
        },
    }


def _mask_overlap(
    development_bundle: Mapping[str, Any],
    heldout_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    development_layers = _layer_geometries(development_bundle)
    heldout_layers = _layer_geometries(heldout_bundle)
    result = {}
    for development_layer, heldout_layer in zip(
        development_layers,
        heldout_layers,
        strict=True,
    ):
        if development_layer.name != heldout_layer.name:
            raise ValueError("Layer geometry changed across assignments.")
        development = _quad_mask(
            development_bundle["structural_quad_eligible"],
            development_layer,
        )
        heldout = _quad_mask(
            heldout_bundle["structural_quad_eligible"],
            heldout_layer,
        )
        if development.shape != heldout.shape:
            raise ValueError("Structural eligibility shape changed across assignments.")
        both_eligible = development & heldout
        development_only = development & ~heldout
        heldout_only = ~development & heldout
        both_failure = ~development & ~heldout
        failure_union = (~development) | (~heldout)
        result[development_layer.name] = {
            "logical_quads": int(development.numel()),
            "development_eligible": int(development.sum().item()),
            "heldout_eligible": int(heldout.sum().item()),
            "both_eligible": int(both_eligible.sum().item()),
            "development_eligible_heldout_failure": int(
                development_only.sum().item()
            ),
            "development_failure_heldout_eligible": int(
                heldout_only.sum().item()
            ),
            "both_structural_failure": int(both_failure.sum().item()),
            "eligibility_agreement_fraction": float(
                (development == heldout).to(torch.float64).mean().item()
            ),
            "structural_failure_jaccard": float(
                both_failure.sum().item() / failure_union.sum().item()
            ),
            "development_mask_sha256": _sha256_tensor(
                development.to(torch.uint8)
            ),
            "heldout_mask_sha256": _sha256_tensor(heldout.to(torch.uint8)),
        }
    return result


def _ordered_manifest_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_protocol_learning_rates(repo_root: Path) -> dict[str, Any]:
    result = {}
    configs = {
        "raw_active_continuous": repo_root
        / "examples/mnist_relu_drn/ibm_om_raw_active_p90_qat/continuous_hwa.json",
        "raw_active_quantized": repo_root
        / "examples/mnist_relu_drn/ibm_om_raw_active_p90_qat/quantized_qat.json",
        "reset_relative_continuous": repo_root
        / "examples/mnist_relu_drn/ibm_om_reset_relative_quantized_hwa/continuous_hwa.json",
        "reset_relative_quantized": repo_root
        / "examples/mnist_relu_drn/ibm_om_reset_relative_quantized_hwa/quantized_qat.json",
    }
    for key, path in configs.items():
        if path.exists():
            config = _load_json(path)
            result[key] = {
                "learning_rates": config["modes"]["train"]["learning_rates"],
                "config_sha256": sha256_file(path),
            }
    return result


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _format_points(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def _render_markdown(report: Mapping[str, Any]) -> str:
    comparison = report["primary_comparison"]
    pipelines = report["pipeline_results"]
    mechanism = report["mechanism"]
    programming = report["programming"]
    overlap = report["assignment_transfer"]["structural_eligibility_overlap"]
    transfer = report["assignment_transfer"]["accuracy_transfer_by_pipeline"]

    lines = [
        "# Evidence-only primary analysis",
        "",
        (
            "The formal study has four valid development runs, all twenty "
            "held-out deployments, and twenty read-only endpoint "
            "decompositions. Each held-out deployment evaluated all 10,000 "
            "test examples. Artifact verification reports `ready_for_review`. "
            "This document records measurements; scientific closeout remains "
            "a human review decision."
        ),
        "",
        "## Primary comparison",
        "",
        "| Endpoint seed | Seven-level QAT | Continuous HWA | QAT minus continuous |",
        "| --- | ---: | ---: | ---: |",
    ]
    for seed in EXPECTED_ENDPOINT_SEEDS:
        qat = comparison["quantized_qat_by_endpoint_seed"][str(seed)]
        continuous = comparison["continuous_hwa_by_endpoint_seed"][str(seed)]
        difference = comparison["paired_differences_by_endpoint_seed"][str(seed)]
        lines.append(
            f"| {seed} | {_format_percent(qat)} | "
            f"{_format_percent(continuous)} | {_format_points(difference)} |"
        )
    lines.extend(
        [
            (
                f"| **Mean** | **{_format_percent(comparison['quantized_qat_mean'])}** | "
                f"**{_format_percent(comparison['continuous_hwa_mean'])}** | "
                f"**{_format_points(comparison['mean_paired_difference'])}** |"
            ),
            "",
            (
                "The paired-difference sample standard deviation is "
                f"{100.0 * comparison['paired_difference_sample_standard_deviation']:.2f} "
                "points. Exact enumeration of all 3,125 paired bootstrap "
                "resamples gives a 95% interval of "
                f"`[{_format_points(comparison['paired_bootstrap']['interval'][0])}, "
                f"{_format_points(comparison['paired_bootstrap']['interval'][1])}]`. "
                "The predeclared +2.00-point mean gate "
                f"was {'met' if comparison['minimum_difference_gate_met'] else 'not met'}."
            ),
            "",
            "## Pipeline decomposition",
            "",
            "| Pipeline | Development selected | Clean selected | Ideal mapped | Apparent mean | Persistent diagnostic | Ideal-to-apparent flips |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in _ARMS:
        pipeline = pipelines[arm["pipeline"]]
        lines.append(
            f"| {arm['label']} | "
            f"{_format_percent(pipeline['development_selected_accuracy'])} | "
            f"{_format_percent(pipeline['clean_selected_test_accuracy'])} | "
            f"{_format_percent(pipeline['ideal_mapped_target_test_accuracy'])} | "
            f"{_format_percent(pipeline['apparent_test_accuracy']['mean'])} | "
            f"{_format_percent(pipeline['persistent_diagnostic_accuracy']['mean'])} | "
            f"{_format_percent(pipeline['prediction_flip_fraction_from_ideal']['mean'])} |"
        )
    lines.extend(
        [
            "",
            (
                "The read-only development decompositions separate ideal mapping "
                "from the saved programmed endpoint. Seven-level QAT is "
                f"{_format_percent(transfer['quantized_qat']['development_ideal_mapped_validation_accuracy'])} "
                "at the development ideal target and "
                f"{_format_percent(transfer['quantized_qat']['development_saved_apparent_validation_accuracy'])} "
                "at its saved development endpoint; on held-out data it is "
                f"{_format_percent(transfer['quantized_qat']['heldout_ideal_mapped_test_accuracy'])} "
                "ideal and only "
                f"{_format_percent(transfer['quantized_qat']['heldout_apparent_test_accuracy_mean'])} "
                "apparent. Continuous HWA similarly changes from "
                f"{_format_percent(transfer['continuous_mapped_hwa']['development_ideal_mapped_validation_accuracy'])} "
                "ideal / "
                f"{_format_percent(transfer['continuous_mapped_hwa']['development_saved_apparent_validation_accuracy'])} "
                "apparent in development to "
                f"{_format_percent(transfer['continuous_mapped_hwa']['heldout_ideal_mapped_test_accuracy'])} "
                "ideal / "
                f"{_format_percent(transfer['continuous_mapped_hwa']['heldout_apparent_test_accuracy_mean'])} "
                "apparent when held out. Thus the large transfer loss occurs in "
                "the programmed endpoint response, not in ideal mapping accuracy. "
                "Repeated checkpoint selection on fixed development endpoint "
                "streams is a plausible exposure mechanism and is recorded as a "
                "limitation rather than asserted as a causal proof."
            ),
            "",
            (
                "The held-out structural mask is not the development mask. "
                f"For W1, {overlap['W1']['development_eligible_heldout_failure']:,} "
                "quads that were eligible during development become structural "
                "failures on assignment 85001, while "
                f"{overlap['W1']['development_failure_heldout_eligible']:,} move "
                "in the opposite direction. Only "
                f"{overlap['W1']['both_structural_failure']:,} W1 quads fail in "
                "both assignments. Baseline and eligibility tensor hashes also "
                "change, although the array-wide coordinate and D90 remain frozen. "
                "Because ideal mapped accuracy stays within half a point, this mask "
                "turnover does not explain the gross accuracy collapse by itself."
            ),
            "",
            "## Differential-signal mechanism",
            "",
            "| Pipeline/layer | Eligible D RMS | Min nonzero D | Apparent D-error RMS, all | Eligible only | Accepted-cell only | Persistent eligible | Eligible apparent SNR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for pipeline_name in ("continuous_mapped_hwa", "quantized_qat"):
        for layer_name in ("W1", "W2"):
            value = mechanism[pipeline_name][layer_name]
            target = value["target"]
            aggregate = value["aggregate"]
            lines.append(
                f"| {pipelines[pipeline_name]['label']}, {layer_name} | "
                f"{target['eligible_D_rms']:.6f} | "
                f"{target['eligible_minimum_nonzero_D']:.3e} | "
                f"{aggregate['apparent_all_quad_D_error_rms']['mean']:.6f} | "
                f"{aggregate['apparent_eligible_D_error_rms']['mean']:.6f} | "
                f"{aggregate['accepted_cell_only_D_error_rms']['mean']:.6f} | "
                f"{aggregate['persistent_eligible_D_error_rms']['mean']:.6f} | "
                f"{aggregate['apparent_eligible_D_snr_db']['mean']:.2f} dB |"
            )
    lines.extend(
        [
            "",
            (
                "For QAT, the eligible minimum nonzero D is the frozen seven-level "
                "spacing. It exceeds the accepted-cell-only D-error RMS in both "
                "layers. The much larger all-quad error is concentrated in the "
                "explicit structural-failure quads; those quads are retained, not "
                "donor-reassigned or clipped."
            ),
            "",
            "## Programming cost",
            "",
            "| Pipeline | Conditioning pulses | Target pulses/cell | Eligible acceptance | Accepted persistent within tolerance | Exhausted cells | Structural-failure cells |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in _ARMS:
        p = programming[arm["pipeline"]]["aggregate"]
        lines.append(
            f"| {arm['label']} | "
            f"{p['conditioning_total_pulses']['mean']:,.0f} | "
            f"{p['target_mean_pulses_per_device']['mean']:.3f} | "
            f"{_format_percent(p['programming_eligible_success_fraction']['mean'])} | "
            f"{p['accepted_persistent_within_tolerance']['mean']:,.1f} | "
            f"{p['budget_exhausted']['mean']:,.1f} | "
            f"{p['structural_target_assignment_failure_cells']['mean']:,.0f} |"
        )
    lines.extend(
        [
            "",
            (
                "Boundary conditioning is separate from the 128-pulse target "
                "budget. Apparent cell acceptance is about 99.86% among "
                "programming-eligible cells, but only about 30% of accepted cells "
                "remain within tolerance in persistent state. Active corrupt and "
                "non-finite counts are zero; published-corrupt identities remain "
                "reported provenance for the counterfactually repaired array."
            ),
            "",
            "## Scope",
            "",
            (
                "The comparison with the earlier RESET-relative study is "
                "descriptive only: coordinate, baseline policy, codebook, "
                "conditioning/controller path, gains, and learning rates all "
                "changed. This study contains no Tiki-Taka, LoRA, or direct "
                "pulse-recovery arm. A remaining deployment gap therefore cannot "
                "establish that on-chip training is necessary."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_root = args.study_root.resolve()
    if study_root.name != EXPECTED_STUDY_ID:
        raise ValueError(
            f"Expected study root {EXPECTED_STUDY_ID!r}, got {study_root.name!r}."
        )

    summary_path = study_root / "analysis/summary.json"
    summary = _load_json(summary_path)
    if summary.get("state") != "ready_for_review" or not summary.get(
        "ready_for_review"
    ):
        raise ValueError("Expected artifact-verified ready-for-review coverage.")
    freeze_path = study_root / "analysis/development_checkpoint_freeze.json"
    freeze = _load_json(freeze_path)
    if freeze.get("heldout_assignment_sampled_before_freeze") is not False:
        raise ValueError("Expected a pre-held-out checkpoint freeze receipt.")
    frozen_by_pipeline = {
        item["pipeline"]: item for item in freeze["checkpoints"]
    }

    reports_by_pipeline: dict[str, dict[int, dict[str, Any]]] = {}
    bundles_by_pipeline: dict[str, dict[int, dict[str, Any]]] = {}
    decomposition_paths: list[Path] = []
    common_cohort_hash: str | None = None
    pipeline_results: dict[str, Any] = {}
    mechanism: dict[str, Any] = {}
    programming: dict[str, Any] = {}
    development_decomposition_paths: list[Path] = []

    for arm in _ARMS:
        pipeline = arm["pipeline"]
        frozen = frozen_by_pipeline[pipeline]
        reports: dict[int, dict[str, Any]] = {}
        bundles: dict[int, dict[str, Any]] = {}
        for seed in EXPECTED_ENDPOINT_SEEDS:
            path = (
                study_root
                / "analysis/decompositions"
                / f"{arm['arm']}_{seed}.json"
            )
            decomposition_paths.append(path)
            report = _load_json(path)
            if report.get("schema") != (
                "ebl.mnist_relu_drn.ibm_om_deployment_decomposition"
            ):
                raise ValueError(f"Unexpected decomposition schema: {path}.")
            if report["cohort"]["examples"] != 10000 or not report["cohort"][
                "identical_across_all_states"
            ]:
                raise ValueError(f"Invalid decomposition cohort: {path}.")
            if report["contract"]["programming_or_resampling_invoked"]:
                raise ValueError(f"Decomposition was not read-only: {path}.")
            if not report["state_restoration"][
                "all_selected_weights_torch_equal_after_every_state"
            ]:
                raise ValueError(f"Checkpoint restoration failed: {path}.")
            if report["inputs"]["selected_weights"]["sha256"] != frozen[
                "weights_sha256"
            ]:
                raise ValueError(f"Frozen checkpoint mismatch: {path}.")
            if report["contract"]["population_fingerprint"] != (
                EXPECTED_POPULATION_FINGERPRINT
            ):
                raise ValueError(f"Held-out population mismatch: {path}.")
            mapping_report = report["contract"]["mapping_report"]
            if mapping_report["p90_eligible_quad_count"] != (
                EXPECTED_P90_ELIGIBLE_QUADS
            ):
                raise ValueError(f"Held-out D90 eligibility mismatch: {path}.")
            cohort_hash = report["cohort"]["ordered_inputs_and_labels_sha256"]
            if common_cohort_hash is None:
                common_cohort_hash = cohort_hash
            elif common_cohort_hash != cohort_hash:
                raise ValueError("Held-out decompositions changed the test cohort.")
            report_seed = report["contract"]["programming_report"]["endpoint_seed"]
            if report_seed != seed:
                raise ValueError(f"Endpoint seed mismatch: {path}.")
            reports[seed] = report
            bundle = _load_bundle(report)
            if bundle["population_sampling_receipt"]["population_sha256"] != (
                EXPECTED_POPULATION_SHA256
            ):
                raise ValueError(f"Held-out population digest mismatch: {path}.")
            bundles[seed] = bundle

        reports_by_pipeline[pipeline] = reports
        bundles_by_pipeline[pipeline] = bundles
        ideal_prediction_hashes = {
            reports[seed]["validation_states"]["ideal_mapped_target"][
                "prediction_sha256"
            ]
            for seed in EXPECTED_ENDPOINT_SEEDS
        }
        clean_prediction_hashes = {
            reports[seed]["validation_states"]["clean_selected"][
                "prediction_sha256"
            ]
            for seed in EXPECTED_ENDPOINT_SEEDS
        }
        if len(ideal_prediction_hashes) != 1 or len(clean_prediction_hashes) != 1:
            raise ValueError(
                f"Ideal or clean prediction changed over endpoint seeds: {pipeline}."
            )

        state_metrics = {
            state_name: _state_summary(reports, state_name)
            for state_name in _STATE_NAMES
        }
        apparent_state = state_metrics["apparent_endpoint"]
        persistent_state = state_metrics["persistent_endpoint_diagnostic"]
        prediction_flip = _series(
            {
                seed: float(
                    reports[seed]["validation_states"]["apparent_endpoint"][
                        "versus_ideal_mapped_target"
                    ]["prediction_flip_fraction"]
                )
                for seed in EXPECTED_ENDPOINT_SEEDS
            }
        )
        ideal_to_apparent_correct_to_incorrect = _series(
            {
                seed: float(
                    reports[seed]["validation_states"]["apparent_endpoint"][
                        "versus_ideal_mapped_target"
                    ]["reference_correct_to_observed_incorrect"]
                )
                for seed in EXPECTED_ENDPOINT_SEEDS
            }
        )
        ideal_to_apparent_incorrect_to_correct = _series(
            {
                seed: float(
                    reports[seed]["validation_states"]["apparent_endpoint"][
                        "versus_ideal_mapped_target"
                    ]["reference_incorrect_to_observed_correct"]
                )
                for seed in EXPECTED_ENDPOINT_SEEDS
            }
        )
        pipeline_results[pipeline] = {
            "label": arm["label"],
            "selected_epoch": frozen["selected_epoch"],
            "development_selected_accuracy": frozen[
                "development_student_accuracy"
            ],
            "clean_selected_test_accuracy": state_metrics["clean_selected"][
                "student_accuracy"
            ]["mean"],
            "ideal_global_requested_test_accuracy": state_metrics[
                "ideal_global_requested"
            ]["student_accuracy"]["mean"],
            "ideal_mapped_target_test_accuracy": state_metrics[
                "ideal_mapped_target"
            ]["student_accuracy"]["mean"],
            "apparent_test_accuracy": apparent_state["student_accuracy"],
            "persistent_diagnostic_accuracy": persistent_state[
                "student_accuracy"
            ],
            "prediction_flip_fraction_from_ideal": prediction_flip,
            "ideal_to_apparent_prediction_transitions": {
                "ideal_correct_to_apparent_incorrect": (
                    ideal_to_apparent_correct_to_incorrect
                ),
                "ideal_incorrect_to_apparent_correct": (
                    ideal_to_apparent_incorrect_to_correct
                ),
            },
            "mapping_accuracy_change_from_clean": (
                state_metrics["ideal_mapped_target"]["student_accuracy"]["mean"]
                - state_metrics["clean_selected"]["student_accuracy"]["mean"]
            ),
            "programming_accuracy_change_from_ideal": (
                apparent_state["student_accuracy"]["mean"]
                - state_metrics["ideal_mapped_target"]["student_accuracy"]["mean"]
            ),
            "persistent_accuracy_change_from_apparent": (
                persistent_state["student_accuracy"]["mean"]
                - apparent_state["student_accuracy"]["mean"]
            ),
            "state_metrics": state_metrics,
        }

        pipeline_layers: dict[str, Any] = {}
        target_references: dict[str, torch.Tensor] = {}
        mask_references: dict[str, torch.Tensor] = {}
        snapshots_by_layer: dict[str, dict[int, dict[str, Any]]] = {
            "W1": {},
            "W2": {},
        }
        for seed in EXPECTED_ENDPOINT_SEEDS:
            bundle = bundles[seed]
            layers = _layer_geometries(bundle)
            for layer in layers:
                snapshot, target, structural_eligible = _layer_seed_snapshot(
                    bundle,
                    layer,
                )
                snapshots_by_layer[layer.name][seed] = snapshot
                if layer.name not in target_references:
                    target_references[layer.name] = target
                    mask_references[layer.name] = structural_eligible
                elif not torch.equal(target_references[layer.name], target):
                    raise ValueError(
                        f"Mapped D target changed across endpoint seeds: "
                        f"{pipeline}/{layer.name}."
                    )
        mapping_report = reports[EXPECTED_ENDPOINT_SEEDS[0]]["contract"][
            "mapping_report"
        ]
        quantized_spacing = mapping_report["quantized_differential_spacing"]
        for layer_name in ("W1", "W2"):
            pipeline_layers[layer_name] = _aggregate_layer_snapshots(
                snapshots_by_layer[layer_name],
                quantized_spacing=(
                    None
                    if quantized_spacing is None
                    else float(quantized_spacing)
                ),
                target=target_references[layer_name],
                structural_eligible=mask_references[layer_name],
            )
        pipeline_layers["mapping_report"] = {
            "raw_active_mode": mapping_report["raw_active_mode"],
            "frozen_differential_budget_D90": mapping_report[
                "frozen_differential_budget"
            ],
            "quantized_differential_spacing_D": quantized_spacing,
            "logical_code_histogram_all_quads": mapping_report[
                "logical_code_histogram"
            ],
            "eligible_baseline": mapping_report["eligible_quad_baseline"],
            "eligible_baseline_sha256": mapping_report[
                "expanded_quad_baseline_sha256"
            ],
            "structural_eligibility_sha256": mapping_report[
                "expanded_quad_eligibility_sha256"
            ],
            "p90_eligible_quad_count": mapping_report[
                "p90_eligible_quad_count"
            ],
            "structural_failure_quad_count": mapping_report[
                "structural_failure_quad_count"
            ],
        }
        mechanism[pipeline] = pipeline_layers
        programming[pipeline] = _programming_summary(reports)

    for arm in _ARMS:
        pipeline = arm["pipeline"]
        frozen = frozen_by_pipeline[pipeline]
        path = (
            study_root
            / "analysis"
            / f"development_decomposition_{pipeline}.json"
        )
        development_decomposition_paths.append(path)
        development = _load_json(path)
        if (
            development.get("schema")
            != "ebl.mnist_relu_drn.ibm_om_deployment_decomposition"
            or development["cohort"]["split"] != "validation"
            or development["cohort"]["examples"] != 5000
            or development["contract"]["programming_or_resampling_invoked"]
            or not development["state_restoration"][
                "all_selected_weights_torch_equal_after_every_state"
            ]
            or development["inputs"]["selected_weights"]["sha256"]
            != frozen["weights_sha256"]
        ):
            raise ValueError(f"Invalid development decomposition: {path}.")
        validation_states = development["validation_states"]
        apparent = validation_states["apparent_endpoint"]
        pipeline_results[pipeline]["development_decomposition"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "examples": 5000,
            "clean_selected_accuracy": validation_states["clean_selected"][
                "metrics"
            ]["student_accuracy"],
            "ideal_mapped_target_accuracy": validation_states[
                "ideal_mapped_target"
            ]["metrics"]["student_accuracy"],
            "saved_apparent_endpoint_accuracy": apparent["metrics"][
                "student_accuracy"
            ],
            "saved_persistent_endpoint_diagnostic_accuracy": validation_states[
                "persistent_endpoint_diagnostic"
            ]["metrics"]["student_accuracy"],
            "apparent_prediction_flip_fraction_from_ideal": apparent[
                "versus_ideal_mapped_target"
            ]["prediction_flip_fraction"],
            "ideal_correct_to_apparent_incorrect": apparent[
                "versus_ideal_mapped_target"
            ]["reference_correct_to_observed_incorrect"],
            "ideal_incorrect_to_apparent_correct": apparent[
                "versus_ideal_mapped_target"
            ]["reference_incorrect_to_observed_correct"],
            "layer_D": {
                layer_name: {
                    "target_D_rms": development["layers"][layer_name][
                        "raw_active_logical_differential_D"
                    ]["mapped_target_signal"]["rms"],
                    "apparent_D_error_rms": development["layers"][layer_name][
                        "raw_active_logical_differential_D"
                    ]["apparent_endpoint"]["error_from_mapped_target"]["rms"],
                    "persistent_D_error_rms": development["layers"][layer_name][
                        "raw_active_logical_differential_D"
                    ]["persistent_endpoint"]["error_from_mapped_target"]["rms"],
                }
                for layer_name in ("W1", "W2")
            },
        }

    continuous = pipeline_results["continuous_mapped_hwa"][
        "apparent_test_accuracy"
    ]["by_endpoint_seed"]
    quantized = pipeline_results["quantized_qat"]["apparent_test_accuracy"][
        "by_endpoint_seed"
    ]
    differences = {
        seed: float(quantized[str(seed)] - continuous[str(seed)])
        for seed in EXPECTED_ENDPOINT_SEEDS
    }
    difference_values = [differences[seed] for seed in EXPECTED_ENDPOINT_SEEDS]
    bootstrap_values = sorted(
        _mean(sample)
        for sample in itertools.product(difference_values, repeat=5)
    )
    primary_comparison = {
        "contrast": (
            "quantized_qat_minus_continuous_mapped_hwa_apparent_forward_"
            "heldout_test_accuracy"
        ),
        "quantized_qat_by_endpoint_seed": quantized,
        "continuous_hwa_by_endpoint_seed": continuous,
        "quantized_qat_mean": _mean(
            [quantized[str(seed)] for seed in EXPECTED_ENDPOINT_SEEDS]
        ),
        "quantized_qat_sample_standard_deviation": _sample_standard_deviation(
            [quantized[str(seed)] for seed in EXPECTED_ENDPOINT_SEEDS]
        ),
        "continuous_hwa_mean": _mean(
            [continuous[str(seed)] for seed in EXPECTED_ENDPOINT_SEEDS]
        ),
        "continuous_hwa_sample_standard_deviation": _sample_standard_deviation(
            [continuous[str(seed)] for seed in EXPECTED_ENDPOINT_SEEDS]
        ),
        "paired_differences_by_endpoint_seed": {
            str(seed): differences[seed] for seed in EXPECTED_ENDPOINT_SEEDS
        },
        "mean_paired_difference": _mean(difference_values),
        "paired_difference_sample_standard_deviation": (
            _sample_standard_deviation(difference_values)
        ),
        "paired_bootstrap": {
            "method": "exact_enumeration_of_all_5_to_the_5_paired_resamples",
            "replicates": len(bootstrap_values),
            "confidence_level": 0.95,
            "interval": [
                _linear_quantile(bootstrap_values, 0.025),
                _linear_quantile(bootstrap_values, 0.975),
            ],
        },
        "predeclared_minimum_mean_difference": 0.02,
        "minimum_difference_gate_met": _mean(difference_values) >= 0.02,
    }

    development_bundle_path = (
        study_root
        / frozen_by_pipeline["quantized_qat"]["result_path"]
    ).parent / "artifacts/ibm_om_deployment.pt"
    development_bundle = torch.load(
        development_bundle_path,
        map_location="cpu",
        weights_only=False,
    )
    heldout_bundle = bundles_by_pipeline["quantized_qat"][
        EXPECTED_ENDPOINT_SEEDS[0]
    ]
    assignment_transfer = {
        "development_assignment_seed": 84001,
        "heldout_assignment_seed": 85001,
        "development_population_fingerprint": development_bundle[
            "population_fingerprint"
        ],
        "heldout_population_fingerprint": heldout_bundle[
            "population_fingerprint"
        ],
        "structural_eligibility_overlap": _mask_overlap(
            development_bundle,
            heldout_bundle,
        ),
        "accuracy_transfer_by_pipeline": {
            pipeline: {
                "development_ideal_mapped_validation_accuracy": (
                    value["development_decomposition"][
                        "ideal_mapped_target_accuracy"
                    ]
                ),
                "heldout_ideal_mapped_test_accuracy": value[
                    "ideal_mapped_target_test_accuracy"
                ],
                "heldout_minus_development_ideal_mapped_accuracy": (
                    value["ideal_mapped_target_test_accuracy"]
                    - value["development_decomposition"][
                        "ideal_mapped_target_accuracy"
                    ]
                ),
                "development_saved_apparent_validation_accuracy": (
                    value["development_decomposition"][
                        "saved_apparent_endpoint_accuracy"
                    ]
                ),
                "heldout_apparent_test_accuracy_mean": value[
                    "apparent_test_accuracy"
                ]["mean"],
                "heldout_minus_development_apparent_accuracy": (
                    value["apparent_test_accuracy"]["mean"]
                    - value["development_decomposition"][
                        "saved_apparent_endpoint_accuracy"
                    ]
                ),
                "development_ideal_correct_to_apparent_incorrect": (
                    value["development_decomposition"][
                        "ideal_correct_to_apparent_incorrect"
                    ]
                ),
                "development_ideal_incorrect_to_apparent_correct": (
                    value["development_decomposition"][
                        "ideal_incorrect_to_apparent_correct"
                    ]
                ),
                "heldout_ideal_correct_to_apparent_incorrect_mean": (
                    value["ideal_to_apparent_prediction_transitions"][
                        "ideal_correct_to_apparent_incorrect"
                    ]["mean"]
                ),
                "heldout_ideal_incorrect_to_apparent_correct_mean": (
                    value["ideal_to_apparent_prediction_transitions"][
                        "ideal_incorrect_to_apparent_correct"
                    ]["mean"]
                ),
            }
            for pipeline, value in pipeline_results.items()
        },
        "development_mapping_report": {
            key: development_bundle["target_mapping_report"][key]
            for key in (
                "p90_eligible_quad_count",
                "structural_failure_quad_count",
                "expanded_quad_baseline_sha256",
                "expanded_quad_eligibility_sha256",
                "eligible_quad_baseline",
            )
        },
        "heldout_mapping_report": {
            key: heldout_bundle["target_mapping_report"][key]
            for key in (
                "p90_eligible_quad_count",
                "structural_failure_quad_count",
                "expanded_quad_baseline_sha256",
                "expanded_quad_eligibility_sha256",
                "eligible_quad_baseline",
            )
        },
    }

    previous_study_path = (
        study_root.parent
        / "mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1"
        / "analysis/primary_analysis.json"
    )
    reset_measurements = None
    if previous_study_path.exists():
        previous = _load_json(previous_study_path)
        reset_measurements = {
            "source_path": str(previous_study_path),
            "source_sha256": sha256_file(previous_study_path),
            "quantized_qat_apparent_accuracy_mean": previous[
                "primary_comparison"
            ]["quantized_qat_mean"],
            "continuous_hwa_apparent_accuracy_mean": previous[
                "primary_comparison"
            ]["continuous_hwa_mean"],
            "mean_paired_qat_minus_continuous": previous[
                "primary_comparison"
            ]["mean_paired_difference"],
        }

    analysis_module = Path(__file__).resolve()
    decomposition_module = (
        _ROOT
        / "ebl/runtime/mnist_relu_drn/ibm_om_deployment_decomposition.py"
    )
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "study_id": EXPECTED_STUDY_ID,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lifecycle_state": summary["state"],
        "coverage": {
            "completed_development_runs": 4,
            "completed_heldout_deployments": 20,
            "expected_heldout_deployments": 20,
            "decomposition_reports": len(decomposition_paths),
            "development_decomposition_reports": len(
                development_decomposition_paths
            ),
            "endpoint_seeds": list(EXPECTED_ENDPOINT_SEEDS),
            "assignment_seed": 85001,
            "test_examples_per_deployment": 10000,
            "ordered_test_cohort_sha256": common_cohort_hash,
            "artifact_verification_passed": True,
            "population_npz_sha256": EXPECTED_POPULATION_SHA256,
            "population_fingerprint": EXPECTED_POPULATION_FINGERPRINT,
            "p90_eligible_quads": EXPECTED_P90_ELIGIBLE_QUADS,
        },
        "primary_comparison": primary_comparison,
        "pipeline_results": pipeline_results,
        "mechanism": mechanism,
        "programming": programming,
        "assignment_transfer": assignment_transfer,
        "descriptive_reset_relative_comparison": {
            "causal_single_intervention_comparison_allowed": False,
            "changed_factors": [
                "coordinate",
                "baseline_policy",
                "D_budget_and_codebook",
                "conditioning_and_controller_path",
                "training_endpoint_model",
                "fixed_forward_gains",
                "learning_rates",
            ],
            "raw_active_measurements": {
                "quantized_qat_apparent_accuracy_mean": primary_comparison[
                    "quantized_qat_mean"
                ],
                "continuous_hwa_apparent_accuracy_mean": primary_comparison[
                    "continuous_hwa_mean"
                ],
                "mean_paired_qat_minus_continuous": primary_comparison[
                    "mean_paired_difference"
                ],
            },
            "reset_relative_measurements": reset_measurements,
            "learning_rate_receipts": _load_protocol_learning_rates(_ROOT),
        },
        "scope": {
            "off_chip_training_only": True,
            "on_chip_recovery_arm_included": False,
            "tiki_taka_included": False,
            "lora_included": False,
            "direct_pulse_recovery_included": False,
            "residual_gap_establishes_on_chip_need": False,
        },
        "provenance": {
            "development_checkpoint_freeze_sha256": sha256_file(freeze_path),
            "artifact_verified_study_summary_sha256": sha256_file(summary_path),
            "analysis_module": str(analysis_module),
            "analysis_module_sha256": sha256_file(analysis_module),
            "decomposition_module": str(decomposition_module),
            "decomposition_module_sha256": sha256_file(decomposition_module),
            "decomposition_report_count": len(decomposition_paths),
            "development_decomposition_report_count": len(
                development_decomposition_paths
            ),
            "ordered_decomposition_sha256_manifest_digest": (
                _ordered_manifest_digest(decomposition_paths)
            ),
            "ordered_development_decomposition_sha256_manifest_digest": (
                _ordered_manifest_digest(development_decomposition_paths)
            ),
            "decomposition_operation": "read_only_saved_endpoint_replay",
            "programming_or_resampling_invoked_by_decomposition": False,
        },
    }
    atomic_write_json(args.output_json, report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    print(args.output_json)
    print(args.output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
