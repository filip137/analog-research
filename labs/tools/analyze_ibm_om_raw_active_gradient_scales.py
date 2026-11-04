#!/usr/bin/env python3
"""Read-only gradient replay for clean and raw-active mapped DRN states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ebl.runtime.store import atomic_write_json, sha256_file
from ebl.registry import resolve_experiment_config
from ebl.runtime.mnist_relu_drn.components import build_student_stack
from ebl.runtime.mnist_relu_drn.runtime import (
    _amplification_index_report,
    _load_teacher,
    _validate_checkpoint_metadata,
)
from networks.data.mnist import build_mnist_loaders, limited
from ebl.schema.base import RunMode
from methods.checkpoint import load_named_weights
from methods.ibm_om.api import (
    _raw_active_structural_cell_mask,
    load_om_array_population,
    map_ibm_reram_array_targets,
    validate_ibm_reram_target_mapping_preflight,
)


_MODES = ("clean", "continuous", "quantized_7_level")
_GAINS = {
    "continuous": 251.18864315095797,
    "quantized_7_level": 223.87211385683378,
}
_LAYOUTS = {
    "base.dense_weight.0": "halves",
    "base.dense_weight.1": "paired",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-batches", type=int, default=64)
    return parser


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def _catalog_sha256(stack: Any) -> str:
    digest = hashlib.sha256()
    for binding in stack.bundle.catalog.trainable:
        digest.update(binding.key.encode("utf-8"))
        digest.update(_tensor_bytes(binding.state))
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(value)).hexdigest()


def _quantile(value: torch.Tensor, probability: float) -> float:
    return float(
        np.quantile(
            value.detach().cpu().to(torch.float64).numpy(),
            probability,
        )
    )


def _gradient_summary(
    values: torch.Tensor,
    batch_rms: list[float],
) -> dict[str, Any]:
    flat = values.to(torch.float64)
    absolute = flat.abs()
    batch = torch.tensor(batch_rms, dtype=torch.float64)
    return {
        "values": int(flat.numel()),
        "gradient_mean": float(flat.mean().item()),
        "gradient_rms": float(flat.square().mean().sqrt().item()),
        "gradient_l2": float(torch.linalg.vector_norm(flat).item()),
        "gradient_minimum": float(flat.min().item()),
        "gradient_maximum": float(flat.max().item()),
        "positive_fraction": float((flat > 0.0).to(torch.float64).mean().item()),
        "negative_fraction": float((flat < 0.0).to(torch.float64).mean().item()),
        "exact_zero_fraction": float((flat == 0.0).to(torch.float64).mean().item()),
        "near_zero_fraction_abs_le_1e_12": float(
            (absolute <= 1e-12).to(torch.float64).mean().item()
        ),
        "absolute_gradient_p50": _quantile(absolute, 0.50),
        "absolute_gradient_p90": _quantile(absolute, 0.90),
        "absolute_gradient_p99": _quantile(absolute, 0.99),
        "batch_gradient_rms": {
            "minimum": float(batch.min().item()),
            "p25": float(torch.quantile(batch, 0.25).item()),
            "median": float(torch.quantile(batch, 0.50).item()),
            "p75": float(torch.quantile(batch, 0.75).item()),
            "maximum": float(batch.max().item()),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.maximum_batches <= 0:
        raise ValueError("Expected --maximum-batches to be positive.")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("Expected --output not to overwrite an artifact.")
    inputs = {
        "config": args.config.expanduser().resolve(),
        "weights": args.weights.expanduser().resolve(),
        "teacher_weights": args.teacher_weights.expanduser().resolve(),
        "population": args.population.expanduser().resolve(),
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Expected readable {name}: {str(path)!r}.")

    _definition, spec = resolve_experiment_config(inputs["config"], RunMode.TRAIN)
    if (
        spec.model.encoding != "single"
        or (spec.model.conductance_min, spec.model.conductance_max) != (0.0, 1.0)
        or spec.settings.weight_modifier.type != "none"
    ):
        raise ValueError("Expected the modifier-free [0,1] calibration config.")
    torch.manual_seed(spec.runtime.seed)
    stack = build_student_stack(spec, enable_measured=False)
    checkpoint_sha_before = sha256_file(inputs["weights"])
    loaded = load_named_weights(inputs["weights"], stack.bundle.catalog)
    teacher_sha = sha256_file(inputs["teacher_weights"])
    _validate_checkpoint_metadata(
        loaded.metadata,
        spec=spec,
        teacher_sha256=teacher_sha,
        expected_amplification_indices=_amplification_index_report(stack),
    )
    clean_gain = float(loaded.metadata["fixed_logit_gain"])
    teacher, _teacher_metadata = _load_teacher(
        inputs["teacher_weights"],
        device=stack.device,
        spec=spec,
    )
    data = build_mnist_loaders(
        spec.data,
        data_seed=spec.runtime.data_seed,
        calibration_examples=spec.mapping.calibration_examples,
        calibration_batch_size=spec.mapping.calibration_batch_size,
    )
    bindings = tuple(stack.bundle.catalog.trainable)
    parameters = tuple(stack.bundle.energy.params())
    clean_states = tuple(binding.state.detach().clone() for binding in bindings)
    catalog_sha_before = _catalog_sha256(stack)
    population = load_om_array_population(inputs["population"]).to(stack.device)
    if (
        population.assignment_seed != 84001
        or population.binding_keys != tuple(binding.key for binding in bindings)
        or population.binding_shapes
        != tuple(tuple(binding.state.shape) for binding in bindings)
    ):
        raise ValueError("Expected the exact assignment-84001 binding layout.")
    global_targets = torch.cat(
        tuple(value.reshape(-1) for value in clean_states)
    )
    structural, lower, upper = _raw_active_structural_cell_mask(
        population,
        _LAYOUTS,
        device=stack.device,
    )
    states: dict[str, tuple[torch.Tensor, ...]] = {"clean": clean_states}
    mapping_reports: dict[str, Any] = {}
    for mode in _MODES[1:]:
        mapped, mapping_report = map_ibm_reram_array_targets(
            global_targets,
            population,
            target_mapping="raw_active_p90_quad",
            dual_rail_layout_by_parameter=_LAYOUTS,
            common_window_margin_fraction=0.0,
            raw_active_mode=mode,
            raw_active_unsupported_quad_policy="structural_failure",
        )
        validate_ibm_reram_target_mapping_preflight(mapping_report)
        supported = (
            structural
            & torch.isfinite(mapped)
            & (mapped >= 0.0)
            & (mapped >= lower)
            & (mapped <= upper)
        )
        surrogate = torch.where(supported, mapped, lower)
        offset = 0
        values = []
        for binding in bindings:
            count = binding.state.numel()
            values.append(
                surrogate[offset : offset + count].reshape(binding.state.shape)
            )
            offset += count
        states[mode] = tuple(values)
        mapping_reports[mode] = {
            "mapping": mapping_report,
            "mapped_training_supported_cells": int(supported.sum().item()),
            "mapped_training_reset_bound_cells": int((~supported).sum().item()),
        }

    mode_gradients: dict[str, dict[str, list[torch.Tensor]]] = {}
    mode_batch_rms: dict[str, dict[str, list[float]]] = {}
    direction_rows: dict[str, dict[str, list[dict[str, float]]]] = {
        mode: {binding.key: [] for binding in bindings}
        for mode in _MODES[1:]
    }
    cohort_hashes: dict[str, str] = {}
    example_counts: dict[str, int] = {}
    clean_by_batch: dict[str, list[torch.Tensor]] = {}
    for mode in _MODES:
        with torch.no_grad():
            for binding, state in zip(bindings, states[mode]):
                binding.state.copy_(state.to(binding.state))
        stack.cost.gain = clean_gain if mode == "clean" else _GAINS[mode]
        chunks = {binding.key: [] for binding in bindings}
        batch_rms = {binding.key: [] for binding in bindings}
        cohort = hashlib.sha256()
        examples = 0
        for batch_index, (inputs_batch, labels_batch) in enumerate(
            limited(data.validation, args.maximum_batches)
        ):
            cohort.update(_tensor_bytes(inputs_batch))
            cohort.update(_tensor_bytes(labels_batch))
            inputs_device = inputs_batch.to(stack.device, dtype=torch.float32)
            labels_device = labels_batch.to(stack.device, dtype=torch.long)
            with torch.no_grad():
                teacher_logits = teacher.logits(inputs_device)
            stack.network.set_input(inputs_device, reset=True)
            stack.minimizer.compute_equilibrium()
            stack.cost.set_teacher(teacher_logits, labels_device)
            gradients = tuple(stack.differentiator.compute_gradient())
            for binding, parameter, gradient in zip(
                bindings,
                parameters,
                gradients,
            ):
                if (
                    gradient.shape != parameter.state.shape
                    or not bool(torch.isfinite(gradient).all())
                ):
                    raise RuntimeError("Expected finite shape-matched gradients.")
                flat = gradient.detach().reshape(-1).cpu().to(torch.float32)
                chunks[binding.key].append(flat)
                batch_rms[binding.key].append(
                    float(flat.to(torch.float64).square().mean().sqrt().item())
                )
                if mode == "clean":
                    clean_by_batch.setdefault(binding.key, []).append(flat)
                else:
                    clean = clean_by_batch[binding.key][batch_index].to(torch.float64)
                    observed = flat.to(torch.float64)
                    clean_norm = torch.linalg.vector_norm(clean)
                    observed_norm = torch.linalg.vector_norm(observed)
                    denominator = clean_norm * observed_norm
                    direction_rows[mode][binding.key].append(
                        {
                            "cosine_to_clean": (
                                float(torch.dot(clean, observed).div(denominator).item())
                                if float(denominator.item()) > 0.0
                                else 0.0
                            ),
                            "relative_l2_difference_from_clean": float(
                                torch.linalg.vector_norm(observed - clean)
                                .div(clean_norm)
                                .item()
                            ),
                        }
                    )
            examples += int(labels_batch.numel())
        cohort_hashes[mode] = cohort.hexdigest()
        example_counts[mode] = examples
        mode_gradients[mode] = chunks
        mode_batch_rms[mode] = batch_rms
    if len(set(cohort_hashes.values())) != 1 or len(set(example_counts.values())) != 1:
        raise RuntimeError("Expected one identical replay cohort for every state.")

    summaries: dict[str, Any] = {}
    for mode in _MODES:
        parameters_report = {}
        for binding in bindings:
            key = binding.key
            values = torch.cat(mode_gradients[mode][key])
            report = _gradient_summary(values, mode_batch_rms[mode][key])
            if mode != "clean":
                rows = direction_rows[mode][key]
                cosines = torch.tensor(
                    [row["cosine_to_clean"] for row in rows],
                    dtype=torch.float64,
                )
                relative = torch.tensor(
                    [
                        row["relative_l2_difference_from_clean"]
                        for row in rows
                    ],
                    dtype=torch.float64,
                )
                report["direction_relative_to_clean_same_batch"] = {
                    "cosine_median": float(cosines.median().item()),
                    "cosine_minimum": float(cosines.min().item()),
                    "cosine_maximum": float(cosines.max().item()),
                    "relative_l2_difference_median": float(
                        relative.median().item()
                    ),
                }
            parameters_report[key] = report
        summaries[mode] = {
            "fixed_logit_gain": clean_gain if mode == "clean" else _GAINS[mode],
            "state_sha256_by_parameter": {
                binding.key: _tensor_sha256(state)
                for binding, state in zip(bindings, states[mode])
            },
            "parameters": parameters_report,
        }

    with torch.no_grad():
        for binding, state in zip(bindings, clean_states):
            binding.state.copy_(state)
    checkpoint_sha_after = sha256_file(inputs["weights"])
    catalog_sha_after = _catalog_sha256(stack)
    if (
        checkpoint_sha_before != checkpoint_sha_after
        or catalog_sha_before != catalog_sha_after
    ):
        raise RuntimeError("Expected read-only replay to leave checkpoint state unchanged.")
    result = {
        "schema": "ebl.ibm_om_raw_active_gradient_scale_replay",
        "schema_version": 1,
        "operation": "read_only_identical_cohort_no_optimizer_step",
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "settings": {
            "split": "validation",
            "maximum_batches": args.maximum_batches,
            "batch_size": spec.data.batch_size,
            "examples": next(iter(example_counts.values())),
            "cohort_sha256": next(iter(cohort_hashes.values())),
            "solver": {
                "inference_iterations": spec.solver.inference_iterations,
                "training_iterations": spec.solver.training_iterations,
                "mode": spec.solver.mode,
                "overrelaxation_factor": spec.solver.overrelaxation_factor,
            },
            "gradient_estimator": type(stack.differentiator).__name__,
            "objective": "teacher_kl",
        },
        "mapping_reports": mapping_reports,
        "states": summaries,
        "integrity": {
            "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after,
            "catalog_sha256_before": catalog_sha_before,
            "catalog_sha256_after": catalog_sha_after,
            "unchanged": True,
        },
    }
    atomic_write_json(output, result)
    print(json.dumps({"output": str(output), "integrity": result["integrity"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
