#!/usr/bin/env python3
"""Measure read-only KD gradients and propose per-parameter thresholds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ebl.runtime.store import sha256_file
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


_POSITIVE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--teacher-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--maximum-batches",
        type=int,
        default=None,
        help="Use the full deterministic validation split when omitted.",
    )
    return parser.parse_args()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _catalog_sha256(stack: Any) -> str:
    digest = hashlib.sha256()
    for binding in stack.bundle.catalog.trainable:
        digest.update(binding.key.encode("utf-8"))
        digest.update(str(tuple(binding.state.shape)).encode("ascii"))
        digest.update(str(binding.state.dtype).encode("ascii"))
        digest.update(_tensor_bytes(binding.state))
    return digest.hexdigest()


def _quantile(values: torch.Tensor, value: float) -> float:
    if values.numel() == 0:
        raise ValueError(
            "Expected at least one positive gradient value. Provided value: 0."
        )
    # torch.quantile uses an indexing path that rejects very large tensors;
    # the first MNIST conductance layer exceeds that limit on a full replay.
    return float(np.quantile(values.numpy(), value))


def _spread(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    tensor = torch.tensor(ordered, dtype=torch.float64)
    return {
        "minimum": ordered[0],
        "p25": float(torch.quantile(tensor, 0.25).item()),
        "median": median(ordered),
        "p75": float(torch.quantile(tensor, 0.75).item()),
        "maximum": ordered[-1],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Expected at least one CSV row. Provided value: 0.")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot_positive_ccdf(
    values_by_parameter: dict[str, torch.Tensor],
    thresholds: dict[str, dict[str, float]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(values_by_parameter),
        1,
        figsize=(9.5, max(3.8, 3.6 * len(values_by_parameter))),
        squeeze=False,
    )
    for axis, (key, values) in zip(axes[:, 0], values_by_parameter.items()):
        positive = values[values > 0.0].sort().values
        if positive.numel() > 200_000:
            indices = torch.linspace(
                0,
                positive.numel() - 1,
                steps=200_000,
                dtype=torch.float64,
            ).round().to(torch.long)
            positive = positive[indices]
        survival = 1.0 - torch.arange(
            positive.numel(), dtype=torch.float64
        ) / float(positive.numel())
        axis.plot(positive.numpy(), survival.numpy(), linewidth=1.6)
        for label, threshold in thresholds[key].items():
            axis.axvline(threshold, linestyle="--", linewidth=1.1, label=label)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(5e-4, 1.05)
        axis.set_title(key)
        axis.set_xlabel("raw positive gradient threshold")
        axis.set_ylabel("fraction of positive gradients above threshold")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    fig.suptitle("Cohort-A initialization: positive-gradient threshold calibration")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    teacher_path = Path(args.teacher_weights).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.maximum_batches is not None and args.maximum_batches <= 0:
        raise ValueError(
            "Expected --maximum-batches to be a positive integer or omitted. "
            f"Provided value: {args.maximum_batches!r}."
        )
    for label, path in (
        ("--config", config_path),
        ("--weights", weights_path),
        ("--teacher-weights", teacher_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected {label} to name an existing file. "
                f"Provided value: {str(path)!r}."
            )
    output_dir.mkdir(parents=True, exist_ok=False)

    _definition, spec = resolve_experiment_config(config_path, RunMode.TRAIN)
    torch.manual_seed(spec.runtime.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.runtime.seed)
    data = build_mnist_loaders(
        spec.data,
        data_seed=spec.runtime.data_seed,
        calibration_examples=spec.mapping.calibration_examples,
        calibration_batch_size=spec.mapping.calibration_batch_size,
    )
    teacher, _teacher_metadata = _load_teacher(
        teacher_path,
        device=torch.device(spec.runtime.device),
        spec=spec,
    )
    stack = build_student_stack(spec, enable_measured=False)
    checkpoint_sha_before = sha256_file(weights_path)
    loaded = load_named_weights(weights_path, stack.bundle.catalog)
    _validate_checkpoint_metadata(
        loaded.metadata,
        spec=spec,
        teacher_sha256=sha256_file(teacher_path),
        expected_amplification_indices=_amplification_index_report(stack),
    )
    stack.cost.gain = float(loaded.metadata["fixed_logit_gain"])
    parameter_sha_before = _catalog_sha256(stack)

    chunks: dict[str, list[torch.Tensor]] = {
        binding.key: [] for binding in stack.bundle.catalog.trainable
    }
    batch_rows: list[dict[str, Any]] = []
    cohort_digest = hashlib.sha256()
    example_count = 0
    batch_count = 0
    parameters = tuple(stack.bundle.energy.params())
    bindings = tuple(stack.bundle.catalog.trainable)
    for batch_index, (inputs, labels) in enumerate(
        limited(data.validation, args.maximum_batches)
    ):
        cohort_digest.update(_tensor_bytes(inputs))
        cohort_digest.update(_tensor_bytes(labels))
        inputs = inputs.to(stack.device, dtype=torch.float32)
        labels = labels.to(stack.device, dtype=torch.long)
        with torch.no_grad():
            teacher_logits = teacher.logits(inputs)
        stack.network.set_input(inputs, reset=True)
        stack.minimizer.compute_equilibrium()
        stack.cost.set_teacher(teacher_logits, labels)
        gradients = tuple(stack.differentiator.compute_gradient())
        if len(gradients) != len(parameters):
            raise RuntimeError(
                "Expected one KL gradient per trainable conductance tensor. "
                f"Provided value: gradients={len(gradients)}, "
                f"parameters={len(parameters)}."
            )
        for binding, parameter, gradient in zip(bindings, parameters, gradients):
            if tuple(gradient.shape) != tuple(parameter.state.shape) or not bool(
                torch.isfinite(gradient).all()
            ):
                raise ValueError(
                    "Expected every replay gradient to be finite and match its "
                    "parameter tensor. Provided value: "
                    f"key={binding.key!r}, gradient_shape={tuple(gradient.shape)!r}, "
                    f"parameter_shape={tuple(parameter.state.shape)!r}."
                )
            flat = gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            chunks[binding.key].append(flat)
            absolute = flat.abs().to(torch.float64)
            batch_rows.append(
                {
                    "batch_index": batch_index,
                    "parameter_name": binding.key,
                    "batch_examples": int(labels.shape[0]),
                    "gradient_l2": float(torch.linalg.vector_norm(absolute).item()),
                    "gradient_rms": float(absolute.square().mean().sqrt().item()),
                    "gradient_mean": float(flat.to(torch.float64).mean().item()),
                    "positive_fraction": float((flat > 0.0).to(torch.float64).mean().item()),
                    "negative_fraction": float((flat < 0.0).to(torch.float64).mean().item()),
                    "zero_fraction": float((flat == 0.0).to(torch.float64).mean().item()),
                    "near_zero_fraction_abs_le_1e_12": float(
                        (absolute <= 1e-12).to(torch.float64).mean().item()
                    ),
                }
            )
        example_count += int(labels.shape[0])
        batch_count = batch_index + 1
    if batch_count == 0:
        raise ValueError("Expected replay to process validation batches. Provided value: 0.")

    values_by_parameter = {
        key: torch.cat(parts) for key, parts in chunks.items()
    }
    threshold_candidates: dict[str, dict[str, float]] = {}
    summary_rows: list[dict[str, Any]] = []
    for key, values in values_by_parameter.items():
        positive = values[values > 0.0]
        absolute = values.abs()
        matching_batches = [
            row for row in batch_rows if row["parameter_name"] == key
        ]
        quantiles = {
            f"p{str(100.0 * item).replace('.', '_')}": _quantile(positive, item)
            for item in _POSITIVE_QUANTILES
        }
        threshold_candidates[key] = {
            "positive_p90": _quantile(positive, 0.9),
            "positive_p99": _quantile(positive, 0.99),
            "positive_p99_9": _quantile(positive, 0.999),
        }
        row: dict[str, Any] = {
            "checkpoint_epoch": 0,
            "checkpoint_role": "isotonic_initialization",
            "parameter_name": key,
            "is_bias": False,
            "values": int(values.numel()),
            "positive_values": int(positive.numel()),
            "positive_fraction": float((values > 0.0).to(torch.float64).mean().item()),
            "negative_fraction": float((values < 0.0).to(torch.float64).mean().item()),
            "zero_fraction": float((values == 0.0).to(torch.float64).mean().item()),
            "near_zero_fraction_abs_le_1e_12": float(
                (absolute <= 1e-12).to(torch.float64).mean().item()
            ),
            "gradient_mean": float(values.to(torch.float64).mean().item()),
            "gradient_rms": float(
                values.to(torch.float64).square().mean().sqrt().item()
            ),
            "gradient_minimum": float(values.min().item()),
            "gradient_maximum": float(values.max().item()),
            "positive_gradient_rms": float(
                positive.to(torch.float64).square().mean().sqrt().item()
            ),
            "absolute_gradient_p50": _quantile(absolute, 0.5),
            "absolute_gradient_p90": _quantile(absolute, 0.9),
            "absolute_gradient_p99": _quantile(absolute, 0.99),
            "checkpoint_path": str(weights_path),
            "checkpoint_file_sha256": checkpoint_sha_before,
            "selected_source_indices_sha256": cohort_digest.hexdigest(),
        }
        row.update({f"positive_gradient_{name}": value for name, value in quantiles.items()})
        for metric in ("gradient_l2", "gradient_rms", "near_zero_fraction_abs_le_1e_12"):
            spread = _spread([float(item[metric]) for item in matching_batches])
            row.update(
                {f"batch_{metric}_{name}": value for name, value in spread.items()}
            )
        summary_rows.append(row)

    parameter_sha_after = _catalog_sha256(stack)
    checkpoint_sha_after = sha256_file(weights_path)
    unchanged = (
        parameter_sha_before == parameter_sha_after
        and checkpoint_sha_before == checkpoint_sha_after
    )
    if not unchanged:
        raise RuntimeError(
            "Expected gradient replay to leave parameters and checkpoint bytes "
            "unchanged. Provided value: unchanged=false."
        )

    candidate_arms = {
        label: {
            key: threshold_candidates[key][label]
            for key in threshold_candidates
        }
        for label in ("positive_p90", "positive_p99", "positive_p99_9")
    }
    report = {
        "schema": "mnist-relu-drn-gradient-threshold-calibration",
        "schema_version": 1,
        "semantics": {
            "gradient": "raw_unscaled_teacher_kl_gradient",
            "gate": "strict_gradient_greater_than_parameter_threshold",
            "update": "at_most_one_pulse_down_per_eligible_cell_per_minibatch",
            "optimizer_step_called": False,
        },
        "inputs": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "weights_path": str(weights_path),
            "weights_sha256_before": checkpoint_sha_before,
            "weights_sha256_after": checkpoint_sha_after,
            "teacher_weights_path": str(teacher_path),
            "teacher_weights_sha256": sha256_file(teacher_path),
        },
        "cohort": {
            "split": "validation",
            "data_seed": spec.runtime.data_seed,
            "batch_size": spec.data.batch_size,
            "batches": batch_count,
            "examples": example_count,
            "maximum_batches": args.maximum_batches,
            "ordered_tensor_sha256": cohort_digest.hexdigest(),
        },
        "replay": {
            "device": spec.runtime.device,
            "solver_mode": spec.solver.mode,
            "training_iterations": spec.solver.training_iterations,
            "overrelaxation_factor": spec.solver.overrelaxation_factor,
            "fixed_logit_gain": stack.cost.gain,
            "parameter_sha256_before": parameter_sha_before,
            "parameter_sha256_after": parameter_sha_after,
            "parameters_unchanged": parameter_sha_before == parameter_sha_after,
            "checkpoint_unchanged": checkpoint_sha_before == checkpoint_sha_after,
        },
        "threshold_candidates": candidate_arms,
        "parameter_summaries": summary_rows,
    }
    _write_csv(output_dir / "gradient_batches.csv", batch_rows)
    _write_csv(output_dir / "gradient_parameter_summary.csv", summary_rows)
    (output_dir / "threshold_calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_positive_ccdf(
        values_by_parameter,
        threshold_candidates,
        output_dir / "positive_gradient_threshold_ccdf.png",
    )
    print(
        json.dumps(
            {
                "batches": batch_count,
                "examples": example_count,
                "parameters_unchanged": True,
                "threshold_candidates": candidate_arms,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
