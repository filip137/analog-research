#!/usr/bin/env python3
"""Replay the MNIST DRN under finite-conductance-floor ablations.

This is a validation-only analysis utility for the one-hidden-layer,
perfect-diode MNIST model used by the IBM PCM/CMO study.  It implements the
same asynchronous closed-form coordinate updates as the repository runtime,
while allowing the conductances used in the MVM numerator and row-sum
denominator to differ.  That separation is useful for diagnosing active
reference-current and denominator-correction proposals; it is not presented
as a passive energy function.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ebl.registry import load_experiment_config
from ebl.runtime.small_network.components import build_data


W1_KEY = "base.dense_weight.0"
W2_KEY = "base.dense_weight.1"
BIAS_KEY = "base.bias.0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze finite-conductance-floor mechanisms on the checked-in "
            "one-hidden-layer MNIST DRN."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "examples/small_drn/mnist_ibm_devices/hwa_from_fp32.json"
        ),
    )
    parser.add_argument("--clean-weights", type=Path, required=True)
    parser.add_argument("--literal-weights", type=Path, required=True)
    parser.add_argument("--affine-weights", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--g-min-us", type=float, default=9.0)
    parser.add_argument("--g-max-us", type=float, default=88.199997)
    parser.add_argument(
        "--drn-conductance-at-g-max",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--replay-batch-size", type=int, default=512)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_weights(
    path: Path,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(
            "Expected a named-weights checkpoint path naming an existing "
            f"file. Provided value: {str(resolved)!r}."
        )
    payload = torch.load(
        resolved,
        map_location=device,
        weights_only=True,
    )
    raw = payload.get("weights") if isinstance(payload, dict) else None
    expected = {W1_KEY, W2_KEY, BIAS_KEY}
    if not isinstance(raw, dict) or set(raw) != expected:
        provided = tuple(raw) if isinstance(raw, dict) else raw
        raise ValueError(
            "Expected named weights to contain exactly the MNIST DRN keys "
            f"{tuple(sorted(expected))!r}. Provided value: {provided!r}."
        )
    weights = {
        key: value.detach().to(device=device, dtype=torch.float32)
        for key, value in raw.items()
    }
    if (
        tuple(weights[W1_KEY].shape) != (1568, 100)
        or tuple(weights[W2_KEY].shape) != (100, 20)
        or tuple(weights[BIAS_KEY].shape) != (100,)
    ):
        raise ValueError(
            "Expected MNIST DRN tensor shapes W1=(1568,100), "
            "W2=(100,20), bias=(100,). Provided value: "
            f"W1={tuple(weights[W1_KEY].shape)!r}, "
            f"W2={tuple(weights[W2_KEY].shape)!r}, "
            f"bias={tuple(weights[BIAS_KEY].shape)!r}."
        )
    return weights


def _load_test_set(
    config: Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    _, document = load_experiment_config(config)
    common = document.common
    if (
        common.model.dims != (1568, 100, 20)
        or common.model.non_linearity.type != "perfect_diode"
        or common.model.voltage_amp != 1.0
        or common.model.current_amp != 1.0
        or common.solver.minimizer_mode != "asynchronous"
    ):
        raise ValueError(
            "Expected the one-hidden-layer perfect-diode MNIST configuration "
            "with dims=(1568,100,20), unit voltage/current amplification, "
            "and asynchronous minimization. Provided value: "
            f"dims={common.model.dims!r}, "
            f"non_linearity={common.model.non_linearity.type!r}, "
            f"voltage_amp={common.model.voltage_amp!r}, "
            f"current_amp={common.model.current_amp!r}, "
            f"mode={common.solver.minimizer_mode!r}."
        )
    replay_common = replace(
        common,
        runtime=replace(common.runtime, device=device.type),
        data=replace(common.data, shuffle=False),
    )
    data = build_data(replay_common)
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch_inputs, batch_targets in data.held_out_loader:
        inputs.append(batch_inputs)
        targets.append(batch_targets)
    return (
        torch.cat(inputs).to(device=device, dtype=torch.float32),
        torch.cat(targets).to(device=device, dtype=torch.long),
        float(common.model.input_gain),
        int(common.solver.inference_iterations),
    )


@torch.no_grad()
def replay(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    numerator_w1: torch.Tensor,
    numerator_w2: torch.Tensor,
    denominator_w1: torch.Tensor | None = None,
    denominator_w2: torch.Tensor | None = None,
    bias: torch.Tensor,
    input_gain: float,
    iterations: int,
    batch_size: int,
) -> dict[str, float | int]:
    """Replay the exact current coordinate order with optional active terms."""

    denominator_w1 = (
        numerator_w1 if denominator_w1 is None else denominator_w1
    )
    denominator_w2 = (
        numerator_w2 if denominator_w2 is None else denominator_w2
    )
    hidden_degree = denominator_w1.sum(0) + denominator_w2.sum(1)
    output_degree = denominator_w2.sum(0)
    if numerator_w2.shape[1] % 2:
        raise ValueError(
            "Expected an even output width for paired differential scores. "
            f"Provided value: {numerator_w2.shape[1]!r}."
        )
    if not torch.isfinite(hidden_degree).all() or not torch.isfinite(
        output_degree
    ).all():
        raise ValueError(
            "Expected all coordinate-update conductance sums to be finite. "
            "Provided value contains a non-finite sum."
        )
    if (hidden_degree <= 0).any() or (output_degree <= 0).any():
        raise ValueError(
            "Expected every coordinate-update conductance sum to be positive. "
            "Provided value contains a disconnected hidden or output node."
        )

    correct = 0
    hidden_square_sum = 0.0
    output_square_sum = 0.0
    hidden_count = 0
    output_count = 0
    for start in range(0, inputs.shape[0], batch_size):
        logical = inputs[start : start + batch_size]
        labels = targets[start : start + batch_size]
        physical_input = input_gain * torch.cat((logical, -logical), dim=1)
        hidden = torch.zeros(
            (logical.shape[0], numerator_w1.shape[1]),
            dtype=logical.dtype,
            device=logical.device,
        )
        output = torch.zeros(
            (logical.shape[0], numerator_w2.shape[1]),
            dtype=logical.dtype,
            device=logical.device,
        )
        for _ in range(iterations):
            hidden = (
                physical_input @ numerator_w1
                + output @ numerator_w2.transpose(0, 1)
                + bias
            ) / hidden_degree
            midpoint = hidden.shape[1] // 2
            hidden = torch.cat(
                (
                    hidden[:, :midpoint].clamp_min(0.0),
                    hidden[:, midpoint:].clamp_max(0.0),
                ),
                dim=1,
            )
            output = hidden @ numerator_w2 / output_degree

        paired = output.reshape(
            output.shape[0],
            output.shape[1] // 2,
            2,
        )
        scores = paired[..., 0] - paired[..., 1]
        correct += int((scores.argmax(1) == labels).sum())
        hidden_square_sum += float(hidden.square().sum())
        output_square_sum += float(output.square().sum())
        hidden_count += hidden.numel()
        output_count += output.numel()

    return {
        "accuracy": correct / targets.numel(),
        "correct": correct,
        "examples": targets.numel(),
        "hidden_degree_mean": float(hidden_degree.mean()),
        "hidden_degree_min": float(hidden_degree.min()),
        "hidden_degree_max": float(hidden_degree.max()),
        "output_degree_mean": float(output_degree.mean()),
        "output_degree_min": float(output_degree.min()),
        "output_degree_max": float(output_degree.max()),
        "hidden_voltage_rms": math.sqrt(hidden_square_sum / hidden_count),
        "output_voltage_rms": math.sqrt(output_square_sum / output_count),
    }


def _top_k_mask(weight: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0 or count > weight.shape[0]:
        raise ValueError(
            "Expected top-k selector count in [1, rows]. Provided value: "
            f"count={count!r}, rows={weight.shape[0]!r}."
        )
    if count == weight.shape[0]:
        return torch.ones_like(weight, dtype=torch.bool)
    indices = torch.topk(weight, count, dim=0).indices
    mask = torch.zeros_like(weight, dtype=torch.bool)
    mask.scatter_(0, indices, True)
    return mask


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(
            "Expected at least one CSV row. Provided value: empty rows."
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_scenarios(
    path: Path,
    scenarios: dict[str, dict[str, Any]],
) -> None:
    names = list(scenarios)
    values = [100.0 * float(scenarios[name]["accuracy"]) for name in names]
    fig, axis = plt.subplots(figsize=(9.0, 5.8))
    positions = list(range(len(names)))
    axis.barh(positions, values, color="#4C78A8")
    axis.set_yticks(positions, labels=[name.replace("_", " ") for name in names])
    axis.invert_yaxis()
    axis.set_xlabel("MNIST accuracy (%)")
    axis.set_xlim(max(0.0, min(values) - 3.0), 100.0)
    axis.grid(axis="x", alpha=0.25)
    for position, value in zip(positions, values):
        axis.text(value + 0.15, position, f"{value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_cancellation(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    cancellation = [float(row["cancellation_fraction"]) for row in rows]
    for key, label, color in (
        ("both_accuracy", "Numerator and denominator", "#4C78A8"),
        ("numerator_only_accuracy", "Numerator only", "#F58518"),
        ("denominator_only_accuracy", "Denominator only", "#54A24B"),
    ):
        axis.plot(
            cancellation,
            [100.0 * float(row[key]) for row in rows],
            marker="o",
            label=label,
            color=color,
        )
    axis.set_xlabel("Subtracted fraction of nominal floor")
    axis.set_ylabel("MNIST accuracy (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_floor_ratio(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    measured_ratio: float,
) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    ratios = [100.0 * float(row["floor_to_gmax_ratio"]) for row in rows]
    axis.plot(
        ratios,
        [100.0 * float(row["hard_floor_accuracy"]) for row in rows],
        marker="o",
        label="Literal hard floor",
        color="#E45756",
    )
    axis.plot(
        ratios,
        [100.0 * float(row["affine_floor_accuracy"]) for row in rows],
        marker="o",
        label="Affine retained floor",
        color="#4C78A8",
    )
    axis.axvline(
        100.0 * measured_ratio,
        color="#777777",
        linestyle="--",
        label="CMO 9/88.2",
    )
    axis.set_xlabel("Minimum / maximum conductance (%)")
    axis.set_ylabel("MNIST accuracy (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = _parser().parse_args()
    if args.g_min_us <= 0.0 or args.g_max_us <= args.g_min_us:
        raise ValueError(
            "Expected 0 < --g-min-us < --g-max-us. Provided value: "
            f"g_min_us={args.g_min_us!r}, g_max_us={args.g_max_us!r}."
        )
    if args.replay_batch_size <= 0:
        raise ValueError(
            "Expected --replay-batch-size to be positive. Provided value: "
            f"{args.replay_batch_size!r}."
        )
    if args.drn_conductance_at_g_max <= 0.0:
        raise ValueError(
            "Expected --drn-conductance-at-g-max to be positive. Provided "
            f"value: {args.drn_conductance_at_g_max!r}."
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    clean = _load_weights(args.clean_weights, device=device)
    literal = _load_weights(args.literal_weights, device=device)
    affine = (
        None
        if args.affine_weights is None
        else _load_weights(args.affine_weights, device=device)
    )
    inputs, targets, input_gain, iterations = _load_test_set(
        args.config,
        device=device,
    )

    clean_w1 = clean[W1_KEY]
    clean_w2 = clean[W2_KEY]
    clean_bias = clean[BIAS_KEY]
    literal_w1 = literal[W1_KEY]
    literal_w2 = literal[W2_KEY]
    floor_ratio = args.g_min_us / args.g_max_us
    reference = args.drn_conductance_at_g_max
    floor = reference * floor_ratio
    span = 1.0 - floor_ratio

    scenarios: dict[str, dict[str, Any]] = {}

    def add_scenario(
        name: str,
        *,
        numerator_w1: torch.Tensor,
        numerator_w2: torch.Tensor,
        denominator_w1: torch.Tensor | None = None,
        denominator_w2: torch.Tensor | None = None,
        bias: torch.Tensor = clean_bias,
    ) -> None:
        scenarios[name] = replay(
            inputs,
            targets,
            numerator_w1=numerator_w1,
            numerator_w2=numerator_w2,
            denominator_w1=denominator_w1,
            denominator_w2=denominator_w2,
            bias=bias,
            input_gain=input_gain,
            iterations=iterations,
            batch_size=args.replay_batch_size,
        )

    add_scenario(
        "clean_hwa",
        numerator_w1=clean_w1,
        numerator_w2=clean_w2,
    )
    deterministic_literal_w1 = clean_w1.clamp_min(floor)
    deterministic_literal_w2 = clean_w2.clamp_min(floor)
    add_scenario(
        "hard_floor_deterministic",
        numerator_w1=deterministic_literal_w1,
        numerator_w2=deterministic_literal_w2,
    )
    add_scenario(
        "hard_floor_realized",
        numerator_w1=literal_w1,
        numerator_w2=literal_w2,
        bias=literal[BIAS_KEY],
    )
    affine_w1 = floor + span * clean_w1
    affine_w2 = floor + span * clean_w2
    add_scenario(
        "affine_floor_deterministic",
        numerator_w1=affine_w1,
        numerator_w2=affine_w2,
        bias=clean_bias,
    )
    if affine is not None:
        add_scenario(
            "affine_floor_realized",
            numerator_w1=affine[W1_KEY],
            numerator_w2=affine[W2_KEY],
            bias=affine[BIAS_KEY],
        )
    add_scenario(
        "affine_numerator_correction",
        numerator_w1=span * clean_w1,
        numerator_w2=span * clean_w2,
        denominator_w1=affine_w1,
        denominator_w2=affine_w2,
        bias=span * clean_bias,
    )
    add_scenario(
        "affine_denominator_correction",
        numerator_w1=affine_w1,
        numerator_w2=affine_w2,
        denominator_w1=span * clean_w1,
        denominator_w2=span * clean_w2,
        bias=span * clean_bias,
    )
    add_scenario(
        "affine_full_active_correction",
        numerator_w1=span * clean_w1,
        numerator_w2=span * clean_w2,
        denominator_w1=span * clean_w1,
        denominator_w2=span * clean_w2,
        bias=span * clean_bias,
    )

    selector_w1 = torch.where(
        _top_k_mask(clean_w1, 384),
        clean_w1.clamp_min(floor),
        torch.zeros_like(clean_w1),
    )
    selector_w2 = torch.where(
        _top_k_mask(clean_w2, 60),
        clean_w2.clamp_min(floor),
        torch.zeros_like(clean_w2),
    )
    add_scenario(
        "selector_topk_384_60",
        numerator_w1=selector_w1,
        numerator_w2=selector_w2,
    )

    cancellation_rows: list[dict[str, Any]] = []
    for fraction in (
        0.0,
        0.25,
        0.5,
        0.75,
        0.9,
        0.95,
        0.98,
        0.99,
        0.995,
        0.999,
        1.0,
    ):
        corrected_w1 = (literal_w1 - fraction * floor).clamp_min(1e-12)
        corrected_w2 = (literal_w2 - fraction * floor).clamp_min(1e-12)
        common = {
            "inputs": inputs,
            "targets": targets,
            "bias": literal[BIAS_KEY],
            "input_gain": input_gain,
            "iterations": iterations,
            "batch_size": args.replay_batch_size,
        }
        both = replay(
            numerator_w1=corrected_w1,
            numerator_w2=corrected_w2,
            **common,
        )
        numerator_only = replay(
            numerator_w1=corrected_w1,
            numerator_w2=corrected_w2,
            denominator_w1=literal_w1,
            denominator_w2=literal_w2,
            **common,
        )
        denominator_only = replay(
            numerator_w1=literal_w1,
            numerator_w2=literal_w2,
            denominator_w1=corrected_w1,
            denominator_w2=corrected_w2,
            **common,
        )
        cancellation_rows.append(
            {
                "cancellation_fraction": fraction,
                "both_accuracy": both["accuracy"],
                "numerator_only_accuracy": numerator_only["accuracy"],
                "denominator_only_accuracy": denominator_only["accuracy"],
                "both_hidden_voltage_rms": both["hidden_voltage_rms"],
                "both_output_voltage_rms": both["output_voltage_rms"],
            }
        )

    gain_iteration_rows: list[dict[str, Any]] = []
    for replay_iterations in (1, 2, 4, 8, 16, 32):
        for gain in (10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0):
            result = replay(
                inputs,
                targets,
                numerator_w1=literal_w1,
                numerator_w2=literal_w2,
                bias=literal[BIAS_KEY],
                input_gain=gain,
                iterations=replay_iterations,
                batch_size=args.replay_batch_size,
            )
            gain_iteration_rows.append(
                {
                    "iterations": replay_iterations,
                    "input_gain": gain,
                    "accuracy": result["accuracy"],
                }
            )

    floor_ratio_rows: list[dict[str, Any]] = []
    ratios = (
        0.0,
        0.001,
        0.003,
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        floor_ratio,
        0.125,
        0.15,
        0.2,
    )
    for ratio in sorted(set(ratios)):
        swept_floor = reference * ratio
        swept_span = 1.0 - ratio
        hard = replay(
            inputs,
            targets,
            numerator_w1=clean_w1.clamp_min(swept_floor),
            numerator_w2=clean_w2.clamp_min(swept_floor),
            bias=clean_bias,
            input_gain=input_gain,
            iterations=iterations,
            batch_size=args.replay_batch_size,
        )
        affine_result = replay(
            inputs,
            targets,
            numerator_w1=swept_floor + swept_span * clean_w1,
            numerator_w2=swept_floor + swept_span * clean_w2,
            bias=clean_bias,
            input_gain=input_gain,
            iterations=iterations,
            batch_size=args.replay_batch_size,
        )
        floor_ratio_rows.append(
            {
                "floor_to_gmax_ratio": ratio,
                "hard_floor_accuracy": hard["accuracy"],
                "affine_floor_accuracy": affine_result["accuracy"],
                "hard_hidden_voltage_rms": hard["hidden_voltage_rms"],
                "affine_hidden_voltage_rms": affine_result[
                    "hidden_voltage_rms"
                ],
                "hard_output_voltage_rms": hard["output_voltage_rms"],
                "affine_output_voltage_rms": affine_result[
                    "output_voltage_rms"
                ],
            }
        )

    clean_hidden_degree = clean_w1.sum(0) + clean_w2.sum(1)
    clean_output_degree = clean_w2.sum(0)
    literal_hidden_degree = literal_w1.sum(0) + literal_w2.sum(1)
    literal_output_degree = literal_w2.sum(0)
    hidden_floor_degree = floor * (clean_w1.shape[0] + clean_w2.shape[1])
    output_floor_degree = floor * clean_w2.shape[0]

    def distribution(values: torch.Tensor) -> dict[str, float]:
        return {
            "minimum": float(values.min()),
            "median": float(values.median()),
            "maximum": float(values.max()),
        }

    floor_loading = {
        "floor_drn_units": floor,
        "hidden_floor_degree": hidden_floor_degree,
        "output_floor_degree": output_floor_degree,
        "clean_hidden_degree_mean": float(clean_hidden_degree.mean()),
        "clean_output_degree_mean": float(clean_output_degree.mean()),
        "affine_hidden_floor_fraction": distribution(
            hidden_floor_degree
            / (span * clean_hidden_degree + hidden_floor_degree)
        ),
        "affine_output_floor_fraction": distribution(
            output_floor_degree
            / (span * clean_output_degree + output_floor_degree)
        ),
        "literal_hidden_nominal_floor_fraction": distribution(
            hidden_floor_degree / literal_hidden_degree
        ),
        "literal_output_nominal_floor_fraction": distribution(
            output_floor_degree / literal_output_degree
        ),
        "clean_target_below_floor_fraction": {
            W1_KEY: float((clean_w1 < floor).float().mean()),
            W2_KEY: float((clean_w2 < floor).float().mean()),
        },
        "literal_realized_at_floor_fraction": {
            W1_KEY: float(
                torch.isclose(
                    literal_w1,
                    torch.as_tensor(
                        floor,
                        dtype=literal_w1.dtype,
                        device=literal_w1.device,
                    ),
                    rtol=0.0,
                    atol=8.0 * torch.finfo(literal_w1.dtype).eps,
                )
                .float()
                .mean()
            ),
            W2_KEY: float(
                torch.isclose(
                    literal_w2,
                    torch.as_tensor(
                        floor,
                        dtype=literal_w2.dtype,
                        device=literal_w2.device,
                    ),
                    rtol=0.0,
                    atol=8.0 * torch.finfo(literal_w2.dtype).eps,
                )
                .float()
                .mean()
            ),
        },
    }
    if affine is not None:
        floor_loading["affine_realized_at_floor_fraction"] = {
            key: float(
                torch.isclose(
                    affine[key],
                    torch.as_tensor(
                        floor,
                        dtype=affine[key].dtype,
                        device=affine[key].device,
                    ),
                    rtol=0.0,
                    atol=8.0 * torch.finfo(affine[key].dtype).eps,
                )
                .float()
                .mean()
            )
            for key in (W1_KEY, W2_KEY)
        }

    source_paths = {
        "analysis_script": Path(__file__).resolve(),
        "config": args.config.expanduser().resolve(),
        "clean_weights": args.clean_weights.expanduser().resolve(),
        "literal_weights": args.literal_weights.expanduser().resolve(),
    }
    if args.affine_weights is not None:
        source_paths["affine_weights"] = (
            args.affine_weights.expanduser().resolve()
        )
    summary = {
        "schema": "mnist-conductance-floor-analysis",
        "schema_version": 5,
        "device": device.type,
        "torch_version": str(torch.__version__),
        "g_min_us": args.g_min_us,
        "g_max_us": args.g_max_us,
        "drn_conductance_at_g_max": reference,
        "input_gain": input_gain,
        "iterations": iterations,
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        },
        "floor_loading": floor_loading,
        "scenarios": scenarios,
        "literal_cancellation_sweep": cancellation_rows,
        "gain_iteration_sweep": gain_iteration_rows,
        "floor_ratio_sweep": floor_ratio_rows,
        "interpretation": {
            "passive_floor_term": (
                "A uniform floor adds a complete-bipartite Laplacian and "
                "changes both numerator currents and conductance-sum divisors."
            ),
            "mapping_distinction": (
                "Hard-floor clipping destroys sub-floor weight information; "
                "affine floor programming retains those distinctions while "
                "leaving the physical floor in the passive equations."
            ),
            "active_correction": (
                "Numerator-only MVM subtraction is not exact for a DRN; exact "
                "floor cancellation also corrects the row-sum denominator "
                "and preserves the relative bias-current scaling."
            ),
            "correction_ablation_bias": (
                "The affine numerator/denominator correction scenarios scale "
                "the bias by the affine span so that full correction is an "
                "exact common rescaling of the clean coordinate equation."
            ),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "cancellation_sweep.csv", cancellation_rows)
    _write_csv(output_dir / "gain_iteration_sweep.csv", gain_iteration_rows)
    _write_csv(output_dir / "floor_ratio_sweep.csv", floor_ratio_rows)
    _write_csv(
        output_dir / "scenarios.csv",
        [
            {"scenario": name, **metrics}
            for name, metrics in scenarios.items()
        ],
    )
    _plot_scenarios(output_dir / "scenario_accuracy.png", scenarios)
    _plot_cancellation(
        output_dir / "literal_cancellation_sweep.png",
        cancellation_rows,
    )
    _plot_floor_ratio(
        output_dir / "floor_ratio_sweep.png",
        floor_ratio_rows,
        measured_ratio=floor_ratio,
    )
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
