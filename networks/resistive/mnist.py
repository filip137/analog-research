from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
import math
import torch
from networks.function.network import Network
from networks.resistive.builders import ParameterCatalog, build_deep_resistive_energy
from networks.resistive.solver import CustomQuadraticMinimizer, MinimizerSettings
from networks.variable.parameter import DenseWeight


def paired_scores(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 2 or output.shape[1] % 2:
        raise ValueError(
            "Expected DRN output to have shape (batch, 2 * classes). "
            f"Provided value: {tuple(output.shape)!r}."
        )
    return output[:, 0::2] - output[:, 1::2]

def _minimizer(energy, *, iterations: int, mode: str, overrelaxation: float):
    settings = MinimizerSettings(
        rel_tol=1e-5,
        vn_tol=1e-6,
        use_polish=False,
        max_newton_iters=32,
        z_thresh=1e10,
        exp_clip=1e5,
        dynamic_polish=False,
        overrelaxation_reject_steps=False,
        overrelaxation_reject_max_tries=3,
        overrelaxation_reject_shrink=0.5,
        overrelaxation_reject_eps=0.0,
    )
    return CustomQuadraticMinimizer(
        fn=energy,
        free_layers=list(energy.layers()[1:]),
        num_iterations=iterations,
        mode=mode,
        non_linearity="perfect_diode",
        quadratic_diode_param={},
        exponential_diode_param={},
        voltage_amp=energy._voltage_amp,
        current_amp=energy._current_amp,
        hard_sigmoid_param={},
        iv_data=None,
        iv_data_path=None,
        double_diode_updater=None,
        adaptive_equilibrium=False,
        overrelaxation_factor=overrelaxation,
        single_diode_updater=None,
        minimizer_settings=settings,
    )

def signed_dual_rail_lift(
    weight: torch.Tensor,
    *,
    target_layout: str,
) -> torch.Tensor:
    """Lift one signed logical matrix into the physical dual-rail topology."""

    if weight.ndim != 2 or target_layout not in {"halves", "paired"}:
        raise ValueError(
            "Expected a rank-2 weight and target_layout 'halves' or 'paired'. "
            f"Provided value: shape={tuple(weight.shape)!r}, "
            f"target_layout={target_layout!r}."
        )
    inputs, outputs = weight.shape
    result = weight.new_zeros((2 * inputs, 2 * outputs))
    positive = weight.clamp_min(0.0)
    negative = (-weight).clamp_min(0.0)
    plus_columns = (
        torch.arange(outputs, device=weight.device)
        if target_layout == "halves"
        else torch.arange(outputs, device=weight.device) * 2
    )
    minus_columns = (
        plus_columns + outputs
        if target_layout == "halves"
        else plus_columns + 1
    )
    result[:inputs, plus_columns] = positive
    result[:inputs, minus_columns] = negative
    result[inputs:, plus_columns] = negative
    result[inputs:, minus_columns] = positive
    return result

def signed_differential_lift(
    weight: torch.Tensor,
    *,
    target_layout: str,
) -> torch.Tensor:
    """Return the signed effective matrix implemented by a G+/G- pair."""

    positive_lift = signed_dual_rail_lift(
        weight,
        target_layout=target_layout,
    )
    negative_lift = signed_dual_rail_lift(
        -weight,
        target_layout=target_layout,
    )
    return 0.5 * (positive_lift - negative_lift)

def collapse_clamped_dual_rail_input(
    conductance_plus: torch.Tensor,
    conductance_minus: torch.Tensor,
) -> torch.Tensor:
    """Collapse an eight-device signed input block into four rail edges.

    The source rows must use the repository's ``[x, -x]`` halves layout.
    At a clamped input boundary, the negative port of the positive source rail
    is the negative source rail, and conversely.  Regrouping branches that see
    the same voltage gives the ordinary dual-rail conductance tensor returned
    here.  It preserves the post-layer energy and KCL on the antipodal input
    subspace.  It does not identify source-side currents of independent
    dynamic rails and therefore must not be applied to an interior edge.
    """

    valid = (
        isinstance(conductance_plus, torch.Tensor)
        and isinstance(conductance_minus, torch.Tensor)
        and conductance_plus.ndim == 2
        and conductance_minus.ndim == 2
        and conductance_plus.shape == conductance_minus.shape
        and conductance_plus.shape[0] % 2 == 0
        and conductance_plus.device == conductance_minus.device
        and conductance_plus.dtype == conductance_minus.dtype
        and bool(torch.isfinite(conductance_plus).all())
        and bool(torch.isfinite(conductance_minus).all())
        and bool((conductance_plus >= 0.0).all())
        and bool((conductance_minus >= 0.0).all())
    )
    if not valid:
        plus_shape = (
            tuple(conductance_plus.shape)
            if isinstance(conductance_plus, torch.Tensor)
            else None
        )
        minus_shape = (
            tuple(conductance_minus.shape)
            if isinstance(conductance_minus, torch.Tensor)
            else None
        )
        raise ValueError(
            "Expected finite non-negative rank-2 G+/G- tensors with "
            "identical dtype, device, shape, and an even source dimension. "
            "Provided value: "
            f"plus_shape={plus_shape!r}, minus_shape={minus_shape!r}."
        )

    logical_inputs = conductance_plus.shape[0] // 2
    plus_source = slice(0, logical_inputs)
    minus_source = slice(logical_inputs, 2 * logical_inputs)
    effective_plus_source = (
        conductance_plus[plus_source]
        + conductance_minus[minus_source]
    )
    effective_minus_source = (
        conductance_minus[plus_source]
        + conductance_plus[minus_source]
    )
    return torch.cat(
        (effective_plus_source, effective_minus_source),
        dim=0,
    )

def _normalized(weight: torch.Tensor) -> torch.Tensor:
    maximum = weight.detach().abs().max()
    if float(maximum.item()) == 0.0:
        raise ValueError("Expected teacher matrix to contain a non-zero weight.")
    return weight.detach() / maximum

def mapped_conductances(
    teacher_weights: tuple[torch.Tensor, torch.Tensor],
    *,
    encoding: str,
    scale_fractions: tuple[float, float],
    conductance_min: float,
    conductance_max: float,
    range_placement: str = "lower",
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    targets: list[torch.Tensor] = []
    layers = []
    layouts = ("halves", "paired")
    for layer_index, (teacher_weight, fraction, layout) in enumerate(
        zip(teacher_weights, scale_fractions, layouts)
    ):
        logical = _normalized(teacher_weight)
        if encoding == "single":
            lifted = signed_dual_rail_lift(logical, target_layout=layout)
            peak = float(lifted.max().item())
            largest_scale = (conductance_max - conductance_min) / peak
            scale = fraction * largest_scale
            if range_placement == "lower":
                baseline = conductance_min
            elif range_placement == "centered":
                baseline = conductance_min + 0.5 * (
                    conductance_max - conductance_min - scale * peak
                )
            else:
                raise ValueError(
                    "Expected range_placement to be 'lower' or 'centered'. "
                    f"Provided value: {range_placement!r}."
                )
            layer_targets = (baseline + scale * lifted,)
        elif encoding == "differential":
            if range_placement != "lower":
                raise ValueError(
                    "Expected differential mapping range_placement to equal "
                    f"'lower'. Provided value: {range_placement!r}."
                )
            lifted = signed_differential_lift(logical, target_layout=layout)
            peak = float(lifted.abs().max().item())
            largest_scale = (conductance_max - conductance_min) / peak
            scale = fraction * largest_scale
            layer_targets = (
                conductance_min + scale * lifted.clamp_min(0.0),
                conductance_min + scale * (-lifted).clamp_min(0.0),
            )
        else:
            raise ValueError(
                "Expected encoding to be 'single' or 'differential'. "
                f"Provided value: {encoding!r}."
            )
        targets.extend(layer_targets)
        layers.append(
            {
                "layer": layer_index,
                "target_layout": layout,
                "teacher_absmax": float(teacher_weight.abs().max().item()),
                "scale_fraction": fraction,
                "largest_nonclipping_scale_s": largest_scale,
                "selected_scale_s": scale,
                "range_placement": range_placement,
                "selected_baseline_s": (
                    baseline if encoding == "single" else conductance_min
                ),
            }
        )
    return tuple(targets), {"encoding": encoding, "layers": layers}

def apply_targets(catalog: ParameterCatalog, targets: tuple[torch.Tensor, ...]) -> None:
    if len(catalog.trainable) != len(targets):
        raise ValueError(
            "Expected one mapped conductance target per trainable tensor. "
            f"Provided value: bindings={len(catalog.trainable)}, targets={len(targets)}."
        )
    with torch.no_grad():
        for binding, target in zip(catalog.trainable, targets):
            binding.state.copy_(target.to(device=binding.state.device, dtype=binding.state.dtype))

def conductance_statistics(catalog: ParameterCatalog, *, encoding: str) -> dict[str, Any]:
    result: dict[str, Any] = {"encoding": encoding, "parameters": {}}
    for binding in catalog.trainable:
        state = binding.state.detach().to(torch.float64)
        raw_lower = binding.parameter.min_cond
        raw_upper = binding.parameter.max_cond
        lower = None if raw_lower is None else float(raw_lower)
        upper = None if raw_upper is None else float(raw_upper)
        result["parameters"][binding.key] = {
            "role": binding.role,
            "minimum_s": float(state.min().item()),
            "maximum_s": float(state.max().item()),
            "mean_s": float(state.mean().item()),
            "rms_s": float(torch.sqrt(state.square().mean()).item()),
            "lower_bound_fraction": (
                None
                if lower is None
                else float((state <= lower).to(torch.float64).mean().item())
            ),
            "upper_bound_fraction": (
                None
                if upper is None
                else float((state >= upper).to(torch.float64).mean().item())
            ),
        }
    if encoding == "differential":
        dense = tuple(
            binding
            for binding in catalog.trainable
            if isinstance(binding.parameter, DenseWeight)
        )
        pairs = list(zip(dense[0::2], dense[1::2]))
        result["differential_layers"] = []
        for plus, minus in pairs:
            difference = plus.state.detach().to(torch.float64) - minus.state.detach().to(torch.float64)
            total = plus.state.detach().to(torch.float64) + minus.state.detach().to(torch.float64)
            result["differential_layers"].append(
                {
                    "difference_rms_s": float(torch.sqrt(difference.square().mean()).item()),
                    "sum_mean_s": float(total.mean().item()),
                    "sum_rms_s": float(torch.sqrt(total.square().mean()).item()),
                }
            )
    return result


def build_student_network(model, solver, device):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Expected config.runtime.device='cuda' to have an available CUDA "
            "device. Provided value: torch.cuda.is_available() is false."
        )
    differential = (0, 1) if model.encoding == "differential" else ()
    bundle = build_deep_resistive_energy(
        layer_shapes=[(item,) for item in model.dims],
        weight_gains=[1.0, 1.0],
        input_gain=model.input_gain,
        non_linearity=model.non_linearity,
        exponential_diode_param=dict(model.exponential_diode_param),
        quadratic_diode_param=dict(model.quadratic_diode_param),
        hard_sigmoid_param=dict(model.hard_sigmoid_param),
        voltage_amp=model.voltage_amp,
        current_amp=model.current_amp,
        weight_min=model.conductance_min,
        weight_max=model.conductance_max,
        differential_dense_edges=differential,
        include_biases=model.include_biases,
        legacy_process_index_amplification=(
            getattr(model, "amplification_indexing", "logical")
            == "legacy_process_global"
        ),
    )
    bundle.energy.set_device(device)
    network = Network(bundle.energy)
    inference = _minimizer(
        bundle.energy,
        iterations=solver.inference_iterations,
        mode=solver.mode,
        overrelaxation=solver.overrelaxation_factor,
    )
    training = _minimizer(
        bundle.energy,
        iterations=solver.training_iterations,
        mode=solver.mode,
        overrelaxation=solver.overrelaxation_factor,
    )
    return bundle, network, inference, training, device
