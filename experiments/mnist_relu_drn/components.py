"""Numerical composition and mapping for teacher-initialized DRN KD."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_relu_drn.config import (
    MEASURED_BACKENDS,
    StudentTrainSpec,
)
from labs.custom_minimizer import CustomQuadraticMinimizer, MinimizerSettings
from model.function.interaction import Function
from model.function.network import Network
from model.resistive.builders import ModelBundle, ParameterCatalog, build_deep_resistive_energy
from model.resistive.device_config import parse_device_programming_config
from model.variable.parameter import Bias, DenseWeight
from training.measured_trace import (
    MeasuredCohortAOptimizer,
    MeasuredCohortAOnePulseDownOptimizer,
    MeasuredCohortBOptimizer,
)
from training.program_verify import ProgramVerifyOptimizer
from training.core.sgd import Backprop
from training.sign_sgd import SignSGD


def paired_scores(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 2 or output.shape[1] % 2:
        raise ValueError(
            "Expected DRN output to have shape (batch, 2 * classes). "
            f"Provided value: {tuple(output.shape)!r}."
        )
    return output[:, 0::2] - output[:, 1::2]


class TeacherKLDivergence(Function):
    """Pure KL(teacher || student) over calibrated paired DRN scores."""

    def __init__(self, output_layer, *, gain: float = 1.0) -> None:
        self._layer = output_layer
        self._num_classes = int(output_layer.shape[0]) // 2
        self._gain = float(gain)
        self._teacher_logits: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        Function.__init__(self, [output_layer], [])

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(
                "Expected fixed readout gain to be positive and finite. "
                f"Provided value: {value!r}."
            )
        self._gain = numeric

    def set_teacher(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.ndim != 2 or logits.shape[1] != self._num_classes:
            raise ValueError(
                "Expected teacher logits with shape "
                f"(batch, {self._num_classes}). "
                f"Provided value: {tuple(logits.shape)!r}."
            )
        self._teacher_logits = logits.detach()
        self._labels = labels.detach().to(device=logits.device, dtype=torch.long)

    def set_target(self, labels: torch.Tensor) -> None:
        if self._teacher_logits is None:
            raise RuntimeError(
                "Expected teacher logits before setting KL labels. Provided value: none."
            )
        self._labels = labels.detach().to(
            device=self._teacher_logits.device,
            dtype=torch.long,
        )

    def student_logits(self) -> torch.Tensor:
        return paired_scores(self._layer.state) * self._gain

    def eval(self) -> torch.Tensor:
        if self._teacher_logits is None:
            raise RuntimeError("Expected teacher logits before evaluating KL.")
        teacher_log_prob = F.log_softmax(self._teacher_logits, dim=1)
        teacher_prob = teacher_log_prob.exp()
        student_log_prob = F.log_softmax(self.student_logits(), dim=1)
        return (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=1)

    def error_fn(self) -> torch.Tensor:
        if self._labels is None:
            raise RuntimeError("Expected labels before evaluating error.")
        return self.student_logits().argmax(dim=1).ne(self._labels)

    def _get_output(self) -> torch.Tensor:
        return self.student_logits()


@dataclass(frozen=True)
class StudentStack:
    bundle: ModelBundle
    network: Network
    minimizer: Any
    training_minimizer: Any
    cost: Any
    differentiator: Backprop
    optimizer: Any
    device: torch.device


@dataclass(frozen=True)
class BoundedDrnTeacher:
    """Frozen equilibrium teacher loaded from stable named DRN weights."""

    stack: StudentStack
    checkpoint_metadata: dict[str, Any]

    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.stack.network.set_input(
                inputs.to(self.stack.device, dtype=torch.float32),
                reset=True,
            )
            self.stack.minimizer.compute_equilibrium()
            return paired_scores(
                self.stack.bundle.energy.layers()[-1].state
            ).detach()

    def copy_named_weights_to(self, catalog: ParameterCatalog) -> None:
        source = self.stack.bundle.catalog
        expected = tuple(binding.key for binding in catalog.trainable)
        provided = tuple(binding.key for binding in source.trainable)
        if provided != expected:
            raise ValueError(
                "Expected bounded DRN teacher and student to expose the "
                "same stable trainable parameter keys. Provided value: "
                f"teacher={provided!r}, student={expected!r}."
            )
        with torch.no_grad():
            for key in expected:
                teacher_state = source.by_key[key].state
                student_state = catalog.by_key[key].state
                if teacher_state.shape != student_state.shape:
                    raise ValueError(
                        "Expected bounded DRN teacher and student parameter "
                        f"{key!r} to have the same shape. Provided value: "
                        f"teacher={tuple(teacher_state.shape)!r}, "
                        f"student={tuple(student_state.shape)!r}."
                    )
                student_state.copy_(teacher_state.to(student_state))


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


def _logical_optimizer(
    catalog: ParameterCatalog,
    *,
    encoding: str,
    rates: tuple[float, ...],
    sign_only: bool = False,
) -> torch.optim.Optimizer:
    bindings = catalog.trainable
    if encoding == "single":
        dense = tuple(
            binding
            for binding in bindings
            if isinstance(binding.parameter, DenseWeight)
        )
        biases = tuple(
            binding
            for binding in bindings
            if isinstance(binding.parameter, Bias)
        )
        if (
            len(dense) != 2
            or len(biases) not in {0, 1}
            or len(bindings) != len(dense) + len(biases)
            or len(rates) != len(bindings)
        ):
            raise ValueError(
                "Expected encoding='single' to expose two dense weights, "
                "zero or one hidden bias, and one learning rate per stable "
                "catalog binding. Provided value: "
                f"bindings={len(bindings)}, dense={len(dense)}, "
                f"biases={len(biases)}, rates={len(rates)}."
            )
        groups = [
            {"params": [binding.state], "lr": rates[index]}
            for index, binding in enumerate(bindings)
        ]
    else:
        if (
            len(bindings) != 4
            or any(
                not isinstance(binding.parameter, DenseWeight)
                for binding in bindings
            )
            or len(rates) != 2
        ):
            raise ValueError(
                "Expected encoding='differential' to expose four dense "
                "conductance tensors and two logical learning rates. "
                f"Provided value: bindings={len(bindings)}, rates={len(rates)}."
            )
        groups = [
            {
                "params": [bindings[2 * index].state, bindings[2 * index + 1].state],
                "lr": rates[index],
            }
            for index in range(2)
        ]
    if sign_only:
        return SignSGD(groups, lr=0.0)
    return torch.optim.SGD(groups, momentum=0.0, weight_decay=0.0)


def measured_optimizer_type(update_backend_type: str) -> type:
    """Resolve the explicitly registered measured optimizer implementation."""

    implementations = {
        "measured_cohort_a": MeasuredCohortAOptimizer,
        "measured_cohort_a_sign_sgd": MeasuredCohortAOptimizer,
        "measured_cohort_a_one_pulse_down": (
            MeasuredCohortAOnePulseDownOptimizer
        ),
        "measured_cohort_b": MeasuredCohortBOptimizer,
    }
    if update_backend_type not in implementations:
        raise ValueError(
            "Expected a registered MNIST measured update backend. Provided "
            f"value: {update_backend_type!r}."
        )
    return implementations[update_backend_type]


def build_student_stack(
    spec: Any,
    *,
    device_data_path=None,
    enable_measured: bool = True,
) -> StudentStack:
    device = torch.device(spec.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Expected config.runtime.device='cuda' to have an available CUDA "
            "device. Provided value: torch.cuda.is_available() is false."
        )
    differential = (0, 1) if spec.model.encoding == "differential" else ()
    bundle = build_deep_resistive_energy(
        layer_shapes=[(item,) for item in spec.model.dims],
        weight_gains=[1.0, 1.0],
        input_gain=spec.model.input_gain,
        non_linearity=spec.model.non_linearity,
        exponential_diode_param=dict(spec.model.exponential_diode_param),
        quadratic_diode_param=dict(spec.model.quadratic_diode_param),
        hard_sigmoid_param=dict(spec.model.hard_sigmoid_param),
        voltage_amp=spec.model.voltage_amp,
        current_amp=spec.model.current_amp,
        weight_min=spec.model.conductance_min,
        weight_max=spec.model.conductance_max,
        differential_dense_edges=differential,
        include_biases=spec.model.include_biases,
        legacy_process_index_amplification=(
            getattr(spec.model, "amplification_indexing", "logical")
            == "legacy_process_global"
        ),
    )
    bundle.energy.set_device(device)
    network = Network(bundle.energy)
    inference = _minimizer(
        bundle.energy,
        iterations=spec.solver.inference_iterations,
        mode=spec.solver.mode,
        overrelaxation=spec.solver.overrelaxation_factor,
    )
    training = _minimizer(
        bundle.energy,
        iterations=spec.solver.training_iterations,
        mode=spec.solver.mode,
        overrelaxation=spec.solver.overrelaxation_factor,
    )
    cost = TeacherKLDivergence(bundle.energy.layers()[-1])
    differentiator = Backprop(
        bundle.energy.params(),
        list(bundle.energy.layers()[1:]),
        cost,
        training,
    )
    rates = getattr(
        spec.settings,
        "learning_rates",
        (0.0,) * (3 if spec.model.include_biases else 2),
    )
    optimizer: Any = _logical_optimizer(
        bundle.catalog,
        encoding=spec.model.encoding,
        rates=rates,
        sign_only=(
            getattr(spec.settings, "update_backend", None) is not None
            and spec.settings.update_backend.type
            == "measured_cohort_a_sign_sgd"
        ),
    )
    update_backend = getattr(spec.settings, "update_backend", None)
    if (
        enable_measured
        and update_backend is not None
        and update_backend.type in MEASURED_BACKENDS
    ):
        if device_data_path is None:
            raise ValueError(
                f"Expected --device-data for {update_backend.type}. "
                "Provided value: None."
            )
        optimizer_type = measured_optimizer_type(update_backend.type)
        optimizer = optimizer_type(
            optimizer,
            bundle.catalog,
            update_backend.parameters,
            device_data_path,
        )
    elif (
        enable_measured
        and update_backend is not None
        and update_backend.type == "program_verify"
    ):
        optimizer = ProgramVerifyOptimizer(
            optimizer,
            bundle.catalog,
            parse_device_programming_config(
                update_backend.parameters["device"],
                path=(
                    "config.modes.train.update_backend.parameters.device"
                ),
            ),
        )
    return StudentStack(
        bundle=bundle,
        network=network,
        minimizer=inference,
        training_minimizer=training,
        cost=cost,
        differentiator=differentiator,
        optimizer=optimizer,
        device=device,
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


def settle_scores(
    stack: StudentStack,
    inputs: torch.Tensor,
    *,
    reset: bool = True,
) -> torch.Tensor:
    stack.network.set_input(inputs.to(stack.device), reset=reset)
    stack.minimizer.compute_equilibrium()
    return paired_scores(stack.bundle.energy.layers()[-1].state)


def fit_positive_logit_gain(
    raw_scores: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    gain_min: float,
    gain_max: float,
    steps: int,
) -> dict[str, Any]:
    scores = raw_scores.detach().to(torch.float64)
    teacher = teacher_logits.detach().to(torch.float64)
    teacher_log_prob = F.log_softmax(teacher, dim=1)
    teacher_prob = teacher_log_prob.exp()

    def kl(gain: torch.Tensor) -> torch.Tensor:
        student_log_prob = F.log_softmax(scores * gain, dim=1)
        return (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=1).mean()

    gains = torch.logspace(
        math.log10(gain_min),
        math.log10(gain_max),
        steps,
        dtype=torch.float64,
        device=scores.device,
    )
    values = torch.stack([kl(gain) for gain in gains])
    index = int(values.argmin().item())
    selected = float(gains[index].item())
    return {
        "gain": selected,
        "raw_kl": float(kl(torch.tensor(1.0, device=scores.device)).item()),
        "calibrated_kl": float(values[index].item()),
        "score_rms": float(torch.sqrt(scores.square().mean()).item()),
        "teacher_logit_rms": float(torch.sqrt(teacher.square().mean()).item()),
        "gain_at_grid_boundary": index in {0, steps - 1},
    }


def collect_calibration(
    stack: StudentStack,
    teacher: BiasFreeReluTeacher | BoundedDrnTeacher,
    loader: Iterable,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = []
    logits = []
    with torch.no_grad():
        for inputs, _labels in loader:
            inputs = inputs.to(stack.device, dtype=torch.float32)
            scores.append(settle_scores(stack, inputs, reset=True).detach())
            logits.append(teacher.logits(inputs).detach())
    return torch.cat(scores), torch.cat(logits)


def select_mapping_and_gain(
    stack: StudentStack,
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    spec: StudentTrainSpec,
    *,
    measured_candidate_projector: MeasuredCohortAOptimizer | None = None,
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    teacher_weights = tuple(item.detach() for item in teacher.parameters())
    candidates = []
    selected_targets = None
    selected_key = None
    fraction_pairs = (
        spec.mapping.scale_fraction_pairs
        if spec.mapping.scale_fraction_pairs is not None
        else product(spec.mapping.scale_fractions, repeat=2)
    )
    for fractions in fraction_pairs:
        targets, mapping = mapped_conductances(
            teacher_weights,
            encoding=spec.model.encoding,
            scale_fractions=(float(fractions[0]), float(fractions[1])),
            conductance_min=spec.model.conductance_min,
            conductance_max=spec.model.conductance_max,
            range_placement=spec.mapping.range_placement,
        )
        apply_targets(stack.bundle.catalog, targets)
        raw, teacher_logits = collect_calibration(stack, teacher, loader)
        nominal_calibration = fit_positive_logit_gain(
            raw,
            teacher_logits,
            gain_min=spec.mapping.logit_gain_min,
            gain_max=spec.mapping.logit_gain_max,
            steps=spec.mapping.logit_gain_steps,
        )
        calibration = nominal_calibration
        if measured_candidate_projector is not None:
            projected = measured_candidate_projector.preview_reset_targets()
            with torch.no_grad():
                for binding in stack.bundle.catalog.trainable:
                    binding.state.copy_(projected[binding.key])
            raw, teacher_logits = collect_calibration(stack, teacher, loader)
            calibration = fit_positive_logit_gain(
                raw,
                teacher_logits,
                gain_min=spec.mapping.logit_gain_min,
                gain_max=spec.mapping.logit_gain_max,
                steps=spec.mapping.logit_gain_steps,
            )
        record = {
            "scale_fractions": list(fractions),
            "mapping": mapping,
            "calibration": calibration,
        }
        if measured_candidate_projector is not None:
            record.update(
                {
                    "selection_domain": "measured_projected",
                    "nominal_calibration": nominal_calibration,
                }
            )
        candidates.append(record)
        key = (calibration["calibrated_kl"], fractions)
        if selected_key is None or key < selected_key:
            selected_key = key
            selected_targets = tuple(target.detach().clone() for target in targets)
    if selected_targets is None or selected_key is None:
        raise RuntimeError("Expected at least one mapping candidate.")
    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["calibration"]["calibrated_kl"],
            tuple(candidates[index]["scale_fractions"]),
        ),
    )
    apply_targets(stack.bundle.catalog, selected_targets)
    return {
        "search_examples": spec.mapping.calibration_examples,
        "selection_domain": (
            "measured_projected"
            if measured_candidate_projector is not None
            else "nominal"
        ),
        "candidates": candidates,
        "selected_index": selected_index,
        "selected": candidates[selected_index],
    }, selected_targets


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


__all__ = [
    "BoundedDrnTeacher",
    "StudentStack",
    "TeacherKLDivergence",
    "apply_targets",
    "build_student_stack",
    "collect_calibration",
    "conductance_statistics",
    "fit_positive_logit_gain",
    "mapped_conductances",
    "measured_optimizer_type",
    "paired_scores",
    "select_mapping_and_gain",
    "settle_scores",
    "collapse_clamped_dual_rail_input",
    "signed_differential_lift",
    "signed_dual_rail_lift",
]
