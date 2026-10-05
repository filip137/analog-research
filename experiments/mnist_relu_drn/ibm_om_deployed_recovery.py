"""Fixed-terminal on-chip recovery from a saved IBM OM deployment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import math
import os
from pathlib import Path
from typing import Any

import torch

from experiments.artifacts import (
    RunStore,
    atomic_write_json,
    content_hash,
    sha256_file,
)
from experiments.mnist_relu_drn.components import (
    BoundedDrnTeacher,
    StudentStack,
    collect_calibration,
    conductance_statistics,
    fit_positive_logit_gain,
    build_student_stack,
)
from experiments.mnist_relu_drn.config import StudentTrainSpec
from experiments.mnist_relu_drn.ibm_om_deployment_decomposition import (
    load_deployment_artifact,
    population_from_deployment,
    validate_deployment_contract,
)
from experiments.mnist_shared import build_mnist_loaders, teacher_batches
from experiments.schema import to_plain_data
from training.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    load_epoch_boundary_checkpoint,
    load_named_weights,
    restore_rng_state,
    save_epoch_boundary_checkpoint,
    save_named_weights,
)
from training.core.engine import ExperimentComponents, GradientsReadyEvent, train_epoch
from training.core.guards import FiniteGradientGuard
from training.ibm_reram_recovery import (
    IbmOmDeployedRecovery,
    sample_physical_fast_population,
)

from .runtime import (
    TeacherAgreementProbe,
    _ROOT,
    _amplification_index_report,
    _evaluate,
    _input,
    _load_teacher,
    _model_checkpoint_metadata,
    _train_epoch,
    _validate_checkpoint_metadata,
)


_GAIN_MIN = 0.001
_GAIN_MAX = 1000.0
_GAIN_STEPS = 121
_ENDPOINT_TOLERANCE = 2e-7


def _exact_source_metadata(
    metadata: Mapping[str, Any],
    *,
    parameters: Mapping[str, Any],
) -> None:
    selection_modifier = metadata.get("selection_weight_modifier")
    selection_parameters = (
        selection_modifier.get("parameters")
        if isinstance(selection_modifier, Mapping)
        else None
    )
    expected = {
        "selection_epoch": parameters["expected_source_selection_epoch"],
        "device_model_sha256": parameters["expected_device_model_sha256"],
    }
    mismatches = {
        name: {"expected": value, "provided": metadata.get(name)}
        for name, value in expected.items()
        if metadata.get(name) != value
    }
    modifier_expected = {
        "target_mapping": parameters["expected_target_mapping"],
        "reset_relative_mode": parameters["expected_reset_relative_mode"],
        "reset_relative_contrast_step": parameters[
            "expected_reset_relative_contrast_step"
        ],
        "reset_read_samples": parameters["expected_reset_read_samples"],
        "reset_guard_standard_errors": parameters[
            "expected_reset_guard_standard_errors"
        ],
        "forward_logit_gain": parameters["source_forward_logit_gain"],
        "dual_rail_layout_by_parameter": dict(
            parameters["source_dual_rail_layout_by_parameter"]
        ),
    }
    if not isinstance(selection_parameters, Mapping):
        mismatches["selection_weight_modifier.parameters"] = {
            "expected": modifier_expected,
            "provided": selection_parameters,
        }
    else:
        for name, value in modifier_expected.items():
            if selection_parameters.get(name) != value:
                mismatches[f"selection_weight_modifier.parameters.{name}"] = {
                    "expected": value,
                    "provided": selection_parameters.get(name),
                }
    if mismatches:
        raise ValueError(
            "Expected the recovery source checkpoint to be the frozen "
            f"RESET-relative quantized-QAT selection. Provided: {mismatches!r}."
        )


def _exact_deployment_config(
    config: Any,
    *,
    parameters: Mapping[str, Any],
) -> None:
    expected = {
        "execution": "pulse_resolved",
        "assignment_seed": parameters["expected_assignment_seed"],
        "endpoint_seed": parameters["expected_endpoint_seed"],
        "corruption_policy": parameters["expected_corruption_policy"],
        "target_mapping": parameters["expected_target_mapping"],
        "reset_relative_mode": parameters["expected_reset_relative_mode"],
        "reset_relative_contrast_step": parameters[
            "expected_reset_relative_contrast_step"
        ],
        "reset_read_samples": parameters["expected_reset_read_samples"],
        "reset_guard_standard_errors": parameters[
            "expected_reset_guard_standard_errors"
        ],
        "forward_logit_gain": parameters["source_forward_logit_gain"],
        "dual_rail_layout_by_parameter": tuple(
            parameters["source_dual_rail_layout_by_parameter"].items()
        ),
        "endpoint_policy": "clip_0_1",
        "target_out_of_support": "error",
        "maximum_program_pulses": 128,
    }
    mismatches = {
        name: {"expected": value, "provided": getattr(config, name)}
        for name, value in expected.items()
        if getattr(config, name) != value
    }
    if mismatches:
        raise ValueError(
            "Expected one exact frozen RESET-relative source deployment. "
            f"Provided: {mismatches!r}."
        )


def _endpoint_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (
            left.detach().to(device="cpu", dtype=torch.float64)
            - right.detach().to(device="cpu", dtype=torch.float64)
        )
        .abs()
        .max()
        .item()
    )


def _calibrate_apparent_gain(
    stack: StudentStack,
    teacher: Any,
    loader: Iterable,
    recovery: IbmOmDeployedRecovery,
) -> dict[str, Any]:
    with recovery.temporary_forward_state("apparent"):
        raw_scores, teacher_logits = collect_calibration(stack, teacher, loader)
    return fit_positive_logit_gain(
        raw_scores,
        teacher_logits,
        gain_min=_GAIN_MIN,
        gain_max=_GAIN_MAX,
        steps=_GAIN_STEPS,
    )


def _evaluate_state(
    stack: StudentStack,
    teacher: Any,
    loader: Iterable,
    recovery: IbmOmDeployedRecovery,
    *,
    state: str,
    gain: float,
    maximum_batches: int | None,
) -> dict[str, Any]:
    stack.cost.gain = float(gain)
    with recovery.temporary_forward_state(state):
        return _evaluate(
            stack,
            teacher,
            loader,
            maximum_batches=maximum_batches,
        )


def _evaluate_gain_pair(
    stack: StudentStack,
    teacher: Any,
    loader: Iterable,
    recovery: IbmOmDeployedRecovery,
    *,
    source_gain: float,
    calibration: Mapping[str, Any],
    maximum_batches: int | None,
) -> dict[str, Any]:
    calibrated_gain = float(calibration["gain"])
    result: dict[str, Any] = {
        "calibration": dict(calibration),
        "source_gain": {},
        "recalibrated_gain": {},
    }
    for state in ("apparent", "persistent"):
        source = _evaluate_state(
            stack,
            teacher,
            loader,
            recovery,
            state=state,
            gain=source_gain,
            maximum_batches=maximum_batches,
        )
        calibrated = _evaluate_state(
            stack,
            teacher,
            loader,
            recovery,
            state=state,
            gain=calibrated_gain,
            maximum_batches=maximum_batches,
        )
        if source["student_accuracy"] != calibrated["student_accuracy"]:
            raise RuntimeError(
                "Expected positive logit-gain recalibration not to change "
                f"argmax accuracy for {state!r}."
            )
        result["source_gain"][state] = source
        result["recalibrated_gain"][state] = calibrated
    stack.cost.gain = calibrated_gain
    return result


def _solve_direct_probability_scale(
    ratios: torch.Tensor,
    *,
    calibration_batches: int,
    total_steps: int,
    pulse_cap: int,
) -> tuple[float, dict[str, Any]]:
    value = ratios.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if value.numel() == 0 or not bool(torch.all(torch.isfinite(value))):
        raise ValueError("Expected finite direct-pulse calibration ratios.")
    target_per_batch = pulse_cap / total_steps

    def expected(scale: float) -> float:
        return float(
            (value * scale)
            .clamp(max=1.0)
            .sum()
            .item()
            / calibration_batches
        )

    if expected(1.0) == 0.0:
        raise RuntimeError("Expected nonzero gradients for direct-pulse calibration.")
    lower = 0.0
    upper = 1.0
    while expected(upper) < target_per_batch:
        upper *= 2.0
        if not math.isfinite(upper):
            raise RuntimeError("Expected a finite direct-pulse probability scale.")
    for _iteration in range(64):
        midpoint = 0.5 * (lower + upper)
        if expected(midpoint) < target_per_batch:
            lower = midpoint
        else:
            upper = midpoint
    scale = 0.5 * (lower + upper)
    report = _direct_probability_scale_report(
        value,
        scale=scale,
        calibration_batches=calibration_batches,
        total_steps=total_steps,
        pulse_cap=pulse_cap,
        algorithm="fixed_probe_expected_full_run_cap_bisection",
    )
    return scale, report


def _direct_probability_scale_report(
    ratios: torch.Tensor,
    *,
    scale: float,
    calibration_batches: int,
    total_steps: int,
    pulse_cap: int,
    algorithm: str,
) -> dict[str, Any]:
    value = ratios.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if value.numel() == 0 or not bool(torch.all(torch.isfinite(value))):
        raise ValueError("Expected finite direct-pulse calibration ratios.")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Expected a positive direct-pulse probability scale.")
    scaled = (value * scale).clamp(max=1.0)
    expected_per_batch = float(scaled.sum().item() / calibration_batches)
    return {
        "algorithm": algorithm,
        "calibration_batches": calibration_batches,
        "ratios": int(value.numel()),
        "nonzero_probability_values": int((value > 0.0).sum().item()),
        "target_expected_pulses_per_batch": pulse_cap / total_steps,
        "expected_pulses_per_batch": expected_per_batch,
        "expected_full_run_pulses": expected_per_batch * total_steps,
        "slow_pulse_cap": pulse_cap,
        "probability_scale": scale,
        "saturated_probability_values": int((value * scale >= 1.0).sum().item()),
        "maximum_probability": float(scaled.max().item()),
        "mean_probability": float(scaled.mean().item()),
    }


def _fixed_direct_gradient_thresholds(
    magnitudes_by_parameter: Mapping[str, list[torch.Tensor]],
    *,
    percentile: float,
    calibration_batches: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Derive fixed strict gates from nonzero absolute raw gradients."""

    thresholds: dict[str, float] = {}
    parameter_reports: dict[str, Any] = {}
    sample_digests: dict[str, str] = {}
    for key in sorted(magnitudes_by_parameter):
        pieces = magnitudes_by_parameter[key]
        if len(pieces) != calibration_batches:
            raise RuntimeError(
                "Expected one gradient-magnitude sample per calibration batch."
            )
        digest = sha256()
        normalized = []
        for piece in pieces:
            value = piece.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            if not bool(torch.all(torch.isfinite(value))):
                raise FloatingPointError(
                    f"Expected finite calibration magnitudes for {key!r}."
                )
            contiguous = value.contiguous()
            digest.update(contiguous.numpy().tobytes(order="C"))
            normalized.append(contiguous)
        joined = torch.cat(normalized)
        nonzero = joined[joined > 0.0]
        if nonzero.numel() == 0:
            raise RuntimeError(
                f"Expected nonzero calibration gradients for {key!r}."
            )
        threshold = float(
            torch.quantile(
                nonzero.to(dtype=torch.float64),
                percentile / 100.0,
                interpolation="linear",
            ).item()
        )
        eligible = joined > threshold
        thresholds[key] = threshold
        sample_digests[key] = digest.hexdigest()
        parameter_reports[key] = {
            "observations": int(joined.numel()),
            "nonzero": int(nonzero.numel()),
            "eligible": int(eligible.sum().item()),
            "suppressed_nonzero": int(
                ((joined > 0.0) & ~eligible).sum().item()
            ),
            "threshold": threshold,
            "raw_gradient_magnitude_sha256": digest.hexdigest(),
        }
    report = {
        "algorithm": "fixed_initial_replay_nonzero_absolute_raw_gradient_quantile",
        "comparison": "strictly_greater_than",
        "interpolation": "linear",
        "percentile": percentile,
        "calibration_batches": calibration_batches,
        "parameters": parameter_reports,
        "sample_digest_by_parameter": sample_digests,
        "threshold_by_parameter": dict(thresholds),
        "training_permutation_consumed": False,
    }
    report["calibration_sha256"] = content_hash(report)
    return thresholds, report


def _calibrate_direct_probabilities(
    stack: StudentStack,
    teacher: Any,
    loader: Iterable,
    train_generator: torch.Generator,
    recovery: IbmOmDeployedRecovery,
    *,
    calibration_batches: int,
    frozen_scale: float | None,
) -> None:
    generator_state = train_generator.get_state().detach().cpu().clone()
    rng_state = capture_rng_state()
    percentile = recovery.parameters.get("direct_gradient_magnitude_percentile")
    ratio_batches: list[torch.Tensor] = []
    magnitude_pieces: dict[str, list[torch.Tensor]] = {
        key: [] for key in recovery.keys
    }
    ratio_pieces: dict[str, list[torch.Tensor]] = {
        key: [] for key in recovery.keys
    }

    def record_snapshot(event: Any) -> None:
        if not isinstance(event, GradientsReadyEvent):
            return
        snapshot = recovery.direct_calibration_snapshot()
        ratio_batches.append(
            torch.cat(
                tuple(
                    snapshot[key]["pulse_probability_ratio"].detach().cpu()
                    for key in recovery.keys
                )
            )
        )
        if percentile is not None:
            for key in recovery.keys:
                magnitude_pieces[key].append(
                    snapshot[key]["raw_gradient_magnitude"].detach().cpu()
                )
                ratio_pieces[key].append(
                    snapshot[key]["pulse_probability_ratio"].detach().cpu()
                )

    try:
        # A gradient pass: the snapshot reads the assigned gradients, and no
        # update runs.
        train_epoch(
            ExperimentComponents(
                stack.network,
                stack.cost,
                stack.minimizer,
                tuple(stack.bundle.energy.params()),
                stack.differentiator,
                None,
            ),
            teacher_batches(
                loader, teacher, stack.device, maximum_batches=calibration_batches
            ),
            event_handlers=(
                FiniteGradientGuard(
                    tuple(binding.key for binding in stack.bundle.catalog.trainable),
                    label="direct-pulse calibration",
                ),
                record_snapshot,
            ),
            reset_input=True,
        )
        if len(ratio_batches) != calibration_batches:
            raise RuntimeError(
                "Expected the declared number of direct-pulse calibration batches."
            )
    finally:
        recovery.zero_grad(set_to_none=True)
        train_generator.set_state(generator_state)
        restore_rng_state(rng_state)
    ungated_ratios = torch.cat(ratio_batches)
    gated_ratios = ungated_ratios
    threshold_report = None
    if percentile is not None:
        thresholds, threshold_report = _fixed_direct_gradient_thresholds(
            magnitude_pieces,
            percentile=float(percentile),
            calibration_batches=calibration_batches,
        )
        recovery.set_direct_gradient_thresholds(
            thresholds,
            report=threshold_report,
        )
        gated_batches = []
        for batch_index in range(calibration_batches):
            gated_batches.append(
                torch.cat(
                    tuple(
                        ratio_pieces[key][batch_index]
                        * (
                            magnitude_pieces[key][batch_index]
                            > thresholds[key]
                        )
                        for key in recovery.keys
                    )
                )
            )
        gated_ratios = torch.cat(gated_batches)
    if frozen_scale is None:
        scale, report = _solve_direct_probability_scale(
            gated_ratios,
            calibration_batches=calibration_batches,
            total_steps=recovery.total_steps,
            pulse_cap=recovery.slow_pulse_cap,
        )
    else:
        scale = float(frozen_scale)
        ungated_report = _direct_probability_scale_report(
            ungated_ratios,
            scale=scale,
            calibration_batches=calibration_batches,
            total_steps=recovery.total_steps,
            pulse_cap=recovery.slow_pulse_cap,
            algorithm="frozen_unthresholded_control_scale_parity",
        )
        expected = float(ungated_report["expected_full_run_pulses"])
        if not math.isclose(
            expected,
            recovery.slow_pulse_cap,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                "Expected the frozen unthresholded probability scale to "
                "reproduce the matched control pulse cap."
            )
        report = _direct_probability_scale_report(
            gated_ratios,
            scale=scale,
            calibration_batches=calibration_batches,
            total_steps=recovery.total_steps,
            pulse_cap=recovery.slow_pulse_cap,
            algorithm="frozen_unthresholded_control_scale_after_gradient_gate",
        )
        report["ungated_control"] = ungated_report
        report["source_recovery_report_sha256"] = recovery.parameters.get(
            "direct_probability_scale_source_report_sha256"
        )
    if threshold_report is not None:
        report["gradient_threshold_calibration_sha256"] = threshold_report[
            "calibration_sha256"
        ]
    recovery.set_direct_probability_scale(scale, report=report)


def _refresh_epoch(
    stack: StudentStack,
    teacher: Any,
    loader: Iterable,
    recovery: IbmOmDeployedRecovery,
    *,
    maximum_batches: int | None,
) -> tuple[dict[str, Any], int]:
    """Gradient-free recovery: settle, measure, then one ``recovery.step()``.

    The recovery writes the apparent conductances itself, so they are not
    clamped afterwards.
    """

    probe = TeacherAgreementProbe()
    with torch.no_grad():
        result = train_epoch(
            ExperimentComponents(
                stack.network,
                stack.cost,
                stack.minimizer,
                tuple(stack.bundle.energy.params()),
                None,
                recovery,
                clamp_after_update=False,
            ),
            teacher_batches(loader, teacher, stack.device, maximum_batches=maximum_batches),
            probes=(probe,),
            reset_input=True,
        )
    return result.probe_value(probe.name), result.batch_count


def _recovery_metadata(
    stack: StudentStack,
    *,
    spec: StudentTrainSpec,
    teacher_sha256: str,
    source_weights_sha256: str,
    deployment_sha256: str,
    device_model_sha256: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = _model_checkpoint_metadata(
        stack,
        spec=spec,
        teacher_sha256=teacher_sha256,
        mapping=None,
        deployment_source={
            "weights_sha256": source_weights_sha256,
            "deployment_sha256": deployment_sha256,
            "assignment_seed": parameters["expected_assignment_seed"],
            "endpoint_seed": parameters["expected_endpoint_seed"],
        },
        device_model_sha256=device_model_sha256,
    )
    metadata.update(
        {
            "initial_target_mapping": "saved_ibm_om_deployment_apparent_state",
            "update_backend": "ibm_om_deployed_recovery",
            "selection_epoch": spec.settings.num_epochs - 1,
            "selection_metric": "fixed_terminal_epoch",
            "recovery_protocol_sha256": content_hash(to_plain_data(parameters)),
            "recovery_method": parameters["method"],
            "slow_pulse_budget_per_cell": parameters[
                "slow_pulse_budget_per_cell"
            ],
            "source_weights_sha256": source_weights_sha256,
            "source_deployment_sha256": deployment_sha256,
        }
    )
    return metadata


def _validate_resume_metadata(
    metadata: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    fields = (
        "experiment_id",
        "update_backend",
        "recovery_protocol_sha256",
        "recovery_method",
        "slow_pulse_budget_per_cell",
        "source_weights_sha256",
        "source_deployment_sha256",
        "device_model_sha256",
        "teacher_sha256",
    )
    mismatches = {
        field: {"expected": expected.get(field), "provided": metadata.get(field)}
        for field in fields
        if metadata.get(field) != expected.get(field)
    }
    if mismatches:
        raise ValueError(
            "Expected resume provenance to match this exact recovery arm. "
            f"Provided: {mismatches!r}."
        )


def run_recovery_train(request: Any) -> int:
    """Run one predeclared fixed-terminal deployed-array recovery arm."""

    spec = request.spec
    if not isinstance(spec, StudentTrainSpec):
        raise TypeError("Expected deployed recovery to resolve StudentTrainSpec.")
    parameters = spec.settings.update_backend.parameters
    source_weights_sha256 = str(parameters["expected_source_weights_sha256"])
    deployment_sha256 = sha256_file(request.deployment)
    device_model_sha256 = sha256_file(request.device_model)
    input_artifacts = [
        _input("teacher_weights", request.teacher_weights),
        _input("deployment", request.deployment),
        _input("device_model", request.device_model),
    ]
    if request.weights is not None:
        input_artifacts.append(_input("weights", request.weights))
    else:
        input_artifacts.append(_input("resume", request.resume))
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=input_artifacts,
    )
    try:
        torch.manual_seed(spec.runtime.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(spec.runtime.seed)
        device = torch.device(spec.runtime.device)
        data = build_mnist_loaders(
            spec.data,
            data_seed=spec.runtime.data_seed,
            calibration_examples=spec.mapping.calibration_examples,
            calibration_batch_size=spec.mapping.calibration_batch_size,
        )
        teacher, _teacher_metadata = _load_teacher(
            request.teacher_weights,
            device=device,
            spec=spec,
        )
        teacher_sha256 = sha256_file(request.teacher_weights)
        stack = build_student_stack(spec, enable_measured=False)
        source_metadata: Mapping[str, Any] = {}
        if request.weights is not None:
            if sha256_file(request.weights) != source_weights_sha256:
                raise ValueError(
                    "Expected --weights to equal the frozen source checkpoint digest."
                )
            loaded = load_named_weights(request.weights, stack.bundle.catalog)
            source_metadata = loaded.metadata
            _validate_checkpoint_metadata(
                source_metadata,
                spec=spec,
                teacher_sha256=teacher_sha256,
                expected_amplification_indices=_amplification_index_report(stack),
            )
            _exact_source_metadata(source_metadata, parameters=parameters)

        deployment = load_deployment_artifact(request.deployment)
        contract_metadata = {
            "selection_epoch": parameters["expected_source_selection_epoch"],
            "device_model_sha256": parameters["expected_device_model_sha256"],
        }
        tensors, deployment_config, deployment_report = validate_deployment_contract(
            deployment,
            expected_selected_weights_sha256=source_weights_sha256,
            expected_binding_keys=tuple(
                binding.key for binding in stack.bundle.catalog.trainable
            ),
            expected_binding_shapes=tuple(
                tuple(binding.state.shape)
                for binding in stack.bundle.catalog.trainable
            ),
            configured_modifiers=(deployment["config"],),
            checkpoint_metadata=contract_metadata,
        )
        _exact_deployment_config(deployment_config, parameters=parameters)
        if deployment_sha256 != sha256_file(request.deployment):
            raise RuntimeError("Expected immutable source deployment bytes.")
        if device_model_sha256 != parameters["expected_device_model_sha256"]:
            raise ValueError("Expected the exact frozen IBM OM device model.")
        population = population_from_deployment(
            deployment,
            config=deployment_config,
        )
        commissioning = deployment["reset_commissioning"]
        reset_baseline = commissioning["observed"]["baseline"]
        steps_per_epoch = min(
            len(data.train),
            spec.settings.max_batches or len(data.train),
        )
        total_steps = steps_per_epoch * spec.settings.num_epochs

        fast_population = None
        fast_receipt = None
        fast_paths: tuple[Path, Path] | None = None
        if (
            parameters["method"] == "tiki_taka"
            and parameters["fast_backend"] == "physical_om"
        ):
            executable = os.environ.get("EBL_AIHWKIT_PYTHON")
            if executable is None:
                raise RuntimeError(
                    "Expected physical-fast recovery to receive "
                    "EBL_AIHWKIT_PYTHON for pinned OM population sampling."
                )
            fast_paths = (
                store.run_dir / "artifacts" / "ibm_om_fast_population.npz",
                store.run_dir
                / "artifacts"
                / "ibm_om_fast_population.receipt.json",
            )
            fast_population, fast_receipt = sample_physical_fast_population(
                stack.bundle.catalog.trainable,
                assignment_seed=parameters["fast_assignment_seed"],
                aihwkit_python=Path(executable),
                population_path=fast_paths[0],
                receipt_path=fast_paths[1],
            )

        recovery = IbmOmDeployedRecovery(
            stack.bundle.catalog.trainable,
            slow_population=population,
            source_raw_apparent=tensors["raw_apparent_endpoint"],
            source_apparent=tensors["apparent_endpoint"],
            source_persistent=tensors["persistent_endpoint"],
            requested_target=tensors["mapped_target"]
            if "mapped_target" in tensors
            else tensors["requested_target"],
            reset_baseline=reset_baseline,
            generator_state_after_programming=deployment[
                "generator_state_after_programming"
            ],
            parameters=parameters,
            learning_rates=spec.settings.learning_rates,
            conductance_min=spec.model.conductance_min,
            conductance_max=spec.model.conductance_max,
            total_steps=total_steps,
            source_deployment_sha256=deployment_sha256,
            device_model_path=request.device_model,
            fast_population=fast_population,
            fast_population_receipt=fast_receipt,
        )
        if (
            _endpoint_error(
                recovery.current_apparent_endpoint(clipped=True),
                tensors["apparent_endpoint"],
            )
            > _ENDPOINT_TOLERANCE
            or _endpoint_error(
                recovery.current_persistent_endpoint(),
                tensors["persistent_endpoint"],
            )
            > _ENDPOINT_TOLERANCE
        ):
            raise RuntimeError(
                "Expected recovery to begin from the saved two-state deployment."
            )

        metadata = _recovery_metadata(
            stack,
            spec=spec,
            teacher_sha256=teacher_sha256,
            source_weights_sha256=source_weights_sha256,
            deployment_sha256=deployment_sha256,
            device_model_sha256=device_model_sha256,
            parameters=parameters,
        )
        resume_path = store.run_dir / "checkpoints" / "resume.pt"
        weights_path = store.run_dir / "checkpoints" / "weights.pt"
        recovery_path = store.run_dir / "artifacts" / "ibm_om_recovery.pt"
        report_path = store.run_dir / "artifacts" / "ibm_om_recovery.report.json"
        direct_calibration_path = (
            store.run_dir
            / "artifacts"
            / "ibm_om_direct_gradient_calibration.json"
        )
        start_epoch = 0
        global_step = 0
        initial: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []

        if request.resume is not None:
            resumed = load_epoch_boundary_checkpoint(
                request.resume,
                catalog=stack.bundle.catalog,
                optimizer=recovery,
                dataloader_generators={"train": data.train_generator},
            )
            _validate_checkpoint_metadata(
                resumed.metadata,
                spec=spec,
                teacher_sha256=teacher_sha256,
                expected_amplification_indices=_amplification_index_report(stack),
            )
            _validate_resume_metadata(resumed.metadata, expected=metadata)
            start_epoch = resumed.epoch
            global_step = resumed.global_step
            initial = deepcopy(resumed.progress_state["initial"])
            history = deepcopy(resumed.progress_state["history"])
            stack.cost.gain = float(resumed.progress_state["current_logit_gain"])
            if recovery.step_index != global_step:
                raise RuntimeError(
                    "Expected recovery and checkpoint global steps to match."
                )
        else:
            initial_calibration = _calibrate_apparent_gain(
                stack,
                teacher,
                data.calibration,
                recovery,
            )
            initial = _evaluate_gain_pair(
                stack,
                teacher,
                data.validation,
                recovery,
                source_gain=parameters["source_forward_logit_gain"],
                calibration=initial_calibration,
                maximum_batches=spec.settings.max_validation_batches,
            )
            initial["endpoint"] = recovery.report()
            initial["deployment_contract"] = deployment_report
            store.append_metric({"mode": "initialization", **initial})
            if parameters["method"] == "direct_pulse":
                frozen_scale = parameters["direct_probability_scale"]
                if (
                    frozen_scale is None
                    or parameters.get("direct_gradient_magnitude_percentile")
                    is not None
                ):
                    _calibrate_direct_probabilities(
                        stack,
                        teacher,
                        data.train,
                        data.train_generator,
                        recovery,
                        calibration_batches=parameters[
                            "direct_probability_calibration_batches"
                        ],
                        frozen_scale=frozen_scale,
                    )
                else:
                    recovery.set_direct_probability_scale(
                        frozen_scale,
                        report={
                            "algorithm": "frozen_explicit_scale",
                            "probability_scale": frozen_scale,
                            "calibration_batches": parameters[
                                "direct_probability_calibration_batches"
                            ],
                        },
                    )
                direct_calibration = {
                    "schema": "ebl.ibm_om.direct_gradient_calibration",
                    "schema_version": 1,
                    "source_deployment_sha256": deployment_sha256,
                    "gradient_gate": recovery.report()[
                        "direct_gradient_gate"
                    ],
                    "probability_calibration": recovery.report()[
                        "direct_probability_calibration"
                    ],
                }
                atomic_write_json(
                    direct_calibration_path,
                    direct_calibration,
                )
                store.append_metric(
                    {
                        "mode": "direct_calibration",
                        **direct_calibration,
                    }
                )

        training_stack = replace(stack, optimizer=recovery)
        for epoch in range(start_epoch, spec.settings.num_epochs):
            if recovery.requires_gradients:
                train_metrics, batches = _train_epoch(
                    training_stack,
                    teacher,
                    data.train,
                    maximum_batches=spec.settings.max_batches,
                )
            else:
                train_metrics, batches = _refresh_epoch(
                    stack,
                    teacher,
                    data.train,
                    recovery,
                    maximum_batches=spec.settings.max_batches,
                )
            global_step += batches
            expected_step = (epoch + 1) * steps_per_epoch
            if recovery.step_index != expected_step or global_step != expected_step:
                raise RuntimeError(
                    "Expected one physical recovery step per training minibatch."
                )
            calibration = _calibrate_apparent_gain(
                stack,
                teacher,
                data.calibration,
                recovery,
            )
            validation = _evaluate_gain_pair(
                stack,
                teacher,
                data.validation,
                recovery,
                source_gain=parameters["source_forward_logit_gain"],
                calibration=calibration,
                maximum_batches=spec.settings.max_validation_batches,
            )
            epoch_record = {
                "epoch": epoch,
                "completed_epochs": epoch + 1,
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation,
                "conductances": conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                ),
                "recovery": recovery.report(),
            }
            history.append(epoch_record)
            store.append_metric({"mode": "train", **epoch_record})
            save_epoch_boundary_checkpoint(
                resume_path,
                catalog=stack.bundle.catalog,
                epoch=epoch + 1,
                global_step=global_step,
                optimizer=recovery,
                progress_state={
                    "initial": initial,
                    "history": history,
                    "current_logit_gain": stack.cost.gain,
                },
                dataloader_generators={"train": data.train_generator},
                metadata=metadata,
            )

        if recovery.step_index != total_steps:
            raise RuntimeError("Expected recovery to reach its fixed terminal step.")
        if parameters["method"] == "direct_pulse":
            atomic_write_json(
                direct_calibration_path,
                {
                    "schema": "ebl.ibm_om.direct_gradient_calibration",
                    "schema_version": 1,
                    "source_deployment_sha256": deployment_sha256,
                    "gradient_gate": recovery.report()[
                        "direct_gradient_gate"
                    ],
                    "probability_calibration": recovery.report()[
                        "direct_probability_calibration"
                    ],
                },
            )
        final = history[-1]
        save_named_weights(
            weights_path,
            stack.bundle.catalog,
            metadata={
                **metadata,
                "fixed_logit_gain": stack.cost.gain,
                "terminal_validation": final["validation"],
            },
        )
        atomic_torch_save(recovery.final_bundle(), recovery_path)
        final_report = recovery.report()
        atomic_write_json(report_path, final_report)
        artifacts = [
            store.artifact_record(weights_path, kind="selected_named_weights"),
            store.artifact_record(resume_path, kind="epoch_boundary_resume"),
            store.artifact_record(recovery_path, kind="ibm_om_recovery_state"),
            store.artifact_record(report_path, kind="ibm_om_recovery_report"),
        ]
        if fast_paths is not None:
            artifacts.extend(
                (
                    store.artifact_record(
                        fast_paths[0], kind="ibm_om_fast_array_population"
                    ),
                    store.artifact_record(
                        fast_paths[1],
                        kind="ibm_om_fast_array_population_receipt",
                    ),
                )
            )
        if parameters["method"] == "direct_pulse":
            artifacts.append(
                store.artifact_record(
                    direct_calibration_path,
                    kind="ibm_om_direct_gradient_calibration",
                )
            )
        store.complete(
            metrics={
                "protocol": "fixed_terminal_deployed_array_recovery",
                "source_weights_sha256": source_weights_sha256,
                "source_deployment_sha256": deployment_sha256,
                "initial": initial,
                "final": final,
                "recovery": final_report,
            },
            artifacts=artifacts,
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_recovery_train"]
