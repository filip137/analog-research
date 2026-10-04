"""Executable boundaries for the staged 784-256-10 IBM-OM crossbar study.

Each invocation crosses exactly one scientific boundary.  In particular,
corruption consumes a named healthy P0 state without programming again, and
on-chip Adam consumes that named healthy or faulted state without remapping it.
The post-write apparent effective state is the only network-facing state;
persistent-state metrics are emitted as explicitly secondary diagnostics.
"""

from __future__ import annotations

from hashlib import sha256
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import torch
import torch.nn.functional as F

from experiments.artifacts import (
    ArtifactRecord,
    RunStore,
    atomic_write_json,
    content_hash,
    sha256_file,
)
from experiments.mnist_analog_relu.runtime import (
    _evaluate,
    _evaluate_plant_states,
    _mapping_report,
    _matched_published_fault_overlay,
    _offchip_adapt,
    _program_endpoint,
    _recovery_seed,
    _sample_population,
    _save_and_verify_on_chip_recovery_state,
    _sample_aihwkit_om_apparent_write_noise,
    population_nominal_step,
)
from experiments.mnist_analog_relu.staged_analysis import (
    load_selection_receipt as load_analyzed_selection_receipt,
    validate_selection_receipt as validate_analyzed_selection_receipt,
)
from experiments.mnist_analog_relu.staged_artifacts import (
    HwaMaster,
    STAGED_PROGRAM_STREAM_ROLE,
    StagedDeviceState,
    load_device_state,
    load_hwa_master,
    save_device_state,
    save_hwa_master,
)
from experiments.mnist_analog_relu.staged_config import (
    DiagnosticLiteralAdamHyperparameters,
    FreshApparentDiagnosticStageSettings,
    LiteralAdamHyperparameters,
    OffchipHwaStageSettings,
    OnChipAdamStageSettings,
    OnChipAdamDiagnosticStageSettings,
    SelectionReceiptAdamHyperparameters,
    StagedCrossbarTrainSpec,
)
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import load_named_weights
from training.ibm_reram_program_verify import derive_seed
from training.ibm_om_standard_crossbar import (
    CROSSBAR_TRAJECTORY_SEED_DERIVATION,
    IbmOmEffectiveCrossbarPlant,
    IbmOmCrossbarStateBundle,
    PulseAdam,
    aihwkit_analog_linear_default_logical_weights,
    build_crossbar_layout,
    build_deterministic_effective_codebook,
    crossbar_state_bundle,
    crossbar_trajectory_seeds,
    map_logical_weights,
    project_to_nearest_effective_code,
    restore_crossbar_state_bundle,
    standard_crossbar_logits,
    tensor_sha256,
)

if TYPE_CHECKING:
    from ebl.cli import TrainRequest
    from experiments.mnist_shared import MnistLoaders


_ROOT = Path(__file__).resolve().parents[2]
_DIMS = (784, 256, 10)
_CELL_COUNT = 203_264
_RECOVERY_SCHEMA = "ebl.ibm_om_crossbar_adam_recovery"
_DIAGNOSTIC_RECOVERY_SCHEMA = "ebl.ibm_om_crossbar_adam_diagnostic_recovery"
_DIAGNOSTIC_RECOVERY_SCHEMA_VERSION = 2
_ADAM_SELECTION_METRIC = "validation.apparent_forward.student_accuracy"
_ADAM_SELECTION_METRIC_KEY = "student_accuracy"
_ADAM_IMPLEMENTATION = {
    "digitally_assisted": True,
    "gradient_engine": "digital_full_apparent_q_autograd",
    "optimizer_moments": "digital",
    "fully_local_learning_primitive": False,
}
_START_STATES = (
    "hwa_healthy_p0",
    "hwa_published_fault",
    "scratch_healthy_p0",
    "scratch_published_fault",
)


def _input(role: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Expected {role!r} to identify an existing file: {resolved}."
        )
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved)}


def _sampler_path() -> Path:
    value = os.environ.get("EBL_AIHWKIT_PYTHON")
    if not value:
        raise RuntimeError(
            "Expected EBL_AIHWKIT_PYTHON to name the separate AIHWKit 1.1.0 "
            "population-sampling interpreter."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            "Expected EBL_AIHWKIT_PYTHON to identify an executable file."
        )
    return path


def _validate_request(request: Any) -> tuple[tuple[dict[str, Any], ...], Path | None]:
    """Enforce the stage-specific public input surface before creating a run."""

    spec = request.spec
    if not isinstance(spec, StagedCrossbarTrainSpec):
        raise TypeError(
            "Expected mnist_ibm_om_crossbar_relu.v2 train to resolve "
            "StagedCrossbarTrainSpec."
        )
    forbidden_common = ("base_weights", "device_data", "device_model")
    for name in forbidden_common:
        if getattr(request, name, None) is not None:
            raise ValueError(
                f"Expected staged crossbar runs not to use --{name.replace('_', '-')}."
            )
    if request.teacher_weights is None:
        raise ValueError("Expected every staged crossbar run to use --teacher-weights.")

    stage = spec.stage
    required: set[str] = {"teacher_weights"}
    permitted: set[str] = {"teacher_weights"}
    sampler: Path | None = None
    if stage.kind == "offchip_hwa":
        sampler = _sampler_path()
    elif stage.kind == "deploy":
        sampler = _sampler_path()
        if stage.source_kind == "hwa_master":
            required.add("weights")
            permitted.add("weights")
    elif stage.kind == "apply_corruption":
        required.add("device_state")
        permitted.add("device_state")
    elif stage.kind == "on_chip_adam":
        required.add("device_state")
        permitted.update(("device_state", "resume"))
        hyperparameters = stage.hyperparameters
        if isinstance(hyperparameters, SelectionReceiptAdamHyperparameters):
            required.add("selection_receipt")
            permitted.add("selection_receipt")
        elif not isinstance(hyperparameters, LiteralAdamHyperparameters):
            raise TypeError("Expected resolved literal or receipt Adam settings.")
    elif stage.kind == "on_chip_adam_diagnostic":
        required.add("device_state")
        permitted.update(("device_state", "resume"))
        if not isinstance(
            stage.hyperparameters, DiagnosticLiteralAdamHyperparameters
        ):
            raise TypeError("Expected resolved literal diagnostic Adam settings.")
    elif stage.kind == "fresh_apparent_diagnostic":
        required.add("device_state")
        permitted.add("device_state")
    else:  # pragma: no cover - strict parser invariant
        raise RuntimeError(f"Unsupported staged kind: {stage.kind!r}.")

    values = {
        name: getattr(request, name, None)
        for name in ("weights", "device_state", "selection_receipt", "resume")
    }
    for name in sorted(required):
        if getattr(request, name, None) is None:
            raise ValueError(f"Expected stage {stage.kind!r} to use --{name.replace('_', '-')}.")
    for name, value in values.items():
        if value is not None and name not in permitted:
            raise ValueError(
                f"Expected stage {stage.kind!r} not to use --{name.replace('_', '-')}."
            )

    records = [_input("teacher_weights", request.teacher_weights)]
    for name, role in (
        ("weights", "hwa_master"),
        ("device_state", "origin_device_state"),
        ("selection_receipt", "adam_selection_receipt"),
        ("resume", "adam_epoch_resume"),
    ):
        value = getattr(request, name, None)
        if value is not None:
            records.append(_input(role, value))
    if sampler is not None:
        records.append(_input("aihwkit_python", sampler))
    return tuple(records), sampler


def _cuda_device(spec: StagedCrossbarTrainSpec) -> torch.device:
    if spec.runtime.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Expected the exact staged runtime to resolve CUDA; CPU fallback is forbidden."
        )
    probe = torch.empty(1, dtype=torch.float32, device="cuda")
    if probe.device.type != "cuda":  # pragma: no cover - defensive runtime gate
        raise RuntimeError("Expected the staged runtime allocation probe to reside on CUDA.")
    return probe.device


def _layout(spec: StagedCrossbarTrainSpec):
    layout = build_crossbar_layout(
        spec.model.dims,
        maximum_input_size=spec.model.maximum_input_size,
    )
    shapes = tuple(tile.shape for tile in layout)
    if (
        spec.model.dims != _DIMS
        or shapes != ((392, 256), (392, 256), (256, 10))
        or sum(tile.cells for tile in layout) != _CELL_COUNT
    ):
        raise RuntimeError("Expected the fixed 784-256-10 three-tile crossbar layout.")
    return layout


def _load_teacher(
    path: Path,
    *,
    spec: StagedCrossbarTrainSpec,
    device: torch.device,
) -> tuple[BiasFreeReluTeacher, dict[str, Any], str]:
    source = path.expanduser().resolve()
    digest = sha256_file(source)
    teacher = BiasFreeReluTeacher(device=device, dims=spec.model.dims)
    loaded = load_named_weights(source, teacher.catalog)
    metadata = dict(loaded.metadata)
    accuracy = metadata.get("selection_accuracy")
    minimum = metadata.get("minimum_validation_accuracy")
    if (
        metadata.get("experiment_id") != spec.source.experiment_id
        or metadata.get("architecture") != spec.source.architecture
        or metadata.get("logical_dims") != list(spec.model.dims)
        or metadata.get("bias") is not False
        or metadata.get("runtime_seed") != spec.runtime.seed
        or metadata.get("data_seed") != spec.runtime.data_seed
        or metadata.get("batch_size") != spec.data.batch_size
        or metadata.get("train_shuffle") is not spec.data.shuffle
        or metadata.get("training_examples") != 55000
        or metadata.get("validation_examples") != 5000
        or metadata.get("validation_split")
        != "torchvision_mnist_train_stratified_per_class_numpy_default_rng_sorted_indices"
        or metadata.get("preprocessing")
        != "to_tensor_flatten_normalize_mean_0.1307_std_0.3"
        or metadata.get("selection_metric") != "validation.cross_entropy"
        or metadata.get("validation_accuracy_comparison_operator") != ">"
        or isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or float(accuracy) <= spec.source.minimum_validation_accuracy
        or isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isclose(
            float(minimum), spec.source.minimum_validation_accuracy, rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise ValueError(
            "Expected an authenticated mnist_relu.v2 784-256-10 teacher selected "
            "by validation cross-entropy with validation accuracy strictly above 97%."
        )
    return teacher, metadata, digest


def _authenticate_selected_teacher_metrics(
    metadata: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    """Bind the reloaded tensor artifact to its recorded selected validation."""

    recorded_loss = metadata.get("selection_value")
    recorded_accuracy = metadata.get("selection_accuracy")
    observed_loss = validation.get("cross_entropy")
    observed_accuracy = validation.get("accuracy")
    if (
        any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in (
                recorded_loss,
                recorded_accuracy,
                observed_loss,
                observed_accuracy,
            )
        )
        or not math.isclose(
            float(recorded_loss),
            float(observed_loss),
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        or not math.isclose(
            float(recorded_accuracy),
            float(observed_accuracy),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "Expected the reloaded teacher tensors to reproduce their recorded "
            "selected validation cross-entropy and accuracy."
        )


def _teacher_validation(
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    loss_sum = 0.0
    correct = 0
    examples = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            logits = teacher.logits(inputs)
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
            correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            examples += int(labels.numel())
    if examples != 5000:
        raise RuntimeError(f"Expected the full 5,000-example validation set, observed {examples}.")
    accuracy = correct / examples
    if accuracy <= threshold:
        raise RuntimeError(
            "Expected the supplied teacher's recomputed validation accuracy to be "
            f"strictly above {threshold:.12g}; observed {accuracy:.12g}."
        )
    return {
        "examples": examples,
        "cross_entropy": loss_sum / examples,
        "accuracy": accuracy,
        "minimum_accuracy": threshold,
        "comparison_operator": ">",
        "passed": True,
    }


def _evaluate_effective_splits(
    *,
    state: torch.Tensor,
    scales: tuple[float, float],
    layout: Sequence,
    teacher: BiasFreeReluTeacher,
    data: "MnistLoaders",
    spec: StagedCrossbarTrainSpec,
    device: torch.device,
) -> dict[str, Any]:
    result = {
        "validation": _evaluate(
            effective_state=state,
            digital_scales=scales,
            layout=tuple(layout),
            teacher=teacher,
            loader=data.validation,
            device=device,
            maximum_batches=None,
            sample_limit=None,
        )
    }
    if result["validation"]["examples"] != 5000:
        raise RuntimeError("Expected an exact 5,000-example validation evaluation.")
    if spec.evaluation.evaluate_test:
        result["test"] = _evaluate(
            effective_state=state,
            digital_scales=scales,
            layout=tuple(layout),
            teacher=teacher,
            loader=data.test,
            device=device,
            maximum_batches=None,
            sample_limit=None,
        )
        if result["test"]["examples"] != 10000:
            raise RuntimeError("Expected an exact 10,000-example test evaluation.")
    return result


def _evaluate_plant_splits(
    *,
    plant: IbmOmEffectiveCrossbarPlant,
    scales: tuple[float, float],
    layout: Sequence,
    teacher: BiasFreeReluTeacher,
    data: "MnistLoaders",
    spec: StagedCrossbarTrainSpec,
    device: torch.device,
) -> dict[str, Any]:
    def evaluate(loader: Iterable) -> dict[str, Any]:
        return _evaluate_plant_states(
            plant=plant,
            digital_scales=scales,
            layout=tuple(layout),
            teacher=teacher,
            loader=loader,
            device=device,
            maximum_batches=None,
            sample_limit=None,
        )

    result = {"validation": evaluate(data.validation)}
    if result["validation"]["apparent_forward"]["examples"] != 5000:
        raise RuntimeError("Expected an exact 5,000-example validation evaluation.")
    if spec.evaluation.evaluate_test:
        result["test"] = evaluate(data.test)
        if result["test"]["apparent_forward"]["examples"] != 10000:
            raise RuntimeError("Expected an exact 10,000-example test evaluation.")
    return result


def _population_mapping(
    population: Any,
    requested: torch.Tensor,
    *,
    maximum_pulses: int,
) -> tuple[Any, dict[str, Any]]:
    codebook = build_deterministic_effective_codebook(
        population,
        maximum_pulses=maximum_pulses,
    )
    projected, indices = project_to_nearest_effective_code(codebook, requested)
    continuous = torch.maximum(
        torch.minimum(requested.detach().cpu(), population.logical_max.detach().cpu()),
        population.logical_min.detach().cpu(),
    )
    report = _mapping_report(
        population=population,
        requested=requested.detach().cpu(),
        continuous=continuous,
        codebook=projected,
        pulse_indices=indices,
        level_counts=codebook.effective_level_counts,
    )
    return codebook, report


def _artifact_records(
    store: RunStore,
    paths: Iterable[tuple[Path, str]],
) -> list[ArtifactRecord]:
    return [store.artifact_record(path, kind=kind) for path, kind in paths]


def _base_metrics(
    *,
    spec: StagedCrossbarTrainSpec,
    teacher_sha256: str,
    teacher_validation: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "stage": spec.stage.kind,
        "architecture": "standard_analog_mvm_digital_relu_analog_mvm",
        "logical_dims": list(spec.model.dims),
        "logical_weights": _CELL_COUNT,
        "programmable_active_om_states": _CELL_COUNT,
        "fixed_intrinsic_references": _CELL_COUNT,
        "fabricated_device_count_claim": None,
        "effective_state": "q_equals_a_minus_r",
        "network_forward_state": "apparent_q",
        "hidden_update_state": "persistent_q",
        "teacher_sha256": teacher_sha256,
        "teacher_validation": dict(teacher_validation),
        "evaluation_profile": spec.evaluation.profile,
        "runtime": _runtime_receipt(device),
        "evidence_class": spec.device.evidence_class,
    }


def _runtime_receipt(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Expected the runtime receipt to describe an active CUDA device.")
    return {
        "configured_device": "cuda",
        "resolved_device": str(device),
        "cuda_available": True,
        "cuda_device_name": torch.cuda.get_device_name(device),
    }


def _validate_staged_hwa_evaluation(
    value: Any,
    *,
    expected_examples: int,
    expected_evaluation_id: str,
    expected_digital_master_sha256: str,
) -> None:
    """Fail closed on the staged HWA apparent-primary evaluation schema."""

    expected_keys = {
        "network_forward_state",
        "primary_state",
        "diagnostic_state_role",
        "persistent_device_state_present",
        "apparent_forward",
        "nonpersistent_support_clamped_digital_master_diagnostic",
        "held_apparent_state_receipt",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("Expected an exact held-apparent HWA evaluation.")
    apparent = value["apparent_forward"]
    diagnostic = value[
        "nonpersistent_support_clamped_digital_master_diagnostic"
    ]
    receipt = value["held_apparent_state_receipt"]
    receipt_keys = {
        "schema",
        "schema_version",
        "evaluation_id",
        "resolved_seed",
        "resampling",
        "support_clamped_digital_master_q_sha256",
        "held_apparent_q_sha256",
        "write_noise_q_sha256",
        "generator_state_before_sha256",
        "generator_state_after_sha256",
        "configured_sigma_q",
        "observed_mean_q",
        "observed_std_q",
        "observed_minimum_q",
        "observed_maximum_q",
    }
    numeric_receipt = (
        receipt.get("configured_sigma_q") if isinstance(receipt, Mapping) else None,
        receipt.get("observed_mean_q") if isinstance(receipt, Mapping) else None,
        receipt.get("observed_std_q") if isinstance(receipt, Mapping) else None,
        receipt.get("observed_minimum_q") if isinstance(receipt, Mapping) else None,
        receipt.get("observed_maximum_q") if isinstance(receipt, Mapping) else None,
    )
    if (
        value["network_forward_state"] != "held_apparent_q"
        or value["primary_state"] != "held_apparent_q"
        or value["diagnostic_state_role"]
        != "nonpersistent_support_clamped_digital_master_q"
        or value["persistent_device_state_present"] is not False
        or not isinstance(apparent, Mapping)
        or not isinstance(diagnostic, Mapping)
        or apparent.get("examples") != expected_examples
        or diagnostic.get("examples") != expected_examples
        or not isinstance(receipt, Mapping)
        or set(receipt) != receipt_keys
        or receipt.get("schema")
        != "ebl.ibm_om_crossbar_held_apparent_hwa_evaluation"
        or receipt.get("schema_version") != 1
        or receipt.get("evaluation_id") != expected_evaluation_id
        or receipt.get("resampling")
        != "one_full_array_draw_held_across_complete_evaluation_cohort"
        or receipt.get("support_clamped_digital_master_q_sha256")
        != expected_digital_master_sha256
        or any(
            not isinstance(receipt.get(name), str)
            or len(receipt[name]) != 64
            for name in (
                "held_apparent_q_sha256",
                "write_noise_q_sha256",
                "generator_state_before_sha256",
                "generator_state_after_sha256",
            )
        )
        or isinstance(receipt.get("resolved_seed"), bool)
        or not isinstance(receipt.get("resolved_seed"), int)
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in numeric_receipt
        )
        or float(receipt.get("configured_sigma_q", 0.0)) <= 0.0
        or float(receipt.get("observed_std_q", -1.0)) < 0.0
        or float(receipt.get("observed_minimum_q", 1.0))
        > float(receipt.get("observed_maximum_q", 0.0))
    ):
        raise RuntimeError(
            "Expected apparent-primary HWA evaluation with one authenticated "
            "held draw and a nonpersistent digital-master diagnostic."
        )


def _run_offchip_hwa(
    *,
    request: Any,
    store: RunStore,
    spec: StagedCrossbarTrainSpec,
    teacher: BiasFreeReluTeacher,
    teacher_metadata: Mapping[str, Any],
    teacher_sha256: str,
    teacher_validation: Mapping[str, Any],
    data: "MnistLoaders",
    layout: tuple,
    sampler: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[ArtifactRecord]]:
    stage = spec.stage
    if not isinstance(stage, OffchipHwaStageSettings):
        raise TypeError("Expected off-chip HWA stage settings.")
    if spec.evaluation.profile != "tuning_validation_only":
        raise RuntimeError(
            "Expected off-chip HWA protocol development to remain validation-only."
        )
    logical_weights = tuple(value.detach().cpu().clone() for value in teacher.parameters())
    requested, scales = map_logical_weights(
        logical_weights,
        layout,
        weight_scaling_omega=spec.mapping.trained_source_weight_scaling_omega,
    )
    population, sampling, population_paths = _sample_population(
        layout=layout,
        assignment_seed=spec.device.assignment_seed,
        corruption_policy=spec.device.healthy_programming_corruption_policy,
        sampler=sampler,
        artifact_root=store.run_dir / "artifacts",
        role="hwa_healthy_assignment",
        bound_policy=spec.device.bound_policy,
        required_aihwkit_version=spec.runtime.required_aihwkit_version,
    )
    codebook, mapping = _population_mapping(
        population,
        requested,
        maximum_pulses=spec.device.deterministic_codebook_pulses,
    )
    compatibility_spec = SimpleNamespace(
        offchip=SimpleNamespace(
            policy=stage.policy,
            epoch_evaluation=(
                "validation_and_test"
                if spec.evaluation.evaluate_test
                else "validation_only"
            ),
            logical_learning_rates=(stage.learning_rate, stage.learning_rate),
            beta_1=stage.beta_1,
            beta_2=stage.beta_2,
            epsilon=stage.epsilon,
            forward_noise=stage.forward_noise,
            epochs=stage.epochs,
            maximum_batches=stage.maximum_batches,
            master_q_bounds=stage.master_q_bounds,
            deployment_target=stage.deployment_target,
            training_protocol=stage.training_protocol,
            objective=stage.objective,
            checkpoint_policy=stage.checkpoint_policy,
            evaluation_forward_policy=stage.evaluation_forward_policy,
        ),
        evaluation=SimpleNamespace(
            maximum_validation_batches=None,
            sample_limit=None,
        ),
    )
    deployment, report, state = _offchip_adapt(
        source_requested=requested,
        population=population,
        codebook_values=codebook.values,
        spec=compatibility_spec,
        layout=layout,
        digital_scales=scales,
        teacher=teacher,
        validation_loader=data.validation,
        train_loader=data.train,
        device=device,
        test_loader=data.test if spec.evaluation.evaluate_test else None,
    )
    master_q = state["fixed_final_master_q"].detach().cpu().to(torch.float32)
    if (
        master_q.shape != (_CELL_COUNT,)
        or not torch.equal(master_q, deployment.detach().cpu())
        or report["fixed_final_epoch"] != 10
        or any(item["examples"] != 55000 for item in report["epochs"])
        or report.get("evaluation_forward_policy")
        != "one_sampled_held_apparent_q_per_evaluation"
        or report.get("primary_evaluation_state") != "held_apparent_q"
        or report.get("diagnostic_evaluation_state")
        != "nonpersistent_support_clamped_digital_master_q"
        or report.get("persistent_device_state_present_during_offchip_hwa")
        is not False
        or not isinstance(report.get("forward_noise"), Mapping)
        or report["forward_noise"].get("base_state")
        != "support_clamped_digital_master_q_surrogate"
        or report.get("fixed_final_test") is not None
    ):
        raise RuntimeError("Expected a complete fixed-final ten-epoch HWA master.")
    _validate_staged_hwa_evaluation(
        report["initial_validation"],
        expected_examples=5000,
        expected_evaluation_id="initial.validation",
        expected_digital_master_sha256=report["initial_realized_sha256"],
    )
    for epoch_index, epoch in enumerate(report["epochs"], start=1):
        if epoch.get("test") is not None:
            raise RuntimeError("Expected validation-only HWA protocol development.")
        _validate_staged_hwa_evaluation(
            epoch["validation"],
            expected_examples=5000,
            expected_evaluation_id=f"epoch_{epoch_index:03d}.validation",
            expected_digital_master_sha256=epoch["realized_sha256"],
        )
    _validate_staged_hwa_evaluation(
        report["fixed_final_validation"],
        expected_examples=5000,
        expected_evaluation_id="epoch_010.validation",
        expected_digital_master_sha256=report["fixed_final_realized_sha256"],
    )
    master = HwaMaster(
        dims=spec.model.dims,
        layout=layout,
        digital_scales=scales,
        fixed_final_master_q=master_q,
        teacher_sha256=teacher_sha256,
        hwa_population_fingerprint=population.fingerprint,
        metadata={
            "stage": "offchip_hwa",
            "config_sha256": content_hash(to_plain_data(spec)),
            "teacher_metadata": dict(teacher_metadata),
            "assignment_seed": spec.device.assignment_seed,
            "forward_state": "sampled_apparent_q",
            "evaluation_forward_state": "one_held_sampled_apparent_q",
            "evaluation_diagnostic_state": (
                "nonpersistent_support_clamped_digital_master_q"
            ),
            "updated_state": "digital_master_q",
            "physical_plant_present": False,
            "persistent_device_updates": False,
            "selection": "fixed_final_epoch_no_selection",
        },
        report={"sampling": sampling, "mapping": mapping, "hwa": report},
    )
    master_path = store.run_dir / "checkpoints" / "hwa_master.pt"
    save_hwa_master(master_path, master)
    loaded = load_hwa_master(master_path)
    if loaded.fixed_final_master_q.shape != (_CELL_COUNT,) or loaded.teacher_sha256 != teacher_sha256:
        raise RuntimeError("Expected the staged HWA master to reload exactly.")
    report_path = store.run_dir / "artifacts" / "hwa_report.json"
    atomic_write_json(report_path, master.report)
    for epoch in report["epochs"]:
        store.append_metric({"stage": "offchip_hwa", **epoch})
    metrics = {
        **_base_metrics(
            spec=spec,
            teacher_sha256=teacher_sha256,
            teacher_validation=teacher_validation,
            device=device,
        ),
        "source_kind": "teacher",
        "assignment_seed": spec.device.assignment_seed,
        "mapping": mapping,
        "hwa": report,
        "final_evaluation": {
            "validation": report["fixed_final_validation"],
            **(
                {"test": report["fixed_final_test"]}
                if report["fixed_final_test"] is not None
                else {}
            ),
        },
        "fixed_final_master_q_sha256": tensor_sha256(master_q),
        "hwa_population_fingerprint": population.fingerprint,
        "network_forward_state": "held_apparent_q",
        "hidden_update_state": "not_applicable_no_physical_plant",
        "updated_state": "digital_master_q",
        "physical_plant_present": False,
        "persistent_device_state_present": False,
        "selection_state": "held_apparent_q_if_selection_were_enabled",
    }
    artifacts = _artifact_records(store, population_paths)
    artifacts.extend(
        (
            store.artifact_record(master_path, kind="crossbar_hwa_master"),
            store.artifact_record(report_path, kind="crossbar_hwa_report"),
        )
    )
    return metrics, artifacts


def _deployment_source(
    *,
    request: Any,
    spec: StagedCrossbarTrainSpec,
    teacher: BiasFreeReluTeacher,
    teacher_sha256: str,
    layout: tuple,
) -> tuple[torch.Tensor, tuple[float, float], str | None, dict[str, Any]]:
    stage = spec.stage
    source_kind = stage.source_kind
    if source_kind == "teacher":
        weights = tuple(value.detach().cpu().clone() for value in teacher.parameters())
        requested, scales = map_logical_weights(
            weights,
            layout,
            weight_scaling_omega=spec.mapping.trained_source_weight_scaling_omega,
        )
        return requested, scales, teacher_sha256, {"policy": "exact_teacher_tensor_mapping"}
    if source_kind == "scratch":
        weights = aihwkit_analog_linear_default_logical_weights(
            spec.model.dims,
            seed=spec.mapping.scratch_initialization_seed,
        )
        requested, scales = map_logical_weights(
            weights,
            layout,
            weight_scaling_omega=spec.mapping.scratch_weight_scaling_omega,
        )
        return requested, scales, None, {
            "policy": spec.mapping.scratch_initialization,
            "seed": spec.mapping.scratch_initialization_seed,
            "teacher_used_for_initialization": False,
        }
    if source_kind == "hwa_master":
        source_path = request.weights.expanduser().resolve()
        master = load_hwa_master(source_path)
        if (
            master.dims != spec.model.dims
            or master.layout != layout
            or master.teacher_sha256 != teacher_sha256
            or master.fixed_final_master_q.shape != (_CELL_COUNT,)
            or not bool(torch.all(torch.isfinite(master.fixed_final_master_q)))
        ):
            raise ValueError("Expected --weights to be the matched fixed-final HWA master.")
        return (
            master.fixed_final_master_q.detach().cpu().clone(),
            master.digital_scales,
            sha256_file(source_path),
            {
                "policy": "fixed_final_hwa_master",
                "source_hwa_population_fingerprint": master.hwa_population_fingerprint,
                "source_metadata": dict(master.metadata),
            },
        )
    raise RuntimeError(f"Unsupported deployment source {source_kind!r}.")


def _run_deploy(
    *,
    request: Any,
    store: RunStore,
    spec: StagedCrossbarTrainSpec,
    teacher: BiasFreeReluTeacher,
    teacher_sha256: str,
    teacher_validation: Mapping[str, Any],
    data: "MnistLoaders",
    layout: tuple,
    sampler: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[ArtifactRecord]]:
    requested, scales, source_sha, source_receipt = _deployment_source(
        request=request,
        spec=spec,
        teacher=teacher,
        teacher_sha256=teacher_sha256,
        layout=layout,
    )
    artifact_root = store.run_dir / "artifacts"
    healthy, healthy_sampling, healthy_paths = _sample_population(
        layout=layout,
        assignment_seed=spec.device.assignment_seed,
        corruption_policy=spec.device.healthy_programming_corruption_policy,
        sampler=sampler,
        artifact_root=artifact_root,
        role="healthy_programming_assignment",
        bound_policy=spec.device.bound_policy,
        required_aihwkit_version=spec.runtime.required_aihwkit_version,
    )
    published, published_sampling, published_paths = _sample_population(
        layout=layout,
        assignment_seed=spec.device.assignment_seed,
        corruption_policy=spec.device.fault_source_corruption_policy,
        sampler=sampler,
        artifact_root=artifact_root,
        role="published_fault_companion",
        bound_policy=spec.device.bound_policy,
        required_aihwkit_version=spec.runtime.required_aihwkit_version,
    )
    _matched_published_fault_overlay(
        healthy=healthy,
        published=published,
        preset_default_corrupt_devices_prob=spec.device.fault_source_preset_default_corrupt_devices_prob,
        enabled_corrupt_devices_prob=spec.device.fault_source_enabled_corrupt_devices_prob,
        corrupt_devices_range=spec.device.fault_source_corrupt_devices_range,
    )
    _codebook, mapping = _population_mapping(
        healthy,
        requested,
        maximum_pulses=spec.device.deterministic_codebook_pulses,
    )
    source_metrics = _evaluate_effective_splits(
        state=requested,
        scales=scales,
        layout=layout,
        teacher=teacher,
        data=data,
        spec=spec,
        device=device,
    )
    if spec.stage.source_kind == "teacher" and any(
        row["student_prediction_sha256"] != row["teacher_prediction_sha256"]
        for row in source_metrics.values()
    ):
        raise RuntimeError(
            "Expected deterministic direct crossbar mapping to preserve every "
            "teacher prediction before device constraints."
        )
    plant, programming = _program_endpoint(
        population=healthy,
        requested=requested,
        assignment_seed=spec.device.assignment_seed,
        endpoint_seed=spec.device.endpoint_seed,
        maximum_pulses=spec.device.maximum_programming_pulses,
        tolerance_x=spec.device.verify_tolerance_x,
        device=device,
        stream_role=STAGED_PROGRAM_STREAM_ROLE,
        random_stream_fingerprint=healthy.fingerprint,
    )
    p0_metrics = _evaluate_plant_splits(
        plant=plant,
        scales=scales,
        layout=layout,
        teacher=teacher,
        data=data,
        spec=spec,
        device=device,
    )
    bundle = crossbar_state_bundle(
        plant,
        layout=layout,
        digital_scales=scales,
        metadata={
            "stage": "deploy",
            "source_kind": spec.stage.source_kind,
            "source_artifact_sha256": source_sha,
            "teacher_sha256": teacher_sha256,
            "assignment_seed": spec.device.assignment_seed,
            "endpoint_seed": spec.device.endpoint_seed,
            "config_sha256": content_hash(to_plain_data(spec)),
            "programming_report_sha256": content_hash(programming),
            "trajectory_seed_origin": programming["trajectory_seed_origin"],
        },
    )
    staged = StagedDeviceState(
        role="healthy_p0",
        source_kind=spec.stage.source_kind,
        dims=spec.model.dims,
        assignment_seed=spec.device.assignment_seed,
        endpoint_seed=spec.device.endpoint_seed,
        teacher_sha256=teacher_sha256,
        source_artifact_sha256=source_sha,
        parent_device_state_sha256=None,
        healthy_population=healthy,
        published_population=published,
        healthy_p0=bundle,
        current=bundle,
        recovery=None,
    )
    state_path = store.run_dir / "checkpoints" / "crossbar_device_state.pt"
    save_device_state(state_path, staged)
    reloaded = load_device_state(state_path)
    if reloaded.current.plant_state_sha256 != bundle.plant_state_sha256:
        raise RuntimeError("Expected the deployed P0 device state to reload exactly.")
    report = {
        "source": source_receipt,
        "mapping": mapping,
        "healthy_population_sampling": healthy_sampling,
        "published_companion_sampling": published_sampling,
        "program_verify": programming,
        "source_evaluation": source_metrics,
        "p0_evaluation": p0_metrics,
        "state_boundary": {
            "role": "healthy_p0",
            "state_sha256": sha256_file(state_path),
            "plant_state_sha256": bundle.plant_state_sha256,
            "reprogrammed_after_p0": False,
        },
    }
    report_path = artifact_root / "deployment_report.json"
    atomic_write_json(report_path, report)
    metrics = {
        **_base_metrics(
            spec=spec,
            teacher_sha256=teacher_sha256,
            teacher_validation=teacher_validation,
            device=device,
        ),
        "source_kind": spec.stage.source_kind,
        "assignment_seed": spec.device.assignment_seed,
        "endpoint_seed": spec.device.endpoint_seed,
        "source_artifact_sha256": source_sha,
        "source": source_metrics,
        "source_evaluation": source_metrics,
        "p0": p0_metrics,
        "final_evaluation": p0_metrics,
        "mapping": mapping,
        "program_verify": programming,
        "device_state_sha256": sha256_file(state_path),
    }
    artifacts = _artifact_records(store, (*healthy_paths, *published_paths))
    artifacts.extend(
        (
            store.artifact_record(state_path, kind="crossbar_deployed_state_bundle"),
            store.artifact_record(report_path, kind="crossbar_deployment_report"),
        )
    )
    return metrics, artifacts


def _restore_current(
    staged: StagedDeviceState,
    *,
    layout: tuple,
    device: torch.device,
) -> IbmOmEffectiveCrossbarPlant:
    faulted = staged.current.state_kind == "faulted"
    if (
        staged.current.plant_state.get("trajectory_seed_derivation")
        != CROSSBAR_TRAJECTORY_SEED_DERIVATION
    ):
        raise ValueError(
            "Expected the staged device state to retain its declared trajectory-seed derivation."
        )
    expected_trajectory_seeds = crossbar_trajectory_seeds(
        staged.healthy_population,
        endpoint_seed=staged.endpoint_seed,
        stream_role=STAGED_PROGRAM_STREAM_ROLE,
        random_stream_population_fingerprint=staged.healthy_population.fingerprint,
    )
    plant, _bundle = restore_crossbar_state_bundle(
        staged.current,
        population=staged.healthy_population,
        trajectory_seeds=expected_trajectory_seeds,
        device=device,
        expected_layout=layout,
        expected_digital_scales=staged.current.digital_scales,
        expected_fault_source_population=(staged.published_population if faulted else None),
        expected_pre_fault_state=(staged.healthy_p0.plant_state if faulted else None),
    )
    return plant


def _validate_origin(
    staged: StagedDeviceState,
    *,
    spec: StagedCrossbarTrainSpec,
    teacher_sha256: str,
    layout: tuple,
    allowed_roles: set[str],
) -> None:
    deployment_metadata = staged.healthy_p0.metadata
    if (
        staged.role not in allowed_roles
        or staged.dims != spec.model.dims
        or staged.assignment_seed != spec.device.assignment_seed
        or staged.endpoint_seed != spec.device.endpoint_seed
        or staged.teacher_sha256 != teacher_sha256
        or staged.healthy_p0.layout != layout
        or staged.current.layout != layout
        or staged.healthy_p0.digital_scales != staged.current.digital_scales
        or deployment_metadata.get("stage") != "deploy"
        or deployment_metadata.get("source_kind") != staged.source_kind
        or deployment_metadata.get("source_artifact_sha256")
        != staged.source_artifact_sha256
        or deployment_metadata.get("teacher_sha256") != staged.teacher_sha256
        or deployment_metadata.get("assignment_seed") != staged.assignment_seed
        or deployment_metadata.get("endpoint_seed") != staged.endpoint_seed
    ):
        raise ValueError("Expected the device state to match this exact staged configuration.")
    if staged.role == "healthy_p0":
        if (
            staged.parent_device_state_sha256 is not None
            or staged.current.plant_state_sha256
            != staged.healthy_p0.plant_state_sha256
            or staged.current.metadata_sha256 != staged.healthy_p0.metadata_sha256
        ):
            raise ValueError("Expected an unchanged, root healthy P0 state.")
    elif staged.role == "faulted_p0":
        current_metadata = staged.current.metadata
        if (
            staged.parent_device_state_sha256 is None
            or current_metadata.get("stage") != "apply_corruption"
            or current_metadata.get("parent_device_state_sha256")
            != staged.parent_device_state_sha256
            or current_metadata.get("source_kind") != staged.source_kind
            or current_metadata.get("source_artifact_sha256")
            != staged.source_artifact_sha256
            or current_metadata.get("teacher_sha256") != staged.teacher_sha256
            or current_metadata.get("assignment_seed") != staged.assignment_seed
            or current_metadata.get("endpoint_seed") != staged.endpoint_seed
            or current_metadata.get("program_verify_calls") != 0
            or current_metadata.get("transition_sha256")
            != content_hash(staged.current.plant_state["fault_transition"])
        ):
            raise ValueError("Expected a cross-authenticated post-P0 fault boundary.")


def _run_apply_corruption(
    *,
    request: Any,
    store: RunStore,
    spec: StagedCrossbarTrainSpec,
    teacher: BiasFreeReluTeacher,
    teacher_sha256: str,
    teacher_validation: Mapping[str, Any],
    data: "MnistLoaders",
    layout: tuple,
    device: torch.device,
) -> tuple[dict[str, Any], list[ArtifactRecord]]:
    origin_path = request.device_state.expanduser().resolve()
    origin_sha = sha256_file(origin_path)
    origin = load_device_state(origin_path)
    _validate_origin(
        origin,
        spec=spec,
        teacher_sha256=teacher_sha256,
        layout=layout,
        allowed_roles={"healthy_p0"},
    )
    if origin.source_kind != spec.stage.source_kind:
        raise ValueError(
            "Expected apply_corruption --device-state source_kind to match the "
            "predeclared stage source_kind."
        )
    if origin.current.plant_state_sha256 != origin.healthy_p0.plant_state_sha256:
        raise ValueError("Expected apply_corruption to consume the exact unchanged healthy P0.")
    plant = _restore_current(origin, layout=layout, device=device)
    before = _evaluate_plant_splits(
        plant=plant,
        scales=origin.current.digital_scales,
        layout=layout,
        teacher=teacher,
        data=data,
        spec=spec,
        device=device,
    )
    mask, stuck_q, fault_report = _matched_published_fault_overlay(
        healthy=origin.healthy_population,
        published=origin.published_population,
        preset_default_corrupt_devices_prob=spec.device.fault_source_preset_default_corrupt_devices_prob,
        enabled_corrupt_devices_prob=spec.device.fault_source_enabled_corrupt_devices_prob,
        corrupt_devices_range=spec.device.fault_source_corrupt_devices_range,
    )
    transition = plant.apply_stuck_at_fault_transition(
        mask=mask,
        stuck_persistent_q=stuck_q,
        transition_id=(
            f"staged-assignment-{spec.device.assignment_seed}-endpoint-{spec.device.endpoint_seed}"
        ),
        source_population_fingerprint=origin.published_population.fingerprint,
    )
    after = _evaluate_plant_splits(
        plant=plant,
        scales=origin.current.digital_scales,
        layout=layout,
        teacher=teacher,
        data=data,
        spec=spec,
        device=device,
    )
    faulted_bundle = crossbar_state_bundle(
        plant,
        layout=layout,
        digital_scales=origin.current.digital_scales,
        metadata={
            "stage": "apply_corruption",
            "parent_device_state_sha256": origin_sha,
            "source_kind": origin.source_kind,
            "source_artifact_sha256": origin.source_artifact_sha256,
            "teacher_sha256": teacher_sha256,
            "assignment_seed": origin.assignment_seed,
            "endpoint_seed": origin.endpoint_seed,
            "transition_sha256": content_hash(transition),
            "program_verify_calls": 0,
        },
    )
    faulted = StagedDeviceState(
        role="faulted_p0",
        source_kind=origin.source_kind,
        dims=origin.dims,
        assignment_seed=origin.assignment_seed,
        endpoint_seed=origin.endpoint_seed,
        teacher_sha256=origin.teacher_sha256,
        source_artifact_sha256=origin.source_artifact_sha256,
        parent_device_state_sha256=origin_sha,
        healthy_population=origin.healthy_population,
        published_population=origin.published_population,
        healthy_p0=origin.healthy_p0,
        current=faulted_bundle,
        recovery=None,
    )
    state_path = store.run_dir / "checkpoints" / "crossbar_device_state.pt"
    save_device_state(state_path, faulted)
    load_device_state(state_path)
    report = {
        "parent_device_state_sha256": origin_sha,
        "fault_source": fault_report,
        "transition": transition,
        "pre_fault": before,
        "post_fault": after,
        "final_evaluation": after,
        "program_verify_calls": 0,
        "reprogrammed": False,
    }
    report_path = store.run_dir / "artifacts" / "corruption_report.json"
    atomic_write_json(report_path, report)
    metrics = {
        **_base_metrics(
            spec=spec,
            teacher_sha256=teacher_sha256,
            teacher_validation=teacher_validation,
            device=device,
        ),
        "source_kind": origin.source_kind,
        "assignment_seed": origin.assignment_seed,
        "endpoint_seed": origin.endpoint_seed,
        "parent_device_state_sha256": origin_sha,
        "pre_fault": before,
        "post_fault": after,
        "final_evaluation": after,
        "fault_transition": transition,
        "program_verify_calls": 0,
        "device_state_sha256": sha256_file(state_path),
    }
    return metrics, [
        store.artifact_record(state_path, kind="crossbar_corrupted_state_bundle"),
        store.artifact_record(report_path, kind="crossbar_corruption_report"),
    ]


def _validate_selection_receipt(value: Any) -> tuple[float, int | None, dict[str, Any]]:
    """Delegate selection replay to the sole tuning-analysis implementation."""

    selection = validate_analyzed_selection_receipt(value)
    if not isinstance(value, Mapping):  # pragma: no cover - delegated validation
        raise ValueError("Expected a strict IBM OM crossbar Adam selection receipt.")
    summary = {
        "study_id": value["study_id"],
        "source_plan_sha256": value["source_plan_sha256"],
        "winner": dict(value["winner"]),
        "candidate_count": len(value["candidates"]),
        "winner_recomputed": True,
        "validated_receipt": dict(value),
    }
    return selection.learning_rate, selection.pulse_cap_per_cell, summary


def _load_selection_receipt(path: Path) -> tuple[float, int | None, dict[str, Any]]:
    value, selection = load_analyzed_selection_receipt(path)
    rate, cap = selection.learning_rate, selection.pulse_cap_per_cell
    summary = {
        "study_id": value["study_id"],
        "source_plan_sha256": value["source_plan_sha256"],
        "winner": dict(value["winner"]),
        "candidate_count": len(value["candidates"]),
        "winner_recomputed": True,
        "validated_receipt": dict(value),
    }
    summary["artifact_sha256"] = sha256_file(path)
    return rate, cap, summary


def _start_state_label(origin: StagedDeviceState) -> str:
    if origin.source_kind == "hwa_master":
        prefix = "hwa"
    elif origin.source_kind == "scratch":
        prefix = "scratch"
    else:
        raise ValueError("Expected on-chip Adam to start from HWA or scratch deployment.")
    if origin.role == "healthy_p0":
        suffix = "healthy_p0"
    elif origin.role == "faulted_p0":
        suffix = "published_fault"
    else:
        raise ValueError("Expected on-chip Adam to start from a named P0 state.")
    label = f"{prefix}_{suffix}"
    if label not in _START_STATES:  # pragma: no cover - construction invariant
        raise RuntimeError("Expected a canonical matched Adam start-state label.")
    return label


def _resolve_adam_settings(
    *,
    stage: OnChipAdamStageSettings | OnChipAdamDiagnosticStageSettings,
    profile: str,
    selection_path: Path | None,
) -> tuple[float, int | None, dict[str, Any]]:
    if isinstance(stage, OnChipAdamDiagnosticStageSettings):
        if (
            profile != "diagnostic_validation_only"
            or selection_path is not None
            or not isinstance(
                stage.hyperparameters, DiagnosticLiteralAdamHyperparameters
            )
        ):
            raise ValueError(
                "Expected diagnostic Adam runs to use the literal diagnostic "
                "grid, diagnostic validation, and no selection receipt."
            )
        return (
            stage.hyperparameters.learning_rate,
            stage.hyperparameters.pulse_cap_per_cell,
            {
                "source": "literal_diagnostic_grid",
                "checkpoint_policy": stage.checkpoint_policy,
                "objective": stage.objective,
                "learning_rate_schedule": stage.learning_rate_schedule,
            },
        )
    if profile == "tuning_validation_only":
        if not isinstance(stage.hyperparameters, LiteralAdamHyperparameters) or selection_path is not None:
            raise ValueError("Expected tuning runs to use literal grid settings and no receipt.")
        return (
            stage.hyperparameters.learning_rate,
            stage.hyperparameters.pulse_cap_per_cell,
            {"source": "literal_predeclared_grid"},
        )
    if not isinstance(stage.hyperparameters, SelectionReceiptAdamHyperparameters) or selection_path is None:
        raise ValueError("Expected production Adam runs to use a strict frozen selection receipt.")
    rate, cap, receipt = _load_selection_receipt(selection_path)
    return rate, cap, {"source": "strict_selection_receipt", **receipt}


def _epoch_stream_digest() -> tuple[Any, Any]:
    return sha256(), sha256()


def _update_epoch_stream_digest(
    digests: tuple[Any, Any],
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    cpu_inputs = inputs.detach().cpu().to(torch.float32).contiguous()
    cpu_labels = labels.detach().cpu().to(torch.int64).contiguous()
    digests[0].update(cpu_inputs.numpy().tobytes())
    digests[1].update(cpu_labels.numpy().tobytes())


def _adam_objective_loss(
    *,
    objective: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    inputs: torch.Tensor,
    teacher: BiasFreeReluTeacher,
) -> torch.Tensor:
    if objective == "supervised_cross_entropy":
        return F.cross_entropy(logits, labels)
    if objective == "teacher_kl":
        with torch.no_grad():
            teacher_logits = teacher.logits(inputs)
            teacher_log_probability = F.log_softmax(teacher_logits, dim=1)
            teacher_probability = teacher_log_probability.exp()
        student_log_probability = F.log_softmax(logits, dim=1)
        return (
            teacher_probability
            * (teacher_log_probability - student_log_probability)
        ).sum(dim=1).mean()
    raise ValueError(f"Unsupported on-chip Adam objective: {objective!r}.")


def _adam_selection_objective_metric(objective: str) -> tuple[str, str]:
    if objective == "supervised_cross_entropy":
        return "validation.apparent_forward.cross_entropy", "cross_entropy"
    if objective == "teacher_kl":
        return "validation.apparent_forward.kl_teacher_student", "kl_teacher_student"
    raise ValueError(f"Unsupported on-chip Adam objective: {objective!r}.")


def _adam_selection_rank(
    *, student_accuracy: float, objective_value: float, epoch: int, kl_first: bool = False
) -> tuple[float, float, int]:
    """Rank held-apparent checkpoints by accuracy, objective, then age."""

    accuracy = float(student_accuracy)
    loss = float(objective_value)
    if (
        not math.isfinite(accuracy)
        or not 0.0 <= accuracy <= 1.0
        or not math.isfinite(loss)
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise ValueError("Expected finite held-apparent checkpoint-selection values.")
    return (loss, -accuracy, epoch) if kl_first else (-accuracy, loss, epoch)


def _diagnostic_recovery_payload(
    *,
    origin_sha: str,
    start_state: str,
    completed_epochs: int,
    total_epochs: int,
    learning_rate: float,
    pulse_cap: int,
    objective: str,
    selection_metric: str,
    selection_objective_metric: str,
    checkpoint_policy: str,
    optimizer_state_dict: Mapping[str, Any],
    train_generator_state: torch.Tensor,
    epoch_reports: Sequence[Mapping[str, Any]],
    initial: Mapping[str, Any],
    selection: Mapping[str, Any],
    best_epoch: int,
    best_accuracy: float,
    best_objective_value: float,
    best_current: IbmOmCrossbarStateBundle,
    best_optimizer_state_dict: Mapping[str, Any],
    best_train_generator_state: torch.Tensor,
) -> dict[str, Any]:
    return {
        "schema": _DIAGNOSTIC_RECOVERY_SCHEMA,
        "schema_version": _DIAGNOSTIC_RECOVERY_SCHEMA_VERSION,
        "origin_device_state_sha256": origin_sha,
        "start_state": start_state,
        "completed_epochs": completed_epochs,
        "total_epochs": total_epochs,
        "learning_rate": learning_rate,
        "pulse_cap_per_cell": pulse_cap,
        "objective": objective,
        "selection_metric": selection_metric,
        "selection_objective_metric": selection_objective_metric,
        "checkpoint_policy": checkpoint_policy,
        "optimizer_state_dict": dict(optimizer_state_dict),
        "train_generator_state": train_generator_state.detach().cpu().clone(),
        "epoch_reports": [dict(item) for item in epoch_reports],
        "initial": dict(initial),
        "selection": dict(selection),
        "best_epoch": best_epoch,
        "best_value": {
            "student_accuracy": float(best_accuracy),
            "objective_value": float(best_objective_value),
        },
        "best_current_state": best_current.state_dict(),
        "best_optimizer_state_dict": dict(best_optimizer_state_dict),
        "best_train_generator_state": (
            best_train_generator_state.detach().cpu().clone()
        ),
    }


def _recovery_payload(
    *,
    origin_sha: str,
    start_state: str,
    completed_epochs: int,
    total_epochs: int,
    learning_rate: float,
    pulse_cap: int | None,
    optimizer: PulseAdam,
    train_generator: torch.Generator,
    epoch_reports: Sequence[Mapping[str, Any]],
    initial: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": _RECOVERY_SCHEMA,
        "schema_version": 1,
        "origin_device_state_sha256": origin_sha,
        "start_state": start_state,
        "completed_epochs": completed_epochs,
        "total_epochs": total_epochs,
        "learning_rate": learning_rate,
        "pulse_cap_per_cell": pulse_cap,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_generator_state": train_generator.get_state().detach().cpu().clone(),
        "epoch_reports": [dict(item) for item in epoch_reports],
        "initial": dict(initial),
        "selection": dict(selection),
    }


def _validate_resume(
    resume: StagedDeviceState,
    *,
    origin: StagedDeviceState,
    origin_sha: str,
    start_state: str,
    learning_rate: float,
    pulse_cap: int | None,
    total_epochs: int,
    selection: Mapping[str, Any],
) -> Mapping[str, Any]:
    recovery = resume.recovery
    required = {
        "schema",
        "schema_version",
        "origin_device_state_sha256",
        "start_state",
        "completed_epochs",
        "total_epochs",
        "learning_rate",
        "pulse_cap_per_cell",
        "optimizer_state_dict",
        "train_generator_state",
        "epoch_reports",
        "initial",
        "selection",
    }
    if (
        resume.role != "adam_final"
        or resume.source_kind != origin.source_kind
        or resume.dims != origin.dims
        or resume.assignment_seed != origin.assignment_seed
        or resume.endpoint_seed != origin.endpoint_seed
        or resume.teacher_sha256 != origin.teacher_sha256
        or resume.source_artifact_sha256 != origin.source_artifact_sha256
        or resume.parent_device_state_sha256 != origin_sha
        or resume.healthy_population.fingerprint != origin.healthy_population.fingerprint
        or resume.published_population.fingerprint != origin.published_population.fingerprint
        or resume.healthy_p0.plant_state_sha256 != origin.healthy_p0.plant_state_sha256
        or resume.current.layout != origin.current.layout
        or resume.current.digital_scales != origin.current.digital_scales
        or resume.current.state_kind != origin.current.state_kind
        or not isinstance(recovery, Mapping)
        or set(recovery) != required
        or recovery.get("schema") != _RECOVERY_SCHEMA
        or recovery.get("schema_version") != 1
        or recovery.get("origin_device_state_sha256") != origin_sha
        or recovery.get("start_state") != start_state
        or recovery.get("total_epochs") != total_epochs
        or not math.isclose(float(recovery.get("learning_rate", math.nan)), learning_rate, rel_tol=0.0, abs_tol=1e-15)
        or recovery.get("pulse_cap_per_cell") != pulse_cap
        or recovery.get("selection") != selection
    ):
        raise ValueError("Expected an exact epoch-boundary Adam resume for this origin.")
    completed = recovery["completed_epochs"]
    reports = recovery["epoch_reports"]
    generator_state = recovery["train_generator_state"]
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 < completed <= total_epochs
        or not isinstance(reports, list)
        or len(reports) != completed
        or [item.get("epoch") for item in reports if isinstance(item, Mapping)]
        != list(range(1, completed + 1))
        or not isinstance(generator_state, torch.Tensor)
        or generator_state.dtype != torch.uint8
        or generator_state.device.type != "cpu"
        or generator_state.ndim != 1
    ):
        raise ValueError("Expected a complete chronological Adam epoch cursor.")
    final_report = reports[-1]
    if (
        final_report.get("apparent_sha256")
        != tensor_sha256(resume.current.plant_state["apparent"])
        or final_report.get("persistent_sha256")
        != tensor_sha256(resume.current.plant_state["persistent"])
    ):
        raise ValueError("Expected the Adam resume cursor to authenticate its final plant state.")
    return recovery


def _validate_diagnostic_resume(
    resume: StagedDeviceState,
    *,
    origin: StagedDeviceState,
    origin_sha: str,
    start_state: str,
    learning_rate: float,
    pulse_cap: int,
    total_epochs: int,
    objective: str,
    selection_metric: str,
    selection_objective_metric: str,
    checkpoint_policy: str,
    selection: Mapping[str, Any],
) -> tuple[Mapping[str, Any], IbmOmCrossbarStateBundle]:
    recovery = resume.recovery
    required = {
        "schema",
        "schema_version",
        "origin_device_state_sha256",
        "start_state",
        "completed_epochs",
        "total_epochs",
        "learning_rate",
        "pulse_cap_per_cell",
        "objective",
        "selection_metric",
        "selection_objective_metric",
        "checkpoint_policy",
        "optimizer_state_dict",
        "train_generator_state",
        "epoch_reports",
        "initial",
        "selection",
        "best_epoch",
        "best_value",
        "best_current_state",
        "best_optimizer_state_dict",
        "best_train_generator_state",
    }
    if (
        resume.role != "adam_final"
        or resume.source_kind != origin.source_kind
        or resume.dims != origin.dims
        or resume.assignment_seed != origin.assignment_seed
        or resume.endpoint_seed != origin.endpoint_seed
        or resume.teacher_sha256 != origin.teacher_sha256
        or resume.source_artifact_sha256 != origin.source_artifact_sha256
        or resume.parent_device_state_sha256 != origin_sha
        or resume.healthy_population.fingerprint
        != origin.healthy_population.fingerprint
        or resume.published_population.fingerprint
        != origin.published_population.fingerprint
        or resume.healthy_p0.plant_state_sha256
        != origin.healthy_p0.plant_state_sha256
        or resume.current.layout != origin.current.layout
        or resume.current.digital_scales != origin.current.digital_scales
        or resume.current.state_kind != origin.current.state_kind
        or not isinstance(recovery, Mapping)
        or set(recovery) != required
        or recovery.get("schema") != _DIAGNOSTIC_RECOVERY_SCHEMA
        or recovery.get("schema_version") != _DIAGNOSTIC_RECOVERY_SCHEMA_VERSION
        or recovery.get("origin_device_state_sha256") != origin_sha
        or recovery.get("start_state") != start_state
        or recovery.get("total_epochs") != total_epochs
        or not math.isclose(
            float(recovery.get("learning_rate", math.nan)),
            learning_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or recovery.get("pulse_cap_per_cell") != pulse_cap
        or recovery.get("objective") != objective
        or recovery.get("selection_metric") != selection_metric
        or recovery.get("selection_objective_metric")
        != selection_objective_metric
        or recovery.get("checkpoint_policy") != checkpoint_policy
        or recovery.get("selection") != selection
    ):
        raise ValueError(
            "Expected an exact epoch-boundary diagnostic Adam resume for this origin."
        )
    completed = recovery["completed_epochs"]
    reports = recovery["epoch_reports"]
    generator_state = recovery["train_generator_state"]
    best_epoch = recovery["best_epoch"]
    best_value = recovery["best_value"]
    best_generator_state = recovery["best_train_generator_state"]
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 <= completed <= total_epochs
        or not isinstance(reports, list)
        or len(reports) != completed
        or [item.get("epoch") for item in reports if isinstance(item, Mapping)]
        != list(range(1, completed + 1))
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 0 <= best_epoch <= completed
        or not isinstance(best_value, Mapping)
        or set(best_value) != {"student_accuracy", "objective_value"}
        or not isinstance(best_value.get("student_accuracy"), (int, float))
        or not isinstance(best_value.get("objective_value"), (int, float))
        or not math.isfinite(float(best_value["student_accuracy"]))
        or not 0.0 <= float(best_value["student_accuracy"]) <= 1.0
        or not math.isfinite(float(best_value["objective_value"]))
        or not isinstance(recovery.get("optimizer_state_dict"), Mapping)
        or not isinstance(recovery.get("best_optimizer_state_dict"), Mapping)
        or any(
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.device.type != "cpu"
            or state.ndim != 1
            for state in (generator_state, best_generator_state)
        )
    ):
        raise ValueError("Expected a complete diagnostic Adam epoch and selection cursor.")

    initial_validation = recovery["initial"]["validation"]["apparent_forward"]
    objective_metric_key = selection_objective_metric.rsplit(".", 1)[-1]
    candidates = [
        (
            float(initial_validation[_ADAM_SELECTION_METRIC_KEY]),
            float(initial_validation[objective_metric_key]),
        )
    ]
    candidates.extend(
        (
            float(item["validation"]["apparent_forward"][_ADAM_SELECTION_METRIC_KEY]),
            float(item["validation"]["apparent_forward"][objective_metric_key]),
        )
        for item in reports
    )
    expected_best_epoch = min(
        range(len(candidates)),
        key=lambda epoch: _adam_selection_rank(
            student_accuracy=candidates[epoch][0],
            objective_value=candidates[epoch][1],
            epoch=epoch,
            kl_first='validation_objective_then' in checkpoint_policy,
        ),
    )
    if (
        best_epoch != expected_best_epoch
        or not math.isclose(
            float(best_value["student_accuracy"]),
            candidates[expected_best_epoch][0],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(best_value["objective_value"]),
            candidates[expected_best_epoch][1],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Expected diagnostic Adam best-state selection to replay exactly.")

    best_current = IbmOmCrossbarStateBundle.from_state_dict(
        recovery["best_current_state"]
    )
    if (
        best_current.layout != origin.current.layout
        or best_current.digital_scales != origin.current.digital_scales
        or best_current.state_kind != origin.current.state_kind
        or best_current.population_fingerprint
        != origin.current.population_fingerprint
    ):
        raise ValueError("Expected the embedded best diagnostic state to match its origin.")
    if completed == 0:
        terminal_matches = (
            resume.current.plant_state_sha256 == origin.current.plant_state_sha256
        )
    else:
        final_report = reports[-1]
        terminal_matches = (
            final_report.get("plant_state_sha256")
            == resume.current.plant_state_sha256
            and final_report.get("apparent_sha256")
            == tensor_sha256(resume.current.plant_state["apparent"])
            and final_report.get("persistent_sha256")
            == tensor_sha256(resume.current.plant_state["persistent"])
        )
    if best_epoch == 0:
        best_matches = (
            best_current.plant_state_sha256 == origin.current.plant_state_sha256
        )
    else:
        best_report = reports[best_epoch - 1]
        best_matches = (
            best_report.get("plant_state_sha256")
            == best_current.plant_state_sha256
            and best_report.get("apparent_sha256")
            == tensor_sha256(best_current.plant_state["apparent"])
            and best_report.get("persistent_sha256")
            == tensor_sha256(best_current.plant_state["persistent"])
        )
    if not terminal_matches or not best_matches:
        raise ValueError(
            "Expected diagnostic Adam terminal and embedded best states to authenticate."
        )
    return recovery, best_current


def _run_on_chip_adam(
    *,
    request: Any,
    store: RunStore,
    spec: StagedCrossbarTrainSpec,
    teacher: BiasFreeReluTeacher,
    teacher_sha256: str,
    teacher_validation: Mapping[str, Any],
    data: "MnistLoaders",
    layout: tuple,
    device: torch.device,
) -> tuple[dict[str, Any], list[ArtifactRecord]]:
    stage = spec.stage
    if not isinstance(
        stage, (OnChipAdamStageSettings, OnChipAdamDiagnosticStageSettings)
    ):
        raise TypeError("Expected on-chip Adam stage settings.")
    diagnostic = isinstance(stage, OnChipAdamDiagnosticStageSettings)
    kl_first = 'validation_objective_then' in stage.checkpoint_policy
    def selection_rank(**values):
        return _adam_selection_rank(**values, kl_first=kl_first)
    schedule = stage.learning_rate_schedule if diagnostic else None
    origin_path = request.device_state.expanduser().resolve()
    origin_sha = sha256_file(origin_path)
    origin = load_device_state(origin_path)
    _validate_origin(
        origin,
        spec=spec,
        teacher_sha256=teacher_sha256,
        layout=layout,
        allowed_roles={"healthy_p0", "faulted_p0"},
    )
    start_state = _start_state_label(origin)
    if start_state != stage.start_state:
        raise ValueError(
            "Expected on_chip_adam --device-state to match the predeclared "
            f"start_state {stage.start_state!r}; observed {start_state!r}."
        )
    selection_path = (
        None if request.selection_receipt is None else request.selection_receipt.expanduser().resolve()
    )
    learning_rate, pulse_cap, selection = _resolve_adam_settings(
        stage=stage,
        profile=spec.evaluation.profile,
        selection_path=selection_path,
    )
    selection_metric = _ADAM_SELECTION_METRIC
    selection_metric_key = _ADAM_SELECTION_METRIC_KEY
    (
        selection_objective_metric,
        selection_objective_metric_key,
    ) = _adam_selection_objective_metric(stage.objective)

    if kl_first:
        selection_metric = selection_objective_metric
    origin_plant = _restore_current(origin, layout=layout, device=device)
    initial = _evaluate_plant_splits(
        plant=origin_plant,
        scales=origin.current.digital_scales,
        layout=layout,
        teacher=teacher,
        data=data,
        spec=spec,
        device=device,
    )
    plant = origin_plant
    reports: list[dict[str, Any]] = []
    completed_epochs = 0
    resume_sha: str | None = None
    resume_recovery: Mapping[str, Any] | None = None
    resume_best_current: IbmOmCrossbarStateBundle | None = None
    if request.resume is not None:
        resume_path = request.resume.expanduser().resolve()
        resume_sha = sha256_file(resume_path)
        resume = load_device_state(resume_path)
        if diagnostic:
            if pulse_cap is None:  # pragma: no cover - strict config invariant
                raise RuntimeError("Expected a finite diagnostic Adam pulse cap.")
            resume_recovery, resume_best_current = _validate_diagnostic_resume(
                resume,
                origin=origin,
                origin_sha=origin_sha,
                start_state=start_state,
                learning_rate=learning_rate,
                pulse_cap=pulse_cap,
                total_epochs=stage.epochs,
                objective=stage.objective,
                selection_metric=selection_metric,
                selection_objective_metric=selection_objective_metric,
                checkpoint_policy=stage.checkpoint_policy,
                selection=selection,
            )
        else:
            resume_recovery = _validate_resume(
                resume,
                origin=origin,
                origin_sha=origin_sha,
                start_state=start_state,
                learning_rate=learning_rate,
                pulse_cap=pulse_cap,
                total_epochs=stage.epochs,
                selection=selection,
            )
        plant = _restore_current(resume, layout=layout, device=device)
        completed_epochs = int(resume_recovery["completed_epochs"])
        reports = [dict(item) for item in resume_recovery["epoch_reports"]]
        if resume_recovery["initial"] != initial:
            raise ValueError("Expected resumed Adam initial metrics to match the origin replay.")

    selection_generator = torch.Generator(device=device.type)
    selection_generator.manual_seed(
        _recovery_seed(
            runtime_seed=spec.runtime.seed,
            assignment_seed=spec.device.assignment_seed,
            endpoint_seed=spec.device.endpoint_seed,
        )
    )
    optimizer = PulseAdam(
        size=_CELL_COUNT,
        layout=layout,
        learning_rates=(learning_rate, learning_rate),
        betas=(stage.beta_1, stage.beta_2),
        epsilon=stage.epsilon,
        layer_scope=stage.layer_scope,
        nominal_dw_min=population_nominal_step(origin.healthy_population),
        pulse_cap_per_cell=pulse_cap,
        generator=selection_generator,
        device=device,
    )
    best_epoch = 0
    best_accuracy = float(
        initial["validation"]["apparent_forward"][selection_metric_key]
    )
    best_objective_value = float(
        initial["validation"]["apparent_forward"][selection_objective_metric_key]
    )
    best_current = origin.current
    best_optimizer_state_dict = optimizer.state_dict()
    best_train_generator_state = data.train_generator.get_state().detach().cpu().clone()
    if resume_recovery is not None:
        optimizer.load_state_dict(resume_recovery["optimizer_state_dict"])
        data.train_generator.set_state(resume_recovery["train_generator_state"])
        if diagnostic:
            if resume_best_current is None:  # pragma: no cover - validated branch
                raise RuntimeError("Expected an embedded diagnostic best state.")
            best_epoch = int(resume_recovery["best_epoch"])
            best_accuracy = float(
                resume_recovery["best_value"]["student_accuracy"]
            )
            best_objective_value = float(
                resume_recovery["best_value"]["objective_value"]
            )
            best_current = resume_best_current
            best_optimizer_state_dict = dict(
                resume_recovery["best_optimizer_state_dict"]
            )
            best_train_generator_state = resume_recovery[
                "best_train_generator_state"
            ].detach().cpu().clone()

    update_port = plant.restricted_recovery_update_port()
    epoch_artifacts: list[ArtifactRecord] = []
    zero_learning_rate_control = diagnostic and learning_rate == 0.0
    for epoch in range(completed_epochs + 1, stage.epochs + 1):
        if schedule and schedule['kind'] == 'exponential':
            progress = min((epoch - 1) / (schedule['decay_epochs'] - 1), 1.0)
            optimizer.learning_rate_scale = schedule['final_factor'] ** progress
        else:
            optimizer.learning_rate_scale = 1.0
        effective_learning_rate = learning_rate * optimizer.learning_rate_scale
        objective_loss_sum = 0.0
        cross_entropy_sum = 0.0
        correct = 0
        examples = 0
        batches = 0
        commanded = 0
        clipped = 0
        blocked = 0
        stream_digests = _epoch_stream_digest()
        for batch_index, (inputs, labels) in enumerate(
            limited(data.train, stage.maximum_batches)
        ):
            _update_epoch_stream_digest(stream_digests, inputs, labels)
            batch_examples = int(labels.numel())
            examples += batch_examples
            batches = batch_index + 1
            if zero_learning_rate_control:
                continue
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            apparent_q = update_port.apparent.to(device=device).requires_grad_(True)
            logits = standard_crossbar_logits(
                inputs,
                apparent_q,
                layout,
                digital_scales=origin.current.digital_scales,
            )
            loss = _adam_objective_loss(
                objective=stage.objective,
                logits=logits,
                labels=labels,
                inputs=inputs,
                teacher=teacher,
            )
            gradient = torch.autograd.grad(loss, apparent_q, only_inputs=True)[0]
            pulse = optimizer.step(gradient, update_port)
            objective_loss_sum += float(loss.item()) * batch_examples
            cross_entropy_sum += float(
                F.cross_entropy(logits.detach(), labels).item()
            ) * batch_examples
            correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            commanded += int(pulse["commanded_pulses"])
            clipped += int(pulse["probability_clipped"])
            blocked += int(pulse["blocked_at_cap"])
        if examples != stage.repair_examples or batches != stage.maximum_batches:
            raise RuntimeError(
                "Expected each on-chip Adam epoch to consume the full ordered "
                f"55,000-example stream; observed examples={examples}, batches={batches}."
            )
        evaluation = _evaluate_plant_splits(
            plant=plant,
            scales=origin.current.digital_scales,
            layout=layout,
            teacher=teacher,
            data=data,
            spec=spec,
            device=device,
        )
        current_bundle = crossbar_state_bundle(
            plant,
            layout=layout,
            digital_scales=origin.current.digital_scales,
            metadata={
                "stage": stage.kind,
                "origin_device_state_sha256": origin_sha,
                "completed_epochs": epoch,
                "learning_rate": learning_rate,
                "pulse_cap_per_cell": pulse_cap,
                "start_state": start_state,
            },
        )
        optimizer_cumulative = optimizer.report()
        zero_learning_rate_noop_check: dict[str, Any] | None = None
        if zero_learning_rate_control:
            zero_learning_rate_noop_check = {
                "origin_plant_state_sha256": origin.current.plant_state_sha256,
                "current_plant_state_sha256": current_bundle.plant_state_sha256,
                "optimizer_steps": optimizer_cumulative["optimizer_steps"],
                "requested_nonzero_commands": optimizer_cumulative[
                    "requested_nonzero_commands"
                ],
                "commanded_pulses": optimizer_cumulative["commanded_pulses"],
                "passed": bool(
                    current_bundle.plant_state_sha256
                    == origin.current.plant_state_sha256
                    and optimizer_cumulative["optimizer_steps"] == 0
                    and optimizer_cumulative["requested_nonzero_commands"] == 0
                    and optimizer_cumulative["commanded_pulses"] == 0
                ),
            }
            if not zero_learning_rate_noop_check["passed"]:
                raise RuntimeError(
                    "Expected learning-rate zero to preserve the complete plant "
                    "and optimizer no-op state."
                )
        epoch_report = {
            "epoch": epoch,
            "batches": batches,
            "examples": examples,
            "train_objective": stage.objective,
            "effective_learning_rate": effective_learning_rate,
            "learning_rate_schedule": schedule,
            "train_objective_loss": (
                None if zero_learning_rate_control else objective_loss_sum / examples
            ),
            "train_cross_entropy": (
                None if zero_learning_rate_control else cross_entropy_sum / examples
            ),
            "train_pre_update_accuracy": (
                None if zero_learning_rate_control else correct / examples
            ),
            "execution": (
                "zero_learning_rate_no_gradient_no_write_control"
                if zero_learning_rate_control
                else "apparent_forward_gradient_to_persistent_pulse_update"
            ),
            "commanded_pulses": commanded,
            "probability_clipped": clipped,
            "blocked_at_cap": blocked,
            "ordered_model_inputs_sha256": stream_digests[0].hexdigest(),
            "ordered_labels_sha256": stream_digests[1].hexdigest(),
            "validation": evaluation["validation"],
            "test": evaluation.get("test"),
            "optimizer_cumulative": optimizer_cumulative,
            "plant_state_sha256": current_bundle.plant_state_sha256,
            "zero_learning_rate_noop_check": zero_learning_rate_noop_check,
            **update_port.state_hash_receipt(),
        }
        reports.append(epoch_report)
        store.append_metric(
            {
                "stage": stage.kind,
                "source_kind": origin.source_kind,
                "start_state": start_state,
                "assignment_seed": origin.assignment_seed,
                "endpoint_seed": origin.endpoint_seed,
                **epoch_report,
            }
        )
        candidate_accuracy = float(
            evaluation["validation"]["apparent_forward"][selection_metric_key]
        )
        candidate_objective_value = float(
            evaluation["validation"]["apparent_forward"][
                selection_objective_metric_key
            ]
        )
        if diagnostic and selection_rank(
            student_accuracy=candidate_accuracy,
            objective_value=candidate_objective_value,
            epoch=epoch,
        ) < selection_rank(
            student_accuracy=best_accuracy,
            objective_value=best_objective_value,
            epoch=best_epoch,
        ):
            best_epoch = epoch
            best_accuracy = candidate_accuracy
            best_objective_value = candidate_objective_value
            best_current = current_bundle
            best_optimizer_state_dict = optimizer.state_dict()
            best_train_generator_state = (
                data.train_generator.get_state().detach().cpu().clone()
            )
        if diagnostic:
            if pulse_cap is None:  # pragma: no cover - strict config invariant
                raise RuntimeError("Expected a finite diagnostic Adam pulse cap.")
            recovery = _diagnostic_recovery_payload(
                origin_sha=origin_sha,
                start_state=start_state,
                completed_epochs=epoch,
                total_epochs=stage.epochs,
                learning_rate=learning_rate,
                pulse_cap=pulse_cap,
                objective=stage.objective,
                selection_metric=selection_metric,
                selection_objective_metric=selection_objective_metric,
                checkpoint_policy=stage.checkpoint_policy,
                optimizer_state_dict=optimizer.state_dict(),
                train_generator_state=data.train_generator.get_state(),
                epoch_reports=reports,
                initial=initial,
                selection=selection,
                best_epoch=best_epoch,
                best_accuracy=best_accuracy,
                best_objective_value=best_objective_value,
                best_current=best_current,
                best_optimizer_state_dict=best_optimizer_state_dict,
                best_train_generator_state=best_train_generator_state,
            )
        else:
            recovery = _recovery_payload(
                origin_sha=origin_sha,
                start_state=start_state,
                completed_epochs=epoch,
                total_epochs=stage.epochs,
                learning_rate=learning_rate,
                pulse_cap=pulse_cap,
                optimizer=optimizer,
                train_generator=data.train_generator,
                epoch_reports=reports,
                initial=initial,
                selection=selection,
            )
        # Keep recoverable epoch 1, every fifth epoch, and the terminal state.
        if stage.epochs > 3 and epoch != 1 and epoch % 5 and epoch != stage.epochs:
            continue
        checkpoint = StagedDeviceState(
            role="adam_final",
            source_kind=origin.source_kind,
            dims=origin.dims,
            assignment_seed=origin.assignment_seed,
            endpoint_seed=origin.endpoint_seed,
            teacher_sha256=origin.teacher_sha256,
            source_artifact_sha256=origin.source_artifact_sha256,
            parent_device_state_sha256=origin_sha,
            healthy_population=origin.healthy_population,
            published_population=origin.published_population,
            healthy_p0=origin.healthy_p0,
            current=current_bundle,
            recovery=recovery,
        )
        checkpoint_path = store.run_dir / "checkpoints" / f"epoch_{epoch:03d}_resume.pt"
        _save_and_verify_on_chip_recovery_state(checkpoint_path, checkpoint.state_dict())
        verified = load_device_state(checkpoint_path)
        if verified.recovery["completed_epochs"] != epoch:
            raise RuntimeError("Expected the exact Adam epoch cursor to reload.")
        epoch_artifacts.append(
            store.artifact_record(checkpoint_path, kind="crossbar_adam_epoch_resume")
        )

    if len(reports) != stage.epochs:
        raise RuntimeError(
            f"Expected the Adam run to contain {stage.epochs} epoch reports."
        )
    training_final_bundle = crossbar_state_bundle(
        plant,
        layout=layout,
        digital_scales=origin.current.digital_scales,
        metadata={
            "stage": stage.kind,
            "origin_device_state_sha256": origin_sha,
            "completed_epochs": stage.epochs,
            "learning_rate": learning_rate,
            "pulse_cap_per_cell": pulse_cap,
            "start_state": start_state,
        },
    )
    training_final_validation = reports[-1]["validation"]
    report_kind = "crossbar_adam_recovery_report"
    report_name = "adam_recovery_report.json"
    if diagnostic:
        if pulse_cap is None:  # pragma: no cover - strict config invariant
            raise RuntimeError("Expected a finite diagnostic Adam pulse cap.")
        selection_candidates = [
            {
                "epoch": 0,
                "value": {
                    "student_accuracy": float(
                        initial["validation"]["apparent_forward"][
                            selection_metric_key
                        ]
                    ),
                    "objective_value": float(
                        initial["validation"]["apparent_forward"][
                            selection_objective_metric_key
                        ]
                    ),
                },
                "accuracy": float(
                    initial["validation"]["apparent_forward"][selection_metric_key]
                ),
                "objective_value": float(
                    initial["validation"]["apparent_forward"][
                        selection_objective_metric_key
                    ]
                ),
                "validation": initial["validation"],
            }
        ]
        selection_candidates.extend(
            {
                "epoch": int(item["epoch"]),
                "value": {
                    "student_accuracy": float(
                        item["validation"]["apparent_forward"][selection_metric_key]
                    ),
                    "objective_value": float(
                        item["validation"]["apparent_forward"][
                            selection_objective_metric_key
                        ]
                    ),
                },
                "accuracy": float(
                    item["validation"]["apparent_forward"][selection_metric_key]
                ),
                "objective_value": float(
                    item["validation"]["apparent_forward"][
                        selection_objective_metric_key
                    ]
                ),
                "validation": item["validation"],
            }
            for item in reports
        )
        recomputed_best = min(
            selection_candidates,
            key=lambda item: selection_rank(
                student_accuracy=float(item["accuracy"]),
                objective_value=float(item["objective_value"]),
                epoch=int(item["epoch"]),
            ),
        )
        if (
            int(recomputed_best["epoch"]) != best_epoch
            or not math.isclose(
                float(recomputed_best["accuracy"]),
                best_accuracy,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(recomputed_best["objective_value"]),
                best_objective_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("Expected diagnostic checkpoint selection to replay.")
        selected_generator = torch.Generator(device=device.type)
        selected_generator.manual_seed(
            _recovery_seed(
                runtime_seed=spec.runtime.seed,
                assignment_seed=spec.device.assignment_seed,
                endpoint_seed=spec.device.endpoint_seed,
            )
        )
        selected_optimizer = PulseAdam(
            size=_CELL_COUNT,
            layout=layout,
            learning_rates=(learning_rate, learning_rate),
            betas=(stage.beta_1, stage.beta_2),
            epsilon=stage.epsilon,
            layer_scope=stage.layer_scope,
            nominal_dw_min=population_nominal_step(origin.healthy_population),
            pulse_cap_per_cell=pulse_cap,
            generator=selected_generator,
            device=device,
        )
        selected_optimizer.load_state_dict(best_optimizer_state_dict)
        final_recovery = _diagnostic_recovery_payload(
            origin_sha=origin_sha,
            start_state=start_state,
            completed_epochs=best_epoch,
            total_epochs=stage.epochs,
            learning_rate=learning_rate,
            pulse_cap=pulse_cap,
            objective=stage.objective,
            selection_metric=selection_metric,
            selection_objective_metric=selection_objective_metric,
            checkpoint_policy=stage.checkpoint_policy,
            optimizer_state_dict=selected_optimizer.state_dict(),
            train_generator_state=best_train_generator_state,
            epoch_reports=reports[:best_epoch],
            initial=initial,
            selection=selection,
            best_epoch=best_epoch,
            best_accuracy=best_accuracy,
            best_objective_value=best_objective_value,
            best_current=best_current,
            best_optimizer_state_dict=selected_optimizer.state_dict(),
            best_train_generator_state=best_train_generator_state,
        )
        final_bundle = best_current
        final = {"validation": recomputed_best["validation"]}
        selected_optimizer_report = selected_optimizer.report()
        zero_learning_rate_noop_summary: dict[str, Any] | None = None
        if zero_learning_rate_control:
            zero_learning_rate_noop_summary = {
                "origin_plant_state_sha256": origin.current.plant_state_sha256,
                "selected_plant_state_sha256": final_bundle.plant_state_sha256,
                "training_final_plant_state_sha256": (
                    training_final_bundle.plant_state_sha256
                ),
                "selected_optimizer_steps": selected_optimizer_report[
                    "optimizer_steps"
                ],
                "training_final_optimizer_steps": optimizer.report()[
                    "optimizer_steps"
                ],
                "all_epoch_checks_passed": all(
                    item["zero_learning_rate_noop_check"]["passed"]
                    for item in reports
                ),
            }
            zero_learning_rate_noop_summary["passed"] = bool(
                zero_learning_rate_noop_summary["origin_plant_state_sha256"]
                == zero_learning_rate_noop_summary["selected_plant_state_sha256"]
                == zero_learning_rate_noop_summary[
                    "training_final_plant_state_sha256"
                ]
                and zero_learning_rate_noop_summary["selected_optimizer_steps"] == 0
                and zero_learning_rate_noop_summary[
                    "training_final_optimizer_steps"
                ]
                == 0
                and zero_learning_rate_noop_summary["all_epoch_checks_passed"]
            )
            if not zero_learning_rate_noop_summary["passed"]:
                raise RuntimeError(
                    "Expected the complete learning-rate-zero run to be a no-op."
                )
        diagnostic_fields: dict[str, Any] = {
            "objective": stage.objective,
            "selection_metric": selection_metric,
            "selection_objective_metric": selection_objective_metric,
            "selection_candidates": selection_candidates,
            "selected_epoch": best_epoch,
            "selected_value": {
                "student_accuracy": best_accuracy,
                "objective_value": best_objective_value,
            },
            "selected_accuracy": best_accuracy,
            "selected_objective_value": best_objective_value,
            "selected_validation": recomputed_best["validation"],
            "non_degradation_passed": bool(
                selection_rank(
                    student_accuracy=best_accuracy,
                    objective_value=best_objective_value,
                    epoch=0,
                )
                <= selection_rank(
                    student_accuracy=float(selection_candidates[0]["accuracy"]),
                    objective_value=float(
                        selection_candidates[0]["objective_value"]
                    ),
                    epoch=0,
                )
            ),
            "training_final_epoch": stage.epochs,
            "training_final_validation": training_final_validation,
            "selected_optimizer": selected_optimizer_report,
            "training_final_optimizer": optimizer.report(),
            "zero_learning_rate_noop_check": zero_learning_rate_noop_summary,
            "selected_plant_state_sha256": final_bundle.plant_state_sha256,
            "training_final_plant_state_sha256": (
                training_final_bundle.plant_state_sha256
            ),
            "training_final_apparent_sha256": tensor_sha256(
                training_final_bundle.plant_state["apparent"]
            ),
            "training_final_persistent_sha256": tensor_sha256(
                training_final_bundle.plant_state["persistent"]
            ),
        }
        report_kind = "crossbar_adam_diagnostic_report"
        report_name = "adam_diagnostic_report.json"
    else:
        final_bundle = training_final_bundle
        final_recovery = _recovery_payload(
            origin_sha=origin_sha,
            start_state=start_state,
            completed_epochs=stage.epochs,
            total_epochs=stage.epochs,
            learning_rate=learning_rate,
            pulse_cap=pulse_cap,
            optimizer=optimizer,
            train_generator=data.train_generator,
            epoch_reports=reports,
            initial=initial,
            selection=selection,
        )
        final = {
            "validation": reports[-1]["validation"],
            **({"test": reports[-1]["test"]} if reports[-1]["test"] is not None else {}),
        }
        selected_optimizer_report = optimizer.report()
        diagnostic_fields = {}
    final_state = StagedDeviceState(
        role="adam_final",
        source_kind=origin.source_kind,
        dims=origin.dims,
        assignment_seed=origin.assignment_seed,
        endpoint_seed=origin.endpoint_seed,
        teacher_sha256=origin.teacher_sha256,
        source_artifact_sha256=origin.source_artifact_sha256,
        parent_device_state_sha256=origin_sha,
        healthy_population=origin.healthy_population,
        published_population=origin.published_population,
        healthy_p0=origin.healthy_p0,
        current=final_bundle,
        recovery=final_recovery,
    )
    final_path = store.run_dir / "checkpoints" / "crossbar_device_state.pt"
    save_device_state(final_path, final_state)
    reloaded_final = load_device_state(final_path)
    if diagnostic:
        if pulse_cap is None:  # pragma: no cover - strict config invariant
            raise RuntimeError("Expected a finite diagnostic Adam pulse cap.")
        _validate_diagnostic_resume(
            reloaded_final,
            origin=origin,
            origin_sha=origin_sha,
            start_state=start_state,
            learning_rate=learning_rate,
            pulse_cap=pulse_cap,
            total_epochs=stage.epochs,
            objective=stage.objective,
            selection_metric=selection_metric,
            selection_objective_metric=selection_objective_metric,
            checkpoint_policy=stage.checkpoint_policy,
            selection=selection,
        )
    report = {
        "source_kind": origin.source_kind,
        "start_state": start_state,
        "assignment_seed": origin.assignment_seed,
        "endpoint_seed": origin.endpoint_seed,
        "origin_device_state_sha256": origin_sha,
        "resume_input_sha256": resume_sha,
        "learning_rate": learning_rate,
        "pulse_cap_per_cell": pulse_cap,
        "selection": selection,
        **diagnostic_fields,
        "resolved_hyperparameters": {
            "learning_rate": learning_rate,
            "pulse_cap_per_cell": pulse_cap,
            "beta_1": stage.beta_1,
            "beta_2": stage.beta_2,
            "epsilon": stage.epsilon,
            "layer_scope": stage.layer_scope,
        },
        "initial": initial,
        "epochs": reports,
        "final": final,
        "final_evaluation": final,
        "optimizer": selected_optimizer_report,
        "learning_implementation": dict(_ADAM_IMPLEMENTATION),
        "checkpoint_policy": stage.checkpoint_policy,
        "network_forward_state": "apparent_q",
        "hidden_update_state": "persistent_q",
    }
    report_path = store.run_dir / "artifacts" / report_name
    atomic_write_json(report_path, report)
    metrics = {
        **_base_metrics(
            spec=spec,
            teacher_sha256=teacher_sha256,
            teacher_validation=teacher_validation,
            device=device,
        ),
        "source_kind": origin.source_kind,
        "input_role": origin.role,
        "start_state": start_state,
        "assignment_seed": origin.assignment_seed,
        "endpoint_seed": origin.endpoint_seed,
        "learning_rate": learning_rate,
        "pulse_cap_per_cell": pulse_cap,
        "origin_device_state_sha256": origin_sha,
        "resume_input_sha256": resume_sha,
        "selection": selection,
        **diagnostic_fields,
        "resolved_hyperparameters": {
            "learning_rate": learning_rate,
            "pulse_cap_per_cell": pulse_cap,
            "beta_1": stage.beta_1,
            "beta_2": stage.beta_2,
            "epsilon": stage.epsilon,
            "layer_scope": stage.layer_scope,
        },
        "initial": initial,
        "epochs": reports,
        "final": final,
        "final_evaluation": final,
        "optimizer": selected_optimizer_report,
        "learning_implementation": dict(_ADAM_IMPLEMENTATION),
        "device_state_sha256": sha256_file(final_path),
    }
    artifacts = epoch_artifacts
    artifacts.extend(
        (
            store.artifact_record(final_path, kind="crossbar_adam_final_state_bundle"),
            store.artifact_record(report_path, kind=report_kind),
        )
    )
    return metrics, artifacts


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Expected finite non-empty diagnostic metric values.")
    normalized = [float(value) for value in values]
    return {
        "mean": sum(normalized) / len(normalized),
        "minimum": min(normalized),
        "maximum": max(normalized),
    }


def _fresh_apparent_diagnostic_seed(
    *,
    configured_seed: int,
    runtime_seed: int,
    assignment_seed: int,
    endpoint_seed: int,
    start_state: str,
    intervention: str,
) -> tuple[int, dict[str, Any]]:
    """Resolve one paired redraw stream shared by healthy/faulted P0."""

    if start_state not in {"hwa_healthy_p0", "hwa_published_fault"}:
        raise ValueError("Expected a named HWA P0 start for paired redraws.")
    resolved = derive_seed(
        configured_seed,
        runtime_seed,
        assignment_seed,
        endpoint_seed,
        intervention,
    )
    return resolved, {
        "scheme": "derive_seed_without_start_state_for_paired_noise_v1",
        "configured_seed": configured_seed,
        "runtime_seed": runtime_seed,
        "assignment_seed": assignment_seed,
        "endpoint_seed": endpoint_seed,
        "intervention": intervention,
        "excluded_pairing_dimension": "start_state",
        "matched_across_start_states": True,
        "resolved_seed": resolved,
    }


def _run_fresh_apparent_diagnostic(
    *,
    request: Any,
    store: RunStore,
    spec: StagedCrossbarTrainSpec,
    teacher: BiasFreeReluTeacher,
    teacher_sha256: str,
    teacher_validation: Mapping[str, Any],
    data: "MnistLoaders",
    layout: tuple,
    device: torch.device,
) -> tuple[dict[str, Any], list[ArtifactRecord]]:
    stage = spec.stage
    if not isinstance(stage, FreshApparentDiagnosticStageSettings):
        raise TypeError("Expected fresh-apparent diagnostic stage settings.")
    origin_path = request.device_state.expanduser().resolve()
    origin_sha = sha256_file(origin_path)
    origin = load_device_state(origin_path)
    _validate_origin(
        origin,
        spec=spec,
        teacher_sha256=teacher_sha256,
        layout=layout,
        allowed_roles={"healthy_p0", "faulted_p0"},
    )
    start_state = _start_state_label(origin)
    if start_state != stage.start_state:
        raise ValueError(
            "Expected fresh_apparent_diagnostic --device-state to match the "
            f"predeclared start_state {stage.start_state!r}; observed {start_state!r}."
        )
    plant = _restore_current(origin, layout=layout, device=device)
    initial = _evaluate_plant_splits(
        plant=plant,
        scales=origin.current.digital_scales,
        layout=layout,
        teacher=teacher,
        data=data,
        spec=spec,
        device=device,
    )
    before = {
        "apparent_sha256": tensor_sha256(plant.apparent),
        "persistent_sha256": tensor_sha256(plant.persistent),
    }
    resolved_seed, seed_derivation = _fresh_apparent_diagnostic_seed(
        configured_seed=stage.seed,
        runtime_seed=spec.runtime.seed,
        assignment_seed=origin.assignment_seed,
        endpoint_seed=origin.endpoint_seed,
        start_state=start_state,
        intervention=stage.intervention,
    )
    generator = torch.Generator(device=device.type)
    generator.manual_seed(resolved_seed)
    generator_before = tensor_sha256(generator.get_state())
    draw_reports: list[dict[str, Any]] = []
    for draw_index in range(1, stage.draws + 1):
        fresh_apparent, noise = _sample_aihwkit_om_apparent_write_noise(
            persistent_q=plant.persistent,
            nominal_dw_min=origin.healthy_population.nominal_dw_min,
            write_noise_std=origin.healthy_population.write_noise_std,
            relative_scale=stage.relative_scale,
            generator=generator,
        )
        evaluation = _evaluate_effective_splits(
            state=fresh_apparent,
            scales=origin.current.digital_scales,
            layout=layout,
            teacher=teacher,
            data=data,
            spec=spec,
            device=device,
        )
        validation = {
            "network_forward_state": "counterfactual_fresh_apparent_q_diagnostic",
            "hidden_update_state": "persistent_q_unchanged",
            "apparent_forward": evaluation["validation"],
            "persistent_diagnostic": initial["validation"][
                "persistent_diagnostic"
            ],
        }
        draw_report = {
            "draw": draw_index,
            "fresh_apparent_q_sha256": tensor_sha256(fresh_apparent),
            "noise_q_sha256": tensor_sha256(noise),
            "validation": validation,
        }
        draw_reports.append(draw_report)
        store.append_metric(
            {
                "stage": stage.kind,
                "start_state": start_state,
                "assignment_seed": origin.assignment_seed,
                "endpoint_seed": origin.endpoint_seed,
                **draw_report,
            }
        )
    after = {
        "apparent_sha256": tensor_sha256(plant.apparent),
        "persistent_sha256": tensor_sha256(plant.persistent),
    }
    mutation_check = {
        "before": before,
        "after": after,
        "passed": before == after,
        "writes_commanded": 0,
        "persistent_updates": 0,
        "held_apparent_updates": 0,
    }
    if not mutation_check["passed"]:
        raise RuntimeError(
            "Expected the counterfactual apparent redraw diagnostic not to mutate the plant."
        )
    metric_names = (
        "student_accuracy",
        "kl_teacher_student",
        "cross_entropy",
    )
    summary = {
        name: _metric_summary(
            [
                float(item["validation"]["apparent_forward"][name])
                for item in draw_reports
            ]
        )
        for name in metric_names
    }
    report = {
        "source_kind": origin.source_kind,
        "start_state": start_state,
        "assignment_seed": origin.assignment_seed,
        "endpoint_seed": origin.endpoint_seed,
        "origin_device_state_sha256": origin_sha,
        "intervention": stage.intervention,
        "interpretation": stage.interpretation,
        "mutation_policy": stage.mutation_policy,
        "draw_count": stage.draws,
        "relative_scale": stage.relative_scale,
        "configured_seed": stage.seed,
        "resolved_seed": resolved_seed,
        "matched_across_start_states": True,
        "seed_derivation": seed_derivation,
        "generator_state_before_sha256": generator_before,
        "generator_state_after_sha256": tensor_sha256(generator.get_state()),
        "noise_equation": (
            "q_fresh_equals_persistent_q_plus_write_noise_std_times_"
            "nominal_dw_min_times_standard_normal"
        ),
        "physical_read_claim": False,
        "initial": initial,
        "draws": draw_reports,
        "summary": summary,
        "mutation_check": mutation_check,
        "network_forward_state": "counterfactual_fresh_apparent_q_diagnostic",
        "hidden_update_state": "persistent_q_unchanged",
    }
    report_path = (
        store.run_dir / "artifacts" / "fresh_apparent_diagnostic_report.json"
    )
    atomic_write_json(report_path, report)
    metrics = {
        **_base_metrics(
            spec=spec,
            teacher_sha256=teacher_sha256,
            teacher_validation=teacher_validation,
            device=device,
        ),
        "input_role": origin.role,
        **report,
    }
    return metrics, [
        store.artifact_record(
            report_path, kind="crossbar_fresh_apparent_diagnostic_report"
        )
    ]


def _run_stage(
    *,
    request: Any,
    store: RunStore,
    sampler: Path | None,
) -> tuple[dict[str, Any], list[ArtifactRecord]]:
    spec = request.spec
    torch.manual_seed(spec.runtime.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.runtime.seed)
    device = _cuda_device(spec)
    # This must be the first streamed metric so launchers can fail closed on
    # the process's resolved training device rather than trusting the config.
    store.append_metric({"mode": "runtime_gate", "runtime": _runtime_receipt(device)})
    layout = _layout(spec)
    data = build_mnist_loaders(spec.data, data_seed=spec.runtime.data_seed)
    teacher, teacher_metadata, teacher_sha256 = _load_teacher(
        request.teacher_weights,
        spec=spec,
        device=device,
    )
    teacher_validation = _teacher_validation(
        teacher,
        data.validation,
        device=device,
        threshold=spec.source.minimum_validation_accuracy,
    )
    _authenticate_selected_teacher_metrics(teacher_metadata, teacher_validation)
    common = {
        "request": request,
        "store": store,
        "spec": spec,
        "teacher": teacher,
        "teacher_sha256": teacher_sha256,
        "teacher_validation": teacher_validation,
        "data": data,
        "layout": layout,
        "device": device,
    }
    if spec.stage.kind == "offchip_hwa":
        if sampler is None:
            raise RuntimeError("Expected the HWA population sampler.")
        return _run_offchip_hwa(
            **common,
            teacher_metadata=teacher_metadata,
            sampler=sampler,
        )
    if spec.stage.kind == "deploy":
        if sampler is None:
            raise RuntimeError("Expected the deployment population sampler.")
        return _run_deploy(**common, sampler=sampler)
    if spec.stage.kind == "apply_corruption":
        return _run_apply_corruption(**common)
    if spec.stage.kind in {"on_chip_adam", "on_chip_adam_diagnostic"}:
        return _run_on_chip_adam(**common)
    if spec.stage.kind == "fresh_apparent_diagnostic":
        return _run_fresh_apparent_diagnostic(**common)
    raise RuntimeError(f"Unsupported staged kind: {spec.stage.kind!r}.")


def run_train(request: "TrainRequest") -> int:
    """Run one strict, CUDA-only stage and finalize its canonical RunStore."""

    input_artifacts, sampler = _validate_request(request)
    spec = request.spec
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=input_artifacts,
        resume_capability=(
            "exact"
            if spec.stage.kind in {"on_chip_adam", "on_chip_adam_diagnostic"}
            else "unsupported"
        ),
    )
    try:
        metrics, artifacts = _run_stage(request=request, store=store, sampler=sampler)
        store.complete(metrics=metrics, artifacts=artifacts)
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_train"]
