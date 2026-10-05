"""Runtime for cohort-A RESET training with KL or label supervision."""

from __future__ import annotations

from collections.abc import Callable
import gc
from pathlib import Path
import random
from typing import Any, Iterable, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_relu_drn.components import (
    conductance_statistics,
    fit_positive_logit_gain,
)
from experiments.mnist_relu_drn_reset.components import (
    PairedSupervision,
    build_reset_student_stack,
)
from experiments.mnist_relu_drn_reset.config import (
    DIFFERENTIAL_EXPERIMENT_ID,
    FACTORIAL_EXPERIMENT_ID,
    ResetTrainSpec,
    ResetValidateSpec,
)
from experiments.mnist_shared import build_mnist_loaders, teacher_batches
from experiments.schema import to_plain_data
from model.resistive.interaction import DenseResistive, SignedDenseResistive
from training.checkpoint import (
    encode_named_weights,
    load_epoch_boundary_checkpoint,
    load_named_weights,
    save_encoded_named_weights,
    save_epoch_boundary_checkpoint,
)
from training.core import engine
from training.core.engine import (
    EvaluationComponents,
    ExperimentComponents,
    GradientsReadyEvent,
)
from training.core.guards import FiniteGradientGuard
from training.core.probes import PerBatchProbe
from training.measured_trace import MeasuredTraceOptimizer

if TYPE_CHECKING:
    from ebl.cli import TrainRequest, ValidateRequest


_ROOT = Path(__file__).resolve().parents[2]


def _requires_production_restart(spec: ResetTrainSpec) -> bool:
    """Return whether LR selection and production must use fresh stacks."""

    return (
        spec.experiment_id
        in {DIFFERENTIAL_EXPERIMENT_ID, FACTORIAL_EXPERIMENT_ID}
        or spec.model.include_biases
    )


def _seed_runtime(seed: int) -> None:
    """Match the historical small_drn.v1 process RNG initialization."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amplification_index_report(stack: Any) -> list[dict[str, Any]]:
    report = []
    for interaction in stack.bundle.energy._interactions:
        if not isinstance(
            interaction, (DenseResistive, SignedDenseResistive)
        ):
            continue
        item = {
            "pre_layer": interaction._layer_pre.name,
            "post_layer": interaction._layer_post.name,
            "resolved_pre_index": interaction._logical_pre_index,
            "resolved_post_index": interaction._logical_post_index,
        }
        if isinstance(interaction, SignedDenseResistive):
            item.update(
                {
                    "interaction": "differential_pair",
                    "voltage_amp": interaction._voltage_amp,
                    "current_amp": interaction._current_amp,
                    "forward_gain": interaction._forward_gain(),
                    "post_layer_metric": interaction._post_metric(),
                }
            )
        report.append(item)
    return report


def _differential_topology_signature(
    report: Any,
) -> tuple[tuple[int, int, float, float], ...] | None:
    """Project provenance onto model-local numerical differential semantics."""

    if not isinstance(report, (list, tuple)):
        return None
    try:
        return tuple(
            (
                int(item["resolved_pre_index"]),
                int(item["resolved_post_index"]),
                float(item["forward_gain"]),
                float(item["post_layer_metric"]),
            )
            for item in report
            if item.get("interaction") == "differential_pair"
        )
    except (KeyError, TypeError, ValueError):
        return None


def _input(role: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Expected --{role.replace('_', '-')} to name an existing file. "
            f"Provided value: {str(resolved)!r}."
        )
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved)}


def _validate_train_request(request: Any) -> None:
    experiment_id = request.definition.experiment_id
    if request.teacher_weights is None:
        raise ValueError(
            f"Expected {experiment_id} training to receive "
            "--teacher-weights. Provided value: None."
        )
    if request.device_data is None:
        raise ValueError(
            f"Expected {experiment_id} training to receive "
            "--device-data. Provided value: None."
        )
    if request.weights is not None or request.base_weights is not None:
        raise ValueError(
            "Expected RESET training not to use --weights or --base-weights. "
            f"Provided value: weights={request.weights!r}, "
            f"base_weights={request.base_weights!r}."
        )


def _load_teacher(
    path: Path, *, device: torch.device
) -> tuple[BiasFreeReluTeacher, dict[str, Any]]:
    teacher = BiasFreeReluTeacher(device=device)
    loaded = load_named_weights(path, teacher.catalog)
    architecture = loaded.metadata.get("architecture")
    if architecture != "bias_free_relu_784_50_10":
        raise ValueError(
            "Expected --teacher-weights metadata architecture to equal "
            "'bias_free_relu_784_50_10'. "
            f"Provided value: {architecture!r}."
        )
    return teacher, dict(loaded.metadata)


def _batch_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str,
) -> dict[str, float | int]:
    teacher_log_prob = F.log_softmax(teacher_logits, dim=1)
    teacher_prob = teacher_log_prob.exp()
    student_log_prob = F.log_softmax(student_logits, dim=1)
    kl = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=1)
    cross_entropy = F.cross_entropy(student_logits, labels, reduction="none")
    targets = F.one_hot(labels, num_classes=student_logits.shape[1]).to(
        dtype=student_logits.dtype
    )
    paired_squared_error = 0.5 * (student_logits - targets).square().sum(dim=1)
    objective_values = {
        "teacher_kl": kl,
        "cross_entropy": cross_entropy,
        "paired_squared_error": paired_squared_error,
    }[objective]
    student_prediction = student_logits.argmax(dim=1)
    teacher_prediction = teacher_logits.argmax(dim=1)
    return {
        "examples": int(labels.shape[0]),
        "objective_sum": float(objective_values.sum().item()),
        "kl_sum": float(kl.sum().item()),
        "cross_entropy_sum": float(cross_entropy.sum().item()),
        "paired_squared_error_sum": float(
            paired_squared_error.sum().item()
        ),
        "student_correct": int(student_prediction.eq(labels).sum().item()),
        "teacher_correct": int(teacher_prediction.eq(labels).sum().item()),
        "agreement": int(student_prediction.eq(teacher_prediction).sum().item()),
        "student_score_squared": float(student_logits.square().sum().item()),
        "teacher_score_squared": float(teacher_logits.square().sum().item()),
    }


def _empty_totals() -> dict[str, float | int]:
    return {
        "examples": 0,
        "objective_sum": 0.0,
        "kl_sum": 0.0,
        "cross_entropy_sum": 0.0,
        "paired_squared_error_sum": 0.0,
        "student_correct": 0,
        "teacher_correct": 0,
        "agreement": 0,
        "student_score_squared": 0.0,
        "teacher_score_squared": 0.0,
    }


def _add_totals(totals: dict[str, float | int], batch: dict[str, float | int]) -> None:
    for name, value in batch.items():
        totals[name] += value


def _finalize_metrics(
    totals: dict[str, float | int],
    *,
    objective: str,
    diagnostic_gain: float | None = None,
    calibrated_kl_sum: float | None = None,
) -> dict[str, Any]:
    examples = int(totals["examples"])
    if examples == 0:
        raise ValueError(
            "Expected RESET DRN evaluation to process examples. Provided value: 0."
        )
    score_values = examples * 10
    result = {
        "examples": examples,
        "objective": objective,
        "objective_loss": float(totals["objective_sum"]) / examples,
        "kl_teacher_student": float(totals["kl_sum"]) / examples,
        "cross_entropy": float(totals["cross_entropy_sum"]) / examples,
        "paired_squared_error": (
            float(totals["paired_squared_error_sum"]) / examples
        ),
        "student_accuracy": int(totals["student_correct"]) / examples,
        "teacher_accuracy": int(totals["teacher_correct"]) / examples,
        "teacher_agreement": int(totals["agreement"]) / examples,
        "student_logit_rms": (
            float(totals["student_score_squared"]) / score_values
        ) ** 0.5,
        "teacher_logit_rms": (
            float(totals["teacher_score_squared"]) / score_values
        ) ** 0.5,
        "fixed_logit_gain": 1.0,
    }
    if diagnostic_gain is not None and calibrated_kl_sum is not None:
        result["posthoc_logit_gain"] = float(diagnostic_gain)
        result["posthoc_calibrated_kl_teacher_student"] = (
            float(calibrated_kl_sum) / examples
        )
    return result


class SupervisionProbe:
    """RESET supervision metrics, shared by training and evaluation passes.

    ``batch_loss`` is the latest batch's mean objective, for the LR-safety
    observer. ``diagnostic_gain`` adds the post-hoc calibrated KL.
    """

    name = "supervision"

    def __init__(self, objective: str, *, diagnostic_gain: float | None = None) -> None:
        self._objective = objective
        self._diagnostic_gain = diagnostic_gain

    def reset(self) -> None:
        self._totals = _empty_totals()
        self._calibrated_kl_sum = 0.0
        self.batch_loss: float | None = None

    def observe(self, event: Any) -> None:
        targets = event.batch.targets
        with torch.no_grad():
            student_logits = event.components.cost_fn.student_logits()
            batch = _batch_metrics(
                student_logits,
                targets.logits,
                targets.labels,
                objective=self._objective,
            )
            _add_totals(self._totals, batch)
            self.batch_loss = float(batch["objective_sum"]) / int(batch["examples"])
            if self._diagnostic_gain is not None:
                teacher_log_prob = F.log_softmax(targets.logits, dim=1)
                teacher_prob = teacher_log_prob.exp()
                calibrated_log_prob = F.log_softmax(
                    student_logits * self._diagnostic_gain,
                    dim=1,
                )
                self._calibrated_kl_sum += float(
                    (
                        teacher_prob
                        * (teacher_log_prob - calibrated_log_prob)
                    ).sum().item()
                )

    def result(self) -> dict[str, Any]:
        return _finalize_metrics(
            self._totals,
            objective=self._objective,
            diagnostic_gain=self._diagnostic_gain,
            calibrated_kl_sum=(
                self._calibrated_kl_sum
                if self._diagnostic_gain is not None
                else None
            ),
        )


def evaluate(
    stack: Any,
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    *,
    maximum_batches: int | None = None,
    sample_limit: int | None = None,
    diagnostic_gain: float | None = None,
) -> dict[str, Any]:
    probe = SupervisionProbe(stack.cost.objective, diagnostic_gain=diagnostic_gain)
    with torch.no_grad():
        result = engine.evaluate(
            EvaluationComponents(stack.network, stack.cost, stack.minimizer),
            teacher_batches(
                loader,
                teacher,
                stack.device,
                maximum_batches=maximum_batches,
                sample_limit=sample_limit,
            ),
            probes=(probe,),
        )
    return result.probe_value(probe.name)


def train_epoch(
    stack: Any,
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    *,
    maximum_batches: int | None,
    reset_input: bool = True,
    observer: Callable[
        [float, tuple[torch.Tensor, ...], tuple[Any, ...]], None
    ]
    | None = None,
) -> tuple[dict[str, Any], int]:
    """One RESET training epoch; ``observer`` sees each batch's gradients.

    The observer runs after the finite-gradient guard and before the update,
    as ``observer(batch_loss, gradients, parameters)``.
    """

    probe = SupervisionProbe(stack.cost.objective)
    handlers: list[Callable[[Any], None]] = [
        FiniteGradientGuard(
            tuple(binding.key for binding in stack.bundle.catalog.trainable),
            label="RESET-training",
        )
    ]
    if observer is not None:

        def notify(event: Any) -> None:
            if isinstance(event, GradientsReadyEvent):
                observer(probe.batch_loss, event.gradients, event.components.parameters)

        handlers.append(notify)
    result = engine.train_epoch(
        ExperimentComponents(
            stack.network,
            stack.cost,
            stack.minimizer,
            tuple(stack.bundle.energy.params()),
            stack.differentiator,
            stack.optimizer,
        ),
        teacher_batches(loader, teacher, stack.device, maximum_batches=maximum_batches),
        probes=(probe,),
        event_handlers=handlers,
        reset_input=reset_input,
    )
    return result.probe_value(probe.name), result.batch_count


def _collect_logits(
    stack: Any,
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
) -> tuple[torch.Tensor, torch.Tensor]:
    probe = PerBatchProbe(
        "logits",
        lambda event: (
            event.components.cost_fn.student_logits().detach(),
            event.batch.targets.logits.detach(),
        ),
    )
    with torch.no_grad():
        result = engine.evaluate(
            EvaluationComponents(stack.network, stack.cost, stack.minimizer),
            teacher_batches(loader, teacher, stack.device),
            probes=(probe,),
        )
    pieces = result.probe_value(probe.name)
    if not pieces:
        raise ValueError(
            "Expected post-hoc calibration to process examples. Provided value: 0."
        )
    return (
        torch.cat([student for student, _teacher in pieces]),
        torch.cat([teacher_logits for _student, teacher_logits in pieces]),
    )


def _posthoc_calibration(
    stack: Any,
    teacher: BiasFreeReluTeacher,
    validation_loader: Iterable,
) -> dict[str, Any]:
    student_logits, teacher_logits = _collect_logits(
        stack, teacher, validation_loader
    )
    result = fit_positive_logit_gain(
        student_logits,
        teacher_logits,
        gain_min=0.001,
        gain_max=1_000_000.0,
        steps=181,
    )
    return {
        **result,
        "fit_split": "validation",
        "operational": False,
        "used_for_training": False,
        "used_for_checkpoint_selection": False,
        "used_for_accuracy": False,
    }


def _selected_payload(
    stack: Any,
    *,
    spec: ResetTrainSpec,
    teacher_path: Path,
    teacher_sha256: str,
    learning_rates: tuple[float, ...],
    epoch: int,
    validation: dict[str, Any],
) -> dict[str, Any]:
    data_report = stack.optimizer.data_report
    return encode_named_weights(
        stack.bundle.catalog,
        metadata={
            "experiment_id": spec.experiment_id,
            "encoding": spec.model.encoding,
            "include_biases": spec.model.include_biases,
            "amplification_indexing": spec.model.amplification_indexing,
            "amplification_indices": _amplification_index_report(stack),
            "objective": spec.settings.objective,
            "initialization": "measured_reset",
            "fixed_logit_gain": 1.0,
            "temperature": 1.0,
            "reset_input_between_batches": (
                spec.settings.reset_input_between_batches
            ),
            "teacher_path": str(teacher_path.expanduser().resolve()),
            "teacher_sha256": teacher_sha256,
            "device_data_sha256": data_report["source_sha256"],
            "device_assignment_sha256_by_parameter": data_report[
                "assignment_sha256_by_parameter"
            ],
            "selected_learning_rates": list(learning_rates),
            "selection_metric": "validation.objective_loss",
            "selection_value": validation["objective_loss"],
            "selection_epoch": epoch,
            "selection_student_accuracy": validation["student_accuracy"],
            "selection_teacher_agreement": validation["teacher_agreement"],
        },
    )


def _validate_checkpoint_metadata(
    metadata: Any,
    *,
    objective: str,
    teacher_sha256: str,
    experiment_id: str = "mnist_relu_drn_reset.v1",
    encoding: str = "single",
    include_biases: bool = False,
    amplification_indexing: str = "logical",
) -> None:
    expected = {
        "experiment_id": experiment_id,
        "encoding": encoding,
        "objective": objective,
        "initialization": "measured_reset",
        "fixed_logit_gain": 1.0,
        "temperature": 1.0,
        "teacher_sha256": teacher_sha256,
    }
    if include_biases:
        expected["include_biases"] = True
    if amplification_indexing != "logical":
        expected["amplification_indexing"] = amplification_indexing
    if experiment_id == FACTORIAL_EXPERIMENT_ID:
        expected["include_biases"] = include_biases
        expected["amplification_indexing"] = amplification_indexing
        expected["reset_input_between_batches"] = True
    if experiment_id == DIFFERENTIAL_EXPERIMENT_ID:
        expected["include_biases"] = False
        expected["amplification_indexing"] = "logical"
        expected["reset_input_between_batches"] = True
    mismatches = {
        name: {"expected": value, "provided": metadata.get(name)}
        for name, value in expected.items()
        if metadata.get(name) != value
    }
    if experiment_id == DIFFERENTIAL_EXPERIMENT_ID:
        expected_signature = (
            (0, 1, 1.0, 1.0),
            (1, 2, 4.0, 0.0625),
        )
        provided_signature = _differential_topology_signature(
            metadata.get("amplification_indices")
        )
        if provided_signature != expected_signature:
            mismatches["amplification_indices"] = {
                "expected": expected_signature,
                "provided": provided_signature,
            }
    if mismatches:
        raise ValueError(
            "Expected RESET checkpoint metadata to match the requested "
            f"experiment and teacher. Provided value: {mismatches!r}."
        )


def run_train(request: "TrainRequest") -> int:
    spec = request.spec
    if not isinstance(spec, ResetTrainSpec):
        raise TypeError(
            "Expected a RESET MNIST train definition to resolve "
            "ResetTrainSpec. "
            f"Provided value: {type(spec).__name__}."
        )
    _validate_train_request(request)
    input_artifacts = [
        _input("teacher_weights", request.teacher_weights),
        _input("device_data", request.device_data),
    ]
    if request.resume is not None:
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
        _seed_runtime(spec.runtime.seed)
        device = torch.device(spec.runtime.device)
        data = build_mnist_loaders(spec.data, data_seed=spec.runtime.data_seed)
        teacher, teacher_metadata = _load_teacher(
            request.teacher_weights, device=device
        )
        teacher_sha = sha256_file(request.teacher_weights)
        if (
            request.resume is not None
            and spec.model.amplification_indexing
            == "legacy_process_global"
        ):
            dummy = build_reset_student_stack(
                spec,
                device_data_path=None,
                enable_measured=False,
            )
            del dummy
            gc.collect()
        stack = build_reset_student_stack(
            spec,
            device_data_path=request.device_data,
            enable_measured=True,
        )
        if not isinstance(stack.cost, PairedSupervision) or stack.cost.gain != 1.0:
            raise RuntimeError(
                "Expected unit-gain paired supervision for RESET training. "
                f"Provided value: {type(stack.cost).__name__}."
            )
        if not isinstance(stack.optimizer, MeasuredTraceOptimizer):
            raise RuntimeError(
                "Expected measured cohort-A optimizer for RESET training. "
                f"Provided value: {type(stack.optimizer).__name__}."
            )

        weights_path = store.run_dir / "checkpoints" / "weights.pt"
        resume_path = store.run_dir / "checkpoints" / "resume.pt"
        selection_path = store.run_dir / "artifacts" / "learning_rate_selection.json"
        start_epoch = 0
        global_step = 0
        selected_weights = None
        selected_epoch = -1
        selected_validation: dict[str, Any] | None = None
        last_train = None
        selection_report: dict[str, Any]

        if request.resume is not None:
            resumed = load_epoch_boundary_checkpoint(
                request.resume,
                catalog=stack.bundle.catalog,
                optimizer=stack.optimizer,
                dataloader_generators={"train": data.train_generator},
            )
            _validate_checkpoint_metadata(
                resumed.metadata,
                objective=spec.settings.objective,
                teacher_sha256=teacher_sha,
                experiment_id=spec.experiment_id,
                encoding=spec.model.encoding,
                include_biases=spec.model.include_biases,
                amplification_indexing=spec.model.amplification_indexing,
            )
            start_epoch = resumed.epoch
            global_step = resumed.global_step
            selected_weights = resumed.selected_weights
            selected_epoch = int(resumed.progress_state["selected_epoch"])
            selected_validation = dict(
                resumed.progress_state["selected_validation"]
            )
            initial = dict(resumed.progress_state["initial_validation"])
            initial_conductances = dict(
                resumed.progress_state["initial_conductances"]
            )
            selection_report = dict(
                resumed.progress_state["learning_rate_selection"]
            )
            if selected_weights is not None:
                save_encoded_named_weights(
                    weights_path,
                    selected_weights,
                    catalog=stack.bundle.catalog,
                )
            store.append_metric(
                {
                    "mode": "resume",
                    "start_epoch": start_epoch,
                    "global_step": global_step,
                    "learning_rates": list(stack.optimizer.learning_rates()),
                    "validation": evaluate(
                        stack,
                        teacher,
                        data.validation,
                        maximum_batches=spec.settings.max_validation_batches,
                    ),
                }
            )
        else:
            from experiments.mnist_relu_drn_reset.learning_rate_selection import (
                select_reset_learning_rates,
            )

            selection_report = select_reset_learning_rates(
                spec,
                stack,
                teacher,
                data,
                store=store,
                train_epoch=train_epoch,
                evaluate=evaluate,
            )
            if _requires_production_restart(spec):
                # Rebuild from an exact reseed so selection and production
                # exercise the repeated-construction lifecycle. Historical
                # controls retain their archived process-global semantics;
                # intended circuits retain model-local topology indices.
                selected_rates = tuple(
                    float(value)
                    for value in selection_report["selection"][
                        "selected_learning_rate_vector"
                    ]
                )
                selection_assignments = dict(
                    stack.optimizer.data_report[
                        "assignment_sha256_by_parameter"
                    ]
                )
                del stack, teacher, data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _seed_runtime(spec.runtime.seed)
                stack = build_reset_student_stack(
                    spec,
                    device_data_path=request.device_data,
                    enable_measured=True,
                )
                data = build_mnist_loaders(
                    spec.data,
                    data_seed=spec.runtime.data_seed,
                )
                teacher, teacher_metadata = _load_teacher(
                    request.teacher_weights,
                    device=device,
                )
                if not isinstance(stack.optimizer, MeasuredTraceOptimizer):
                    raise RuntimeError(
                        "Expected the rebuilt production "
                        "stack to use MeasuredTraceOptimizer. Provided value: "
                        f"{type(stack.optimizer).__name__}."
                    )
                production_assignments = dict(
                    stack.optimizer.data_report[
                        "assignment_sha256_by_parameter"
                    ]
                )
                if production_assignments != selection_assignments:
                    raise RuntimeError(
                        "Expected exact measured-device assignment parity "
                        "between LR selection and rebuilt production. "
                        "Provided value: "
                        f"selection={selection_assignments!r}, "
                        f"production={production_assignments!r}."
                    )
                stack.optimizer.set_learning_rates(selected_rates)
                production_initialization = (
                    stack.optimizer.initialize_at_pulse_zero()
                )
                store.append_metric(
                    {
                        "mode": "production_restart",
                        "semantics": (
                            "controlled_exact_reseed_and_rebuild"
                            if spec.experiment_id == FACTORIAL_EXPERIMENT_ID
                            else (
                                "differential_model_local_reseed_and_rebuild"
                                if spec.experiment_id
                                == DIFFERENTIAL_EXPERIMENT_ID
                                else "historical_exact_reseed_and_rebuild"
                            )
                        ),
                        "selected_learning_rates": list(selected_rates),
                        "assignment_sha256_by_parameter": (
                            production_assignments
                        ),
                        "initialization": production_initialization,
                        "amplification_indices": (
                            _amplification_index_report(stack)
                        ),
                    }
                )
            initial = evaluate(
                stack,
                teacher,
                data.validation,
                maximum_batches=spec.settings.max_validation_batches,
            )
            initial_conductances = conductance_statistics(
                stack.bundle.catalog,
                encoding=spec.model.encoding,
            )
            store.append_metric(
                {
                    "mode": "initialization",
                    "initialization": "measured_reset",
                    "fixed_logit_gain": 1.0,
                    "learning_rates": list(stack.optimizer.learning_rates()),
                    "programming": stack.optimizer.programming_report,
                    "validation": initial,
                    "conductances": initial_conductances,
                }
            )
        atomic_write_json(selection_path, selection_report)
        learning_rates = tuple(
            float(value) for value in stack.optimizer.learning_rates()
        )
        expected_rate_count = len(stack.optimizer.param_groups)
        if (
            len(learning_rates) != expected_rate_count
            or any(value <= 0.0 for value in learning_rates)
        ):
            raise RuntimeError(
                f"Expected {expected_rate_count} positive selected learning "
                "rates before production "
                f"training. Provided value: {learning_rates!r}."
            )

        if selected_validation is None:
            selected_validation = dict(initial)
            selected_weights = _selected_payload(
                stack,
                spec=spec,
                teacher_path=request.teacher_weights,
                teacher_sha256=teacher_sha,
                learning_rates=learning_rates,
                epoch=-1,
                validation=initial,
            )
            save_encoded_named_weights(
                weights_path, selected_weights, catalog=stack.bundle.catalog
            )

        last_validation = evaluate(
            stack,
            teacher,
            data.validation,
            maximum_batches=spec.settings.max_validation_batches,
        )
        for epoch in range(start_epoch, spec.settings.num_epochs):
            last_train, batches = train_epoch(
                stack,
                teacher,
                data.train,
                maximum_batches=spec.settings.max_batches,
                reset_input=spec.settings.reset_input_between_batches,
            )
            global_step += batches
            last_validation = evaluate(
                stack,
                teacher,
                data.validation,
                maximum_batches=spec.settings.max_validation_batches,
            )
            improved = (
                last_validation["objective_loss"]
                < selected_validation["objective_loss"]
            )
            if improved:
                selected_epoch = epoch
                selected_validation = dict(last_validation)
                selected_weights = _selected_payload(
                    stack,
                    spec=spec,
                    teacher_path=request.teacher_weights,
                    teacher_sha256=teacher_sha,
                    learning_rates=learning_rates,
                    epoch=epoch,
                    validation=last_validation,
                )
                save_encoded_named_weights(
                    weights_path,
                    selected_weights,
                    catalog=stack.bundle.catalog,
                )
            if selected_weights is None:
                raise RuntimeError(
                    "Expected selected RESET weights at every epoch boundary."
                )
            progress = {
                "fixed_logit_gain": 1.0,
                "objective": spec.settings.objective,
                "selected_learning_rates": list(learning_rates),
                "selected_epoch": selected_epoch,
                "selected_validation": selected_validation,
                "initial_validation": initial,
                "initial_conductances": initial_conductances,
                "learning_rate_selection": selection_report,
            }
            checkpoint_metadata = {
                "experiment_id": spec.experiment_id,
                "encoding": spec.model.encoding,
                "include_biases": spec.model.include_biases,
                "amplification_indexing": spec.model.amplification_indexing,
                "amplification_indices": _amplification_index_report(stack),
                "objective": spec.settings.objective,
                "initialization": "measured_reset",
                "fixed_logit_gain": 1.0,
                "temperature": 1.0,
                "reset_input_between_batches": (
                    spec.settings.reset_input_between_batches
                ),
                "teacher_sha256": teacher_sha,
                "device_data_sha256": stack.optimizer.data_report[
                    "source_sha256"
                ],
                "device_assignment_sha256_by_parameter": stack.optimizer.data_report[
                    "assignment_sha256_by_parameter"
                ],
            }
            save_epoch_boundary_checkpoint(
                resume_path,
                catalog=stack.bundle.catalog,
                epoch=epoch + 1,
                global_step=global_step,
                optimizer=stack.optimizer,
                progress_state=progress,
                selected_weights=selected_weights,
                dataloader_generators={"train": data.train_generator},
                metadata=checkpoint_metadata,
            )
            if (
                (epoch + 1) % spec.settings.log_every == 0
                or epoch + 1 == spec.settings.num_epochs
            ):
                store.append_metric(
                    {
                        "mode": "train",
                        "epoch": epoch,
                        "completed_epochs": epoch + 1,
                        "global_step": global_step,
                        "train": last_train,
                        "validation": last_validation,
                        "selected": improved,
                        "selected_epoch": selected_epoch,
                        "selected_objective_loss": selected_validation[
                            "objective_loss"
                        ],
                        "learning_rates": list(learning_rates),
                        "conductances": conductance_statistics(
                            stack.bundle.catalog,
                            encoding=spec.model.encoding,
                        ),
                        "measured_projection": stack.optimizer.programming_report,
                    }
                )

        if selected_weights is None or selected_validation is None:
            raise RuntimeError("Expected training to provide selected RESET weights.")
        if not resume_path.exists():
            save_epoch_boundary_checkpoint(
                resume_path,
                catalog=stack.bundle.catalog,
                epoch=start_epoch,
                global_step=global_step,
                optimizer=stack.optimizer,
                progress_state={
                    "fixed_logit_gain": 1.0,
                    "objective": spec.settings.objective,
                    "selected_learning_rates": list(learning_rates),
                    "selected_epoch": selected_epoch,
                    "selected_validation": selected_validation,
                    "initial_validation": initial,
                    "initial_conductances": initial_conductances,
                    "learning_rate_selection": selection_report,
                },
                selected_weights=selected_weights,
                dataloader_generators={"train": data.train_generator},
                metadata={
                    "experiment_id": spec.experiment_id,
                    "encoding": spec.model.encoding,
                    "include_biases": spec.model.include_biases,
                    "amplification_indexing": spec.model.amplification_indexing,
                    "amplification_indices": _amplification_index_report(
                        stack
                    ),
                    "objective": spec.settings.objective,
                    "initialization": "measured_reset",
                    "fixed_logit_gain": 1.0,
                    "temperature": 1.0,
                    "reset_input_between_batches": (
                        spec.settings.reset_input_between_batches
                    ),
                    "teacher_sha256": teacher_sha,
                    "device_data_sha256": stack.optimizer.data_report[
                        "source_sha256"
                    ],
                    "device_assignment_sha256_by_parameter": stack.optimizer.data_report[
                        "assignment_sha256_by_parameter"
                    ],
                },
            )

        final_conductances = conductance_statistics(
            stack.bundle.catalog, encoding=spec.model.encoding
        )
        final_programming = stack.optimizer.programming_report
        load_named_weights(weights_path, stack.bundle.catalog)
        posthoc = _posthoc_calibration(stack, teacher, data.validation)
        selected_conductances = conductance_statistics(
            stack.bundle.catalog, encoding=spec.model.encoding
        )
        store.complete(
            metrics={
                "teacher": {"sha256": teacher_sha, "metadata": teacher_metadata},
                "encoding": spec.model.encoding,
                "include_biases": spec.model.include_biases,
                "amplification_indexing": spec.model.amplification_indexing,
                "amplification_indices": _amplification_index_report(stack),
                "objective": spec.settings.objective,
                "initialization": "measured_reset",
                "fixed_logit_gain": 1.0,
                "temperature": 1.0,
                "reset_input_between_batches": (
                    spec.settings.reset_input_between_batches
                ),
                "update_backend": "measured_cohort_a",
                "selected_learning_rates": list(learning_rates),
                "learning_rate_selection": selection_report["selection"],
                "initial_validation": initial,
                "last_train": last_train,
                "last_validation": last_validation,
                "selected": {"epoch": selected_epoch, **selected_validation},
                "posthoc_validation_calibration": posthoc,
                "acceptance_gate": {
                    "minimum_validation_accuracy": spec.settings.minimum_validation_accuracy,
                    "passed": selected_validation["student_accuracy"]
                    >= spec.settings.minimum_validation_accuracy,
                },
                "initial_conductances": initial_conductances,
                "final_conductances": final_conductances,
                "selected_conductances": selected_conductances,
                "measured_programming": final_programming,
            },
            artifacts=(
                store.artifact_record(weights_path, kind="selected_named_weights"),
                store.artifact_record(resume_path, kind="epoch_boundary_resume"),
                store.artifact_record(
                    selection_path, kind="learning_rate_selection"
                ),
            ),
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


def run_validate(request: "ValidateRequest") -> int:
    spec = request.spec
    if not isinstance(spec, ResetValidateSpec):
        raise TypeError(
            "Expected a RESET MNIST validate definition to resolve "
            f"ResetValidateSpec. Provided value: {type(spec).__name__}."
        )
    if request.teacher_weights is None:
        raise ValueError(
            f"Expected {spec.experiment_id} validation to receive "
            "--teacher-weights. Provided value: None."
        )
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("weights", request.weights),
            _input("teacher_weights", request.teacher_weights),
        ),
    )
    try:
        torch.manual_seed(spec.runtime.seed)
        device = torch.device(spec.runtime.device)
        data = build_mnist_loaders(spec.data, data_seed=spec.runtime.data_seed)
        teacher, teacher_metadata = _load_teacher(
            request.teacher_weights, device=device
        )
        teacher_sha = sha256_file(request.teacher_weights)
        stack = build_reset_student_stack(
            spec,
            device_data_path=None,
            enable_measured=False,
        )
        loaded = load_named_weights(request.weights, stack.bundle.catalog)
        _validate_checkpoint_metadata(
            loaded.metadata,
            objective=spec.settings.objective,
            teacher_sha256=teacher_sha,
            experiment_id=spec.experiment_id,
            encoding=spec.model.encoding,
            include_biases=spec.model.include_biases,
            amplification_indexing=spec.model.amplification_indexing,
        )
        posthoc = _posthoc_calibration(stack, teacher, data.validation)
        loader = data.validation if spec.settings.split == "validation" else data.test
        metrics = evaluate(
            stack,
            teacher,
            loader,
            sample_limit=spec.settings.sample_limit,
            diagnostic_gain=float(posthoc["gain"]),
        )
        store.append_metric(
            {"mode": "validate", "split": spec.settings.split, **metrics}
        )
        store.complete(
            metrics={
                "split": spec.settings.split,
                **metrics,
                "encoding": spec.model.encoding,
                "include_biases": spec.model.include_biases,
                "amplification_indexing": spec.model.amplification_indexing,
                "amplification_indices": _amplification_index_report(stack),
                "objective": spec.settings.objective,
                "initialization": "measured_reset",
                "fixed_logit_gain": 1.0,
                "posthoc_validation_calibration": posthoc,
                "teacher": {"sha256": teacher_sha, "metadata": teacher_metadata},
                "checkpoint_metadata": dict(loaded.metadata),
                "conductances": conductance_statistics(
                    stack.bundle.catalog, encoding=spec.model.encoding
                ),
            }
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["evaluate", "run_train", "run_validate", "train_epoch"]
