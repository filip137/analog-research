"""Application runtime for the bias-free MNIST ReLU teacher."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import torch
import torch.nn.functional as F

from experiments.artifacts import RunStore, sha256_file
from experiments.lifecycle import BestCheckpoint, EpochResult, run_epochs, run_phase
from experiments.lifecycle.train_phase import lower
from experiments.mnist_relu.config import TeacherTrainSpec, TeacherValidateSpec
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import (
    encode_named_weights,
    load_epoch_boundary_checkpoint,
    load_named_weights,
    save_epoch_boundary_checkpoint,
)

if TYPE_CHECKING:
    from ebl.cli import TrainRequest, ValidateRequest


_ROOT = Path(__file__).resolve().parents[2]


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Expected config.runtime.device='cuda' to have an available CUDA "
            "device. Provided value: torch.cuda.is_available() is false."
        )
    return torch.device(name)


def _input(role: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Expected --{role.replace('_', '-')} to name an existing file. "
            f"Provided value: {str(resolved)!r}."
        )
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved)}


def _validate_request(request: Any, *, training: bool) -> None:
    if getattr(request, "base_weights", None) is not None:
        raise ValueError(
            "Expected mnist_relu.v1 not to use --base-weights. Provided "
            f"value: {str(request.base_weights)!r}."
        )
    if getattr(request, "device_data", None) is not None:
        raise ValueError(
            "Expected mnist_relu.v1 not to use --device-data. Provided "
            f"value: {str(request.device_data)!r}."
        )
    if getattr(request, "teacher_weights", None) is not None:
        raise ValueError(
            "Expected mnist_relu.v1 not to use --teacher-weights. Provided "
            f"value: {str(request.teacher_weights)!r}."
        )
    if training and request.resume is not None and request.weights is not None:
        raise ValueError(
            "Expected at most one of --weights or --resume. Provided value: both."
        )


def _evaluate(
    model: BiasFreeReluTeacher,
    loader: Iterable,
    *,
    device: torch.device,
    maximum_batches: int | None = None,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    loss_sum = 0.0
    correct = 0
    examples = 0
    with torch.no_grad():
        for inputs, labels in limited(loader, maximum_batches):
            if sample_limit is not None:
                remaining = sample_limit - examples
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
                labels = labels[:remaining]
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            logits = model.logits(inputs)
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            examples += int(labels.shape[0])
    if examples == 0:
        raise ValueError("Expected evaluation to process at least one example. Provided value: 0.")
    return {
        "examples": examples,
        "cross_entropy": loss_sum / examples,
        "accuracy": correct / examples,
        "error_fraction": 1.0 - correct / examples,
    }


def _train_epoch(
    model: BiasFreeReluTeacher,
    optimizer: torch.optim.Optimizer,
    loader: Iterable,
    *,
    device: torch.device,
    maximum_batches: int | None,
) -> tuple[dict[str, Any], int]:
    model.train()
    loss_sum = 0.0
    correct = 0
    examples = 0
    batches = 0
    for batch_index, (batch_inputs, labels) in enumerate(limited(loader, maximum_batches)):
        batch_inputs = batch_inputs.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(batch_inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * labels.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        examples += int(labels.shape[0])
        batches = batch_index + 1
    if examples == 0:
        raise ValueError("Expected teacher training to process examples. Provided value: 0.")
    return (
        {
            "examples": examples,
            "cross_entropy": loss_sum / examples,
            "accuracy": correct / examples,
        },
        batches,
    )


def run_train(request: "TrainRequest") -> int:
    spec = request.spec
    if not isinstance(spec, TeacherTrainSpec):
        raise TypeError(
            "Expected mnist_relu.v1 train to resolve TeacherTrainSpec. "
            f"Provided value: {type(spec).__name__}."
        )
    _validate_request(request, training=True)
    inputs = []
    for role in ("weights", "resume"):
        value = getattr(request, role)
        if value is not None:
            inputs.append(_input(role, value))
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=inputs,
    )

    def phase():
        torch.manual_seed(spec.runtime.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(spec.runtime.seed)
        device = _device(spec.runtime.device)
        data = build_mnist_loaders(
            spec.data,
            data_seed=spec.runtime.data_seed,
        )
        model = BiasFreeReluTeacher(device=device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=spec.settings.learning_rate,
            weight_decay=spec.settings.weight_decay,
        )
        weights_path = store.run_dir / "checkpoints" / "weights.pt"
        resume_path = store.run_dir / "checkpoints" / "resume.pt"
        best = BestCheckpoint(weights_path, model.catalog, better=lower("cross_entropy"))
        start_epoch = 0
        global_step = 0
        if request.resume is not None:
            resumed = load_epoch_boundary_checkpoint(
                request.resume,
                catalog=model.catalog,
                optimizer=optimizer,
                dataloader_generators={"train": data.train_generator},
            )
            start_epoch = resumed.epoch
            global_step = resumed.global_step
            best.restore(
                epoch=int(resumed.progress_state["selected_epoch"]),
                selection={
                    "cross_entropy": float(
                        resumed.progress_state["selected_cross_entropy"]
                    ),
                    "accuracy": float(resumed.progress_state["selected_accuracy"]),
                },
                payload=resumed.selected_weights,
            )
        elif request.weights is not None:
            load_named_weights(request.weights, model.catalog)

        def validate(_epoch: int) -> dict[str, Any]:
            return _evaluate(
                model,
                data.validation,
                device=device,
                maximum_batches=spec.settings.max_validation_batches,
            )

        initial = validate(-1)
        store.append_metric({"mode": "initialization", "validation": initial})

        def train_epoch(_epoch: int, step: int) -> tuple[dict[str, Any], int]:
            trained, batches = _train_epoch(
                model,
                optimizer,
                data.train,
                device=device,
                maximum_batches=spec.settings.max_batches,
            )
            return trained, step + batches

        def payload(epoch: int, validation: dict[str, Any]) -> dict[str, Any]:
            return encode_named_weights(
                model.catalog,
                metadata={
                    "experiment_id": spec.experiment_id,
                    "architecture": "bias_free_relu_784_50_10",
                    "selection_metric": "validation.cross_entropy",
                    "selection_value": float(validation["cross_entropy"]),
                    "selection_accuracy": float(validation["accuracy"]),
                    "selection_epoch": epoch,
                },
            )

        def save_resume(completed: int, step: int, _last: EpochResult | None) -> None:
            save_epoch_boundary_checkpoint(
                resume_path,
                catalog=model.catalog,
                epoch=completed,
                global_step=step,
                optimizer=optimizer,
                progress_state={
                    "selected_cross_entropy": float(best.selection["cross_entropy"]),
                    "selected_accuracy": float(best.selection["accuracy"]),
                    "selected_epoch": best.epoch,
                },
                selected_weights=best.payload,
                dataloader_generators={"train": data.train_generator},
                metadata={"experiment_id": spec.experiment_id},
            )

        def record(result: EpochResult) -> dict[str, Any]:
            return {
                "mode": "train",
                "epoch": result.epoch,
                "completed_epochs": result.completed_epochs,
                "global_step": result.global_step,
                "train": result.train,
                "validation": result.validation,
                "selected": result.improved,
                "selected_epoch": best.epoch,
                "selected_cross_entropy": float(best.selection["cross_entropy"]),
            }

        last, _completed, _step = run_epochs(
            store=store,
            start_epoch=start_epoch,
            num_epochs=spec.settings.num_epochs,
            global_step=global_step,
            log_every=spec.settings.log_every,
            train_epoch=train_epoch,
            validate=validate,
            best=best,
            selection=lambda validation: validation,
            payload=payload,
            save_resume=save_resume,
            resume_path=resume_path,
            record=record,
            missing_selection="Expected a selected teacher checkpoint after validation.",
        )
        best.require("Expected training or resume to provide selected teacher weights.")
        selected_accuracy = float(best.selection["accuracy"])
        return (
            {
                "initial_validation": initial,
                "last_train": None if last is None else last.train,
                "last_validation": initial if last is None else last.validation,
                "selected": {
                    "epoch": best.epoch,
                    "cross_entropy": float(best.selection["cross_entropy"]),
                    "accuracy": selected_accuracy,
                },
                "acceptance_gate": {
                    "minimum_validation_accuracy": spec.settings.minimum_validation_accuracy,
                    "passed": selected_accuracy >= spec.settings.minimum_validation_accuracy,
                },
            },
            (
                store.artifact_record(weights_path, kind="selected_named_weights"),
                store.artifact_record(resume_path, kind="epoch_boundary_resume"),
            ),
        )

    run_phase(store, phase)
    return 0


def run_validate(request: "ValidateRequest") -> int:
    spec = request.spec
    if not isinstance(spec, TeacherValidateSpec):
        raise TypeError(
            "Expected mnist_relu.v1 validate to resolve TeacherValidateSpec. "
            f"Provided value: {type(spec).__name__}."
        )
    _validate_request(request, training=False)
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(_input("weights", request.weights),),
    )
    try:
        torch.manual_seed(spec.runtime.seed)
        device = _device(spec.runtime.device)
        data = build_mnist_loaders(spec.data, data_seed=spec.runtime.data_seed)
        model = BiasFreeReluTeacher(device=device)
        loaded = load_named_weights(request.weights, model.catalog)
        loader = data.validation if spec.settings.split == "validation" else data.test
        metrics = _evaluate(
            model,
            loader,
            device=device,
            sample_limit=spec.settings.sample_limit,
        )
        store.append_metric({"mode": "validate", "split": spec.settings.split, **metrics})
        store.complete(
            metrics={
                "split": spec.settings.split,
                **metrics,
                "checkpoint_metadata": loaded.metadata,
            }
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_train", "run_validate"]
