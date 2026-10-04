"""Application runtime for the bias-free MNIST ReLU teacher."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import torch
import torch.nn.functional as F

from experiments.artifacts import RunStore, sha256_file
from experiments.mnist_relu.config import (
    V2_EXPERIMENT_ID,
    TeacherTrainSpec,
    TeacherValidateSpec,
)
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import (
    encode_named_weights,
    load_epoch_boundary_checkpoint,
    load_named_weights,
    save_encoded_named_weights,
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


def _runtime_device_receipt(
    *,
    configured_device: str,
    resolved_device: torch.device,
) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    receipt: dict[str, Any] = {
        "configured_device": configured_device,
        "resolved_device": str(resolved_device),
        "cuda_available": cuda_available,
    }
    if resolved_device.type == "cuda":
        receipt["device_name"] = torch.cuda.get_device_name(resolved_device)
    return receipt


def _input(role: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Expected --{role.replace('_', '-')} to name an existing file. "
            f"Provided value: {str(resolved)!r}."
        )
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved)}


def _experiment_id(request: Any) -> str:
    return str(request.spec.experiment_id)


def _validate_request(request: Any, *, training: bool) -> None:
    experiment_id = _experiment_id(request)
    for name in (
        "base_weights",
        "device_data",
        "teacher_weights",
        "device_state",
        "selection_receipt",
    ):
        value = getattr(request, name, None)
        if value is not None:
            raise ValueError(
                f"Expected {experiment_id} not to use "
                f"--{name.replace('_', '-')}. Provided value: {str(value)!r}."
            )
    if training and request.resume is not None and request.weights is not None:
        raise ValueError(
            "Expected at most one of --weights or --resume. Provided value: both."
        )


def _validation_accuracy_passes(
    *,
    experiment_id: str,
    accuracy: float,
    threshold: float,
) -> bool:
    if experiment_id == V2_EXPERIMENT_ID:
        return accuracy > threshold
    return accuracy >= threshold


def _teacher_acceptance_gate(
    *,
    experiment_id: str,
    accuracy: float,
    threshold: float,
) -> dict[str, Any]:
    passed = _validation_accuracy_passes(
        experiment_id=experiment_id,
        accuracy=accuracy,
        threshold=threshold,
    )
    if experiment_id == V2_EXPERIMENT_ID:
        if not passed:
            raise RuntimeError(
                "Expected the validation-cross-entropy-selected teacher to "
                "have validation accuracy strictly above "
                f"{threshold:.12g}. Provided value: {accuracy:.12g}."
            )
        return {
            "minimum_validation_accuracy": threshold,
            "comparison_operator": ">",
            "passed": True,
        }
    # Preserve the v1 result payload exactly: its historical gate is
    # inclusive and did not record an operator field.
    return {
        "minimum_validation_accuracy": threshold,
        "passed": passed,
    }


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


def run_train(request: "TrainRequest") -> int:
    spec = request.spec
    if not isinstance(spec, TeacherTrainSpec):
        raise TypeError(
            "Expected a registered MNIST ReLU teacher train mode to resolve "
            "TeacherTrainSpec. "
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
    try:
        torch.manual_seed(spec.runtime.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(spec.runtime.seed)
        device = _device(spec.runtime.device)
        runtime_device = _runtime_device_receipt(
            configured_device=spec.runtime.device,
            resolved_device=device,
        )
        if spec.experiment_id == V2_EXPERIMENT_ID:
            store.append_metric({"mode": "runtime_device", **runtime_device})
        data = build_mnist_loaders(
            spec.data,
            data_seed=spec.runtime.data_seed,
        )
        model = BiasFreeReluTeacher(device=device, dims=spec.model.dims)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=spec.settings.learning_rate,
            weight_decay=spec.settings.weight_decay,
        )
        weights_path = store.run_dir / "checkpoints" / "weights.pt"
        resume_path = store.run_dir / "checkpoints" / "resume.pt"
        start_epoch = 0
        global_step = 0
        selected_loss: float | None = None
        selected_accuracy: float | None = None
        selected_epoch: int | None = None
        selected_weights = None
        if request.resume is not None:
            resumed = load_epoch_boundary_checkpoint(
                request.resume,
                catalog=model.catalog,
                optimizer=optimizer,
                dataloader_generators={"train": data.train_generator},
            )
            start_epoch = resumed.epoch
            global_step = resumed.global_step
            selected_weights = resumed.selected_weights
            selected_loss = float(resumed.progress_state["selected_cross_entropy"])
            selected_accuracy = float(resumed.progress_state["selected_accuracy"])
            selected_epoch = int(resumed.progress_state["selected_epoch"])
            if selected_weights is not None:
                save_encoded_named_weights(weights_path, selected_weights, catalog=model.catalog)
        elif request.weights is not None:
            load_named_weights(request.weights, model.catalog)

        initial = _evaluate(
            model,
            data.validation,
            device=device,
            maximum_batches=spec.settings.max_validation_batches,
        )
        store.append_metric({"mode": "initialization", "validation": initial})

        last_train = None
        last_validation = initial
        for epoch in range(start_epoch, spec.settings.num_epochs):
            model.train()
            loss_sum = 0.0
            correct = 0
            examples = 0
            batches = 0
            for batch_index, (batch_inputs, labels) in enumerate(
                limited(data.train, spec.settings.max_batches)
            ):
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
            global_step += batches
            if examples == 0:
                raise ValueError("Expected teacher training to process examples. Provided value: 0.")
            last_train = {
                "examples": examples,
                "cross_entropy": loss_sum / examples,
                "accuracy": correct / examples,
            }
            last_validation = _evaluate(
                model,
                data.validation,
                device=device,
                maximum_batches=spec.settings.max_validation_batches,
            )
            improved = selected_loss is None or (
                last_validation["cross_entropy"] < selected_loss
            )
            if improved:
                selected_loss = float(last_validation["cross_entropy"])
                selected_accuracy = float(last_validation["accuracy"])
                selected_epoch = epoch
                checkpoint_metadata = {
                    "experiment_id": spec.experiment_id,
                    "architecture": "bias_free_relu_" + "_".join(
                        str(item) for item in spec.model.dims
                    ),
                    "selection_metric": "validation.cross_entropy",
                    "selection_value": selected_loss,
                    "selection_accuracy": selected_accuracy,
                    "selection_epoch": selected_epoch,
                }
                if spec.experiment_id == V2_EXPERIMENT_ID:
                    checkpoint_metadata.update(
                        {
                            "logical_dims": list(spec.model.dims),
                            "bias": False,
                            "runtime_seed": spec.runtime.seed,
                            "data_seed": spec.runtime.data_seed,
                            "batch_size": spec.data.batch_size,
                            "train_shuffle": spec.data.shuffle,
                            "training_examples": last_train["examples"],
                            "validation_examples": last_validation["examples"],
                            "validation_split": (
                                "torchvision_mnist_train_stratified_per_class_"
                                "numpy_default_rng_sorted_indices"
                            ),
                            "preprocessing": (
                                "to_tensor_flatten_normalize_mean_0.1307_std_0.3"
                            ),
                            "validation_accuracy_comparison_operator": ">",
                            "minimum_validation_accuracy": (
                                spec.settings.minimum_validation_accuracy
                            ),
                        }
                    )
                selected_weights = encode_named_weights(
                    model.catalog,
                    metadata=checkpoint_metadata,
                )
                save_encoded_named_weights(weights_path, selected_weights, catalog=model.catalog)
            if selected_weights is None or selected_epoch is None:
                raise RuntimeError("Expected a selected teacher checkpoint after validation.")
            save_epoch_boundary_checkpoint(
                resume_path,
                catalog=model.catalog,
                epoch=epoch + 1,
                global_step=global_step,
                optimizer=optimizer,
                progress_state={
                    "selected_cross_entropy": selected_loss,
                    "selected_accuracy": selected_accuracy,
                    "selected_epoch": selected_epoch,
                },
                selected_weights=selected_weights,
                dataloader_generators={"train": data.train_generator},
                metadata={"experiment_id": spec.experiment_id},
            )
            if (epoch + 1) % spec.settings.log_every == 0 or epoch + 1 == spec.settings.num_epochs:
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
                        "selected_cross_entropy": selected_loss,
                    }
                )

        if selected_weights is None or selected_epoch is None or selected_accuracy is None:
            raise RuntimeError("Expected training or resume to provide selected teacher weights.")
        acceptance_gate = _teacher_acceptance_gate(
            experiment_id=spec.experiment_id,
            accuracy=selected_accuracy,
            threshold=spec.settings.minimum_validation_accuracy,
        )
        artifacts = (
            store.artifact_record(weights_path, kind="selected_named_weights"),
            store.artifact_record(resume_path, kind="epoch_boundary_resume"),
        )
        store.complete(
            metrics={
                "initial_validation": initial,
                "last_train": last_train,
                "last_validation": last_validation,
                "selected": {
                    "epoch": selected_epoch,
                    "cross_entropy": selected_loss,
                    "accuracy": selected_accuracy,
                },
                "acceptance_gate": acceptance_gate,
                **(
                    {"runtime_device": runtime_device}
                    if spec.experiment_id == V2_EXPERIMENT_ID
                    else {}
                ),
            },
            artifacts=artifacts,
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


def run_validate(request: "ValidateRequest") -> int:
    spec = request.spec
    if not isinstance(spec, TeacherValidateSpec):
        raise TypeError(
            "Expected a registered MNIST ReLU teacher validate mode to resolve "
            "TeacherValidateSpec. "
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
        runtime_device = _runtime_device_receipt(
            configured_device=spec.runtime.device,
            resolved_device=device,
        )
        if spec.experiment_id == V2_EXPERIMENT_ID:
            store.append_metric({"mode": "runtime_device", **runtime_device})
        data = build_mnist_loaders(spec.data, data_seed=spec.runtime.data_seed)
        model = BiasFreeReluTeacher(device=device, dims=spec.model.dims)
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
                **(
                    {"runtime_device": runtime_device}
                    if spec.experiment_id == V2_EXPERIMENT_ID
                    else {}
                ),
            }
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_train", "run_validate"]
