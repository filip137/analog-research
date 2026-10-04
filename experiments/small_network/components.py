"""Numerical composition for the versioned small-DRN experiment.

This module is the only place where ``small_drn.v1`` configuration objects
are translated into legacy datasets, energy functions, minimizers, gradient
estimators, and optimizers.  Persistence and run lifecycle policy live in
``runtime.py``; the generic training loop remains configuration-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from experiments.small_network.config import CommonSettings, TrainSpec
from labs.custom_minimizer import (
    CustomAnderssonMinimizer,
    CustomQuadraticMinimizer,
    MinimizerSettings,
)
from labs.datasets import DigitsDataset, MnistDataset, MoonsDataset, YinYangDataset
from model.function.cost import SquaredError, SquaredErrorPairedOutputs
from model.function.network import Network
from model.resistive.builders import ModelBundle, build_deep_resistive_energy
from model.resistive.digital_low_rank import DigitalLowRankReadout
from model.resistive.device_config import parse_device_programming_config
from model.variable.parameter import Bias
from training.add_normal import AddNormalConfig, build_add_normal_modifier
from training.adam import AdamOptimizer
from training.core.engine import EvaluationComponents, ExperimentComponents
from training.core.modifier import ParameterModifier
from training.core.optimizers import build_optimizer
from training.core.sgd import (
    AugmentedFunction,
    Backprop,
    DirectReadoutGradient,
    EquilibriumProp,
)
from training.ibm_om_fp32_bounds import IbmOmFp32BoundsOptimizer
from training.measured_trace import (
    MeasuredCohortAOptimizer,
    MeasuredCohortBOptimizer,
    MeasuredCohortBLoRAOptimizer,
)
from training.program_verify import ProgramVerifyOptimizer
from training.tiki_taka import parse_update_pipeline


_CLASS_COUNTS = {"moons": 2, "yinyang": 3, "digits": 10, "mnist": 10}
_DTYPES = {"float32": torch.float32, "float64": torch.float64}


class _FlattenedImageDataset(Dataset):
    """Flatten torchvision images while preserving labels and optional indices."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        return (sample[0].reshape(-1), *sample[1:])


class TypedNetwork:
    """Preserve the configured dtype when the legacy network resets layers."""

    def __init__(self, network: Network, dtype: torch.dtype) -> None:
        self._network = network
        self._dtype = dtype

    def set_input(self, inputs: torch.Tensor, reset: bool = False) -> None:
        self._network.set_input(inputs.to(dtype=self._dtype), reset=reset)
        for layer in self._network.layers():
            if layer.state.dtype != self._dtype:
                layer.state = layer.state.to(dtype=self._dtype)

    def layers(self):
        return self._network.layers()

    def free_layers(self):
        return self._network.free_layers()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._network, name)


class LayerStateCheckpoint:
    """Tensor-bearing continuation state for legacy equilibrium carry-over.

    Training intentionally uses ``reset_input=False``.  The next minibatch can
    therefore start from the states left by the preceding clean evaluation.
    Layer names are process-global in the legacy implementation, so this
    codec uses stable composition-order keys rather than ``Layer_N`` names.
    """

    _SCHEMA = "small_drn.layer-state"
    _SCHEMA_VERSION = 1

    def __init__(self, layers: Iterable[Any]) -> None:
        self._layers = tuple(layers)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self._SCHEMA,
            "schema_version": self._SCHEMA_VERSION,
            "layers": {
                f"layer.{index}": layer.state.detach().cpu().clone()
                for index, layer in enumerate(self._layers)
            },
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise ValueError(
                "Expected small-DRN runtime layer state to be an object. "
                f"Provided value: {value!r}."
            )
        if (
            value.get("schema") != self._SCHEMA
            or value.get("schema_version") != self._SCHEMA_VERSION
        ):
            raise ValueError(
                "Expected small-DRN runtime layer state schema "
                f"{self._SCHEMA!r} version {self._SCHEMA_VERSION}. "
                "Provided value: "
                f"schema={value.get('schema')!r}, "
                f"schema_version={value.get('schema_version')!r}."
            )
        raw_layers = value.get("layers")
        expected_keys = tuple(
            f"layer.{index}" for index in range(len(self._layers))
        )
        if not isinstance(raw_layers, Mapping) or set(raw_layers) != set(
            expected_keys
        ):
            provided = (
                tuple(raw_layers)
                if isinstance(raw_layers, Mapping)
                else raw_layers
            )
            raise ValueError(
                "Expected small-DRN runtime layer state to contain exactly "
                f"{expected_keys!r}. Provided value: {provided!r}."
            )

        staged: list[torch.Tensor] = []
        for key, layer in zip(expected_keys, self._layers):
            tensor = raw_layers[key]
            expected_shape = tuple(layer.shape)
            valid_shape = (
                isinstance(tensor, torch.Tensor)
                and tensor.ndim == len(expected_shape) + 1
                and tensor.shape[0] > 0
                and tuple(tensor.shape[1:]) == expected_shape
            )
            if not valid_shape:
                raise ValueError(
                    f"Expected runtime state {key!r} to be a batched tensor "
                    f"with trailing shape {expected_shape!r}. "
                    f"Provided value: {tensor!r}."
                )
            if tensor.dtype != layer.state.dtype:
                raise ValueError(
                    f"Expected runtime state {key!r} dtype to be "
                    f"{layer.state.dtype}. Provided value: {tensor.dtype}."
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"Expected runtime state {key!r} to contain only finite "
                    f"values. Provided value: {tensor!r}."
                )
            staged.append(
                tensor.detach().to(device=layer.state.device).clone()
            )

        snapshots = tuple(layer.state for layer in self._layers)
        try:
            for layer, tensor in zip(self._layers, staged):
                layer.state = tensor
        except Exception:
            for layer, snapshot in zip(self._layers, snapshots):
                layer.state = snapshot
            raise


@dataclass(frozen=True)
class DataBundle:
    train_loader: Iterable[Any]
    held_out_loader: Iterable[Any]
    test_loader: Iterable[Any]
    dataloader_generators: Mapping[str, torch.Generator]


@dataclass(frozen=True)
class ModelStack:
    bundle: ModelBundle
    network: TypedNetwork
    cost_fn: Any
    free_layers: tuple[Any, ...]
    device: torch.device
    dtype: torch.dtype
    logical_input_dim: int
    num_classes: int
    effective_non_linearity: str


@dataclass(frozen=True)
class EvaluationRuntime:
    stack: ModelStack
    components: EvaluationComponents


@dataclass(frozen=True)
class ValidationRuntime(EvaluationRuntime):
    data: DataBundle


@dataclass(frozen=True)
class TrainRuntime:
    stack: ModelStack
    data: DataBundle
    training_components: ExperimentComponents
    evaluation_components: EvaluationComponents
    optimizer: Any
    modifier: ParameterModifier | None
    noisy_evaluation: bool
    runtime_state: LayerStateCheckpoint
    resume_capability: str

    @property
    def modifier_resolved_seed(self) -> int | None:
        seed = getattr(self.modifier, "resolved_seed", None)
        return None if seed is None else int(seed)


def seed_runtime(seed: int | None) -> None:
    """Match the legacy training entry point's process RNG initialization."""

    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model_stack(common: CommonSettings) -> ModelStack:
    """Build the legacy dense resistive model behind stable parameter names."""

    adapter = common.model.adapter
    device = _resolve_device(common.runtime.device)
    dtype = _DTYPES[common.runtime.dtype]
    dims = common.model.dims
    if dims[0] % 2:
        raise ValueError(
            "Expected config.model.dims[0] to be even because the physical "
            "input layer stores positive and negative copies. "
            f"Provided value: {dims[0]!r}."
        )
    logical_input_dim = dims[0] // 2
    _validate_dataset_shape(
        common.data.dataset,
        logical_input_dim=logical_input_dim,
        output_width=dims[-1],
    )

    non_linearity = common.model.non_linearity
    # ``double_diode`` was the legacy spelling for the hard-sigmoid device.
    effective_non_linearity = (
        "hard_sigmoid"
        if non_linearity.type == "double_diode"
        else non_linearity.type
    )
    bundle = build_deep_resistive_energy(
        layer_shapes=[(dimension,) for dimension in dims],
        weight_gains=list(common.model.weight_gains),
        input_gain=common.model.input_gain,
        non_linearity=effective_non_linearity,
        exponential_diode_param=dict(
            non_linearity.exponential_diode_param
        ),
        quadratic_diode_param=dict(non_linearity.quadratic_diode_param),
        hard_sigmoid_param=dict(non_linearity.hard_sigmoid_param),
        voltage_amp=common.model.voltage_amp,
        current_amp=common.model.current_amp,
        weight_min=common.model.weight_min,
        weight_max=common.model.weight_max,
        weight_init_mode=common.model.weight_init_mode,
        include_biases=common.model.include_biases,
        passive_low_rank_adapter=(
            dict(adapter.parameters)
            if adapter.type == "passive_low_rank"
            else None
        ),
        digital_low_rank_adapter=(
            dict(adapter.parameters)
            if adapter.type == "digital_low_rank"
            else None
        ),
        passive_layerwise_low_rank_adapter=(
            dict(adapter.parameters)
            if adapter.type == "passive_layerwise_low_rank"
            else None
        ),
    )
    energy = bundle.energy
    energy.set_device(device)
    with torch.no_grad():
        for binding in bundle.catalog.all:
            binding.parameter.state = binding.state.to(
                device=device,
                dtype=dtype,
            )
        for layer in energy.layers():
            layer.state = layer.state.to(device=device, dtype=dtype)

    network = TypedNetwork(Network(energy), dtype)
    output_layer = energy.layers()[-1]
    num_classes = _CLASS_COUNTS[common.data.dataset]
    if adapter.type == "digital_low_rank":
        input_factor, output_factor = energy.adapter_params()
        cost_fn = DigitalLowRankReadout(
            energy.layers()[0],
            output_layer,
            input_factor=input_factor,
            output_factor=output_factor,
            logical_input_dim=logical_input_dim,
            num_classes=num_classes,
            input_gain=common.model.input_gain,
            alpha=float(adapter.parameters["alpha"]),
        )
    elif dims[-1] == num_classes:
        cost_fn = SquaredError(output_layer)
    else:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes)

    return ModelStack(
        bundle=bundle,
        network=network,
        cost_fn=cost_fn,
        free_layers=tuple(network.free_layers()),
        device=device,
        dtype=dtype,
        logical_input_dim=logical_input_dim,
        num_classes=num_classes,
        effective_non_linearity=effective_non_linearity,
    )


def build_data(common: CommonSettings) -> DataBundle:
    """Build legacy datasets and an explicitly seeded training sampler."""

    device = _resolve_device(common.runtime.device)
    logical_input_dim = common.model.dims[0] // 2
    data_seed = (
        common.runtime.data_seed
        if common.runtime.data_seed is not None
        else common.runtime.seed
    )
    if common.data.dataset in {"moons", "yinyang"}:
        if common.data.num_points is None:
            raise ValueError(
                "Expected config.data.num_points to be an integer for "
                f"dataset {common.data.dataset!r}. Provided value: None."
            )

    if common.data.dataset == "moons":
        dataset = MoonsDataset(
            name="moons",
            batch_size=common.data.batch_size,
            device=device,
            num_samples=common.data.num_points,
            input_dim=logical_input_dim,
        )
    elif common.data.dataset == "yinyang":
        dataset = YinYangDataset(
            name="yinyang",
            batch_size=common.data.batch_size,
            device=device,
            num_samples=common.data.num_points,
            seed=42 if data_seed is None else data_seed,
        )
    elif common.data.dataset == "digits":
        dataset = DigitsDataset(
            name="digits",
            batch_size=common.data.batch_size,
            device=device,
            num_samples=common.data.num_points,
            seed=0 if data_seed is None else data_seed,
        )
    elif common.data.dataset == "mnist":
        mnist_root = Path(
            os.environ.get(
                "EBL_MNIST_ROOT",
                str(Path.home() / "datasets" / "mnist"),
            )
        ).expanduser()
        dataset = MnistDataset(
            name="mnist",
            batch_size=common.data.batch_size,
            device=torch.device("cpu"),
            root=str(mnist_root),
            train=True,
            download=False,
            normalize=True,
            normalize_mean=0.1307,
            normalize_std=0.3,
        )
    else:  # pragma: no cover - rejected by the pure schema
        raise ValueError(
            "Expected config.data.dataset to be 'moons', 'yinyang', "
            f"'digits', or 'mnist'. Provided value: {common.data.dataset!r}."
        )

    try:
        legacy_train, held_out = dataset.build()
    except RuntimeError as exc:
        if common.data.dataset != "mnist":
            raise
        raise RuntimeError(
            "Expected an existing torchvision MNIST dataset root containing "
            f"MNIST/raw or MNIST/processed. Provided root: {mnist_root}"
        ) from exc

    train_dataset = legacy_train.dataset
    test_loader = held_out
    if common.data.dataset == "mnist":
        raw_train_dataset = train_dataset
        flattened_train = _FlattenedImageDataset(raw_train_dataset)
        test_loader = DataLoader(
            _FlattenedImageDataset(held_out.dataset),
            batch_size=common.data.batch_size,
            shuffle=False,
        )
        if common.data.validation_points is None:
            train_dataset = flattened_train
            held_out = test_loader
        else:
            train_indices, validation_indices = _stratified_split_indices(
                raw_train_dataset,
                validation_points=common.data.validation_points,
                seed=0 if data_seed is None else data_seed,
            )
            train_dataset = Subset(flattened_train, train_indices)
            held_out = DataLoader(
                Subset(flattened_train, validation_indices),
                batch_size=common.data.batch_size,
                shuffle=False,
            )
        if common.data.num_points is not None:
            if common.data.num_points > len(train_dataset):
                raise ValueError(
                    "Expected config.data.num_points to be no larger than the "
                    "available MNIST training split after validation holdout. "
                    f"Provided value: {common.data.num_points}."
                )
            subset_generator = torch.Generator()
            subset_generator.manual_seed(0 if data_seed is None else data_seed)
            indices = torch.randperm(
                len(train_dataset),
                generator=subset_generator,
            )[: common.data.num_points].tolist()
            train_dataset = Subset(train_dataset, indices)

    generator = torch.Generator()
    if data_seed is None:
        generator.seed()
    else:
        generator.manual_seed(data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=common.data.batch_size,
        shuffle=common.data.shuffle,
        generator=generator,
    )
    return DataBundle(
        train_loader=train_loader,
        held_out_loader=held_out,
        test_loader=test_loader,
        dataloader_generators={"train": generator},
    )


def _stratified_split_indices(
    dataset: Dataset,
    *,
    validation_points: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return deterministic class-stratified train/validation indices."""

    raw_targets = getattr(dataset, "targets", None)
    if raw_targets is None and hasattr(dataset, "tensors"):
        tensors = getattr(dataset, "tensors")
        if isinstance(tensors, (tuple, list)) and len(tensors) >= 2:
            raw_targets = tensors[1]
    if raw_targets is None:
        raw_targets = [dataset[index][1] for index in range(len(dataset))]
    targets = np.asarray(
        raw_targets.detach().cpu().numpy()
        if isinstance(raw_targets, torch.Tensor)
        else raw_targets,
        dtype=np.int64,
    ).reshape(-1)
    if targets.size != len(dataset):
        raise ValueError(
            "Expected MNIST target count to match the training dataset. "
            f"Provided value: targets={targets.size}, dataset={len(dataset)}."
        )
    if validation_points >= targets.size:
        raise ValueError(
            "Expected config.data.validation_points to be smaller than the "
            f"MNIST training set ({targets.size}). Provided value: "
            f"{validation_points}."
        )
    labels = sorted(int(value) for value in np.unique(targets))
    if not labels:
        raise ValueError(
            "Expected MNIST training targets to contain at least one class. "
            "Provided value: empty."
        )
    base, remainder = divmod(validation_points, len(labels))
    generator = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for class_index, label in enumerate(labels):
        indices = np.flatnonzero(targets == label)
        take = base + (1 if class_index < remainder else 0)
        if take >= indices.size:
            raise ValueError(
                "Expected every MNIST class to retain at least one training "
                "example after the stratified validation split. Provided "
                f"value: class={label}, available={indices.size}, holdout={take}."
            )
        shuffled = generator.permutation(indices)
        validation.extend(int(value) for value in shuffled[:take])
        train.extend(int(value) for value in shuffled[take:])
    # Sort the fixed subsets; only the explicit training DataLoader controls
    # epoch order, so split membership and shuffle order stay independent.
    return sorted(train), sorted(validation)


def build_evaluation_runtime(common: CommonSettings) -> EvaluationRuntime:
    stack = build_model_stack(common)
    minimizer = _build_minimizer(
        common,
        stack,
        function=stack.bundle.energy,
        iterations=common.solver.inference_iterations,
    )
    _reset_minimizer_statistics(minimizer)
    return EvaluationRuntime(
        stack=stack,
        components=EvaluationComponents(
            network=stack.network,
            cost_fn=stack.cost_fn,
            energy_minimizer=minimizer,
        ),
    )


def build_validation_runtime(common: CommonSettings) -> ValidationRuntime:
    evaluation = build_evaluation_runtime(common)
    return ValidationRuntime(
        stack=evaluation.stack,
        components=evaluation.components,
        data=build_data(common),
    )


def build_train_runtime(
    spec: TrainSpec,
    *,
    device_data_path: Path | None = None,
) -> TrainRuntime:
    """Compose the authoritative engine with legacy numerical primitives."""

    stack = build_model_stack(spec.common)
    data = build_data(spec.common)
    energy = stack.bundle.energy
    cost_fn = stack.cost_fn
    modifier: ParameterModifier | None = None
    noisy_evaluation = False
    modifier_settings = spec.settings.weight_modifier
    if modifier_settings.type == "add_normal":
        modifier_config = AddNormalConfig(
            std_dev=float(modifier_settings.parameters["std_dev"]),
            seed=modifier_settings.parameters["seed"],
            noisy_evaluation=bool(
                modifier_settings.parameters["noisy_evaluation"]
            ),
            scale_mode=str(modifier_settings.parameters["scale_mode"]),
        )
        modifier = build_add_normal_modifier(
            stack.bundle.catalog.trainable_parameters,
            modifier_config,
            run_seed=spec.common.runtime.seed,
        )
        noisy_evaluation = (
            modifier is not None and modifier_config.noisy_evaluation
        )
    elif modifier_settings.type != "none":
        raise NotImplementedError(
            "Expected the hardware-aware branch to use weight modifier "
            "'none' or 'add_normal'. Provided value: "
            f"{modifier_settings.type!r}."
        )

    inference_minimizer = _build_minimizer(
        spec.common,
        stack,
        function=energy,
        iterations=spec.common.solver.inference_iterations,
    )
    if spec.settings.algorithm == "ep":
        training_function = AugmentedFunction(energy, cost_fn)
        training_minimizer = _build_minimizer(
            spec.common,
            stack,
            function=training_function,
            iterations=spec.common.solver.training_iterations,
        )
        differentiator = EquilibriumProp(
            energy.params(),
            list(stack.free_layers),
            training_function,
            cost_fn,
            training_minimizer,
            variant="centered",
            nudging=spec.settings.nudging,
        )
    elif spec.settings.algorithm == "backprop":
        training_minimizer = _build_minimizer(
            spec.common,
            stack,
            function=energy,
            iterations=spec.common.solver.training_iterations,
        )
        differentiator = Backprop(
            energy.params(),
            list(stack.free_layers),
            cost_fn,
            training_minimizer,
        )
    elif spec.settings.algorithm == "digital":
        if spec.common.model.adapter.type != "digital_low_rank":
            raise ValueError(
                "Expected algorithm='digital' only with "
                "model.adapter.type='digital_low_rank'. Provided value: "
                f"adapter={spec.common.model.adapter.type!r}."
            )
        training_minimizer = _build_minimizer(
            spec.common,
            stack,
            function=energy,
            iterations=spec.common.solver.training_iterations,
        )
        differentiator = DirectReadoutGradient(cost_fn)
    else:  # pragma: no cover - rejected by the pure schema
        raise ValueError(
            "Expected config.modes.train.algorithm to be 'ep', 'backprop', "
            "or 'digital'. Provided value: "
            f"{spec.settings.algorithm!r}."
        )

    parameters = tuple(energy.params()) + tuple(cost_fn.params())
    learning_rates = _parameter_learning_rates(
        parameters,
        weight_rates=spec.settings.learning_rates,
        bias_rates=spec.settings.bias_learning_rates,
    )
    update_pipeline = {
        "type": spec.settings.update_backend.type,
        **dict(spec.settings.update_backend.parameters),
    }
    program_verify_config = None
    measured_parameters = None
    om_bounds_parameters = None
    if spec.settings.update_backend.type == "program_verify":
        program_verify_config = parse_device_programming_config(
            spec.settings.update_backend.parameters["device"],
            path="config.modes.train.update_backend.parameters.device",
        )
        parsed_pipeline = None
    elif spec.settings.update_backend.type in {
        "measured_cohort_a",
        "measured_cohort_b",
        "measured_cohort_b_lora",
    }:
        if device_data_path is None:
            raise ValueError(
                "Expected --device-data for update backend "
                f"{spec.settings.update_backend.type!r}. Provided value: None."
            )
        measured_parameters = spec.settings.update_backend.parameters
        parsed_pipeline = None
    elif spec.settings.update_backend.type == "direct_adam":
        parsed_pipeline = None
    elif spec.settings.update_backend.type == "ibm_om_fp32_bounds":
        if device_data_path is None:
            raise ValueError(
                "Expected --device-data for update backend "
                "'ibm_om_fp32_bounds'. Provided value: None."
            )
        om_bounds_parameters = spec.settings.update_backend.parameters
        parsed_pipeline = None
    else:
        parsed_pipeline = parse_update_pipeline(update_pipeline)
    if (
        spec.common.model.adapter.type
        in {"passive_low_rank", "passive_layerwise_low_rank"}
        and parsed_pipeline is not None
        and parsed_pipeline.aihwkit_preset is not None
    ):
        raise ValueError(
            "Expected low-rank energy-adapter training to use direct updates "
            "or the ideal-tensor Tiki-Taka backend. Provided value: "
            f"aihwkit_preset={parsed_pipeline.aihwkit_preset!r}."
        )
    if spec.settings.update_backend.type == "direct_adam":
        adam = spec.settings.update_backend.parameters
        optimizer = AdamOptimizer(
            energy,
            cost_fn,
            learning_rates,
            betas=(float(adam["beta1"]), float(adam["beta2"])),
            eps=float(adam["epsilon"]),
            weight_decay=float(adam["weight_decay"]),
            amsgrad=bool(adam["amsgrad"]),
        )
    elif om_bounds_parameters is not None and (
        om_bounds_parameters["optimizer"]["type"] == "adam"
    ):
        adam = om_bounds_parameters["optimizer"]["parameters"]
        optimizer = AdamOptimizer(
            energy,
            cost_fn,
            learning_rates,
            betas=(float(adam["beta1"]), float(adam["beta2"])),
            eps=float(adam["epsilon"]),
            weight_decay=float(adam["weight_decay"]),
            amsgrad=bool(adam["amsgrad"]),
        )
    else:
        sgd_parameters = (
            om_bounds_parameters["optimizer"]["parameters"]
            if om_bounds_parameters is not None
            else {}
        )
        optimizer = build_optimizer(
            energy,
            cost_fn,
            learning_rates,
            update_pipeline=parsed_pipeline,
            momentum=float(sgd_parameters.get("momentum", 0.0)),
            weight_decay=float(sgd_parameters.get("weight_decay", 0.0)),
        )
    if program_verify_config is not None:
        optimizer = ProgramVerifyOptimizer(
            optimizer,
            stack.bundle.catalog,
            program_verify_config,
        )
    elif measured_parameters is not None:
        measured_optimizer = {
            "measured_cohort_a": MeasuredCohortAOptimizer,
            "measured_cohort_b": MeasuredCohortBOptimizer,
            "measured_cohort_b_lora": MeasuredCohortBLoRAOptimizer,
        }[spec.settings.update_backend.type]
        optimizer = measured_optimizer(
            optimizer,
            stack.bundle.catalog,
            measured_parameters,
            device_data_path,
        )
    elif om_bounds_parameters is not None:
        optimizer = IbmOmFp32BoundsOptimizer(
            optimizer,
            stack.bundle.catalog,
            om_bounds_parameters,
            device_data_path,
        )
    resume_capability = (
        "stateful_nondeterministic"
        if parsed_pipeline is not None
        and parsed_pipeline.aihwkit_preset is not None
        else "exact"
    )
    _reset_minimizer_statistics(inference_minimizer)

    evaluation_components = EvaluationComponents(
        network=stack.network,
        cost_fn=cost_fn,
        energy_minimizer=inference_minimizer,
    )
    return TrainRuntime(
        stack=stack,
        data=data,
        training_components=ExperimentComponents(
            network=stack.network,
            cost_fn=cost_fn,
            energy_minimizer=inference_minimizer,
            parameters=parameters,
            differentiator=differentiator,
            optimizer=optimizer,
        ),
        evaluation_components=evaluation_components,
        optimizer=optimizer,
        modifier=modifier,
        noisy_evaluation=noisy_evaluation,
        runtime_state=LayerStateCheckpoint(stack.network.layers()),
        resume_capability=resume_capability,
    )


def configured_resume_capability(spec: TrainSpec) -> str:
    """Resolve capability before importing optional numerical backends."""

    if spec.settings.update_backend.type != "tiki_taka":
        return "exact"
    preset = spec.settings.update_backend.parameters.get("aihwkit_preset")
    return "stateful_nondeterministic" if preset is not None else "exact"


def _build_minimizer(
    common: CommonSettings,
    stack: ModelStack,
    *,
    function: Any,
    iterations: int,
) -> Any:
    solver = common.solver
    if solver.random_initialization:
        raise NotImplementedError(
            "Expected config.solver.random_initialization to be false because "
            "the legacy small-network solver has no reproducible random-state "
            f"initialization contract. Provided value: {solver.random_initialization!r}."
        )
    settings = MinimizerSettings(
        rel_tol=solver.tolerances.relative,
        vn_tol=solver.tolerances.voltage,
        use_polish=solver.polish.enabled,
        max_newton_iters=solver.polish.max_newton_iterations,
        z_thresh=solver.polish.z_threshold,
        exp_clip=solver.polish.exponential_clip,
        dynamic_polish=solver.polish.dynamic,
        overrelaxation_reject_steps=solver.overrelaxation.reject_steps,
        overrelaxation_reject_max_tries=(
            solver.overrelaxation.reject_max_tries
        ),
        overrelaxation_reject_shrink=solver.overrelaxation.reject_shrink,
        overrelaxation_reject_eps=solver.overrelaxation.reject_epsilon,
        experimental_exponential_newton_tol_progressive=(
            solver.experimental_exponential.progressive_tolerance
        ),
        experimental_exponential_newton_tol_start=(
            solver.experimental_exponential.tolerance_start
        ),
        experimental_exponential_newton_tol_end=(
            solver.experimental_exponential.tolerance_end
        ),
        experimental_exponential_newton_tol_switch_hi=(
            solver.experimental_exponential.tolerance_switch_high
        ),
        experimental_exponential_newton_tol_switch_lo=(
            solver.experimental_exponential.tolerance_switch_low
        ),
    )
    minimizer_cls = (
        CustomAnderssonMinimizer
        if solver.minimizer_impl == "anderson"
        else CustomQuadraticMinimizer
    )
    extra: dict[str, Any] = {}
    if minimizer_cls is CustomAnderssonMinimizer:
        extra = {
            "anderson_m": solver.anderson.memory,
            "anderson_omega": solver.anderson.omega,
            "anderson_tol_floor": solver.anderson.tolerance_floor,
            "anderson_reg": solver.anderson.regularization,
        }
    non_linearity = common.model.non_linearity
    return minimizer_cls(
        fn=function,
        free_layers=list(stack.free_layers),
        num_iterations=iterations,
        mode=solver.minimizer_mode,
        non_linearity=stack.effective_non_linearity,
        quadratic_diode_param=dict(non_linearity.quadratic_diode_param),
        exponential_diode_param=dict(non_linearity.exponential_diode_param),
        voltage_amp=common.model.voltage_amp,
        current_amp=common.model.current_amp,
        hard_sigmoid_param=dict(non_linearity.hard_sigmoid_param),
        iv_data=None,
        iv_data_path=non_linearity.iv_data_path,
        double_diode_updater=solver.updaters.double_diode,
        adaptive_equilibrium=solver.adaptive_equilibrium,
        overrelaxation_factor=solver.overrelaxation.factor,
        single_diode_updater=solver.updaters.single_diode,
        damping=solver.experimental_exponential.damping,
        experimental_newton_max_steps=(
            solver.experimental_exponential.newton_max_steps
        ),
        minimizer_settings=settings,
        **extra,
    )


def _parameter_learning_rates(
    parameters: tuple[Any, ...],
    *,
    weight_rates: tuple[float, ...],
    bias_rates: tuple[float, ...],
) -> list[float]:
    weights = [parameter for parameter in parameters if not isinstance(parameter, Bias)]
    biases = [parameter for parameter in parameters if isinstance(parameter, Bias)]
    if len(weights) != len(weight_rates):
        raise ValueError(
            "Expected one configured learning rate per non-bias trainable "
            f"parameter ({len(weights)}). Provided value: {len(weight_rates)} rates."
        )
    if len(biases) != len(bias_rates):
        raise ValueError(
            "Expected one configured bias learning rate per trainable Bias "
            f"parameter ({len(biases)}). Provided value: {len(bias_rates)} rates."
        )
    weight_iter = iter(weight_rates)
    bias_iter = iter(bias_rates)
    return [
        float(next(bias_iter) if isinstance(parameter, Bias) else next(weight_iter))
        for parameter in parameters
    ]


def _validate_dataset_shape(
    dataset: str,
    *,
    logical_input_dim: int,
    output_width: int,
) -> None:
    if dataset == "moons":
        valid_input = logical_input_dim == 2 or (
            logical_input_dim >= 2 and logical_input_dim % 2 == 0
        )
    elif dataset == "yinyang":
        valid_input = logical_input_dim == 2
    elif dataset == "digits":
        valid_input = logical_input_dim == 64
    elif dataset == "mnist":
        valid_input = logical_input_dim == 784
    else:
        valid_input = False
    if not valid_input:
        raise ValueError(
            f"Expected the logical input width for dataset {dataset!r} to "
            + (
                "be 2 or an even expanded moons width >= 2"
                if dataset == "moons"
                else (
                    "be 2"
                    if dataset == "yinyang"
                    else ("be 64" if dataset == "digits" else "be 784")
                )
            )
            + f". Provided value: {logical_input_dim!r}."
        )
    num_classes = _CLASS_COUNTS[dataset]
    if output_width not in (num_classes, 2 * num_classes):
        raise ValueError(
            f"Expected the output width for dataset {dataset!r} to be "
            f"{num_classes} or {2 * num_classes}. "
            f"Provided value: {output_width!r}."
        )


def _resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            "Expected config.runtime.device to be a valid torch device. "
            f"Provided value: {value!r}."
        ) from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Expected CUDA to be available for config.runtime.device. "
            f"Provided value: {value!r}."
        )
    return device


def _reset_minimizer_statistics(minimizer: Any) -> None:
    for name in (
        "reset_equilibrium_iteration_stats",
        "reset_overrelaxation_reject_stats",
    ):
        method = getattr(minimizer, name, None)
        if callable(method):
            method()


__all__ = [
    "DataBundle",
    "EvaluationRuntime",
    "LayerStateCheckpoint",
    "ModelStack",
    "TrainRuntime",
    "TypedNetwork",
    "ValidationRuntime",
    "build_data",
    "build_evaluation_runtime",
    "build_model_stack",
    "build_train_runtime",
    "build_validation_runtime",
    "configured_resume_capability",
    "seed_runtime",
]
