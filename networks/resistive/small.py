from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from collections.abc import Iterable
import torch
from networks.resistive.solver import CustomAnderssonMinimizer, CustomQuadraticMinimizer, MinimizerSettings
from networks.function.cost import SquaredError, SquaredErrorPairedOutputs
from networks.function.network import Network
from networks.resistive.builders import ModelBundle, build_deep_resistive_energy
from networks.resistive.digital_low_rank import DigitalLowRankReadout


_CLASS_COUNTS = {"moons": 2, "yinyang": 3, "digits": 10, "mnist": 10}

_DTYPES = {"float32": torch.float32, "float64": torch.float64}

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

def build_model_stack(common: Any) -> ModelStack:
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

def _build_minimizer(
    common: Any,
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
