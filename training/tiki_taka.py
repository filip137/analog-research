"""Tiki-Taka gradient accumulation for EqProp/DRN parameters.

The DRN differentiators produce a complete gradient tensor for every model
parameter.  This module treats those tensors as writes to a hidden, signed
gradient crossbar and periodically transfers slices of that crossbar to the
visible model weights.

The default backend is an ideal tensor implementation of AIHWKit's
``TransferCompound`` update pipeline. An optional native AIHWKit backend sends
each complete gradient through real pulsed device tiles and maps their realized
slow-device states into the DRN conductance range.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite, prod
from numbers import Integral, Real
from typing import Optional, Union

import torch

from model.variable.parameter import Bias, ConvWeight, DenseWeight, PoolWeight


AIHWKIT_TIKI_TAKA_PRESETS = (
    "TikiTakaIdealizedPreset",
    "TikiTakaReRamESPreset",
    "TikiTakaReRamSBPreset",
    "TikiTakaCapacitorPreset",
    "TikiTakaEcRamPreset",
    "TikiTakaEcRamMOPreset",
)

AIHWKIT_WRITE_NOISE_PRESETS = (
    "TikiTakaReRamESPreset",
    "TikiTakaReRamSBPreset",
)


@dataclass(frozen=True)
class TikiTakaConfig:
    """Configuration for the hidden-gradient-crossbar update pipeline.

    ``transfer_every`` counts optimizer steps, which correspond to minibatches
    in the existing DRN trainers.  The auxiliary array stores the negative
    gradient/update direction.  Consequently, transfers add selected
    auxiliary slices to the visible weights.
    """

    fast_lr: float = 1.0
    transfer_every: int = 1
    units_in_mbatch: bool = True
    n_reads_per_transfer: int = 1
    gamma: float = 0.0
    transfer_lr: float = 1.0
    scale_transfer_lr: bool = True
    transfer_columns: bool = True
    with_reset_prob: float = 0.0
    random_selection: bool = False
    fast_weight_min: Optional[float] = None
    fast_weight_max: Optional[float] = None
    accumulate_biases: bool = False
    aihwkit_preset: Optional[str] = None
    aihwkit_conductance_min: Optional[float] = None
    aihwkit_conductance_max: Optional[float] = None
    aihwkit_construction_seed: Optional[int] = None

    def __post_init__(self) -> None:
        _require_real("fast_lr", self.fast_lr, lower=0.0)
        _require_integer("transfer_every", self.transfer_every, lower=0)
        _require_bool("units_in_mbatch", self.units_in_mbatch)
        _require_integer("n_reads_per_transfer", self.n_reads_per_transfer, lower=1)
        _require_real("gamma", self.gamma)
        _require_real("transfer_lr", self.transfer_lr, lower=0.0)
        _require_bool("scale_transfer_lr", self.scale_transfer_lr)
        _require_bool("transfer_columns", self.transfer_columns)
        _require_real(
            "with_reset_prob",
            self.with_reset_prob,
            lower=0.0,
            upper=1.0,
        )
        _require_bool("random_selection", self.random_selection)
        _require_optional_real("fast_weight_min", self.fast_weight_min)
        _require_optional_real("fast_weight_max", self.fast_weight_max)
        _require_bool("accumulate_biases", self.accumulate_biases)
        _require_optional_real(
            "aihwkit_conductance_min",
            self.aihwkit_conductance_min,
        )
        _require_optional_real(
            "aihwkit_conductance_max",
            self.aihwkit_conductance_max,
        )
        if self.aihwkit_construction_seed is not None:
            _require_integer(
                "aihwkit_construction_seed",
                self.aihwkit_construction_seed,
                lower=1,
            )
        if not self.units_in_mbatch:
            raise ValueError(
                "Expected update_pipeline.units_in_mbatch to be true because "
                "server_code supplies one averaged EqProp gradient per minibatch. "
                f"Provided value: {self.units_in_mbatch!r}."
            )
        if self.gamma != 0.0:
            raise ValueError(
                "Expected update_pipeline.gamma to be 0.0 so the main DRN "
                "crossbar is the only array visible to forward and inference. "
                f"Provided value: {self.gamma!r}."
            )
        if not self.transfer_columns and self.with_reset_prob > 0.0:
            raise ValueError(
                "Expected update_pipeline.with_reset_prob to be 0.0 when "
                "transfer_columns is false; AIHWKit resets transferred columns "
                "only. "
                f"Provided value: transfer_columns={self.transfer_columns!r}, "
                f"with_reset_prob={self.with_reset_prob!r}."
            )
        if (
            self.fast_weight_min is not None
            and self.fast_weight_max is not None
            and self.fast_weight_min >= self.fast_weight_max
        ):
            raise ValueError(
                "Expected fast_weight_min to be smaller than fast_weight_max. "
                f"Provided value: min={self.fast_weight_min!r}, "
                f"max={self.fast_weight_max!r}."
            )
        if self.aihwkit_preset is not None:
            if (
                not isinstance(self.aihwkit_preset, str)
                or self.aihwkit_preset not in AIHWKIT_TIKI_TAKA_PRESETS
            ):
                raise ValueError(
                    "Expected update_pipeline.aihwkit_preset to be null or one "
                    "of: "
                    f"{', '.join(AIHWKIT_TIKI_TAKA_PRESETS)}. "
                    f"Provided value: {self.aihwkit_preset!r}."
                )
            if (
                self.aihwkit_conductance_min is not None
                and self.aihwkit_conductance_min < 0.0
            ):
                raise ValueError(
                    "Expected update_pipeline.aihwkit_conductance_min to be "
                    "non-negative. "
                    f"Provided value: {self.aihwkit_conductance_min!r}."
                )
            if (
                self.aihwkit_conductance_max is not None
                and self.aihwkit_conductance_max <= 0.0
            ):
                raise ValueError(
                    "Expected update_pipeline.aihwkit_conductance_max to be "
                    "positive. "
                    f"Provided value: {self.aihwkit_conductance_max!r}."
                )
            if (
                self.aihwkit_conductance_min is not None
                and self.aihwkit_conductance_max is not None
                and self.aihwkit_conductance_min
                >= self.aihwkit_conductance_max
            ):
                raise ValueError(
                    "Expected aihwkit_conductance_min to be smaller than "
                    "aihwkit_conductance_max. "
                    "Provided value: "
                    f"min={self.aihwkit_conductance_min!r}, "
                    f"max={self.aihwkit_conductance_max!r}."
                )
            if self.fast_weight_min is not None and self.fast_weight_min > 0.0:
                raise ValueError(
                    "Expected update_pipeline.fast_weight_min <= 0 for an "
                    "AIHWKit pulsed fast device. "
                    f"Provided value: {self.fast_weight_min!r}."
                )
            if self.fast_weight_max is not None and self.fast_weight_max < 0.0:
                raise ValueError(
                    "Expected update_pipeline.fast_weight_max >= 0 for an "
                    "AIHWKit pulsed fast device. "
                    f"Provided value: {self.fast_weight_max!r}."
                )
        elif any(
            value is not None
            for value in (
                self.aihwkit_conductance_min,
                self.aihwkit_conductance_max,
                self.aihwkit_construction_seed,
            )
        ):
            raise ValueError(
                "Expected update_pipeline.aihwkit_preset to name a preset when "
                "AIHWKit conductance or seed settings are provided. "
                "Provided value: "
                f"aihwkit_preset={self.aihwkit_preset!r}, "
                "aihwkit_conductance_min="
                f"{self.aihwkit_conductance_min!r}, "
                "aihwkit_conductance_max="
                f"{self.aihwkit_conductance_max!r}, "
                "aihwkit_construction_seed="
                f"{self.aihwkit_construction_seed!r}."
            )


def _require_bool(name: str, value) -> None:
    if not isinstance(value, bool):
        raise ValueError(
            f"Expected update_pipeline.{name} to be a boolean. "
            f"Provided value: {value!r}."
        )


def _require_integer(name: str, value, *, lower: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
        raise ValueError(
            f"Expected update_pipeline.{name} to be an integer >= {lower}. "
            f"Provided value: {value!r}."
        )


def _require_real(
    name: str,
    value,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    lower_inclusive: bool = True,
) -> None:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and isfinite(float(value))
    )
    if valid and lower is not None:
        valid = value >= lower if lower_inclusive else value > lower
    if valid and upper is not None:
        valid = value <= upper
    if valid:
        return

    if lower is not None and upper is not None:
        expectation = f"a number in [{lower}, {upper}]"
    elif lower is not None and lower_inclusive:
        expectation = f"a number >= {lower}"
    elif lower is not None:
        expectation = f"a number > {lower}"
    else:
        expectation = "a number"
    raise ValueError(
        f"Expected update_pipeline.{name} to be {expectation}. "
        f"Provided value: {value!r}."
    )


def _require_optional_real(name: str, value) -> None:
    valid = (
        value is None
        or (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and isfinite(float(value))
        )
    )
    if not valid:
        raise ValueError(
            f"Expected update_pipeline.{name} to be a number or null. "
            f"Provided value: {value!r}."
        )


def parse_update_pipeline(
    value: Optional[Union[TikiTakaConfig, Mapping]],
) -> Optional[TikiTakaConfig]:
    """Parse the optional ``update_pipeline`` configuration block."""

    if value is None or isinstance(value, TikiTakaConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            "Expected update_pipeline to be an object with "
            "'type': 'direct' or 'type': 'tiki_taka'. "
            f"Provided value: {value!r}."
        )

    pipeline_type = value.get("type")
    if pipeline_type == "direct":
        if len(value) != 1:
            raise ValueError(
                "Expected update_pipeline with type='direct' to contain no "
                "additional keys. "
                f"Provided value: {value!r}."
            )
        return None
    if pipeline_type != "tiki_taka":
        raise ValueError(
            "Expected update_pipeline.type to be 'direct' or 'tiki_taka'. "
            f"Provided value: {pipeline_type!r}."
        )

    config_values = {key: item for key, item in value.items() if key != "type"}
    allowed = {field.name for field in fields(TikiTakaConfig)}
    unknown = sorted(set(config_values) - allowed)
    if unknown:
        raise ValueError(
            "Expected update_pipeline keys to be drawn from: "
            f"{', '.join(sorted(allowed | {'type'}))}. "
            f"Provided value: unknown keys {unknown!r} in {value!r}."
        )
    return TikiTakaConfig(**config_values)


def _crossbar_layout(parameter) -> tuple[str, Optional[int]]:
    if isinstance(parameter, DenseWeight):
        return "dense", prod(parameter._layer_pre_shape)
    if isinstance(parameter, ConvWeight):
        return "conv", None
    if isinstance(parameter, Bias):
        return "bias", None
    return "flat", None


def _as_crossbar_matrix(
    tensor: torch.Tensor,
    layout: str,
    dense_input_size: Optional[int],
) -> torch.Tensor:
    """Return a logical ``[out, in]`` view of a parameter tensor."""

    if layout == "dense":
        return tensor.reshape(dense_input_size, -1).transpose(0, 1)
    if layout == "conv":
        return tensor.reshape(tensor.shape[0], -1)
    if layout == "bias":
        return tensor.reshape(-1, 1)
    return tensor.reshape(1, -1)


def _finite_parameter_bound(value) -> Optional[float]:
    """Return a finite scalar parameter bound, or ``None`` when unbounded."""

    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if isfinite(converted) else None


def _resolve_conductance_bounds(
    parameter,
    config: TikiTakaConfig,
) -> tuple[float, float]:
    """Resolve the DRN-side range used by the AIHWKit affine mapping."""

    parameter_min = _finite_parameter_bound(getattr(parameter, "min_cond", None))
    parameter_max = _finite_parameter_bound(getattr(parameter, "max_cond", None))
    if parameter_min is None and getattr(parameter, "_non_negative", False):
        parameter_min = 0.0

    conductance_min = config.aihwkit_conductance_min
    if conductance_min is None:
        conductance_min = parameter_min
    conductance_max = config.aihwkit_conductance_max
    if conductance_max is None:
        conductance_max = parameter_max

    if conductance_min is None or conductance_max is None:
        raise ValueError(
            "Expected finite AIHWKit conductance bounds, supplied either by "
            "update_pipeline.aihwkit_conductance_min/max or by the DRN "
            "parameter clamp. "
            "Provided value: "
            f"parameter={getattr(parameter, 'name', type(parameter).__name__)!r}, "
            f"parameter_min={parameter_min!r}, parameter_max={parameter_max!r}, "
            f"configured_min={config.aihwkit_conductance_min!r}, "
            f"configured_max={config.aihwkit_conductance_max!r}."
        )
    conductance_min = float(conductance_min)
    conductance_max = float(conductance_max)
    if conductance_min < 0.0 or conductance_min >= conductance_max:
        raise ValueError(
            "Expected finite AIHWKit conductance bounds with "
            "0 <= min < max. "
            f"Provided value: min={conductance_min!r}, "
            f"max={conductance_max!r}."
        )
    if parameter_min is not None and conductance_min < parameter_min:
        raise ValueError(
            "Expected update_pipeline.aihwkit_conductance_min to lie within "
            "the DRN parameter clamp. "
            f"Provided value: conductance_min={conductance_min!r}, "
            f"parameter_min={parameter_min!r}."
        )
    if parameter_max is not None and conductance_max > parameter_max:
        raise ValueError(
            "Expected update_pipeline.aihwkit_conductance_max to lie within "
            "the DRN parameter clamp. "
            f"Provided value: conductance_max={conductance_max!r}, "
            f"parameter_max={parameter_max!r}."
        )
    return conductance_min, conductance_max


class TikiTakaOptimizer(torch.optim.Optimizer):
    """Optimizer implementing a two-crossbar Tiki-Taka update pipeline.

    Dense and convolutional weights use the auxiliary crossbar.  Biases use
    direct digital SGD unless ``accumulate_biases`` is enabled.  Pooling
    weights remain frozen, matching :class:`training.core.optimizers.SGDOptimizer`.

    The auxiliary state and its per-parameter transfer counters/cursors live in
    the standard PyTorch optimizer state, so ``state_dict()`` preserves them.
    """

    def __init__(
        self,
        energy_fn,
        cost_fn,
        learning_rates,
        config: Union[TikiTakaConfig, Mapping],
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        provided_config = config
        config = parse_update_pipeline(config)
        if config is None:
            raise ValueError(
                "Expected a TikiTakaConfig or Tiki-Taka update_pipeline object. "
                f"Provided value: {provided_config!r}."
            )
        if momentum != 0.0 or weight_decay != 0.0:
            raise ValueError(
                "Expected momentum=0 and weight_decay=0 for Tiki-Taka updates; "
                "their placement in the two-crossbar pipeline is undefined. "
                f"Provided value: momentum={momentum!r}, "
                f"weight_decay={weight_decay!r}."
            )

        parameters = [
            parameter
            for parameter in energy_fn.params() + cost_fn.params()
            if not isinstance(parameter, PoolWeight)
        ]
        if len(learning_rates) != len(parameters):
            raise ValueError(
                f"Expected {len(parameters)} learning rates after filtering "
                "PoolWeight parameters. "
                f"Provided value: {len(learning_rates)} learning rates."
            )

        parameter_groups = []
        for parameter, learning_rate in zip(parameters, learning_rates):
            _require_real("learning_rate", learning_rate, lower=0.0)
            layout, dense_input_size = _crossbar_layout(parameter)
            use_auxiliary = not isinstance(parameter, Bias) or config.accumulate_biases
            conductance_bounds = (None, None)
            if config.aihwkit_preset is not None and use_auxiliary:
                conductance_bounds = _resolve_conductance_bounds(parameter, config)
            parameter_groups.append(
                {
                    "params": [parameter.state],
                    "lr": float(learning_rate),
                    "use_tiki_taka": use_auxiliary,
                    "crossbar_layout": layout,
                    "dense_input_size": dense_input_size,
                    "conductance_min": conductance_bounds[0],
                    "conductance_max": conductance_bounds[1],
                }
            )

        defaults = {"lr": 0.1}
        super().__init__(parameter_groups, defaults)
        self._learning_rates = list(learning_rates)
        self._config = config
        self._aihwkit_tiles = {}
        self._aihwkit_mappings = {}
        self._aihwkit_identity_cache = {}
        self._aihwkit_version = None
        if config.aihwkit_preset is not None:
            self._initialize_aihwkit_backend()

    @property
    def config(self) -> TikiTakaConfig:
        """Return the immutable Tiki-Taka configuration."""

        return self._config

    @property
    def uses_aihwkit(self) -> bool:
        """Return whether updates use native AIHWKit pulsed device tiles."""

        return self._config.aihwkit_preset is not None

    @property
    def aihwkit_version(self) -> Optional[str]:
        """Return the loaded AIHWKit version for the native backend."""

        return self._aihwkit_version

    def _initialize_aihwkit_backend(self) -> None:
        try:
            import aihwkit
            from aihwkit.simulator import presets
            from aihwkit.simulator.rpu_base import cuda as aihwkit_cuda
            from aihwkit.simulator.tiles import AnalogTile
        except ImportError as exc:
            raise RuntimeError(
                "Expected the optional 'aihwkit' package in order to use "
                "update_pipeline.aihwkit_preset. "
                f"Provided value: {self._config.aihwkit_preset!r}."
            ) from exc

        self._aihwkit_version = getattr(aihwkit, "__version__", "unknown")
        preset_class = getattr(presets, self._config.aihwkit_preset, None)
        if preset_class is None:
            raise RuntimeError(
                "Expected the installed AIHWKit version to provide preset "
                f"{self._config.aihwkit_preset!r}. "
                f"Provided value: aihwkit_version={self._aihwkit_version!r}."
            )

        for group_index, group in enumerate(self.param_groups):
            if not group["use_tiki_taka"]:
                continue
            parameter = group["params"][0]
            if parameter.device.type not in ("cpu", "cuda"):
                raise RuntimeError(
                    "Expected AIHWKit-backed DRN parameters on a CPU or CUDA "
                    "device. "
                    f"Provided value: parameter_device={str(parameter.device)!r}."
                )
            if parameter.device.type == "cuda" and not aihwkit_cuda.is_compiled():
                raise RuntimeError(
                    "Expected an AIHWKit build with CUDA support for CUDA DRN "
                    "parameters, or a CPU training device. "
                    f"Provided value: parameter_device={str(parameter.device)!r}, "
                    f"aihwkit_version={self._aihwkit_version!r}."
                )

            rpu_config = preset_class()
            compound = rpu_config.device
            for name in (
                "fast_lr",
                "transfer_every",
                "units_in_mbatch",
                "n_reads_per_transfer",
                "gamma",
                "transfer_lr",
                "scale_transfer_lr",
                "transfer_columns",
                "with_reset_prob",
                "random_selection",
            ):
                setattr(compound, name, getattr(self._config, name))

            fast_device = compound.unit_cell_devices[0]
            if self._config.fast_weight_min is not None:
                fast_device.w_min = float(self._config.fast_weight_min)
            if self._config.fast_weight_max is not None:
                fast_device.w_max = float(self._config.fast_weight_max)

            seed = self._config.aihwkit_construction_seed
            if seed is not None:
                seed_offset = group_index * (
                    len(compound.unit_cell_devices) + 1
                )
                if hasattr(compound, "construction_seed"):
                    compound.construction_seed = int(seed) + seed_offset
                for device_index, device in enumerate(compound.unit_cell_devices):
                    if hasattr(device, "construction_seed"):
                        device.construction_seed = (
                            int(seed)
                            + seed_offset
                            + device_index
                            + 1
                        )

            matrix = _as_crossbar_matrix(
                parameter,
                group["crossbar_layout"],
                group["dense_input_size"],
            )
            tile = AnalogTile(
                matrix.shape[0],
                matrix.shape[1],
                rpu_config,
                bias=False,
            )
            if parameter.device.type == "cuda":
                tile = tile.cuda(parameter.device)

            mapping = self._mapping_from_tile(tile, group)
            conductance_min = mapping["conductance_min"]
            conductance_max = mapping["conductance_max"]
            tolerance = max(1e-7, 1e-6 * (conductance_max - conductance_min))
            matrix_min = float(matrix.detach().min().item())
            matrix_max = float(matrix.detach().max().item())
            if (
                matrix_min < conductance_min - tolerance
                or matrix_max > conductance_max + tolerance
            ):
                raise ValueError(
                    "Expected initial DRN conductances to lie inside the "
                    "AIHWKit conductance mapping. "
                    "Provided value: "
                    f"parameter_min={matrix_min!r}, parameter_max={matrix_max!r}, "
                    f"mapping_min={conductance_min!r}, "
                    f"mapping_max={conductance_max!r}."
                )

            ratio = (
                (matrix.detach() - conductance_min)
                / (conductance_max - conductance_min)
            ).clamp_(0.0, 1.0)
            initial_device_state = (
                mapping["state_min"]
                + ratio * (mapping["state_max"] - mapping["state_min"])
            )
            # AIHWKit's weight setter accepts host tensors for both CPU and
            # CUDA tiles. Its public getters likewise return host tensors.
            tile.set_weights(initial_device_state.detach().cpu().contiguous())
            tile.set_learning_rate(float(group["lr"]))

            self._aihwkit_tiles[parameter] = tile
            self._aihwkit_mappings[parameter] = self._mapping_from_tile(tile, group)
            state = self.state[parameter]
            state["step"] = 0
            state["transfer_index"] = 0
            state["transfer_count"] = 0
            self._sync_parameter_from_aihwkit(parameter, group)

    def _mapping_from_tile(self, tile, group: dict) -> dict:
        hidden = tile.get_hidden_parameters()
        parameter = group["params"][0]
        required = (
            "min_bound_1",
            "max_bound_1",
            "hidden_weights_0",
            "hidden_weights_1",
        )
        missing = [name for name in required if name not in hidden]
        if missing:
            raise RuntimeError(
                "Expected a two-device AIHWKit TransferCompound exposing "
                "min_bound_1, max_bound_1, hidden_weights_0, and "
                "hidden_weights_1. "
                f"Provided value: preset={self._config.aihwkit_preset!r}, "
                f"aihwkit_version={self._aihwkit_version!r}, "
                f"missing={missing!r}, available={list(hidden)!r}."
            )
        state_min = hidden["min_bound_1"].detach().to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        state_max = hidden["max_bound_1"].detach().to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        if not bool(torch.all(state_max > state_min)):
            raise RuntimeError(
                "Expected every AIHWKit slow-device upper bound to exceed its "
                "lower bound. "
                "Provided value: "
                f"minimum_span={float((state_max - state_min).min().item())!r}."
            )
        return {
            "state_min": state_min,
            "state_max": state_max,
            "conductance_min": float(group["conductance_min"]),
            "conductance_max": float(group["conductance_max"]),
        }

    @staticmethod
    def _tile_visible_weights(tile) -> torch.Tensor:
        weights = tile.get_weights()
        return weights[0] if isinstance(weights, tuple) else weights

    def _sync_parameter_from_aihwkit(
        self,
        parameter: torch.Tensor,
        group: dict,
    ) -> None:
        tile = self._aihwkit_tiles[parameter]
        mapping = self._aihwkit_mappings[parameter]
        device_state = self._tile_visible_weights(tile).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        state_fraction = (
            (device_state - mapping["state_min"])
            / (mapping["state_max"] - mapping["state_min"])
        ).clamp_(0.0, 1.0)
        conductance = mapping["conductance_min"] + state_fraction * (
            mapping["conductance_max"] - mapping["conductance_min"]
        )
        visible = _as_crossbar_matrix(
            parameter,
            group["crossbar_layout"],
            group["dense_input_size"],
        )
        visible.copy_(conductance.to(device=visible.device, dtype=visible.dtype))

    def _matrix_in_parameter_layout(
        self,
        matrix: torch.Tensor,
        parameter: torch.Tensor,
        group: dict,
    ) -> torch.Tensor:
        result = torch.empty_like(parameter, memory_format=torch.preserve_format)
        _as_crossbar_matrix(
            result,
            group["crossbar_layout"],
            group["dense_input_size"],
        ).copy_(matrix.to(device=result.device, dtype=result.dtype))
        return result

    def auxiliary_state(self, parameter: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the live auxiliary array for a parameter tensor, if created."""

        if self.uses_aihwkit:
            tile = self._aihwkit_tiles.get(parameter)
            if tile is None:
                return None
            matrix = tile.get_hidden_parameters().get("hidden_weights_0")
            if matrix is None:
                return None
            group = next(
                group
                for group in self.param_groups
                if group["params"][0] is parameter
            )
            return self._matrix_in_parameter_layout(matrix, parameter, group)
        return self.state[parameter].get("auxiliary")

    def aihwkit_device_state(
        self,
        parameter: torch.Tensor,
    ) -> Optional[OrderedDict]:
        """Return cloned normalized device state for an AIHWKit parameter.

        ``hidden_weights_0`` and ``hidden_weights_1`` are the apparent fast and
        slow device states. Presets with write noise can additionally expose
        persistent device states.
        """

        if not self.uses_aihwkit:
            return None
        tile = self._aihwkit_tiles.get(parameter)
        if tile is None:
            return None
        return OrderedDict(
            (name, value.detach().clone())
            for name, value in tile.get_hidden_parameters().items()
        )

    def aihwkit_slow_conductance(
        self,
        parameter: torch.Tensor,
        *,
        persistent: bool = False,
    ) -> Optional[torch.Tensor]:
        """Return the slow AIHWKit state mapped into the DRN layout and range.

        The apparent slow state is used by default. For devices with write
        noise, ``persistent=True`` returns the underlying programmed state.
        Devices that do not expose a persistent slow state return ``None``.
        """

        if not self.uses_aihwkit:
            return None
        tile = self._aihwkit_tiles.get(parameter)
        mapping = self._aihwkit_mappings.get(parameter)
        if tile is None or mapping is None:
            return None
        state_name = "persistent_weights_1" if persistent else "hidden_weights_1"
        state_matrix = tile.get_hidden_parameters().get(state_name)
        if state_matrix is None:
            return None
        state_matrix = state_matrix.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        state_fraction = (
            (state_matrix - mapping["state_min"])
            / (mapping["state_max"] - mapping["state_min"])
        ).clamp_(0.0, 1.0)
        conductance = mapping["conductance_min"] + state_fraction * (
            mapping["conductance_max"] - mapping["conductance_min"]
        )
        group = next(
            group
            for group in self.param_groups
            if group["params"][0] is parameter
        )
        return self._matrix_in_parameter_layout(
            conductance,
            parameter,
            group,
        )

    def _snapshot_aihwkit_tile(
        self,
        tile,
        parameter: torch.Tensor,
    ) -> dict:
        return {
            "parameter_shape": tuple(parameter.shape),
            "visible": self._tile_visible_weights(tile).detach().clone(),
            "hidden": OrderedDict(
                (name, value.detach().clone())
                for name, value in tile.get_hidden_parameters().items()
            ),
            "extra": dict(tile.tile.dump_extra()),
            "learning_rate": float(tile.get_learning_rate()),
            "aihwkit_version": self._aihwkit_version,
        }

    def state_dict(self) -> dict:
        """Serialize auxiliary arrays, transfer progress, and pipeline config."""

        state_dict = super().state_dict()
        if self.uses_aihwkit:
            for saved_group, live_group in zip(
                state_dict["param_groups"],
                self.param_groups,
            ):
                for saved_parameter, parameter in zip(
                    saved_group["params"],
                    live_group["params"],
                ):
                    tile = self._aihwkit_tiles.get(parameter)
                    if tile is not None:
                        parameter_state = dict(
                            state_dict["state"].get(saved_parameter, {})
                        )
                        parameter_state[
                            "aihwkit_tile_state"
                        ] = self._snapshot_aihwkit_tile(tile, parameter)
                        state_dict["state"][saved_parameter] = parameter_state
        state_dict["tiki_taka_config"] = asdict(self._config)
        return state_dict

    def _preflight_parameter_groups(self, state_dict: Mapping) -> None:
        """Reject structurally incompatible optimizer state before loading it."""

        saved_groups = state_dict.get("param_groups")
        saved_states = state_dict.get("state")
        if not isinstance(saved_groups, list) or not isinstance(saved_states, Mapping):
            raise ValueError(
                "Expected a Tiki-Taka optimizer checkpoint with list "
                "'param_groups' and mapping 'state' entries. "
                "Provided value: "
                f"param_groups_type={type(saved_groups).__name__!r}, "
                f"state_type={type(saved_states).__name__!r}."
            )
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                "Expected a Tiki-Taka optimizer checkpoint with the same "
                "number of parameter groups as the current optimizer. "
                f"Provided value: saved={len(saved_groups)!r}, "
                f"current={len(self.param_groups)!r}."
            )

        structural_keys = (
            "use_tiki_taka",
            "crossbar_layout",
            "dense_input_size",
            "conductance_min",
            "conductance_max",
        )
        for group_index, (saved_group, live_group) in enumerate(
            zip(saved_groups, self.param_groups)
        ):
            saved_parameters = saved_group.get("params", [])
            live_parameters = live_group["params"]
            if len(saved_parameters) != len(live_parameters):
                raise ValueError(
                    "Expected a Tiki-Taka optimizer checkpoint with the same "
                    "parameters in every group. "
                    "Provided value: "
                    f"group_index={group_index!r}, "
                    f"saved={len(saved_parameters)!r}, "
                    f"current={len(live_parameters)!r}."
                )
            for key in structural_keys:
                if saved_group.get(key) != live_group.get(key):
                    raise ValueError(
                        "Expected a Tiki-Taka optimizer checkpoint with the "
                        "same parameter-group structure and conductance bounds "
                        "as the current optimizer. "
                        "Provided value: "
                        f"group_index={group_index!r}, field={key!r}, "
                        f"saved={saved_group.get(key)!r}, "
                        f"current={live_group.get(key)!r}."
                    )

            for saved_parameter, live_parameter in zip(
                saved_parameters,
                live_parameters,
            ):
                parameter_state = saved_states.get(saved_parameter, {})
                auxiliary = (
                    parameter_state.get("auxiliary")
                    if isinstance(parameter_state, Mapping)
                    else None
                )
                if (
                    auxiliary is not None
                    and tuple(auxiliary.shape) != tuple(live_parameter.shape)
                ):
                    raise ValueError(
                        "Expected a Tiki-Taka auxiliary tensor with the same "
                        "shape as its current parameter. "
                        "Provided value: "
                        f"group_index={group_index!r}, "
                        f"saved_shape={tuple(auxiliary.shape)!r}, "
                        f"current_shape={tuple(live_parameter.shape)!r}."
                    )

    def _preflight_aihwkit_checkpoint(self, state_dict: Mapping) -> None:
        """Validate every native snapshot before mutating optimizer state."""

        saved_groups = state_dict.get("param_groups")
        saved_states = state_dict.get("state")
        if not isinstance(saved_groups, list) or not isinstance(saved_states, Mapping):
            raise ValueError(
                "Expected an AIHWKit optimizer checkpoint with list "
                "'param_groups' and mapping 'state' entries. "
                "Provided value: "
                f"param_groups_type={type(saved_groups).__name__!r}, "
                f"state_type={type(saved_states).__name__!r}."
            )
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                "Expected an AIHWKit optimizer checkpoint with the same number "
                "of parameter groups as the current optimizer. "
                f"Provided value: saved={len(saved_groups)!r}, "
                f"current={len(self.param_groups)!r}."
            )

        expected_snapshot_keys = {
            "parameter_shape",
            "visible",
            "hidden",
            "extra",
            "learning_rate",
            "aihwkit_version",
        }
        for group_index, (saved_group, live_group) in enumerate(
            zip(saved_groups, self.param_groups)
        ):
            if not live_group["use_tiki_taka"]:
                continue
            saved_parameters = saved_group.get("params", [])
            live_parameters = live_group["params"]
            if len(saved_parameters) != len(live_parameters):
                raise ValueError(
                    "Expected an AIHWKit optimizer checkpoint with the same "
                    "parameters in every accumulated group. "
                    "Provided value: "
                    f"group_index={group_index!r}, "
                    f"saved={len(saved_parameters)!r}, "
                    f"current={len(live_parameters)!r}."
                )
            for saved_parameter, live_parameter in zip(
                saved_parameters,
                live_parameters,
            ):
                parameter_state = saved_states.get(saved_parameter)
                snapshot = (
                    parameter_state.get("aihwkit_tile_state")
                    if isinstance(parameter_state, Mapping)
                    else None
                )
                if not isinstance(snapshot, Mapping):
                    raise ValueError(
                        "Expected AIHWKit optimizer state to include a native "
                        "tile snapshot for every accumulated parameter. "
                        "Provided value: "
                        f"group_index={group_index!r}, "
                        f"saved_parameter={saved_parameter!r}."
                    )
                missing = sorted(expected_snapshot_keys - set(snapshot))
                if missing:
                    raise ValueError(
                        "Expected an AIHWKit tile snapshot containing visible, "
                        "hidden, extra, learning_rate, aihwkit_version, and "
                        "parameter_shape state. "
                        f"Provided value: missing={missing!r}, "
                        f"keys={list(snapshot)!r}."
                    )
                saved_version = snapshot["aihwkit_version"]
                if saved_version != self._aihwkit_version:
                    raise ValueError(
                        "Expected an AIHWKit tile snapshot created by the same "
                        "AIHWKit version as the current optimizer. "
                        f"Provided value: saved={saved_version!r}, "
                        f"current={self._aihwkit_version!r}."
                    )

                tile = self._aihwkit_tiles[live_parameter]
                expected_parameter_shape = tuple(live_parameter.shape)
                expected_visible_shape = tuple(
                    _as_crossbar_matrix(
                        live_parameter,
                        live_group["crossbar_layout"],
                        live_group["dense_input_size"],
                    ).shape
                )
                visible = snapshot["visible"]
                hidden = snapshot["hidden"]
                try:
                    saved_parameter_shape = tuple(snapshot["parameter_shape"])
                except TypeError:
                    saved_parameter_shape = None
                current_hidden = tile.get_hidden_parameters()
                saved_hidden_names = (
                    list(hidden) if isinstance(hidden, Mapping) else None
                )
                current_hidden_names = list(current_hidden)
                hidden_shapes_match = (
                    saved_hidden_names == current_hidden_names
                    and all(
                        hasattr(hidden[name], "shape")
                        and tuple(hidden[name].shape)
                        == tuple(current_hidden[name].shape)
                        for name in current_hidden_names
                    )
                )
                if (
                    saved_parameter_shape != expected_parameter_shape
                    or not hasattr(visible, "shape")
                    or tuple(visible.shape) != expected_visible_shape
                    or not hidden_shapes_match
                ):
                    saved_visible_shape = (
                        tuple(visible.shape)
                        if hasattr(visible, "shape")
                        else None
                    )
                    raise ValueError(
                        "Expected an AIHWKit native tile snapshot with the "
                        "same parameter, visible-tile, and hidden-state shapes "
                        "as the current optimizer. "
                        "Provided value: "
                        f"group_index={group_index!r}, "
                        "saved_parameter_shape="
                        f"{saved_parameter_shape!r}, "
                        f"current_parameter_shape={expected_parameter_shape!r}, "
                        f"saved_visible_shape={saved_visible_shape!r}, "
                        f"current_visible_shape={expected_visible_shape!r}, "
                        f"saved_hidden_names={saved_hidden_names!r}, "
                        f"current_hidden_names={current_hidden_names!r}."
                    )

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore state while guarding against a different transfer pipeline."""

        saved_config = state_dict.get("tiki_taka_config")
        current_config = asdict(self._config)
        normalized_saved_config = saved_config
        if isinstance(saved_config, Mapping):
            try:
                normalized_saved_config = asdict(TikiTakaConfig(**saved_config))
            except (TypeError, ValueError):
                normalized_saved_config = saved_config
        if (
            normalized_saved_config is not None
            and normalized_saved_config != current_config
        ):
            raise ValueError(
                "Expected optimizer state with the same Tiki-Taka configuration "
                "as the current optimizer. "
                f"Provided value: saved={saved_config!r}, "
                f"current={current_config!r}."
            )
        self._preflight_parameter_groups(state_dict)
        if (
            self.uses_aihwkit
            and self._config.aihwkit_preset in AIHWKIT_WRITE_NOISE_PRESETS
        ):
            raise RuntimeError(
                "Expected an AIHWKit Tiki-Taka checkpoint whose device state "
                "can be restored without resampling persistent write noise. "
                "AIHWKit's public hidden-parameter setter changes both "
                "apparent and programmed values for this preset, so loading "
                "was rejected before mutating the optimizer. "
                "Provided value: "
                f"preset={self._config.aihwkit_preset!r}, "
                f"aihwkit_version={self._aihwkit_version!r}."
            )
        if self.uses_aihwkit:
            self._preflight_aihwkit_checkpoint(state_dict)
        optimizer_state = {
            key: value
            for key, value in state_dict.items()
            if key != "tiki_taka_config"
        }
        result = super().load_state_dict(optimizer_state)

        if self.uses_aihwkit:
            for group in self.param_groups:
                if not group["use_tiki_taka"]:
                    continue
                parameter = group["params"][0]
                snapshot = self.state[parameter].pop("aihwkit_tile_state", None)
                if snapshot is None:
                    raise ValueError(
                        "Expected AIHWKit optimizer state to include a native "
                        "tile snapshot for every accumulated parameter. "
                        "Provided value: "
                        f"parameter_shape={tuple(parameter.shape)!r}."
                    )
                self._restore_aihwkit_tile(parameter, group, snapshot)
        return result

    def _restore_aihwkit_tile(
        self,
        parameter: torch.Tensor,
        group: dict,
        snapshot: Mapping,
    ) -> None:
        expected = {
            "parameter_shape",
            "visible",
            "hidden",
            "extra",
            "learning_rate",
            "aihwkit_version",
        }
        missing = sorted(expected - set(snapshot))
        if missing:
            raise ValueError(
                "Expected an AIHWKit tile snapshot containing visible, hidden, "
                "extra, learning_rate, aihwkit_version, and parameter_shape "
                "state. "
                f"Provided value: missing={missing!r}, "
                f"keys={list(snapshot)!r}."
            )

        tile = self._aihwkit_tiles[parameter]
        visible = snapshot["visible"].detach().cpu()
        hidden = OrderedDict(
            (name, value.detach().cpu())
            for name, value in snapshot["hidden"].items()
        )

        # AIHWKit 1.1 restores hidden parameters before set_weights(), but the
        # latter resets TransferCompound's fast device. Preserve it by restoring
        # in the opposite order. Presets with persistent write noise are
        # rejected before this method because their public restore is lossy.
        tile.set_weights(visible)
        tile.set_hidden_parameters(hidden)
        tile.tile.load_extra(snapshot["extra"], True)
        tile.set_learning_rate(float(snapshot["learning_rate"]))
        self._aihwkit_mappings[parameter] = self._mapping_from_tile(tile, group)
        self._sync_parameter_from_aihwkit(parameter, group)

    def _identity_for_aihwkit(
        self,
        size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (size, device.type, device.index, dtype)
        identity = self._aihwkit_identity_cache.get(key)
        if identity is None:
            identity = torch.eye(size, device=device, dtype=dtype)
            self._aihwkit_identity_cache[key] = identity
        return identity

    def _step_aihwkit_parameter(
        self,
        parameter: torch.Tensor,
        gradient: torch.Tensor,
        group: dict,
    ) -> None:
        tile = self._aihwkit_tiles[parameter]
        mapping = self._aihwkit_mappings[parameter]
        gradient_matrix = _as_crossbar_matrix(
            gradient,
            group["crossbar_layout"],
            group["dense_input_size"],
        )

        conductance_per_state = (
            mapping["conductance_max"] - mapping["conductance_min"]
        ) / (mapping["state_max"] - mapping["state_min"])
        device_gradient = gradient_matrix / conductance_per_state
        output_size, input_size = device_gradient.shape

        # AIHWKit accepts rank-one update batches. Choose the exact
        # factorization with the smaller batch dimension while preserving
        # d.T @ x == device_gradient.
        if input_size <= output_size:
            x_input = self._identity_for_aihwkit(
                input_size,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            d_input = device_gradient.transpose(0, 1).contiguous()
        else:
            x_input = device_gradient.contiguous()
            d_input = self._identity_for_aihwkit(
                output_size,
                device=parameter.device,
                dtype=parameter.dtype,
            )

        tile.set_learning_rate(float(group["lr"]))
        tile.update(x_input, d_input)
        # AnalogSGD normally performs this hook. It applies configured decay
        # and diffusion once per optimizer/minibatch update.
        tile.post_update_step()

        state = self.state[parameter]
        state["step"] += 1
        transfer_every = self._config.transfer_every
        if transfer_every > 0 and state["step"] % transfer_every == 0:
            state["transfer_count"] += 1
            extra = tile.tile.dump_extra()
            slice_indices = extra.get("rpu.rpu_device.current_slice_indices")
            if slice_indices:
                state["transfer_index"] = int(round(slice_indices[0]))
        self._sync_parameter_from_aihwkit(parameter, group)

    def _initialize_auxiliary_state(self, parameter: torch.Tensor) -> dict:
        state = self.state[parameter]
        if "auxiliary" not in state:
            state["auxiliary"] = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )
            state["step"] = 0
            state["transfer_index"] = 0
            state["transfer_count"] = 0
        return state

    def _transfer_slices(self, parameter: torch.Tensor, group: dict, state: dict) -> None:
        config = self._config
        visible = _as_crossbar_matrix(
            parameter,
            group["crossbar_layout"],
            group["dense_input_size"],
        )
        auxiliary = _as_crossbar_matrix(
            state["auxiliary"],
            group["crossbar_layout"],
            group["dense_input_size"],
        )

        axis = 1 if config.transfer_columns else 0
        axis_size = visible.shape[axis]
        n_reads = min(config.n_reads_per_transfer, axis_size)
        if config.random_selection:
            start = int(torch.randint(axis_size, (), device=parameter.device).item())
        else:
            start = int(state["transfer_index"])
        indices = [(start + offset) % axis_size for offset in range(n_reads)]

        transfer_scale = config.transfer_lr
        if config.scale_transfer_lr:
            transfer_scale *= group["lr"]

        if transfer_scale != 0.0:
            for index in indices:
                if config.transfer_columns:
                    visible[:, index].add_(auxiliary[:, index], alpha=transfer_scale)
                else:
                    visible[index, :].add_(auxiliary[index, :], alpha=transfer_scale)

                if config.with_reset_prob > 0.0:
                    should_reset = config.with_reset_prob == 1.0 or bool(
                        torch.rand((), device=parameter.device) < config.with_reset_prob
                    )
                    if should_reset:
                        if config.transfer_columns:
                            auxiliary[:, index].zero_()
                        else:
                            auxiliary[index, :].zero_()

        state["transfer_index"] = (start + n_reads) % axis_size
        state["transfer_count"] += 1

    @torch.no_grad()
    def step(self, closure=None):
        """Write gradients to auxiliary arrays and perform due transfers."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("TikiTakaOptimizer does not support sparse gradients")

                if not group["use_tiki_taka"]:
                    parameter.add_(gradient, alpha=-group["lr"])
                    continue

                if self.uses_aihwkit:
                    self._step_aihwkit_parameter(
                        parameter,
                        gradient,
                        group,
                    )
                    continue

                state = self._initialize_auxiliary_state(parameter)
                auxiliary = state["auxiliary"]
                fast_learning_rate = self._config.fast_lr
                if fast_learning_rate == 0.0:
                    fast_learning_rate = group["lr"]
                auxiliary.add_(gradient, alpha=-fast_learning_rate)
                if (
                    self._config.fast_weight_min is not None
                    or self._config.fast_weight_max is not None
                ):
                    auxiliary.clamp_(
                        min=self._config.fast_weight_min,
                        max=self._config.fast_weight_max,
                    )

                state["step"] += 1
                transfer_every = self._config.transfer_every
                if transfer_every > 0 and state["step"] % transfer_every == 0:
                    self._transfer_slices(parameter, group, state)

        return loss

    def __str__(self) -> str:
        config = self._config
        direction = "columns" if config.transfer_columns else "rows"
        backend = (
            f"AIHWKit {config.aihwkit_preset}"
            if self.uses_aihwkit
            else "ideal tensor"
        )
        return (
            f"Tiki-Taka ({backend}) -- initial learning rates = "
            f"{self._learning_rates}, fast_lr={config.fast_lr}, "
            f"transfer_every={config.transfer_every}, "
            f"n_reads_per_transfer={config.n_reads_per_transfer}, "
            f"transfer_lr={config.transfer_lr}, direction={direction}"
        )
