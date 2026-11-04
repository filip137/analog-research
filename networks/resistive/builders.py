"""Stable model-parameter contracts for resistive-network builders.

The legacy energy functions expose two subtly different parameter views:
``_params`` contains everything that is saved, while ``params()`` contains
parameters that are updated by training.  This module makes those roles
explicit without changing the model's numerical construction.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

import torch

from networks.variable.parameter import Bias, ConvWeight, DenseWeight, PoolWeight


_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")
_SEGMENT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _snake_case(name: str) -> str:
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()


def _as_parameter_tuple(value: Iterable[Any], *, name: str) -> tuple[Any, ...]:
    try:
        parameters = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"Expected {name} to be an iterable of parameter objects. "
            f"Provided value: {value!r}."
        ) from exc
    for index, parameter in enumerate(parameters):
        state = getattr(parameter, "state", None)
        if not isinstance(state, torch.Tensor):
            raise TypeError(
                f"Expected {name}[{index}] to expose a torch.Tensor state. "
                f"Provided value: {parameter!r}."
            )
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError(
            f"Expected {name} to contain each parameter object exactly once. "
            f"Provided value: {parameters!r}."
        )
    return parameters


def _call_parameter_view(model: Any, name: str) -> Optional[tuple[Any, ...]]:
    method = getattr(model, name, None)
    if not callable(method):
        return None
    return _as_parameter_tuple(method(), name=f"{type(model).__name__}.{name}()")


def _all_model_parameters(model: Any) -> tuple[Any, ...]:
    explicit = _call_parameter_view(model, "all_params")
    if explicit is not None:
        return explicit
    for attribute in ("_all_params", "_params"):
        value = getattr(model, attribute, None)
        if value is not None:
            return _as_parameter_tuple(
                value,
                name=f"{type(model).__name__}.{attribute}",
            )
    trainable = _call_parameter_view(model, "params")
    if trainable is not None:
        return trainable
    raise TypeError(
        "Expected model to expose all_params(), _all_params, _params, or "
        f"params(). Provided value: {model!r}."
    )


def _trainable_model_parameters(model: Any) -> tuple[Any, ...]:
    explicit = _call_parameter_view(model, "trainable_params")
    if explicit is not None:
        return explicit
    trainable = _call_parameter_view(model, "params")
    if trainable is not None:
        return trainable
    return _all_model_parameters(model)


def _checkpoint_model_parameters(model: Any) -> tuple[Any, ...]:
    explicit = _call_parameter_view(model, "checkpoint_params")
    if explicit is not None:
        return explicit
    value = getattr(model, "_params", None)
    if value is not None:
        return _as_parameter_tuple(
            value,
            name=f"{type(model).__name__}._params",
        )
    return _all_model_parameters(model)


def _default_role(parameter: Any) -> str:
    declared = getattr(parameter, "checkpoint_role", None)
    if isinstance(declared, str) and declared:
        return declared
    if isinstance(parameter, Bias):
        return "bias"
    if isinstance(parameter, PoolWeight):
        return "pool_weight"
    if isinstance(parameter, ConvWeight):
        return "conv_weight"
    if isinstance(parameter, DenseWeight):
        return "dense_weight"
    return _snake_case(type(parameter).__name__)


@dataclass(frozen=True)
class ParameterBinding:
    """One stable name and role assignment for a mutable model parameter."""

    key: str
    parameter: Any
    group: str = "base"
    role: str = "parameter"
    trainable: bool = True
    checkpointed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not _KEY_PATTERN.fullmatch(self.key):
            raise ValueError(
                "Expected parameter key to contain at least two lowercase "
                "dot-separated identifier segments, for example "
                f"'base.dense_weight.0'. Provided value: {self.key!r}."
            )
        for name, value in (("group", self.group), ("role", self.role)):
            if not isinstance(value, str) or not _SEGMENT_PATTERN.fullmatch(value):
                raise ValueError(
                    f"Expected parameter {name} to be a lowercase identifier. "
                    f"Provided value: {value!r}."
                )
        for name, value in (
            ("trainable", self.trainable),
            ("checkpointed", self.checkpointed),
        ):
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected parameter {name} to be a bool. "
                    f"Provided value: {value!r}."
                )
        state = getattr(self.parameter, "state", None)
        if not isinstance(state, torch.Tensor):
            raise TypeError(
                "Expected bound parameter to expose a torch.Tensor state. "
                f"Provided value: {self.parameter!r}."
            )

    @property
    def state(self) -> torch.Tensor:
        return self.parameter.state

    def descriptor(self) -> dict[str, Any]:
        """Return stable, JSON-compatible structural metadata."""

        return {
            "key": self.key,
            "group": self.group,
            "role": self.role,
            "trainable": self.trainable,
            "checkpointed": self.checkpointed,
            "shape": list(self.state.shape),
            "dtype": str(self.state.dtype),
        }


class ParameterCatalog(Sequence[ParameterBinding]):
    """An ordered, immutable catalog with explicit parameter-role views."""

    def __init__(self, bindings: Iterable[ParameterBinding]):
        resolved = tuple(bindings)
        keys = [binding.key for binding in resolved]
        if len(set(keys)) != len(keys):
            duplicates = sorted({key for key in keys if keys.count(key) > 1})
            raise ValueError(
                "Expected parameter keys to be unique. "
                f"Provided value: duplicate keys {duplicates!r}."
            )
        identities = [id(binding.parameter) for binding in resolved]
        if len(set(identities)) != len(identities):
            raise ValueError(
                "Expected each parameter object to have exactly one binding. "
                f"Provided value: {resolved!r}."
            )
        self._bindings = resolved
        self._by_key: Mapping[str, ParameterBinding] = MappingProxyType(
            {binding.key: binding for binding in resolved}
        )

    def __len__(self) -> int:
        return len(self._bindings)

    def __iter__(self) -> Iterator[ParameterBinding]:
        return iter(self._bindings)

    def __getitem__(self, index):
        return self._bindings[index]

    @property
    def all(self) -> tuple[ParameterBinding, ...]:
        return self._bindings

    @property
    def trainable(self) -> tuple[ParameterBinding, ...]:
        return tuple(binding for binding in self if binding.trainable)

    @property
    def checkpointed(self) -> tuple[ParameterBinding, ...]:
        return tuple(binding for binding in self if binding.checkpointed)

    @property
    def by_key(self) -> Mapping[str, ParameterBinding]:
        return self._by_key

    @property
    def all_parameters(self) -> tuple[Any, ...]:
        return tuple(binding.parameter for binding in self.all)

    @property
    def trainable_parameters(self) -> tuple[Any, ...]:
        return tuple(binding.parameter for binding in self.trainable)

    @property
    def checkpointed_parameters(self) -> tuple[Any, ...]:
        return tuple(binding.parameter for binding in self.checkpointed)

    def for_group(
        self,
        group: str,
        *,
        checkpointed_only: bool = False,
    ) -> tuple[ParameterBinding, ...]:
        if not isinstance(group, str) or not _SEGMENT_PATTERN.fullmatch(group):
            raise ValueError(
                "Expected group to be a lowercase identifier. "
                f"Provided value: {group!r}."
            )
        source = self.checkpointed if checkpointed_only else self.all
        return tuple(binding for binding in source if binding.group == group)

    def descriptors(
        self,
        *,
        checkpointed_only: bool = False,
    ) -> list[dict[str, Any]]:
        source = self.checkpointed if checkpointed_only else self.all
        return [binding.descriptor() for binding in source]

    @classmethod
    def infer(cls, model: Any) -> "ParameterCatalog":
        """Infer stable base-parameter bindings from a legacy model."""

        all_parameters = _all_model_parameters(model)
        all_ids = {id(parameter) for parameter in all_parameters}
        trainable_parameters = _trainable_model_parameters(model)
        checkpoint_parameters = _checkpoint_model_parameters(model)
        for view_name, view in (
            ("trainable", trainable_parameters),
            ("checkpoint", checkpoint_parameters),
        ):
            unknown = [
                parameter for parameter in view if id(parameter) not in all_ids
            ]
            if unknown:
                raise ValueError(
                    f"Expected every {view_name} parameter to appear in the "
                    f"all-parameter view. Provided value: {unknown!r}."
                )

        trainable_ids = {id(parameter) for parameter in trainable_parameters}
        checkpoint_ids = {id(parameter) for parameter in checkpoint_parameters}
        counters: dict[tuple[str, str], int] = {}
        bindings = []
        for parameter in all_parameters:
            group = getattr(parameter, "checkpoint_group", "base")
            role = _default_role(parameter)
            if not isinstance(group, str):
                raise ValueError(
                    "Expected checkpoint_group to be a lowercase identifier. "
                    f"Provided value: {group!r}."
                )
            counter_key = (group, role)
            index = counters.get(counter_key, 0)
            counters[counter_key] = index + 1
            declared_key = getattr(parameter, "checkpoint_key", None)
            key = declared_key or f"{group}.{role}.{index}"
            bindings.append(
                ParameterBinding(
                    key=key,
                    parameter=parameter,
                    group=group,
                    role=role,
                    trainable=id(parameter) in trainable_ids,
                    checkpointed=id(parameter) in checkpoint_ids,
                )
            )
        return cls(bindings)


@dataclass(frozen=True)
class ModelBundle:
    """A model paired with the explicit catalog used by runtime tooling."""

    energy: Any
    parameters: ParameterCatalog

    def __post_init__(self) -> None:
        expected = {id(parameter) for parameter in _all_model_parameters(self.energy)}
        provided = {id(parameter) for parameter in self.parameters.all_parameters}
        if expected != provided:
            raise ValueError(
                "Expected model bundle catalog to bind every model parameter "
                "exactly once. "
                f"Provided value: missing={len(expected - provided)}, "
                f"unexpected={len(provided - expected)}."
            )

    @property
    def model(self) -> Any:
        return self.energy

    @property
    def catalog(self) -> ParameterCatalog:
        return self.parameters

    @classmethod
    def wrap(
        cls,
        energy: Any,
        *,
        catalog: Optional[ParameterCatalog] = None,
    ) -> "ModelBundle":
        return cls(
            energy=energy,
            parameters=catalog or ParameterCatalog.infer(energy),
        )


def build_deep_resistive_energy(
    *args,
    catalog_factory: Optional[Callable[[Any], ParameterCatalog]] = None,
    **kwargs,
) -> ModelBundle:
    """Build the current DRN unchanged and attach an explicit catalog."""

    from networks.resistive.network import DeepResistiveEnergy

    energy = DeepResistiveEnergy(*args, **kwargs)
    catalog = (
        catalog_factory(energy)
        if catalog_factory is not None
        else ParameterCatalog.infer(energy)
    )
    return ModelBundle.wrap(energy, catalog=catalog)


__all__ = [
    "ModelBundle",
    "ParameterBinding",
    "ParameterCatalog",
    "build_deep_resistive_energy",
]
