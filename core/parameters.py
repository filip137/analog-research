"""Tensor parameters and explicit, stable checkpoint bindings for crossbar workflows."""

from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from abc import ABC
import numpy as np
import torch
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


class Variable(ABC):
    """
    Abstract class for variables (layers and parameters).

    Attributes
    ----------
    shape (tuple of int): shape of the tensor used to represent the state of the variable
    state (Tensor): the state of the variable

    Methods
    -------
    init_state()
        Initializes the state of the variable
    set_device(device)
        Set the variable's Tensor state on a given device
    to(device)
        Create a copy of the variable on a given device
    """

    def __init__(self, shape):
        """Initializes an instance of Variable

        Args:
            shape (tuple of int): shape of the tensor used to represent the variable's state
        """

        self._shape = shape

        # TODO: should one initialize the variable in the Variable construcor?

    @property
    def shape(self):
        """Gets the shape of the variable (the tensor)"""

        return self._shape

    @property
    def state(self):
        """Gets and sets the current state of the variable"""

        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def set_device(self, device):
        """Set the variable's Tensor state on a given device

        Args:
            device (str): The device, e.g. 'cpu' or 'cuda'.
        """

        self._state = self._state.to(device)

    def to(self, device):
        """Create a copy of the variable on a given device

        Args:
            device (str): The device, e.g. 'cpu' or 'cuda'.
        """
        variable = copy.deepcopy(self)
        variable._state = variable._state.to(device)
        return variable

    @abstractmethod
    def init_state(self):
        """Initializes the variable"""
        pass


class Parameter(Variable, ABC):
    """Abstract class for parameter variables

    Attributes
    ----------
    _non_negative (bool): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False)

    Methods
    -------
    get()):
        Returns the state of the parameter
    clamp_():
        Clamps the parameter's state in its range of permissible values (in place operation)
    """

    def __init__(self, shape, device, non_negative = True, min_cond = None, max_cond = None):
        """Initializes an instance of Parameter

        Args:
            shape (tuple of ints): Shape of the tensor used to represent the parameter. Type is float32.
            device (str): Either 'cpu' or 'cuda'.
            non_negative (bool, optional): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False). Default: False
            min_cond (float, optional): Lower clamp bound for the parameter's state. Defaults to 0 for non-negative parameters, otherwise no clamp.
            max_cond (float, optional): Upper clamp bound for the parameter's state. Defaults to inf for non-negative parameters, otherwise no clamp.
        """

        Variable.__init__(self, shape)

        self._state = torch.empty(*shape, dtype=torch.float32, device=device)
        self._non_negative = non_negative

        self.min_cond = min_cond
        self.max_cond = max_cond

    def get(self):
        """Returns the state of the parameter"""
        return self._state

    def clamp_(self):
        """Clamps the parameter's state in its range of permissible values (in place operation)"""
        clamp_min = self.min_cond
        clamp_max = self.max_cond

        if self._non_negative:
            if clamp_min is None:
                clamp_min = 0.0
            if clamp_max is None:
                clamp_max = float('inf')

        if clamp_min is not None or clamp_max is not None:
            self._state.clamp_(min=clamp_min, max=clamp_max)


class DenseWeight(Parameter):
    """Class for dense ('fully connected') weights

    Methods
    -------
    init_state(gain, mode):
        Initializes the weight tensor
    """

    _counter = 0

    def __init__(self, layer_pre_shape, layer_post_shape, gain, device, clamp=False,
                 clamp_min=None, clamp_max=None, init_mode='kaiming_uniform'):
        """Initializes an instance of DenseWeight

        Args:
            layer_pre_shape (tuple of ints): shape of the pre-synaptic layer
            layer_post_shape (tuple of ints): shape of the post-synaptic layer
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            clamp (bool, optional): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False). Default: False
            clamp_min (float, optional): Lower clamp bound applied when clamp=True. Defaults to 0.
            clamp_max (float, optional): Upper clamp bound applied when clamp=True. Defaults to 100e-6.
        """

        shape = layer_pre_shape + layer_post_shape
        Parameter.__init__(self, shape, device=device, non_negative=clamp,
                           min_cond=clamp_min, max_cond=clamp_max)

        self._layer_pre_shape = layer_pre_shape
        self._layer_post_shape = layer_post_shape

        self.init_state(gain, mode=init_mode)
        self.clamp_()

        self.name = 'DenseWeight_{}'.format(DenseWeight._counter)

        DenseWeight._counter += 1

    def init_state(self, gain, mode='kaiming_uniform'):
        """Initializes the weight tensor according to a uniform or normal distribution.
        Args:
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            mode (str, optional): method to initialize the weight tensor. Either 'xavier_uniform', 'xavier_normal', 'kaiming_uniform' or 'kaiming_normal'. Default: 'xavier_uniform'.
        """

        size_pre = 1
        for dim in self._layer_pre_shape: size_pre *= dim
        size_post = 1
        for dim in self._layer_post_shape: size_post *= dim

        if mode == 'xavier_uniform':
            # half xavier uniform
            scale = gain * 0.5 * np.sqrt(6. / (size_pre + size_post))
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif mode == 'xavier_normal':
            # half xavier normal
            scale = gain * 0.5 * np.sqrt(2. / (size_pre + size_post))
            torch.nn.init.normal_(self._state, std=scale)
        elif mode == 'kaiming_uniform':
            # half kaiming uniform
            # scale = gain * 0.5 * np.sqrt(3. / size_pre)
            scale = gain * np.sqrt(1. / size_pre)
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif mode == 'bounded_uniform':
            # direct range [0, gain]
            torch.nn.init.uniform_(self._state, 0.0, self.max_cond)
        elif mode == 'bounded_range_uniform':
            if self.min_cond is None or self.max_cond is None:
                raise ValueError(
                    "Expected bounded_range_uniform initialization to have "
                    "finite min_cond and max_cond bounds. Provided value: "
                    f"min_cond={self.min_cond!r}, max_cond={self.max_cond!r}."
                )
            torch.nn.init.uniform_(
                self._state,
                float(self.min_cond),
                float(self.max_cond),
            )
        elif mode == 'floor_shifted_kaiming_uniform':
            if self.min_cond is None or self.max_cond is None:
                raise ValueError(
                    "Expected floor_shifted_kaiming_uniform initialization "
                    "to have finite min_cond and max_cond bounds. Provided "
                    f"value: min_cond={self.min_cond!r}, "
                    f"max_cond={self.max_cond!r}."
                )
            scale = gain * np.sqrt(1. / size_pre)
            torch.nn.init.uniform_(self._state, -scale, +scale)
            self._state.clamp_(min=0.0)
            self._state.add_(float(self.min_cond))
        elif mode == 'Kendall':
            lower = 1e-7
            upper = 0.08 / np.sqrt(size_pre + size_post)
            torch.nn.init.uniform_(self._state, lower, upper)
        else:  #  mode == 'kaiming_normal'
            # half kaiming normal
            scale = gain * 0.5 * np.sqrt(1. / size_pre)
            torch.nn.init.normal_(self._state, std=scale)


_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")


_SEGMENT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


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
