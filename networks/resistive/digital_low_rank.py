"""Ideal signed low-rank residual applied after a DRN equilibrium solve."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from networks.function.cost import CostFunction
from networks.function.interaction import Function
from networks.resistive.digital_low_rank_config import (
    DigitalLowRankAdapterConfig,
    Wan2022ProgrammingConfig,
    parse_digital_low_rank_adapter,
)
from networks.variable.parameter import Parameter


_ROLES = ("input_factor", "output_factor")


class DigitalLowRankWeight(Parameter):
    """An unbounded signed FP32 factor with stable adapter checkpoint roles."""

    checkpoint_group = "adapter"

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        role: str,
        gain: float,
        zero: bool,
        device=None,
    ) -> None:
        if role not in _ROLES:
            raise ValueError(
                "Expected digital low-rank weight role to be 'input_factor' "
                f"or 'output_factor'. Provided value: {role!r}."
            )
        super().__init__(
            shape,
            device=device,
            non_negative=False,
            min_cond=None,
            max_cond=None,
        )
        self.adapter_role = role
        self.role = role
        self.checkpoint_role = role
        self.init_state(gain=gain, zero=zero)
        suffix = "InputFactor" if role == "input_factor" else "OutputFactor"
        self.name = f"DigitalLowRank{suffix}Weight"

    def init_state(self, gain: float, zero: bool = False) -> None:
        """Initialize one factor without imposing conductance constraints."""

        if zero:
            torch.nn.init.zeros_(self._state)
        else:
            bound = float(gain) / math.sqrt(self.shape[0])
            torch.nn.init.uniform_(self._state, -bound, bound)


class DigitalLowRankReadout(CostFunction, Function):
    """Squared-error classifier with an ideal digital ``x A B`` residual."""

    def __init__(
        self,
        input_layer,
        output_layer,
        *,
        input_factor: DigitalLowRankWeight,
        output_factor: DigitalLowRankWeight,
        logical_input_dim: int,
        num_classes: int,
        input_gain: float,
        alpha: float,
    ) -> None:
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._input_factor = input_factor
        self._output_factor = output_factor
        self._logical_input_dim = int(logical_input_dim)
        self._num_classes = int(num_classes)
        self._input_gain = float(input_gain)
        self._scale = float(alpha) / input_factor.shape[1]
        self._label = None
        self._target = None
        Function.__init__(
            self,
            [input_layer, output_layer],
            [input_factor, output_factor],
        )

    def set_target(self, label) -> None:
        output = self._output_layer.state
        self._label = label.to(device=output.device, dtype=torch.long)
        self._target = F.one_hot(
            self._label,
            num_classes=self._num_classes,
        ).to(dtype=output.dtype)

    def _logical_inputs(self) -> torch.Tensor:
        return (
            self._input_layer.state[:, : self._logical_input_dim]
            / self._input_gain
        )

    def _base_scores(self) -> torch.Tensor:
        output = self._output_layer.state
        expected = 2 * self._num_classes
        if output.ndim != 2 or output.shape[1] != expected:
            raise ValueError(
                "Expected digital_low_rank output state to have shape "
                f"(batch, {expected}) for differential class scores. "
                f"Provided value: shape={tuple(output.shape)!r}."
            )
        paired = output.reshape(output.shape[0], self._num_classes, 2)
        return paired[..., 0] - paired[..., 1]

    def scores(self) -> torch.Tensor:
        residual = (
            self._logical_inputs()
            .matmul(self._input_factor.state)
            .matmul(self._output_factor.state)
        )
        return self._base_scores() + self._scale * residual

    def _target_vector(self) -> torch.Tensor:
        if self._target is None:
            raise RuntimeError(
                "DigitalLowRankReadout targets are not set. "
                "Call set_target(label=...)."
            )
        return self._target

    def eval(self) -> torch.Tensor:
        difference = self.scores() - self._target_vector()
        return 0.5 * difference.square().sum(dim=1)

    def error_fn(self) -> torch.Tensor:
        if self._label is None:
            raise RuntimeError(
                "DigitalLowRankReadout labels are not set. "
                "Call set_target(label=...)."
            )
        return torch.argmax(self.scores(), dim=1).ne(self._label)

    def top_five_error_fn(self) -> torch.Tensor:
        if self._label is None:
            raise RuntimeError(
                "DigitalLowRankReadout labels are not set. "
                "Call set_target(label=...)."
            )
        indices = torch.topk(
            self.scores(),
            k=min(5, self._num_classes),
            dim=1,
        ).indices
        return ~(indices == self._label.reshape(-1, 1)).any(dim=1)

    def _get_output(self) -> torch.Tensor:
        return self.scores()


__all__ = [
    "DigitalLowRankAdapterConfig",
    "DigitalLowRankReadout",
    "DigitalLowRankWeight",
    "Wan2022ProgrammingConfig",
    "parse_digital_low_rank_adapter",
]
