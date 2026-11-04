"""Passive dense conductance factors for low-rank DRN adapters."""

from __future__ import annotations

import torch

from networks.resistive.low_rank_config import (
    PassiveLowRankAdapterConfig,
    parse_passive_low_rank_adapter,
)
from networks.variable.parameter import DenseWeight


_INPUT_FACTOR_ROLE = "input_factor"
_OUTPUT_FACTOR_ROLE = "output_factor"
_FACTOR_ROLES = (_INPUT_FACTOR_ROLE, _OUTPUT_FACTOR_ROLE)


class PassiveLowRankDenseWeight(DenseWeight):
    """A passive factor retaining the normal dense-crossbar tensor layout."""

    checkpoint_group = "adapter"

    def __init__(
        self,
        layer_pre_shape,
        layer_post_shape,
        *,
        role: str,
        gain: float,
        conductance_min: float,
        conductance_max: float,
        initial_value: float | None = None,
    ) -> None:
        if role not in _FACTOR_ROLES:
            raise ValueError(
                "Expected passive low-rank weight role to be 'input_factor' "
                f"or 'output_factor'. Provided value: {role!r}."
            )

        self.adapter_role = role
        self.role = role
        self.checkpoint_role = role
        super().__init__(
            layer_pre_shape,
            layer_post_shape,
            gain,
            device=None,
            clamp=True,
            clamp_min=conductance_min,
            clamp_max=conductance_max,
            init_mode="kaiming_uniform",
        )
        if initial_value is not None:
            with torch.no_grad():
                self._state.fill_(initial_value)

        dense_index = self.name.rsplit("_", 1)[-1]
        role_name = (
            "InputFactor"
            if role == _INPUT_FACTOR_ROLE
            else "OutputFactor"
        )
        self.name = f"PassiveLowRank{role_name}Weight_{dense_index}"


__all__ = [
    "PassiveLowRankAdapterConfig",
    "PassiveLowRankDenseWeight",
    "parse_passive_low_rank_adapter",
]
