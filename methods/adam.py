"""Adam optimizer adapter for the legacy resistive-network parameters."""

from __future__ import annotations

import torch

from networks.variable.parameter import PoolWeight


class AdamOptimizer(torch.optim.Adam):
    """Apply ordinary digital Adam to the trainable DRN parameter tensors."""

    def __init__(
        self,
        energy_fn,
        cost_fn,
        learning_rates,
        *,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        amsgrad: bool,
    ) -> None:
        rates = tuple(float(value) for value in learning_rates)
        parameters = [
            parameter
            for parameter in energy_fn.params() + cost_fn.params()
            if not isinstance(parameter, PoolWeight)
        ]
        if len(rates) != len(parameters):
            raise ValueError(
                "learning_rates length "
                f"({len(rates)}) does not match parameter count "
                f"({len(parameters)} after filtering PoolWeight)"
            )
        groups = [
            {"params": parameter.state, "lr": rate}
            for parameter, rate in zip(parameters, rates, strict=True)
        ]
        super().__init__(
            groups,
            lr=1.0,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self._learning_rates = rates
        self._betas = betas
        self._eps = float(eps)
        self._weight_decay = float(weight_decay)
        self._amsgrad = bool(amsgrad)

    def __str__(self) -> str:
        return (
            "Adam -- initial learning rates = "
            f"{self._learning_rates}, betas={self._betas}, "
            f"eps={self._eps}, weight_decay={self._weight_decay}, "
            f"amsgrad={self._amsgrad}"
        )
