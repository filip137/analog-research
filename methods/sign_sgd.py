"""Sign-only stochastic-gradient descent.

The optimizer discards gradient magnitude and applies one fixed parameter-unit
step per nonzero coordinate::

    parameter <- parameter - learning_rate * sign(gradient)

Momentum and weight decay are deliberately absent so the backend has exactly
the signSGD semantics advertised by the experiment capability matrix.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch


class SignSGD(torch.optim.Optimizer):
    """Apply fixed-magnitude descent steps using only gradient signs."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float,
    ) -> None:
        if isinstance(lr, bool):
            raise ValueError(
                "Expected SignSGD learning rate to be finite and "
                f"non-negative. Provided value: {lr!r}."
            )
        numeric = float(lr)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                "Expected SignSGD learning rate to be finite and "
                f"non-negative. Provided value: {lr!r}."
            )
        super().__init__(params, defaults={"lr": numeric})
        for index, group in enumerate(self.param_groups):
            if isinstance(group["lr"], bool):
                raise ValueError(
                    "Expected every SignSGD parameter-group learning rate "
                    "to be finite and non-negative. Provided value: "
                    f"group={index}, learning_rate={group['lr']!r}."
                )
            group_lr = float(group["lr"])
            if not math.isfinite(group_lr) or group_lr < 0.0:
                raise ValueError(
                    "Expected every SignSGD parameter-group learning rate "
                    "to be finite and non-negative. Provided value: "
                    f"group={index}, learning_rate={group['lr']!r}."
                )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError(
                "Expected SignSGD updates without an optimizer closure. "
                f"Provided value: {closure!r}."
            )
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise ValueError(
                        "Expected SignSGD gradients to be dense. Provided "
                        "value: sparse gradient."
                    )
                parameter.add_(gradient.sign(), alpha=-learning_rate)
        return None


__all__ = ["SignSGD"]
