"""Gradient estimator for parameters that occur only in a digital readout."""

from __future__ import annotations


class DirectReadoutGradient:
    """Differentiate the settled-batch cost directly with respect to readout parameters."""

    def __init__(self, cost_fn) -> None:
        self._cost_fn = cost_fn

    def compute_gradient(self):
        return [
            self._cost_fn._grad(parameter, mean=True)
            for parameter in self._cost_fn.params()
        ]

    def __str__(self) -> str:
        return "Direct digital-readout gradient"
