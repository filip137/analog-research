"""Engine event handlers that stop a training step before a bad update."""

import torch

from training.core.engine import GradientsReadyEvent


class FiniteGradientGuard:
    """Reject invalid gradients before an engine reaches the update backend.

    With ``names`` (one per parameter, e.g. catalog keys) the error names the
    parameter and batch; ``label`` says which pass computed the gradients.
    Register it before observers that react to gradient values.
    """

    def __init__(self, names=None, *, label="computed"):
        self.names = None if names is None else tuple(names)
        self.label = label

    def __call__(self, event):
        if not isinstance(event, GradientsReadyEvent):
            return
        for index, gradient in enumerate(event.gradients):
            if not isinstance(gradient, torch.Tensor) or not torch.isfinite(gradient).all():
                if self.names is None:
                    raise FloatingPointError(
                        "Expected every computed gradient to be a finite tensor. "
                        f"Provided value: gradient index {index}, type={type(gradient).__name__}."
                    )
                raise FloatingPointError(
                    f"Expected finite {self.label} gradients. Provided value: "
                    f"parameter={self.names[index]!r}, batch={event.batch_index}."
                )
