"""Engine event handlers that stop a training step before a bad update."""

import torch

from training.core.engine import GradientsReadyEvent


class FiniteGradientGuard:
    """Reject invalid gradients before an engine reaches the update backend."""

    def __call__(self, event):
        if not isinstance(event, GradientsReadyEvent):
            return
        for index, gradient in enumerate(event.gradients):
            if not isinstance(gradient, torch.Tensor) or not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    "Expected every computed gradient to be a finite tensor. "
                    f"Provided value: gradient index {index}, type={type(gradient).__name__}."
                )
