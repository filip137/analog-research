"""Numerical composition for RESET-trained single-device MNIST DRNs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from experiments.mnist_relu_drn.components import (
    StudentStack,
    build_student_stack,
    paired_scores,
)
from model.function.interaction import Function
from training.core.sgd import Backprop


class PairedSupervision(Function):
    """Unit-gain supervision over paired DRN output rails."""

    def __init__(self, output_layer: Any, *, objective: str) -> None:
        if objective not in {
            "teacher_kl",
            "cross_entropy",
            "paired_squared_error",
        }:
            raise ValueError(
                "Expected objective to be 'teacher_kl', 'cross_entropy', "
                "or 'paired_squared_error'. "
                f"Provided value: {objective!r}."
            )
        self._layer = output_layer
        self._objective = objective
        self._num_classes = int(output_layer.shape[0]) // 2
        self._teacher_logits: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        Function.__init__(self, [output_layer], [])

    @property
    def objective(self) -> str:
        return self._objective

    @property
    def gain(self) -> float:
        return 1.0

    def set_batch(self, teacher_logits: torch.Tensor, labels: torch.Tensor) -> None:
        if teacher_logits.ndim != 2 or teacher_logits.shape[1] != self._num_classes:
            raise ValueError(
                "Expected teacher logits with shape "
                f"(batch, {self._num_classes}). Provided value: "
                f"{tuple(teacher_logits.shape)!r}."
            )
        if labels.ndim != 1 or labels.shape[0] != teacher_logits.shape[0]:
            raise ValueError(
                "Expected labels with shape (batch,) matching teacher logits. "
                f"Provided value: labels={tuple(labels.shape)!r}, "
                f"teacher={tuple(teacher_logits.shape)!r}."
            )
        self._teacher_logits = teacher_logits.detach()
        self._labels = labels.detach().to(
            device=teacher_logits.device,
            dtype=torch.long,
        )

    def set_target(self, targets: Any) -> None:
        """Engine target hook: ``targets`` carries ``logits`` and ``labels``."""

        self.set_batch(targets.logits, targets.labels)

    def student_logits(self) -> torch.Tensor:
        return paired_scores(self._layer.state)

    def eval(self) -> torch.Tensor:
        if self._teacher_logits is None or self._labels is None:
            raise RuntimeError(
                "Expected teacher logits and labels before evaluating supervision."
            )
        student = self.student_logits()
        if self._objective == "cross_entropy":
            return F.cross_entropy(student, self._labels, reduction="none")
        if self._objective == "paired_squared_error":
            target = F.one_hot(
                self._labels,
                num_classes=self._num_classes,
            ).to(dtype=student.dtype)
            return 0.5 * (student - target).square().sum(dim=1)
        teacher_log_prob = F.log_softmax(self._teacher_logits, dim=1)
        teacher_prob = teacher_log_prob.exp()
        student_log_prob = F.log_softmax(student, dim=1)
        return (
            teacher_prob * (teacher_log_prob - student_log_prob)
        ).sum(dim=1)

    def error_fn(self) -> torch.Tensor:
        if self._labels is None:
            raise RuntimeError("Expected labels before evaluating error.")
        return self.student_logits().argmax(dim=1).ne(self._labels)

    def _get_output(self) -> torch.Tensor:
        return self.student_logits()


def build_reset_student_stack(
    spec: Any,
    *,
    device_data_path: Any,
    enable_measured: bool = True,
) -> StudentStack:
    """Build the original dual-rail topology with reset-training supervision."""

    stack = build_student_stack(
        spec,
        device_data_path=device_data_path,
        enable_measured=enable_measured,
    )
    cost = PairedSupervision(
        stack.bundle.energy.layers()[-1],
        objective=spec.settings.objective,
    )
    differentiator = Backprop(
        stack.bundle.energy.params(),
        list(stack.bundle.energy.layers()[1:]),
        cost,
        stack.training_minimizer,
    )
    return replace(stack, cost=cost, differentiator=differentiator)


__all__ = ["PairedSupervision", "build_reset_student_stack"]
