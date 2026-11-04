from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn.functional as F
from networks.function.interaction import Function
from networks.resistive.mnist import paired_scores


class TeacherKLDivergence(Function):
    """Pure KL(teacher || student) over calibrated paired DRN scores."""

    def __init__(self, output_layer, *, gain: float = 1.0) -> None:
        self._layer = output_layer
        self._num_classes = int(output_layer.shape[0]) // 2
        self._gain = float(gain)
        self._teacher_logits: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        Function.__init__(self, [output_layer], [])

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(
                "Expected fixed readout gain to be positive and finite. "
                f"Provided value: {value!r}."
            )
        self._gain = numeric

    def set_teacher(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.ndim != 2 or logits.shape[1] != self._num_classes:
            raise ValueError(
                "Expected teacher logits with shape "
                f"(batch, {self._num_classes}). "
                f"Provided value: {tuple(logits.shape)!r}."
            )
        self._teacher_logits = logits.detach()
        self._labels = labels.detach().to(device=logits.device, dtype=torch.long)

    def set_target(self, targets: Any) -> None:
        """Engine target hook: ``targets`` carries ``logits`` and ``labels``."""

        self.set_teacher(targets.logits, targets.labels)

    def student_logits(self) -> torch.Tensor:
        return paired_scores(self._layer.state) * self._gain

    def eval(self) -> torch.Tensor:
        if self._teacher_logits is None:
            raise RuntimeError("Expected teacher logits before evaluating KL.")
        teacher_log_prob = F.log_softmax(self._teacher_logits, dim=1)
        teacher_prob = teacher_log_prob.exp()
        student_log_prob = F.log_softmax(self.student_logits(), dim=1)
        return (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=1)

    def error_fn(self) -> torch.Tensor:
        if self._labels is None:
            raise RuntimeError("Expected labels before evaluating error.")
        return self.student_logits().argmax(dim=1).ne(self._labels)

    def _get_output(self) -> torch.Tensor:
        return self.student_logits()
