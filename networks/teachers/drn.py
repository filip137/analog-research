from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
import torch
from networks.resistive.builders import ParameterCatalog
from networks.resistive.mnist import paired_scores


@dataclass(frozen=True)
class BoundedDrnTeacher:
    """Frozen equilibrium teacher loaded from stable named DRN weights."""

    stack: Any
    checkpoint_metadata: dict[str, Any]

    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.stack.network.set_input(
                inputs.to(self.stack.device, dtype=torch.float32),
                reset=True,
            )
            self.stack.minimizer.compute_equilibrium()
            return paired_scores(
                self.stack.bundle.energy.layers()[-1].state
            ).detach()

    def copy_named_weights_to(self, catalog: ParameterCatalog) -> None:
        source = self.stack.bundle.catalog
        expected = tuple(binding.key for binding in catalog.trainable)
        provided = tuple(binding.key for binding in source.trainable)
        if provided != expected:
            raise ValueError(
                "Expected bounded DRN teacher and student to expose the "
                "same stable trainable parameter keys. Provided value: "
                f"teacher={provided!r}, student={expected!r}."
            )
        with torch.no_grad():
            for key in expected:
                teacher_state = source.by_key[key].state
                student_state = catalog.by_key[key].state
                if teacher_state.shape != student_state.shape:
                    raise ValueError(
                        "Expected bounded DRN teacher and student parameter "
                        f"{key!r} to have the same shape. Provided value: "
                        f"teacher={tuple(teacher_state.shape)!r}, "
                        f"student={tuple(student_state.shape)!r}."
                    )
                student_state.copy_(teacher_state.to(student_state))
