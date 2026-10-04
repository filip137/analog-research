"""Parameter-modifier seam used by the generic training runtime."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, ContextManager, Optional, Protocol, runtime_checkable


@runtime_checkable
class ParameterModifier(Protocol):
    """Temporarily alter parameters for a complete training or eval phase.

    The method names intentionally match the existing hardware-aware modifier,
    but this protocol contains no hardware-specific policy.
    """

    def training_context(self) -> ContextManager[Any]:
        """Return one context for a complete training minibatch."""

    def evaluation_context(self) -> ContextManager[Any]:
        """Return one context that may span a complete evaluation loader."""

    def state_dict(self) -> dict:
        """Return modifier state required for continuation."""

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Restore modifier continuation state."""


class NoOpParameterModifier:
    """A stateless modifier that leaves parameters untouched."""

    def training_context(self) -> ContextManager["NoOpParameterModifier"]:
        return nullcontext(self)

    def evaluation_context(self) -> ContextManager["NoOpParameterModifier"]:
        return nullcontext(self)

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: Mapping) -> None:
        if not isinstance(state_dict, Mapping) or state_dict:
            raise ValueError(
                "Expected the no-op parameter modifier state to be an empty "
                f"mapping. Provided value: {state_dict!r}."
            )


NO_OP_PARAMETER_MODIFIER = NoOpParameterModifier()


class SplitParameterModifier:
    """Use independent modifiers for minibatch training and evaluation.

    This keeps a clean master parameter tensor while allowing, for example,
    compact endpoint sampling during BPTT and a pulse-resolved deployment for
    model selection. Both delegates are checkpointed as one modifier so an
    epoch-boundary resume preserves their independent random streams.
    """

    def __init__(
        self,
        *,
        training: Optional[ParameterModifier],
        evaluation: Optional[ParameterModifier],
    ) -> None:
        self.training = modifier_or_default(training)
        self.evaluation = modifier_or_default(evaluation)

    def training_context(self) -> ContextManager[Any]:
        return self.training.training_context()

    def evaluation_context(self) -> ContextManager[Any]:
        return self.evaluation.evaluation_context()

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "training": self.training.state_dict(),
            "evaluation": self.evaluation.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        expected = {"version", "training", "evaluation"}
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected:
            raise ValueError(
                "Expected split parameter-modifier state with exact keys "
                f"{sorted(expected)!r}. Provided value: {state_dict!r}."
            )
        if state_dict["version"] != 1:
            raise ValueError(
                "Expected split parameter-modifier state version 1. "
                f"Provided value: {state_dict['version']!r}."
            )
        self.training.load_state_dict(state_dict["training"])
        self.evaluation.load_state_dict(state_dict["evaluation"])


def modifier_or_default(
    modifier: Optional[ParameterModifier],
) -> ParameterModifier:
    """Resolve ``None`` to the shared stateless no-op modifier."""

    if modifier is None:
        return NO_OP_PARAMETER_MODIFIER
    return modifier
