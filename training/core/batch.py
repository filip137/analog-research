"""Canonical minibatch representation for the experiment runtime.

Legacy loaders in this repository yield either ``(inputs, targets)`` pairs or
``(inputs, targets, indices)`` triples.  Converting both forms at the runtime
boundary keeps training, evaluation, and probes independent of the loader
implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar, Union


InputT = TypeVar("InputT")
TargetT = TypeVar("TargetT")
IndexT = TypeVar("IndexT")


@dataclass(frozen=True)
class Batch(Generic[InputT, TargetT, IndexT]):
    """A minibatch with optional source-dataset indices."""

    inputs: InputT
    targets: TargetT
    indices: Optional[IndexT] = None

    @property
    def example_count(self) -> int:
        """Return the number of examples represented by this minibatch.

        Dataset indices are the least ambiguous source when present.  Targets
        are preferred otherwise because an input can itself be a sequence of
        model arguments.
        """

        for value in (self.indices, self.targets, self.inputs):
            size = _leading_size(value)
            if size is not None:
                return size
        return 1

    @classmethod
    def from_raw(cls, raw_batch: Any) -> "Batch[Any, Any, Any]":
        """Adapt a pair, triple, or existing :class:`Batch`."""

        return as_batch(raw_batch)


RawBatch = Union[Batch[Any, Any, Any], Sequence[Any]]


def as_batch(raw_batch: RawBatch) -> Batch[Any, Any, Any]:
    """Return the canonical representation of a loader-produced minibatch."""

    if isinstance(raw_batch, Batch):
        return raw_batch

    if not isinstance(raw_batch, Sequence) or isinstance(
        raw_batch, (str, bytes, bytearray)
    ):
        raise ValueError(
            "Expected a batch to be Batch, (inputs, targets), or "
            "(inputs, targets, indices). "
            f"Provided value: {raw_batch!r}."
        )

    if len(raw_batch) == 2:
        inputs, targets = raw_batch
        return Batch(inputs=inputs, targets=targets)
    if len(raw_batch) == 3:
        inputs, targets, indices = raw_batch
        return Batch(inputs=inputs, targets=targets, indices=indices)

    raise ValueError(
        "Expected a batch to contain exactly two or three items: "
        "(inputs, targets) or (inputs, targets, indices). "
        f"Provided value: {raw_batch!r}."
    )


def _leading_size(value: Any) -> Optional[int]:
    if value is None:
        return None

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            if len(shape) > 0:
                return int(shape[0])
        except (TypeError, ValueError):
            pass

    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        return len(value)
    except TypeError:
        return None
