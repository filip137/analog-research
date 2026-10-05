"""The shared train-phase lifecycle.

Every train phase runs the same epoch loop::

    for each epoch:
        train one epoch through training.core.engine
        validate
        offer the validation to the global-best selector
            (on improvement: encode the selected payload, write weights.pt)
        write the epoch-boundary resume.pt
        append the epoch metric when the log cadence says so
        notify epoch observers
    ensure a resume.pt exists even when no epoch ran

The order is part of the contract: the resume checkpoint records the global
and loader RNG states after validation, so any RNG-consuming step moved
across it changes every later epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Mapping, Optional, TypeVar

from experiments.artifacts import ArtifactRecord, RunStore
from training.checkpoint import save_encoded_named_weights


V = TypeVar("V")


class BestCheckpoint:
    """Global-best selection whose payload is materialized as ``weights.pt``.

    ``better(candidate, incumbent)`` compares two selection records and must
    be strict, so ties keep the earlier epoch.  Without an incumbent every
    candidate is selected.
    """

    def __init__(
        self,
        weights_path: Path,
        catalog: Any,
        *,
        better: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
    ) -> None:
        self.weights_path = weights_path
        self.catalog = catalog
        self._better = better
        self.epoch: Optional[int] = None
        self.selection: Optional[Mapping[str, Any]] = None
        self.payload: Optional[Mapping[str, Any]] = None

    def _adopt(
        self,
        epoch: int,
        selection: Mapping[str, Any],
        payload: Optional[Mapping[str, Any]],
    ) -> None:
        self.epoch = epoch
        self.selection = selection
        self.payload = payload
        if payload is not None:
            save_encoded_named_weights(
                self.weights_path, payload, catalog=self.catalog
            )

    def restore(
        self,
        *,
        epoch: int,
        selection: Mapping[str, Any],
        payload: Optional[Mapping[str, Any]],
    ) -> None:
        """Adopt the selection stored in a resume checkpoint."""

        self._adopt(epoch, selection, payload)

    def seed(
        self,
        *,
        epoch: int,
        selection: Mapping[str, Any],
        payload: Callable[[], Mapping[str, Any]],
    ) -> None:
        """Make an unselected run start from this candidate (e.g. epoch -1)."""

        if self.selection is None:
            self._adopt(epoch, selection, payload())

    def offer(
        self,
        *,
        epoch: int,
        selection: Mapping[str, Any],
        payload: Callable[[], Mapping[str, Any]],
    ) -> bool:
        """Select ``selection`` if it strictly improves; return whether it did."""

        improved = self.selection is None or self._better(selection, self.selection)
        if improved:
            self._adopt(epoch, selection, payload())
        return improved

    def require(self, message: str) -> Mapping[str, Any]:
        if self.payload is None or self.selection is None or self.epoch is None:
            raise RuntimeError(message)
        return self.payload


def lower(metric: str) -> Callable[[Mapping[str, Any], Mapping[str, Any]], bool]:
    """Strictly lower ``metric`` is better."""

    return lambda candidate, incumbent: candidate[metric] < incumbent[metric]


@dataclass(frozen=True)
class EpochResult(Generic[V]):
    epoch: int
    completed_epochs: int
    global_step: int
    train: Any
    validation: V
    improved: bool
    logged: bool


def run_epochs(
    *,
    store: RunStore,
    start_epoch: int,
    num_epochs: int,
    global_step: int,
    log_every: int,
    train_epoch: Callable[[int, int], tuple[Any, int]],
    validate: Callable[[int], V],
    best: BestCheckpoint,
    selection: Callable[[V], Mapping[str, Any]],
    payload: Callable[[int, V], Mapping[str, Any]],
    save_resume: Callable[[int, int], Any],
    resume_path: Path,
    record: Callable[[EpochResult[V]], Mapping[str, Any]],
    after_epoch: Iterable[Callable[[EpochResult[V]], Any]] = (),
    missing_selection: str = "Expected a selected checkpoint at every epoch boundary.",
) -> tuple[Optional[EpochResult[V]], int, int]:
    """Run epochs ``start_epoch .. num_epochs - 1`` of one train phase.

    ``train_epoch(epoch, global_step)`` returns the epoch's train metrics and
    the next global step.  ``save_resume(completed_epochs, global_step)``
    writes the epoch-boundary checkpoint from the caller's current state and
    ``best``.  Returns the last epoch's result (``None`` when no epoch ran),
    the completed-epoch count and the global step.
    """

    observers = tuple(after_epoch)
    last: Optional[EpochResult[V]] = None
    completed = start_epoch
    for epoch in range(start_epoch, num_epochs):
        trained, global_step = train_epoch(epoch, global_step)
        validation = validate(epoch)
        improved = best.offer(
            epoch=epoch,
            selection=selection(validation),
            payload=lambda: payload(epoch, validation),
        )
        best.require(missing_selection)
        completed = epoch + 1
        save_resume(completed, global_step)
        logged = completed % log_every == 0 or completed == num_epochs
        last = EpochResult(
            epoch=epoch,
            completed_epochs=completed,
            global_step=global_step,
            train=trained,
            validation=validation,
            improved=improved,
            logged=logged,
        )
        if logged:
            store.append_metric(record(last))
        for observer in observers:
            observer(last)
    best.require(missing_selection)
    # A continuation whose target epoch is already complete still owns a
    # fresh, self-contained resume artifact.
    if not resume_path.exists():
        save_resume(completed, global_step)
    return last, completed, global_step


def run_phase(
    store: RunStore,
    body: Callable[[], tuple[Mapping[str, Any], Iterable[ArtifactRecord]]],
) -> Path:
    """Run ``body`` and record its terminal state in ``store``.

    ``body`` returns the result metrics and artifact records; any exception,
    including an interrupt, marks the run failed and is re-raised unchanged.
    """

    try:
        metrics, artifacts = body()
        return store.complete(metrics=metrics, artifacts=artifacts)
    except BaseException as error:
        store.fail(error)
        raise
