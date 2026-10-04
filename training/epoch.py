"""Loader and statistic bindings for the shared numerical engine."""

from training.engine import (
    AfterUpdateEvent,
    EvaluationComponents,
    ExperimentComponents,
    FreePhaseEvent,
    GradientsReadyEvent,
    evaluate,
    train_epoch,
)


class Epoch:
    """Own named statistic collections, without implementing a batch loop."""

    def __init__(self, components, dataloader, *, phases, event_handlers=()):
        self.components = components
        self.dataloader = dataloader
        self.event_handlers = tuple(event_handlers)
        self._stats = {phase: [] for phase in phases}
        self.epoch = 0

    @property
    def statistics(self):
        return tuple(stat for stats in self._stats.values() for stat in stats)

    def add_statistic(self, statistic, *, phase="free"):
        if phase not in self._stats:
            raise ValueError(f"Unknown measurement phase {phase!r}; expected {tuple(self._stats)}.")
        if any(statistic is existing for existing in self.statistics):
            raise ValueError("A statistic can only be registered once per runner.")
        if statistic.name and any(
            (statistic.name, statistic.option, statistic.display_name)
            == (existing.name, existing.option, existing.display_name)
            for existing in self.statistics
        ):
            raise ValueError(f"Duplicate named statistic: {statistic.name!r}, {statistic.option!r}.")
        self._stats[phase].append(statistic)

    def dataset_size(self):
        return len(self.dataloader.dataset)

    def _reset(self):
        for statistic in self.statistics:
            statistic.reset()
        for handler in self.event_handlers:
            reset = getattr(handler, "reset", None)
            if callable(reset):
                reset()

    def _measure(self, phase):
        for statistic in self._stats[phase]:
            statistic.do_measurement()

    def __str__(self):
        return self.label + ", ".join(str(stat) for stat in self.statistics if stat.display)


class Trainer(Epoch):
    """Bind training components and measurements to ``train_epoch``."""

    label = "TRAIN -- "

    def __init__(
        self,
        components: ExperimentComponents,
        dataloader,
        *,
        reset_input: bool,
        modifier=None,
        event_handlers=(),
    ):
        super().__init__(
            components, dataloader, phases=("free", "gradients"),
            event_handlers=event_handlers,
        )
        self.reset_input = reset_input
        self.modifier = modifier
        self.global_step = 0

    def run(self, verbose=False):
        self._reset()

        def measure(event):
            if isinstance(event, FreePhaseEvent):
                self._measure("free")
            elif isinstance(event, GradientsReadyEvent):
                self._measure("gradients")
            elif isinstance(event, AfterUpdateEvent) and verbose:
                print(f"\r{self}", end="", flush=True)

        result = train_epoch(
            self.components,
            self.dataloader,
            modifier=self.modifier,
            event_handlers=(*self.event_handlers, measure),
            epoch=self.epoch,
            start_global_step=self.global_step,
            reset_input=self.reset_input,
        )
        self.global_step = result.next_global_step
        self.epoch += 1
        if verbose:
            print()
        return result


class Evaluator(Epoch):
    """Bind evaluation components, statistics, and probes to ``evaluate``."""

    label = "TEST  -- "

    def __init__(
        self,
        components: EvaluationComponents,
        dataloader,
        *,
        reset_input: bool = True,
        modifier=None,
        event_handlers=(),
        probes=(),
    ):
        super().__init__(
            components, dataloader, phases=("free",), event_handlers=event_handlers,
        )
        self.reset_input = reset_input
        self.modifier = modifier
        self.probes = tuple(probes)
        self.idx = None

    def run(self, verbose=False):
        self._reset()

        def measure(event):
            self.idx = event.batch.indices
            self._measure("free")
            if verbose:
                print(f"\r{self}", end="", flush=True)

        result = evaluate(
            self.components,
            self.dataloader,
            modifier=self.modifier,
            probes=self.probes,
            event_handlers=(*self.event_handlers, measure),
            epoch=self.epoch,
            reset_input=self.reset_input,
        )
        self.epoch += 1
        if verbose:
            print()
        return result
