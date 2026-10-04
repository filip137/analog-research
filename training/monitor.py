"""Epoch orchestration and recording of statistics registered by experiment setup."""

from datetime import datetime
from pathlib import Path
import pickle
import time


def _create_summary_writer(path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as error:
        print(f"[monitor] Tensorboard disabled: {error}")
        return None
    return SummaryWriter(str(path))


class Monitor:
    """Coordinate runners and record results without constructing numerical objects."""

    def __init__(self, *, trainer, evaluator, scheduler, save_model, path=None,
                 use_tensorboard=True):
        if not callable(save_model):
            raise TypeError("Expected save_model to be a callable accepting a path.")
        self._trainer = trainer
        self._evaluator = evaluator
        self._scheduler = scheduler
        self._save_model = save_model
        self._path = Path(path) if path is not None else Path(datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._series = [
            TimeSeries(statistic, train=train)
            for runner, train in ((trainer, True), (evaluator, False))
            for statistic in runner.statistics
            if statistic.name
        ]
        # Weight-distribution variants intentionally retain their historical
        # shared tag and registration order: existing plotters deinterleave
        # the six values. Epoch rejects duplicate *statistics*, not those tags.
        error_series = [series for series in self._series if series.name == "Error/test"]
        if len(error_series) != 1:
            raise ValueError("Register exactly one evaluation Error statistic before constructing Monitor.")
        self._test_error_curve = error_series[0]
        self._epoch = 0
        self._use_tensorboard = bool(use_tensorboard)
        self._writer = None
        self._start_time = time.time()

    def run(self, num_epochs, verbose=True):
        self._path.mkdir(parents=True, exist_ok=True)
        try:
            if self._use_tensorboard:
                self._writer = _create_summary_writer(self._path)
            for _ in range(num_epochs):
                self.one_epoch(verbose=verbose)
        finally:
            writer, self._writer = self._writer, None
            if writer is not None:
                writer.close()

    def one_epoch(self, verbose=True):
        self._path.mkdir(parents=True, exist_ok=True)
        self._epoch += 1
        print(f"Epoch {self._epoch}")
        self._trainer.run(verbose=verbose)
        self._scheduler.step()
        self._evaluator.run(verbose=verbose)
        for series in self._series:
            series.update()
        if self._writer is not None:
            for series in self._series:
                self._writer.add_scalar(series.name, series.last_value(), self._epoch)
        if not verbose:
            print(str(self._trainer))
            print(str(self._evaluator))
        self.save_series()
        if self._test_error_curve.is_minimum():
            self.save_network()
        seconds = time.time() - self._start_time
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"Duration = {hours:.0f} hours {minutes:.0f} min {seconds:.0f} sec\n")

    def test_error(self):
        return self._test_error_curve.last_value()

    def save_network(self):
        self._save_model(str(self._path / "model.pt"))

    def save_series(self):
        with (self._path / "time_series.pkl").open("wb") as handle:
            pickle.dump(
                {series.name: series.get() for series in self._series},
                handle, protocol=pickle.HIGHEST_PROTOCOL,
            )


class TimeSeries:
    """One registered statistic sampled at each epoch boundary."""

    def __init__(self, statistic, train=True):
        self._statistic = statistic
        self._time_series = []
        self.name = statistic.name + ("/train" if train else "/test")
        if statistic.option:
            self.name += "_" + statistic.option

    def update(self):
        self._time_series.append(self._statistic.get())

    def get(self):
        return self._time_series

    def last_value(self):
        return self._time_series[-1]

    def minimum(self):
        return min(self._time_series)

    def is_minimum(self):
        return len(self._time_series) == 1 or self.last_value() < min(self._time_series[:-1])
