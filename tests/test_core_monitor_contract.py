import pickle
from types import SimpleNamespace

import pytest

from training.core.engine import EvaluationComponents, ExperimentComponents
from training.lab.epoch import Trainer, Evaluator
from training.lab.monitor import Monitor
from training.lab.statistics import add_standard_statistics
from test_small_network_numerical_parity import _build_stack, _load_oracle


class _Statistic:
    name = "Error"
    option = None
    display_name = "Error"
    display = True

    def __init__(self):
        self.value = 0.0

    def get(self):
        return self.value


def _monitor_parts(errors, *, fail=False):
    order = []
    train_stat, evaluation_stat = _Statistic(), _Statistic()
    remaining = iter(errors)

    def train(verbose=False):
        order.append("train")
        if fail:
            raise RuntimeError("injected training failure")
        train_stat.value = 30.0

    def evaluate(verbose=False):
        order.append("evaluate")
        evaluation_stat.value = next(remaining)

    trainer = SimpleNamespace(statistics=(train_stat,), run=train)
    evaluator = SimpleNamespace(statistics=(evaluation_stat,), run=evaluate)
    scheduler = SimpleNamespace(step=lambda: order.append("schedule"))
    return trainer, evaluator, scheduler, order


def test_monitor_preserves_series_units_order_and_strict_selection(tmp_path):
    trainer, evaluator, scheduler, order = _monitor_parts([40.0, 40.0, 20.0])
    saved = []
    path = tmp_path / "new" / "run"
    monitor = Monitor(trainer=trainer, evaluator=evaluator, scheduler=scheduler,
                      save_model=saved.append, path=path, use_tensorboard=False)
    before = trainer.statistics, evaluator.statistics
    monitor.run(3, verbose=False)
    assert (trainer.statistics, evaluator.statistics) == before
    assert order == ["train", "schedule", "evaluate"] * 3
    assert saved == [str(path / "model.pt")] * 2
    assert monitor.test_error() == 20.0
    with (path / "time_series.pkl").open("rb") as handle:
        assert pickle.load(handle) == {
            "Error/train": [30.0, 30.0, 30.0], "Error/test": [40.0, 40.0, 20.0],
        }


@pytest.mark.parametrize("fail", [False, True])
def test_monitor_closes_writer_on_completion_or_failure(tmp_path, monkeypatch, fail):
    closed = []
    writer = SimpleNamespace(add_scalar=lambda *args: None, close=lambda: closed.append(True))
    monkeypatch.setattr("training.lab.monitor._create_summary_writer", lambda _: writer)
    trainer, evaluator, scheduler, _ = _monitor_parts([20.0], fail=fail)
    monitor = Monitor(trainer=trainer, evaluator=evaluator, scheduler=scheduler,
                      save_model=lambda _: None, path=tmp_path)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            monitor.run(1)
    else:
        monitor.run(1)
    assert closed == [True]
    assert monitor._writer is None


def test_monitor_can_run_incrementally_with_fresh_writers(tmp_path, monkeypatch):
    writers, writes, closed = [], [], []

    def create_writer(path):
        writer = SimpleNamespace(
            add_scalar=lambda *args: writes.append(args),
            close=lambda: closed.append(writer),
        )
        writers.append(writer)
        return writer

    monkeypatch.setattr("training.lab.monitor._create_summary_writer", create_writer)
    trainer, evaluator, scheduler, _ = _monitor_parts([40.0, 20.0])
    monitor = Monitor(trainer=trainer, evaluator=evaluator, scheduler=scheduler,
                      save_model=lambda _: None, path=tmp_path)
    monitor.run(1, verbose=False)
    monitor.run(1, verbose=False)
    assert len(writers) == 2
    assert all(writer is finished for writer, finished in zip(writers, closed))
    assert closed == writers
    assert [step for name, _, step in writes if name == "Error/test"] == [1, 2]
    with (tmp_path / "time_series.pkl").open("rb") as handle:
        assert pickle.load(handle)["Error/test"] == [40.0, 20.0]


def test_standard_statistics_have_one_owner_and_no_duplicate_measurements(tmp_path):
    stack = _build_stack(_load_oracle())
    energy, cost = stack.bundle.energy, stack.cost_fn
    loader = SimpleNamespace(dataset=range(2))
    trainer = Trainer(
        ExperimentComponents(stack.network, cost, None, tuple(energy.params()), None, None),
        loader, reset_input=True,
    )
    evaluator = Evaluator(EvaluationComponents(stack.network, cost, None), loader)
    add_standard_statistics(trainer, energy, cost, training=True)
    add_standard_statistics(evaluator, energy, cost)
    before = trainer.statistics, evaluator.statistics
    Monitor(trainer=trainer, evaluator=evaluator, scheduler=None,
            save_model=lambda _: None, path=tmp_path, use_tensorboard=False)
    assert (trainer.statistics, evaluator.statistics) == before
    for runner in (trainer, evaluator):
        names = [(stat.name, stat.option, stat.display_name) for stat in runner.statistics if stat.name]
        assert len(names) == len(set(names))
        assert sum(stat.name == "Error" for stat in runner.statistics) == 1
    with pytest.raises(ValueError, match="once"):
        trainer.add_statistic(trainer.statistics[0])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.add_statistic(_Statistic())


def test_monitor_preserves_interleaved_weight_distribution_format(tmp_path, monkeypatch):
    """Historical plotters depend on shared tags and the six-value order."""
    import torch
    from training.lab.statistics import WeightDistributionStat

    trainer, evaluator, scheduler, _ = _monitor_parts([20.0])
    weight = SimpleNamespace(name="DenseWeight_0", state=torch.tensor([[1.0, 2.0], [4.0, 8.0]]))
    kinds = ("mean", "std", "min", "max", "abs_mean", "abs_std")
    statistics = [WeightDistributionStat(weight, kind) for kind in kinds]
    for statistic in statistics:
        statistic.do_measurement()
    trainer.statistics = (*trainer.statistics, *statistics)
    writes = []
    writer = SimpleNamespace(add_scalar=lambda *args: writes.append(args), close=lambda: None)
    monkeypatch.setattr("training.lab.monitor._create_summary_writer", lambda _: writer)
    Monitor(trainer=trainer, evaluator=evaluator, scheduler=scheduler,
            save_model=lambda _: None, path=tmp_path).run(1, verbose=False)
    tag = "WeightDist/train_DenseWeight_0"
    assert [value for name, value, step in writes if name == tag] == [stat.get() for stat in statistics]
    with (tmp_path / "time_series.pkl").open("rb") as handle:
        assert pickle.load(handle)[tag] == [statistics[-1].get()]
