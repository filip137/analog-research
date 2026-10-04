from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments.small_network import learning_rate_selection as selection
from experiments.small_network import runtime as small_runtime
from model.variable.parameter import DenseWeight
from training.core.engine import FreePhaseEvent, GradientsReadyEvent


class _StateCodec:
    def __init__(self, value: float) -> None:
        self.value = torch.tensor([value], dtype=torch.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"value": self.value.detach().clone()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.value.copy_(state["value"])


class _SnapshotOptimizer(_StateCodec):
    cohort = "A"

    def __init__(self, parameter: Any) -> None:
        super().__init__(0.0)
        self.parameter = parameter
        self.rates = (0.0, 0.0, 0.0)

    def initialize_at_pulse_zero(self) -> None:
        self.parameter.state.fill_(3.0)
        self.value.fill_(7.0)

    def set_learning_rates(self, rates: tuple[float, ...]) -> None:
        self.rates = tuple(float(rate) for rate in rates)

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "rates": self.rates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.rates = tuple(state["rates"])


class _MetricStore:
    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []

    def append_metric(self, metric: dict[str, Any]) -> None:
        self.metrics.append(deepcopy(metric))


def _parameter(value: float) -> Any:
    return SimpleNamespace(state=torch.tensor([value], dtype=torch.float32))


def _snapshot_runtime() -> Any:
    parameter = _parameter(1.0)
    generator = torch.Generator().manual_seed(1234)
    return SimpleNamespace(
        training_components=SimpleNamespace(parameters=(parameter,)),
        optimizer=_SnapshotOptimizer(parameter),
        runtime_state=_StateCodec(5.0),
        data=SimpleNamespace(dataloader_generators={"train": generator}),
    )


def _selection_spec() -> Any:
    settings = {
        "weight_relative_update_targets": (0.1, 0.2),
        "max_center_reductions": 0,
        "grid_multipliers": (0.5, 1.0),
        "plateau_relative_tolerance": 0.0,
        "safety": {},
    }
    return SimpleNamespace(
        settings=SimpleNamespace(
            learning_rate_selection=SimpleNamespace(
                type="bounded_relative_update_grid",
                parameters=settings,
            ),
            max_batches=None,
            max_validation_batches=None,
        )
    )


def _probe_report() -> dict[str, Any]:
    return {
        "parameter_names": ["weight.0", "weight.1", "bias.0"],
        "weight_names": ["weight.0", "weight.1"],
        "bias_to_weight": {"bias.0": "weight.0"},
        "normalization_unit_by_weight": {
            "weight.0": 2.0,
            "weight.1": 4.0,
        },
        "bias_q90_unit_by_parameter": {"bias.0": 4.0},
        "used_batches": 4,
    }


def _assert_pristine(runtime: Any, generator_state: torch.Tensor) -> None:
    torch.testing.assert_close(
        runtime.training_components.parameters[0].state,
        torch.tensor([3.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        runtime.optimizer.value,
        torch.tensor([7.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        runtime.runtime_state.value,
        torch.tensor([5.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert runtime.optimizer.rates == (0.0, 0.0, 0.0)
    assert torch.equal(
        runtime.data.dataloader_generators["train"].get_state(),
        generator_state,
    )


def _dirty(runtime: Any) -> None:
    runtime.training_components.parameters[0].state.add_(100.0)
    runtime.optimizer.value.add_(100.0)
    runtime.runtime_state.value.add_(100.0)
    torch.rand((), generator=runtime.data.dataloader_generators["train"])


def test_runtime_snapshot_restores_parameters_optimizer_layers_and_generators(
) -> None:
    runtime = _snapshot_runtime()
    generator_state = runtime.data.dataloader_generators[
        "train"
    ].get_state().clone()
    snapshot = selection._RuntimeSnapshot.capture(runtime)

    _dirty(runtime)
    runtime.optimizer.rates = (1.0, 2.0, 3.0)
    snapshot.restore(runtime)

    torch.testing.assert_close(
        runtime.training_components.parameters[0].state,
        torch.tensor([1.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        runtime.optimizer.value,
        torch.tensor([0.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        runtime.runtime_state.value,
        torch.tensor([5.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert runtime.optimizer.rates == (0.0, 0.0, 0.0)
    assert torch.equal(
        runtime.data.dataloader_generators["train"].get_state(),
        generator_state,
    )


def test_grid_selection_restarts_every_trial_and_uses_stable_tie_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _snapshot_runtime()
    initial_generator_state = runtime.data.dataloader_generators[
        "train"
    ].get_state().clone()
    canary_targets: list[tuple[float, float]] = []
    candidate_targets: list[tuple[float, float]] = []

    monkeypatch.setattr(selection, "MeasuredTraceOptimizer", _SnapshotOptimizer)

    def run_probe(candidate_runtime: Any, _settings: Any) -> dict[str, Any]:
        _assert_pristine(candidate_runtime, initial_generator_state)
        _dirty(candidate_runtime)
        return _probe_report()

    def targets_from_rates(rates: tuple[float, ...]) -> tuple[float, float]:
        return (2.0 * rates[0], 4.0 * rates[1])

    def run_canary(
        _spec: Any,
        candidate_runtime: Any,
        *,
        rates: tuple[float, ...],
        settings: Any,
    ) -> dict[str, Any]:
        del settings
        _assert_pristine(candidate_runtime, initial_generator_state)
        targets = targets_from_rates(rates)
        canary_targets.append(targets)
        candidate_runtime.optimizer.set_learning_rates(rates)
        _dirty(candidate_runtime)
        unsafe = targets == pytest.approx((0.05, 0.2))
        failure = (
            {
                "kind": "gradient_rms_explosion",
                "parameter": "weight.1",
                "onset_step": 2,
                "confirmed_step": 3,
            }
            if unsafe
            else None
        )
        return {
            "status": "rejected_safety" if unsafe else "clean",
            "failure": failure,
            "safety": {},
        }

    def run_candidate(
        _spec: Any,
        candidate_runtime: Any,
        *,
        rates: tuple[float, ...],
        settings: Any,
    ) -> dict[str, Any]:
        del settings
        _assert_pristine(candidate_runtime, initial_generator_state)
        targets = targets_from_rates(rates)
        candidate_targets.append(targets)
        candidate_runtime.optimizer.set_learning_rates(rates)
        _dirty(candidate_runtime)
        return {
            "status": "complete",
            "failure": None,
            "epochs": [],
            "final_validation_mean_cost": 1.0,
            "final_validation_accuracy": 0.8,
            "safety": {},
            "projection": {},
        }

    monkeypatch.setattr(selection, "_run_probe", run_probe)
    monkeypatch.setattr(selection, "_run_canary", run_canary)
    monkeypatch.setattr(selection, "_run_candidate", run_candidate)

    report = selection.select_measured_learning_rates(
        _selection_spec(),
        runtime,
        store=_MetricStore(),
    )

    assert canary_targets == pytest.approx(
        [(0.1, 0.2), (0.05, 0.1), (0.05, 0.2), (0.1, 0.1)]
    )
    assert candidate_targets == pytest.approx(
        [(0.05, 0.1), (0.1, 0.1), (0.1, 0.2)]
    )
    assert report["cells"][1]["candidate"]["status"] == (
        "skipped_unsafe_canary"
    )
    assert report["selection"]["plateau_cell_ids"] == [
        "cell_00",
        "cell_02",
        "cell_03",
    ]
    assert report["selection"]["selected_cell_id"] == "cell_00"
    assert report["selection"]["selected_learning_rate_vector"] == (
        pytest.approx([0.025, 0.025, 0.0125])
    )


def test_canary_reports_persistent_gradient_safety_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = DenseWeight(
        (1,),
        (1,),
        gain=1.0,
        device="cpu",
        clamp=False,
    )
    weight.state.fill_(1.0)
    cost = SimpleNamespace(eval=lambda: torch.tensor([1.0]))
    components = SimpleNamespace(cost_fn=cost, parameters=(weight,))
    batch = SimpleNamespace()

    def train_with_explosion(
        _components: Any,
        _loader: Any,
        *,
        event_handlers: Any,
        **_kwargs: Any,
    ) -> None:
        monitor = tuple(event_handlers)[0]
        for step, gradient in enumerate((1.0, 3.0, 4.0)):
            monitor(
                FreePhaseEvent(
                    epoch=0,
                    batch_index=step,
                    global_step=step,
                    batch=batch,
                    components=components,
                )
            )
            monitor(
                GradientsReadyEvent(
                    epoch=0,
                    batch_index=step,
                    global_step=step,
                    batch=batch,
                    components=components,
                    gradients=(torch.tensor([[gradient]]),),
                )
            )
        raise AssertionError("the persistent safety gate should reject")

    monkeypatch.setattr(
        selection,
        "_parameter_topology",
        lambda _runtime: {"by_name": {"weight.0": weight}},
    )
    monkeypatch.setattr(selection, "train_epoch", train_with_explosion)
    optimizer = SimpleNamespace(set_learning_rates=lambda _rates: None)
    runtime = SimpleNamespace(
        optimizer=optimizer,
        training_components=components,
        data=SimpleNamespace(train_loader=(None,) * 3),
    )
    settings = {
        "canary_batches": 3,
        "safety": {
            "warmup_batches": 1,
            "loss_ema_decay": 0.0,
            "loss_growth_factor": 100.0,
            "gradient_growth_factor": 2.0,
            "persistence_batches": 2,
        },
    }

    result = selection._run_canary(
        SimpleNamespace(),
        runtime,
        rates=(0.1,),
        settings=settings,
    )

    assert result["status"] == "rejected_safety"
    assert result["failure"] == {
        "kind": "gradient_rms_explosion",
        "parameter": "weight.0",
        "onset_step": 2,
        "confirmed_step": 3,
    }
    assert result["safety"]["processed_batches"] == 3
    assert result["safety"]["maximum_gradient_streak"] == {"weight.0": 2}


def test_execute_train_reseeds_and_rebuilds_before_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ProductionOptimizer:
        def __init__(self, initial_token: str) -> None:
            self.initial_token = initial_token
            self.rates: tuple[float, ...] | None = None

        def set_learning_rates(self, rates: tuple[float, ...]) -> None:
            self.rates = tuple(float(rate) for rate in rates)

    class Store:
        def __init__(self) -> None:
            self.run_dir = tmp_path / "run"
            self.completed_metrics: dict[str, Any] | None = None

        def artifact_record(self, path: Path, *, kind: str) -> dict[str, Any]:
            return {"path": str(path), "kind": kind}

        def complete(self, *, metrics: Any, artifacts: Any) -> Path:
            del artifacts
            self.completed_metrics = dict(metrics)
            return self.run_dir / "result.json"

        def fail(self, error: Exception) -> None:
            raise AssertionError(f"unexpected run failure: {error!r}")

    spec = SimpleNamespace(
        common=SimpleNamespace(
            runtime=SimpleNamespace(seed=41),
            data=SimpleNamespace(validation_points=5000),
        ),
        settings=SimpleNamespace(
            learning_rate_selection=SimpleNamespace(
                type="bounded_relative_update_grid"
            ),
            weight_modifier=SimpleNamespace(type="none", parameters={}),
        ),
    )
    request = SimpleNamespace(
        spec=object(),
        resume=None,
        weights=None,
        base_weights=None,
        device_data=None,
    )
    store = Store()
    seed_calls: list[int] = []
    built: list[Any] = []
    written: list[tuple[Path, Any]] = []
    executed: list[Any] = []

    def seed_runtime(seed: int) -> None:
        seed_calls.append(seed)

    def build_runtime(_spec: Any, *, device_data_path: Any) -> Any:
        assert device_data_path is None
        candidate = SimpleNamespace(
            optimizer=ProductionOptimizer(f"seed-{seed_calls[-1]}"),
            resume_capability="exact",
        )
        built.append(candidate)
        return candidate

    report = {
        "selection": {
            "selected_learning_rate_vector": [0.25, 0.5, 0.125],
            "selected_cell_id": "cell_00",
        }
    }

    def select_rates(
        _spec: Any,
        candidate: Any,
        *,
        store: Any,
        initial_weights_path: Any,
    ) -> dict[str, Any]:
        del store, initial_weights_path
        assert candidate is built[0]
        candidate.optimizer.initial_token = "mutated-by-selection"
        return report

    def execute(
        _request: Any,
        _spec: Any,
        production: Any,
        _store: Any,
        *,
        observers: Any,
    ) -> tuple[dict[str, Any], tuple[Any, ...], None]:
        assert tuple(observers) == ()
        executed.append(production)
        return (
            {
                "completed_epochs": 1,
                "global_step": 2,
                "resume_capability": "exact",
                "selected": {
                    "epoch": 0,
                    "value": 0.4,
                    "error_fraction": 0.2,
                    "accuracy": 0.8,
                },
            },
            (),
            None,
        )

    monkeypatch.setattr(small_runtime, "_expect_spec", lambda *_a, **_k: spec)
    monkeypatch.setattr(
        small_runtime,
        "_validate_training_initialization",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        small_runtime,
        "configured_resume_capability",
        lambda _spec: "exact",
    )
    monkeypatch.setattr(
        small_runtime,
        "_create_run_store",
        lambda *_a, **_k: store,
    )
    monkeypatch.setattr(small_runtime, "seed_runtime", seed_runtime)
    monkeypatch.setattr(small_runtime, "build_train_runtime", build_runtime)
    monkeypatch.setattr(selection, "select_measured_learning_rates", select_rates)
    monkeypatch.setattr(small_runtime, "_execute_training", execute)
    monkeypatch.setattr(
        small_runtime,
        "atomic_write_json",
        lambda path, payload: written.append((path, payload)),
    )
    monkeypatch.setattr(
        small_runtime,
        "MeasuredTraceOptimizer",
        ProductionOptimizer,
    )
    monkeypatch.setattr(small_runtime.torch.cuda, "is_available", lambda: False)

    outcome = small_runtime.execute_train(request)

    assert seed_calls == [41, 41]
    assert len(built) == 2
    assert built[0] is not built[1]
    assert built[0].optimizer.initial_token == "mutated-by-selection"
    assert built[1].optimizer.initial_token == "seed-41"
    assert built[0].optimizer.rates is None
    assert built[1].optimizer.rates == (0.25, 0.5, 0.125)
    assert executed == [built[1]]
    assert written == [
        (tmp_path / "run/artifacts/lr_selection.json", report)
    ]
    assert store.completed_metrics is not None
    assert store.completed_metrics["learning_rate_selection"] == report[
        "selection"
    ]
    assert outcome.run_dir == tmp_path / "run"
