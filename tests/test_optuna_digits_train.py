from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from ebl.registry import resolve_experiment_config
from ebl.schema.base import ConfigError, RunMode
from ebl.runtime.small_network.runtime import (
    TrainingEpochReport,
    TrainingOutcome,
)
from labs.tools.optuna_digits_train import (
    OptunaEpochObserver,
    PruningPolicy,
    build_parser,
    build_pruning_policy,
    build_search_space,
    build_trial_config,
    load_base_experiment,
    pruner_score,
    selected_validation_objective,
)


_REPO_ROOT = Path(__file__).parents[1]


def _digits_payload() -> dict:
    payload = json.loads(
        (_REPO_ROOT / "examples" / "small_drn" / "base.json").read_text(
            encoding="utf-8"
        )
    )
    payload["data"].update(
        {
            "dataset": "digits",
            "num_points": 20,
            "batch_size": 2,
            "shuffle": False,
        }
    )
    payload["model"]["dims"] = [128, 8, 10]
    payload["solver"].update(
        {"inference_iterations": 1, "training_iterations": 1}
    )
    payload["modes"]["train"].update(
        {
            "num_epochs": 2,
            "max_batches": 1,
            "max_validation_batches": 1,
        }
    )
    return payload


def _write_payload(tmp_path: Path, payload: dict, name: str = "base.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _Trial:
    def __init__(self, *, should_prune: bool = False) -> None:
        self.suggestions = {}
        self.reports = []
        self._should_prune = should_prune

    def suggest_float(self, name, low, high, *, log=False):
        value = low * 2.0 if log else (low + high) / 2.0
        self.suggestions[name] = {
            "low": low,
            "high": high,
            "log": log,
            "value": value,
        }
        return value

    def report(self, value, *, step):
        self.reports.append((value, step))

    def should_prune(self):
        return self._should_prune


class _Pruned(RuntimeError):
    pass


def _report(
    *,
    completed_epochs: int = 2,
    validation_error: float = 0.4,
) -> TrainingEpochReport:
    return TrainingEpochReport(
        epoch_index=completed_epochs - 1,
        completed_epochs=completed_epochs,
        global_step=completed_epochs,
        train_examples=2,
        train_mean_cost=0.3,
        train_error_fraction=0.25,
        train_accuracy=0.75,
        validation_examples=2,
        validation_mean_cost=0.2,
        validation_error_fraction=validation_error,
        validation_accuracy=1.0 - validation_error,
        selected_this_epoch=True,
        selected_epoch_index=completed_epochs - 1,
        selected_validation_cost=0.2,
        selected_validation_error_fraction=validation_error,
        selected_validation_accuracy=1.0 - validation_error,
    )


def _changed_paths(before, after, prefix=()):
    if isinstance(before, dict) and isinstance(after, dict):
        paths = set()
        for key in before.keys() | after.keys():
            paths.update(
                _changed_paths(
                    before.get(key),
                    after.get(key),
                    (*prefix, key),
                )
            )
        return paths
    if before != after:
        return {".".join(prefix)}
    return set()


def test_strict_digits_config_and_trial_mutation_use_registry(
    tmp_path: Path,
) -> None:
    payload = _digits_payload()
    original = json.loads(json.dumps(payload))
    base_path = _write_payload(tmp_path, payload)
    base = load_base_experiment(base_path)
    args = build_parser().parse_args(
        [
            "--base-config",
            str(base_path),
            "--tune-input-gain",
            "--input-gain-min",
            "1.0",
            "--input-gain-max",
            "20.0",
            "--nudging-min",
            "0.01",
            "--nudging-max",
            "0.2",
            "--nudging-log",
        ]
    )
    search = build_search_space(args, base.spec)
    trial = _Trial()
    trial_payload = build_trial_config(base.payload, trial, search)

    assert payload == original
    assert _changed_paths(payload, trial_payload) == {
        "model.input_gain",
        "model.weight_gains",
        "modes.train.learning_rates",
        "modes.train.nudging",
    }
    trial_path = _write_payload(tmp_path, trial_payload, "trial.json")
    _definition, trial_spec = resolve_experiment_config(
        trial_path,
        RunMode.TRAIN,
    )
    assert tuple(trial_payload["model"]["weight_gains"]) == (
        trial_spec.common.model.weight_gains
    )
    assert tuple(trial_payload["modes"]["train"]["learning_rates"]) == (
        trial_spec.settings.learning_rates
    )


def test_legacy_flat_or_non_digits_base_config_is_rejected(
    tmp_path: Path,
) -> None:
    legacy = _write_payload(
        tmp_path,
        {
            "dims": [128, 8, 10],
            "learning_rate": [0.1, 0.1],
            "dataset": "digits",
        },
        "legacy.json",
    )
    with pytest.raises(ConfigError, match="experiment_id"):
        load_base_experiment(legacy)

    moons = _write_payload(
        tmp_path,
        json.loads(
            (_REPO_ROOT / "examples" / "small_drn" / "base.json").read_text(
                encoding="utf-8"
            )
        ),
        "moons.json",
    )
    with pytest.raises(ConfigError, match="dataset.*digits"):
        load_base_experiment(moons)


def test_epoch_observer_uses_fraction_thresholds_and_one_based_cutoff(
    tmp_path: Path,
) -> None:
    trial = _Trial()
    observer = OptunaEpochObserver(
        trial=trial,
        log_path=tmp_path / "epochs.jsonl",
        policy=PruningPolicy(
            train_cost_threshold=None,
            train_cost_epoch=1,
            validation_error_threshold=0.3,
            validation_error_epoch=2,
            pruner_metric="test_accuracy",
            use_trial_pruner=False,
        ),
        pruned_error=_Pruned,
    )
    observer(_report(completed_epochs=1, validation_error=0.4))
    with pytest.raises(_Pruned, match="validation error fraction"):
        observer(_report(completed_epochs=2, validation_error=0.4))
    records = [
        json.loads(line)
        for line in (tmp_path / "epochs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["completed_epochs"] for record in records] == [1, 2]
    assert records[-1]["validation_error_fraction"] == 0.4

    bad_args = build_parser().parse_args(
        [
            "--base-config",
            "unused.json",
            "--prune-test-error-threshold",
            "40",
        ]
    )
    with pytest.raises(ValueError, match="fraction in \\[0, 1\\]"):
        build_pruning_policy(bad_args)


def test_pruner_scores_and_objective_use_clean_fractional_metrics(
    tmp_path: Path,
) -> None:
    report = _report(validation_error=0.4)
    assert pruner_score("test_accuracy", report) == 0.6
    assert pruner_score("test_error", report) == -0.4
    assert pruner_score("train_accuracy", report) == 0.75
    assert pruner_score("train_error", report) == -0.25

    trial = _Trial()
    reporting = OptunaEpochObserver(
        trial=trial,
        log_path=tmp_path / "reported.jsonl",
        policy=PruningPolicy(
            train_cost_threshold=None,
            train_cost_epoch=1,
            validation_error_threshold=None,
            validation_error_epoch=1,
            pruner_metric="test_accuracy",
            use_trial_pruner=True,
        ),
        pruned_error=_Pruned,
    )
    reporting(report)
    assert trial.reports == [(0.6, 2)]

    outcome = TrainingOutcome(
        run_dir=tmp_path / "run",
        result_path=tmp_path / "run" / "result.json",
        weights_path=tmp_path / "run" / "checkpoints" / "weights.pt",
        resume_path=tmp_path / "run" / "checkpoints" / "resume.pt",
        completed_epochs=2,
        global_step=2,
        selected_epoch_index=1,
        selected_validation_cost=0.2,
        selected_validation_error_fraction=0.4,
        selected_validation_accuracy=0.6,
        last_epoch_report=report,
        resume_capability="exact",
    )
    assert selected_validation_objective(outcome) == 0.6
    with pytest.raises(RuntimeError, match="accuracy"):
        selected_validation_objective(
            replace(outcome, selected_validation_accuracy=None)
        )
