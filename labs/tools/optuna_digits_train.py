#!/usr/bin/env python3
"""Optuna search over the public ``small_drn.v1`` training runtime.

This is an intentional clean break from the legacy ``small_network`` config
and callback conventions:

* ``--base-config`` must be strict nested ``small_drn.v1`` JSON for digits;
* trials change only ``modes.train.learning_rates``,
  ``model.weight_gains``, ``modes.train.nudging``, and (when requested)
  ``model.input_gain``;
* every error, accuracy, and error threshold is a fraction in ``[0, 1]``;
* legacy ``test_*`` pruning flag names refer to the clean held-out validation
  pass used for checkpoint selection;
* the objective is the clean validation accuracy at the selected
  minimum-validation-cost epoch, not final training accuracy.

Run this as an installed console environment or as a module:
``python -m labs.tools.optuna_digits_train ...``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

from ebl.cli import TrainRequest
from ebl.registry import (
    parse_experiment_config,
    resolve_experiment_config,
)
from ebl.schema.base import ConfigError, RunMode
from ebl.schema.small_network import TrainSpec
from ebl.runtime.small_network.runtime import (
    TrainingEpochReport,
    TrainingOutcome,
    execute_train,
)


@dataclass(frozen=True)
class BaseExperiment:
    path: Path
    payload: Mapping[str, Any]
    spec: TrainSpec


@dataclass(frozen=True)
class SearchSpace:
    input_gain_bounds: tuple[float, float] | None
    input_gain_log: bool
    nudging_bounds: tuple[float, float]
    nudging_log: bool


@dataclass(frozen=True)
class PruningPolicy:
    train_cost_threshold: float | None
    train_cost_epoch: int
    validation_error_threshold: float | None
    validation_error_epoch: int
    pruner_metric: str
    use_trial_pruner: bool


class OptunaEpochObserver:
    """Report immutable runtime epochs to one Optuna-like trial."""

    def __init__(
        self,
        *,
        trial: Any,
        log_path: Path,
        policy: PruningPolicy,
        pruned_error: Callable[[str], Exception],
    ) -> None:
        self._trial = trial
        self._log_path = log_path
        self._policy = policy
        self._pruned_error = pruned_error

    def __call__(self, report: TrainingEpochReport) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(asdict(report), allow_nan=False, sort_keys=True)
                + "\n"
            )

        completed = report.completed_epochs
        threshold = self._policy.train_cost_threshold
        if (
            threshold is not None
            and completed >= self._policy.train_cost_epoch
            and report.train_mean_cost is not None
            and report.train_mean_cost > threshold
        ):
            raise self._pruned_error(
                f"Pruned after epoch {completed}: train_mean_cost="
                f"{report.train_mean_cost:.6g} > {threshold:.6g}."
            )

        error_threshold = self._policy.validation_error_threshold
        if (
            error_threshold is not None
            and completed >= self._policy.validation_error_epoch
            and report.validation_error_fraction >= error_threshold
        ):
            raise self._pruned_error(
                f"Pruned after epoch {completed}: clean validation error "
                f"fraction={report.validation_error_fraction:.6g} >= "
                f"{error_threshold:.6g}."
            )

        if not self._policy.use_trial_pruner:
            return
        score = pruner_score(self._policy.pruner_metric, report)
        if score is None:
            return
        self._trial.report(score, step=completed)
        if self._trial.should_prune():
            raise self._pruned_error(
                f"Pruned after epoch {completed}: "
                f"{self._policy.pruner_metric} score={score:.6g}."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna sweep for strict nested small_drn.v1 digits training. "
            "Accuracy/error values and thresholds are fractions in [0, 1]; "
            "the objective is selected clean-validation accuracy."
        )
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="Strict nested small_drn.v1 JSON configured for digits.",
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument(
        "--tune-input-gain",
        action="store_true",
        help="Include model.input_gain in the Optuna search space.",
    )
    parser.add_argument("--input-gain-min", type=float, default=None)
    parser.add_argument("--input-gain-max", type=float, default=None)
    parser.add_argument(
        "--input-gain-log",
        action="store_true",
        help="Sample model.input_gain logarithmically.",
    )
    parser.add_argument(
        "--nudging-min",
        type=float,
        default=None,
        help="Minimum modes.train.nudging.",
    )
    parser.add_argument(
        "--nudging-max",
        type=float,
        default=None,
        help="Maximum modes.train.nudging.",
    )
    parser.add_argument(
        "--nudging-log",
        action="store_true",
        help="Sample nudging logarithmically with explicit nudging bounds.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory containing trial configs, logs, and run directories.",
    )
    parser.add_argument(
        "--storage",
        default="auto",
        help=(
            "Optuna storage URI or path. Use 'none' for in-memory. "
            "Default: 'auto' -> <output_dir>/optuna.db."
        ),
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name (default: base config stem).",
    )
    parser.add_argument(
        "--load-if-exists",
        action="store_true",
        help="Reuse an existing named study when storage is enabled.",
    )
    parser.add_argument(
        "--prune-train-cost-threshold",
        type=float,
        default=None,
        help="Prune when train mean cost exceeds this threshold.",
    )
    parser.add_argument(
        "--prune-train-cost-epoch",
        type=int,
        default=20,
        help="Completed epoch (1-based, inclusive) enabling train-cost pruning.",
    )
    parser.add_argument(
        "--prune-test-error-threshold",
        type=float,
        default=None,
        help=(
            "Prune when clean held-out validation error is at least this "
            "fraction in [0, 1]. The test_* name is retained for CLI "
            "compatibility."
        ),
    )
    parser.add_argument(
        "--prune-test-error-epoch",
        type=int,
        default=50,
        help=(
            "Completed epoch (1-based, inclusive) enabling clean-validation "
            "error pruning."
        ),
    )
    parser.add_argument(
        "--pruner",
        choices=("none", "median"),
        default="none",
    )
    parser.add_argument(
        "--pruner-metric",
        choices=(
            "test_accuracy",
            "train_accuracy",
            "test_error",
            "train_error",
            "test_cost",
            "train_cost",
        ),
        default="test_accuracy",
        help=(
            "Metric reported for pruning. test_* means clean held-out "
            "validation; errors are fractions."
        ),
    )
    parser.add_argument("--pruner-startup-trials", type=int, default=5)
    parser.add_argument("--pruner-warmup-epochs", type=int, default=5)
    parser.add_argument("--pruner-interval-epochs", type=int, default=1)
    return parser


def load_base_experiment(path: Path | str) -> BaseExperiment:
    """Load and registry-resolve one strict nested digits experiment."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(
            "Expected --base-config to reference readable strict JSON. "
            f"Provided value: {str(source)!r}. {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            "Expected --base-config to contain valid JSON. "
            f"Provided value: {str(source)!r} "
            f"(line {exc.lineno}, column {exc.colno}: {exc.msg})."
        ) from exc
    definition, document = parse_experiment_config(payload)
    if definition.experiment_id != "small_drn.v1":
        raise ConfigError(
            "Expected --base-config experiment_id to be 'small_drn.v1'. "
            f"Provided value: {definition.experiment_id!r}."
        )
    spec = definition.resolve(document, RunMode.TRAIN)
    if not isinstance(spec, TrainSpec):
        raise ConfigError(
            "Expected --base-config to resolve a small_drn.v1 TrainSpec. "
            f"Provided value: {type(spec).__name__}."
        )
    if spec.common.data.dataset != "digits":
        raise ConfigError(
            "Expected --base-config config.data.dataset to be 'digits'. "
            f"Provided value: {spec.common.data.dataset!r}."
        )
    return BaseExperiment(path=source, payload=payload, spec=spec)


def build_search_space(
    args: argparse.Namespace,
    spec: TrainSpec,
) -> SearchSpace:
    """Validate CLI bounds and resolve defaults from the strict base spec."""

    input_bounds: tuple[float, float] | None = None
    if not args.tune_input_gain and (
        args.input_gain_min is not None
        or args.input_gain_max is not None
        or args.input_gain_log
    ):
        raise ValueError(
            "Expected --input-gain-min/--input-gain-max/--input-gain-log "
            "only with --tune-input-gain. Provided value: input-gain tuning "
            "was disabled."
        )
    if args.tune_input_gain:
        base = float(spec.common.model.input_gain)
        lower = (
            float(args.input_gain_min)
            if args.input_gain_min is not None
            else base * 0.1
        )
        upper = (
            float(args.input_gain_max)
            if args.input_gain_max is not None
            else base * 10.0
        )
        _validate_bounds(
            "input gain",
            lower,
            upper,
            logarithmic=args.input_gain_log,
        )
        input_bounds = (lower, upper)

    if (args.nudging_min is None) != (args.nudging_max is None):
        raise ValueError(
            "Expected --nudging-min and --nudging-max to be provided "
            "together. Provided value: exactly one bound."
        )
    if args.nudging_min is not None:
        nudging_lower = float(args.nudging_min)
        nudging_upper = float(args.nudging_max)
        nudging_log = bool(args.nudging_log)
    else:
        base_nudging = float(spec.settings.nudging)
        if base_nudging > 0.0:
            nudging_lower = base_nudging * 0.1
            nudging_upper = base_nudging * 10.0
            nudging_log = True
        else:
            nudging_lower = 0.0
            nudging_upper = 5.0
            nudging_log = False
        if args.nudging_log:
            raise ValueError(
                "Expected --nudging-log to be accompanied by explicit "
                "--nudging-min and --nudging-max. Provided value: no bounds."
            )
    _validate_bounds(
        "nudging",
        nudging_lower,
        nudging_upper,
        logarithmic=nudging_log,
    )
    return SearchSpace(
        input_gain_bounds=input_bounds,
        input_gain_log=bool(args.input_gain_log),
        nudging_bounds=(nudging_lower, nudging_upper),
        nudging_log=nudging_log,
    )


def build_trial_config(
    base_payload: Mapping[str, Any],
    trial: Any,
    search: SearchSpace,
) -> dict[str, Any]:
    """Copy JSON and mutate only the four declared nested search fields."""

    # A JSON round-trip both deep-copies and rejects non-JSON caller objects.
    config = json.loads(
        json.dumps(base_payload, allow_nan=False, sort_keys=True)
    )
    model = _required_object(config, "model")
    modes = _required_object(config, "modes")
    train = _required_object(modes, "train")

    learning_rates = _positive_sequence(
        train.get("learning_rates"),
        name="config.modes.train.learning_rates",
    )
    weight_gains = _positive_sequence(
        model.get("weight_gains"),
        name="config.model.weight_gains",
    )
    train["learning_rates"] = [
        trial.suggest_float(
            f"learning_rate_{index}",
            value * 0.1,
            value * 10.0,
            log=True,
        )
        for index, value in enumerate(learning_rates)
    ]
    model["weight_gains"] = [
        trial.suggest_float(
            f"weight_gain_{index}",
            value * 0.1,
            value * 10.0,
            log=True,
        )
        for index, value in enumerate(weight_gains)
    ]
    train["nudging"] = trial.suggest_float(
        "nudging",
        search.nudging_bounds[0],
        search.nudging_bounds[1],
        log=search.nudging_log,
    )
    if search.input_gain_bounds is not None:
        model["input_gain"] = trial.suggest_float(
            "input_gain",
            search.input_gain_bounds[0],
            search.input_gain_bounds[1],
            log=search.input_gain_log,
        )
    return config


def build_pruning_policy(args: argparse.Namespace) -> PruningPolicy:
    train_threshold = args.prune_train_cost_threshold
    if train_threshold is not None:
        train_threshold = float(train_threshold)
        if not math.isfinite(train_threshold) or train_threshold < 0.0:
            raise ValueError(
                "Expected --prune-train-cost-threshold to be a finite "
                f"non-negative number. Provided value: {train_threshold!r}."
            )
    validation_threshold = args.prune_test_error_threshold
    if validation_threshold is not None:
        validation_threshold = float(validation_threshold)
        if (
            not math.isfinite(validation_threshold)
            or not 0.0 <= validation_threshold <= 1.0
        ):
            raise ValueError(
                "Expected --prune-test-error-threshold to be a fraction in "
                f"[0, 1]. Provided value: {validation_threshold!r}."
            )
    for name, value in (
        ("--prune-train-cost-epoch", args.prune_train_cost_epoch),
        ("--prune-test-error-epoch", args.prune_test_error_epoch),
    ):
        if value < 1:
            raise ValueError(
                f"Expected {name} to be an integer >= 1. "
                f"Provided value: {value!r}."
            )
    for name, value, minimum in (
        ("--pruner-startup-trials", args.pruner_startup_trials, 0),
        ("--pruner-warmup-epochs", args.pruner_warmup_epochs, 0),
        ("--pruner-interval-epochs", args.pruner_interval_epochs, 1),
    ):
        if value < minimum:
            raise ValueError(
                f"Expected {name} to be an integer >= {minimum}. "
                f"Provided value: {value!r}."
            )
    return PruningPolicy(
        train_cost_threshold=train_threshold,
        train_cost_epoch=args.prune_train_cost_epoch,
        validation_error_threshold=validation_threshold,
        validation_error_epoch=args.prune_test_error_epoch,
        pruner_metric=args.pruner_metric,
        use_trial_pruner=args.pruner == "median",
    )


def pruner_score(
    metric_name: str,
    report: TrainingEpochReport,
) -> float | None:
    """Map all metrics to a maximize score without changing their units."""

    if metric_name == "train_accuracy":
        return report.train_accuracy
    if metric_name == "test_accuracy":
        return report.validation_accuracy
    if metric_name == "train_error":
        return (
            None
            if report.train_error_fraction is None
            else -report.train_error_fraction
        )
    if metric_name == "test_error":
        return -report.validation_error_fraction
    if metric_name == "train_cost":
        return (
            None
            if report.train_mean_cost is None
            else -report.train_mean_cost
        )
    if metric_name == "test_cost":
        return -report.validation_mean_cost
    raise ValueError(
        "Expected pruner metric to be a configured accuracy, error, or cost "
        f"name. Provided value: {metric_name!r}."
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    command = (
        sys.executable,
        "-m",
        "labs.tools.optuna_digits_train",
        *raw_argv,
    )
    if args.trials < 1:
        raise SystemExit(
            f"Expected --trials to be >= 1. Provided value: {args.trials!r}."
        )
    try:
        base = load_base_experiment(args.base_config)
        search = build_search_space(args, base.spec)
        pruning = build_pruning_policy(args)
    except (ConfigError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    optuna = _import_optuna()
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else base.path.parent / "optuna_runs"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    storage = _resolve_storage(args.storage, output_root=output_root)
    study_name = args.study_name or base.path.stem

    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_epochs,
            interval_steps=args.pruner_interval_epochs,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    def objective(trial: Any) -> float:
        config = build_trial_config(base.payload, trial, search)
        trial_dir = output_root / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "config.json"
        config_path.write_text(
            json.dumps(config, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        definition, spec = resolve_experiment_config(
            config_path,
            RunMode.TRAIN,
        )
        request = TrainRequest(
            definition=definition,
            spec=spec,
            config_path=config_path,
            output_dir=trial_dir / "runs",
            weights=None,
            base_weights=None,
            resume=None,
            command=command,
        )
        observer = OptunaEpochObserver(
            trial=trial,
            log_path=trial_dir / "epoch_metrics.jsonl",
            policy=pruning,
            pruned_error=optuna.TrialPruned,
        )
        outcome = execute_train(request, observers=(observer,))
        _record_trial_outcome(trial, outcome)
        return selected_validation_objective(outcome)

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage) and args.load_if_exists,
    )
    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    summary = {
        "objective": "selected_clean_validation_accuracy_fraction",
        "best_value": best.value,
        "best_params": best.params,
        "best_attrs": best.user_attrs,
    }
    summary_path = output_root / "best_trial.json"
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    return 0


def _record_trial_outcome(trial: Any, outcome: TrainingOutcome) -> None:
    trial.set_user_attr("run_dir", str(outcome.run_dir))
    trial.set_user_attr(
        "selected_validation_accuracy",
        outcome.selected_validation_accuracy,
    )
    trial.set_user_attr(
        "selected_validation_error_fraction",
        outcome.selected_validation_error_fraction,
    )
    trial.set_user_attr(
        "selected_validation_cost",
        outcome.selected_validation_cost,
    )
    trial.set_user_attr(
        "selected_completed_epoch",
        outcome.selected_epoch_index + 1,
    )
    if outcome.last_epoch_report is not None:
        trial.set_user_attr(
            "last_train_accuracy",
            outcome.last_epoch_report.train_accuracy,
        )


def selected_validation_objective(outcome: TrainingOutcome) -> float:
    """Return the clean selected accuracy used as the maximization objective."""

    accuracy = outcome.selected_validation_accuracy
    if accuracy is None:
        raise RuntimeError(
            "Expected a fresh Optuna training run to select a clean "
            "validation accuracy. Provided value: null."
        )
    return float(accuracy)


def _resolve_storage(value: Any, *, output_root: Path) -> str | None:
    storage_value = "" if value is None else str(value).strip()
    if not storage_value or storage_value.lower() in {
        "none",
        "memory",
        "in-memory",
    }:
        return None
    if "://" in storage_value:
        return storage_value
    path = (
        output_root / "optuna.db"
        if storage_value.lower() == "auto"
        else Path(storage_value).expanduser().resolve()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _import_optuna():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "Optuna is not installed. Install it (for example, "
            "`pip install optuna`) or use an environment that provides it."
        ) from exc
    return optuna


def _required_object(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(
            f"Expected copied trial config {key!r} to be an object. "
            f"Provided value: {item!r}."
        )
    return item


def _positive_sequence(value: Any, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"Expected {name} to be a non-empty JSON array. "
            f"Provided value: {value!r}."
        )
    resolved = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item <= 0.0 for item in resolved):
        raise ValueError(
            f"Expected every {name} value to be finite and positive for "
            f"logarithmic search. Provided value: {value!r}."
        )
    return resolved


def _validate_bounds(
    name: str,
    lower: float,
    upper: float,
    *,
    logarithmic: bool,
) -> None:
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower >= upper
    ):
        raise ValueError(
            f"Expected {name} bounds to be finite with minimum < maximum. "
            f"Provided value: minimum={lower!r}, maximum={upper!r}."
        )
    if logarithmic and lower <= 0.0:
        raise ValueError(
            f"Expected logarithmic {name} minimum to be positive. "
            f"Provided value: {lower!r}."
        )


if __name__ == "__main__":
    raise SystemExit(main())
