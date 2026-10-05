"""Golden cases for routing the MNIST DRN loops through ``training.core.engine``.

Every case drives a per-batch loop of ``mnist_relu_drn_kd.v1`` or
``mnist_relu_drn_reset*.v1`` (training, evaluation, calibration collection,
LR probing, deployed-array recovery, decomposition replay) on a tiny CPU DRN.
Each case records the trainable states at every settle, the gradients and
states before every update, the final states and gradients, the returned
metrics, and the global and loader RNG states.  A change in step order, RNG
consumption or any floating-point expression therefore changes the record.

Run ``python tests/training_loop_golden_cases.py capture`` on the reference
commit to (re)write ``tests/golden/training_loops``;
``tests/test_training_loop_golden.py`` replays and compares bit-exactly.
Failures record only the exception class: the migration may reword messages.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import DataLoader, TensorDataset

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ibm_om_golden_cases as om  # noqa: E402

GOLDEN_DIR = ROOT / "tests" / "golden" / "training_loops"
assert_same = om.assert_same
environment = om.environment

_LITERAL_GAIN = "literal_compact_repaired_noisy"  # forward_logit_gain=2.5
_RESET_GAIN = "reset_compact_q9_repaired"  # forward_logit_gain=1.25
_RECOVERY_SOURCE = om.RECOVERY_SOURCE_CASE  # reset_pulse_q9_repaired, gain 1.75


# --------------------------------------------------------------------------
# Recording helpers
# --------------------------------------------------------------------------


def _states(bindings) -> list[torch.Tensor]:
    return [binding.state.detach().clone() for binding in bindings]


def _grads(bindings) -> list[torch.Tensor | None]:
    return [
        None if binding.state.grad is None else binding.state.grad.detach().clone()
        for binding in bindings
    ]


@contextlib.contextmanager
def _recorded(stack, bindings, *, step_owner=None) -> Iterator[dict[str, Any]]:
    """Record states at every free/training settle and before every update."""

    record: dict[str, Any] = {"settle": [], "train_settle": [], "updates": []}
    patched: list[tuple[Any, str, Any]] = []

    def wrap(owner, name, key, before):
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            record[key].append(before())
            return original(*args, **kwargs)

        setattr(owner, name, wrapper)
        patched.append((owner, name, original))

    wrap(stack.minimizer, "compute_equilibrium", "settle", lambda: _states(bindings))
    if stack.training_minimizer is not stack.minimizer:
        wrap(
            stack.training_minimizer,
            "compute_equilibrium",
            "train_settle",
            lambda: _states(bindings),
        )
    owner = stack.optimizer if step_owner is None else step_owner
    wrap(
        owner,
        "step",
        "updates",
        lambda: {"grads": _grads(bindings), "states": _states(bindings)},
    )
    try:
        yield record
    finally:
        for target, name, original in reversed(patched):
            setattr(target, name, original)


def _outcome(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"ok": True, "value": fn()}
    except Exception as error:  # noqa: BLE001 - behaviour includes failures
        return {"ok": False, "type": type(error).__name__}


def _finish(record: dict[str, Any], bindings, generator=None) -> dict[str, Any]:
    record["final"] = _states(bindings)
    record["final_grads"] = _grads(bindings)
    record["global_rng"] = torch.get_rng_state().clone()
    if generator is not None:
        record["loader_rng"] = generator.get_state().clone()
    return record


# --------------------------------------------------------------------------
# Stacks, teachers and loaders
# --------------------------------------------------------------------------


def _seed_states(bindings, gmin: float, gmax: float, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for binding in bindings:
            binding.state.copy_(
                gmin
                + (gmax - gmin)
                * (0.05 + 0.9 * torch.rand(binding.state.shape, generator=generator))
            )


def _kd_stack(case_name: str = _LITERAL_GAIN):
    case = om.HWA_CASES[case_name]
    stack = om._tiny_stack(case)
    gmin, gmax = om._case_bounds(case)
    _seed_states(stack.bundle.catalog.trainable, gmin, gmax, seed=91)
    return stack


def _reset_stack(objective: str):
    from experiments.mnist_relu_drn_reset.components import (
        build_reset_student_stack,
    )

    case = om.HWA_CASES[_RESET_GAIN]
    gmin, gmax = om._case_bounds(case)
    spec = SimpleNamespace(
        runtime=SimpleNamespace(device="cpu"),
        model=SimpleNamespace(
            dims=(8, 8, 4),
            input_gain=1.0,
            non_linearity="perfect_diode",
            exponential_diode_param={},
            quadratic_diode_param={},
            hard_sigmoid_param={},
            voltage_amp=4.0,
            current_amp=0.25,
            conductance_min=gmin,
            conductance_max=gmax,
            encoding="single",
            include_biases=False,
        ),
        solver=SimpleNamespace(
            inference_iterations=4,
            training_iterations=4,
            mode="asynchronous",
            overrelaxation_factor=1.1,
        ),
        settings=SimpleNamespace(learning_rates=(0.05, 0.05), objective=objective),
    )
    stack = build_reset_student_stack(
        spec, device_data_path=None, enable_measured=False
    )
    _seed_states(stack.bundle.catalog.trainable, gmin, gmax, seed=93)
    return stack


def _data(count: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    return TensorDataset(
        torch.rand(count, 4, generator=generator),
        torch.randint(0, 2, (count,), generator=generator),
    )


def _train_loader(count: int = 10, batch_size: int = 4, seed: int = 5):
    generator = torch.Generator().manual_seed(1234)
    loader = DataLoader(
        _data(count, seed), batch_size=batch_size, shuffle=True, generator=generator
    )
    return loader, generator


def _eval_loader(count: int = 10, batch_size: int = 4, seed: int = 6):
    return DataLoader(_data(count, seed), batch_size=batch_size, shuffle=False)


def _teacher():
    return om._LinearTeacher()


@contextlib.contextmanager
def _workdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as raw:
        yield Path(raw)


def _ibm(case_name: str, bindings, workdir: Path):
    directory = workdir / case_name
    directory.mkdir(exist_ok=True)
    return om.build_hwa_modifier(
        om.HWA_CASES[case_name], directory, bindings=bindings
    )[0]


def _modifier(kind: str, stack, workdir: Path):
    from training.add_normal import AddNormalConfig, build_add_normal_modifier
    from training.core.modifier import SplitParameterModifier

    bindings = tuple(stack.bundle.catalog.trainable)
    if kind == "clean":
        return None
    if kind == "add_normal":
        return build_add_normal_modifier(
            stack.bundle.catalog.trainable_parameters,
            AddNormalConfig(std_dev=0.05, seed=17, noisy_evaluation=True),
        )
    if kind == "ibm_gain":
        return _ibm(_LITERAL_GAIN, bindings, workdir)
    if kind == "split_gain":
        return SplitParameterModifier(
            training=_ibm(_RESET_GAIN, bindings, workdir),
            evaluation=_ibm(_LITERAL_GAIN, bindings, workdir),
        )
    raise KeyError(kind)


def _modifier_state(modifier) -> Any:
    return None if modifier is None else om._snapshot(modifier.state_dict())


# --------------------------------------------------------------------------
# KD (mnist_relu_drn_kd.v1)
# --------------------------------------------------------------------------

KD_EVAL_CASES = {
    "clean": {"modifier": "clean"},
    "add_normal": {"modifier": "add_normal"},
    "ibm_gain": {"modifier": "ibm_gain"},
    "split_gain": {"modifier": "split_gain"},
    "sample_limit_boundary": {"modifier": "clean", "sample_limit": 8},
    "sample_limit_partial": {
        "modifier": "ibm_gain",
        "sample_limit": 6,
        "maximum_batches": 2,
    },
}


def run_kd_eval(name: str) -> dict[str, Any]:
    from experiments.mnist_relu_drn import runtime

    spec = KD_EVAL_CASES[name]
    stack = _kd_stack()
    bindings = tuple(stack.bundle.catalog.trainable)
    torch.manual_seed(2024)
    with _workdir() as workdir:
        modifier = _modifier(spec["modifier"], stack, workdir)
        with _recorded(stack, bindings) as record:
            results = [
                _outcome(
                    lambda: runtime._evaluate(
                        stack,
                        _teacher(),
                        _eval_loader(),
                        maximum_batches=spec.get("maximum_batches"),
                        sample_limit=spec.get("sample_limit"),
                        modifier=modifier,
                    )
                )
                for _repeat in range(2)
            ]
        record["results"] = results
        record["gain_after"] = float(stack.cost.gain)
        record["modifier_state"] = _modifier_state(modifier)
    return _finish(record, bindings)


KD_SELECTION_CASES = {
    "clean": {"selection": "clean", "modifier": "clean"},
    "modifier_repeats": {"selection": "modifier", "modifier": "ibm_gain"},
    "split_repeats": {"selection": "modifier", "modifier": "split_gain"},
}


def run_kd_selection(name: str) -> dict[str, Any]:
    from experiments.mnist_relu_drn import runtime

    case = KD_SELECTION_CASES[name]
    stack = _kd_stack()
    bindings = tuple(stack.bundle.catalog.trainable)
    torch.manual_seed(2025)
    spec = SimpleNamespace(
        settings=SimpleNamespace(
            selection_evaluation=case["selection"],
            selection_noise_repeats=2,
            max_validation_batches=None,
        )
    )
    with _workdir() as workdir:
        modifier = _modifier(case["modifier"], stack, workdir)
        with _recorded(stack, bindings) as record:
            record["result"] = _outcome(
                lambda: runtime._selection_evaluate(
                    stack, _teacher(), _eval_loader(), spec=spec, modifier=modifier
                )
            )
        record["modifier_state"] = _modifier_state(modifier)
    return _finish(record, bindings)


KD_TRAIN_CASES = {
    "clean": {"modifier": "clean"},
    "add_normal": {"modifier": "add_normal"},
    "ibm_gain": {"modifier": "ibm_gain"},
    "split_gain": {"modifier": "split_gain"},
}


def run_kd_train(name: str) -> dict[str, Any]:
    from experiments.mnist_relu_drn import runtime

    spec = KD_TRAIN_CASES[name]
    stack = _kd_stack()
    bindings = tuple(stack.bundle.catalog.trainable)
    torch.manual_seed(2026)
    loader, generator = _train_loader()
    with _workdir() as workdir:
        modifier = _modifier(spec["modifier"], stack, workdir)
        with _recorded(stack, bindings) as record:
            # Full epoch, an exact-length limit, then a truncating limit.
            record["epochs"] = [
                _outcome(
                    lambda maximum=maximum: runtime._train_epoch(
                        stack,
                        _teacher(),
                        loader,
                        maximum_batches=maximum,
                        modifier=modifier,
                    )
                )
                for maximum in (None, 3, 2)
            ]
        record["gain_after"] = float(stack.cost.gain)
        record["modifier_state"] = _modifier_state(modifier)
    return _finish(record, bindings, generator)


def run_kd_train_nonfinite() -> dict[str, Any]:
    from experiments.mnist_relu_drn import runtime

    stack = _kd_stack()
    bindings = tuple(stack.bundle.catalog.trainable)
    torch.manual_seed(2027)
    dataset = _data(8, seed=8)
    dataset.tensors[0][5, 1] = float("nan")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    with _recorded(stack, bindings) as record:
        record["outcome"] = _outcome(
            lambda: runtime._train_epoch(
                stack, _teacher(), loader, maximum_batches=None, modifier=None
            )
        )
    return _finish(record, bindings)


KD_COLLECT_CASES = ("clean", "ibm_evaluation_context")


def run_kd_collect(name: str) -> dict[str, Any]:
    from experiments.mnist_relu_drn.components import collect_calibration

    stack = _kd_stack()
    bindings = tuple(stack.bundle.catalog.trainable)
    torch.manual_seed(2028)
    with _workdir() as workdir:
        modifier = _modifier("ibm_gain", stack, workdir)
        context = (
            modifier.evaluation_context
            if name == "ibm_evaluation_context"
            else contextlib.nullcontext
        )
        with _recorded(stack, bindings) as record:
            with context():
                record["result"] = _outcome(
                    lambda: collect_calibration(stack, _teacher(), _eval_loader())
                )
    return _finish(record, bindings)


KD_TRACE_CASES = {
    "full": {"maximum_batches": None, "sample_limit": None},
    "sample_limit": {"maximum_batches": None, "sample_limit": 6},
    "batch_limit": {"maximum_batches": 2, "sample_limit": None},
}


def run_kd_trace(name: str) -> dict[str, Any]:
    from experiments.mnist_relu_drn.ibm_om_deployment_decomposition import (
        collect_evaluation_trace,
    )

    spec = KD_TRACE_CASES[name]
    stack = _kd_stack()
    stack.cost.gain = 1.5
    bindings = tuple(stack.bundle.catalog.trainable)
    torch.manual_seed(2029)
    with _recorded(stack, bindings) as record:
        record["result"] = _outcome(
            lambda: asdict(
                collect_evaluation_trace(
                    stack,
                    _teacher(),
                    _eval_loader(),
                    maximum_batches=spec["maximum_batches"],
                    sample_limit=spec["sample_limit"],
                )
            )
        )
    return _finish(record, bindings)


# --------------------------------------------------------------------------
# Deployed-array recovery (mnist_relu_drn_kd.v1 update_backend)
# --------------------------------------------------------------------------

RECOVERY_LOOP_CASES = {
    "rail_refresh": {"method": "rail_refresh"},
    "direct_pulse": {"method": "direct_pulse"},
    # The p90 gate leaves ~10% of cells eligible, so the pulse budget must be
    # small enough for the bisected probability scale to reach it.
    "direct_pulse_p90": {"method": "direct_pulse", "percentile": 90.0, "budget": 0.25},
}

_RECOVERY_EPOCHS = 2
_RECOVERY_BATCHES = 2


def _stack_recovery(spec: dict[str, Any], workdir: Path):
    from training.ibm_reram_recovery import IbmOmDeployedRecovery

    case = om.HWA_CASES[_RECOVERY_SOURCE]
    stack = _kd_stack(_RECOVERY_SOURCE)
    bindings = tuple(stack.bundle.catalog.trainable)
    modifier, _bindings, population, device_model = om.build_hwa_modifier(
        case, workdir, bindings=bindings
    )
    with modifier.evaluation_context():
        pass
    deployment = modifier.last_deployment_bundle
    gmin, gmax = om._case_bounds(case)
    parameters = om._recovery_parameters(
        spec["method"],
        fast_backend=None,
        device_model=device_model,
        percentile=spec.get("percentile"),
    )
    if "budget" in spec:
        parameters["slow_pulse_budget_per_cell"] = spec["budget"]
    recovery = IbmOmDeployedRecovery(
        bindings,
        slow_population=population,
        source_raw_apparent=deployment["raw_apparent_endpoint"],
        source_apparent=deployment["apparent_endpoint"],
        source_persistent=deployment["persistent_endpoint"],
        requested_target=deployment.get(
            "mapped_target", deployment.get("requested_target")
        ),
        reset_baseline=deployment["reset_commissioning"]["observed"]["baseline"],
        generator_state_after_programming=deployment[
            "generator_state_after_programming"
        ],
        parameters=parameters,
        learning_rates=(0.6, 0.4),
        conductance_min=gmin,
        conductance_max=gmax,
        total_steps=_RECOVERY_EPOCHS * _RECOVERY_BATCHES,
        source_deployment_sha256="d" * 64,
        device_model_path=device_model,
        fast_population=None,
    )
    return stack, bindings, recovery


def run_recovery_loop(name: str) -> dict[str, Any]:
    from experiments.mnist_relu_drn import ibm_om_deployed_recovery as deployed
    from experiments.mnist_relu_drn import runtime

    spec = RECOVERY_LOOP_CASES[name]
    torch.manual_seed(2030)
    with _workdir() as workdir:
        stack, bindings, recovery = _stack_recovery(spec, workdir)
        stack.cost.gain = 1.75
        loader, generator = _train_loader(count=8, batch_size=4)
        with _recorded(stack, bindings, step_owner=recovery) as record:
            if recovery.requires_gradients:
                record["calibration"] = _outcome(
                    lambda: deployed._calibrate_direct_probabilities(
                        stack,
                        _teacher(),
                        loader,
                        generator,
                        recovery,
                        calibration_batches=_RECOVERY_BATCHES,
                        frozen_scale=None,
                    )
                )
                record["after_calibration"] = {
                    "grads": _grads(bindings),
                    "loader_rng": generator.get_state().clone(),
                    "global_rng": torch.get_rng_state().clone(),
                }
                training = replace(stack, optimizer=recovery)
                record["epochs"] = [
                    _outcome(
                        lambda: runtime._train_epoch(
                            training,
                            _teacher(),
                            loader,
                            maximum_batches=None,
                            modifier=None,
                        )
                    )
                    for _epoch in range(_RECOVERY_EPOCHS)
                ]
            else:
                record["epochs"] = [
                    _outcome(
                        lambda: deployed._refresh_epoch(
                            stack,
                            _teacher(),
                            loader,
                            recovery,
                            maximum_batches=None,
                        )
                    )
                    for _epoch in range(_RECOVERY_EPOCHS)
                ]
        record["step_index"] = int(recovery.step_index)
        record["report"] = om._snapshot(recovery.report())
        record["state"] = om._snapshot(recovery.state_dict())
        record["apparent"] = recovery.current_apparent_endpoint(clipped=False).clone()
        record["persistent"] = recovery.current_persistent_endpoint().clone()
    return _finish(record, bindings, generator)


# --------------------------------------------------------------------------
# RESET (mnist_relu_drn_reset*.v1)
# --------------------------------------------------------------------------

_SAFETY = {
    "warmup_batches": 1,
    "loss_ema_decay": 0.5,
    "loss_growth_factor": 4.0,
    "gradient_growth_factor": 100.0,
    "persistence_batches": 2,
}

RESET_TRAIN_CASES = {
    "kl_reset_observer": {"objective": "teacher_kl", "reset_input": True, "observer": True},
    "ce_carry": {"objective": "cross_entropy", "reset_input": False, "observer": False},
    "pse_carry_observer": {
        "objective": "paired_squared_error",
        "reset_input": False,
        "observer": True,
    },
}


def run_reset_train(name: str) -> dict[str, Any]:
    from experiments.mnist_relu_drn_reset import runtime
    from experiments.mnist_relu_drn_reset.learning_rate_selection import (
        SafetyMonitor,
    )

    spec = RESET_TRAIN_CASES[name]
    stack = _reset_stack(spec["objective"])
    bindings = tuple(stack.bundle.catalog.trainable)
    torch.manual_seed(2031)
    loader, generator = _train_loader()
    monitor = (
        SafetyMonitor(tuple(binding.key for binding in bindings), _SAFETY)
        if spec["observer"]
        else None
    )
    with _recorded(stack, bindings) as record:
        record["epochs"] = [
            _outcome(
                lambda maximum=maximum: runtime.train_epoch(
                    stack,
                    _teacher(),
                    loader,
                    maximum_batches=maximum,
                    reset_input=spec["reset_input"],
                    observer=None if monitor is None else monitor.observe,
                )
            )
            for maximum in (None, 2)
        ]
    if monitor is not None:
        record["safety"] = om._snapshot(monitor.summary())
    return _finish(record, bindings, generator)


def run_reset_train_nonfinite() -> dict[str, Any]:
    from experiments.mnist_relu_drn_reset import runtime
    from experiments.mnist_relu_drn_reset.learning_rate_selection import (
        SafetyMonitor,
    )

    stack = _reset_stack("teacher_kl")
    bindings = tuple(stack.bundle.catalog.trainable)
    torch.manual_seed(2032)
    dataset = _data(8, seed=9)
    dataset.tensors[0][6, 0] = float("nan")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    monitor = SafetyMonitor(tuple(binding.key for binding in bindings), _SAFETY)
    with _recorded(stack, bindings) as record:
        record["outcome"] = _outcome(
            lambda: runtime.train_epoch(
                stack,
                _teacher(),
                loader,
                maximum_batches=None,
                reset_input=True,
                observer=monitor.observe,
            )
        )
    return _finish(record, bindings)


RESET_EVAL_CASES = {
    "kl": {"objective": "teacher_kl"},
    "ce_diagnostic_gain": {"objective": "cross_entropy", "diagnostic_gain": 1.7},
    "pse_sample_limit": {
        "objective": "paired_squared_error",
        "sample_limit": 6,
        "maximum_batches": 2,
    },
    "kl_sample_limit_boundary": {"objective": "teacher_kl", "sample_limit": 8},
}


def run_reset_eval(name: str) -> dict[str, Any]:
    from experiments.mnist_relu_drn_reset import runtime

    spec = RESET_EVAL_CASES[name]
    stack = _reset_stack(spec["objective"])
    bindings = tuple(stack.bundle.catalog.trainable)
    torch.manual_seed(2033)
    with _recorded(stack, bindings) as record:
        record["result"] = _outcome(
            lambda: runtime.evaluate(
                stack,
                _teacher(),
                _eval_loader(),
                maximum_batches=spec.get("maximum_batches"),
                sample_limit=spec.get("sample_limit"),
                diagnostic_gain=spec.get("diagnostic_gain"),
            )
        )
    return _finish(record, bindings)


def run_reset_collect() -> dict[str, Any]:
    from experiments.mnist_relu_drn_reset import runtime

    stack = _reset_stack("teacher_kl")
    bindings = tuple(stack.bundle.catalog.trainable)
    torch.manual_seed(2034)
    with _recorded(stack, bindings) as record:
        record["result"] = _outcome(
            lambda: runtime._collect_logits(stack, _teacher(), _eval_loader())
        )
        record["posthoc"] = _outcome(
            lambda: runtime._posthoc_calibration(stack, _teacher(), _eval_loader())
        )
    return _finish(record, bindings)


RESET_PROBE_CASES = {
    "stops_early": {"tolerance": 1000.0, "reset_input": None},
    "runs_to_end": {"tolerance": 1e-12, "reset_input": False},
    "explicit_reset": {"tolerance": 1000.0, "reset_input": True},
}


def run_reset_probe(name: str) -> dict[str, Any]:
    from experiments.mnist_relu_drn_reset.learning_rate_selection import _run_probe

    spec = RESET_PROBE_CASES[name]
    stack = _reset_stack("teacher_kl")
    bindings = tuple(stack.bundle.catalog.trainable)
    torch.manual_seed(2035)
    loader, generator = _train_loader(count=12, batch_size=2)
    settings = {
        "probe_batches": [2, 4, 6],
        "stability_tolerance": spec["tolerance"],
        "bias_quantile": 0.9,
    }
    with _recorded(stack, bindings) as record:
        record["result"] = _outcome(
            lambda: _run_probe(
                stack,
                _teacher(),
                loader,
                settings,
                training_reset_input=spec["reset_input"],
            )
        )
    return _finish(record, bindings, generator)


# --------------------------------------------------------------------------
# Registry, capture
# --------------------------------------------------------------------------


def all_cases() -> dict[str, Callable[[], Any]]:
    cases: dict[str, Callable[[], Any]] = {}
    for name in KD_EVAL_CASES:
        cases[f"kd_eval/{name}"] = lambda name=name: run_kd_eval(name)
    for name in KD_SELECTION_CASES:
        cases[f"kd_selection/{name}"] = lambda name=name: run_kd_selection(name)
    for name in KD_TRAIN_CASES:
        cases[f"kd_train/{name}"] = lambda name=name: run_kd_train(name)
    cases["kd_train/nonfinite"] = run_kd_train_nonfinite
    for name in KD_COLLECT_CASES:
        cases[f"kd_collect/{name}"] = lambda name=name: run_kd_collect(name)
    for name in KD_TRACE_CASES:
        cases[f"kd_trace/{name}"] = lambda name=name: run_kd_trace(name)
    for name in RECOVERY_LOOP_CASES:
        cases[f"recovery/{name}"] = lambda name=name: run_recovery_loop(name)
    for name in RESET_TRAIN_CASES:
        cases[f"reset_train/{name}"] = lambda name=name: run_reset_train(name)
    cases["reset_train/nonfinite"] = run_reset_train_nonfinite
    for name in RESET_EVAL_CASES:
        cases[f"reset_eval/{name}"] = lambda name=name: run_reset_eval(name)
    cases["reset_collect/logits"] = run_reset_collect
    for name in RESET_PROBE_CASES:
        cases[f"reset_probe/{name}"] = lambda name=name: run_reset_probe(name)
    return cases


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.pt"


def capture(selected: list[str] | None = None) -> None:
    cases = all_cases()
    names = selected or list(cases)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    for name in names:
        result = cases[name]()
        path = golden_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, path)
        print(f"captured {name} -> {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    meta = {**environment(), "commit": commit, "cases": sorted(cases)}
    (GOLDEN_DIR / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "capture":
        capture(sys.argv[2:] or None)
    else:
        print("usage: python tests/training_loop_golden_cases.py capture [case ...]")
        print("\n".join(sorted(all_cases())))
