"""Bounded circuit/gradient checks, published through native EBL RunStore."""
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch
from scipy.special import softmax

from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from methods.energy_transformer import (
    PARAMETERS, finite_difference_gradient, gradient_errors, implicit_gradient,
    phase_gradient, update,
)
from networks.energy_transformer.circuit import (
    Circuit, Components, Parameters, dissipation, energy, equilibrium_reference,
    matrix_force, perturb, simulate, torch_energy,
)
from networks.energy_transformer.spice import executable_identity, run_spice

ROOT = Path(__file__).resolve().parents[2]


def _gradient_pass(errors, tolerance):
    return all(row["absolute_l2"] <= 1e-8 if row["relative_l2"] is None
               else row["relative_l2"] <= tolerance for row in errors.values())


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Declared CPU evidence budget exhausted")


def negative_controls(circuit, target):
    p = circuit.parameters
    changed = Circuit(perturb(p, "K", (0, 0), 0.08), circuit.components)
    state = np.linspace(0.2, 0.6, p.size)
    stale = changed.normalized_force(state, target, compensation=circuit.compensation)
    nominal = matrix_force(changed.parameters, state, target)
    stale_error = float(np.max(np.abs(stale-nominal)))
    # Fixed, analytically constructed counterexample, not a search for failure.
    q = Parameters(np.zeros((2, 1)), np.zeros((0, 1)), np.zeros(1),
                   np.array([-2., -1.]), np.zeros(0), p.kappa)
    s = torch.zeros(q.size, dtype=torch.float64, requires_grad=True)
    e = torch_energy(q, s, np.zeros(1))
    grad, = torch.autograd.grad(e, s)
    unequal_rc_velocity = np.array([0., -2., -10.])
    derivative = float(grad.detach().numpy()@unequal_rc_velocity)
    return {"frozen_compensation_force_error": stale_error,
            "unequal_logit_rc_energy_derivative": derivative,
            "frozen_compensation_detected": stale_error > 1e-4,
            "unequal_logit_rc_detected": derivative > 0}


def characterize_case(config, case, store, deadline):
    _deadline(deadline)
    p = Parameters(np.array(case.K), np.array(case.W).reshape(len(case.c), len(case.a)),
                   np.array(case.a), np.array(case.b), np.array(case.c), config.kappa)
    parts = Components(**{name: getattr(config, name) for name in Components.__dataclass_fields__})
    circuit, target = Circuit(p, parts), np.array(case.target)
    root = store.run_dir/"artifacts"/case.name
    root.mkdir()
    initial_states = [np.zeros(p.size), np.linspace(-.45, .55, p.size), np.linspace(.5, -.4, p.size)]
    reference_state = equilibrium_reference(p, target)
    d, l, m = p.sizes
    trajectories, spine = [], []
    metrics = {"branch_kcl_max_error": 0., "energy_identity_relative_error": 0.,
               "maximum_energy_derivative": -float("inf"), "maximum_energy_increment": 0.,
               "terminal_kcl_residual": 0., "equilibrium_start_disagreement": 0.,
               "spice_transient_max_error": 0., "spice_equilibrium_max_error": 0.,
               "spice_softmax_max_error": 0., "spice_current_max_error_normalized": 0.,
               "spice_terminal_kcl_residual": 0., "probability_sum_error": 0.}
    energies = []
    for index, initial in enumerate(initial_states):
        _deadline(deadline)
        trace = simulate(circuit, target, initial=initial)
        trajectories.append(trace)
        values = np.array([energy(p, state, target) for state in trace.states])
        energies.extend([float(values.min()), float(values.max())])
        metrics["maximum_energy_increment"] = max(metrics["maximum_energy_increment"], float(np.max(np.diff(values))))
        metrics["terminal_kcl_residual"] = max(metrics["terminal_kcl_residual"], trace.residual)
        metrics["equilibrium_start_disagreement"] = max(metrics["equilibrium_start_disagreement"], float(np.max(np.abs(trace.states[-1]-reference_state))))
        for state in trace.states[::50]:
            physical = circuit.normalized_force(state, target)
            compact = matrix_force(p, state, target)
            metrics["branch_kcl_max_error"] = max(metrics["branch_kcl_max_error"], float(np.max(np.abs(physical-compact))))
            raw = torch.tensor(state, dtype=torch.float64, requires_grad=True)
            grad, = torch.autograd.grad(torch_energy(p, raw, target), raw)
            actual = float(grad.detach().numpy()@circuit.rhs(0, state, target))
            expected = dissipation(circuit, state, target)
            metrics["energy_identity_relative_error"] = max(metrics["energy_identity_relative_error"], abs(actual-expected)/max(1, abs(actual), abs(expected)))
            metrics["maximum_energy_derivative"] = max(metrics["maximum_energy_derivative"], actual)
        np.savez_compressed(root/f"numerical-start-{index}.npz", time=trace.time,
                            states=trace.states, energy=values)
        spine.append((f"free-start-{index}", 0., initial))
    free = trajectories[0].states[-1]
    if m and np.min(np.abs(p.split(free)[2])) <= 1e-5:
        raise ValueError("ReLU equilibrium too close to a switching surface for smooth gradient reference")
    implicit = implicit_gradient(p, free, target)
    fd = finite_difference_gradient(p, target, config.finite_difference_step)
    metrics["finite_difference_gradient"] = gradient_errors(fd, implicit)
    beta_rows = []
    gradients = {f"implicit_{key}": value for key, value in implicit.items()}
    gradients.update({f"finite_difference_{key}": value for key, value in fd.items()})
    best_beta = min(config.betas)
    for beta in config.betas:
        _deadline(deadline)
        minus = simulate(circuit, target, -beta).states[-1]
        plus = simulate(circuit, target, beta).states[-1]
        center = phase_gradient(p, minus, plus, 2*beta)
        one_sided = phase_gradient(p, free, plus, beta)
        beta_rows.append({"beta": beta, "centered": gradient_errors(center, implicit),
                          "one_sided": gradient_errors(one_sided, implicit)})
        for key in PARAMETERS:
            gradients[f"centered_{beta:g}_{key}"] = center[key]
            gradients[f"one_sided_{beta:g}_{key}"] = one_sided[key]
        gradients[f"minus_state_{beta:g}"] = minus
        gradients[f"plus_state_{beta:g}"] = plus
    spine.extend([("minus", -best_beta, initial_states[0]), ("plus", best_beta, initial_states[0])])
    spice_phases = {}
    for name, nudge, initial in spine:
        _deadline(deadline)
        timeout = min(config.spice_timeout, max(0.001, deadline-time.monotonic()))
        trace = run_spice(circuit, target, root/name, executable=config.spice_executable,
                          nudge=nudge, initial=initial, timeout=timeout)
        numerical = simulate(circuit, target, nudge, initial, trace.time)
        metrics["spice_transient_max_error"] = max(metrics["spice_transient_max_error"], float(np.max(np.abs(trace.states-numerical.states))))
        metrics["spice_equilibrium_max_error"] = max(metrics["spice_equilibrium_max_error"], float(np.max(np.abs(trace.states[-1]-numerical.states[-1]))))
        ideal_prob = softmax(config.kappa*trace.states[:, d:d+l], axis=1)
        metrics["spice_softmax_max_error"] = max(metrics["spice_softmax_max_error"], float(np.max(np.abs(trace.probabilities-ideal_prob))))
        metrics["probability_sum_error"] = max(metrics["probability_sum_error"], float(np.max(np.abs(trace.probabilities.sum(axis=1)-1))))
        if np.min(trace.probabilities) <= 0:
            raise RuntimeError("Nonpositive BJT softmax probability")
        for state, measured_current in zip(trace.states[::50], trace.rail_currents[::50]):
            metrics["spice_current_max_error_normalized"] = max(metrics["spice_current_max_error_normalized"],
                float(np.max(np.abs(measured_current-circuit.rail_currents(state))))/config.conductance)
        output = (trace.states[-1, :d], trace.probabilities[-1], trace.rectified[-1])
        current = trace.rail_currents[-1]
        measured_force = (np.concatenate((p.a, p.b, p.c))-trace.states[-1]
                          +circuit.compensation*np.concatenate(output)
                          -(current[::2]-current[1::2])/(2*config.conductance))
        measured_force[:d] -= nudge*(output[0]-target)
        metrics["spice_terminal_kcl_residual"] = max(metrics["spice_terminal_kcl_residual"], float(np.max(np.abs(measured_force))))
        spice_phases[name] = (trace.states[-1], output)
        store.append_metric({"case": case.name, "stage": "spice", "phase": name})
    spice_gradient = phase_gradient(p, spice_phases["minus"][0], spice_phases["plus"][0],
                                    2*best_beta, spice_phases["minus"][1], spice_phases["plus"][1])
    metrics["spice_gradient"] = gradient_errors(spice_gradient, implicit)
    for key in PARAMETERS:
        gradients[f"spice_{key}"] = spice_gradient[key]
    # Input a stays clamped. Only K/W and hidden biases b/c are trainable here.
    training_gradient = {**spice_gradient, "a": np.zeros_like(p.a)}
    learned = update(circuit, training_gradient, config.learning_rate)
    learned_state = simulate(learned, target).states[-1]
    _deadline(deadline)
    learned_spice = run_spice(learned, target, root/"after-update",
                              executable=config.spice_executable,
                              timeout=min(config.spice_timeout, max(.001, deadline-time.monotonic())))
    metrics["updated_spice_equilibrium_error"] = float(np.max(np.abs(learned_spice.states[-1]-learned_state)))
    metrics["loss_before"] = float(0.5*np.sum((spice_phases["free-start-0"][0][:d]-target)**2))
    metrics["loss_after"] = float(0.5*np.sum((learned_spice.states[-1, :d]-target)**2))
    metrics["learning_signal"] = "centered EqProp from measured SPICE activation ports"
    metrics["learning_input_unchanged"] = bool(np.array_equal(learned.parameters.a, p.a))
    metrics["learning_rate"] = config.learning_rate
    metrics["minimum_stability_margin"] = min(p.stability_margin(-max(config.betas)), learned.parameters.stability_margin())
    metrics["relu_active_set_margin"] = float(np.min(np.abs(free[d+l:]))) if m else None
    metrics["free_relu_active_count"] = int(np.sum(free[d+l:] > 0))
    metrics["energy_min"], metrics["energy_max"] = min(energies), max(energies)
    metrics["beta_curve"] = beta_rows
    controls = negative_controls(circuit, target)
    final = next(row for row in beta_rows if row["beta"] == best_beta)
    checks = {
        "physical_kcl": metrics["branch_kcl_max_error"] <= 1e-10,
        "energy_identity": metrics["energy_identity_relative_error"] <= 1e-9,
        "energy_descent": metrics["maximum_energy_derivative"] <= 1e-9 and metrics["maximum_energy_increment"] <= 1e-9,
        "equilibrium": metrics["terminal_kcl_residual"] <= 1e-9 and metrics["equilibrium_start_disagreement"] <= 1e-7,
        "spice_transient": metrics["spice_transient_max_error"] <= 1e-4,
        "spice_equilibrium": metrics["spice_equilibrium_max_error"] <= 1e-5,
        "bjt_softmax": metrics["spice_softmax_max_error"] <= 5e-6 and metrics["probability_sum_error"] <= 5e-6,
        "spice_currents": metrics["spice_current_max_error_normalized"] <= 1e-5 and metrics["spice_terminal_kcl_residual"] <= 1e-7,
        "finite_difference_gradient": _gradient_pass(metrics["finite_difference_gradient"], 1e-6),
        "centered_eqprop": _gradient_pass(final["centered"], 2e-4),
        "one_sided_eqprop": _gradient_pass(final["one_sided"], 2e-2),
        "spice_eqprop": _gradient_pass(metrics["spice_gradient"], 2e-3),
        "learning": metrics["loss_after"] < metrics["loss_before"] and metrics["learning_input_unchanged"] and metrics["updated_spice_equilibrium_error"] <= 1e-5,
        "stable_regime": metrics["minimum_stability_margin"] > 0,
        "negative_controls": controls["frozen_compensation_detected"] and controls["unequal_logit_rc_detected"],
    }
    np.savez_compressed(root/"gradients.npz", **gradients)
    np.savez_compressed(root/"updated-parameters.npz", **{key: getattr(learned.parameters, key) for key in PARAMETERS})
    result = {"case": case.name, "checks": checks, "passed": all(checks.values()),
              "metrics": metrics, "negative_controls": controls,
              "coverage": {"free_starts": 3, "numerical_nudged_phases": 2*len(config.betas),
                           "spice_phases": len(spine)+1, "gradient_groups": list(PARAMETERS), "learning_steps": 1}}
    atomic_write_json(root/"summary.json", result)
    store.append_metric({"case": case.name, "stage": "complete", "passed": result["passed"]})
    print(f"CASE={case.name} PASSED={result['passed']}", flush=True)
    return result


def run_characterize(request):
    config = request.spec
    torch.set_num_threads(1)
    executable, version = executable_identity(config.spice_executable)
    deadline = time.monotonic()+config.wall_seconds
    all_passed = True
    # Each independently executable case owns a native bundle, even when one
    # public invocation requests several cases under the same wall budget.
    for case in config.cases:
        _deadline(deadline)
        resolved = {**asdict(config), "cases": [asdict(case)]}
        store = RunStore.create(output_root=request.output_dir, experiment_id=config.experiment_id,
                                resolved_config=resolved, command=request.command, repo_root=ROOT,
                                input_artifacts=[{"path": str(executable), "sha256": sha256_file(executable)}],
                                resume_capability="unsupported")
        print(f"RUN_DIR={store.run_dir}", flush=True)
        try:
            atomic_write_json(store.run_dir/"artifacts"/"spice-environment.json",
                              {"executable": str(executable), "sha256": sha256_file(executable),
                               "version": version, "numpy": np.__version__, "evidence": "exploratory-ideal-circuit"})
            result = characterize_case(config, case, store, deadline)
            _deadline(deadline)
            summary = {"schema": "analog-energy-transformer.result.v1", "status": "complete",
                       "passed": result["passed"], "cases": [result]}
            atomic_write_json(store.run_dir/"artifacts"/"summary.json", summary)
            files = sorted((store.run_dir/"artifacts").rglob("*"))
            artifacts = [store.artifact_record(path, kind="circuit_evidence") for path in files if path.is_file()]
            if sum(record.size_bytes for record in artifacts) > 100_000_000/len(config.cases):
                raise RuntimeError("Declared per-case share of artifact size budget exceeded")
            store.complete(metrics={"cases": 1, "case": case.name, "passed": result["passed"]}, artifacts=artifacts)
            all_passed = all_passed and result["passed"]
        except BaseException as error:
            store.fail(error)
            raise
    return 0 if all_passed else 1
