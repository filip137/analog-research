"""Independent circuit, energy, equilibrium-gradient and EBL boundary checks."""
from dataclasses import replace
import json
from pathlib import Path
import os

import numpy as np
import pytest
import torch

from ebl.schema.base import ConfigError, RunMode
from ebl.schema.analog_energy_transformer import parse_config, resolve_config
from methods.energy_transformer import (eqprop, finite_difference_gradient, gradient_errors,
                                         implicit_gradient, update)
from networks.energy_transformer.circuit import (Circuit, Components, Parameters, dissipation,
    equilibrium_reference, matrix_force, simulate, torch_energy)
from networks.energy_transformer.spice import export_netlist, run_spice

ROOT = Path(__file__).resolve().parents[1]


def fixture_parameters():
    return Parameters(np.array([[.2, -.1], [-.15, .12], [.08, .16]]),
                      np.array([[.2, -.1], [-.1, .15]]), np.array([-.2, .25]),
                      np.array([.1, -.1, .05]), np.array([.3, -.6]))


@pytest.mark.parametrize("nudge", [-.04, 0., .04])
def test_physical_currents_equal_kcl_and_decrease_independent_energy(nudge):
    p = fixture_parameters()
    c = Circuit(p)
    target = np.array([.2, -.1])
    rng = np.random.default_rng(811)
    for state in rng.normal(size=(12, p.size)):
        physical = c.normalized_force(state, target, nudge)
        np.testing.assert_allclose(physical, matrix_force(p, state, target, nudge), atol=1e-14)
        raw = torch.tensor(state, dtype=torch.float64, requires_grad=True)
        gradient, = torch.autograd.grad(torch_energy(p, raw, target, nudge), raw)
        slope = float(gradient.detach().numpy()@c.rhs(0, state, target, nudge))
        assert slope <= 1e-10
        assert slope == pytest.approx(dissipation(c, state, target, nudge), rel=1e-12, abs=1e-10)


def test_phase_gradients_match_implicit_and_loss_differences():
    p = fixture_parameters()
    c = Circuit(p)
    target = np.array([.2, -.1])
    state = simulate(c, target).states[-1]
    np.testing.assert_allclose(state, equilibrium_reference(p, target), atol=1e-10)
    reference = implicit_gradient(p, state, target)
    for gradients, tolerance in ((finite_difference_gradient(p, target), 1e-6),
                                  (eqprop(c, target, .005)[0], 2e-4)):
        for row in gradient_errors(gradients, reference).values():
            assert row["absolute_l2"] < 1e-8 if row["relative_l2"] is None else row["relative_l2"] < tolerance
    gradient = eqprop(c, target, .005)[0]
    gradient["a"] = np.zeros_like(p.a)
    changed = update(c, gradient, .1)
    assert np.array_equal(changed.parameters.a, p.a)
    final = simulate(changed, target).states[-1]
    assert np.sum((final[:2]-target)**2) < np.sum((state[:2]-target)**2)


def test_sign_changes_rebuild_devices_and_compensation():
    c = Circuit(fixture_parameters())
    gradients = {name: np.zeros_like(getattr(c.parameters, name)) for name in ("K", "W", "a", "b", "c")}
    gradients["K"][0, 0] = 3.5  # +0.20 -> -0.15 at eta=0.1
    changed = update(c, gradients, .1)
    assert changed.parameters.K[0, 0] < 0
    assert np.min([edge[2] for edge in changed.edges]) > 0
    state, target = np.linspace(-.3, .5, c.parameters.size), np.array([.2, -.1])
    np.testing.assert_allclose(changed.normalized_force(state, target), matrix_force(changed.parameters, state, target), atol=1e-14)
    assert np.max(np.abs(changed.normalized_force(state, target, compensation=c.compensation)-matrix_force(changed.parameters, state, target))) > 1e-4


def test_invalid_physical_contract_and_stability_fail_closed():
    p = fixture_parameters()
    with pytest.raises(ValueError):
        Components(logit_capacitance=np.array([1., 2.]))
    with pytest.raises(ValueError, match="stability"):
        simulate(Circuit(replace(p, W=10*p.W)), np.zeros(2))
    with pytest.raises(ValueError, match="Target"):
        simulate(Circuit(p), np.zeros(3))
    with pytest.raises(RuntimeError, match="did not settle"):
        simulate(Circuit(p), np.zeros(2), times=np.array([0., 1e-5]))
    from ebl.runtime.analog_energy_transformer import negative_controls
    controls = negative_controls(Circuit(p), np.zeros(2))
    assert controls["frozen_compensation_detected"]
    assert controls["unequal_logit_rc_detected"]


def test_strict_schema_and_public_registry():
    from ebl.registry import resolve_experiment_config
    path = ROOT/"examples/analog_energy_transformer_circuit.json"
    raw = json.loads(path.read_text())
    config = parse_config(raw)
    assert len(config.cases) == 3
    definition, spec = resolve_experiment_config(path, RunMode.CHARACTERIZE)
    assert definition.experiment_id == config.experiment_id
    assert spec == config
    for malformed in ({**raw, "unknown": 1}, {**raw, "wall_seconds": 601},
                      {**raw, "logit_capacitance": [1e-7, 2e-7]},
                      {**raw, "kappa": float("nan")}, {**raw, "betas": [0]},
                      {**raw, "schema_version": True}, {**raw, "cases": raw["cases"]*2}):
        with pytest.raises(ConfigError):
            parse_config(malformed)
    with pytest.raises(ConfigError):
        resolve_config(config, RunMode.TRAIN)


def test_netlist_has_explicit_shared_components_and_no_matrix_behavioral_force():
    circuit = Circuit(fixture_parameters())
    lines = export_netlist(circuit, np.zeros(2)).splitlines()
    assert sum(line.startswith("Rweight") for line in lines) == 4*(6+4)
    assert sum(line.startswith("Qsoft") for line in lines) == 3
    assert sum(line.startswith("Cstate") for line in lines) == circuit.parameters.size
    assert all(line.startswith("Brect") for line in lines if line.startswith("B"))
    assert any(line.startswith("Fportp") for line in lines)
    assert any(line.startswith("Gcomp") for line in lines)


def test_spice_zero_coupling_has_analytic_equilibrium(tmp_path):
    executable = os.environ.get("ENERGY_TRANSFORMER_NGSPICE")
    if not executable:
        pytest.skip("Set ENERGY_TRANSFORMER_NGSPICE for the explicit circuit integration test")
    p = Parameters(np.zeros((2, 2)), np.zeros((0, 2)), np.array([.1, -.2]),
                   np.array([-.2, .3]), np.zeros(0))
    target = np.array([-.1, .1])
    trace = run_spice(Circuit(p), target, tmp_path/"spice", executable=executable, nudge=.02)
    expected = np.concatenate(((p.a+.02*target)/1.02, p.b))
    np.testing.assert_allclose(trace.states[-1], expected, atol=1e-7)
    np.testing.assert_allclose(trace.probabilities[-1], p.activations(expected)[1], atol=1e-7)


def test_each_requested_case_has_its_own_native_bundle(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ebl.runtime import analog_energy_transformer as runtime
    config = parse_config(json.loads((ROOT/"examples/analog_energy_transformer_circuit.json").read_text()))
    executable = tmp_path/"tool"
    executable.write_text("fake executable identity for artifact test")
    monkeypatch.setattr(runtime, "executable_identity", lambda _: (executable, "fixture"))
    monkeypatch.setattr(runtime, "characterize_case", lambda config, case, store, deadline:
                        {"case": case.name, "passed": True})
    assert runtime.run_characterize(SimpleNamespace(spec=config, output_dir=tmp_path/"runs", command=("fixture",))) == 0
    bundles = list((tmp_path/"runs").iterdir())
    assert len(bundles) == 3
    identities = []
    for bundle in bundles:
        resolved = json.loads((bundle/"config.resolved.json").read_text())
        assert len(resolved["cases"]) == 1
        identities.append(resolved["cases"][0]["name"])
        assert json.loads((bundle/"result.json").read_text())["status"] == "complete"
    assert set(identities) == {case.name for case in config.cases}
