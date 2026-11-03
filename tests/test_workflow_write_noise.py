"""Write-only noise factor for the CMO update laws: physical identity and the exact-write oracle."""
import json
from pathlib import Path

import pytest
import torch

from core.contracts import ConfigError
from hardware.devices.cmo import ACTIVE, HALF_RANGE_US, REFERENCE, CmoMatrix
from ebl.schema.lifecycle import parse_lifecycle
from methods.onchip import IdealizedWriteAdam, ThresholdedWriteAdam, _adam_command
from test_workflow_cmo import cmo_inputs, small, template, case_doc
from workflow_helpers import run_lifecycle

ROOT = Path(__file__).resolve().parents[1]
DELTA = torch.tensor([.01, -.02, .03, 0.])


def faulty(seed=7):
    """Index 0 has a stuck active device, index 1 a stuck reference."""

    mask = torch.tensor([[True, False, False, False], [False, True, False, False]])
    x = CmoMatrix(4, seed, 2., 1., "cpu", (mask, torch.full((2, 4), 30.)))
    x.program(torch.zeros(4))
    return x


def thresholded(array, scale=None):
    return ThresholdedWriteAdam(array, learning_rate=.003, threshold_sigma=1, accumulate_updates=True,
                                acceptance_percent=2, noise_scale=1., write_noise_scale=scale)


def without_rewrite(doc):
    doc["onchip"]["arms"] = [arm for arm in doc["onchip"]["arms"] if arm["weights"] != "rewrite"]
    return doc


def test_unit_factor_is_bitwise_the_physical_write_law():
    physical, declared = faulty(), faulty()
    physical.write(DELTA); declared.write(DELTA, 1.0)
    assert torch.equal(physical.g, declared.g)
    assert torch.equal(physical.generator.get_state(), declared.generator.get_state())
    assert physical.cost() == declared.cost()


def test_zero_factor_writes_exactly_and_keeps_faults_counters_and_stream():
    physical, exact = faulty(), faulty()
    initial = exact.g.clone()
    physical.write(DELTA); exact.write(DELTA, 0.0)
    healthy = torch.tensor([False, True, True, False])
    assert torch.equal(exact.g[ACTIVE, healthy], initial[ACTIVE, healthy] + DELTA[healthy] * HALF_RANGE_US)
    assert exact.g[ACTIVE, 0] == initial[ACTIVE, 0] == 30.
    assert torch.equal(exact.g[REFERENCE], initial[REFERENCE])
    assert exact.g[ACTIVE, 3] == initial[ACTIVE, 3]
    assert not torch.equal(physical.g[ACTIVE, healthy], exact.g[ACTIVE, healthy])
    assert exact.cost() == physical.cost() and torch.equal(exact.writes, physical.writes)
    assert torch.equal(exact.generator.get_state(), physical.generator.get_state())


@pytest.mark.parametrize("factor", [-1., float("nan"), float("inf"), True])
def test_device_rejects_invalid_write_noise_factors(factor):
    with pytest.raises(ValueError):
        faulty().write(DELTA, factor)


def test_threshold_and_issued_commands_do_not_depend_on_write_noise(monkeypatch):
    import methods.onchip as module
    physical, exact = faulty(), faulty()
    initial = exact.g.clone()
    wp, we = thresholded(physical), thresholded(exact, 0.)
    assert we.threshold_q == wp.threshold_q == thresholded(faulty(), 1.).threshold_q
    commands = [torch.tensor([.02, .01, -.02, .005]) for _ in range(3)]
    sequence = iter(commands + commands)
    monkeypatch.setattr(module, "_adam_command", lambda *_: next(sequence))
    for _ in commands:
        wp.step(None)
    for _ in commands:
        we.step(None)
    assert torch.equal(wp.pending, we.pending) and physical.cost() == exact.cost()
    issued = torch.stack(commands).sum(0) - we.pending
    healthy = torch.tensor([False, True, True, True])
    assert torch.allclose(exact.g[ACTIVE, healthy], initial[ACTIVE, healthy] + issued[healthy] * HALF_RANGE_US)
    assert we.diagnostics()["write_noise_scale"] == 0. and "write_noise_scale" not in wp.diagnostics()


def test_dense_exact_writer_applies_the_digital_adam_trajectory():
    array = faulty(); writer = IdealizedWriteAdam(array, learning_rate=.003, write_noise_scale=0.)
    digital = IdealizedWriteAdam(faulty(), learning_rate=.003)
    start = array.read(); total = torch.zeros(4)
    for step in range(5):
        gradient = torch.tensor([1., -2., .5, 3.]) * (step + 1)
        total += _adam_command(digital, gradient)
        writer.step(gradient)
    expected = start + total
    # The stuck active device (index 0) keeps its weight; the others follow digital Adam.
    assert torch.allclose(array.read()[1:], expected[1:], atol=1e-6)
    assert array.read()[0] == start[0]
    assert writer.state_dict()["write_noise_scale"] == 0.
    assert "write_noise_scale" not in digital.state_dict()


@pytest.mark.parametrize("law", ["cmo_idealized_write_adam", "cmo_thresholded_write_adam"])
def test_schema_records_write_noise_only_when_declared_and_rejects_rewrite(law):
    doc = without_rewrite(template())
    doc["onchip"]["update_law"] = law
    if law == "cmo_thresholded_write_adam":
        doc["onchip"].update(threshold_sigma=2., accumulate_updates=True)
    original = parse_lifecycle(doc)
    assert "write_noise_scale" not in original.to_dict()["onchip"]
    doc["onchip"]["write_noise_scale"] = 0
    oracle = parse_lifecycle(doc)
    assert oracle.onchip.write_noise_scale == 0. and oracle.to_dict()["onchip"]["write_noise_scale"] == 0.
    assert parse_lifecycle(oracle.to_dict()) == oracle and oracle.digest() != original.digest()
    del doc["onchip"]["write_noise_scale"]
    assert parse_lifecycle(doc).digest() == original.digest()
    rewrite = template(); rewrite["onchip"].update(doc["onchip"], arms=template()["onchip"]["arms"], write_noise_scale=0)
    with pytest.raises(ConfigError, match="rewrite"):
        parse_lifecycle(rewrite)


@pytest.mark.parametrize("value", [-1, float("nan"), True, "0"])
def test_schema_rejects_invalid_write_noise_scale(value):
    doc = without_rewrite(template()); doc["onchip"]["write_noise_scale"] = value
    with pytest.raises(ConfigError):
        parse_lifecycle(doc)


def test_schema_rejects_write_noise_scale_outside_cmo_laws():
    doc = json.loads((ROOT / "examples/lifecycles/opt125m-om-mlp4-template.json").read_text())
    doc["onchip"]["write_noise_scale"] = 0
    with pytest.raises(ConfigError):
        parse_lifecycle(doc)


def test_exp013_lifecycles_keep_their_onchip_sections():
    for path in sorted((ROOT / "campaigns/opt-cmo-corruption/lifecycles").glob("opt125m-cmo-l6-exp7-recovery-regimes-*.json")):
        doc = json.loads(path.read_text())
        assert parse_lifecycle(doc).to_dict()["onchip"] == doc["onchip"], path.name


@pytest.mark.parametrize("law", ["cmo_idealized_write_adam", "cmo_thresholded_write_adam"])
def test_exact_write_lifecycle_replays(cmo_inputs, tmp_path, law):
    weights, digest = cmo_inputs
    options = {
        "data.corpus": {"name": "synthetic", "sha256": digest},
        "data.sequence_length": 9, "data.max_examples": 8,
        "defects.cases": [case_doc("random", 50000)], "hwa.arms": [{"id": "digital", "method": "none"}],
        "onchip.batch_size": 4, "onchip.epochs": 1, "onchip.learning_rate": .01,
        "onchip.update_law": law, "onchip.write_noise_scale": 0., "onchip.development_every_steps": 1,
        "onchip.arms": [{"id": "none", "weights": "hold", "calibration": False},
                        {"id": "onchip_weights", "weights": "learn", "calibration": False}],
    }
    if law == "cmo_thresholded_write_adam":
        options.update({"onchip.threshold_sigma": 1., "onchip.accumulate_updates": True})
    lc = small(**options)
    assert parse_lifecycle(lc.to_dict()) == lc
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    run, result = results["onchip__digital__271001"]
    assert result["metrics"]["exact_replays"] == 2
    metrics = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
    checks = [m for m in metrics if m.get("stage") == "onchip_development"]
    assert checks and all(m["cost"]["verify_reads"] == 0 for m in checks)
    if law == "cmo_thresholded_write_adam":
        assert all(m["writer"]["write_noise_scale"] == 0. for m in checks)
