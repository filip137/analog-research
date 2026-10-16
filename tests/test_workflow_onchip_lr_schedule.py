"""Declared learning-rate schedule for CMO on-chip weight learning (``onchip.lr_schedule``)."""
import json
from pathlib import Path

import pytest
import torch

from experiments.schema import ConfigError
from workflow.lifecycle import parse_lifecycle
from workflow.onchip import IdealizedWriteAdam, _adam_command
from test_workflow_cmo import cmo_inputs, small, template, case_doc
from test_workflow_write_noise import faulty, thresholded, without_rewrite
from workflow_helpers import run_lifecycle

ROOT = Path(__file__).resolve().parents[1]
COSINE = {"kind": "cosine_warmup_decay", "total_steps": 2, "warmup_steps": 1, "min_factor": 0.0}


def scheduled(law="cmo_idealized_write_adam"):
    doc = without_rewrite(template())
    doc["onchip"]["update_law"] = law
    if law == "cmo_thresholded_write_adam":
        doc["onchip"].update(threshold_sigma=2., accumulate_updates=True)
    return doc


def steps(doc):
    onchip, data = doc["onchip"], doc["data"]
    cohort = min(data["recovery_sequences"], data["max_examples"] or data["recovery_sequences"])
    return onchip["epochs"] * -(-cohort // onchip["batch_size"])


@pytest.mark.parametrize("law", ["cmo_idealized_write_adam", "cmo_thresholded_write_adam"])
def test_schema_records_a_schedule_only_when_declared(law):
    doc = scheduled(law)
    original = parse_lifecycle(doc)
    assert "lr_schedule" not in original.to_dict()["onchip"] and original.onchip.lr_schedule is None
    total = steps(doc)
    doc["onchip"]["lr_schedule"] = dict(COSINE, total_steps=total, warmup_steps=min(3, total - 1))
    declared = parse_lifecycle(doc)
    assert declared.to_dict()["onchip"]["lr_schedule"] == doc["onchip"]["lr_schedule"]
    assert parse_lifecycle(declared.to_dict()) == declared and declared.digest() != original.digest()
    del doc["onchip"]["lr_schedule"]
    assert parse_lifecycle(doc).digest() == original.digest()


def test_schema_requires_the_full_onchip_budget():
    doc = scheduled()
    doc["onchip"]["lr_schedule"] = dict(COSINE, total_steps=steps(doc) + 1)
    with pytest.raises(ConfigError, match="total_steps"):
        parse_lifecycle(doc)


def test_schema_rejects_a_schedule_outside_cmo_laws():
    doc = json.loads((ROOT / "examples/lifecycles/opt125m-om-mlp4-template.json").read_text())
    doc["onchip"]["lr_schedule"] = COSINE
    with pytest.raises(ConfigError):
        parse_lifecycle(doc)


def test_existing_onchip_sections_are_unchanged():
    for path in sorted((ROOT / "campaigns/opt-cmo-corruption/lifecycles").glob("opt125m-cmo-l6-exp*.json")):
        doc = json.loads(path.read_text())
        if "onchip" in doc:
            assert parse_lifecycle(doc).to_dict()["onchip"] == doc["onchip"], path.name


@pytest.mark.parametrize("make", [lambda a: IdealizedWriteAdam(a, learning_rate=.003, write_noise_scale=0.),
                                  lambda a: thresholded(a, 0.)])
def test_factor_scales_the_adam_command_and_leaves_moments_unchanged(make):
    constant, scaled = make(faulty()), make(faulty())
    for step in range(4):
        gradient = torch.tensor([1., -2., .5, 3.]) * (step + 1)
        scaled.lr_factor = .25
        expected = .25 * _adam_command(constant, gradient)
        assert torch.allclose(_adam_command(scaled, gradient), expected)
        assert torch.equal(scaled.first, constant.first) and torch.equal(scaled.second, constant.second)


def test_unit_factor_is_bitwise_the_constant_law():
    constant, unit = IdealizedWriteAdam(faulty(), learning_rate=.003), IdealizedWriteAdam(faulty(), learning_rate=.003)
    for step in range(3):
        gradient = torch.tensor([1., -2., .5, 3.]) * (step + 1)
        unit.lr_factor = 1.0
        constant.step(gradient); unit.step(gradient)
    assert torch.equal(constant.array.read(), unit.array.read())


@pytest.mark.parametrize("law", ["cmo_idealized_write_adam", "cmo_thresholded_write_adam"])
def test_scheduled_lifecycle_replays_and_records_the_factor(cmo_inputs, tmp_path, law):
    weights, digest = cmo_inputs
    options = {
        "data.corpus": {"name": "synthetic", "sha256": digest},
        "data.sequence_length": 9, "data.max_examples": 8,
        "defects.cases": [case_doc("random", 50000)], "hwa.arms": [{"id": "digital", "method": "none"}],
        "onchip.batch_size": 4, "onchip.epochs": 1, "onchip.learning_rate": .01,
        "onchip.update_law": law, "onchip.development_every_steps": 1, "onchip.lr_schedule": COSINE,
        "onchip.arms": [{"id": "none", "weights": "hold", "calibration": False},
                        {"id": "onchip_weights", "weights": "learn", "calibration": False}],
    }
    if law == "cmo_thresholded_write_adam":
        options.update({"onchip.threshold_sigma": 1., "onchip.accumulate_updates": True})
    lc = small(**options)
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    run, result = results["onchip__digital__271001"]
    assert result["metrics"]["exact_replays"] == 2
    metrics = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
    # Update 1 is the one warm-up step; update 2 is the end of the cosine decay.
    assert [m["lr_factor"] for m in metrics if m.get("stage") == "onchip_development"] == [1.0, 0.0]
