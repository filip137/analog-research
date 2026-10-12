"""Physical no-write semantics and signed command conservation for CMO gating."""
import json

import pytest
import torch

from experiments.schema import ConfigError
from training.reram_cmo import CmoMatrix
from workflow.lifecycle import parse_lifecycle
from workflow.onchip import IdealizedWriteAdam, ThresholdedWriteAdam
from test_workflow_cmo import cmo_inputs, small, template, case_doc
from workflow_helpers import run_lifecycle


def writer(array, *, accumulate=True, lr=.003, sigma=1):
    return ThresholdedWriteAdam(array, learning_rate=lr, threshold_sigma=sigma,
                               accumulate_updates=accumulate, acceptance_percent=2,
                               noise_scale=1.)


def matrix():
    x = CmoMatrix(3, 7, 2., 1., "cpu")
    x.program(torch.zeros(3))
    return x


def test_discard_suppresses_writes_and_noise_without_changing_adam_moments():
    array = matrix(); w = writer(array, accumulate=False)
    before = array.g.clone(); cost = array.cost()
    reference = IdealizedWriteAdam(matrix(), learning_rate=.003)
    for _ in range(30):
        gradient = torch.tensor([1., -2., 0.])
        w.step(gradient); reference.step(gradient)
    assert torch.equal(array.g, before) and array.cost() == cost
    assert torch.equal(w.first, reference.first) and torch.equal(w.second, reference.second)
    assert w.pending is None


def test_accumulation_delays_then_issues_the_sum_without_rounding_up():
    array = matrix(); w = writer(array)
    initial = array.g.clone(); count = array.cost()["device_writes"]
    gradient = torch.tensor([1., -1., 0.])
    for _ in range(4):
        w.step(gradient)
    assert torch.equal(array.g, initial)
    assert torch.allclose(w.pending, torch.tensor([-.012, .012, 0.]))
    w.step(gradient)
    assert array.cost()["device_writes"] - count == 2
    assert torch.count_nonzero(w.pending) == 0
    assert torch.equal(array.g[1], initial[1])
    assert array.g[0, 2] == initial[0, 2]
    assert not torch.equal(array.g[0, :2], initial[0, :2])


def test_accumulator_conserves_signed_commands_even_for_stuck_write_attempts(monkeypatch):
    import workflow.onchip as module
    mask = torch.tensor([[True, False, False], [False, False, False]])
    stuck = torch.full((2, 3), 48.6)
    array = CmoMatrix(3, 7, 2., 1., "cpu", (mask, stuck))
    array.program(torch.zeros(3)); before = array.g.clone(); cost = array.cost()
    w = writer(array)
    commands = iter([torch.tensor([.01, .01, .01]), torch.tensor([.01, -.01, .001])])
    monkeypatch.setattr(module, "_adam_command", lambda *_: next(commands))
    w.step(None); w.step(None)
    assert torch.allclose(w.pending, torch.tensor([0., 0., .011]))
    assert torch.equal(array.g, before)
    assert array.cost()["stuck_write_attempts"] - cost["stuck_write_attempts"] == 1
    assert array.cost()["device_writes"] == cost["device_writes"]


def test_zero_lr_does_not_flush_pending_updates_or_mutate_array():
    array = matrix(); w = writer(array, lr=0)
    w.pending.fill_(.1); before = array.g.clone()
    w.step(torch.ones(3))
    assert torch.equal(array.g, before) and torch.all(w.pending == .1)
    saved = w.state_dict(); w.pending.zero_()
    assert torch.all(saved["pending"] == .1)


@pytest.mark.parametrize("field,value", [("threshold_sigma", 0), ("threshold_sigma", float("nan")),
                                        ("accumulate_updates", "yes"), ("development_every_steps", 0)])
def test_threshold_schema_rejects_invalid_fields(field, value):
    doc = template()
    doc["onchip"].update(update_law="cmo_thresholded_write_adam", threshold_sigma=2., accumulate_updates=True)
    doc["onchip"][field] = value
    with pytest.raises(ConfigError):
        parse_lifecycle(doc)


def test_threshold_fields_do_not_silently_change_legacy_law():
    doc = template(); doc["onchip"]["threshold_sigma"] = 2.
    with pytest.raises(ConfigError):
        parse_lifecycle(doc)


@pytest.mark.parametrize("accumulate", [False, True])
def test_threshold_lifecycle_replays_and_records_fixed_development(cmo_inputs, tmp_path, accumulate):
    weights, digest = cmo_inputs
    lc = small(**{
        "data.corpus": {"name": "synthetic", "sha256": digest},
        "data.sequence_length": 9, "data.max_examples": 8,
        "defects.cases": [case_doc("random", 50000)], "hwa.arms": [{"id":"digital", "method":"none"}],
        "onchip.batch_size": 4, "onchip.epochs": 1, "onchip.learning_rate": .01,
        "onchip.update_law": "cmo_thresholded_write_adam", "onchip.threshold_sigma": 1.,
        "onchip.accumulate_updates": accumulate, "onchip.development_every_steps": 1,
        "onchip.arms": [{"id":"none", "weights":"hold", "calibration":False},
                        {"id":"onchip_weights", "weights":"learn", "calibration":False}],
    })
    assert parse_lifecycle(lc.to_dict()) == lc
    results = run_lifecycle(lc, weights, tmp_path/"runs")
    run, result = results["onchip__digital__271001"]
    assert result["metrics"]["exact_replays"] == 2
    metrics = [json.loads(s) for s in (run/"metrics.jsonl").read_text().splitlines()]
    checks = [m for m in metrics if m.get("stage") == "onchip_development"]
    assert [m["global_step"] for m in checks] == [1, 2]
    assert all(m["writer"]["accumulate_updates"] == accumulate for m in checks)
    assert all(m["development"]["scored_tokens"] == 64 for m in checks)
    if not accumulate:
        assert all(m["cost"]["device_writes"] == 0 for m in checks)
