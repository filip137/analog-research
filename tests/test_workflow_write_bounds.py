"""Bounded CMO write targets: G + dG clipped to [G_min, G_max] before the write noise draw."""
import json
from pathlib import Path

import pytest
import torch

from experiments.schema import ConfigError
from training.reram_cmo import ACTIVE, G_MAX_US, G_MIN_US, HALF_RANGE_US, REFERENCE, CmoMatrix
from workflow.lifecycle import parse_lifecycle
from workflow.onchip import IdealizedWriteAdam, ThresholdedWriteAdam
from test_workflow_cmo import cmo_inputs, small, template, case_doc
from workflow_helpers import run_lifecycle

ROOT = Path(__file__).resolve().parents[1]
# Index 0 is stuck active, 1 is pushed above G_max, 2 below G_min, 3 stays in range, 4 is not written.
DELTA = torch.tensor([.2, .2, -2.2, -.02, 0.])


def edge(seed=7):
    """Weights programmed to q = 1, so active devices start near G_max; index 0 has a stuck active device."""

    mask = torch.zeros((2, 5), dtype=torch.bool); mask[ACTIVE, 0] = True
    x = CmoMatrix(5, seed, 2., 1., "cpu", (mask, torch.full((2, 5), 30.)))
    x.program(torch.ones(5))
    return x


def test_default_write_is_bitwise_the_unbounded_law():
    physical, declared = edge(), edge()
    physical.write(DELTA); declared.write(DELTA, clip_targets=False)
    assert torch.equal(physical.g, declared.g)
    assert torch.equal(physical.generator.get_state(), declared.generator.get_state())
    assert physical.cost() == declared.cost() and "clipped_write_targets" not in physical.cost()
    assert "clipped_write_targets" not in physical.state_dict()
    # Without bounds the requested targets leave the physical range.
    assert physical.g[ACTIVE, 1] > G_MAX_US and physical.g[ACTIVE, 2] < G_MIN_US


def test_exact_bounded_writes_land_on_the_clipped_target():
    bounded = edge(); start = bounded.g.clone(); stuck_before = bounded.stuck_write_attempts
    bounded.write(DELTA, 0.0, clip_targets=True)
    assert bounded.g[ACTIVE, 1] == G_MAX_US and bounded.g[ACTIVE, 2] == G_MIN_US
    assert bounded.g[ACTIVE, 3] == start[ACTIVE, 3] + DELTA[3] * HALF_RANGE_US
    assert bounded.g[ACTIVE, 0] == 30. and bounded.g[ACTIVE, 4] == start[ACTIVE, 4]
    assert torch.equal(bounded.g[REFERENCE], start[REFERENCE])
    # Only healthy written devices whose target moved count; the stuck attempt does not.
    assert bounded.cost()["clipped_write_targets"] == 2
    assert bounded.cost()["stuck_write_attempts"] - stuck_before == 1


def test_noisy_bounded_write_draws_around_the_clipped_target():
    bounded, reference = edge(), edge()
    target = (reference.g[ACTIVE] + DELTA * HALF_RANGE_US).clamp(G_MIN_US, G_MAX_US)
    bounded.write(DELTA, clip_targets=True)
    # The same noise stream around the clipped target reproduces the bounded endpoint.
    reference.write((target - reference.g[ACTIVE]) / HALF_RANGE_US * (DELTA != 0))
    assert torch.allclose(bounded.g, reference.g, atol=1e-5)
    assert torch.equal(bounded.generator.get_state(), reference.generator.get_state())
    # The noisy endpoint itself is held unclipped.
    assert (bounded.g[ACTIVE, 1] - G_MAX_US).abs() > 0


def test_role_mapped_writes_clip_the_reference_target_too():
    bounded = edge()
    roles = torch.tensor([1, 1, 0, 0, -1], dtype=torch.int8)
    # A reference write moves the reference by -dG; at q = 1 the reference sits at mid-range.
    delta = torch.tensor([1.5, 1.5, -2.2, 0., .3])
    bounded.write(delta, 0.0, roles=roles, clip_targets=True)
    assert bounded.g[REFERENCE, 0] == G_MIN_US and bounded.g[REFERENCE, 1] == G_MIN_US
    assert bounded.g[ACTIVE, 2] == G_MIN_US
    assert bounded.cost()["clipped_write_targets"] == 3


def test_bounded_state_replays_with_its_counter():
    original = edge(); original.write(DELTA, clip_targets=True)
    state = original.state_dict()
    restored = edge(); restored.load_state_dict(state)
    assert torch.equal(restored.g, original.g) and restored.cost() == original.cost()
    unbounded = edge(); restored.load_state_dict(unbounded.state_dict())
    assert "clipped_write_targets" not in restored.cost()


@pytest.mark.parametrize("make", [
    lambda a, clip: IdealizedWriteAdam(a, learning_rate=.5, write_noise_scale=0., clip_write_targets=clip),
    lambda a, clip: ThresholdedWriteAdam(a, learning_rate=.5, threshold_sigma=1, accumulate_updates=True,
                                         acceptance_percent=2, noise_scale=1., write_noise_scale=0.,
                                         clip_write_targets=clip),
])
def test_writers_keep_bounded_conductances_in_range(make):
    bounded, unbounded = edge(), edge()
    wb, wu = make(bounded, True), make(unbounded, False)
    for _ in range(5):
        gradient = torch.tensor([-1., -1., 1., 1., -1.])
        wb.step(gradient); wu.step(gradient)
    healthy = ~bounded.mask
    assert bool((bounded.g[healthy] <= G_MAX_US).all() and (bounded.g[healthy] >= G_MIN_US).all())
    assert bool((unbounded.g[healthy] > G_MAX_US).any())
    assert wb.state_dict()["clip_write_targets"] is True and "clip_write_targets" not in wu.state_dict()
    assert bounded.cost()["clipped_write_targets"] > 0


@pytest.mark.parametrize("law", ["cmo_idealized_write_adam", "cmo_thresholded_write_adam"])
def test_schema_records_bounds_only_when_declared(law):
    doc = template()
    doc["onchip"].pop("clip_write_targets", None)
    doc["onchip"]["update_law"] = law
    if law == "cmo_thresholded_write_adam":
        doc["onchip"].update(threshold_sigma=2., accumulate_updates=True)
        doc["onchip"]["arms"] = [arm for arm in doc["onchip"]["arms"] if arm["weights"] != "rewrite"]
    original = parse_lifecycle(doc)
    assert original.onchip.clip_write_targets is None and "clip_write_targets" not in original.to_dict()["onchip"]
    doc["onchip"]["clip_write_targets"] = True
    bounded = parse_lifecycle(doc)
    assert bounded.onchip.clip_write_targets is True and bounded.to_dict()["onchip"]["clip_write_targets"] is True
    assert parse_lifecycle(bounded.to_dict()) == bounded and bounded.digest() != original.digest()
    del doc["onchip"]["clip_write_targets"]
    assert parse_lifecycle(doc).digest() == original.digest()


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_schema_rejects_non_boolean_bounds(value):
    doc = template(); doc["onchip"]["clip_write_targets"] = value
    with pytest.raises(ConfigError):
        parse_lifecycle(doc)


def test_schema_rejects_bounds_outside_cmo_laws():
    doc = json.loads((ROOT / "examples/lifecycles/opt125m-om-mlp4-template.json").read_text())
    doc["onchip"]["clip_write_targets"] = True
    with pytest.raises(ConfigError):
        parse_lifecycle(doc)


def test_cmo_template_declares_bounded_write_targets():
    doc = json.loads((ROOT / "examples/lifecycles/opt125m-cmo-mlp4-template.json").read_text())
    assert doc["onchip"]["clip_write_targets"] is True
    assert parse_lifecycle(doc).onchip.clip_write_targets is True


def test_existing_campaign_lifecycles_stay_unbounded():
    for path in sorted((ROOT / "campaigns/opt-cmo-corruption/lifecycles").glob("*.json")):
        doc = json.loads(path.read_text())
        assert "clip_write_targets" not in (doc.get("onchip") or {}), path.name


def test_bounded_lifecycle_replays_and_reports_clipped_targets(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    lc = small(**{
        "data.corpus": {"name": "synthetic", "sha256": digest},
        "data.sequence_length": 9, "data.max_examples": 8,
        "defects.cases": [case_doc("random", 50000)], "hwa.arms": [{"id": "digital", "method": "none"}],
        "onchip.batch_size": 4, "onchip.epochs": 1, "onchip.learning_rate": .5,
        "onchip.update_law": "cmo_idealized_write_adam", "onchip.clip_write_targets": True,
        "onchip.development_every_steps": 1,
        "onchip.arms": [{"id": "none", "weights": "hold", "calibration": False},
                        {"id": "onchip_weights", "weights": "learn", "calibration": False}],
    })
    assert parse_lifecycle(lc.to_dict()) == lc
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    run, result = results["onchip__digital__271001"]
    assert result["metrics"]["exact_replays"] == 2
    metrics = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
    checks = [m for m in metrics if m.get("stage") == "onchip_development"]
    assert checks and all(m["cost"]["clipped_write_targets"] >= 0 for m in checks)
    assert checks[-1]["cost"]["clipped_write_targets"] > 0
