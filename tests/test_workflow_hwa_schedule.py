"""Fixed-update HWA schedules, observed at real Adam updates on a tiny OPT."""

import json
from pathlib import Path

import pytest
import torch

from experiments.schema import ConfigError
from workflow import deployment, hwa
from workflow.lifecycle import parse_lifecycle
from test_workflow_opt import tiny_teacher, tokens
from workflow_helpers import Store


TEMPLATE = Path(__file__).resolve().parents[1] / "examples/lifecycles/opt125m-cmo-mlp4-template.json"


def document(kind="linear_warmup_decay", *, scope="model"):
    doc = json.loads(TEMPLATE.read_text())
    doc["network"]["analog_decoder_layers"] = 2
    doc["deployment"]["tile_size"] = 8
    doc["data"].update(hwa_sequences=5, max_examples=0, sequence_length=9)
    doc["hwa"].update(epochs=2, batch_size=2)
    if scope == "model":
        doc["hwa"]["arms"][1].update(scope="model", digital_learning_rate=2e-4)
    if kind is not None:
        doc["hwa"]["lr_schedule"] = {"kind": kind, "total_steps": 6, "warmup_steps": 2}
        if kind == "warmup_step_decay":
            doc["hwa"]["lr_schedule"].update(milestone_step=4, decay_factor=0.1)
    return doc


def train(doc, tmp_path, *, cached_examples=None):
    lifecycle = parse_lifecycle(doc)
    teacher = tiny_teacher()
    arm = lifecycle.hwa.arm("standard_hwa")
    network = deployment.build_network(teacher, lifecycle, arm.arm_id)
    cache = {
        "hwa": {"tokens": tokens(cached_examples or lifecycle.data.cohort_size("hwa"))},
        "development": {"tokens": tokens(3, seed=9)},
    }
    store = Store(tmp_path)
    result = hwa.train_source(
        lifecycle=lifecycle, arm=arm, network=network, teacher=teacher, cache=cache,
        device_bundle={}, store=store, device="cpu",
    )
    return result, store.records


@pytest.mark.parametrize(
    "kind,factors",
    [
        ("linear_warmup_decay", [0, 0.5, 1, 0.75, 0.5, 0.25, 0]),
        ("warmup_step_decay", [0, 0.5, 1, 1, 1, 0.1, 0.1]),
    ],
)
@pytest.mark.parametrize("scope", ["analog", "model"])
def test_real_optimizer_uses_schedule_for_every_parameter_group(monkeypatch, tmp_path, kind, factors, scope):
    observed, deltas = [], []
    step = torch.optim.Adam.step

    def recorded_step(optimizer, *args, **kwargs):
        observed.append([group["lr"] for group in optimizer.param_groups])
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        before = [p.detach().clone() for p in parameters]
        result = step(optimizer, *args, **kwargs)
        deltas.append(any(not torch.equal(p, previous) for p, previous in zip(parameters, before)))
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", recorded_step)
    doc = document(kind, scope=scope)
    lifecycle = parse_lifecycle(doc)
    schedule = lifecycle.hwa.lr_schedule
    assert [schedule.factor(s) for s in range(7)] == factors
    for bad_step in (-1, 7):
        with pytest.raises(ValueError, match="HWA update"):
            schedule.factor(bad_step)
    result, records = train(doc, tmp_path)
    bases = [1e-4] + ([2e-4] if scope == "model" else [])
    assert len(observed) == 6
    for rates, factor in zip(observed, factors[1:]):
        assert rates == pytest.approx([base * factor for base in bases])
    assert all(deltas[:-1])
    assert deltas[-1] is (kind != "linear_warmup_decay")
    fit = result["fit"]
    assert fit["status"] == "complete"
    assert fit["lr_schedule"]["steps_run"] == 6
    assert fit["lr_schedule"]["base_learning_rates"] == bases
    assert fit["lr_schedule"]["final_learning_rates"] == observed[-1]
    logged = {r["global_step"]: r for r in records if r["stage"] in ("hwa", "hwa_schedule")}
    assert schedule.boundary_steps() <= logged.keys()
    for s, record in logged.items():
        assert record["learning_rates"] == pytest.approx([base * factors[s] for base in bases])
    selection = [r for r in records if r["stage"] == "hwa_selection"]
    assert [r["global_step"] for r in selection] == [0, 3, 6]


@pytest.mark.parametrize(
    "key,value",
    [
        ("kind", "cosine"), ("total_steps", 0), ("total_steps", 7),
        ("total_steps", True), ("warmup_steps", -1), ("warmup_steps", 6),
        ("warmup_steps", 2.5), ("unknown", 1), ("decay_factor", 0.1),
    ],
)
def test_linear_schedule_rejects_invalid_configuration(key, value):
    doc = document()
    doc["hwa"]["lr_schedule"][key] = value
    with pytest.raises(ConfigError, match="hwa.lr_schedule"):
        parse_lifecycle(doc)


@pytest.mark.parametrize(
    "key,value",
    [("milestone_step", 2), ("milestone_step", 6), ("decay_factor", 0),
     ("decay_factor", float("nan")), ("decay_factor", 1.01)],
)
def test_step_schedule_rejects_invalid_configuration(key, value):
    doc = document("warmup_step_decay")
    doc["hwa"]["lr_schedule"][key] = value
    with pytest.raises(ConfigError, match="hwa.lr_schedule"):
        parse_lifecycle(doc)


@pytest.mark.parametrize("setting", ["milestones", "decay", "extension", "null", "missing"])
def test_schedule_rejects_ambiguous_or_incomplete_configuration(setting):
    doc = document("warmup_step_decay")
    if setting == "milestones":
        doc["hwa"]["lr_milestones"] = [1]
    elif setting == "decay":
        doc["hwa"]["lr_decay"] = 0.1
    elif setting == "extension":
        doc["hwa"]["selection"]["extension_epochs"] = 1
    elif setting == "null":
        doc["hwa"]["lr_schedule"] = None
    else:
        del doc["hwa"]["lr_schedule"]["milestone_step"]
    with pytest.raises(ConfigError, match="hwa.lr_schedule"):
        parse_lifecycle(doc)


def test_legacy_serialization_and_digests_are_unchanged():
    doc = json.loads(TEMPLATE.read_text())
    lifecycle = parse_lifecycle(doc)
    assert lifecycle.hwa.lr_schedule is None
    assert "lr_schedule" not in lifecycle.hwa.shared_dict()
    assert lifecycle.to_dict() == doc
    assert lifecycle.digest() == "7b1d87a8ad9aecf176ca3f16855d7591224c74594a29a26f82cbe5e2ca314101"
    assert lifecycle.hwa_arm_digest("standard_hwa") == "3fa10ed44bd457c65b8ba100737b45461ec01c470af9fee4f2b775d09311f553"
    scheduled = parse_lifecycle(document())
    assert parse_lifecycle(scheduled.to_dict()) == scheduled
    assert scheduled.digest() != parse_lifecycle(document(None)).digest()
    assert scheduled.hwa_arm_digest("standard_hwa") != parse_lifecycle(document(None)).hwa_arm_digest("standard_hwa")


def test_legacy_epoch_decay_retains_exact_training_behavior(monkeypatch, tmp_path):
    legacy = document(None)
    legacy["hwa"].update(lr_milestones=[1], lr_decay=0.1)
    scheduled = document("warmup_step_decay")
    scheduled["hwa"]["lr_schedule"].update(warmup_steps=0, milestone_step=3)
    observed = []
    step = torch.optim.Adam.step

    def recorded_step(optimizer, *args, **kwargs):
        observed.append([group["lr"] for group in optimizer.param_groups])
        return step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", recorded_step)
    old_result, old_records = train(legacy, tmp_path / "legacy")
    new_result, _ = train(scheduled, tmp_path / "scheduled")
    expected = [[1e-4, 2e-4]] * 3 + [[1e-5, 2e-5]] * 3
    for rates, target in zip(observed, expected * 2):
        assert rates == pytest.approx(target)
    assert len(observed) == 12
    assert all(torch.equal(value, new_result["model"][key]) for key, value in old_result["model"].items())
    assert [r["teacher_kl"] for r in old_result["fit"]["history"]] == [
        r["teacher_kl"] for r in new_result["fit"]["history"]
    ]
    assert "lr_schedule" not in old_result["fit"]
    assert all("global_step" not in record and "learning_rates" not in record for record in old_records)


def test_smoke_cap_requires_an_explicit_matching_schedule(tmp_path):
    doc = document()
    doc["data"]["max_examples"] = 3
    with pytest.raises(ConfigError, match="diagnostic smoke"):
        parse_lifecycle(doc)
    doc["hwa"]["lr_schedule"].update(total_steps=4, warmup_steps=1)
    result, _ = train(doc, tmp_path)
    assert result["fit"]["lr_schedule"]["steps_run"] == 4


def test_streamed_budget_and_partial_minibatches_are_counted(tmp_path):
    doc = document()
    doc["data"]["hwa_sequences"] = 11
    doc["hwa"]["sequences_per_epoch"] = 5
    result, _ = train(doc, tmp_path)
    assert result["fit"]["lr_schedule"]["steps_run"] == 6
    assert result["fit"]["stream"]["passes"] == pytest.approx(10 / 11)


def test_cache_cardinality_is_verified_even_if_step_count_is_unchanged(tmp_path):
    with pytest.raises(ValueError, match="cache/budget mismatch"):
        train(document(), tmp_path, cached_examples=6)


def test_rejected_run_records_only_completed_updates(monkeypatch, tmp_path):
    objective = hwa.family(parse_lifecycle(document()).network).objective
    calls = 0

    def nonfinite_second_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        return objective(*args, **kwargs) if calls == 1 else torch.tensor(float("nan"))

    monkeypatch.setattr(hwa.family(parse_lifecycle(document()).network), "objective", nonfinite_second_batch)
    result, _ = train(document(), tmp_path)
    assert result["fit"]["status"] == "rejected_nonfinite"
    assert result["fit"]["lr_schedule"]["steps_run"] == 1


@pytest.mark.parametrize("kind", ["linear_warmup_decay", "warmup_step_decay"])
def test_warmup_can_be_explicitly_disabled(kind):
    doc = document(kind)
    doc["hwa"]["lr_schedule"]["warmup_steps"] = 0
    schedule = parse_lifecycle(doc).hwa.lr_schedule
    assert schedule.factor(0) == 1
    assert schedule.factor(1) == pytest.approx(5 / 6 if kind == "linear_warmup_decay" else 1)
