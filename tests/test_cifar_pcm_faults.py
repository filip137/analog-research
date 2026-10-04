"""Failure identity, endpoint parity and matched one-epoch recovery contracts."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from experiments.cifar_crossbar.fault_config import (
    FaultSpec,
    default_config,
    parse_config,
)
from experiments.cifar_crossbar.model import CifarResNet32, CrossbarSuffix
from experiments.cifar_crossbar.pcm_faults import GaussianEndpointArray, PermanentFaults
from experiments.cifar_crossbar.pcm_reference import PcmInferenceNoise


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_physical_failures_nested_and_persistent():
    maps = {
        kind: PermanentFaults.sample(1000, kind, 0.1, 19)
        for kind in ("open", "gmax", "random")
    }
    assert all(torch.equal(f.mask, maps["open"].mask) for f in maps.values())
    larger = PermanentFaults.sample(1000, "random", 0.5, 19)
    assert not bool((maps["random"].mask & ~larger.mask).any())
    assert torch.equal(maps["random"].conductance, larger.conductance)
    for kind, fault in maps.items():
        target = torch.linspace(-1, 1, 1000)
        array = GaussianEndpointArray(target, fault, 8)
        stuck = array.g[fault.mask].clone()
        before = array.g[~fault.mask].clone()
        for i in range(4):
            array.program(target.flip(0) * (i + 1) / 4)
            array.verify_permanence()
        assert torch.equal(array.g[fault.mask], stuck)
        assert not torch.equal(array.g[~fault.mask], before)
        if kind == "open":
            assert bool((stuck == 0).all())
        if kind == "gmax":
            assert bool((stuck == 25).all())
        if kind == "random":
            assert bool(((stuck > 0) & (stuck < 25)).all())


def test_faults_are_not_logical_weight_replacement():
    fault = PermanentFaults.sample(3, "gmax", 0, 4)
    fault.mask[:] = torch.tensor([[True, False, True], [False, True, True]])
    array = GaussianEndpointArray(
        torch.tensor([0.2, -0.2, 0.3]), fault, 8, noise_scale=0
    )
    torch.testing.assert_close(array.read(), torch.tensor([1.0, -1.0, 0.0]))
    assert fault.receipt()["failed_devices"] == 4
    assert fault.receipt()["affected_weights"] == 3
    assert fault.receipt()["both_banks_failed"] == 1


def test_gaussian_programming_matches_existing_pcm_inference_and_replays():
    student = CrossbarSuffix(CifarResNet32(10).eval(), "head")
    reference = PcmInferenceNoise(student, 33, "cpu")
    programmed, _ = reference.realization(student.q)
    fault = PermanentFaults.sample(student.q.numel(), "open", 0, 19)
    array = GaussianEndpointArray(student.q.detach(), fault, 33)
    assert torch.equal(array.read(), programmed)
    state = array.state_dict()
    array.program(student.q.detach() / 2)
    expected = array.read()
    array.load_state_dict(state)
    array.program(student.q.detach() / 2)
    assert torch.equal(array.read(), expected)
    wrong = deepcopy(state)
    wrong["faults"]["mask"][0, 0] = True
    with pytest.raises(ValueError, match="fault identities"):
        array.load_state_dict(wrong)


def test_fault_spec_is_explicit_and_one_epoch():
    assert parse_config(default_config()).epochs == 1
    for changes in (
        {"epochs": 2},
        {"fault_rates_ppm": [100]},
        {"fault_kinds": ["middle"]},
        {"methods": ["onchip_weights"]},
        {"noise_scale": float("nan")},
    ):
        with pytest.raises(ValueError):
            default_config(**changes)


def test_zero_rate_learning_performs_no_writes_and_preserves_prefix(tmp_path):
    from experiments.cifar_crossbar.fault_runtime import adapt

    teacher = CifarResNet32(10).eval()
    student = CrossbarSuffix(teacher, "head")
    prefix = student.prefix_hash()
    features = torch.randn(4, 64)
    cache = {
        "features": features,
        "teacher_logits": teacher.fc(features).detach(),
        "labels": torch.arange(4),
    }
    fault = PermanentFaults.sample(student.q.numel(), "random", 0.1, 18)
    array = GaussianEndpointArray(student.q.detach(), fault, 33)
    before = array.state_dict()
    spec = FaultSpec(
        **default_config(
            suffix="head", recovery_images=500, learning_rate=0.0, calibration_lr=0.0
        )
    )
    outcome = adapt(
        student,
        array,
        student.q.detach(),
        cache,
        cache,
        spec,
        "onchip_weights",
        SimpleNamespace(append_metric=lambda x: None),
        "case",
        "cpu",
    )
    assert outcome["cost"]["array_reprogram_calls"] == 0
    assert torch.equal(before["rng"], array.generator.get_state())
    assert torch.equal(before["g"], array.g)
    assert student.prefix_hash() == prefix


def test_native_fault_screen_restores_all_controls_and_writes_valid_bundle(
    tmp_path, monkeypatch
):
    from ebl.cli import main
    from experiments.cifar_crossbar import fault_runtime as runtime
    from experiments.cifar_crossbar.runtime import save
    from experiments.study_workflow import prepare_study, summarize_study

    monkeypatch.setenv("EBL_DEFER_CURRENT_SIMULATIONS", "1")
    torch.manual_seed(21)
    teacher = CifarResNet32(10).eval().requires_grad_(False)
    monkeypatch.setattr(runtime, "source_model", lambda *a: (teacher, "fixture"))
    spec = FaultSpec(
        **default_config(
            suffix="head",
            cpu_threads=1,
            recovery_images=500,
            fault_kinds=("gmax",),
            fault_rates_ppm=(0, 10000),
            max_examples=4,
            batch_size=2,
        )
    )
    student = CrossbarSuffix(teacher, "head")
    features = torch.randn(4, 64)
    logits = teacher.fc(features).detach()

    def cache(n):
        return {
            "features": features.repeat(n // 4, 1),
            "teacher_logits": logits.repeat(n // 4, 1),
            "labels": logits.argmax(1).repeat(n // 4),
            "cohort": {"count": n, "indices_sha256": "fixture"},
        }

    data = tmp_path / "cache.pt"
    save(
        data,
        {
            "context": runtime.cache_context(spec, "fixture", student),
            "training": cache(500),
            "test": cache(10000),
        },
    )
    teacher_file = tmp_path / "teacher.pt"
    torch.save({}, teacher_file)
    config = tmp_path / "screen.json"
    from dataclasses import asdict

    config.write_text(json.dumps(asdict(spec)))
    plan = {
        "schema_version": 1,
        "study_id": "pcm-fault-cli-test",
        "title": "Fault screen CLI test",
        "hypothesis": "Paired controls restore the same fault state.",
        "motivation": "Verify native artifacts.",
        "evidence_class": "example",
        "completion_criteria": ["One completed native bundle."],
        "analysis_plan": ["Verify artifacts and paired state."],
        "arms": [
            {
                "arm_id": "screen",
                "configs": ["screen.json"],
                "description": "Screen",
                "experiment_id": spec.experiment_id,
                "mode": "train",
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    study = prepare_study(plan_path, tmp_path / "results")
    output = study / "runs/screen"
    assert (
        main(
            [
                "train",
                "--config",
                str(config),
                "--output-dir",
                str(output),
                "--teacher-weights",
                str(teacher_file),
                "--device-data",
                str(data),
            ]
        )
        == 0
    )
    run = next(output.iterdir())
    result = json.loads((run / "result.json").read_text())["metrics"]
    assert result["completed_controls"] == 10
    assert result["cases"] == 2
    for case in ("nominal", "gmax_10000ppm"):
        paired = [r for r in result["measurements"] if r["case"] == case]
        assert len({r["p0_sha256"] for r in paired}) == 1
        assert len({r["initial_apparent_sha256"] for r in paired}) == 1
        for row in paired:
            expected = (
                2
                if row["method"] in ("rewrite", "onchip_weights", "onchip_calibration")
                else 0
            )
            assert row["cost"]["array_reprogram_calls"] == expected
    assert summarize_study(study, verify_artifacts=True)["ready_for_review"]


def test_campaign_declares_all_cases_and_native_cache(tmp_path):
    from experiments.cifar_crossbar.fault_campaign import prepare

    campaign = prepare(tmp_path, "cifar10")
    assert len(campaign["nodes"]) == 4
    for node in campaign["nodes"]:
        spec = parse_config(json.loads((tmp_path / node["config"]).read_text()))
        assert spec.recovery_images == 5000 and spec.epochs == 1
        if spec.stage == "screen":
            assert spec.fault_rates_ppm == (0, 100, 1000, 10000)
            assert node["inputs"]["device-data"]["artifact"] == "checkpoints/cache.pt"


def test_fault_report_uses_paired_array_differences():
    from experiments.cifar_crossbar.fault_campaign import paired_recovery_summary

    pairs = {}
    for seed, deployed, improvement in (
        (1, 10.0, 1.0),
        (2, 20.0, -1.0),
        (3, 30.0, 3.0),
    ):
        pairs[("cifar100", "gmax", 10000, seed)] = {
            method: {"teacher_kl": deployed - change}
            for method, change in (
                ("none", 0),
                ("calibration", 0.5),
                ("rewrite", 0.25),
                ("onchip_weights", improvement),
                ("onchip_calibration", improvement + 0.1),
            )
        }
    comparisons = paired_recovery_summary(pairs)
    row = next(
        r
        for r in comparisons
        if r["candidate"] == "onchip_weights" and r["control"] == "none"
    )
    assert row["kl_reduction_mean"] == pytest.approx(1.0)
    assert row["kl_reduction_std"] == pytest.approx(2.0)
    assert row["arrays_improved"] == 2
    pairs.pop(("cifar100", "gmax", 10000, 3))
    with pytest.raises(ValueError, match="three paired arrays"):
        paired_recovery_summary(pairs)
