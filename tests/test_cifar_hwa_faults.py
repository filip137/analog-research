"""Population resampling, selection isolation, and native HWA-to-recovery chain."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from experiments.cifar_crossbar.hwa_fault_config import (
    ComparisonSpec,
    default_config,
    parse_config,
)
from experiments.cifar_crossbar.hwa_fault_runtime import augment_images, sampled_weights
from experiments.cifar_crossbar.model import CifarResNet32, CrossbarSuffix
from experiments.cifar_crossbar.pcm_faults import GaussianEndpointArray, PermanentFaults


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_population_faults_resample_preserve_master_and_gradient():
    q = torch.linspace(-1, 1, 10000).requires_grad_()
    before = q.detach().clone()
    rng = torch.Generator().manual_seed(73)
    first, m1 = sampled_weights(q, rng, 1.0, "random", 0.1)
    second, m2 = sampled_weights(q, rng, 1.0, "random", 0.1)
    assert not torch.equal(m1, m2) and not torch.equal(first, second)
    assert torch.equal(q, before)
    first.sum().backward()
    torch.testing.assert_close(q.grad, torch.ones_like(q))
    replay, r1 = sampled_weights(
        q, torch.Generator().manual_seed(73), 1.0, "random", 0.1
    )
    assert torch.equal(first, replay) and torch.equal(m1, r1)


def test_hwa_healthy_distribution_matches_deployment():
    q = torch.linspace(-1, 1, 1000)
    faults = PermanentFaults.sample(len(q), "open", 0, 123)
    array = GaussianEndpointArray(q, faults, 10)
    sampled, mask = sampled_weights(q, torch.Generator().manual_seed(10))
    torch.testing.assert_close(sampled, array.read(), atol=1e-7, rtol=1e-6)
    assert not mask.any()
    for kind in ("open", "gmax"):
        failed, _ = sampled_weights(q, torch.Generator().manual_seed(10), 0, kind, 1)
        assert torch.equal(failed, torch.zeros_like(failed))


def test_gpu_style_augmentation_is_replayable_and_normalizes_after_padding():
    raw = torch.full((8, 3, 32, 32), 255, dtype=torch.uint8)
    first = augment_images(raw, "cifar10", torch.Generator().manual_seed(1))
    second = augment_images(raw, "cifar10", torch.Generator().manual_seed(1))
    assert first.shape == raw.shape and torch.equal(first, second)
    from experiments.cifar_crossbar.data import NORMALIZATION

    mean, sd = NORMALIZATION["cifar10"]
    for c in range(3):
        assert float(first[:, c].min()) == pytest.approx(-mean[c] / sd[c])
        assert float(first[:, c].max()) == pytest.approx((1 - mean[c]) / sd[c])


def test_hwa_spec_requires_separate_final_array_namespace():
    assert parse_config(default_config()).hwa_epochs == 30
    for changes in (
        {"array_seed": 51001},
        {"endpoint_seed": 72001},
        {"cdt_rates": (1.1,)},
        {"hwa_noise_scales": (2.0,)},
        {"validation_images": 25},
        {"hwa_objectives": ("teacher_kl", "teacher_kl")},
    ):
        with pytest.raises(ValueError):
            default_config(**changes)


@pytest.mark.parametrize("reject_sgd", [False, True])
def test_full_native_fit_tune_screen_uses_only_development_for_selection(
    tmp_path, monkeypatch, reject_sgd
):
    from ebl.cli import main
    from experiments.cifar_crossbar import hwa_fault_runtime as runtime
    from experiments.cifar_crossbar.runtime import load
    from experiments.study_workflow import prepare_study, summarize_study

    monkeypatch.setenv("EBL_DEFER_CURRENT_SIMULATIONS", "1")
    receipt_path = tmp_path / "source-receipt.json"
    receipt_path.write_text(json.dumps({"files": {}}))
    monkeypatch.setenv("EBL_SOURCE_RECEIPT", str(receipt_path))
    if reject_sgd:

        def unstable_step(optimizer, *args, **kwargs):
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("nan"))

        monkeypatch.setattr(torch.optim.SGD, "step", unstable_step)
    torch.manual_seed(11)
    teacher = CifarResNet32(10).eval().requires_grad_(False)
    monkeypatch.setattr(runtime, "source_model", lambda *args: (teacher, "fixture"))
    spec = ComparisonSpec(
        **default_config(
            suffix="head",
            cpu_threads=1,
            max_examples=4,
            hwa_epochs=1,
            extension_epochs=1,
            validation_every=1,
            hwa_batch_size=4,
            batch_size=4,
            hwa_learning_rates=(0.0001,),
            hwa_sgd_learning_rates=(0.05,),
            hwa_objectives=("teacher_kl",),
            hwa_noise_scales=(1.0,),
            cdt_rates=(0.01,),
            development_arrays=1,
            tuning_arrays=1,
            recovery_learning_rates=(0.0001,),
            calibration_learning_rates=(0.0003,),
        )
    )
    student = CrossbarSuffix(teacher, "head")
    prefix = student.prefix_hash()
    features = torch.randn(1, 64)
    logits = teacher.fc(features).detach()

    def cohort(n):
        return {
            "features": features.expand(n, -1),
            "teacher_logits": logits.expand(n, -1),
            "labels": torch.zeros(n, dtype=torch.long),
            "cohort": {"count": n, "indices_sha256": "fixture"},
        }

    cache = {
        "context": runtime.cache_context(spec, "fixture", student),
        "hwa_raw": torch.zeros(1, 3, 32, 32, dtype=torch.uint8).expand(
            45000, -1, -1, -1
        ),
        "hwa_labels": torch.zeros(45000, dtype=torch.long),
        "training": cohort(5000),
        "development": cohort(1000),
        "test": cohort(10000),
    }
    cache_path = tmp_path / "cache.pt"
    torch.save(cache, cache_path)
    teacher_path = tmp_path / "teacher.pt"
    torch.save({}, teacher_path)
    arms = []
    for stage in ("fit", "tune", "screen"):
        path = tmp_path / f"{stage}.json"
        path.write_text(json.dumps({**asdict(spec), "stage": stage}))
        arms.append(
            {
                "arm_id": stage,
                "configs": [path.name],
                "description": stage,
                "experiment_id": spec.experiment_id,
                "mode": "train",
            }
        )
    plan = {
        "schema_version": 1,
        "study_id": "hwa-chain-test",
        "title": "HWA chain",
        "hypothesis": "Selection precedes final arrays.",
        "motivation": "Native lifecycle regression.",
        "evidence_class": "example",
        "completion_criteria": ["Three completed native stages."],
        "analysis_plan": ["Check selection isolation and paired state."],
        "arms": arms,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    study = prepare_study(plan_path, tmp_path / "results")
    captured = []
    original_evaluate = runtime.evaluate

    def audited_evaluate(model, c, device, **kwargs):
        captured.append(c)
        return original_evaluate(model, c, device, **kwargs)

    monkeypatch.setattr(runtime, "evaluate", audited_evaluate)
    from experiments.cifar_crossbar import fault_runtime

    monkeypatch.setattr(fault_runtime, "evaluate", audited_evaluate)
    artifacts = {}
    for stage in ("fit", "tune", "screen"):
        captured.clear()
        args = [
            "train",
            "--config",
            str(tmp_path / f"{stage}.json"),
            "--output-dir",
            str(study / "runs" / stage),
            "--teacher-weights",
            str(teacher_path),
            "--device-data",
            str(cache_path),
        ]
        if stage != "fit":
            args += ["--weights", str(artifacts["fit" if stage == "tune" else "tune"])]
        assert main(args) == 0
        if stage != "screen":
            assert captured and all(c["cohort"]["count"] == 1000 for c in captured)
        run = next((study / "runs" / stage).iterdir())
        result = json.loads((run / "result.json").read_text())["metrics"]
        if stage == "fit":
            artifacts[stage] = run / "checkpoints/sources.pt"
            bundle = load(artifacts[stage])
            assert len(bundle["sources"]) == 6
            for source in bundle["sources"].values():
                student.load_state_dict(source["model"])
                assert student.prefix_hash() == prefix
            assert result["training_images"] == 4
            assert result["rejected_candidates"] == int(reject_sgd)
            assert all(
                x is None or x["status"] == "complete"
                for x in result["sources"].values()
            )
        elif stage == "tune":
            artifacts[stage] = run / "checkpoints/recovery.pt"
            assert load(artifacts[stage])["source_sha256"] == runtime.sha256_file(
                artifacts["fit"]
            )
        else:
            assert result["completed_controls"] == 210 and result["cases"] == 42
            for row in result["measurements"]:
                assert row["image_presentations"] in (0, 4)
            by_case = {}
            for row in result["measurements"]:
                by_case.setdefault((row["source"], row["case"]), []).append(row)
            for rows in by_case.values():
                assert len({r["p0_sha256"] for r in rows}) == 1
                assert len({r["initial_apparent_sha256"] for r in rows}) == 1
    assert summarize_study(study, verify_artifacts=True)["ready_for_review"]


def test_nonfinite_candidate_cannot_win_using_its_earlier_finite_checkpoint():
    from experiments.cifar_crossbar.hwa_fault_runtime import select_completed

    def candidate(status, score):
        return {
            "fit": {
                "status": status,
                "all_cases": {"cases": [{"case": "nominal", "teacher_kl": score}]},
            }
        }

    rejected = candidate("rejected_nonfinite", 0)
    completed = candidate("complete", 1)
    assert select_completed([rejected, completed], [("open", 0)]) is completed
    with pytest.raises(RuntimeError, match="All candidates"):
        select_completed([rejected], [("open", 0)])


def test_campaign_has_full_hwa_and_three_disjoint_final_arrays(tmp_path):
    from experiments.cifar_crossbar.hwa_fault_campaign import prepare

    campaign = prepare(tmp_path, "cifar100")
    assert len(campaign["nodes"]) == 6
    screens = []
    for node in campaign["nodes"]:
        c = parse_config(json.loads((tmp_path / node["config"]).read_text()))
        if c.stage == "screen":
            screens.append(c.array_seed)
            assert set(node["inputs"]) == {"device-data", "weights"}
    assert screens == [81001, 81002, 81003]


def test_cross_source_contrasts_pair_array_identity_and_preserve_losses():
    from experiments.cifar_crossbar.hwa_fault_campaign import cross_source_comparisons

    rows = []
    for seed, control, recovered in (
        (81001, 1.0, 0.8),
        (81002, 2.0, 2.3),
        (81003, 3.0, 2.9),
    ):
        for source, method, kl, accuracy in (
            ("noise_hwa", "calibration", control, 70.0),
            ("cdt_gmax", "onchip_weights", recovered, 71.0),
        ):
            rows.append(
                dict(
                    dataset="cifar100",
                    fault_kind="gmax",
                    fault_rate_ppm=10000,
                    source=source,
                    method=method,
                    array_seed=seed,
                    teacher_kl=kl,
                    accuracy_percent=accuracy,
                    fault_identity_sha256=str(seed),
                )
            )
    result = cross_source_comparisons(rows[::2] + rows[1::2][::-1])
    assert len(result) == 1
    assert result[0]["arrays_kl_improved"] == 2
    assert result[0]["mean_kl_reduction"] == pytest.approx(0, abs=1e-12)
    assert result[0]["std_kl_reduction"] == pytest.approx(0.2645751311064591)
    assert result[0]["mean_accuracy_gain_pp"] == 1
    rows[1]["fault_identity_sha256"] = "wrong-array"
    with pytest.raises(ValueError, match="unmatched physical faults"):
        cross_source_comparisons(rows)
