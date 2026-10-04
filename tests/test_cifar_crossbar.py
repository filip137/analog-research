"""Numerical and experimental-contract checks for CIFAR suffix recovery."""

from copy import deepcopy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from experiments.cifar_crossbar.config import default_config, parse_config
from experiments.cifar_crossbar.data import split_indices
from experiments.cifar_crossbar.devices import (
    EndpointKernel,
    PulseWriter,
    make_plant,
    prepare_population,
)
from experiments.cifar_crossbar.model import CifarResNet32, CrossbarSuffix
from experiments.cifar_crossbar.pcm_reference import PcmInferenceNoise


def raw_population(kind="om", size=12):
    shape = (1 if kind == "om" else 2, size)
    ones = torch.ones(shape)
    h = {
        "min_bound": ones * (-1 if kind == "om" else 0),
        "max_bound": ones * (1 if kind == "om" else 2),
        "dwmin_up": ones * 0.01,
        "dwmin_down": ones * 0.01,
        "reference": ones * 0,
        "corrupt": ones.bool() & False,
        "published_corrupt": ones.bool() & False,
        "reset_bias": ones * 0.01,
    }
    c = {
        "dw_min": 0.01,
        "dw_min_std": 0.2,
        "write_noise_std": 0.3,
        "reset": 0.01,
        "reset_std": 0.01,
        "reset_dtod": 0.02,
        "w_max": 2 if kind == "pcm" else 1,
        "w_min": 0 if kind == "pcm" else -1,
        "A_up": -27.235,
        "gamma_up": 2.5,
        "a": -1.0,
        "b": 0.0,
        "dw_min_std_add": 0.042,
        "dw_min_std_slope": 0.108,
    }
    return {
        "kind": kind,
        "size": size,
        "seed": 123,
        "policy": "repaired",
        "variation": 1.0,
        "constants": c,
        "hidden": h,
    }


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize(
    "suffix,tiles", [("head", 1), ("last_block", 5), ("last_two_blocks", 9)]
)
def test_mapping_logits_and_gradients(suffix, tiles):
    torch.manual_seed(31)
    teacher = CifarResNet32(10).eval()
    student = CrossbarSuffix(teacher, suffix)
    x = torch.randn(3, 3, 32, 32)
    prefix = student.prefix_hash()
    torch.testing.assert_close(student(x), teacher(x), atol=2e-6, rtol=2e-5)
    assert len(student.mapping_receipt()["logical_tiles"]) == tiles
    student(x).square().sum().backward()
    teacher(x).square().sum().backward()
    for spec in student.layout:
        expected = teacher.get_submodule(spec.name).weight.grad.flatten() * spec.scale
        actual = student.q.grad[spec.offset : spec.offset + spec.size]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
    assert student.prefix_hash() == prefix
    assert all(p.grad is None for p in student.model.parameters())


def test_prefix_cache_preserves_function_and_bn():
    teacher = CifarResNet32(100).eval()
    student = CrossbarSuffix(teacher, "last_two_blocks")
    x = torch.randn(4, 3, 32, 32)
    saved = student.prefix_hash()
    student.enable_calibration(True)
    student.train()
    torch.testing.assert_close(
        student(x), student.forward_features(student.features(x))
    )
    student(x).sum().backward()
    assert student.prefix_hash() == saved
    assert not student.model.layer3[3].bn1.training


@pytest.mark.parametrize("classes", [10, 100])
def test_nested_balanced_cohorts_exclude_validation(classes):
    labels = np.arange(50000) % classes
    train, validation = split_indices(labels)
    assert len(train) == 45000 and len(validation) == 5000
    assert not set(train) & set(validation)
    for count in (500, 5000):
        assert np.all(np.bincount(labels[train[:count]]) == count // classes)
    assert set(train[:500]) <= set(train[:5000])


@pytest.mark.parametrize("kind", ["om", "pcm"])
@pytest.mark.parametrize("method", ["open_loop", "closed_loop"])
def test_exact_pulse_resume_and_zero_update(kind, method):
    pop = prepare_population(raw_population(kind))
    plant = make_plant(pop, 42, "cpu")
    writer = PulseWriter(
        plant.port(),
        method=method,
        learning_rate=0.01,
        tolerance=0.005,
        cap=30,
        maximum_pulses=4,
        seed=44,
    )
    grad = torch.linspace(-1, 1, plant.size)
    for _ in range(3):
        writer.step(grad)
    state, writer_state = plant.state_dict(), writer.state_dict()
    other = make_plant(pop, 999, "cpu")
    other.load_state_dict(state)
    again = PulseWriter(
        other.port(),
        method=method,
        learning_rate=0.01,
        tolerance=0.005,
        cap=30,
        maximum_pulses=4,
        seed=11,
    )
    again.load_state_dict(writer_state)
    for _ in range(3):
        writer.step(grad)
        again.step(grad)
    assert torch.equal(plant.read(), other.read())
    assert plant.cost() == other.cost()
    hold = PulseWriter(
        plant.port(), method=method, learning_rate=0.0, tolerance=0.005, seed=4
    )
    value = plant.read().clone()
    costs = plant.cost()
    hold.step(grad)
    assert torch.equal(value, plant.read()) and costs == plant.cost()
    assert not hasattr(plant.port(), "population")


def test_faults_are_physical_and_hidden_from_writer():
    pop = prepare_population(
        raw_population("om"), fault_rate=0.5, fault_kind="high", fault_seed=4
    )
    plant = make_plant(pop, 2, "cpu")
    mask = pop["hidden"]["injected_faults"].flatten()
    before = plant.engine.persistent.clone()
    plant.pulse(-torch.ones(plant.size, dtype=torch.int8))
    assert torch.equal(plant.engine.persistent[mask], before[mask])
    assert int(mask.sum()) == 6
    pop["hidden"]["dwmin_up"][0, 0] += 0.1
    with pytest.raises(ValueError, match="fingerprint"):
        make_plant(pop, 2, "cpu")


def test_pcm_set_equation_and_refresh_cost():
    raw = raw_population("pcm")
    raw["constants"]["dw_min_std"] = 0.0
    raw["constants"]["reset_std"] = 0.0
    plant = make_plant(prepare_population(raw), 5, "cpu")
    plant.g.fill_(0.5)
    plant.pulse(torch.ones(plant.size, dtype=torch.int8))
    expected = 0.5 + 0.01 * (1 + 27.235 * np.exp(-1.25))
    torch.testing.assert_close(plant.g[0], torch.full((plant.size,), expected))
    torch.testing.assert_close(plant.g[1], torch.full((plant.size,), 0.5))
    plant.g[0].fill_(1.8)
    plant.g[1].fill_(0.8)
    plant.finish_step()
    assert plant.refreshes == plant.size
    assert int(plant.resets.sum()) == 4 * plant.size
    assert plant.cost()["physical_pulses"] > 4 * plant.size


def test_pcm_refresh_cannot_exceed_physical_write_cap():
    plant = make_plant(prepare_population(raw_population("pcm")), 5, "cpu")
    plant.g[0].fill_(1.8)
    plant.g[1].fill_(0.8)
    baseline = (plant.pulses + plant.resets).clone()
    plant.begin_recovery(3)
    plant.finish_step()
    for _ in range(20):
        plant.pulse(torch.ones(plant.size, dtype=torch.int8))
    assert int((plant.pulses + plant.resets - baseline).max()) <= 3


def test_programming_counts_all_observed_cells():
    from experiments.cifar_crossbar.devices import program

    plant = make_plant(prepare_population(raw_population()), 3, "cpu")
    result = program(
        plant.port(), torch.ones(plant.size) * 100, tolerance=0.001, maximum_pulses=2
    )
    assert result["verify_reads"] == 3 * plant.size


def test_endpoint_kernel_ste_and_rng_replay():
    table = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 100
    kernel = EndpointKernel(table, 11, "cpu")
    q = torch.linspace(-1, 1, 100, requires_grad=True)
    state = kernel.generator.get_state()
    first = kernel.perturb(q)
    kernel.generator.set_state(state)
    assert torch.equal(first, kernel.perturb(q))
    first.sum().backward()
    assert torch.equal(q.grad, torch.ones_like(q))


def test_inference_noise_is_a_separate_reproducible_track():
    student = CrossbarSuffix(CifarResNet32(10).eval(), "head")
    model = PcmInferenceNoise(student, 21, "cpu", time_seconds=86400)
    state = model.generator.get_state()
    a, b = model.realization(student.q)
    model.generator.set_state(state)
    c, d = model.realization(student.q)
    assert torch.equal(a, c) and torch.equal(b, d)
    assert not hasattr(model, "pulse")


def test_teacher_kl_direction_and_per_image_average():
    from experiments.cifar_crossbar.runtime import evaluate

    class LogitStudent:
        read_generator = torch.Generator().manual_seed(19)

        def eval(self):
            return self

        def forward_features(self, x, q=None):
            return x

    teacher = torch.tensor([[0.8, 0.2], [0.25, 0.75], [0.5, 0.5]], dtype=torch.float64)
    student = torch.tensor([[0.6, 0.4], [0.4, 0.6], [0.75, 0.25]], dtype=torch.float64)
    cache = {
        "features": student.log(),
        "teacher_logits": teacher.log(),
        "labels": torch.tensor([0, 1, 0]),
    }
    expected = float((teacher * (teacher / student).log()).sum(1).mean())
    reverse = float((student * (student / teacher).log()).sum(1).mean())
    assert abs(expected - reverse) > 0.001
    for batch_size in (1, 2, 3):
        measured = evaluate(LogitStudent(), cache, "cpu", batch_size=batch_size)
        assert measured["teacher_kl"] == pytest.approx(expected)


def test_report_kl_uses_paired_array_means_and_preserves_accuracy(
    tmp_path, monkeypatch
):
    from experiments.cifar_crossbar import analysis
    import csv

    rows = []
    # Two endpoints on array 1, one on array 2: each array gets equal weight.
    for array, endpoint, open_kl in ((1, 11, 0.2), (1, 12, 0.4), (2, 21, 0.1)):
        for method in ("none", "calibration", "rewrite", "open_loop", "closed_loop"):
            kl = open_kl if method == "open_loop" else 0.5
            accuracy = 89.0 if method == "open_loop" else 90.0
            config = default_config(
                stage="recover",
                backend="pcm",
                method=method,
                assignment_seed=array,
                endpoint_seed=endpoint,
                max_examples=500,
            )
            rows.append(
                {
                    "config": config,
                    "run": tmp_path / f"{array}-{endpoint}-{method}",
                    "metrics": {
                        "initial": {"accuracy_percent": 85.0, "teacher_kl": 1.0},
                        "final": {
                            "accuracy_percent": accuracy,
                            "teacher_kl": kl,
                            "examples": 500,
                        },
                        "gain_pp": accuracy - 85.0,
                        "cost": {"physical_pulses": 0, "refresh_reads": 0},
                        "controller_verify_reads": 0,
                        "rewrite": None,
                        "image_presentations": 500,
                        "p0_sha256": f"p0-{array}-{endpoint}",
                        "initial_apparent_sha256": f"apparent-{array}-{endpoint}",
                    },
                }
            )
    monkeypatch.setattr(analysis, "records", lambda path: ({"phase": "pilot"}, rows))
    monkeypatch.setattr(analysis, "plot", lambda *args, **kwargs: None)
    source = tmp_path / "execution.json"
    source.write_text("{}")
    payload = analysis.report(source, tmp_path / "report")
    effect = next(
        r
        for r in payload["comparisons"]
        if r["method"] == "open_loop" and r["control"] == "calibration"
    )
    assert effect["gain_pp"] == -1.0
    assert effect["teacher_kl_reduction_nats"] == pytest.approx(0.3)
    assert not effect["inferential_claim_supported"]
    with (tmp_path / "report/teacher_metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 30
    assert all("legacy" in row["model"] for row in metrics)
    assert "Teacher KL (nats)" in (tmp_path / "report/report.md").read_text()
    rows[-1]["metrics"]["initial"]["teacher_kl"] = 2.0
    with pytest.raises(ValueError, match="initial_teacher_kl"):
        analysis.paired_rows(rows)


def test_pcm_gaussian_is_the_default_pcm_benchmark():
    from experiments.cifar_crossbar.campaign import default_backends

    assert default_backends("hwa") == ["om", "pcm_inference"]
    assert default_backends("reference") == ["pcm_inference"]
    for phase in ("pilot", "tune", "screen", "confirm"):
        assert default_backends(phase) == ["om"]


def test_strict_config_and_confirmation_boundaries():
    parse_config(default_config())
    with pytest.raises(ValueError):
        parse_config({**default_config(), "unknown": 1})
    with pytest.raises(ValueError):
        default_config(evaluation="confirmation", max_examples=10)
    with pytest.raises(ValueError):
        default_config(stage="recover", objective="cross_entropy")
    with pytest.raises(ValueError):
        default_config(stage="recover", backend="pcm_inference")
    with pytest.raises(ValueError):
        default_config(epochs=True)


def test_population_shift_does_not_allow_mixing_pcm_reference_and_pulse_hwa():
    from experiments.cifar_crossbar.runtime import verify_context

    context = {
        k: "same" for k in ("schema", "dataset", "suffix", "teacher_sha256", "mapping")
    }
    context.update(
        device_settings={"backend": "om", "noise_scale": 1.0}, read_noise=0.0
    )
    shifted = deepcopy(context)
    shifted["device_settings"]["noise_scale"] = 2.0
    verify_context(shifted, context, devices=False)
    shifted["device_settings"]["backend"] = "pcm_inference"
    with pytest.raises(ValueError, match="device family"):
        verify_context(shifted, context, devices=False)


def test_paired_statistics_cluster_endpoints_and_holm():
    from experiments.cifar_crossbar.analysis import paired_statistics, holm

    result = paired_statistics(np.ones(15), np.repeat([41, 43, 47], 5), draws=100)
    assert result["independent_arrays"] == 15
    assert result["ci95_pp"] == [1.0, 1.0]
    assert result["p_one_sided"] == 1 / 32768
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_campaign_declares_exact_full_model_pilot(tmp_path):
    from argparse import Namespace
    from experiments.cifar_crossbar.campaign import materialize
    import json

    args = Namespace(
        output=tmp_path,
        study_id="cifar-test",
        phase="pilot",
        device="cpu",
        suffix="last_two_blocks",
        disable_cudnn=False,
        datasets=["cifar10"],
        backends=["om"],
        hwa_selection=None,
        recovery_selection=None,
        screen_selection=None,
        hwa_epochs=30,
    )
    materialize(args)
    campaign = json.loads((tmp_path / "campaign.json").read_text())
    configs = [
        json.loads((tmp_path / n["config"]).read_text()) for n in campaign["nodes"]
    ]
    assert len(configs) == 10
    assert {c["method"] for c in configs if c["stage"] == "recover"} == {
        "none",
        "calibration",
        "rewrite",
        "open_loop",
        "closed_loop",
    }
    assert all(c["suffix"] == "last_two_blocks" for c in configs)
    audit = next(c for c in configs if c["stage"] == "audit")
    assert audit["evaluation"] == "confirmation" and audit["max_examples"] == 0


def test_native_cli_source_receipt_and_result_bundle(tmp_path, monkeypatch):
    import json
    from ebl.cli import main
    from experiments.cifar_crossbar import runtime
    from experiments.artifacts import sha256_file

    source = tmp_path / "source.py"
    source.write_text("verified source\n")
    receipt = tmp_path / "source-receipt.json"
    receipt.write_text(json.dumps({"files": {"source.py": sha256_file(source)}}))
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setenv("EBL_SOURCE_RECEIPT", str(receipt))
    monkeypatch.setenv("EBL_DEFER_CURRENT_SIMULATIONS", "1")
    teacher = CifarResNet32(10).eval().requires_grad_(False)
    monkeypatch.setattr(
        runtime, "source_model", lambda *args: (teacher, "teacher-fixture")
    )

    def cache(teacher, student, spec, device):
        x = torch.randn(4, 3, 32, 32)
        return {
            "features": student.features(x),
            "teacher_logits": teacher(x),
            "labels": torch.arange(4),
        }, {"count": 4}

    monkeypatch.setattr(runtime, "build_cache", cache)
    weights = tmp_path / "weights.pt"
    torch.save({}, weights)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(default_config(max_examples=4)))
    from experiments.study_workflow import prepare_study, summarize_study

    plan = {
        "schema_version": 1,
        "study_id": "cifar-cli-check",
        "title": "CLI contract check",
        "hypothesis": "The audit emits a valid native result.",
        "motivation": "Verify the complete run contract.",
        "evidence_class": "example",
        "arms": [
            {
                "arm_id": "audit",
                "configs": ["config.json"],
                "description": "Audit",
                "experiment_id": "cifar_resnet_suffix_recovery.v1",
                "mode": "train",
            }
        ],
        "completion_criteria": ["One valid native bundle."],
        "analysis_plan": ["Verify all artifact hashes."],
    }
    plan_path = tmp_path / "study-plan.json"
    plan_path.write_text(json.dumps(plan))
    study = prepare_study(plan_path, tmp_path / "results")
    output = study / "runs/audit"
    assert (
        main(
            [
                "train",
                "--config",
                str(config),
                "--teacher-weights",
                str(weights),
                "--selection-receipt",
                str(plan_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    run = next(output.iterdir())
    result = json.loads((run / "result.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    assert result["status"] == "complete"
    assert result["metrics"]["metrics"]["teacher_agreement_percent"] == 100
    assert manifest["inputs"][-1]["sha256"] == sha256_file(receipt)
    summary = summarize_study(study, verify_artifacts=True)
    assert summary["ready_for_review"]


def test_hwa_continuation_restores_optimizer_rng_and_inherited_best(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from experiments.cifar_crossbar import runtime
    from experiments.cifar_crossbar.config import Spec

    torch.manual_seed(77)
    teacher = CifarResNet32(10).eval().requires_grad_(False)
    x = torch.randn(6, 3, 32, 32)
    y = torch.arange(6)
    dataset = torch.utils.data.TensorDataset(x, y)
    dataset.targets = y.tolist()
    monkeypatch.setattr(runtime, "get_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(
        runtime,
        "split_indices",
        lambda *args, **kwargs: (np.arange(4), np.arange(4, 6)),
    )
    spec = Spec(
        **default_config(
            stage="hwa",
            suffix="head",
            epochs=3,
            batch_size=2,
            max_examples=4,
            ramp_epochs=1,
        )
    )
    kernel = tmp_path / "kernel.pt"
    runtime.save(
        kernel,
        {
            "device_settings": runtime.scientific_device(spec),
            "adequate": True,
            "residuals": torch.linspace(-0.05, 0.05, 15).reshape(3, 5),
            "heldout_residuals": torch.linspace(-0.03, 0.03, 15).reshape(3, 5),
        },
    )

    def run(name, resume=None):
        torch.manual_seed(spec.seed)
        student = CrossbarSuffix(teacher, "head")
        student.read_noise = 0.01
        context = runtime.checkpoint_context(spec, "fixture", student)
        cache = {
            "features": student.features(x[4:]),
            "teacher_logits": teacher(x[4:]),
            "labels": y[4:],
        }
        store = SimpleNamespace(run_dir=tmp_path / name, append_metric=lambda v: None)
        request = SimpleNamespace(device_model=kernel, resume=resume)
        result = runtime.fit_hwa(
            request, spec, teacher, student, context, cache, store, "cpu"
        )
        return result, runtime.load(store.run_dir / "checkpoints/weights.pt")

    expected, weights = run("continuous")
    original_save = runtime.save

    def interrupt(path, value):
        original_save(path, value)
        if path.name == "resume.pt" and value["epoch"] == 1:
            raise InterruptedError("Simulated interruption after a durable checkpoint")

    monkeypatch.setattr(runtime, "save", interrupt)
    with pytest.raises(InterruptedError):
        run("interrupted")
    monkeypatch.setattr(runtime, "save", original_save)
    actual, replayed = run("continued", tmp_path / "interrupted/checkpoints/resume.pt")
    assert actual["history"] == expected["history"]
    for key in weights["model"]:
        assert torch.equal(weights["model"][key], replayed["model"][key])
    completed, again = run("completed", tmp_path / "continued/checkpoints/resume.pt")
    assert completed["history"] == expected["history"]
    assert again["selected_epoch"] == weights["selected_epoch"]
