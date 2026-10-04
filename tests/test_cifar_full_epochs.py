"""Full-cohort isolation, permanent OM faults, paired recovery and exact replay."""

from dataclasses import asdict
from types import SimpleNamespace
import json

import pytest
import torch

from experiments.cifar_crossbar.full_epoch_config import FullEpochSpec, default_config, parse_config
from experiments.cifar_crossbar.full_epoch_runtime import make_array, om_population, restore_endpoint, trajectory
from experiments.cifar_crossbar.devices import clone_cpu
from experiments.cifar_crossbar.fault_runtime import cache_context
from experiments.cifar_crossbar.model import CifarResNet32, CrossbarSuffix
from experiments.cifar_crossbar.runtime import evaluate, save, load


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def population(size, seed=151001):
    ones = torch.ones(1, size)
    return dict(kind="om", size=size, seed=seed, policy="repaired", variation=1.,
                aihwkit_version="1.1.0",
                constants=dict(dw_min=.01, dw_min_std=.2, write_noise_std=.3),
                hidden=dict(min_bound=-ones, max_bound=ones, reference=ones*.1,
                            dwmin_up=ones*.01, dwmin_down=ones*.01,
                            corrupt=torch.zeros_like(ones, dtype=torch.bool),
                            published_corrupt=torch.zeros_like(ones, dtype=torch.bool)))


def fixture(backend):
    torch.manual_seed(19)
    teacher = CifarResNet32(10).eval().requires_grad_(False)
    student = CrossbarSuffix(teacher, "head")
    features = torch.randn(6, 64)
    short = dict(features=features, teacher_logits=teacher.fc(features).detach(), labels=torch.arange(6))
    spec = parse_config(default_config(backend=backend, suffix="head", batch_size=2, max_examples=6, hwa_epochs=1,
                                       recovery_model="gaussian_endpoint_reprogramming" if backend == "pcm" else "om_closed_loop_pulse_adam"))
    target = student.q.detach().clone()
    bundle = {"dataset": "cifar10", "arrays": {"151001": population(len(target))}}
    array = make_array(target, spec, "gmax", 100000, bundle)
    return teacher, student, {k: short for k in ("training", "development", "test")}, spec, array, target, bundle


def test_contract_does_not_relax_historical_budgets():
    from experiments.cifar_crossbar.epoch_config import default_config as previous
    with pytest.raises(ValueError):
        previous(recovery_images=45000)
    assert parse_config(default_config()).recovery_images == 45000
    for change in ({"epochs": 20}, {"epochs": True}, {"recovery_images": 5000},
                   {"backend": "om"}, {"array_seed": 101001}, {"update_pulses": 128}):
        with pytest.raises(ValueError):
            default_config(**change)


@pytest.mark.parametrize("kind", ["open", "gmax", "random"])
def test_om_faults_keep_exact_reference_and_healthy_identities(kind):
    raw = population(100)
    p = om_population(raw, kind, .3, 15)
    mask = p["hidden"]["injected_faults"]
    assert mask.any() and (~mask).any()
    assert torch.equal(p["hidden"]["reference"], raw["hidden"]["reference"])
    for key, original in raw["hidden"].items():
        assert torch.equal(p["hidden"][key][~mask], original[~mask])
    assert torch.equal(p["hidden"]["max_bound"][mask], p["hidden"]["min_bound"][mask])
    assert not p["hidden"]["dwmin_up"][mask].any()


@pytest.mark.parametrize("backend", ["pcm", "om"])
@pytest.mark.parametrize("method", ["calibration", "rewrite", "onchip_weights", "onchip_calibration"])
def test_five_epochs_replay_and_fixed_devices(tmp_path, backend, method):
    _, student, cache, spec, array, target, _ = fixture(backend)
    directory = tmp_path / "case"
    p0 = {"model": clone_cpu(student.state_dict()), "array": array.state_dict()}
    save(directory / "p0.pt", p0)
    initial = array.read().clone()
    store = SimpleNamespace(run_dir=tmp_path, append_metric=lambda row: None)
    result = trajectory(student, array, target, cache, spec, method, store, directory)
    assert [r["epoch"] for r in result["curve"]] == [1, 2, 3, 4, 5]
    assert result["curve"][-1]["image_presentations"] == 30
    if method == "calibration":
        assert torch.equal(initial, array.read())
    for epoch in (1, 5):
        state = load(directory / f"{method}_epoch{epoch}.pt")
        restore_endpoint(student, array, p0, state)
        assert evaluate(student, cache["test"], "cpu", q=array.read()) == result["curve"][epoch-1]["test"]
    if backend == "pcm":
        assert result["curve"][-1]["cost"]["array_reprogram_calls"] == (0 if method == "calibration" else 15)
    else:
        assert result["curve"][-1]["cost"]["max_recovery_cell_pulses"] <= 15
        if method.startswith("onchip"):
            assert state["writer"]["step_index"] == 15


@pytest.mark.parametrize("backend", ["pcm", "om"])
def test_native_screen_artifacts_and_pairing(tmp_path, monkeypatch, backend):
    from ebl.cli import main
    from experiments.study_workflow import prepare_study, summarize_study
    from experiments.cifar_crossbar import full_epoch_runtime as runtime
    monkeypatch.setenv("EBL_DEFER_CURRENT_SIMULATIONS", "1")
    teacher, student, cache, spec, _, _, population_bundle = fixture(backend)
    monkeypatch.setattr(runtime, "source_model", lambda *a: (teacher, "fixture"))
    context = cache_context(spec, "fixture", student)
    teacher_path, data, sources, populations = [tmp_path / f"{n}.pt" for n in ("teacher", "data", "sources", "populations")]
    save(teacher_path, {})
    save(data, {"context": context, **cache})
    save(sources, {"context": context, "backend": backend, "sources": {k: {"model": student.state_dict()} for k in ("digital", "standard_hwa")}})
    save(populations, population_bundle)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(asdict(spec)))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(schema_version=1, study_id="full-epoch-test", title="Test", hypothesis="Paired recovery.",
                                    motivation="Native coverage.", evidence_class="example", completion_criteria=["One screen."],
                                    analysis_plan=["Pairing and replay."], arms=[dict(arm_id="screen", configs=[config.name], description="Test", experiment_id=spec.experiment_id, mode="train")])))
    study = prepare_study(plan, tmp_path / "results")
    command = ["train", "--config", str(config), "--output-dir", str(study / "runs/screen"),
               "--teacher-weights", str(teacher_path), "--weights", str(sources), "--device-data", str(data)]
    if backend == "om":
        command += ["--device-model", str(populations)]
    assert main(command) == 0
    run = next((study / "runs/screen").iterdir())
    result = json.loads((run / "result.json").read_text())["metrics"]
    assert result["completed_controls"] == 40
    for source in ("digital", "standard_hwa"):
        rows = [r for r in result["measurements"] if r["source"] == source and r["case"] == "nominal"]
        assert len({r["p0_sha256"] for r in rows}) == 1
        assert len({r["initial_apparent_sha256"] for r in rows}) == 1
    assert summarize_study(study, verify_artifacts=True)["ready_for_review"]


def test_native_cache_and_import_emit_required_terminal_metrics(tmp_path, monkeypatch):
    from ebl.cli import main
    from experiments.study_workflow import prepare_study, summarize_study
    from experiments.cifar_crossbar import full_epoch_runtime as runtime
    monkeypatch.setenv("EBL_DEFER_CURRENT_SIMULATIONS", "1")
    teacher, student, cache, spec, _, _, _ = fixture("pcm")
    for value in cache.values():
        value["cohort"] = {"count": 6, "indices_sha256": "test"}
    monkeypatch.setattr(runtime, "source_model", lambda *a: (teacher, "fixture"))
    monkeypatch.setattr(runtime, "build_cache", lambda *a: cache)
    teacher_path = tmp_path / "teacher.pt"
    prior_path = tmp_path / "prior.pt"
    save(teacher_path, {})
    context = cache_context(spec, "fixture", student)
    save(prior_path, {"context": {**context, "recovery_images": 5000},
                      "sources": {k: {"model": student.state_dict(), "fit": {}} for k in ("digital", "standard_hwa")}})
    arms = []
    for stage in ("cache", "sources"):
        (tmp_path / f"{stage}.json").write_text(json.dumps({**asdict(spec), "stage": stage}))
        arms.append(dict(arm_id=stage, configs=[f"{stage}.json"], description=stage, experiment_id=spec.experiment_id, mode="train"))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(schema_version=1, study_id="full-cache-test", title="Test", hypothesis="Caches have lifecycle metrics.",
                                    motivation="Native regression.", evidence_class="example", completion_criteria=["Both stages."],
                                    analysis_plan=["Artifact verification."], arms=arms)))
    study = prepare_study(plan, tmp_path / "results")
    cached = None
    for stage in ("cache", "sources"):
        command = ["train", "--config", str(tmp_path / f"{stage}.json"), "--output-dir", str(study / "runs" / stage), "--teacher-weights", str(teacher_path)]
        if stage == "sources":
            command += ["--weights", str(prior_path), "--device-data", str(cached)]
        assert main(command) == 0
        run = next((study / "runs" / stage).iterdir())
        assert (run / "metrics.jsonl").stat().st_size
        cached = run / "checkpoints/cache.pt"
    assert summarize_study(study, verify_artifacts=True)["ready_for_review"]
