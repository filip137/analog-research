"""Backward parity, persistent optimization, paired faults and native coverage."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.cifar_crossbar.epoch_config import EpochSpec, default_config, parse_config
from experiments.cifar_crossbar.epoch_runtime import lr_factor, restore_compact, select_policies, trajectory
from experiments.cifar_crossbar.fault_runtime import adapt, cache_context
from experiments.cifar_crossbar.model import CifarResNet32, CrossbarSuffix
from experiments.cifar_crossbar.pcm_faults import GaussianEndpointArray, PermanentFaults
from experiments.cifar_crossbar.runtime import evaluate, load, save


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture():
    torch.manual_seed(19)
    teacher = CifarResNet32(10).eval().requires_grad_(False)
    student = CrossbarSuffix(teacher, "head")
    x = torch.randn(6, 64)
    cache = {"features": x, "teacher_logits": teacher.fc(x).detach(), "labels": torch.arange(6)}
    target = student.q.detach().clone()
    array = GaussianEndpointArray(target, PermanentFaults.sample(len(target), "gmax", .1, 11), 27)
    return student, cache, array, target


def test_contract_preserves_old_one_pass_and_validates_new_milestones():
    from experiments.cifar_crossbar.fault_config import default_config as old
    with pytest.raises(ValueError):
        old(epochs=20)
    spec = parse_config(default_config())
    assert spec.epochs == 20
    assert [lr_factor(spec, "step_decay", e) for e in (1, 5, 6, 10, 11, 20)] == pytest.approx([1, 1, .3, .3, .09, .09])
    for change in ({"milestones": [1, 5]}, {"array_seed": 81001}, {"decay_factor": 1}, {"epochs": True}, {"noise_scale": 0}):
        with pytest.raises(ValueError):
            default_config(**change)


@pytest.mark.parametrize("method", ["calibration", "rewrite", "onchip_weights", "onchip_calibration"])
def test_epoch_one_matches_original_and_later_states_replay(tmp_path, method):
    student, cache, array, target = fixture()
    spec = EpochSpec(**default_config(suffix="head", epochs=3, milestones=(1, 2, 3), batch_size=2, decay_after=(1,)))
    p0 = {"model": {k: v.detach().clone() for k, v in student.state_dict().items()}, "array": array.state_dict()}
    store = SimpleNamespace(append_metric=lambda row: None)
    old = adapt(student, array, target, cache, cache, spec, method, store, "old", "cpu")
    student.load_state_dict(p0["model"])
    array.load_state_dict(p0["array"])
    save(tmp_path / "p0.pt", p0)
    result = trajectory(student, array, target, cache, cache, spec, method, "constant", store, "new", "cpu", test=cache, checkpoint=tmp_path / "curve.pt")
    states = load(tmp_path / "curve.pt")["milestones"]
    assert result["curve"][0]["test"] == old["final"]
    assert torch.equal(states["1"]["conductance"], old["array"]["g"])
    assert torch.equal(states["1"]["rng"], old["array"]["rng"])
    assert result["curve"][-1]["cost"]["array_reprogram_calls"] == (0 if method == "calibration" else 9)
    assert result["curve"][-1]["image_presentations"] == 18
    for state in states["3"]["optimizer"]["state"].values():
        assert state["step"].item() == 9  # Adam was not reset between passes.
    for e in (1, 2, 3):
        restore_compact(student, array, p0, states[str(e)])
        assert evaluate(student, cache, "cpu", q=array.read()) == result["curve"][e-1]["test"]
    if method == "calibration":
        assert torch.equal(array.g, p0["array"]["g"])


def test_selection_uses_development_only_and_has_no_final_array_inputs():
    spec = EpochSpec(**default_config(epochs=2, milestones=(1, 2)))
    rows = []
    for method in ("calibration", "onchip_weights", "onchip_calibration"):
        for schedule in spec.schedules:
            rows.append({"method": method, "schedule": schedule,
                         "initial_development": {"teacher_kl": 2},
                         "curve": [{"development": {"teacher_kl": 1}},
                                   {"development": {"teacher_kl": .8 if schedule == "constant" else .4},
                                    "test": {"teacher_kl": -999 if schedule == "constant" else 999}}]})
    selected = select_policies(rows, spec)
    assert all(r["schedule"] == "step_decay" and r["selected_epoch"] == 2 for r in selected.values())
    assert selected["rewrite"] == selected["calibration"]


def test_native_tune_screen_artifacts_pairing_and_no_test_in_tuning(tmp_path, monkeypatch):
    from ebl.cli import main
    from experiments.cifar_crossbar import epoch_runtime as runtime
    from experiments.study_workflow import prepare_study, summarize_study

    monkeypatch.setenv("EBL_DEFER_CURRENT_SIMULATIONS", "1")
    student, short, _, _ = fixture()
    teacher = student.model
    monkeypatch.setattr(runtime, "source_model", lambda *a: (teacher, "fixture"))
    spec = EpochSpec(**default_config(stage="tune", suffix="head", epochs=2, milestones=(1, 2),
                                      tuning_arrays=1, fault_kinds=("gmax",), max_examples=6, batch_size=2))
    context = cache_context(spec, "fixture", student)

    def cache(n):
        return {k: v.repeat((n + len(v) - 1) // len(v), *([1] * (v.ndim - 1)))[:n] for k, v in short.items()}

    data = tmp_path / "cache.pt"
    save(data, {"context": context, "training": cache(5000), "development": cache(1000), "test": cache(10000)})
    teacher_file = tmp_path / "teacher.pt"
    save(teacher_file, {})
    bundle = {"context": context, "sources": {k: {"model": student.state_dict(), "fit": {"convergence_review_required": True}} for k in ("digital", "standard_hwa", "noise_hwa", "cdt_open", "cdt_gmax", "cdt_random")}}
    source = tmp_path / "source.pt"
    save(source, {"context": context, "source_bundle": bundle, "selected": {"calibration": .0003, "onchip_weights": .0001}})
    plans = []
    for stage in ("tune", "screen"):
        path = tmp_path / f"{stage}.json"
        path.write_text(json.dumps(asdict(replace(spec, stage=stage))))
        plans.append({"arm_id": stage, "configs": [path.name], "description": stage, "experiment_id": spec.experiment_id, "mode": "train"})
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "study_id": "epoch-test", "title": "Epoch test",
                               "hypothesis": "Paired recovery preserves state.", "motivation": "Native check.",
                               "evidence_class": "example", "completion_criteria": ["Both stages."],
                               "analysis_plan": ["Exact parity."], "arms": plans}))
    study = prepare_study(plan, tmp_path / "results")
    original_trajectory = runtime.trajectory

    def no_test_tuning(*args, **kwargs):
        assert kwargs.get("test") is None
        return original_trajectory(*args, **kwargs)

    monkeypatch.setattr(runtime, "trajectory", no_test_tuning)
    for stage in ("tune", "screen"):
        if stage == "screen":
            monkeypatch.setattr(runtime, "trajectory", original_trajectory)
        output = study / "runs" / stage
        assert main(["train", "--config", str(tmp_path / f"{stage}.json"), "--output-dir", str(output),
                     "--teacher-weights", str(teacher_file), "--device-data", str(data), "--weights", str(source)]) == 0
        run = next(output.iterdir())
        value = json.loads((run / "result.json").read_text())["metrics"]
        if stage == "tune":
            assert value["completed_trajectories"] == 12
            source = run / "checkpoints/selection.pt"
        else:
            assert value["completed_controls"] == 60
            for name in bundle["sources"]:
                paired = [r for r in value["measurements"] if r["source"] == name and r["case"] == "nominal"]
                assert len({r["p0_sha256"] for r in paired}) == 1
                assert len({r["initial_apparent_sha256"] for r in paired}) == 1
                for r in paired:
                    if r["method"] != "none":
                        assert r["curve"][-1]["cost"]["array_reprogram_calls"] == (0 if r["method"] == "calibration" else 6)
    assert summarize_study(study, verify_artifacts=True)["ready_for_review"]
