"""Focused HWA development must not release partial or invalid evidence."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from experiments.artifacts import sha256_file
from workflow.lifecycle import parse_lifecycle, provenance


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp1e_development", ROOT / "campaigns/opt-cmo-corruption/launch/exp1e_development.py"
)
development = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(development)
DECISION_SPEC = importlib.util.spec_from_file_location(
    "exp1e_decision",
    ROOT / "campaigns/opt-cmo-corruption/series/001-hwa-robustness/analysis/exp1e_decision.py",
)
decision = importlib.util.module_from_spec(DECISION_SPEC)
DECISION_SPEC.loader.exec_module(decision)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def case_bundle(root, rate=1e-4, schedule="linear", *, score=0.1, rejected=False, duration=False):
    lc = development.duration_lifecycle(rate) if duration else development.lifecycle(rate, schedule)
    config = root / f"{lc.lifecycle_id}.json"
    run = root / lc.lifecycle_id / "attempt-1"
    result_path = run / "result.json"
    write_json(config, lc.to_dict())
    write_json(run / "config.resolved.json", {
        "lifecycle": lc.to_dict(), "stage": {"kind": "hwa", "hwa_arm": development.ARM, "array_seed": None},
    })
    epochs = range(0, 5 if rejected else lc.hwa.epochs + 1, 2)
    history = [{
        "epoch": epoch, "global_step": epoch * 256, "elapsed_seconds": float(epoch),
        "teacher_kl": score + (50 - epoch) * 0.001,
        "cases": [{"case": case.label, "teacher_kl": score + (50 - epoch) * 0.001}
                  for case in lc.hwa.arms[0].selection_cases],
    } for epoch in epochs]
    steps_run = 1200 if rejected else lc.hwa.lr_schedule.total_steps
    fit = {
        "status": "rejected_nonfinite" if rejected else "complete", "history": history,
        "epochs_run": 4 if rejected else lc.hwa.epochs, "maximum_epochs": lc.hwa.epochs,
        "selected_epoch": history[-1]["epoch"], "development_kl": history[-1]["teacher_kl"],
        "convergence_review_required": True,
        "rejection": {"epoch": 5, "batch": 176, "reason": "nonfinite HWA objective"} if rejected else None,
        "lr_schedule": {**lc.hwa.lr_schedule.to_dict(), "steps_run": steps_run,
                        "step_convention": "one-based; learning rates applied before optimizer.step",
                        "base_learning_rates": [rate, rate],
                        "final_learning_rates": [rate * lc.hwa.lr_schedule.factor(steps_run)] * 2},
    }
    source = run / "checkpoints/source.pt"
    source.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"provenance": provenance(lc, "hwa_source", hwa_arm=development.ARM),
                "fit": fit, "arm": lc.hwa.arms[0].to_dict(), "model": {"q": torch.tensor([0.0])}}, source)
    result = {
        "status": "complete", "duration_seconds": 50.0,
        "metrics": {"fit": fit, "test_used_for_selection": False,
                    "clean_development": {"teacher_kl": 0.01}},
        "artifacts": [{"path": "checkpoints/source.pt", "sha256": sha256_file(source),
                       "size_bytes": source.stat().st_size, "kind": "hwa_source"}],
    }
    write_json(result_path, result)
    return config, result_path


def mutate_result(path, edit):
    result = json.loads(path.read_text())
    edit(result)
    write_json(path, result)


def test_grid_is_matched_and_tests_are_held_out(tmp_path):
    prepared = development.prepare(tmp_path)
    assert len(prepared) == 12
    cases = [parse_lifecycle(json.loads(Path(item["path"]).read_text())) for item in prepared]
    full = [lc for lc in cases if not lc.lifecycle_id.endswith("-smoke")]
    assert len(full) == 10 and len({lc.lifecycle_id for lc in cases}) == 12
    assert {(lc.hwa.learning_rate, lc.hwa.lr_schedule.kind) for lc in full} == {
        (r, s) for r in (1e-5, 1e-4, 1e-3, 5e-3, 1e-2)
        for s in ("linear_warmup_decay", "warmup_step_decay")
    }
    for lc in cases:
        assert lc.data.evaluation == "development" and lc.transfer_tests is None
        assert lc.data.test_sequences == 0
        assert lc.data.corpus == "opt125m-synthetic-50m"
        assert lc.hwa.noise_std == 0.03 and lc.hwa.arms[0].noise_strength == 1
        assert lc.hwa.arms[0].corruption is None and lc.hwa.arms[0].scope == "model"
        assert lc.hwa.learning_rate == lc.hwa.arms[0].digital_learning_rate
        assert lc.hwa.lr_milestones == () and lc.hwa.lr_decay == 1 and lc.hwa.extension_epochs == 0
        assert lc.hwa.selection_every == 2
        smoke = lc.lifecycle_id.endswith("-smoke")
        assert lc.hwa.lr_schedule.total_steps == (512 if smoke else 12800)
        assert lc.hwa.lr_schedule.warmup_steps == (9 if smoke else 205)
        if lc.hwa.lr_schedule.kind == "warmup_step_decay":
            assert lc.hwa.lr_schedule.milestone_step == (409 if smoke else 10240)
            assert lc.hwa.lr_schedule.decay_factor == 0.1
    first = full[0]
    for lc in full:
        for section in ("network", "data", "devices", "defects", "deployment", "program_verify"):
            assert lc.section_digest(section) == first.section_digest(section)
        assert lc.hwa.seed == first.hwa.seed
        assert lc.hwa.arms[0].selection_cases == first.hwa.arms[0].selection_cases
    assert development.prepare(tmp_path) == prepared
    target = Path(prepared[0]["path"])
    write_json(target, {})
    with pytest.raises(ValueError, match="replace"):
        development.prepare(tmp_path)


def test_read_case_distinguishes_nonfinite_science_from_incomplete_execution(tmp_path):
    config, result = case_bundle(tmp_path / "complete")
    case = development.read_case(config, result)
    assert case["fit_status"] == "complete" and case["selected_step"] == 12800
    config, result = case_bundle(tmp_path / "rejected", rejected=True)
    case = development.read_case(config, result)
    assert case["fit_status"] == "rejected_nonfinite"
    assert development.recommend([case]) is None
    mutate_result(result, lambda r: r.update(status="failed"))
    with pytest.raises(ValueError, match="incomplete"):
        development.read_case(config, result)


def test_expanded_grid_preserves_old_ids_digests_and_existing_file_bytes(tmp_path):
    for rate, label in ((1e-5, "1em5"), (1e-4, "1em4"), (1e-3, "1em3"), (5e-3, "5em3")):
        for schedule in development.SCHEDULES:
            lc = development.lifecycle(rate, schedule)
            assert lc.lifecycle_id == f"opt125m-cmo-l6-exp1e-dev-{schedule}-lr{label}"
            old = parse_lifecycle(json.loads((development.LIFECYCLES / f"{lc.lifecycle_id}.json").read_text()))
            assert lc.digest() == old.digest()
            assert lc.hwa_arm_digest(development.ARM) == old.hwa_arm_digest(development.ARM)
            # Equivalent pre-existing formatting must not be rewritten.
            (tmp_path / f"{lc.lifecycle_id}.json").write_text(json.dumps(old.to_dict(), separators=(",", ":")))
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
    development.prepare(tmp_path)
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items())
    for schedule in development.SCHEDULES:
        lc = development.lifecycle(1e-2, schedule)
        assert lc.lifecycle_id == f"opt125m-cmo-l6-exp1e-dev-{schedule}-lr1em2"
        expanded, prior = lc.to_dict(), development.lifecycle(5e-3, schedule).to_dict()
        expanded["lifecycle_id"] = prior["lifecycle_id"]
        expanded["question"]["text"] = prior["question"]["text"]
        expanded["hwa"]["learning_rate"] = prior["hwa"]["learning_rate"]
        expanded["hwa"]["arms"][0]["digital_learning_rate"] = prior["hwa"]["arms"][0]["digital_learning_rate"]
        assert expanded == prior


@pytest.mark.parametrize("failure", [
    "test_receipt", "nonfinite_history", "nonfinite_case", "nonfinite_clean", "history_missing",
    "history_duplicate", "history_order", "short_run", "step_count", "schedule_mismatch",
    "selected_mismatch", "not_minimum", "artifact_digest", "artifact_missing",
])
def test_read_case_rejects_invalid_evidence(tmp_path, failure):
    config, result = case_bundle(tmp_path)
    payload = json.loads(result.read_text())
    fit = payload["metrics"]["fit"]
    if failure == "test_receipt":
        payload["metrics"]["test_used_for_selection"] = True
    elif failure == "nonfinite_history":
        fit["history"][1]["teacher_kl"] = float("nan")
    elif failure == "nonfinite_case":
        fit["history"][1]["cases"][0]["teacher_kl"] = float("inf")
    elif failure == "nonfinite_clean":
        payload["metrics"]["clean_development"]["teacher_kl"] = float("nan")
    elif failure == "history_missing":
        del fit["history"][1]
    elif failure == "history_duplicate":
        fit["history"].insert(1, fit["history"][0])
    elif failure == "history_order":
        fit["history"][1], fit["history"][2] = fit["history"][2], fit["history"][1]
    elif failure == "short_run":
        fit["epochs_run"] = 49
    elif failure == "step_count":
        fit["lr_schedule"]["steps_run"] = 12799
    elif failure == "schedule_mismatch":
        fit["lr_schedule"]["warmup_steps"] = 200
    elif failure == "selected_mismatch":
        fit["development_kl"] = 0.01
    elif failure == "not_minimum":
        fit["selected_epoch"] = 0
        fit["development_kl"] = fit["history"][0]["teacher_kl"]
    elif failure == "artifact_digest":
        payload["artifacts"][0]["sha256"] = "0" * 64
    else:
        (result.parent / "checkpoints/source.pt").unlink()
    write_json(result, payload)
    with pytest.raises((ValueError, FileNotFoundError)):
        development.read_case(config, result)


def test_result_cannot_redefine_a_grid_cell(tmp_path):
    config, result = case_bundle(tmp_path)
    doc = json.loads(config.read_text())
    doc["hwa"]["noise_std"] = 0.09
    write_json(config, doc)
    resolved = json.loads((result.parent / "config.resolved.json").read_text())
    resolved["lifecycle"] = doc
    write_json(result.parent / "config.resolved.json", resolved)
    with pytest.raises(ValueError):
        development.read_case(config, result)


def inspection_receipt(result):
    payload = json.loads(result.read_text())
    receipt = {"schema": "exp1e.artifact_inspection.v1", "result_sha256": sha256_file(result),
               "verified": True, "verify_artifacts": True, "require_complete": True,
               "exit_code": 0, "artifacts": payload["artifacts"]}
    write_json(result.parent / "artifact-inspection.json", receipt)
    return receipt


def test_remote_artifacts_require_a_bound_successful_inspection_receipt(tmp_path):
    config, result = case_bundle(tmp_path)
    source = result.parent / "checkpoints/source.pt"
    receipt = inspection_receipt(result)
    source.unlink()
    case = development.read_case(config, result)
    assert case["source_verification"]["kind"] == "remote_inspection"
    for field, value in (("verified", False), ("verify_artifacts", False), ("require_complete", False),
                         ("exit_code", 1), ("result_sha256", "0" * 64), ("artifacts", [])):
        invalid = {**receipt, field: value}
        write_json(result.parent / "artifact-inspection.json", invalid)
        with pytest.raises(ValueError, match="receipt"):
            development.read_case(config, result)
    write_json(result.parent / "artifact-inspection.json", receipt)
    source.write_bytes(b"corrupt local bytes")
    with pytest.raises(ValueError, match="digest"):
        development.read_case(config, result)


def test_exact_checkpoint_score_ties_keep_the_earliest_checkpoint(tmp_path):
    config, result = case_bundle(tmp_path)
    payload = json.loads(result.read_text())
    fit = payload["metrics"]["fit"]
    for item in fit["history"][-2:]:
        item["teacher_kl"] = 0.1
        for case in item["cases"]:
            case["teacher_kl"] = 0.1
    fit["development_kl"] = 0.1
    write_json(result, payload)
    with pytest.raises(ValueError, match="minimum"):
        development.read_case(config, result)
    fit["selected_epoch"] = 48
    write_json(result, payload)
    assert development.read_case(config, result)["selected_step"] == 48 * 256


def test_recommendation_ties_choose_lower_lr_then_linear():
    def cell(name, lr, schedule, score, status="complete"):
        return {"id": name, "peak_lr": lr, "schedule": schedule,
                "selected_development_kl": score, "fit_status": status}
    high = cell("high", 1e-3, "linear_warmup_decay", 1.0)
    low_step = cell("low-step", 1e-5, "warmup_step_decay", 1.009)
    low_linear = cell("low-linear", 1e-5, "linear_warmup_decay", 1.01)
    rejected = cell("unstable", 1e-5, "linear_warmup_decay", 0.1, "rejected_nonfinite")
    assert development.recommend([high, low_step, rejected]) == "low-step"
    assert development.recommend([high, low_step, low_linear, rejected]) == "low-linear"
    low_linear["selected_development_kl"] = 1.011
    low_step["selected_development_kl"] = 1.011
    assert development.recommend([high, low_step, low_linear]) == "high"


def test_missing_cases_never_release_recommendation_and_duplicate_attempts_fail(tmp_path):
    entries = []
    for rate in development.RATES:
        for schedule in development.SCHEDULES:
            config, result = case_bundle(tmp_path / "cases", rate, schedule)
            entries.append({"lifecycle": str(config), "result": str(result)})
    index = tmp_path / "index.json"
    write_json(index, entries[:-1])
    partial = development.summarize(index, tmp_path / "partial")
    assert partial["coverage_complete"] is False and partial["recommendation"] is None
    assert len(partial["missing"]) == 1
    write_json(index, entries)
    report = development.summarize(index, tmp_path / "complete")
    assert report["coverage_complete"] and report["recommendation"] is not None
    assert report["downstream_gate"] == "awaiting_scientific_review"
    assert report["test_used_for_selection"] is False
    assert report["plots"] == ["development-curves.png", "development-curves.pdf"]
    assert all((tmp_path / "complete" / filename).stat().st_size > 1000 for filename in report["plots"])
    assert {item["kind"] for item in report["warnings"]} == {"late_checkpoint", "grid_edge"}
    write_json(index, entries + entries[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        development.summarize(index, tmp_path / "duplicate")


def test_initial_checkpoint_warning_does_not_trigger_an_automatic_grid_extension():
    cases = [{"id": "initial", "fit_status": "complete", "selected_step": 0,
              "total_steps": 12800, "peak_lr": 1e-5}]
    warnings = development.review_warnings(cases, "initial")
    assert [item["kind"] for item in warnings] == ["initial_checkpoint", "grid_edge"]
    assert "No automatic extension" in warnings[-1]["message"]


def test_previous_high_rates_are_interior_and_one_em2_is_the_high_boundary():
    for rate, is_boundary in ((1e-5, True), (1e-3, False), (5e-3, False), (1e-2, True)):
        cases = [{"id": "winner", "fit_status": "complete", "selected_step": 512,
                  "total_steps": 12800, "peak_lr": rate}]
        warnings = development.review_warnings(cases, "winner")
        assert any(item["kind"] == "grid_edge" for item in warnings) is is_boundary


def test_original_eight_results_remain_partial_until_both_new_rates_finish(tmp_path, monkeypatch):
    entries = []
    for rate in (1e-5, 1e-4, 1e-3, 5e-3):
        for schedule in development.SCHEDULES:
            config, result = case_bundle(tmp_path / "cases", rate, schedule)
            entries.append({"lifecycle": str(config), "result": str(result)})
    index = tmp_path / "index.json"
    write_json(index, entries)
    monkeypatch.setattr(development, "plot_development", lambda *args: [])
    report = development.summarize(index, tmp_path / "review")
    assert report["coverage_complete"] is False and report["recommendation"] is None
    assert len(report["expected"]) == 10
    assert set(report["missing"]) == {development.case_id(1e-2, schedule) for schedule in development.SCHEDULES}


def test_duration_preparation_changes_only_budget_and_preserves_grid_bytes(tmp_path):
    development.prepare(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
    prepared = development.prepare_duration(tmp_path)
    assert len(prepared) == 2
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items())
    for rate, record in zip(development.DURATION_RATES, prepared):
        short = development.lifecycle(rate, "step")
        long = parse_lifecycle(json.loads(Path(record["path"]).read_text()))
        assert long.lifecycle_id == short.lifecycle_id + "-tail4pass"
        assert long.digest() == record["lifecycle_sha256"]
        assert long.hwa.epochs == 100 and long.hwa.lr_schedule.total_steps == 25600
        assert long.hwa.lr_schedule.warmup_steps == 205
        assert long.hwa.lr_schedule.milestone_step == 10240
        assert long.hwa.lr_schedule.decay_factor == 0.1
        assert [long.hwa.lr_schedule.factor(step) for step in range(12801)] == [
            short.hwa.lr_schedule.factor(step) for step in range(12801)]
        assert long.hwa.lr_schedule.factor(25600) == 0.1
        doc, original = long.to_dict(), short.to_dict()
        doc["lifecycle_id"] = original["lifecycle_id"]
        doc["question"]["text"] = original["question"]["text"]
        doc["hwa"]["epochs"] = original["hwa"]["epochs"]
        doc["hwa"]["lr_schedule"]["total_steps"] = original["hwa"]["lr_schedule"]["total_steps"]
        assert doc == original
    assert development.prepare_duration(tmp_path) == prepared
    with pytest.raises(ValueError, match="predeclared"):
        development.duration_lifecycle(1e-2)
    write_json(Path(prepared[0]["path"]), {})
    with pytest.raises(ValueError, match="replace"):
        development.prepare_duration(tmp_path)


@pytest.mark.parametrize("failure", ["none", "epochs", "maximum_epochs", "steps", "history", "config"])
def test_duration_sources_require_full_frozen_100_epoch_budget(tmp_path, failure):
    config, result = case_bundle(tmp_path, 1e-3, "step", duration=True)
    payload = json.loads(result.read_text())
    fit = payload["metrics"]["fit"]
    if failure == "epochs":
        fit["epochs_run"] = 50
    elif failure == "maximum_epochs":
        fit["maximum_epochs"] = 50
    elif failure == "steps":
        fit["lr_schedule"]["steps_run"] = 12800
    elif failure == "history":
        fit["history"] = fit["history"][:26]
    elif failure == "config":
        doc = json.loads(config.read_text())
        doc["hwa"]["lr_schedule"]["warmup_steps"] = 410
        write_json(config, doc)
    write_json(result, payload)
    if failure == "none":
        case = development.read_case(config, result)
        assert case["comparison"] == "duration" and case["steps_run"] == 25600
        assert len(case["history"]) == 51 and case["selected_epoch"] == 100
    else:
        with pytest.raises(ValueError):
            development.read_case(config, result)


def test_duration_summary_is_separate_and_reports_matched_terminal_components(tmp_path, monkeypatch):
    monkeypatch.setattr(development, "plot_development", lambda *args: [])
    entries = []
    for rate in development.RATES:
        for schedule in development.SCHEDULES:
            config, result = case_bundle(tmp_path, rate, schedule, score=0.0999 if rate == 5e-3 else 0.1)
            entries.append({"lifecycle": str(config), "result": str(result)})
    for rate in development.DURATION_RATES:
        config, result = case_bundle(tmp_path, rate, "step", duration=True,
                                    score=0.0999 if rate == 5e-3 else 0.1)
        entries.append({"lifecycle": str(config), "result": str(result)})
    # Give healthy/faulty components different values while preserving the mean.
    for entry in entries:
        result = Path(entry["result"])
        def different_components(payload):
            for point in payload["metrics"]["fit"]["history"]:
                point["cases"][0]["teacher_kl"] = point["teacher_kl"] * 0.8
                point["cases"][1]["teacher_kl"] = point["teacher_kl"] * 1.2
        mutate_result(result, different_components)
    index = tmp_path / "index.json"
    write_json(index, entries)
    report = development.summarize(index, tmp_path / "review")
    assert len(report["cases"]) == len(report["expected"]) == 10
    assert report["recommendation"] == development.case_id(1e-5, "linear")
    duration = report["duration_comparison"]
    assert duration["coverage_complete"] and duration["reference_coverage_complete"]
    assert len(duration["cases"]) == 2 and not duration["missing"]
    assert duration["recommendation"] == development.duration_lifecycle(1e-3).lifecycle_id
    assert duration["downstream_gate"] == report["downstream_gate"] == "awaiting_scientific_review"
    assert duration["test_used_for_selection"] is False
    assert "more selection opportunities" in duration["selection_caveat"]
    for comparison in duration["comparisons"]:
        assert comparison["history_prefix_identical"] is True
        assert comparison["prior_replay_change"]["mean_teacher_kl_delta"] == 0
        assert comparison["checkpoint_12800"] == comparison["prior_checkpoint_12800"]
        assert comparison["terminal_25600"]["step"] == 25600
        for kind in ("selected", "terminal"):
            delta = comparison[f"{kind}_change_12800_to_25600"]
            assert delta["mean_teacher_kl_delta"] == pytest.approx(-0.05)
            assert delta["case_teacher_kl_deltas"]["nominal"] == pytest.approx(-0.04)
            assert delta["case_teacher_kl_deltas"]["random_30000ppm"] == pytest.approx(-0.06)
        assert comparison["terminal_change_20480_to_25600"]["mean_teacher_kl_delta"] == pytest.approx(-0.02)
    assert (tmp_path / "review/duration/duration.md").exists()
    assert json.loads((tmp_path / "review/duration/duration.json").read_text()) == duration
    write_json(index, entries[:-1])
    partial = development.summarize(index, tmp_path / "partial")
    assert partial["coverage_complete"] and partial["recommendation"] == report["recommendation"]
    assert not partial["duration_comparison"]["coverage_complete"]
    assert partial["duration_comparison"]["recommendation"] is None
    monkeypatch.setattr("sys.argv", ["exp1e_development.py", "summarize", "--index", str(index),
                                     "--output", str(tmp_path / "partial-cli")])
    assert development.main() == 2
    write_json(index, entries + entries[-1:])
    with pytest.raises(ValueError, match="Duplicate"):
        development.summarize(index, tmp_path / "duplicate")


def test_duration_prefix_difference_and_missing_reference_remain_visible(tmp_path):
    originals, durations = [], []
    for rate in development.DURATION_RATES:
        originals.append(development.read_case(*case_bundle(tmp_path, rate, "step")))
        durations.append(development.read_case(*case_bundle(tmp_path, rate, "step", duration=True, score=0.1001)))
    report = development.duration_review(durations, originals)
    assert report["coverage_complete"]
    assert all(item["history_prefix_identical"] is False for item in report["comparisons"])
    assert all(item["prior_replay_change"]["mean_teacher_kl_delta"] == pytest.approx(0.0001)
               for item in report["comparisons"])
    assert sum(item["kind"] == "prefix_replay_difference" for item in report["warnings"]) == 2
    missing = development.duration_review(durations, originals[:1])
    assert not missing["reference_coverage_complete"] and missing["recommendation"] is None
    assert missing["missing_complete_references"] == [development.case_id(5e-3, "step")]


def test_rejected_duration_does_not_fabricate_terminal_or_selected_improvement(tmp_path):
    rejected = development.read_case(*case_bundle(tmp_path, 1e-3, "step", duration=True, rejected=True))
    report = development.duration_review([rejected], [])
    comparison = report["comparisons"][0]
    assert comparison["checkpoint_12800"] is None and comparison["terminal_25600"] is None
    assert comparison["selected_change_12800_to_25600"] is None
    assert comparison["terminal_change_12800_to_25600"] is None
    assert report["recommendation"] is None


@pytest.mark.parametrize("rejected", [False, True])
def test_summary_cli_returns_failure_for_partial_or_all_rejected_grid(tmp_path, monkeypatch, capsys, rejected):
    entries = []
    grid = [(r, s) for r in development.RATES for s in development.SCHEDULES]
    for rate, schedule in grid if rejected else grid[:1]:
        config, result = case_bundle(tmp_path / "cases", rate, schedule, rejected=rejected)
        entries.append({"lifecycle": str(config), "result": str(result)})
    index, output = tmp_path / "index.json", tmp_path / "review"
    write_json(index, entries)
    monkeypatch.setattr("sys.argv", ["exp1e_development.py", "summarize", "--index", str(index), "--output", str(output)])
    assert development.main() == 2
    report = json.loads((output / "development.json").read_text())
    assert report["coverage_complete"] is rejected
    assert report["recommendation"] is None
    assert report["downstream_gate"] == "awaiting_scientific_review"
    capsys.readouterr()


def deployment_rows(*, legacy=False):
    arms = ["pretrained", *decision.BASES]
    arms += [f"{base}_model" + ("_dlr1em4" if legacy else "") for base in decision.BASES]
    rows = []
    for arm in arms:
        for rate in decision.RATES:
            for seed in decision.SEEDS:
                rows.append({
                    "hwa_arm": arm, "defect_rate_ppm": rate, "array_seed": seed,
                    "duplicate_coverage": False, "lifecycle_id": decision.MAIN,
                    "evaluation": "test" if legacy else "development", "technology": "cmo",
                    "defect_kind": "random" if rate else "none", "defect_die_concentration": None,
                    "defect_array_concentration": None, "hwa_convergence_review_required": False,
                    # Raw deployed_teacher_kl deliberately differs: it is development.
                    "deployed_teacher_kl": 1.0 if legacy else 100.0,
                    "deployed_transfer_wikitext_teacher_kl": 1.0,
                    "deployed_transfer_generated_teacher_kl": 0.5,
                })
    return rows


def write_deployments(root, data):
    write_json(root / decision.MAIN / "collected/deployments.json", data)
    reference = [{**row, "lifecycle_id": decision.REFERENCE, "evaluation": "test", "deployed_teacher_kl": 1.0}
                 for row in data if row["hwa_arm"] == "pretrained"]
    write_json(root / decision.REFERENCE / "collected/deployments.json", reference)


def test_final_decision_uses_transfer_tests_and_reproduces_only_pretrained(tmp_path, capsys):
    write_deployments(tmp_path, deployment_rows())
    # No earlier analog-only result files exist: the controls are freshly trained.
    assert decision.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "pretrained reproduction" in out and "absolute KL change" in out
    assert "WikiText KL 1" in out and "generated KL 0.5" in out


@pytest.mark.parametrize("failure", ["missing", "duplicate", "duplicate_flag", "nonfinite", "reproduction",
                                      "missing_domain", "wrong_evaluation", "unknown_arm", "wrong_seed"])
def test_final_decision_fails_before_interpretation_for_invalid_evidence(tmp_path, capsys, failure):
    data = deployment_rows()
    write_deployments(tmp_path, data)
    if failure == "missing":
        data.pop()
    elif failure == "duplicate":
        data.append(dict(data[-1]))
    elif failure == "duplicate_flag":
        data[-1]["duplicate_coverage"] = True
    elif failure == "nonfinite":
        data[-1]["deployed_transfer_generated_teacher_kl"] = float("nan")
    elif failure == "reproduction":
        data[0]["deployed_transfer_wikitext_teacher_kl"] = 1.1
    elif failure == "missing_domain":
        del data[-1]["deployed_transfer_wikitext_teacher_kl"]
    elif failure == "wrong_evaluation":
        data[-1]["evaluation"] = "test"
    elif failure == "unknown_arm":
        data[-1]["hwa_arm"] = "standard_hwa_model_old"
    else:
        data[-1]["array_seed"] = 9999
    write_json(tmp_path / decision.MAIN / "collected/deployments.json", data)
    assert decision.main([str(tmp_path)]) != 0
    captured = capsys.readouterr()
    assert "FAIL" in captured.err and "scope effect" not in captured.out


def test_legacy_analysis_requires_explicit_contract_and_model_tag(tmp_path, capsys):
    write_deployments(tmp_path, deployment_rows(legacy=True))
    assert decision.main([str(tmp_path)]) != 0
    assert decision.main([str(tmp_path), "--contract", "legacy-v1"]) != 0
    assert decision.main([str(tmp_path), "--contract", "legacy-v1", "--legacy-model-tag", "dlr1em4"]) == 0
    assert "analysis contract: legacy-v1" in capsys.readouterr().out


def test_material_comparison_never_downgrades_missing_arrays_to_no_difference():
    a = {seed: 0.8 for seed in decision.SEEDS}
    b = {seed: 1.0 for seed in decision.SEEDS}
    assert decision.compare(a, b) == (5, 5, pytest.approx(0.2), True)
    a.pop(decision.SEEDS[-1])
    with pytest.raises(ValueError, match="all five"):
        decision.compare(a, b)
