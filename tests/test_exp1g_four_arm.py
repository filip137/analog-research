"""Scientific coverage, frozen-source reuse and readiness release boundaries."""
import importlib.util
import json
from pathlib import Path

import pytest

from workflow.lifecycle import parse_lifecycle, provenance, verify_provenance

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp1g_four_arm", ROOT / "campaigns/opt-cmo-corruption/launch/exp1g_four_arm.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_four_recipes_change_only_declared_noise_and_corruption():
    settings = {"std3": (1., None), "std9": (3., None), "cdt3": (1., 30000), "cdt5": (1., 50000)}
    for label, (noise, faults) in settings.items():
        lc = study.lifecycle(label)
        arm = lc.to_dict()["hwa"]["arms"][0]
        assert arm["scope"] == "model" and arm["digital_learning_rate"] == 1e-5
        assert arm["noise_strength"] == noise
        assert arm["corruption"] == (None if faults is None else {"kind": "random", "rates_ppm": [faults]})
        assert lc.hwa.learning_rate == .005 and lc.hwa.noise_std == .03
        assert lc.hwa.lr_schedule.total_steps == 25600
        assert lc.hwa.lr_schedule.warmup_steps == 205
        assert lc.hwa.lr_schedule.factor(25600) == .2
        assert lc.network.analog_decoder_layers == 6
        assert len(lc.devices.assignment_seeds) == 5 and len(lc.defects.cases) == 6
        assert lc.hwa.seed == 41
    assert study.lifecycle("cdt5-smoke").hwa.lr_schedule.total_steps == 512
    assert study.lifecycle("cdt5-smoke").transfer_tests is None


@pytest.mark.parametrize("label,filename", [("std3", "opt125m-cmo-l6-exp1e-digital-lr1em5.json"), ("cdt3", "opt125m-cmo-l6-exp1f-cdt-lr1em5.json")])
def test_reused_sources_have_exact_compatible_provenance(label, filename):
    original = parse_lifecycle(json.loads((ROOT / "campaigns/opt-cmo-corruption/lifecycles" / filename).read_text()))
    arm = study.CASES[label][1]
    verify_provenance(provenance(original, "hwa_source", hwa_arm=arm), study.lifecycle(label), "hwa_source", hwa_arm=arm)


def rows_for(*labels):
    rows = []
    for i, label in enumerate(labels):
        for seed in study.lifecycle(label).devices.assignment_seeds:
            for case in study.lifecycle(label).defects.cases:
                row = {"lifecycle_id": study.lifecycle(label).lifecycle_id, "hwa_arm": study.CASES[label][1], "array_seed": seed, "case": case.label}
                for domain in ("wikitext", "generated"):
                    row["deployed_transfer_" + domain + "_teacher_kl"] = .1 - i * .015
                    row["deployed_transfer_" + domain + "_perplexity"] = 10. - i
                rows.append(row)
    return rows


def test_only_complete_four_arm_coverage_can_report_completion():
    report = study.compare_rows(rows_for("std3", "cdt3", "cdt5", "std9"))
    assert report["coverage_complete"] and report["deployment_rows"] == 120
    assert len(report["comparisons"]) == 72
    pair = next(r for r in report["comparisons"] if r["control"] == "std3" and r["selected"] == "cdt3")
    assert pair["material_rule_passed"] and pair["wins"] == 5
    assert pair["mean_paired_relative_reduction"] == pytest.approx(.15)
    assert not study.compare_rows(rows_for("std3", "cdt3"))["coverage_complete"]
    assert not study.compare_rows([])["coverage_complete"]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nonfinite", "unexpected"])
def test_invalid_coverage_cannot_be_silently_pooled(problem):
    rows = rows_for("std3")
    if problem == "missing": rows.pop()
    elif problem == "duplicate": rows.append(rows[0])
    elif problem == "nonfinite": rows[0]["deployed_transfer_wikitext_teacher_kl"] = float("nan")
    else: rows[0]["hwa_arm"] = "wrong"
    with pytest.raises(ValueError): study.compare_rows(rows)


def test_failed_canary_validation_never_releases_training(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    (evidence / "readiness").mkdir(parents=True)
    (evidence / "readiness/preflight.json").write_text('{"passed": true}')
    folder = tmp_path / "canary"
    source = folder / ("hwa__" + study.CASES["cdt5-smoke"][1])
    source.mkdir(parents=True)
    (source / "result.json").write_text('{"metrics": {"fit": {}}}')
    monkeypatch.setattr(study, "checked_collection", lambda _: {"job": "canary", "attempt": "one"})
    def invalid(*_): raise ValueError("incomplete fit")
    monkeypatch.setattr(study, "validate_source_fit", invalid)
    monkeypatch.setattr(study, "transition", lambda *_: pytest.fail("failed canary released a job"))
    with pytest.raises(ValueError, match="incomplete fit"):
        study.release_after_canary(tmp_path / "queue", evidence, folder)


def test_successful_canary_releases_only_the_two_training_cases(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    (evidence / "readiness").mkdir(parents=True)
    (evidence / "readiness/preflight.json").write_text('{"passed": true}')
    folder = tmp_path / "canary"
    source = folder / ("hwa__" + study.CASES["cdt5-smoke"][1])
    source.mkdir(parents=True)
    (source / "result.json").write_text('{"metrics": {"fit": {}}}')
    monkeypatch.setattr(study, "checked_collection", lambda _: {"job": "canary", "attempt": "one"})
    monkeypatch.setattr(study, "validate_source_fit", lambda *_: None)
    monkeypatch.setattr(study, "gate", lambda *args: None)
    calls = []
    monkeypatch.setattr(study, "transition", lambda _, action, job: calls.append((action, job)))
    assert study.release_after_canary(tmp_path / "queue", evidence, folder)["full_runs_released"]
    assert calls == [("release", study.lifecycle(l).lifecycle_id) for l in ("cdt5", "std9")]
