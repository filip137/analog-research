"""Reject incomplete or unpaired five-full-epoch evidence before reporting."""

from copy import deepcopy
import pytest

from experiments.cifar_crossbar.full_epoch_report import CASES, METHODS, SOURCES, case_label, summarize, validate


def fixture():
    metrics = dict(examples=10000, teacher_kl=.1, accuracy_percent=90., teacher_agreement_percent=95.)
    rows = []
    for source in SOURCES:
        for case in CASES:
            for method in METHODS:
                curve = [] if method == "none" else [dict(epoch=e, image_presentations=e*45000, test=metrics,
                    cost={"array_reprogram_calls": 0 if method=="calibration" else e*704}) for e in range(1,6)]
                rows.append(dict(source=source, case=case, method=method, curve=curve, p0_sha256=source+case,
                                 initial_apparent_sha256=source+case, identity={"seed":151001}, initial_test=metrics, clean=metrics))
    return dict(smoke_only=False, recovery_images=45000, epochs=5, backend="pcm", measurements=rows)


def test_accepts_complete_five_full_epochs():
    assert len(validate(fixture())) == 40
    assert case_label("open_10000ppm", "om") == "stuck-low 1%"
    assert case_label("open_10000ppm", "pcm") == "open 1%"


@pytest.mark.parametrize("error", ["missing", "p0", "identity", "epochs", "cohort", "writes", "test"])
def test_rejects_unmatched_or_incomplete_evidence(error):
    value = deepcopy(fixture())
    row = value["measurements"][1]
    if error == "missing": value["measurements"].pop()
    if error == "p0": row["p0_sha256"] = "bad"
    if error == "identity": row["identity"] = {"seed":2}
    if error == "epochs": row["curve"].pop()
    if error == "cohort": row["curve"][0]["image_presentations"] = 5000
    if error == "writes": row["curve"][0]["cost"]["array_reprogram_calls"] = 1
    if error == "test": row["curve"][0]["test"]["examples"] = 1000
    with pytest.raises(ValueError): validate(value)


def test_summary_requires_three_distinct_arrays():
    rows = [dict(group="a", array_seed=s, score=float(s)) for s in (1,2,3)]
    result = summarize(rows, ["group"], ["score"])
    assert result[0]["score_mean"] == 2
    assert result[0]["score_sd"] == 1
    rows[2]["array_seed"] = 2
    with pytest.raises(ValueError): summarize(rows, ["group"], ["score"])


def test_collected_four_study_report_covers_all_arrays_and_hwa_controls(tmp_path, monkeypatch):
    import json
    from experiments.artifacts import atomic_write_json, sha256_file
    from experiments.cifar_crossbar import full_epoch_report as module
    monkeypatch.setattr(module, "plot", lambda *a: None)
    for dataset in ("cifar10", "cifar100"):
        for backend in ("pcm", "om"):
            study = tmp_path / "results" / f"{dataset}-{backend}-five-full-epochs-v1"
            completed = {}
            for stage in ("cache", "sources", "screen1", "screen2", "screen3"):
                node = f"{dataset}_{backend}_{stage}"
                run = study / "runs" / node / "test-run"
                if stage.startswith("screen"):
                    value = deepcopy(fixture())
                    value.update(stage="screen", dataset=dataset, backend=backend)
                    for row in value["measurements"]:
                        row.update(array_seed=151000+int(stage[-1]), backend=backend)
                        row["identity"]["physical_devices"] = 100
                        if backend == "om":
                            for point in row["curve"]:
                                point["cost"] = {"max_recovery_cell_pulses": 0, "capped_cells": 0}
                else:
                    value = dict(stage=stage, sources={"standard_hwa": {"sha256": "fixture", "fit": {}}})
                atomic_write_json(run / "result.json", {"metrics": value})
                remote = "/home/filip/cifar_full_epoch_sources/v2/" + str(run.relative_to(tmp_path))
                completed[node] = dict(run_dir=remote, result_sha256=sha256_file(run / "result.json"))
            atomic_write_json(study / "campaign-execution.json", {"completed": completed, "active": None})
    result = module.report(tmp_path, tmp_path / "analysis")
    assert len(result["summary"]) == 800
    assert len(result["paired"]) == 640
    assert len(result["cross_source"]) == 960
    assert len(result["costs"]) == 384
    assert {r["control"] for r in result["cross_source"]} == {"none", "calibration", "rewrite"}
    assert len((tmp_path / "analysis/per_array.csv").read_text().splitlines()) == 2401
    om = module.report(tmp_path, tmp_path / "om-analysis", ("om",))
    assert om["controls"] == 240
    assert om["post_recovery_test_endpoints"] == 960
    assert {r["backend"] for r in om["summary"]} == {"om"}
