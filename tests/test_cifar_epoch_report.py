"""Report guards reject missing coverage and unequal programming budgets."""

from copy import deepcopy
import json

import pytest

from experiments.cifar_crossbar.epoch_report import METHODS, cross_source_pairs, flatten, paired, summarize


def evidence():
    sources = ("digital", "standard_hwa", "noise_hwa", "cdt_open", "cdt_gmax", "cdt_random")
    common = dict(smoke_only=False, epochs=20, milestones=[1, 2, 5, 10, 20],
                  recovery_images=5000, evaluation_images=10000, development_images=1000)
    runs = [{"config": {"stage": "tune", "dataset": "cifar10"},
             "metrics": {**common, "completed_trajectories": 48,
                         "selected": {m: {"schedule": "constant", "selected_epoch": 20} for m in METHODS}}}]
    for seed in (101001, 101002, 101003):
        records = []
        for source in sources:
            kinds = (source[4:],) if source.startswith("cdt_") else ("open", "gmax", "random")
            for kind in ("none", *kinds):
                case = "nominal" if kind == "none" else f"{kind}_10000ppm"
                for method in ("none", *METHODS):
                    curve = []
                    if method != "none":
                        for epoch in range(1, 21):
                            metrics = {"teacher_kl": 1 / epoch, "accuracy_percent": 90 + epoch / 10, "examples": 10000}
                            curve.append({"epoch": epoch, "development": metrics, "test": metrics,
                                          "teacher_kl_train_mean": 1 / epoch,
                                          "image_presentations": 5000 * epoch,
                                          "cost": {"array_reprogram_calls": 0 if method == "calibration" else 79 * epoch}})
                    records.append(dict(source=source, case=case, method=method, array_seed=seed,
                                        fault_kind=kind, fault_rate_ppm=0 if kind == "none" else 10000,
                                        faults={"identity_sha256": f"{seed}/{kind}"},
                                        p0_sha256=f"{source}/{kind}/{seed}", initial_apparent_sha256="same",
                                        initial_test={"teacher_kl": 2, "accuracy_percent": 80},
                                        schedule="constant" if method != "none" else "none", curve=curve))
        runs.append({"config": {"stage": "screen", "dataset": "cifar10", "array_seed": seed},
                     "metrics": {**common, "completed_controls": 90, "measurements": records}})
    return runs


def test_counts_and_paired_epoch_gains():
    rows, development, policies = flatten(evidence())
    assert len(rows) == 1350
    assert len(development) == 4320
    assert len(summarize(rows)) == 450
    assert policies["cifar10"]["calibration"]["schedule"] == "constant"
    contrasts = paired(rows)
    value = next(r for r in contrasts if r["epoch"] == 20 and r["comparison"] == "same_method_epoch1")
    assert value["kl_reduction_mean"] == pytest.approx(.95)
    assert value["accuracy_gain_pp_mean"] == pytest.approx(1.9)
    assert value["arrays_both_improved"] == 3
    contrasts = cross_source_pairs(rows)
    assert contrasts and all(r["arrays"] == 3 for r in contrasts)
    assert any(r["source"] == "digital" and r["control_source"] == "noise_hwa" for r in contrasts)
    changed = deepcopy(rows)
    next(r for r in changed if r["source"] == "digital" and r["method"] == "onchip_weights" and r["epoch"] == 20)["fault_identity_sha256"] = "unpaired"
    with pytest.raises(ValueError, match="physical failure"):
        cross_source_pairs(changed)


@pytest.mark.parametrize("fault", ["budget", "p0", "array", "smoke"])
def test_rejects_invalid_comparison(fault):
    runs = deepcopy(evidence())
    if fault == "budget":
        runs[1]["metrics"]["measurements"][2]["curve"][0]["cost"]["array_reprogram_calls"] = 0
    elif fault == "p0":
        runs[1]["metrics"]["measurements"][2]["p0_sha256"] = "different"
    elif fault == "array":
        runs[-1] = deepcopy(runs[1])
    else:
        runs[1]["metrics"]["smoke_only"] = True
    with pytest.raises(ValueError):
        flatten(runs)


def test_complete_report_exports_all_budgets_and_selected_endpoints(tmp_path, monkeypatch):
    from experiments.cifar_crossbar import epoch_report
    execution = tmp_path / "execution.json"
    execution.write_text("{}\n")
    monkeypatch.setattr(epoch_report, "records", lambda path: ({"phase": "recovery_epochs"}, evidence()))
    monkeypatch.setattr(epoch_report, "plots", lambda *args: None)
    counts = epoch_report.report([execution], tmp_path / "report")
    assert counts == {"final_controls": 270, "post_adaptation_test_endpoints": 1080, "final_learning_trajectories": 216}
    value = json.loads((tmp_path / "report/results.json").read_text())
    assert value["counts"] == counts
    assert len((tmp_path / "report/development_selected_endpoints.csv").read_text().splitlines()) == 271
    assert "No test-based schedule" in (tmp_path / "report/report.md").read_text()
