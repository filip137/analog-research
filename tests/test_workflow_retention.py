"""Native retention stages, source reuse, matched fault mapping and collection."""

import json
from pathlib import Path

import pytest
import torch

from experiments.cifar_crossbar.runtime import load
from experiments.schema import ConfigError
from training import reram_cmo as cmo
from workflow import retention
from workflow.__main__ import main as workflow_main, summary
from workflow.lifecycle import parse_lifecycle, provenance, verify_provenance, stage_selections
from workflow.plan import build_plan
from workflow.lifecycle import Execution
from test_workflow_cmo import cmo_inputs, small, case_doc
from workflow_helpers import run_lifecycle


def spec():
    return {"models": ["pcm", "reram_cmo", "reram_wan2022"], "seconds": [1, 86400, 172800],
            "assignment_seeds": [271001], "case_labels": ["nominal", "random_50000ppm"],
            "seed": 391013, "read_noise": False}


def lifecycle(**changes):
    return small(**{"onchip": None, "defects.cases": [case_doc(), case_doc("random", 50000)],
                    "retention": spec(), **changes})


def test_retention_is_optional_and_preserves_frozen_source_provenance():
    lc = lifecycle()
    original = lc.to_dict()
    del original["retention"]
    original = parse_lifecycle(original)
    assert "retention" not in original.to_dict()
    assert parse_lifecycle(lc.to_dict()) == lc
    for kind in ("feature_cache", "device_bundle", "hwa_source"):
        arm = "standard_hwa" if kind == "hwa_source" else None
        verify_provenance(provenance(original, kind, hwa_arm=arm), lc, kind, hwa_arm=arm)
    assert {s.kind for s in stage_selections(lc)} == {"prepare", "devices", "hwa", "retention"}
    assert summary(lc)["deployed_cases"] == 0
    assert summary(lc)["retention_snapshots"] == 48  # two sources, three models, two cases, four snapshots


@pytest.mark.parametrize("key,value", [
    ("seconds", [1, 3600, 86400]), ("seconds", [86400, 1]), ("seconds", [1, 1]),
    ("seconds", [float("nan")]), ("models", ["pcm", "pcm"]), ("models", ["unknown"]),
    ("case_labels", ["unknown"]), ("assignment_seeds", [3]), ("seed", 271001), ("read_noise", 1),
    ("wan_noise_scales", [2, 4]), ("wan_noise_scales", [1, 4, 2]), ("wan_noise_scales", [1, 1]),
    ("wan_noise_scales", [0, 1]), ("wan_noise_scales", [1, float("inf")]),
])
def test_retention_rejects_invalid_regimes(key, value):
    document = spec()
    document[key] = value
    with pytest.raises(ConfigError):
        lifecycle(**{"retention": document})


@pytest.mark.parametrize("model", retention.MODELS)
def test_mapping_preserves_weights_without_noise_and_maps_fault_quantiles(model):
    q = torch.linspace(-1, 1, 101)
    target = retention.targets(q, model)
    torch.testing.assert_close((target[0] - target[1]) / retention.MODELS[model]["normalizer"], q,
                               rtol=1e-5, atol=2e-7)
    case = lifecycle().defects.cases[1]
    mask = torch.tensor([[True, False, True]])
    quantiles = torch.tensor([[0., .5, 1.]])
    stuck = cmo.G_MIN_US + quantiles * (cmo.G_MAX_US - cmo.G_MIN_US)
    actual_mask, values = retention.mapped_faults(mask, stuck, case, model, "cpu")
    assert torch.equal(actual_mask, mask)
    expected = retention.MODELS[model]["g_min"] + quantiles * (
        retention.MODELS[model]["g_max"] - retention.MODELS[model]["g_min"])
    torch.testing.assert_close(values[mask], expected[mask])


@pytest.mark.parametrize("model", retention.MODELS)
def test_programming_matches_the_pinned_aihwkit_model(model):
    aihwkit = pytest.importorskip("aihwkit")
    assert aihwkit.__version__ == "1.1.0"
    from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
    from aihwkit.inference.noise.reram import ReRamCMONoiseModel, ReRamWan2022NoiseModel
    reference = {"pcm": lambda: PCMLikeNoiseModel(),
                 "reram_cmo": lambda: ReRamCMONoiseModel(acceptance_range=2),
                 "reram_wan2022": lambda: ReRamWan2022NoiseModel(g_max=40)}[model]()
    target = retention.targets(torch.linspace(-1, 1, 101), model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(615)
        expected = reference.apply_programming_noise_to_conductance(target)
    actual = retention.program(target, model, torch.Generator().manual_seed(615), acceptance=2, noise_scale=1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_native_retention_runs_and_collects_all_ages(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    lc = lifecycle(**{"data.corpus": {"name": "synthetic", "sha256": digest},
                      "data.sequence_length": 9, "data.max_examples": 4,
                      "hwa.arms": [{"id": "digital", "method": "none"}]})
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    [(run_dir, result)] = [v for k, v in results.items() if k.startswith("retention__")]
    assert result["metrics"]["source_unchanged"]
    assert result["metrics"]["snapshots"] == result["metrics"]["exact_replays"] == 24
    rows = json.loads((run_dir / "retention_measurements.json").read_text())
    assert all(row["cost"]["retention_device_writes"] == 0 for row in rows)
    for case in ("nominal", "random_50000ppm"):
        wan = [r for r in rows if r["case"] == case and r["model"] == "reram_wan2022"]
        assert wan[0]["apparent_sha256"] == wan[1]["apparent_sha256"]
        assert wan[0]["metrics"] == wan[1]["metrics"]
        assert wan[0]["baseline_seconds"] == 1
        group = [r for r in rows if r["case"] == case]
        assert len({json.dumps(r["fault_identity"], sort_keys=True) for r in group}) == 1
    assert workflow_main(["collect", str(tmp_path / "runs"), "--output", str(tmp_path / "collected")]) == 0
    collected = json.loads((tmp_path / "collected/retention.json").read_text())
    assert len(collected) == 24 and not any(r["duplicate_coverage"] for r in collected)
    assert all("retained_development_teacher_kl" in r for r in collected)
    # Native plan source substitution must remove HWA without weakening its downstream gate.
    source_dir, source_result = results["hwa__digital"]
    source = source_dir / next(a["path"] for a in source_result["artifacts"] if a["kind"] == "hwa_source")
    from workflow.networks import family
    from experiments.artifacts import sha256_file
    from unittest.mock import patch
    with patch.object(family(lc.network), "pinned_digest", return_value=sha256_file(weights)):
        plan = build_plan(lc, teacher_weights=weights, execution=Execution("cpu", 1, False),
                          python=Path("/usr/bin/python"), frozen_sources={"digital": source})
    manifest = json.loads(plan["campaign.json"])
    assert not any(s["id"].startswith("hwa__") for s in manifest["stages"])
    stage = next(s for s in manifest["stages"] if s["id"].startswith("retention__"))
    assert stage["inputs"]["weights"] == {"path": str(source.resolve())}
    assert "hwa__digital" not in stage["depends_on"]


def test_wan_noise_scales_are_optional_and_only_scale_aged_wan_snapshots():
    lc = lifecycle()
    assert "wan_noise_scales" not in lc.to_dict()["retention"]
    default = dict(spec(), wan_noise_scales=[1])
    assert lifecycle(retention=default).to_dict() == lc.to_dict()
    scaled = lifecycle(retention=dict(spec(), wan_noise_scales=[1, 2, 8]))
    assert scaled.to_dict()["retention"]["wan_noise_scales"] == [1.0, 2.0, 8.0]
    assert parse_lifecycle(scaled.to_dict()) == scaled
    assert scaled.retention.snapshots("pcm") == [(1.0, 1.0), (86400.0, 1.0), (172800.0, 1.0)]
    assert scaled.retention.snapshots("reram_wan2022") == [
        (1.0, 1.0), (86400.0, 1.0), (86400.0, 2.0), (86400.0, 8.0),
        (172800.0, 1.0), (172800.0, 2.0), (172800.0, 8.0)]
    assert summary(scaled)["retention_snapshots"] == 64  # 48 plus two sources x two cases x four scaled Wan ages
    for document in (dict(spec(), models=["pcm"], wan_noise_scales=[1, 2]),
                     dict(spec(), models=["reram_wan2022"], seconds=[1], wan_noise_scales=[1, 2])):
        with pytest.raises(ConfigError):
            lifecycle(retention=document)


def test_native_wan_noise_scales_replay_nominal_and_collect(cmo_inputs, tmp_path):
    from workflow.lifecycle import StageSelection, parse_lifecycle as parse
    from workflow_helpers import run_stage
    weights, digest = cmo_inputs
    common = {"data.corpus": {"name": "synthetic", "sha256": digest}, "data.sequence_length": 9,
              "data.max_examples": 4, "hwa.arms": [{"id": "digital", "method": "none"}]}
    nominal = lifecycle(**common, retention=dict(spec(), models=["reram_wan2022"], case_labels=["nominal"]))
    scaled = lifecycle(**common, retention=dict(spec(), models=["reram_wan2022"], case_labels=["nominal"],
                                                 seconds=[172800], wan_noise_scales=[1, 2, 6]))
    base = run_lifecycle(nominal, weights, tmp_path / "nominal")
    base_rows = json.loads((base[next(k for k in base if k.startswith("retention__"))][0]
                            / "retention_measurements.json").read_text())
    results = run_lifecycle(scaled, weights, tmp_path / "scaled")
    [(run_dir, result)] = [v for k, v in results.items() if k.startswith("retention__")]
    assert result["metrics"]["snapshots"] == result["metrics"]["exact_replays"] == 4
    rows = json.loads((run_dir / "retention_measurements.json").read_text())
    assert [r["snapshot"] for r in rows] == ["programmed", "t172800s", "t172800s-x2", "t172800s-x6"]
    assert [r["noise_scale"] for r in rows] == [1.0, 1.0, 2.0, 6.0]
    by_snapshot = {r["snapshot"]: r for r in base_rows}
    for row in rows[:2]:
        assert row["apparent_sha256"] == by_snapshot[row["snapshot"]]["apparent_sha256"]
        assert row["metrics"] == by_snapshot[row["snapshot"]]["metrics"]
    assert rows[1]["residual_rms"] < rows[2]["residual_rms"] < rows[3]["residual_rms"]
    assert all(r["baseline"] == rows[0]["metrics"] for r in rows)
    doc = scaled.to_dict(); doc["lifecycle_id"] += "-compensation"
    doc["drift_compensation"] = {"methods": ["global", "row_column"], "calibration_sequences": 2,
                                 "row_column_iterations": 3, "gain_bounds": [0.1, 10.0]}
    comp = parse(doc)
    assert summary(comp)["compensated_snapshots"] == 6
    comp_dir, comp_result = run_stage(comp, StageSelection("compensate", "digital", 271001), results, weights,
                                      tmp_path / "compensation")
    measured = json.loads((comp_dir / "compensation_measurements.json").read_text())
    assert [(r["noise_scale"], r["method"]) for r in measured] == [
        (s, m) for s in (1.0, 2.0, 6.0) for m in ("global", "row_column")]
    snapshots = {(x["seconds"], x["noise_scale"]): x for x in rows}
    assert all(r["uncompensated"] == snapshots[r["seconds"], r["noise_scale"]]["metrics"] for r in measured)
    for root, kind, count in ((tmp_path / "scaled", "retention", 4), (tmp_path / "compensation", "compensation", 6)):
        assert workflow_main(["collect", str(root), "--output", str(tmp_path / f"collected-{kind}")]) == 0
        flat = json.loads((tmp_path / f"collected-{kind}/{kind}.json").read_text())
        assert len(flat) == count and not any(r["duplicate_coverage"] for r in flat)
        assert {r["noise_scale"] for r in flat} == {1.0, 2.0, 6.0}
