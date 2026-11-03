"""Pilot provenance, full-pass coverage, and fail-closed measured release."""
import importlib.util
import json
from pathlib import Path

import pytest

from ebl.schema.lifecycle import parse_lifecycle, provenance, verify_provenance

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp2_dense_baseline", ROOT / "campaigns/opt-cmo-corruption/launch/exp2_dense_baseline.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_full_pass_preserves_frozen_source_and_excludes_test_training():
    original = parse_lifecycle(json.loads((ROOT / "campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1e-digital-lr1em5.json").read_text()))
    for label, examples in (("l6-canary", 8192), ("50m", 195226)):
        lc = study.lifecycle(label)
        verify_provenance(provenance(original, "hwa_source", hwa_arm=study.ARM), lc, "hwa_source", hwa_arm=study.ARM)
        assert lc.data.cohort_size(lc.onchip.cohort_key) == examples
        assert lc.onchip.learning_rate == .005 and lc.onchip.epochs == 1
        assert lc.onchip.batch_size == 32 and lc.onchip.calibration_learning_rate == 0
        assert not any(a.calibration for a in lc.onchip.arms)
        assert len(lc.defects.cases) == 1 and lc.defects.cases[0].rate_ppm == 30000
        assert lc.network.analog_decoder_layers == 6
        assert lc.data.development_sequences == 256 and lc.data.recovery_sequences == 8192
    assert study.lifecycle("l1-smoke").network.analog_decoder_layers == 1


def test_frozen_source_manifest_trains_one_array_without_hwa_retraining(tmp_path, monkeypatch):
    import ebl.lifecycle.plan as plan
    from core.execution import Execution
    # Only the pinned teacher hash is mocked; the actual native stage graph is built.
    monkeypatch.setattr(plan, "sha256_file", lambda _: study.PINNED["facebook/opt-125m"]["sha256"])
    lc = study.lifecycle("50m")
    files = plan.build_plan(lc, teacher_weights=tmp_path / "teacher.pt", execution=Execution("cpu", 1, False), python=Path("/usr/bin/python3"))
    plan.write_plan(files, tmp_path)
    path = tmp_path / "campaign.json"
    manifest = study.execution_manifest(path, "50m")
    ids = [s["id"] for s in manifest["stages"]]
    assert len([s for s in ids if s.startswith("deploy__")]) == 5
    assert [s for s in ids if s.startswith("onchip__")] == [f"onchip__{study.ARM}__{study.ARRAY}"]
    assert not any(s.startswith("hwa__") for s in ids)
    for stage in manifest["stages"]:
        if stage["id"].startswith("deploy__"):
            assert stage["inputs"]["weights"] == {"path": study.source()["original_source"]}


@pytest.mark.parametrize("failure", [None, "budget", "memory", "coverage"])
def test_canary_projection_does_not_use_performance_to_choose_lr(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(study, "checked_collection", lambda _: {})
    monkeypatch.setattr(study, "DEADLINE", "2099-01-01T00:00:00+00:00")
    stage = tmp_path / f"onchip__{study.ARM}__{study.ARRAY}"
    stage.mkdir()
    learned = {"onchip_arm": "onchip_weights", "sequence_presentations": 8192,
               "curve": [{"epoch_seconds": 60., "development": {"teacher_kl": 1000.}}]}
    if failure == "coverage": learned["sequence_presentations"] = 195226
    if failure == "budget": learned["curve"][0]["epoch_seconds"] = 1000.
    (stage / "measurements.json").write_text(json.dumps([learned]))
    memory = 27000 if failure == "memory" else 16000
    (stage / "metrics.jsonl").write_text(json.dumps({"stage": "onchip_memory", "peak_reserved_bytes": memory * 1024**2}) + "\n")
    if failure:
        with pytest.raises(ValueError): study.measured_projection(tmp_path)
    else:
        projection = study.measured_projection(tmp_path)
        assert projection["full_total_seconds_with_margin"] == 2831
