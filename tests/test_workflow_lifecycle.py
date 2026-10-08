"""Strict lifecycle schema, section provenance and stage planning."""

import copy
import json
from pathlib import Path

import pytest

from campaigns.schema import CampaignSpec
from experiments.definitions import load_experiment_config, parse_experiment_config
from experiments.schema import ConfigError
from workflow.networks import cifar_resnet32
from workflow.__main__ import main as workflow_main
from workflow.lifecycle import (
    ARTIFACT_SECTIONS,
    EXPERIMENT_ID,
    StageSelection,
    load_lifecycle,
    parse_execution,
    parse_lifecycle,
    parse_stage_config,
    provenance,
    stage_inputs,
    stage_selections,
    verify_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
LIFECYCLES = sorted(ROOT.glob("campaigns/**/lifecycles/*.json"))
PCM = ROOT / "campaigns/cifar-crossbar-hwa-recovery/lifecycles/cifar10-pcm-conv4-smoke.json"
OM = ROOT / "campaigns/cifar-crossbar-hwa-recovery/lifecycles/cifar10-om-conv4-smoke.json"


def document(path=PCM):
    return json.loads(path.read_text())


def changed(doc, dotted, value):
    doc = copy.deepcopy(doc)
    *parents, key = dotted.split(".")
    target = doc
    for name in parents:
        target = target[int(name)] if isinstance(target, list) else target[name]
    if isinstance(target, list):
        target[int(key)] = value
    else:
        target[key] = value
    return doc


@pytest.mark.parametrize("path", LIFECYCLES, ids=lambda p: p.stem)
def test_campaign_lifecycles_parse_and_round_trip(path):
    lifecycle = load_lifecycle(path)
    assert lifecycle.lifecycle_id == path.stem
    assert parse_lifecycle(lifecycle.to_dict()) == lifecycle
    assert parse_lifecycle(json.loads(json.dumps(lifecycle.to_dict()))).digest() == lifecycle.digest()


def test_campaign_has_lifecycles_for_both_technologies():
    assert {load_lifecycle(p).devices.technology for p in LIFECYCLES} == {"om", "pcm"}


@pytest.mark.parametrize(
    "dotted,value,message",
    [
        ("unexpected", 1, "lifecycle fields"),
        ("question.text", " padded", "question.text"),
        ("network.analog_convolutions", 6, "analog_convolutions"),
        ("data.development_images", 1001, "multiple of the 10"),
        ("devices.selection_seeds", [271001], "disjoint"),
        ("devices.assignment_seeds", [211001, 211001], "unique seeds"),
        ("devices.endpoint_seed_offset", 60000, "endpoint seed outside"),
        ("defects.cases.0", {"kind": "none", "rate_ppm": 5}, "kind 'none' exactly"),
        ("defects.placement", "bernoulli_per_active_cell", "defects.placement"),
        ("deployment.encoding", "single_cell_intrinsic_reference", "deployment.encoding"),
        ("hwa.noise_model", "om_endpoint_kernel", "hwa.noise_model"),
        ("hwa.arms.1.corruption", {"kind": "gmax", "rates_ppm": [50000, 20000]}, "increasing"),
        ("hwa.arms.0", {"id": "digital", "method": "none", "noise_strength": 0.0}, r"hwa\.arms\[0\] fields"),
        ("program_verify", {"method": "closed_loop", "max_cycles": 128, "tolerance_steps": 0.5,
                            "relaxation": {"model": "none"}}, "program_verify.method"),
        ("program_verify.relaxation", {"model": "pcm_drift", "seconds": 0.0, "compensation": True}, "seconds"),
        ("onchip.update_law", "om_closed_loop_pulse_adam", "onchip.update_law"),
        ("onchip.arms.0.weights", "learn", "unique weight/calibration"),
    ],
)
def test_lifecycle_rejects_invalid_or_unmatched_settings(dotted, value, message):
    doc = document() if dotted != "unexpected" else {**document(), "unexpected": 1}
    if dotted != "unexpected":
        doc = changed(doc, dotted, value)
    with pytest.raises(ConfigError, match=message):
        parse_lifecycle(doc)


def test_weight_writing_arm_requires_matched_no_write_control():
    doc = document()
    doc["onchip"]["arms"] = [arm for arm in doc["onchip"]["arms"] if arm["id"] != "calibration"]
    with pytest.raises(ConfigError, match="hold arm with calibration=True"):
        parse_lifecycle(doc)


def test_open_loop_law_has_no_verified_rewrite_control():
    doc = document(OM)
    doc["onchip"] = {
        key: value for key, value in doc["onchip"].items() if key not in ("pulses_per_update", "tolerance_steps")
    }
    doc["onchip"]["update_law"] = "om_open_loop_pulse_adam"
    with pytest.raises(ConfigError, match="rewrite"):
        parse_lifecycle(doc)
    doc["onchip"]["arms"] = [arm for arm in doc["onchip"]["arms"] if arm["weights"] != "rewrite"]
    assert parse_lifecycle(doc).onchip.max_pulses_per_cell == 640


def test_om_relaxation_is_not_modelled():
    doc = changed(document(OM), "program_verify.relaxation",
                  {"model": "pcm_drift", "seconds": 1.0, "compensation": True})
    with pytest.raises(ConfigError, match="relaxation.model"):
        parse_lifecycle(doc)


def test_hwa_training_requires_selection_arrays():
    doc = changed(document(), "devices.selection_seeds", [])
    with pytest.raises(ConfigError, match="selection arrays"):
        parse_lifecycle(doc)
    doc["hwa"]["arms"] = [{"id": "digital", "method": "none"}]
    assert parse_lifecycle(doc).devices.selection_seeds == ()


def test_section_provenance_allows_reuse_only_for_unchanged_dependencies():
    base = load_lifecycle(PCM)
    record = provenance(base, "hwa_source", hwa_arm="standard_hwa")
    # The question, on-chip settings and other HWA arms do not determine this source.
    later = changed(changed(document(), "question.decision", "A later decision."), "onchip.epochs", 9)
    later["hwa"]["arms"] = later["hwa"]["arms"][:2]
    verify_provenance(record, parse_lifecycle(later), "hwa_source", hwa_arm="standard_hwa")
    for dotted, value in (("program_verify.relaxation", {"model": "pcm_drift", "seconds": 1.0, "compensation": False}),
                          ("hwa.learning_rate", 0.001), ("hwa.arms.1.noise_strength", 2.0)):
        with pytest.raises(ValueError):
            verify_provenance(record, parse_lifecycle(changed(document(), dotted, value)), "hwa_source",
                              hwa_arm="standard_hwa")
    deployment = provenance(base, "deployment", hwa_arm="digital")
    with pytest.raises(ValueError, match="defects"):
        verify_provenance(deployment, parse_lifecycle(changed(document(), "defects.cases.1.rate_ppm", 30000)),
                          "deployment", hwa_arm="digital")
    assert set(ARTIFACT_SECTIONS["feature_cache"]) == {"network", "data"}


def test_stage_dag_covers_every_arm_and_assignment():
    lifecycle = load_lifecycle(PCM)
    stages = stage_selections(lifecycle)
    ids = [s.stage_id for s in stages]
    assert ids[:2] == ["prepare", "devices"] and len(ids) == len(set(ids)) == 2 + 3 + 3 + 3
    onchip = StageSelection("onchip", "cdt_gmax", 271001)
    assert stage_inputs(onchip) == {
        "teacher_weights": None,
        "device_data": ("prepare", "feature_cache"),
        "device_model": ("devices", "device_bundle"),
        "device_state": ("deploy__cdt_gmax__271001", "deployment"),
    }
    assert stage_inputs(StageSelection("deploy", "digital", 271001))["weights"] == ("hwa__digital", "hwa_source")


def test_stage_config_is_a_registered_experiment():
    lifecycle = document()
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": 1,
        "stage": {"kind": "deploy", "hwa_arm": "cdt_gmax", "array_seed": 271001},
        "execution": {"device": "cpu", "cpu_threads": 1, "disable_cudnn": False},
        "lifecycle": lifecycle,
    }
    definition, config = parse_experiment_config(payload)
    assert definition.experiment_id == EXPERIMENT_ID and config.stage.stage_id == "deploy__cdt_gmax__271001"
    assert parse_stage_config(config.to_dict()) == config
    for stage, message in (
        ({"kind": "deploy", "hwa_arm": "cdt_gmax", "array_seed": 1}, "assignment seed"),
        ({"kind": "hwa", "hwa_arm": "missing", "array_seed": None}, "HWA arm"),
        ({"kind": "prepare", "hwa_arm": "digital", "array_seed": None}, "no arm"),
    ):
        with pytest.raises(ConfigError, match=message):
            parse_stage_config({**payload, "stage": stage})
    with pytest.raises(ConfigError, match="execution.device"):
        parse_execution({"device": "gpu", "cpu_threads": 1, "disable_cudnn": False})


def test_plan_writes_runner_manifest_and_registered_stage_configs(tmp_path, monkeypatch):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"synthetic teacher")
    from experiments.artifacts import sha256_file

    monkeypatch.setitem(cifar_resnet32.SOURCES, "cifar10", ("teacher.pt", sha256_file(teacher), 93.53))
    output = tmp_path / "plan"
    assert workflow_main(["plan", str(PCM), "--teacher-weights", str(teacher), "--output", str(output)]) == 0
    manifest = json.loads((output / "campaign.json").read_text())
    spec = CampaignSpec.parse(manifest, base_dir=output)
    assert [s.stage_id for s in spec.stages][:2] == ["prepare", "devices"]
    onchip = {s.stage_id: s for s in spec.stages}["onchip__standard_hwa__271001"]
    assert onchip.inputs["device_state"].stage == "deploy__standard_hwa__271001"
    assert onchip.inputs["teacher_weights"].path == teacher.resolve()
    assert set(onchip.depends_on) == {"prepare", "devices", "deploy__standard_hwa__271001"}
    for stage in spec.stages:
        definition, config = load_experiment_config(stage.config)
        assert config.stage.stage_id == stage.stage_id and config.lifecycle == load_lifecycle(PCM)
    # Re-planning is idempotent; a different plan in the same directory is refused.
    assert workflow_main(["plan", str(PCM), "--teacher-weights", str(teacher), "--output", str(output)]) == 0
    assert workflow_main(["plan", str(PCM), "--teacher-weights", str(teacher), "--output", str(output),
                          "--device", "cuda:0"]) == 2
    other = tmp_path / "other.pt"
    other.write_bytes(b"not the teacher")
    assert workflow_main(["plan", str(PCM), "--teacher-weights", str(other), "--output", str(tmp_path / "x")]) == 2


def test_check_command_validates_campaign_lifecycles(capsys):
    assert workflow_main(["check"]) == 0
    output = capsys.readouterr().out
    assert all(p.stem in output for p in LIFECYCLES)
