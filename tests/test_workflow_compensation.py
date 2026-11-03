"""Native replay, calibrated held-state inference, strict inputs and collection."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from experiments.artifacts import sha256_file
from experiments.schema import ConfigError
from training.drift_compensation import METHODS
from workflow.__main__ import main, summary
from workflow.lifecycle import parse_lifecycle, StageSelection, Execution
from workflow.plan import build_plan
from workflow.networks import family
from workflow_helpers import run_lifecycle, run_stage
from test_workflow_cmo import cmo_inputs
from test_workflow_retention import lifecycle


def test_shared_teacher_evaluation_exactly_matches_individual_methods():
    from test_workflow_opt import tiny_teacher
    from workflow.networks import opt_mlp
    from workflow.compensation import fit
    from types import SimpleNamespace
    network = opt_mlp.OptMlpSuffix(tiny_teacher(), 2, tile_size=8, scope="model")
    tokens = torch.randint(0, 64, (5, 9))
    q = network.q.detach() * .6
    spec = SimpleNamespace(row_column_iterations=10, gain_bounds=(.1, 10))
    corrections = {method: fit(network, network.q.detach(), q, method, spec) for method in METHODS}
    expected = {}
    for method, correction in corrections.items():
        network.drift_compensation = correction
        expected[method] = opt_mlp.evaluate(network, {"tokens": tokens}, "cpu", q=q, batch_size=2)
    network.drift_compensation = None
    actual = opt_mlp.evaluate_compensations(network, {"tokens": tokens}, "cpu", q=q,
                                          corrections=corrections, batch_size=2)
    assert actual == expected
    assert network.drift_compensation is None


def settings():
    return {"methods": list(METHODS), "calibration_sequences": 4, "row_column_iterations": 10, "gain_bounds": [.1, 10.]}


@pytest.mark.parametrize("key,value", [("methods", ["column", "column"]), ("methods", ["bad"]),
                                     ("gain_bounds", [2., 3.]), ("calibration_sequences", 1000000),
                                     ("row_column_iterations", 0)])
def test_invalid_compensation_rejected(key, value):
    spec = settings(); spec[key] = value
    with pytest.raises(ConfigError):
        lifecycle(**{"drift_compensation": spec})


def test_compensation_requires_matching_retention_and_no_read_noise():
    with pytest.raises(ConfigError):
        lifecycle(**{"retention.read_noise": True, "drift_compensation": settings()})


def test_native_compensation_reuses_states_and_collects(cmo_inputs, tmp_path):
    teacher, digest = cmo_inputs
    lc = lifecycle(**{"data.corpus": {"name": "synthetic", "sha256": digest},
                      "data.sequence_length": 9, "data.max_examples": 4,
                      "hwa.arms": [{"id": "digital", "method": "none"}]})
    results = run_lifecycle(lc, teacher, tmp_path / "parent")
    doc = lc.to_dict(); doc["drift_compensation"] = settings(); doc["lifecycle_id"] += "-compensation"
    comp = parse_lifecycle(doc)
    assert summary(comp)["compensated_snapshots"] == 126
    selected = StageSelection("compensate", "digital", 271001)
    run_dir, result = run_stage(comp, selected, results, teacher, tmp_path / "compensation")
    assert result["metrics"]["source_unchanged"] and result["metrics"]["control_exact"]
    rows = json.loads((run_dir / "compensation_measurements.json").read_text())
    assert len(rows) == 126
    assert all(r["cost"]["device_writes"] == 0 and r["state_unchanged"] for r in rows)
    for row in rows:
        if row["identity_snapshot"]:
            assert row["compensated"] == row["baseline"]
            assert row["model"] == "reram_wan2022" and row["seconds"] == 1
    assert main(["collect", str(tmp_path / "compensation"), "--output", str(tmp_path / "collected")]) == 0
    flat = json.loads((tmp_path / "collected/compensation.json").read_text())
    assert len(flat) == 126 and not any(r["duplicate_coverage"] for r in flat)
    assert all(r["development_aging_kl_regained_percent"] is None for r in flat if r["identity_snapshot"])
    source = results["hwa__digital"][0] / "checkpoints/source.pt"
    parent = results["retention__digital__271001"][0] / "retention_states.json"
    with patch.object(family(lc.network), "pinned_digest", return_value=sha256_file(teacher)):
        with pytest.raises(ValueError, match="retention-source"):
            build_plan(comp, teacher_weights=teacher, execution=Execution("cpu", 1, False), python=Path('/usr/bin/python'))
        plan = build_plan(comp, teacher_weights=teacher, execution=Execution("cpu", 1, False), python=Path('/usr/bin/python'),
                          frozen_sources={"digital": source}, retention_sources={"digital:271001": parent})
    stage = next(s for s in json.loads(plan['campaign.json'])['stages'] if s['id'].startswith('compensate__'))
    assert stage['inputs']['device_state'] == {'path': str(parent.resolve())}
    # A changed parent config must be rejected before the saved state is consumed.
    config_path = parent.with_name('config.resolved.json')
    content = json.loads(config_path.read_text()); content['lifecycle']['retention']['seed'] += 1
    config_path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="parent retention configuration"):
        run_stage(comp, selected, results, teacher, tmp_path / "bad-parent")
