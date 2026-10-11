"""Digital LoRA must learn without altering the physical array or source model."""
import json
import pytest
import torch

from experiments.cifar_crossbar.runtime import load
from experiments.schema import ConfigError
from workflow.lifecycle import DigitalLora, parse_lifecycle
from workflow.networks.opt_mlp import OptMlpSuffix
from test_workflow_cmo import cmo_inputs, small, case_doc
from test_workflow_opt import tiny_teacher, tokens
from workflow_helpers import run_lifecycle


def arm(rank=4, rate=.001):
    return {"id": "lora", "weights": "hold", "calibration": False,
            "digital_adapter": {"kind": "lora", "rank": rank, "alpha": rank, "learning_rate": rate}}


def test_lora_zero_initialization_and_logical_weight_correction():
    net = OptMlpSuffix(tiny_teacher(), 2, 8, scope="model")
    ids = tokens()
    x = net.features(ids)
    before = net.forward_features(x).detach()
    base = net.base_state_hash()
    net.enable_lora(DigitalLora(4, 8, .001), 53)
    assert torch.equal(before, net.forward_features(x))
    assert net.base_state_hash() == base
    with torch.no_grad():
        net.lora[0]["b"].fill_(.03)
    inputs = torch.randn(2, 3, 16)
    spec = net.layout[0]
    weights = net.q[:spec.size].reshape(spec.shape) * net.scales[0]
    delta = net.lora[0]["b"] @ net.lora[0]["a"] * 2
    expected = (inputs @ (weights + delta).T) * net.output_gains[0] + net.mlp_biases[0]
    assert torch.allclose(net._matrix(inputs, 0, net.q), expected, atol=1e-6)
    net.enable_lora(None)
    assert torch.equal(before, net.forward_features(x))


@pytest.mark.parametrize("bad", [0, -1, 769])
def test_lora_rank_is_strict(bad):
    with pytest.raises(ConfigError):
        small(**{"onchip.arms": [{"id": "none", "weights": "hold", "calibration": False}, arm(bad)]})


def test_lora_requires_frozen_array_and_matching_hold():
    for arms in ([arm()], [dict(arm(), weights="learn")], [dict(arm(), calibration=True)]):
        with pytest.raises(ConfigError):
            small(**{"onchip.arms": arms})


def test_lora_native_roundtrip_preserves_all_base_parameters_and_device_state(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    lc = small(**{
        "data.corpus": {"name": "synthetic", "sha256": digest},
        "data.sequence_length": 9, "data.max_examples": 8,
        "defects.cases": [case_doc("random", 30000)],
        "hwa.arms": [{"id": "digital", "method": "hwa", "scope": "model", "digital_learning_rate": 0, "corruption": None, "noise_strength": 1.0, "selection_cases": [case_doc()]}],
        "onchip.batch_size": 4,
        "onchip.arms": [{"id": "none", "weights": "hold", "calibration": False}, arm(),
                        dict(arm(2), id="lora2")],
    })
    assert parse_lifecycle(lc.to_dict()) == lc
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    from workflow.portable_deployment import materialize, validate
    portable = tmp_path / "portable.pt"
    deployed, deploy_result = results["deploy__digital__271001"]
    device_run, device_result = results["devices"]
    bundle = device_run / next(a["path"] for a in device_result["artifacts"] if a["kind"] == "device_bundle")
    receipt = materialize(deployed, bundle, portable, "cpu")
    assert validate(portable) == receipt
    assert load(portable)["materialized_from_sha256"] == next(a["sha256"] for a in deploy_result["artifacts"] if a["kind"] == "deployment")
    assert all(c["array"]["kind"] == "layered_cmo" for c in load(portable)["cases"].values())
    run, _ = results["onchip__digital__271001"]
    rows = json.loads((run / "measurements.json").read_text())
    for row in rows[1:]:
        assert row["replay_exact"]
        assert row["initial_apparent_sha256"] == row["final_apparent_sha256"] == rows[0]["initial_apparent_sha256"]
        assert row["digital_adapter"]["base_unchanged"]
        assert row["digital_adapter"]["array_unchanged"]
        assert row["cost"]["device_writes"] == row["cost"]["stuck_write_attempts"] == 0
        assert row["cost"]["device_programs"] == row["cost"]["verify_reads"] == 0
        state = load(run / "checkpoints/onchip" / row["case"] / (row["onchip_arm"] + ".pt"))
        assert not state["calibration"]
        assert state["array"]["kind"] != "layered_cmo_deployment"
        assert any(bool(v.ne(0).any()) for k,v in state["digital_adapter"]["state"].items() if k.endswith(".b"))
        assert row["sequence_presentations"] == lc.onchip.epochs * 8
    # Build a recovery-only campaign around the native trajectory with all
    # upstream inputs pinned explicitly, then exercise transport validation.
    from campaigns.recovery_collect import export, checked
    from campaigns.runner import _campaign_contract_record
    from campaigns.schema import CampaignSpec
    from experiments.artifacts import sha256_file
    import copy
    import sys
    root = tmp_path / "runs"
    native_manifest = json.loads((run / "manifest.json").read_text())
    inputs = {r["role"]: {k:r[k] for k in ("path","sha256")} for r in native_manifest["inputs"]}
    selection = json.loads((run / "config.resolved.json").read_text())["stage"]
    stage_id = "onchip__digital__271001"
    manifest = {"schema_version":1,"campaign_id":"runs",
        "targets":[{"id":"local","worktree":str(tmp_path),"python":sys.executable}],
        "stages":[{"id":stage_id,"case_id":"digital__271001","target":"local","command":"train",
            "config":str(run / "config.resolved.json"),"depends_on":[],
            "inputs":{k:{"path":v["path"]} for k,v in inputs.items()}}]}
    manifest_path = tmp_path / "campaign.json"; manifest_path.write_text(json.dumps(manifest))
    contract = _campaign_contract_record(CampaignSpec.parse(manifest,base_dir=tmp_path),manifest_path=manifest_path)
    (root / "campaign.resolved.json").write_text(json.dumps(contract))
    (root / "aggregate").mkdir()
    (root / "aggregate/results.json").write_text(json.dumps({"stages":[{"stage_id":stage_id,"status":"complete",
        "run_result":str(run / "result.json"),"run_result_sha256":sha256_file(run / "result.json") }]}))
    pinned = {"artifacts":inputs,"array_seed":271001,
              "apparent_sha256":{r["case"]:r["initial_apparent_sha256"] for r in rows}}
    payload = export(root,pinned)
    assert checked(payload,pinned) == payload
    for field in ("rows_json","rows_csv"):
        altered = copy.deepcopy(payload); altered[field] = "[]"
        with pytest.raises(ValueError): checked(altered,pinned)
    altered_inputs = copy.deepcopy(pinned)
    altered_inputs["artifacts"]["device_state"]["sha256"] = "0" * 64
    with pytest.raises(ValueError): checked(payload,altered_inputs)
