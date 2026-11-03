"""Analog LoRA adapters on a frozen HWA base (exp-019): network, targeted faults, schema and a full lifecycle.

A tiny random OPT and a tiny PIQA-format task stand in for OPT-125M. The
whole-model HWA base comes from a native run of a base lifecycle, as the
std3 source does for the real experiment.
"""

import copy
import json

import pytest
import torch

from core.artifacts import sha256_file
from core.tensor_io_native import load
from core.contracts import ConfigError
import hardware.arrays as device_stage
import ebl.runtime.lifecycle as runtime
from ebl.lifecycle.cli import main as workflow_main
from ebl.schema.lifecycle import BASE_SOURCE, parse_lifecycle, stage_inputs, stage_selections
from networks.config import AnalogLora
from hardware.config import DefectCase
import networks.opt as opt_mlp
from ebl.execution.runner import _validate_run_result
from test_workflow_cmo import NAMESPACE, case_doc, cmo_inputs, small  # noqa: F401  (fixture)
from test_workflow_opt import tiny_teacher, tokens
from test_workflow_tasks import tiny_task
from workflow_helpers import request_for
from networks.mapping import fault_domains

LORA = AnalogLora(rank=2, alpha=4.0)


def adapter_network(seed=7):
    return opt_mlp.OptMlpSuffix(tiny_teacher(seed), 2, tile_size=8, scaling="per_output_channel_absmax",
                                analog_lora=LORA)


def test_zero_adapters_reproduce_the_base_and_set_adapters_add_their_low_rank_path():
    network = adapter_network()
    base = opt_mlp.OptMlpSuffix(tiny_teacher(), 2, tile_size=8, scaling="per_output_channel_absmax")
    assert network.base_matrices == len(base.layout) == 4 and len(network.layout) == 12
    assert [s.name for s in network.layout[4:6]] == ["decoder.layers.1.fc1.lora_a", "decoder.layers.1.fc1.lora_b"]
    assert network.q.numel() == base.q.numel() + 2 * 2 * 2 * (16 + 32)
    ids = tokens()
    with torch.no_grad():
        assert torch.equal(network(ids), base(ids))
        q = network.q.detach().clone()
        generator = torch.Generator().manual_seed(3)
        x = torch.randn(5, 16, generator=generator)
        a, b = network.adapters[0]
        for index in (a, b):
            spec = network.layout[index]
            q[spec.offset:spec.offset + spec.size] = torch.randn(spec.size, generator=generator)
        expected = network._matrix(x, 0, network.q)
        A = q[network.layout[a].offset:network.layout[a].offset + network.layout[a].size].reshape(network.layout[a].shape)
        B = q[network.layout[b].offset:network.layout[b].offset + network.layout[b].size].reshape(network.layout[b].shape)
        added = ((x @ A.T) @ B.T) * LORA.alpha / LORA.rank * network.output_gains[0]
        assert torch.allclose(network._matrix(x, 0, q), expected + added, atol=1e-5)
    assert len(network.output_gains) == 4 and len(network.calibration_parameters()) == 0


def test_targeted_faults_split_the_untargeted_fault_map_exactly():
    layout = adapter_network().layout
    maps = {target: device_stage.hierarchical_faults(layout, 8, 12, DefectCase("random", 50000, hierarchical=True,
                                                                                  target=target), 271002, NAMESPACE, domains=fault_domains(layout))
            for target in ("all", "base", "adapter")}
    for spec in layout:
        adapter = fault_domains(layout)[spec.name].adapter
        mask_all = maps["all"]["matrices"][spec.name][0]
        assert torch.equal(maps["adapter" if adapter else "base"]["matrices"][spec.name][0], mask_all)
        assert not bool(maps["base" if adapter else "adapter"]["matrices"][spec.name][0].any())
    receipt = maps["adapter"]["receipt"]
    assert receipt["failed_devices"] == receipt["failed_adapter_devices"] > 0 and receipt["target"] == "adapter"
    assert maps["all"]["receipt"]["failed_devices"] == (maps["base"]["receipt"]["failed_devices"]
                                                        + maps["adapter"]["receipt"]["failed_devices"])


# --- schema --------------------------------------------------------------------------------


def lora_doc(base_doc, task_sha, base_sha, **changes):
    doc = copy.deepcopy(base_doc)
    doc["lifecycle_id"] = "tiny-analog-lora"
    doc["network"]["analog_lora"] = {"rank": 2, "alpha": 4.0}
    doc["task"] = {"name": "piqa", "choices": {"name": "tinytask", "sha256": task_sha}, "decision": "acc_norm",
                   "development_examples": 4, "evaluation_examples": 6, "training_examples": 8}
    doc["question"]["primary_metric"] = "task_accuracy_norm_percent"
    doc["hwa"]["objective"] = "task_choice_token_mean"
    doc["hwa"]["arms"] = [{"id": "lora", "method": "lora", "noise_strength": 1.0, "selection_cases": [case_doc()],
                           "base_source": {"name": "tiny-std", "sha256": base_sha}, "learning_rate": 0.01,
                           "warmup_fraction": 0.1}]
    doc["defects"]["cases"] = [case_doc(), {**case_doc("random", 50000), "target": "base"},
                               {**case_doc("random", 50000), "target": "adapter"}, case_doc("random", 50000)]
    doc["onchip"]["arms"] = [{"id": "none", "weights": "hold", "calibration": False},
                             {"id": "pair_rewrite_verify", "weights": "hold", "calibration": False,
                              "fault_rewrite": {"detection": "verify", "roles": "pair", "seed": 241016,
                                                "threshold_sigma": 5.0}}]
    for key, value in changes.items():
        *parents, last = key.split(".")
        target = doc
        for part in parents:
            target = target[part]
        target[last] = value
    return doc


def base_doc(digest):
    doc = small(**{"data.corpus": {"name": "synthetic", "sha256": digest}, "data.sequence_length": 9,
                   "data.max_examples": 8, "defects.cases": [case_doc()], "hwa.batch_size": 4,
                   "onchip.batch_size": 4}).to_dict()
    doc["lifecycle_id"] = "tiny-std-base"
    doc["hwa"]["arms"] = [{"id": "std_model", "method": "hwa", "noise_strength": 1.0, "corruption": None,
                           "selection_cases": [case_doc()], "scope": "model", "digital_learning_rate": 1e-5}]
    return doc


def test_schema_keeps_digests_without_adapters_and_rejects_partial_lora_settings():
    plain = parse_lifecycle(base_doc("1" * 64))
    assert "analog_lora" not in plain.to_dict()["network"]
    assert all("target" not in case for case in plain.to_dict()["defects"]["cases"])
    lc = parse_lifecycle(lora_doc(base_doc("1" * 64), "2" * 64, "3" * 64))
    assert [c.label for c in lc.defects.cases] == ["nominal", "random_50000ppm_base", "random_50000ppm_adapter",
                                                   "random_50000ppm"]
    assert lc.network.adapter_weights == 2 * 2 * 2 * (768 + 3072)
    assert stage_inputs(stage_selections(lc)[2], lc)["weights"] == BASE_SOURCE
    assert "task" in runtime.provenance(lc, "hwa_source", hwa_arm="lora")["sections"]
    for changes, message in (({"hwa.objective": "teacher_kl"}, "LoRA arms"),
                             ({"task.training_examples": None}, "training_examples"),
                             ({"deployment.scaling": "per_layer_absmax"}, "per_output_channel_absmax")):
        doc = lora_doc(base_doc("1" * 64), "2" * 64, "3" * 64, **changes)
        if doc["task"].get("training_examples") is None:
            del doc["task"]["training_examples"]
        with pytest.raises(ConfigError, match=message):
            parse_lifecycle(doc)
    doc = base_doc("1" * 64)
    doc["defects"]["cases"] = [{**case_doc("random", 50000), "target": "adapter"}]
    with pytest.raises(ConfigError, match="analog_lora"):
        parse_lifecycle(doc)


# --- the full lifecycle -----------------------------------------------------------------


def run_with_base(lc, teacher, base_path, root):
    results = {}
    for selection in stage_selections(lc):
        inputs = {}
        for role, upstream in stage_inputs(selection, lc).items():
            if upstream is None:
                inputs[role] = teacher
            elif upstream == BASE_SOURCE:
                inputs[role] = base_path
            else:
                run_dir, result = results[upstream[0]]
                [artifact] = [a for a in result["artifacts"] if a["kind"] == upstream[1]]
                inputs[role] = run_dir / artifact["path"]
        output = root / selection.stage_id
        assert runtime.run_train(request_for(lc, selection, inputs, output)) == 0
        [run_dir] = sorted(output.iterdir())
        results[selection.stage_id] = (run_dir, dict(_validate_run_result(run_dir / "result.json")))
    return results


def test_lora_lifecycle_trains_adapters_on_the_frozen_base_and_deploys_targeted_faults(cmo_inputs, tmp_path, monkeypatch):
    weights, digest = cmo_inputs
    base_lc = parse_lifecycle(base_doc(digest))
    base = run_with_base(base_lc, weights, None, tmp_path / "base")
    base_run, base_result = base["hwa__std_model"]
    [artifact] = [a for a in base_result["artifacts"] if a["kind"] == "hwa_source"]
    base_path = base_run / artifact["path"]
    _, task_sha = tiny_task(tmp_path / "corpora")
    lc = parse_lifecycle(lora_doc(base_doc(digest), task_sha, sha256_file(base_path)))
    results = run_with_base(lc, weights, base_path, tmp_path / "runs")
    _, hwa = results["hwa__lora"]
    fit = hwa["metrics"]["fit"]
    assert fit["kind"] == "lora" and fit["base_unchanged"] and fit["base_source"]["lifecycle_id"] == "tiny-std-base"
    assert [item["epoch"] for item in fit["history"]] == [0, 1] and fit["adapters"]["weights"] == 2 * 2 * 2 * 48
    source = load(results["hwa__lora"][0] / "checkpoints/source.pt")
    base_model = load(base_path)["model"]
    base_size = base_model["q"].numel()
    assert torch.equal(source["model"]["q"][:base_size], base_model["q"])
    assert all(torch.equal(source["model"][k], v) for k, v in base_model.items() if k.startswith("digital."))
    run, onchip = results["onchip__lora__271001"]
    rows = {(r["case"], r["onchip_arm"]): r for r in json.loads((run / "measurements.json").read_text())}
    identities = {case: rows[case, "none"]["identity"] for case, _ in rows}
    assert identities["random_50000ppm_adapter"]["failed_adapter_devices"] == identities["random_50000ppm_adapter"]["failed_devices"] > 0
    assert identities["random_50000ppm_base"]["failed_adapter_devices"] == 0
    assert identities["random_50000ppm"]["failed_devices"] == (identities["random_50000ppm_base"]["failed_devices"]
                                                               + identities["random_50000ppm_adapter"]["failed_devices"])
    assert all(row["replay_exact"] and "task_test" in row["final"] for row in rows.values())
    from core.execution import Execution
    from ebl.lifecycle.plan import build_plan
    monkeypatch.setattr(opt_mlp, "pinned_digest", lambda network: sha256_file(weights))
    plan = dict(lc=lc, teacher_weights=weights, execution=Execution("cpu", 1, False), python=weights)
    with pytest.raises(ValueError, match="pinned LoRA base"):
        build_plan(plan.pop("lc"), **plan, lora_bases={"lora": weights})
    files = build_plan(lc, teacher_weights=weights, execution=Execution("cpu", 1, False), python=weights,
                       lora_bases={"lora": base_path})
    manifest = json.loads(files["campaign.json"])
    [stage] = [s for s in manifest["stages"] if s["id"] == "hwa__lora"]
    assert stage["inputs"]["weights"] == {"path": str(base_path.resolve())}
    collected = tmp_path / "collected"
    assert workflow_main(["collect", str(tmp_path / "runs"), "--output", str(collected)]) == 0
    assert len(json.loads((collected / "rows.json").read_text())) == 8


def test_adapter_only_recovery_toward_the_clean_source_on_a_reused_lora_source(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    base = run_with_base(parse_lifecycle(base_doc(digest)), weights, None, tmp_path / "base")
    base_run, base_result = base["hwa__std_model"]
    [artifact] = [a for a in base_result["artifacts"] if a["kind"] == "hwa_source"]
    base_path = base_run / artifact["path"]
    _, task_sha = tiny_task(tmp_path / "corpora")
    doc = lora_doc(base_doc(digest), task_sha, sha256_file(base_path))
    lora = run_with_base(parse_lifecycle(doc), weights, base_path, tmp_path / "lora")
    source_run, source_result = lora["hwa__lora"]
    [artifact] = [a for a in source_result["artifacts"] if a["kind"] == "hwa_source"]
    source = source_run / artifact["path"]
    rewrite = {"detection": "verify", "roles": "pair", "seed": 241016, "threshold_sigma": 5.0}
    doc["onchip"].update(
        objective="source_kl", update_law="cmo_thresholded_write_adam", threshold_sigma=0.5, accumulate_updates=True,
        max_write_attempts_per_device=4, epochs=2, learning_rate=0.01, calibration_learning_rate=0.001,
        epoch_evaluation="development", arms=[
            {"id": "none", "weights": "hold", "calibration": False},
            {"id": "calibration", "weights": "hold", "calibration": True},
            {"id": "rewrite", "weights": "hold", "calibration": False, "fault_rewrite": rewrite},
            {"id": "rewrite_adapters", "weights": "learn", "calibration": False, "fault_rewrite": rewrite,
             "write_scope": "adapter", "learning_rate": 0.05},
            {"id": "rewrite_all", "weights": "learn", "calibration": False, "fault_rewrite": rewrite}])
    lc = parse_lifecycle(doc)
    assert lc.onchip.arms[3].to_dict()["write_scope"] == "adapter"
    results = {}
    # The LoRA source is reused: only prepare, devices, deploy and on-chip run.
    for selection in stage_selections(lc):
        if selection.kind == "hwa":
            results[selection.stage_id] = lora[selection.stage_id]
            continue
        inputs = {}
        for role, upstream in stage_inputs(selection, lc).items():
            if upstream is None:
                inputs[role] = weights
            elif role == "weights":
                inputs[role] = source
            else:
                run_dir, result = results[upstream[0]]
                [a] = [a for a in result["artifacts"] if a["kind"] == upstream[1]]
                inputs[role] = run_dir / a["path"]
        output = tmp_path / "recovery" / selection.stage_id
        assert runtime.run_train(request_for(lc, selection, inputs, output)) == 0
        [run_dir] = sorted(output.iterdir())
        results[selection.stage_id] = (run_dir, dict(_validate_run_result(run_dir / "result.json")))
    run, _ = results["onchip__lora__271001"]
    rows = {(r["case"], r["onchip_arm"]): r for r in json.loads((run / "measurements.json").read_text())}
    for (case, arm), row in rows.items():
        assert row["replay_exact"]
        if arm == "rewrite_adapters":
            assert row["cost"]["base_device_writes"] == 0
            assert [set(epoch) >= {"task_test"} for epoch in row["curve"]] == [False, True]
        if arm == "rewrite_all" and case != "nominal":
            assert row["cost"]["touched_devices"] > 0
    assert all("adapter_device_writes" in rows[c, "rewrite_adapters"]["cost"] for c, _ in rows)
    # A healthy array starts at the clean source, so the source-KL objective starts at zero loss.
    assert rows["nominal", "rewrite"]["final"]["task_test"]["predictions"] == rows["nominal", "none"]["final"]["task_test"]["predictions"]
