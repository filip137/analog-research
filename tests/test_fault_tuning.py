"""Fault-recovery paper LR screen: config, Child-Tuning_D mask, analog-LoRA tiles and schedule."""
import copy
import json
import math
from pathlib import Path

import pytest
import torch

import methods.fault_tuning as ft
from ebl.runtime import fault_tuning as runtime
from ebl.schema.lifecycle import parse_lifecycle, stage_inputs, stage_selections
from hardware.mapping import MatrixSpec
from hardware.config import _defects

ROOT = Path(__file__).resolve().parents[1]
LIFECYCLES = ROOT / "campaigns/fault-recovery-paper/lifecycles"


def _doc(unit="deploy"):
    return json.loads((LIFECYCLES / f"opt125m-cmo-l6-fault-paper-tune-{unit}.json").read_text())


def test_lifecycle_parses_and_plans_one_stage():
    lc = parse_lifecycle(_doc())
    kinds = [s.kind for s in stage_selections(lc)]
    assert kinds == ["prepare", "devices", "hwa", "fault_tuning"]
    selection = stage_selections(lc)[-1]
    assert selection.array_seed == 271001
    assert stage_inputs(selection, lc)["weights"] == ("hwa__standard_hwa_model", "hwa_source")
    assert parse_lifecycle(lc.to_dict()).digest() == lc.digest()


@pytest.mark.parametrize("path,value", [
    (("schedule", "lr_schedule"), "constant"),
    (("schedule", "warmup_steps"), 5),
    (("targets", "kind"), "kl"),
    (("write_law", "max_write_attempts_per_device"), 4),
    (("fault_rewrite", "roles"), "active"),
])
def test_rejects_changed_scientific_settings(path, value):
    doc = _doc()
    doc["fault_tuning"][path[0]][path[1]] = value
    with pytest.raises(Exception):
        parse_lifecycle(doc)


def test_analog_arm_requires_pinned_b_scale():
    doc = _doc()
    doc["fault_tuning"]["arms"].append({"label": "a", "method": "analog_lora", "learning_rate": 1e-3, "rank": 16})
    with pytest.raises(Exception):
        parse_lifecycle(doc)
    doc["fault_tuning"]["analog_lora"]["b_scale"] = {"name": "x", "sha256": "0" * 64}
    assert parse_lifecycle(doc).fault_tuning.arms[-1].method == "analog_lora"


def test_two_cases_are_rejected():
    doc = _doc()
    doc["defects"]["cases"].append(dict(doc["defects"]["cases"][0], rate_ppm=10000))
    with pytest.raises(Exception):
        parse_lifecycle(doc)


def test_cosine_matches_torch_scheduler():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300)
    for step in range(300):
        assert math.isclose(opt.param_groups[0]["lr"], ft.cosine(3e-3, step, 300), rel_tol=1e-6, abs_tol=1e-15)
        opt.step()
        sched.step()
    assert ft.cosine(1.0, 0, 300) == 1.0 and ft.cosine(1.0, 300, 300) == 0.0


def test_child_mask_is_top_fraction_per_matrix():
    layout = [MatrixSpec("m0", (4, 25), 0, 100, 1.0), MatrixSpec("m1", (10, 10), 100, 100, 1.0)]
    fisher = torch.cat((torch.arange(100.), 1000 + torch.arange(100.).flip(0)))
    mask = ft.child_mask(fisher, layout, 0.1)
    assert int(mask[:100].sum()) == 10 and int(mask[100:].sum()) == 10
    assert bool(mask[90:100].all()) and bool(mask[100:110].all())


def test_sample_loss_is_mean_nll_of_samples():
    logits = torch.randn(2, 3, 5)
    drawn = torch.randint(0, 5, (2, 3, 4))
    lp = logits.log_softmax(-1)
    expected = -torch.stack([lp[b, t, drawn[b, t, s]] for b in range(2) for t in range(3) for s in range(4)]).mean()
    assert torch.allclose(ft.sample_loss(logits, drawn), expected)


def test_adapter_targets_match_digital_lora_init():
    base = [MatrixSpec("decoder.layers.6.fc1", (12, 8), 0, 96, 1.0), MatrixSpec("decoder.layers.6.fc2", (8, 12), 96, 96, 1.0)]
    target, a_scales = ft.adapter_targets(base, 4, 271001)
    layout = ft.adapter_layout(base, 4)
    assert [m.name for m in layout] == ["decoder.layers.6.fc1.lora_a", "decoder.layers.6.fc1.lora_b",
                                        "decoder.layers.6.fc2.lora_a", "decoder.layers.6.fc2.lora_b"]
    assert target.numel() == sum(m.size for m in layout)
    assert float(target.abs().max()) == 1.0
    # enable_lora's draws: one CPU generator, uniform +/- fan_in^-0.5, layout order
    g = torch.Generator().manual_seed(271001)
    for spec, (k, scale) in zip(base, enumerate(a_scales)):
        a = torch.empty(4, spec.shape[1]).uniform_(-spec.shape[1] ** -0.5, spec.shape[1] ** -0.5, generator=g)
        part = layout[2 * k]
        assert torch.allclose(target[part.offset:part.offset + part.size].view(part.shape) * scale[:, None], a)
        b = layout[2 * k + 1]
        assert not bool(target[b.offset:b.offset + b.size].any())


def test_adapter_gradient_is_chain_rule_to_normalized_values():
    class Net:
        pass
    base = [MatrixSpec("m.fc1", (6, 5), 0, 30, 1.0)]
    layout = ft.adapter_layout(base, 2)
    net = Net()
    net.lora = torch.nn.ModuleList([torch.nn.ParameterDict({"a": torch.nn.Parameter(torch.zeros(2, 5)),
                                                             "b": torch.nn.Parameter(torch.zeros(6, 2))})])
    a_s, b_s = [torch.rand(2) + 0.5], [torch.rand(6) + 0.5]
    q = torch.randn(sum(m.size for m in layout))
    x = torch.randn(3, 5)

    def loss_of(values):
        a = values[:10].view(2, 5) * a_s[0][:, None]
        b = values[10:].view(6, 2) * b_s[0][:, None]
        return ((x @ a.T) @ b.T).square().sum()
    runtime._set_adapter(net, q, layout, a_s, b_s)
    module = net.lora[0]
    ((x @ module["a"].T) @ module["b"].T).square().sum().backward()
    direct = q.clone().requires_grad_(True)
    loss_of(direct).backward()
    assert torch.allclose(runtime._adapter_gradient(net, a_s, b_s), direct.grad, atol=1e-5)


def test_adapter_faults_have_their_own_stream_and_rate():
    from hardware.arrays import hierarchical_faults
    from networks.mapping import fault_domains
    base = [MatrixSpec("decoder.layers.6.fc1", (512, 256), 0, 512 * 256, 1.0)]
    adapters = ft.adapter_layout(base, 16)
    case = _defects({"placement": "hierarchical_die_array", "persistence": "permanent", "arrays_per_die": 48,
                     "cases": [{"kind": "random", "rate_ppm": 30000, "die_concentration": None,
                                "array_concentration": None}]}, "cmo").cases[0]
    alone = hierarchical_faults(base, 256, 48, case, 271001, "facebook/opt-125m", domains=fault_domains(base))
    both = hierarchical_faults(base + adapters, 256, 48, case, 271001, "facebook/opt-125m",
                               domains=fault_domains(base + adapters))
    own = hierarchical_faults(adapters, 256, 48, case, 271001, "facebook/opt-125m", domains=fault_domains(adapters))
    for part in (0, 1):
        assert torch.equal(alone["matrices"][base[0].name][part], both["matrices"][base[0].name][part])
        for m in adapters:
            assert torch.equal(own["matrices"][m.name][part], both["matrices"][m.name][part])
    devices = 2 * sum(m.size for m in adapters)
    assert own["receipt"]["failed_devices"] == round(0.03 * devices)


def test_stored_sample_file_is_optional_and_named():
    doc = _doc()
    doc["fault_tuning"]["targets"]["file"] = "teacher-samples8-c919329d-df73ce66"
    assert parse_lifecycle(doc).fault_tuning.targets["file"] == "teacher-samples8-c919329d-df73ce66"
    doc["fault_tuning"]["targets"]["file"] = "../elsewhere"
    with pytest.raises(Exception):
        parse_lifecycle(doc)
    a1 = parse_lifecycle(_doc("a1")).fault_tuning
    j1 = parse_lifecycle(_doc("j1")).fault_tuning
    assert a1.arms == j1.arms and a1.targets["sha256"] == j1.targets["sha256"]


def test_l12_units_change_only_depth_source_and_grids():
    """exp-002 (12 MLPs): every other setting equals exp-001's stored-draw units."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_fpt", ROOT / "campaigns/fault-recovery-paper/make_lifecycles.py")
    make = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make)
    pins = {"l12": {"deployment": {"name": "d", "sha256": "1" * 64}, "b_scale": {"name": "b", "sha256": "2" * 64}}}
    six = _doc("a1")
    labels = []
    for unit in ("l12-s1", "l12-s2", "l12-s3"):
        doc = make.lifecycle(pins, unit)
        lc = parse_lifecycle(copy.deepcopy(doc))
        assert lc.network.analog_decoder_layers == 12
        ft12, ft6 = doc["fault_tuning"], six["fault_tuning"]
        assert ft12["source_sha256"] == make.EXP2_SOURCE and ft12["targets"] == ft6["targets"]
        for key in set(ft6) - {"source_sha256", "deployment", "analog_lora", "arms"}:
            assert ft12[key] == ft6[key], key
        for key in set(six) - {"lifecycle_id", "question", "network", "fault_tuning"}:
            assert doc[key] == six[key], key
        assert {k: v for k, v in doc["network"].items() if k != "analog_decoder_layers"} == \
            {k: v for k, v in six["network"].items() if k != "analog_decoder_layers"}
        labels += [a["label"] for a in ft12["arms"]]
    rates = {}
    for label in labels:
        arm, _, tag = label.rpartition("-")
        rates.setdefault(arm, []).append(tag)
    grid = ["1em4", "3em4", "1em3", "3em3", "1em2"]
    assert rates.pop("hold") == ["naive"] and rates.pop("hold-fault") == ["aware"]
    assert rates == {"noisy": grid, "noiseless": grid, "childd": grid, "dlora-r16": grid, "dlora-r64": grid,
                     "alora-r16": grid + ["3em2"]}
    canary = make.lifecycle(pins, "l12-canary")["fault_tuning"]
    assert [a["label"] for a in canary["arms"]] == labels and canary["schedule"]["steps"] == 4
