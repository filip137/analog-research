"""CMO fault-aware pair rewrite: verify reads, partner programming, the controller and the arm contract.

The verify arm decides from controller-visible reads only. Arrays that read
identically but differ in their hidden fault maps receive identical commands,
and the hidden map is read only after programming, for accounting.
"""
import json
from pathlib import Path

import pytest
import torch

from experiments.cifar_crossbar.runtime import load
from experiments.schema import ConfigError
from training import reram_cmo as cmo
from training.reram_cmo import ACTIVE, REFERENCE, CmoMatrix
from workflow import deployment, program_verify
from workflow import devices as device_stage
from workflow.lifecycle import parse_lifecycle
from test_workflow_cmo import case_doc, cmo_inputs, mlp_layout, scenario, small, template, NAMESPACE
from test_workflow_opt import tiny_teacher
from workflow_helpers import run_lifecycle

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PAIR = {"detection": "oracle", "roles": "pair", "seed": 241016}
VERIFY_PAIR = {"detection": "verify", "roles": "pair", "seed": 241016, "threshold_sigma": 5.0}


def faulty(seed=7):
    """Index 0 has a stuck active device, index 1 a stuck reference."""

    mask = torch.tensor([[True, False, False, False], [False, True, False, False]])
    x = CmoMatrix(4, seed, 2., 1., "cpu", (mask, torch.full((2, 4), 30.)))
    x.program(torch.zeros(4))
    return x


def arms(*extra):
    return [{"id": "none", "weights": "hold", "calibration": False}, *extra]


# --- device primitives ----------------------------------------------------------------


def test_verify_read_returns_held_conductances_and_counts_two_reads_per_weight():
    x = faulty()
    state, stream = x.g.clone(), x.generator.get_state()
    reads = x.read_devices()
    assert torch.equal(reads, state) and reads.data_ptr() != x.g.data_ptr()
    assert x.cost()["verify_reads"] == 8 and torch.equal(x.generator.get_state(), stream)


def test_program_devices_draws_one_endpoint_from_its_own_stream_and_keeps_faults():
    x = faulty()
    before, stream = x.g.clone(), x.generator.get_state()
    targets = torch.tensor([[20., 40., 60., 0.], [0., 0., 70., 0.]])
    selected = torch.tensor([[True, True, False, False], [False, False, True, False]])
    generator = torch.Generator().manual_seed(11)
    noise = torch.randn((2, 4), generator=torch.Generator().manual_seed(11))
    x.program_devices(targets, selected, generator)
    expected = targets + cmo.programming_sigma(targets, 2.) * noise
    assert x.g[ACTIVE, 0] == before[ACTIVE, 0] == 30.  # stuck active keeps its value
    assert x.g[ACTIVE, 1] == expected[ACTIVE, 1] and x.g[REFERENCE, 2] == expected[REFERENCE, 2]
    unselected = ~selected
    assert torch.equal(x.g[unselected], before[unselected])
    assert torch.equal(x.generator.get_state(), stream)
    cost = x.cost()
    assert (cost["device_programs"], cost["stuck_write_attempts"], cost["array_program_calls"]) == (8 - 2 + 2, 2 + 1, 2)
    assert x.writes.tolist() == [0, 2, 1, 1]
    x.verify_permanence()


def test_a_devices_draw_does_not_depend_on_which_other_devices_are_selected():
    one, two = faulty(), faulty()
    targets = torch.full((2, 4), 50.)
    one.program_devices(targets, torch.tensor([[False, False, True, False], [False] * 4]), torch.Generator().manual_seed(3))
    two.program_devices(targets, torch.tensor([[False, True, True, True], [False, False, True, False]]),
                        torch.Generator().manual_seed(3))
    assert one.g[ACTIVE, 2] == two.g[ACTIVE, 2]


def test_states_record_verify_reads_only_once_taken_and_round_trip():
    x = faulty()
    assert len(x.state_dict()["counters"]) == 4
    x.read_devices()
    state = x.state_dict()
    assert state["counters"][4] == 8
    y = faulty(); y.load_state_dict(state)
    assert y.cost() == x.cost() and torch.equal(y.g, x.g)
    old = faulty(); old.load_state_dict(faulty().state_dict())
    assert old.cost()["verify_reads"] == 0
    with pytest.raises(ValueError):
        y.load_state_dict({**state, "counters": [1, 2, 3]})


# --- controller -----------------------------------------------------------------------


def test_detection_flags_devices_beyond_the_declared_programming_sigmas():
    target = torch.tensor([0., .5, -.5])
    expected = torch.stack((cmo.to_conductance(target), torch.full((3,), cmo.MID_US)))
    sigma = cmo.programming_sigma(expected, 2.)
    reads = expected + torch.tensor([[4.9, 5.1, 0.], [0., -5.1, 4.]]) * sigma
    flags = program_verify.detect_stuck_devices(reads, target, acceptance=2., noise_scale=1., threshold_sigma=5.)
    assert flags.tolist() == [[False, True, False], [False, True, False]]


def test_pair_plan_cancels_each_stuck_device_through_its_partner():
    target = torch.tensor([.1, -.2, .3, .0, .5])
    flags = torch.tensor([[True, False, False, True, True], [False, True, False, True, False]])
    observed = torch.tensor([[70., 0., 0., 20., 9.], [0., 30., 0., 20., 0.]])
    selected, targets, clipped = program_verify.pair_rewrite_plan(flags, observed, target, "pair")
    half = cmo.HALF_RANGE_US
    assert selected.tolist() == [[False, True, False, False, False], [True, False, False, False, True]]
    assert targets[REFERENCE, 0] == pytest.approx(70. - .1 * half)        # r = a_s - 39.6 q
    assert targets[ACTIVE, 1] == pytest.approx(30. - .2 * half)           # a = r_s + 39.6 q
    assert targets[REFERENCE, 4] == cmo.G_MIN_US and clipped == 1          # 9 - 19.8 is clipped
    # Both devices stuck: left as read. Active roles never reprogram a reference.
    active, _, _ = program_verify.pair_rewrite_plan(flags, observed, target, "active")
    assert active.tolist() == [[False, True, False, False, False], [False] * 5]


def tiny_arrays(lc, case, noise_scale=1.0):
    network = deployment.build_network(tiny_teacher(), lc)
    target = network.q.detach()
    faults = None if case.kind == "none" else device_stage.hierarchical_faults(
        network.layout, lc.deployment.tile_size, lc.defects.arrays_per_die, case, 271001, lc.network.binding_namespace)
    array = device_stage.CmoArray(network.layout, lc.network.binding_namespace, lc.devices.endpoint_seed(271001),
                                  2.0, noise_scale, "cpu", faults)
    array.program(target)
    return array, target


def test_exact_pair_rewrite_restores_every_weight_with_one_healthy_device():
    lc = small()
    array, target = tiny_arrays(lc, scenario("random", 50000), noise_scale=0.0)
    mask, _ = device_stage.hidden_faults(array)
    receipt = program_verify.fault_rewrite(array, target, lc, parse_arm(ORACLE_PAIR).fault_rewrite, 271001)
    single = mask.any(0) & ~mask.all(0)
    error = (array.read() - target).abs()
    assert receipt["controller"]["programmed_active"] + receipt["controller"]["programmed_reference"] == int(single.sum())
    unclipped = error[single] < 1e-5
    assert int((~unclipped).sum()) == receipt["controller"]["clipped_targets"]
    assert bool((error[~mask.any(0)] < 1e-6).all())
    assert receipt["analysis"]["false_positive_devices"] == receipt["analysis"]["false_negative_devices"] == 0
    assert receipt["cost"]["verify_reads"] == 0 and receipt["fault_map"].startswith("hidden oracle")


def test_active_roles_leave_stuck_active_devices_uncorrected():
    lc = small()
    array, target = tiny_arrays(lc, scenario("random", 50000), noise_scale=0.0)
    mask, _ = device_stage.hidden_faults(array)
    rewrite = parse_arm({**ORACLE_PAIR, "roles": "active"}).fault_rewrite
    receipt = program_verify.fault_rewrite(array, target, lc, rewrite, 271001)
    assert receipt["controller"]["programmed_reference"] == 0
    assert receipt["controller"]["programmed_active"] == int((mask[REFERENCE] & ~mask[ACTIVE]).sum())


def test_verify_decisions_depend_only_on_reads_and_precede_any_fault_map_access(monkeypatch):
    lc = small()
    faulted, target = tiny_arrays(lc, scenario("random", 50000))
    healthy, _ = tiny_arrays(lc, scenario("none", 0))
    # The healthy array is made to read exactly like the faulted one.
    for a, b in zip(healthy.children, faulted.children):
        a.g = b.g.clone()
    events = []
    hidden, plant = device_stage.hidden_faults, device_stage.CmoArray.program_devices
    monkeypatch.setattr(device_stage, "hidden_faults", lambda array: events.append("hidden") or hidden(array))
    monkeypatch.setattr(device_stage.CmoArray, "program_devices",
                        lambda self, *a: events.append("program") or plant(self, *a))
    rewrite = parse_arm(VERIFY_PAIR).fault_rewrite
    receipts = [program_verify.fault_rewrite(array, target, lc, rewrite, 271001) for array in (faulted, healthy)]
    assert events == ["program", "hidden"] * 2
    assert receipts[0]["controller"] == receipts[1]["controller"] and receipts[0]["controller"]["programmed_active"] > 0
    assert receipts[0]["analysis"] != receipts[1]["analysis"]
    # The plant differs: the faulted array's stuck devices keep their value, the healthy one's do not.
    faulted.verify_permanence()
    assert not torch.equal(faulted.read(), healthy.read())
    assert receipts[0]["cost"]["verify_reads"] == 2 * target.numel()


def test_verify_finds_every_stuck_device_away_from_its_target():
    lc = small()
    array, target = tiny_arrays(lc, scenario("random", 50000))
    mask, stuck = device_stage.hidden_faults(array)
    receipt = program_verify.fault_rewrite(array, target, lc, parse_arm(VERIFY_PAIR).fault_rewrite, 271001)
    expected = torch.stack((cmo.to_conductance(target), torch.full_like(target, cmo.MID_US)))
    near = mask & ((stuck - expected).abs() <= 5 * cmo.programming_sigma(expected, 2.))
    assert receipt["analysis"]["false_negative_devices"] == int(near.sum())
    assert receipt["analysis"]["false_positive_devices"] == 0


def test_stuck_kinds_share_fault_positions_and_healthy_programming():
    """exp-016: gmin/gmax/open deployments differ from random ones only in stuck values."""

    lc = small()
    arrays = {kind: tiny_arrays(lc, scenario(kind, 30000))[0] for kind in ("random", "gmin", "gmax", "open")}
    other, _ = tiny_arrays(lc, scenario("random", 50000))
    masks = {kind: device_stage.hidden_faults(a)[0] for kind, a in arrays.items()}
    g = {kind: torch.cat([c.g for c in a.children], 1) for kind, a in arrays.items()}
    healthy = ~masks["random"]
    for kind in ("gmin", "gmax", "open"):
        assert torch.equal(masks[kind], masks["random"])
        assert torch.equal(g[kind][healthy], g["random"][healthy])
    # Healthy devices of one assignment also share their endpoints across rates.
    both = healthy & ~device_stage.hidden_faults(other)[0]
    assert torch.equal(torch.cat([c.g for c in other.children], 1)[both], g["random"][both])


# --- schema ---------------------------------------------------------------------------


def parse_arm(rewrite, weights="hold"):
    doc = template()
    doc["onchip"]["arms"] = arms({"id": "fixed", "weights": weights, "calibration": False, "fault_rewrite": rewrite})
    return parse_lifecycle(doc).onchip.arms[1]


def test_schema_round_trips_fault_rewrites_and_omits_them_otherwise():
    doc = template()
    plain = parse_lifecycle(doc)
    assert all("fault_rewrite" not in arm for arm in plain.to_dict()["onchip"]["arms"])
    doc["onchip"]["arms"] = arms(
        {"id": "active_rewrite_oracle", "weights": "hold", "calibration": False, "fault_rewrite": {**ORACLE_PAIR, "roles": "active"}},
        {"id": "pair_rewrite_oracle", "weights": "hold", "calibration": False, "fault_rewrite": ORACLE_PAIR},
        {"id": "pair_rewrite_verify", "weights": "hold", "calibration": False, "fault_rewrite": VERIFY_PAIR},
        {"id": "pair_rewrite_verify_recovery", "weights": "learn", "calibration": False, "fault_rewrite": VERIFY_PAIR},
    )
    lc = parse_lifecycle(doc)
    assert parse_lifecycle(lc.to_dict()) == lc and lc.to_dict()["onchip"]["arms"] == doc["onchip"]["arms"]
    assert [arm.trains for arm in lc.onchip.arms] == [False, False, False, False, True]
    assert lc.onchip.arms[3].fault_rewrite.threshold_sigma == 5.0 and lc.onchip.arms[1].fault_rewrite.threshold_sigma is None


def test_fault_rewrites_are_allowed_under_a_thresholded_write_cap():
    doc = template()
    doc["onchip"].update(update_law="cmo_thresholded_write_adam", threshold_sigma=2., accumulate_updates=True,
                         max_write_attempts_per_device=4)
    doc["onchip"]["arms"] = arms({"id": "learn", "weights": "learn", "calibration": False},
                                 {"id": "rewrite_learn", "weights": "learn", "calibration": False, "fault_rewrite": VERIFY_PAIR})
    assert parse_lifecycle(doc).onchip.max_write_attempts_per_device == 4


@pytest.mark.parametrize("arm, message", [
    ({"weights": "rewrite", "calibration": True, "fault_rewrite": ORACLE_PAIR}, "rewrite faults"),
    ({"weights": "hold", "calibration": False, "fault_rewrite": {**ORACLE_PAIR, "threshold_sigma": 5.}}, "exactly"),
    ({"weights": "hold", "calibration": False, "fault_rewrite": {k: v for k, v in VERIFY_PAIR.items() if k != "threshold_sigma"}}, "exactly"),
    ({"weights": "hold", "calibration": False, "fault_rewrite": {**ORACLE_PAIR, "roles": "reference"}}, "roles"),
    ({"weights": "hold", "calibration": False, "fault_rewrite": {**ORACLE_PAIR, "detection": "read"}}, "detection"),
    ({"weights": "hold", "calibration": False, "fault_rewrite": {**ORACLE_PAIR, "seed": 281001}}, "separate"),
    ({"weights": "hold", "calibration": False, "fault_rewrite": {**ORACLE_PAIR, "seed": 41}}, "separate"),
    ({"weights": "hold", "calibration": False, "fault_rewrite": {**VERIFY_PAIR, "threshold_sigma": 0}}, "threshold"),
])
def test_schema_rejects_invalid_fault_rewrites(arm, message):
    doc = template()
    doc["onchip"]["arms"] = arms({"id": "fixed", **arm})
    with pytest.raises(ConfigError, match=message):
        parse_lifecycle(doc)


def test_fault_rewrites_need_a_plain_hold_arm_cmo_and_no_write_noise_oracle():
    doc = template()
    doc["onchip"]["arms"] = [{"id": "fixed", "weights": "hold", "calibration": False, "fault_rewrite": ORACLE_PAIR}]
    with pytest.raises(ConfigError, match="hold arm"):
        parse_lifecycle(doc)
    doc = template()
    doc["onchip"]["arms"] = arms({"id": "fixed", "weights": "hold", "calibration": False, "fault_rewrite": ORACLE_PAIR})
    doc["onchip"]["write_noise_scale"] = 0
    with pytest.raises(ConfigError, match="fault_rewrite"):
        parse_lifecycle(doc)
    om = json.loads((ROOT / "examples/lifecycles/opt125m-om-mlp4-template.json").read_text())
    om["onchip"]["arms"] = arms({"id": "fixed", "weights": "hold", "calibration": False, "fault_rewrite": ORACLE_PAIR})
    with pytest.raises(ConfigError, match="CMO"):
        parse_lifecycle(om)


# --- the lifecycle --------------------------------------------------------------------


def test_fault_rewrite_arms_run_replay_and_leave_learning_unchanged_on_healthy_arrays(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    lc = small(**{
        "data.corpus": {"name": "synthetic", "sha256": digest},
        "data.sequence_length": 9, "data.max_examples": 8,
        "defects.cases": [case_doc(), case_doc("random", 50000)], "hwa.arms": [{"id": "digital", "method": "none"}],
        "onchip.batch_size": 4, "onchip.epochs": 1, "onchip.learning_rate": .01,
        "onchip.update_law": "cmo_thresholded_write_adam", "onchip.threshold_sigma": 1.,
        "onchip.accumulate_updates": True, "onchip.max_write_attempts_per_device": 4,
        "onchip.arms": arms(
            {"id": "active_rewrite_oracle", "weights": "hold", "calibration": False,
             "fault_rewrite": {**ORACLE_PAIR, "roles": "active"}},
            {"id": "pair_rewrite_oracle", "weights": "hold", "calibration": False, "fault_rewrite": ORACLE_PAIR},
            {"id": "pair_rewrite_verify", "weights": "hold", "calibration": False, "fault_rewrite": VERIFY_PAIR},
            {"id": "pair_rewrite_verify_recovery", "weights": "learn", "calibration": False, "fault_rewrite": VERIFY_PAIR},
            {"id": "onchip_weights", "weights": "learn", "calibration": False}),
    })
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    run, result = results["onchip__digital__271001"]
    assert result["metrics"]["exact_replays"] == 12
    rows = {(r["case"], r["onchip_arm"]): r for r in json.loads((run / "measurements.json").read_text())}
    analog = 2 * 2 * 16 * 32
    for (case, arm), row in rows.items():
        assert row["initial_apparent_sha256"] == rows[case, "none"]["initial_apparent_sha256"] and row["replay_exact"]
        rewrite = row.get("fault_rewrite")
        assert (rewrite is not None) == (arm.startswith(("active_", "pair_")))
        if rewrite and not arm.endswith("recovery"):
            assert row["cost"] == {} and row["final_apparent_sha256"] == rewrite["rewritten_apparent_sha256"]
        if rewrite and rewrite["detection"] == "verify":
            assert rewrite["cost"]["verify_reads"] == 2 * analog
        if arm.endswith("recovery"):
            cost = row["cost"]
            assert cost["max_onchip_device_write_attempts"] <= 4 and cost["verify_reads"] == 0
            assert cost["issued_write_attempts"] == cost["device_writes"] + cost["stuck_write_attempts"]
            assert rewrite["rewritten_apparent_sha256"] == rows[case, "pair_rewrite_verify"]["final_apparent_sha256"]
    # A healthy array: nothing is flagged or programmed, and learning after the rewrite is bitwise plain learning.
    for arm in ("pair_rewrite_oracle", "pair_rewrite_verify"):
        row = rows["nominal", arm]
        assert row["fault_rewrite"]["cost"]["device_programs"] == 0 and row["final"] == row["initial"]
    plain, after = rows["nominal", "onchip_weights"], rows["nominal", "pair_rewrite_verify_recovery"]
    assert after["final_apparent_sha256"] == plain["final_apparent_sha256"] and after["final"] == plain["final"]
    assert after["cost"] == plain["cost"]
    faulted = [label for label in {c for c, _ in rows} if label != "nominal"][0]
    pair, active = rows[faulted, "pair_rewrite_oracle"]["fault_rewrite"], rows[faulted, "active_rewrite_oracle"]["fault_rewrite"]
    assert pair["controller"]["programmed_reference"] > 0 and active["controller"]["programmed_reference"] == 0
    assert pair["controller"]["programmed_active"] == active["controller"]["programmed_active"] > 0
    state = load(run / "checkpoints/onchip" / faulted / "pair_rewrite_verify.pt")
    assert state["array"]["kind"] == "layered_cmo"
