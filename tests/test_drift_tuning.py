"""Crossbar fault-tuning learner under AIHWKit CMO log-time drift: law, reads, writes, lifecycles."""
import json
import math
from pathlib import Path

import pytest
import torch

import hardware.arrays as devices
import methods.drift_tuning as dt
from ebl.schema.lifecycle import parse_lifecycle, stage_selections
from hardware.config import _defects
from hardware.mapping import MatrixSpec
from networks.mapping import fault_domains

ROOT = Path(__file__).resolve().parents[1]
DRIFT = ROOT / "campaigns/pilots/20261116-drift-token-budget/lifecycles"
FAULT = ROOT / "campaigns/fault-recovery-paper/lifecycles/fault-token-budget-20261116/full/fault-token-budget-20261116-xb6-n8192.json"
FULL = DRIFT / "full/drift-token-budget-20261116-xb6-m4.json"


def law(**changes):
    raw = json.loads(FULL.read_text())["fault_tuning"]["drift"]
    raw.update(changes)
    return dt.DriftTuning.parse(raw)


def array(rate_ppm=0):
    layout = [MatrixSpec("decoder.layers.6.fc1", (64, 32), 0, 64 * 32, 1.0),
              MatrixSpec("decoder.layers.6.fc2", (32, 64), 64 * 32, 32 * 64, 1.0)]
    case = _defects({"placement": "hierarchical_die_array", "persistence": "permanent", "arrays_per_die": 48,
                     "cases": [{"kind": "none" if not rate_ppm else "random", "rate_ppm": rate_ppm,
                                "die_concentration": None, "array_concentration": None}]}, "cmo").cases[0]
    a = devices.fresh_array(technology="cmo", layout=layout, assignment_seed=271001, case=case, populations=None,
                            device="cpu", namespace="facebook/opt-125m", endpoint_seed=281001, noise_scale=1.0,
                            tile_size=256, arrays_per_die=48, acceptance_percent=2.0, domains=fault_domains(layout))
    target = torch.linspace(-0.8, 0.8, 2 * 64 * 32)
    a.program(target)
    return a


def test_law_marginal_and_release_identity():
    d = law()
    held = torch.full((300, 600), 50.0)
    assert torch.equal(dt.drifted(held, 0, 0.0, d), held)
    read = dt.drifted(held, 0, 86400.0, d).double()
    log_t = math.log(86400.0)
    assert float(read.mean()) == pytest.approx(50 + 4 * dt.AIHWKIT_MEAN_US * log_t, abs=0.02)
    assert float(read.std()) == pytest.approx(4 * (dt.AIHWKIT_SPREAD_SLOPE_US * log_t + dt.AIHWKIT_SPREAD_OFFSET_US),
                                              rel=0.01)


def test_draw_key_matches_the_drn_backend_convention():
    # Same text key and truncation as drn methods.drift_tuning.draw_key.
    import hashlib
    expected = int.from_bytes(hashlib.sha256(b"391103:2:900000").digest()[:8], "little") & ((1 << 63) - 1)
    assert dt.draw_key(391103, 2, 900.0) == expected


def test_drifted_weights_read_without_accounting_and_cancel_the_mean_in_pairs():
    d = law()
    a = array()
    held = a.read().clone()
    reads = a.cost()["verify_reads"]
    assert torch.equal(dt.drifted_weights(a, d, 0.0), held)
    q = dt.drifted_weights(a, d, 3600.0)
    assert torch.equal(q, dt.drifted_weights(a, d, 3600.0))
    assert a.cost()["verify_reads"] == reads and torch.equal(a.read(), held)
    # The conductance-independent mean shift cancels between active and reference devices.
    assert float((q - held).mean()) == pytest.approx(0.0, abs=0.01)
    spread = 4 * (dt.AIHWKIT_SPREAD_SLOPE_US * math.log(3600.0) + dt.AIHWKIT_SPREAD_OFFSET_US)
    assert float((q - held).std()) == pytest.approx(math.sqrt(2) * spread / dt.to_weight(1.0, 0.0) ** -1, rel=0.05)


def test_a_write_shifts_held_devices_and_keeps_every_draw():
    d = law()
    a, b = array(), array()
    delta = torch.zeros_like(a.read())
    delta[:100] = 0.05
    a.write(delta)
    changed = a.read() != b.read()
    assert bool(changed[:100].any()) and not bool(changed[100:].any())
    qa, qb = dt.drifted_weights(a, d, 7200.0), dt.drifted_weights(b, d, 7200.0)
    assert torch.equal(qa[~changed], qb[~changed])
    assert torch.allclose((qa - qb)[changed], (a.read() - b.read())[changed], atol=1e-5)


def test_drift_lifecycles_plan_one_stage_and_keep_the_fault_contract():
    fault = json.loads(FAULT.read_text())
    for path in sorted(DRIFT.glob("*/*.json")):
        raw = json.loads(path.read_text())
        lc = parse_lifecycle(raw)
        assert [s.kind for s in stage_selections(lc)] == ["prepare", "devices", "hwa", "fault_tuning"]
        assert parse_lifecycle(lc.to_dict()).digest() == lc.digest()
        f, g = dict(raw["fault_tuning"]), dict(fault["fault_tuning"])
        assert f.pop("drift")["multiplier"] == 4.0 and f["analog_lora"] is None
        for x in (f, g):
            for key in ("arms", "deployment", "schedule", "analog_lora"):
                x.pop(key)
        assert f == g
        assert raw["fault_tuning"]["schedule"]["steps"] in (32, 2400)
        assert {k: v for k, v in raw["fault_tuning"]["schedule"].items() if k != "steps"} == \
            {k: v for k, v in fault["fault_tuning"]["schedule"].items() if k != "steps"}
    old = parse_lifecycle(fault)
    assert old.fault_tuning.drift is None and "drift" not in old.fault_tuning.to_dict()


def test_invalid_drift_lifecycles_are_rejected():
    raw = json.loads(FULL.read_text())

    def broken(edit):
        doc = json.loads(json.dumps(raw))
        edit(doc)
        with pytest.raises(Exception):
            parse_lifecycle(doc)

    broken(lambda d: d["defects"].update(cases=[{"kind": "random", "rate_ppm": 30000, "die_concentration": None,
                                                 "array_concentration": None}]))
    broken(lambda d: d["fault_tuning"]["drift"].update(evaluation_seconds=[0, 61, 86400]))
    broken(lambda d: d["fault_tuning"]["drift"].update(mapping_scale=3.0))
    broken(lambda d: d["fault_tuning"]["drift"].update(write_clock="restarts"))
    broken(lambda d: d["fault_tuning"].update(arms=[{"label": "L1", "method": "onchip_noisy", "learning_rate": 1e-3}]))
