"""The threshold pilot's full-pass gate uses complete development evidence only."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/"campaigns/opt-cmo-corruption/launch/exp5_threshold.py"
spec = importlib.util.spec_from_file_location("exp5_threshold_test", PATH)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def reports():
    return [{"label":tag+"-dev", "sequence_presentations":8192,
             "development_kl":.11, "initial_development_kl":.1,
             "cost":{"device_writes":100}, "threshold_sigma":sigma,
             "accumulate_updates":acc, "full_seconds_with_margin":2000,
             "collection":"native/"+tag}
            for tag,(acc,sigma) in pilot.SETTINGS.items()]


def test_no_full_fit_for_negative_grid_and_no_incomplete_selection():
    values = reports()
    assert not pilot.select(values)["triggered"]
    with pytest.raises(ValueError, match="incomplete"):
        pilot.select(values[:-1])


def test_full_gate_uses_final_dev_then_physical_cost_and_ignores_test():
    values = reports()
    values[1]["development_kl"] = values[2]["development_kl"] = .09
    values[2]["cost"]["device_writes"] = 50
    for index,row in enumerate(values):
        row["test_kl"] = index
    decision = pilot.select(values)
    assert decision["triggered"] and decision["label"] == "accum-s2-50m"
    assert decision["learning_rate"] == .003


def test_all_candidates_preserve_upstream_and_final_selection_protocol():
    from workflow.lifecycle import parse_lifecycle
    import json
    base = parse_lifecycle(json.loads((PATH.parents[1]/"lifecycles/opt125m-cmo-l6-exp2-dense-50m.json").read_text()))
    for label in pilot.labels():
        lc = pilot.lifecycle(label)
        for key in ("network", "data", "devices", "defects", "deployment", "hwa", "program_verify", "transfer_tests"):
            assert lc.to_dict().get(key) == base.to_dict().get(key)
        assert lc.onchip.learning_rate == .003 and lc.onchip.epochs == 1
        assert lc.onchip.cohort_key == ("training" if label.endswith("-dev") else "hwa")
