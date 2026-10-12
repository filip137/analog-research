"""The write-budget sweep preserves the source and rejects cap violations."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/"campaigns/opt-cmo-corruption/launch/exp6_writecap.py"
spec = importlib.util.spec_from_file_location("writecap_pilot", PATH)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_caps_change_only_declared_controller_budget():
    base = json.loads((PATH.parents[1]/"lifecycles/opt125m-cmo-l6-exp5-threshold-accum-s2-dev.json").read_text())
    for cap, label in zip(pilot.CAPS, pilot.labels()):
        doc = pilot.lifecycle(label).to_dict()
        assert doc['onchip'].pop('max_write_attempts_per_device') == cap
        for key in base:
            if key not in {'lifecycle_id', 'question'}:
                assert doc[key] == base[key]


def evidence():
    metrics = {split: {'teacher_kl': .1} for split in ['development', 'transfer_generated', 'transfer_wikitext']}
    rows = [{'onchip_arm': 'none', 'final': metrics},
            {'onchip_arm': 'onchip_weights', 'final': metrics, 'sequence_presentations': 8192,
             'cost': {'max_onchip_device_writes': 1, 'max_onchip_device_write_attempts': 1,
                      'issued_write_attempts': 5, 'device_writes': 4, 'stuck_write_attempts': 1,
                      'capped_devices': 5}}]
    records = [{'stage': 'onchip_memory', 'peak_reserved_bytes': 1000}]
    reference = {'sequence_presentations': 8192, 'initial_development_kl': .1, 'development_kl': .05}
    return rows, records, reference


def test_report_uses_gain_from_held_array_and_fails_closed_on_cap_violation():
    rows, records, reference = evidence()
    report = pilot.checked_report('cap1-dev', rows, records, 'collection.json', reference)
    assert report['retained_development_gain'] == 0
    rows[1]['cost']['max_onchip_device_write_attempts'] = 2
    with pytest.raises(ValueError, match='cap'):
        pilot.checked_report('cap1-dev', rows, records, 'collection.json', reference)


def test_report_rejects_mismatched_initial_condition():
    rows, records, reference = evidence()
    reference['initial_development_kl'] = .11
    with pytest.raises(ValueError, match='matched'):
        pilot.checked_report('cap1-dev', rows, records, 'collection.json', reference)
