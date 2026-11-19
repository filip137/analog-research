"""exp-002 configs are current and differ from their templates only in the declared fields."""
import importlib.util
import json
from pathlib import Path

from ebl.schema.opt_cmo_drn import parse_config

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / 'examples/opt_cmo_drn'
PILOT = ROOT / 'campaigns/pilots/20261119-nohwa-drift'


def flat(x, p=''):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flat(v, f'{p}.{k}'))
        return out
    return {p: x}


def changed(a, b):
    a, b = flat(a), flat(b)
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def load_make():
    spec = importlib.util.spec_from_file_location('make_configs', PILOT / 'make_configs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configs_current_and_declared_changes_only():
    make = load_make()
    files = sorted((EX / 'nohwa_drift').rglob('*.json'))
    assert len(files) == 2 + 48 + 8
    for path in files:
        doc = json.loads(path.read_text())
        parse_config(doc)
        layers = doc['analog_layers']
        kind = path.parent.name
        if kind == 'prepare':
            assert json.loads(path.read_text()) == make.prepare(layers)
            base = json.loads((EX / ('prepare.json' if layers == 6 else 'l12/prepare.json')).read_text())
            assert changed(base, doc) == {'.source_sha256'} and doc['source_sha256'] == make.NOHWA[layers]
        elif kind == 'faults':
            base = json.loads((EX / 'mapping_faults' / path.name.replace('-nohwa', '', 1)).read_text())
            assert changed(base, doc) == {'.source_sha256'} and doc['source_sha256'] == make.NOHWA[layers]
        else:
            replay = path.name.endswith('-replay.json')
            base = json.loads((EX / 'cmo_drift_recovery' / ('canary' if replay else 'full') /
                               f"drn{layers}-ratio3-m4{'-canary' if replay else ''}.json").read_text())
            allowed = {'.fault_tuning.arms', '.drift_tuning.drift_off_arms', '.drift_tuning.mapping_scale',
                       '.source_sha256', '.training.max_wall_seconds'}
            assert changed(base, doc) <= allowed
            assert [a['label'] for a in doc['fault_tuning']['arms']] == ['A0', 'A0-twin', 'CAL']
            assert doc['fault_tuning']['arms'][2]['learning_rate'] == 1e-2
            assert doc['drift_tuning']['drift_off_arms'] == ['A0-twin']
            source = 'nohwa' if '-nohwa-' in path.name else 'hwa'
            assert doc['source_sha256'] == (make.NOHWA if source == 'nohwa' else make.HWA)[layers]
            assert doc['drift_tuning']['mapping_scale'] == (3.0 if 'ratio3' in path.name else 1.0)
            if replay:
                assert changed(base, doc) <= {'.fault_tuning.arms', '.drift_tuning.drift_off_arms'}
