"""Attention size ladder: configs parse, geometry and mismatch reach the netlist, v1/v2 netlists unchanged."""
import json
from pathlib import Path
import numpy as np
import pytest
from networks.energy_transformer import end_to_end as e2e
from ebl.schema.attention_size import parse_config
from ebl.runtime.attention_size import spec_for
from ebl.runtime.crossbar_qk_attention import toy_examples

ROOT = Path(__file__).resolve().parents[1]


def test_configs_and_netlists():
    paths = sorted((ROOT / 'examples').glob('attention_size_*.json'))
    assert len(paths) == 8
    for p in paths:
        c = parse_config(json.loads(p.read_text()))
        spec = spec_for(c, toy_examples(np.random.default_rng(0))[0])
        net = e2e.netlist(spec, c.model)[0]
        inv = e2e.inventory(net)
        assert inv['gate_area_total_um2'] > 0 and inv['capacitance_F'] > 0
        mm = e2e.netlist(spec | {'mismatch': {'avt': 7e-3, 'abeta': .01, 'seed': 1}, 'ktc_seed': 2}, c.model)[0]
        assert 'delvto=' in mm and 'Vktc_k00' in mm and 'k00_g' in mm and 'delvto=' not in net


def test_cell_geometry_shrinks_area():
    z1 = parse_config(json.loads((ROOT / 'examples/attention_size_B_Z1_bsim3.json').read_text()))
    z2 = parse_config(json.loads((ROOT / 'examples/attention_size_B_Z2_bsim3.json').read_text()))
    ex = toy_examples(np.random.default_rng(0))[0]
    a1 = e2e.inventory(e2e.netlist(spec_for(z1, ex), 'bsim3')[0])['gate_area_um2']
    a2 = e2e.inventory(e2e.netlist(spec_for(z2, ex), 'bsim3')[0])['gate_area_um2']
    assert a2['cell_mid'] < a1['cell_mid'] / 100 and a2['translinear'] == a1['translinear']


def test_rejects_unknown_family():
    raw = json.loads((ROOT / 'examples/attention_size_B_Z2_bsim3.json').read_text())
    with pytest.raises(Exception):
        parse_config(raw | {'geometry': {'nonsense': [1, 1]}})
