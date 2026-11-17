"""Attention layer: configs parse, netlist structure, block laws."""
import json
from pathlib import Path
import numpy as np
from networks.energy_transformer import attention_layer as al
from ebl.schema.attention_layer import parse_config
from ebl.runtime.attention_layer import spec_for, examples

ROOT = Path(__file__).resolve().parents[1]


def test_configs_and_netlist():
    for m in ('bsim3', 'ideal'):
        c = parse_config(json.loads((ROOT / f'examples/attention_layer_{m}.json').read_text()))
        net, cols, count = al.netlist(spec_for(c, examples(c.seed, 1)[0]), m)
        assert count['crossbar_resistors'] == 32 and count['drn_resistors'] == 6 * 8 + 8 + 6 * 8 + 6
        assert 'Eu0 du0' in net and 'Bdi_0' in net and 'Ieps' in net and f'Vrect vrect 0 {c.v_rect:.17g}' in net


def test_block_laws():
    n = al.law_norm([0.3, -0.1], 0.05)
    assert np.allclose(n, [0.3 / 0.45, -0.1 / 0.45])
    u = al.drn_inputs(n, 0.3)
    assert u.shape == (6,) and np.isclose(u[4], 0.3) and np.isclose(u[2], -u[0])
    g1, g2 = np.full((6, 4), 2e-5), np.full((4, 3), 3e-5)
    h, o = al.law_drn(u, g1, g2, 9e-6)
    assert np.all(h >= 0) and o.shape == (3,)
