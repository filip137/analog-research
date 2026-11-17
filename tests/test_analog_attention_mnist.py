"""MNIST chain A: model shapes, MOS count, SPICE netlist structure."""
import json
from pathlib import Path
import numpy as np
import torch
from networks.energy_transformer.mnist_chain_a import ChainA, inputs, mos_count
from networks.energy_transformer import mnist_spice as ms
from ebl.schema.analog_attention_mnist import parse_config

ROOT = Path(__file__).resolve().parents[1]


def test_model_and_configs():
    for p in sorted((ROOT / 'examples').glob('analog_attention_mnist_*.json')):
        if 'spice' not in p.name:
            parse_config(json.loads(p.read_text()))
    m = ChainA(2, 4, 8).double()
    x = inputs(torch.randint(0, 255, (3, 28, 28), dtype=torch.uint8)).double()
    assert x.shape == (3, 16, 65) and m(x).shape == (3, 10)


def test_spice_netlist_counts():
    m = ChainA(2, 2, 3).double()
    E = m.export()
    x = inputs(torch.zeros(1, 28, 28, dtype=torch.uint8)).double()[0].numpy()
    spec = dict(tokens=x, key_gp=E['key_gp_S'], key_gm=E['key_gm_S'], value_gp=E['value_gp_S'], value_gm=E['value_gm_S'],
                query_V=E['query_V'], t_integrate_heads=[1e-4, 2e-4], v_cm=1.05, scale=.5, s_ratio=16., tail_current=1e-9,
                amplitude=.57, v_low=.6, c_score=1e-12, c_hold=1e-12, t_write=2e-6, t_reset=2e-4, t_warm=2e-4, t_settle=.04,
                score_offset=[1e-10, 1e-10])
    net, cols, count = ms.netlist(spec)
    assert count['crossbar_resistors'] == 2 * 65 * (2 + 3) and count['hold_capacitors'] == 16 * 5 + 1
    assert 'Ioff_0_0' in net and 'Vq1_1' in net and sum(1 for c in cols if c[0].startswith('out_')) == 2 * 3
