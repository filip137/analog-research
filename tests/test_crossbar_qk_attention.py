"""Crossbar wiring, formula and end-to-end ideal-SPICE retrieval tests."""
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from networks.energy_transformer import crossbar_qk as xbar, bump_scorer as bump
from networks.energy_transformer.pseudoconductance import export, simulate, reference, IREF
from ebl.schema.crossbar_qk_attention import parse_config
from ebl.schema.base import ConfigError

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'examples/crossbar_qk_ideal.json').read_text())
GP, GM = np.array(CFG['conductance_pos']), np.array(CFG['conductance_neg'])
CTX = np.array([[.3, 1., 0.], [-.2, 0., 1.], [.1, -1., 0.]])
XQ = np.array([0., 0., 1.])
V = np.array([[-.6, .4], [.2, -.5], [.7, .1]])


def spec(xq=XQ):
    return {'tail_current': 8e-10, 'crossbar': {'q': xq, 'k': CTX, 'gp': GP, 'gm': GM, 'v_cm': 1.05, 'scale': .75}}


def test_config_and_netlist_contract():
    parse_config(dict(CFG))
    bad = dict(CFG); bad['weights_source_sha256'] = 'x'
    with pytest.raises(ConfigError):
        parse_config(bad)
    text, columns, counts = export(V, None, scorer=spec())
    assert sum(counts.values()) == 97 and text.count('\nRx') == 48
    assert 'Vq0 ' not in text and 'Vk00 ' not in text          # gates driven by the crossbar
    assert 'Rxqp00 xqp0 q0' in text and 'Rxk2m12 xk2m2 k21' in text


def test_formula_is_weighted_mean():
    q = xbar.formula(XQ, GP, GM, 1.05, .75)
    inputs = np.concatenate([1.05 + .75 * XQ, 1.05 - .75 * XQ])
    for d in range(2):
        g = np.concatenate([GP[d], GM[d]])
        assert abs(q[d] - g @ inputs / g.sum()) < 1e-12


def test_ideal_spice_rows_and_selection(tmp_path):
    exe = shutil.which('ngspice') or '/tmp/pseudoconductance-spice-env/bin/ngspice'
    if not Path(exe).exists():
        pytest.skip('ngspice unavailable')
    r = simulate(V, None, tmp_path / 'r', exe, model='ideal', scorer=spec())
    q, k = xbar.rows(r['port'])
    assert np.max(abs(q - xbar.formula(XQ, GP, GM, 1.05, .75))) < 1e-6
    assert np.max(abs(k - xbar.formula(CTX, GP, GM, 1.05, .75))) < 1e-6
    a = bump.measured(r['port'], IREF)
    assert np.argmax(a) == 1                                    # query equals key 1
    assert np.max(abs(r['y'] - reference(V, a))) < 1e-3
