"""Wiring, law and default-netlist tests for the bump-chain scorer."""
import shutil
from pathlib import Path
import numpy as np
import pytest
from networks.energy_transformer import bump_scorer as bump
from networks.energy_transformer.pseudoconductance import export, simulate, reference, IREF
from ebl.schema.bump_scorer_attention import parse_config
from ebl.schema.base import ConfigError

V = np.array([[-.6, .4], [.2, -.5], [.7, .1]])
K = np.array([[.85, .90], [1.20, .95], [1.00, 1.25]])


def test_scorer_replaces_attention_sources_only():
    with pytest.raises(ConfigError):
        parse_config({'tail_current': 1e-6})
    text, columns, counts = export(V, None, scorer=dict(query=K[0], keys=K, tail_current=8e-10))
    assert counts['scorer'] == 46 and sum(counts.values()) == 97
    assert 'Iref_a0' not in text and 'Iref_h vdd g_h' in text
    assert 'Vsc_2 so2 g_a2 0' in text
    default, _, base = export(V, np.array([.001, .001, 5.]))
    assert sum(base.values()) == 51 and 'scorer' not in base and 'Vsc_' not in default
    assert 'abstol=1e-20' in default
    tr, _, _ = export(V, np.array([.2, .6, 1.5]), transient=True, transient_abstol=1e-18)
    dc, _, _ = export(V, np.array([.2, .6, 1.5]), transient_abstol=1e-18)
    assert 'abstol=1e-18' in tr and 'abstol=1e-20' in dc


def test_law_gradient_matches_finite_difference():
    q = K[0] + np.array([.04, -.025])
    dq, dk = bump.law_gradients(q, K, 8e-10, IREF, .7)
    for d in range(2):
        s = np.zeros(2); s[d] = 1e-7
        fd = (bump.law(q + s, K, 8e-10, IREF, .7) - bump.law(q - s, K, 8e-10, IREF, .7)) / 2e-7
        assert np.allclose(fd, dq[:, d], rtol=1e-5, atol=1e-12)
    assert np.allclose(dk, -dq)


def test_ideal_spice_scorer_follows_law_and_selects(tmp_path):
    exe = shutil.which('ngspice') or '/tmp/pseudoconductance-spice-env/bin/ngspice'
    if not Path(exe).exists():
        pytest.skip('ngspice unavailable')
    for sel in (0, 2):
        r = simulate(V, None, tmp_path / str(sel), exe, model='ideal',
                     scorer=dict(query=K[sel], keys=K, tail_current=8e-10))
        a = bump.measured(r['port'], IREF)
        assert np.allclose(a, bump.law(K[sel], K, 8e-10, IREF, 2 / 3), rtol=1e-3, atol=5e-5)  # ~5 fA numerical floor at starved branches
        assert np.max(abs(r['y'] - reference(V, a))) < 1e-3
        assert np.max(abs(r['y'] - V[sel])) < .03
