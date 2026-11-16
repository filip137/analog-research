"""Anti-bump scorer wiring, law and ideal-SPICE tests."""
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from networks.energy_transformer import antibump_scorer as ab
from networks.energy_transformer.pseudoconductance import export, simulate, IREF
from ebl.schema.antibump_scorer_attention import parse_config

ROOT = Path(__file__).resolve().parents[1]
V = np.array([[-.6, .4], [.2, -.5], [.7, .1]])
K = np.array([[.85, .90], [1.20, .95], [1.00, 1.25]])
EXE = shutil.which('ngspice') or '/tmp/pseudoconductance-spice-env/bin/ngspice'


def spec(variant, q=K[0]):
    return {'kind': 'antibump', 'variant': variant, 'query': q, 'keys': K, 's_ratio': 16, 'tail_current': 1e-9,
            'resistance': 1.2e8, 'amplitude': .57, 'iref': IREF, 'v_low': .6, 'drives_gate': variant == 'star'}


def test_configs_and_netlists():
    for name in ['star_ideal', 'star_bsim3', 'softmax_ideal', 'softmax_bsim3']:
        parse_config(json.loads((ROOT / f'examples/antibump_{name}.json').read_text()))
    star, _, cs = export(V, None, scorer=spec('star'))
    soft, _, cx = export(V, None, scorer=spec('softmax'))
    assert 'Mbias_a0' not in star and 'Mbias_a0' in soft          # star drives the coupler gates
    assert sum(cs.values()) == 92 and sum(cx.values()) == 93
    assert 'Eva va_buf 0 va 0 1' in star and 'Vlow vlow 0' in soft
    with pytest.raises(ValueError):
        export(V, None, scorer=spec('star') | {'drives_gate': False})


def test_law_limits():
    f = ab.law_currents(K[0], K, 16, .7)
    assert abs(f[0, 0] - 0.2) < 1e-12 and np.all(f < 1) and np.all(f >= 0.2 - 1e-12)
    a = ab.law_attention(K[0], K, spec('softmax'), .7)
    assert abs(a.sum() - .57) < 1e-12 and np.argmax(a) == 0


@pytest.mark.parametrize('variant', ['star', 'softmax'])
def test_ideal_spice_matches_law_and_selects(tmp_path, variant):
    if not Path(EXE).exists():
        pytest.skip('ngspice unavailable')
    r = simulate(V, None, tmp_path / variant, EXE, model='ideal', scorer=spec(variant, K[2]))
    isum = ab.measured_scores(r['port'])
    assert np.allclose(isum, ab.law_currents(K[2], K, 16, 2 / 3).sum(1) * 1e-9, rtol=1e-3)
    assert np.argmin(isum) == 2 and np.max(abs(r['y'] - V[2])) < .1


def test_ideal_sum_is_additive(tmp_path):
    if not Path(EXE).exists():
        pytest.skip('ngspice unavailable')
    d = np.linspace(-.2, .2, 8)
    total = ab.run_sum_test(tmp_path / 'all', EXE, deltas=d, active=range(8), model='ideal', s_ratio=16, tail_current=1e-9)[0]
    law = ab.law_currents(np.zeros(8) + 1.05, (1.05 - d)[None, :], 16, 2 / 3).sum() * 1e-9
    assert abs(total - law) / law < 1e-3
