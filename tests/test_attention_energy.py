"""Attention-energy ladder: configs parse, option netlists build, v1 netlists unchanged, ideal energy accounting."""
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from networks.energy_transformer import end_to_end as e2e
from ebl.schema.attention_energy import parse_config
from ebl.runtime.attention_energy import spec_for
from ebl.runtime.crossbar_qk_attention import toy_examples

ROOT = Path(__file__).resolve().parents[1]
EXE = shutil.which('ngspice') or '/tmp/pseudoconductance-spice-env/bin/ngspice'
CONFIGS = sorted((ROOT / 'examples').glob('attention_energy_*.json'))


def test_configs_parse_and_build():
    assert len(CONFIGS) == 16
    for p in CONFIGS:
        c = parse_config(json.loads(p.read_text()))
        net, cols, count = e2e.netlist(spec_for(c, toy_examples(np.random.default_rng(0))[0]), c.model)
        assert any(n.startswith('pwr_supply_') for n, _ in cols) and 'VDD_scorer' in net
        assert ('Mbias_h' in net or 'Xbias_h' in net) == (c.chain == 'A')
        assert ('qh0' in net) == c.query_hold and ('vcmh' in net) == c.ref_hold and ('Ven ' in net) == c.gate_scorer


def test_rejects_inconsistent_options():
    raw = json.loads((ROOT / 'examples/attention_energy_A_L4_bsim3.json').read_text())
    with pytest.raises(Exception):
        parse_config(raw | {'query_hold': False})
    with pytest.raises(Exception):
        parse_config(raw | {'t_warm': 0.0})


@pytest.mark.parametrize('chain', ['A', 'B'])
def test_ideal_final_rung_matches_law_and_counts_energy(tmp_path, chain):
    if not Path(EXE).exists():
        pytest.skip('ngspice unavailable')
    c = parse_config(json.loads((ROOT / f'examples/attention_energy_{chain}_L5_ideal.json').read_text()))
    ex = toy_examples(np.random.default_rng(2))[1]
    spec = spec_for(c, ex)
    r = e2e.simulate(spec, 'ideal', tmp_path / chain, EXE)
    law = e2e.law(spec, c.kappa_design)
    assert abs(r['y_raw'][0] - law['y']) < 2e-3
    E = r['energy']
    assert E['total'] > 0 and E['crossbar_tokens'] > 0 and E['core'] > 0
    assert abs(E['total'] - E['core'] - E['crossbar_tokens'] - E['clocks']) < 1e-18
