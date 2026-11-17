"""End-to-end attention step: configs, netlist structure, ideal-SPICE agreement with the law."""
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from networks.energy_transformer import end_to_end as e2e
from ebl.schema.end_to_end_attention import parse_config
from ebl.runtime.end_to_end_attention import spec_for
from ebl.runtime.crossbar_qk_attention import toy_examples

ROOT = Path(__file__).resolve().parents[1]
EXE = shutil.which('ngspice') or '/tmp/pseudoconductance-spice-env/bin/ngspice'


def cfg(chain, model):
    return parse_config(json.loads((ROOT / f'examples/e2e_attention_{chain}_{model}.json').read_text()))


def test_configs_and_netlists():
    for chain in 'AB':
        for model in ('ideal', 'bsim3'):
            c = cfg(chain, model)
            spec = spec_for(c, toy_examples(np.random.default_rng(0))[0])
            net, cols, count = e2e.netlist(spec, model)
            assert count['crossbar_resistors'] == 18 and count['hold_capacitors'] == 9
            assert ('Mbias_h' in net) == (chain == 'A' and model == 'bsim3')
            assert (count.get('switches', 0) == 36) == (model == 'bsim3')
            assert count['scorer'] == 37 and count['softmax'] == 5
    with pytest.raises(Exception):
        parse_config(json.loads((ROOT / 'examples/e2e_attention_A_ideal.json').read_text()) | {'chain': 'C'})


def test_law_selects_query_key():
    c = cfg('A', 'bsim3')
    ex = toy_examples(np.random.default_rng(1))[0]
    law = e2e.law(spec_for(c, ex), c.kappa_design)
    assert np.argmax(law['p']) == ex['selected'] and abs(law['p'].sum() - 1) < 1e-12


@pytest.mark.parametrize('chain', ['A', 'B'])
def test_ideal_spice_matches_law(tmp_path, chain):
    if not Path(EXE).exists():
        pytest.skip('ngspice unavailable')
    c = cfg(chain, 'ideal')
    ex = toy_examples(np.random.default_rng(2))[1]
    spec = spec_for(c, ex) | {'t_settle': .3 if chain == 'A' else .01}
    r = e2e.simulate(spec, 'ideal', tmp_path / chain, EXE)
    law = e2e.law(spec, c.kappa_design)
    assert np.allclose(r['softmax'], law['p'], atol=2e-3)
    assert abs(r['y_raw'][0] - law['y']) < 2e-3 and abs(r['y_raw'][1] + law['y']) < 2e-3
    assert max(abs(h['held'] - h['column']) for h in r['kv'].values()) < 1e-6
