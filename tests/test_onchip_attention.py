"""On-chip learning of chain A: configs, nudge netlist, rule bookkeeping."""
import json
from pathlib import Path
import numpy as np
from networks.energy_transformer import end_to_end as e2e
from ebl.schema.onchip_attention import parse_config
from ebl.runtime import onchip_attention as oa

ROOT = Path(__file__).resolve().parents[1]


def test_configs_and_nudge_netlist():
    for n in ('finetune', 'scratch'):
        c = parse_config(json.loads((ROOT / f'examples/onchip_attention_{n}_bsim3.json').read_text()))
        d = oa.Params(c).design(c)
        ex = oa.task(np.random.default_rng(0), 1)[0]
        sp = oa.energy_spec(d, ex, t_settle=c.t_settle_train) | {'nudge': {'beta': c.beta, 'x_target': 1.0, 't_phase': c.t_phase}}
        net = e2e.netlist(sp, 'bsim3')[0]
        assert 'Bnud z3_0 0 I' in net and 'Vnb nb 0 PWL' in net and 'Ibias_0 psum0 0' in net
        T = e2e.timing(sp)
        assert np.isclose(T['t_end'] - T['t_free'], 2 * c.t_phase)


def test_adam_moves_against_gradient():
    c = parse_config(json.loads((ROOT / 'examples/onchip_attention_scratch_bsim3.json').read_text()))
    p = oa.Params(c)
    before = {k: v.copy() for k, v in p.v.items()}
    p.adam({k: np.ones_like(v) for k, v in p.v.items()}, 0.01)
    assert all(np.all(p.v[k] < before[k]) for k in p.v)
