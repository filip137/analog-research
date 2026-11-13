"""Circuit-learning contracts and native SPICE EP regression."""
from dataclasses import replace
from pathlib import Path
import json
import os
import time
import numpy as np
import pytest

from networks.energy_transformer.convergent import PARAMETERS
from networks.energy_transformer.convergent_spice import export_netlist
from networks.energy_transformer.learning import initial_parameters,boundary,dataset
from networks.energy_transformer import transistor
from ebl.schema.mos_attention_learning import Config,parse_config
from ebl.schema.base import ConfigError,RunMode


def test_task_requires_cross_token_information_and_has_frozen_initialization():
    p=initial_parameters();b=boundary([.3,-.1])
    assert p.shape==(3,3) and p.size==12
    assert sum(getattr(p,n).size for n in PARAMETERS)==27
    np.testing.assert_array_equal(b.mask.ravel(),[0,0,1])
    assert b.inputs[2,0]==0 and b.target[2,0]==pytest.approx(.1)
    assert boundary([-.3,-.1]).target[2,0]==pytest.approx(-.2)
    assert not np.array_equal(dataset(3100,12),dataset(3102,12))


def test_native_attention_has_no_behavioral_products_or_exponential_channel():
    p=initial_parameters();text,_=export_netlist(p,boundary([.2,.1]),
        softmax_device=transistor.SOFTMAX_DEVICE,multiplier='mos_triode',terminal_only=True)
    lines=text.splitlines()
    assert sum(line.startswith('Mproduct') for line in lines)==108
    assert sum(line.startswith('Msoft') for line in lines)==9
    assert not any(line.startswith(('Bmultiply','Bmoschannel','Qsoft')) for line in lines)
    assert sum(line.startswith('Eregulated') for line in lines)==3
    assert 'setscale finaltime' in lines
    for invalid in (0.,float('nan')):
        with pytest.raises(ValueError): export_netlist(p,boundary([.2,.1]),duration=invalid)


def test_public_train_config_rejects_silent_scope_changes():
    from ebl.registry import resolve_experiment_config
    from ebl.schema.mos_attention_learning import resolve_config
    path=Path(__file__).resolve().parents[1]/'examples/mos_attention_learning.json'
    raw=json.loads(path.read_text());definition,c=resolve_experiment_config(path,RunMode.TRAIN)
    assert definition.experiment_id=='mos_attention_learning.v1' and c==parse_config(raw)
    for change in ({'unrecognized':1},{'epochs':201},{'beta':True},{'beta':float('nan')},
                   {'wall_seconds':1801},{'train_seed':3102},{'schema_version':True}):
        with pytest.raises(ConfigError): parse_config({**raw,**change})
    with pytest.raises(ConfigError): resolve_config(c,RunMode.CHARACTERIZE)


def test_measured_ep_never_uses_ideal_ports(monkeypatch):
    from methods import convergent_energy_transformer as method
    p=initial_parameters();b=boundary([.3,-.1])
    measured={'g':np.ones((3,3)),'q':np.ones((1,3,1)),'k':np.ones((1,3,1)),
              'p':np.full((1,3,3),1/3),'r':np.ones((3,2)),'y':np.ones((3,1))}
    def forbidden(*args,**kwargs): raise AssertionError('Ideal ports cannot train the circuit')
    monkeypatch.setattr(method,'ports',forbidden)
    gradient=method.phase_gradient(p,b,np.zeros(12),np.zeros(12),.005,measured,measured)
    assert all(np.count_nonzero(value)==0 for value in gradient.values())


def test_native_bank_and_ep_match_physical_circuit_differences(tmp_path):
    from ebl.runtime.mos_attention_learning import Circuit,primitive_checks,gradient_check
    executable=os.environ.get('ENERGY_TRANSFORMER_NGSPICE')
    if not executable: pytest.skip('Set ENERGY_TRANSFORMER_NGSPICE for native MOS circuit checks')
    circuit=Circuit(tmp_path,executable,Config(),time.monotonic()+90)
    assert primitive_checks(circuit)['passed']
    result=gradient_check(circuit,initial_parameters(),'gradient')
    assert result['passed'],result
    assert circuit.worst['residual']<1e-7
    assert circuit.worst['product_error']<1e-7
