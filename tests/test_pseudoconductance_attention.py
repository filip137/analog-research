"""Small SPICE wiring/sensitivity test, skipped when the simulator is absent."""
import shutil
from pathlib import Path
import numpy as np
import pytest
from networks.energy_transformer.pseudoconductance import export,simulate,reference
from ebl.schema.pseudoconductance_attention import parse_config
from ebl.schema.base import ConfigError

def test_current_ports_and_model_contract():
    with pytest.raises(ConfigError):
        parse_config({'model':'level1'})
    v=np.array([[-.6,.4],[.2,-.5],[.7,.1]])
    a=np.array([.001,.001,5.])
    text,_,counts=export(v,a)
    assert sum(counts.values())==51
    assert 'Bch' not in text and '.model nc nmos level=49' in text
    assert 'Inudge_0 0 z3_0' in text
    assert 'Medge_0_0 edge_0_0 g_a0 z3_0 0 nc' in text

def test_spice_selects_changed_control(tmp_path):
    exe=shutil.which('ngspice') or '/tmp/pseudoconductance-spice-env/bin/ngspice'
    if not Path(exe).exists(): pytest.skip('ngspice unavailable')
    v=np.array([[-.6,.4],[.2,-.5],[.7,.1]])
    for selected in [0,2]:
        a=np.full(3,.001);a[selected]=5
        r=simulate(v,a,tmp_path/str(selected),exe,model='ideal')
        assert np.max(abs(r['y']-reference(v,a)))<1e-3
        assert np.max(abs(r['y']-v[selected]))<.03

def test_rejects_aborted_analysis_with_zero_exit(tmp_path,monkeypatch):
    from types import SimpleNamespace
    v=np.zeros((3,2));a=np.ones(3)
    _,columns,_=export(v,a)
    def fake_run(*args,**kwargs):
        path=Path(kwargs['cwd'])/'trace.txt'
        np.savetxt(path,np.zeros((1,len(columns)+1)),header='mock header')
        return SimpleNamespace(returncode=0,stdout='tran simulation(s) aborted',stderr='')
    monkeypatch.setattr('networks.energy_transformer.pseudoconductance.subprocess.run',fake_run)
    with pytest.raises(RuntimeError,match='aborted'):
        simulate(v,a,tmp_path/'aborted','unused',transient=True)
