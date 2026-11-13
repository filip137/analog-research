"""Energy, reciprocal circuit and EP checks for the multi-token extension."""
from dataclasses import replace
import os
import numpy as np
import pytest
import torch

from networks.energy_transformer.convergent import (PARAMETERS, pilot_case, ports, force,
    branch_force, energy, torch_energy, dissipation, simulate, loss)
from networks.energy_transformer.convergent_spice import export_netlist, run_spice
from methods.convergent_energy_transformer import (eqprop, phase_gradient,
    implicit_gradient, finite_difference_directions, update)


@pytest.mark.parametrize("nudge",[-.04,0.,.04])
def test_reciprocal_kcl_and_energy_identity(nudge):
    p,b=pilot_case()
    rng=np.random.default_rng(37)
    tau=np.r_[np.ones(p.position.size),np.full(np.prod(p.output_shape),1.3)]
    for state in rng.normal(size=(8,p.size)):
        np.testing.assert_allclose(branch_force(p,state,b,nudge),force(p,state,b,nudge),atol=1e-13)
        raw=torch.tensor(state,dtype=torch.float64,requires_grad=True)
        e=torch_energy(p,raw,b,nudge)
        assert float(e.detach())==pytest.approx(energy(p,state,b,nudge),abs=1e-12)
        gradient,=torch.autograd.grad(e,raw)
        slope=gradient.detach().numpy()@(force(p,state,b,nudge)/tau)
        assert slope<=1e-12
        assert slope==pytest.approx(dissipation(p,state,b,nudge),abs=1e-10)


def test_both_attention_paths_are_required_and_attention_is_token_dependent():
    p,b=pilot_case()
    state=np.linspace(-.4,.7,p.size)
    assert np.max(np.abs(force(p,state,b)-force(p,state,b,omit_key_return=True)))>1e-5
    altered=state.copy(); altered[0]+=.8
    assert np.max(np.abs(ports(p,state)["p"]-ports(p,altered)["p"]))>1e-5
    raw=torch.tensor(state,dtype=torch.float64,requires_grad=True)
    gradient,=torch.autograd.grad(torch_energy(p,raw,b),raw)
    tau=np.r_[np.ones(p.position.size),np.full(np.prod(p.output_shape),1.3)]
    wrong=gradient.detach().numpy()@(force(p,state,b,omit_key_return=True)/tau)
    assert abs(wrong-dissipation(p,state,b))>1e-6


def test_eqprop_matches_independent_implicit_energy_and_loss_differences():
    p,b=pilot_case()
    free=simulate(p,b)
    for start in (np.linspace(-.4,.5,p.size),np.linspace(.5,-.4,p.size)):
        np.testing.assert_allclose(simulate(p,b,initial=start).states[-1],free.states[-1],atol=1e-8)
    values=[energy(p,s,b) for s in free.states]
    assert np.max(np.diff(values))<1e-9
    reference,min_eig=implicit_gradient(p,free.states[-1],b)
    assert min_eig>0
    estimate,_,_=eqprop(p,b,.005,free.states[-1])
    for name in PARAMETERS:
        np.testing.assert_allclose(estimate[name],reference[name],rtol=2e-3,atol=1e-8)
    fd=finite_difference_directions(p,b,reference)
    assert max(row["relative_error"] for row in fd.values())<2e-4
    changed=update(p,estimate,.05)
    assert loss(changed,simulate(changed,b).states[-1],b)<loss(p,free.states[-1],b)


def test_contract_rejects_invalid_boundaries_and_incomplete_settling():
    p,b=pilot_case()
    with pytest.raises(ValueError): replace(p,epsilon=0.)
    with pytest.raises(ValueError): force(p,np.zeros(p.size),replace(b,mask=-b.mask))
    with pytest.raises(ValueError): force(p,np.zeros(p.size),b,-1.)
    with pytest.raises(ValueError): eqprop(p,b,0.)
    with pytest.raises(RuntimeError,match="did not settle"): simulate(p,b,times=np.array([0.,1e-5]))


def test_netlist_uses_explicit_weight_resistors_and_bjt_pools():
    p,b=pilot_case()
    text,_=export_netlist(p,b)
    lines=text.splitlines()
    assert sum(line.startswith("Cstate") for line in lines)==p.size
    assert sum(line.startswith("Qsoft") for line in lines)==2*4*4
    assert sum(line.startswith("Rweight") for line in lines)>400
    assert sum(line.startswith("Bmultiply") for line in lines)==3*2*4*4*2
    assert any(line.startswith("Bnormalize") for line in lines)
    assert all(line.startswith(("Bmean","Bvariance","Bsqrt","Bnormalize","Bmultiply","Brectify")) for line in lines if line.startswith("B"))


def test_public_characterization_config_is_strict():
    import json
    from pathlib import Path
    from ebl.schema.base import ConfigError, RunMode
    from ebl.schema.convergent_energy_transformer import parse_config,resolve_config
    from ebl.registry import resolve_experiment_config
    path=Path(__file__).resolve().parents[1]/"examples/convergent_energy_transformer_circuit.json"
    raw=json.loads(path.read_text())
    config=parse_config(raw)
    definition,resolved=resolve_experiment_config(path,RunMode.CHARACTERIZE)
    assert definition.experiment_id==config.experiment_id and resolved==config
    for change in ({"unknown":1},{"schema_version":True},{"betas":[True]},{"betas":[0.]},{"betas":[.01,.01]},
                   {"wall_seconds":601},{"case":"unfrozen"},{"learning_rate":float("nan")}):
        with pytest.raises(ConfigError): parse_config({**raw,**change})
    with pytest.raises(ConfigError): resolve_config(config,RunMode.TRAIN)


def test_spice_equilibria_and_measured_phase_gradients(tmp_path):
    executable=os.environ.get("ENERGY_TRANSFORMER_NGSPICE")
    if not executable: pytest.skip("Set ENERGY_TRANSFORMER_NGSPICE for actual circuit simulation")
    p,b=pilot_case()
    free=simulate(p,b).states[-1]
    reference,_=implicit_gradient(p,free,b)
    traces={}
    for label,nudge in (("free",0.),("minus",-.005),("plus",.005)):
        trace=run_spice(p,b,tmp_path/label,executable=executable,nudge=nudge)
        numerical=simulate(p,b,nudge,times=trace.time)
        np.testing.assert_allclose(trace.states[-1],numerical.states[-1],atol=1e-5)
        assert np.max(np.abs(trace.states-numerical.states))<1e-4
        actual={key:value[-1] for key,value in trace.ports.items()}
        ideal=ports(p,trace.states[-1])
        for key in ("g","q","k","p","r"):
            np.testing.assert_allclose(actual[key],ideal[key],atol=5e-6)
        assert np.max(np.abs(actual["token_drive"]-actual["z"]))<1e-6
        assert np.max(np.abs(actual["output_drive"]-actual["y"]-nudge*b.mask*(actual["y"]-b.target)))<1e-6
        traces[label]=(trace,actual)
    minus,mp=traces["minus"];plus,pp=traces["plus"]
    estimate=phase_gradient(p,b,minus.states[-1],plus.states[-1],.005,mp,pp)
    for name in PARAMETERS:
        error=np.linalg.norm(estimate[name]-reference[name])
        assert error/max(np.linalg.norm(reference[name]),1e-8)<2e-3
    changed=update(p,estimate,.05)
    after=run_spice(changed,b,tmp_path/"updated",executable=executable)
    assert loss(changed,after.states[-1],b)<loss(p,traces["free"][0].states[-1],b)
