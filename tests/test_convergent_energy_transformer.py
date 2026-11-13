"""Energy, reciprocal circuit and EP checks for the multi-token extension."""
from dataclasses import replace
import os
import numpy as np
import pytest
import torch

from networks.energy_transformer.convergent import (PARAMETERS, pilot_case, ports, force,
    branch_force, energy, torch_energy, dissipation, simulate, loss)
from networks.energy_transformer.convergent_spice import Builder, export_netlist, run_spice
from networks.energy_transformer import mosfet
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


@pytest.mark.parametrize("device",["bjt",mosfet.DEVICE])
def test_netlist_uses_explicit_weight_resistors_and_declared_device_pools(device):
    p,b=pilot_case()
    text,_=export_netlist(p,b,softmax_device=device)
    lines=text.splitlines()
    assert sum(line.startswith("Cstate") for line in lines)==p.size
    assert sum(line.startswith("Qsoft") for line in lines)==(32 if device=="bjt" else 0)
    assert sum(line.startswith("Xsoftmos") for line in lines)==(32 if device==mosfet.DEVICE else 0)
    assert sum(line.startswith("Rweight") for line in lines)>400
    assert sum(line.startswith("Bmultiply") for line in lines)==3*2*4*4*2
    assert any(line.startswith("Bnormalize") for line in lines)
    assert all(line.startswith(("Bmean","Bvariance","Bsqrt","Bnormalize","Bmultiply","Brectify","Bmoschannel")) for line in lines if line.startswith("B"))
    with pytest.raises(ValueError): export_netlist(p,b,softmax_device="unknown")


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
                   {"wall_seconds":601},{"case":"unfrozen"},{"learning_rate":float("nan")},{"softmax_device":"unknown"}):
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
        device=mosfet.operating_metrics(trace.ports["softmax_sources"],trace.ports["softmax_gates"],trace.ports["p"])
        assert device["weak_inversion_margin"]>0 and device["maximum_branch_current_over_i0"]<1
        assert device["current_law_error_over_tail"]<1e-6 and device["tail_fraction_error"]<1e-6
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


def test_mosfet_bank_transfer_common_mode_and_jacobian(tmp_path):
    import subprocess
    from scipy.special import softmax
    executable=os.environ.get("ENERGY_TRANSFORMER_NGSPICE")
    if not executable: pytest.skip("Set ENERGY_TRANSFORMER_NGSPICE for device-law checks")
    counter=0
    def evaluate(scores):
        nonlocal counter
        folder=tmp_path/str(counter);folder.mkdir();counter+=1
        net=Builder(mosfet.DEVICE)
        for i,value in enumerate(scores): net.lines.append(f"Vinput{i} input{i} 0 {value:.17g}")
        outputs=net.softmax([f"input{i}" for i in range(len(scores))],1.)
        bank=net.banks[0]
        vectors=outputs+[bank["source"]]+bank["gates"]
        net.lines += [".control","set noaskquit","set numdgt=15","set wr_singlescale","set wr_vecnames","op",
                      "wrdata ports.tsv "+" ".join(f"v({node})" for node in vectors),"quit",".endc",".end"]
        (folder/"bank.cir").write_text("\n".join(net.lines)+"\n")
        with (folder/"stdout.log").open("w") as log:
            subprocess.run([executable,"-n","-b","-o","spice.log","bank.cir"],cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=10,check=True)
        values=np.loadtxt(folder/"ports.tsv",skiprows=1,ndmin=2)[-1,1:]
        n=len(scores);prob=values[:n]
        metrics=mosfet.operating_metrics(np.array(values[n]),values[n+1:],prob)
        assert metrics["weak_inversion_margin"]>0 and metrics["maximum_branch_current_over_i0"]<1
        assert metrics["current_law_error_over_tail"]<1e-6
        np.testing.assert_allclose(prob,softmax(scores),rtol=1e-6,atol=1e-8)
        return prob
    for point in (np.zeros(3),np.array([-6.,0.,6.])): evaluate(point)
    point=np.array([.2,-.3,.1]);base=evaluate(point)
    np.testing.assert_allclose(evaluate(point+5.),base,atol=1e-8)
    delta=1e-4
    jac=np.column_stack([(evaluate(point+delta*e)-evaluate(point-delta*e))/(2*delta) for e in np.eye(3)])
    np.testing.assert_allclose(jac,np.diag(base)-np.outer(base,base),rtol=1e-5,atol=1e-7)
