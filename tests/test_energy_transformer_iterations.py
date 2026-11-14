from dataclasses import asdict
import pytest
import torch
from networks.energy_transformer.gpu import EnergyNetwork,NAMES,ep_gradient
from networks.energy_transformer.finite_time import relax,finite_ep,bptt_gradient,agreement,select_cap
from ebl.schema.energy_transformer_iterations import Config,parse_config
from ebl.schema.base import ConfigError

torch.set_num_threads(1)


def case():
    model=EnergyNetwork(3,4,2,heads=1,dtype=torch.float64)
    gen=torch.Generator().manual_seed(7)
    x=torch.randn(2,3,4,dtype=torch.float64,generator=gen)*.2
    target=torch.randn(2,3,2,dtype=torch.float64,generator=gen)*.2
    return model,x,target,torch.ones_like(target)


def test_exact_cap_and_matching_forward_for_ep_bptt():
    model,x,target,mask=case();original={k:v.clone() for k,v in model.state_dict().items()}
    ep,phases=finite_ep(model,x,target,mask,8)
    bp,phase=bptt_gradient(model,x,target,mask,8)
    assert all((p.iterations==8).all() for p in phases)
    torch.testing.assert_close(phases[0].y,phase.y,rtol=0,atol=0)
    assert phases[0].residual>1e-6
    assert all(not p.requires_grad and torch.equal(p,original[name]) for name,p in model.named_parameters())
    assert all(not value.requires_grad for gradient in (ep,bp) for value in gradient.values())


def test_bptt_matches_finite_differences_of_same_truncated_map():
    model,x,target,mask=case();gradient,_=bptt_gradient(model,x,target,mask,16)
    gen=torch.Generator().manual_seed(11)
    for name in NAMES:
        p=getattr(model,name);base=p.clone();direction=torch.randn(p.shape,dtype=p.dtype,generator=gen)
        direction/=direction.norm();cost=[]
        for sign in (-1,1):
            p.copy_(base+sign*1e-5*direction)
            phase=relax(model,x,target,mask,16)
            cost.append((.5*(mask*(phase.y-target).square()).sum()/mask.sum()).item())
        p.copy_(base)
        numerical=(cost[1]-cost[0])/2e-5
        expected=(gradient[name]*direction).sum().item()
        assert numerical==pytest.approx(expected,rel=1e-4,abs=1e-9)


def test_long_cap_recovers_equilibrium_ep_and_bptt_agreement():
    model,x,target,mask=case()
    ep,phases=finite_ep(model,x,target,mask,128)
    exact,_=ep_gradient(model,x,target,mask,tolerance=1e-11)
    bp,_=bptt_gradient(model,x,target,mask,128)
    assert agreement(ep,exact)['global']['relative_l2']<1e-6
    assert agreement(ep,bp)['global']['cosine']>.999
    assert all(p.residual<1e-10 for p in phases)


def test_selection_requires_all_batches_and_a_qualified_reference():
    def row(cap,batch,value=.95,valid=True):
        return {'iterations':cap,'batch':batch,'reference_valid':valid,
                **{k:{'global':{'cosine':value}} for k in ('ep_vs_bptt','ep_vs_reference','bptt_vs_reference')}}
    rows=[row(16,0),row(16,1,.8),row(32,0),row(32,1)]
    assert select_cap(rows,(16,32),2)==32
    rows[-1]['reference_valid']=False
    assert select_cap(rows,(16,32),2) is None
    assert select_cap([row(16,0)],(16,),2) is None


def test_candidate_caps_and_training_mode_are_bounded():
    from ebl.schema.energy_transformer_mnist import parse_config as training
    for change in ({'iterations':[256]},{'iterations':[16,8]},{'batches':4},{'threshold':.8},{'case':'wider'}):
        with pytest.raises(ConfigError):parse_config({**asdict(Config()),**change})
    assert training({'phase_mode':'fixed_steps','max_steps':128}).max_steps==128
    with pytest.raises(ConfigError):training({'phase_mode':'fixed_steps','max_steps':129})
    assert training({}).max_steps==4096  # Historical strict protocol remains reproducible.


def test_public_validate_configs_and_finite_training_prediction():
    from pathlib import Path
    from ebl.registry import resolve_experiment_config
    from ebl.schema.base import RunMode
    from ebl.schema.energy_transformer_mnist import Config as TrainingConfig
    from ebl.runtime.energy_transformer_mnist import predict
    root=Path(__file__).resolve().parents[1]/'examples/energy_transformer_iterations'
    paths=list(root.glob('*.json'));assert len(paths)==3
    for path in paths:
        definition,config=resolve_experiment_config(path,RunMode.VALIDATE)
        assert definition.experiment_id=='energy_transformer_iterations.v1'
        assert max(config.iterations)==128
    model,x,target,mask=case()
    result,diagnostic=predict(model,x,target,mask,TrainingConfig(phase_mode='fixed_steps',max_steps=8),None)
    assert diagnostic['iterations']==8 and diagnostic['residual']>1e-6
    assert not result.requires_grad
