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


@pytest.mark.parametrize('solver_dt,solver_rule', [(.5,'energy_backtracking'),(1.,'energy_backtracking'),(1.,'cycle_backtracking')])
def test_bptt_matches_finite_differences_of_same_truncated_map(solver_dt,solver_rule):
    model,x,target,mask=case()
    model.solver_dt=solver_dt
    model.solver_rule=solver_rule
    model.device_domain='physical' if solver_dt==1 else 'declared4'
    gradient,_=bptt_gradient(model,x,target,mask,16)
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
    for change in ({'iterations':[256]},{'iterations':[16,8]},{'batches':9},{'threshold':.8},{'case':'wider'}):
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


def test_physical_domain_accepts_valid_products_beyond_old_guard_and_rejects_bad_voltages():
    from networks.energy_transformer.gpu import mos_domain,check_domain,EquilibriumError,triode_product
    model,_,_,_=case()
    q=torch.full((1,1,3,1),4.2,dtype=torch.float64)
    k=torch.full_like(q,4.1)
    logits=q@k.transpose(-1,-2)
    v={'q':q,'k':k,'p':logits.softmax(-1),'logits':logits}
    with pytest.raises(EquilibriumError,match='operand='):check_domain(model,v,'test')
    model.device_domain='physical';d=check_domain(model,v,'test')
    # Independently enumerate every device polarity and score/return product.
    margins=[];drains=[]
    for left,right in ((4.2,4.1),(1/3,4.2),(1/3,4.1)):
        for sg in (-1,1):
            for sd in (-1,1):
                gate=1.5+sg*.05*left;drain=.3+sd*.05*right
                margins.append(gate-.5-drain);drains.append(drain)
        actual=triode_product(torch.tensor(left,dtype=torch.float64),torch.tensor(right,dtype=torch.float64))
        assert actual.item()==pytest.approx(left*right,abs=1e-11)
    assert d['triode_margin'].item()==pytest.approx(min(margins))
    assert d['drain_margin'].item()==pytest.approx(min(drains))
    assert d['weak_inversion_margin'].item()>0
    v['q']=torch.full_like(q,6.1)
    with pytest.raises(EquilibriumError,match='drain_margin='):check_domain(model,v,'test')
    v['q']=q;v['logits']=torch.full_like(logits,140.)
    with pytest.raises(EquilibriumError,match='weak_inversion_margin='):check_domain(model,v,'test')


def test_full_step_repair_keeps_energy_force_and_reports_physical_margins():
    model,x,target,mask=case();model.device_domain='physical';model.solver_dt=1.
    ep,phases=finite_ep(model,x,target,mask,128)
    bp,phase=bptt_gradient(model,x,target,mask,128)
    assert agreement(ep,bp)['global']['cosine']>.999
    for p in phases:
        assert p.minimum_triode_margin>0 and p.minimum_drain_margin>0 and p.minimum_weak_inversion_margin>0
        assert p.energy_increase<1e-10
    torch.testing.assert_close(phase.y,phases[0].y,atol=0,rtol=0)


def test_final_repair_diagnostic_includes_original_failure_cohort():
    import json
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'examples/energy_transformer_iterations/repaired128/more_tokens-final.json'
    c=parse_config(json.loads(path.read_text()))
    assert c.case=='more_tokens' and c.iterations==(128,)
    assert c.batches*c.batch_size==64  # Includes the original failing validation position 35.
    assert c.reference_steps==(512,1024) and c.threshold==.9


def test_cycle_guard_resolves_real_energy_network_period_two_within_cap():
    import math
    model=EnergyNetwork(1,4,2,heads=1,dtype=torch.float64,
                        device_domain='physical',solver_dt=1.)
    for p in model.parameters():p.zero_()
    # Opposite Q/K maps create a stable continuous-time mode whose Euler
    # multiplier is near -1. Non-increasing energy alone permits oscillation.
    a=math.sqrt(math.sqrt(model.epsilon)/2)
    model.query[0].copy_(torch.eye(4,dtype=torch.float64)*a)
    model.key.copy_(-model.query)
    model.position[0,0]=.01;model.position[0,1]=-.01
    x=torch.zeros(1,1,4,dtype=torch.float64)
    target=torch.zeros(1,1,2,dtype=torch.float64);mask=torch.ones_like(target)
    before={k:v.clone() for k,v in model.state_dict().items()}
    cycling=relax(model,x,target,mask,128)
    assert cycling.residual>1e-3 and cycling.cycle_reductions==0
    model.solver_rule='cycle_backtracking'
    repaired=relax(model,x,target,mask,128)
    assert repaired.residual<1e-10 and repaired.cycle_reductions>0
    assert repaired.minimum_step_size<=.5
    assert repaired.energy_increase<1e-10
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert repaired.minimum_triode_margin>0 and repaired.minimum_drain_margin>0
    torch.testing.assert_close(repaired.y,cycling.y,atol=0,rtol=0)


def test_bptt_finite_difference_through_active_cycle_rejection():
    import math
    model=EnergyNetwork(1,4,2,heads=1,dtype=torch.float64,device_domain='physical',
                        solver_dt=1.,solver_rule='cycle_backtracking')
    for p in model.parameters():p.zero_()
    a=math.sqrt(math.sqrt(model.epsilon)/2)
    model.query[0].copy_(torch.eye(4,dtype=torch.float64)*a);model.key.copy_(-model.query)
    model.position[0,0]=.01;model.position[0,1]=-.01
    model.decoder[0,0]=.02;model.decoder[0,1]=-.02
    x=torch.zeros(1,1,4,dtype=torch.float64)
    target=torch.full((1,1,2),.2,dtype=torch.float64);mask=torch.ones_like(target)
    gradient,phase=bptt_gradient(model,x,target,mask,32)
    assert phase.cycle_reductions>0
    gen=torch.Generator().manual_seed(92)
    for name in ('query','key','decoder','position'):
        p=getattr(model,name);base=p.clone()
        direction=torch.randn(p.shape,dtype=p.dtype,generator=gen);direction/=direction.norm()
        costs=[]
        for sign in (-1,1):
            p.copy_(base+sign*1e-6*direction)
            state=relax(model,x,target,mask,32)
            assert state.cycle_reductions==phase.cycle_reductions
            costs.append((.5*(mask*(state.y-target).square()).sum()/mask.sum()).item())
        p.copy_(base)
        finite=(costs[1]-costs[0])/2e-6
        assert finite==pytest.approx((gradient[name]*direction).sum().item(),rel=1e-3,abs=1e-9)
