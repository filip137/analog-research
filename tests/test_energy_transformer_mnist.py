from dataclasses import asdict,replace
import pytest
import torch
from networks.energy_transformer.gpu import EnergyNetwork,DigitalNetwork,settle,ep_gradient,EquilibriumError,diode,diode_residual
from networks.energy_transformer.mnist import hidden_masks,patchify,unpatchify,batch
from ebl.schema.energy_transformer_mnist import Config,parse_config,CASES
from ebl.schema.base import ConfigError

torch.set_num_threads(1)


def test_equal_counts_and_shared_initial_maps():
    for tokens,width,patch in CASES.values():
        e=EnergyNetwork(tokens,width,patch**2);d=DigitalNetwork(tokens,width,patch**2)
        assert sum(p.numel() for p in e.parameters())==sum(p.numel() for p in d.parameters())
        assert sum(getattr(e,n).numel() for n in ('query','key','memory'))==6*width**2
        for name in ('query','key','encoder','decoder','position'):
            assert torch.equal(getattr(e,name),getattr(d,name))


def test_masks_have_no_target_leakage_and_match_across_tokenizations():
    gen=torch.Generator().manual_seed(3);m=hidden_masks(5,gen)
    assert (m.sum((1,2))==392).all()
    images=torch.randint(256,(5,28,28),generator=gen)
    for patch in (4,7):
        x,y,mask=batch(images,m,patch,'cpu');p=patch**2
        assert torch.equal(unpatchify(mask,patch).bool(),m)
        assert torch.equal(x[...,:p][mask.bool()],torch.zeros_like(y[mask.bool()]))
        assert torch.equal(x[...,p:],1-mask)
        assert torch.equal(unpatchify(patchify(images,patch),patch),images)


def test_diode_complementarity_switching_and_energy_minimum():
    drive=torch.tensor([-3.,-1e-5,0.,1e-5,3.],dtype=torch.float64);r=diode(drive)
    assert diode_residual(drive,r)==0
    e=.5*r.square()-r*drive
    for change in (-.1,.1):
        alternative=(r+change).clamp_min(0)
        assert (.5*alternative.square()-alternative*drive>=e-1e-15).all()


def test_energy_force_identity_and_descent():
    model=EnergyNetwork(3,4,2,heads=1,dtype=torch.float64)
    gen=torch.Generator().manual_seed(32)
    x=torch.randn(2,3,4,generator=gen,dtype=torch.float64)*.1
    target=torch.zeros(2,3,2,dtype=torch.float64);mask=torch.ones_like(target)
    z=torch.randn(2,3,4,generator=gen,dtype=torch.float64).requires_grad_()
    y=torch.randn(2,3,2,generator=gen,dtype=torch.float64).requires_grad_()
    v=model.ports(z,y);dz,dy=model.force(v,x,target,mask,.005)
    gz,gy=torch.autograd.grad(model.energy(v,x,target,mask,.005).sum(),(z,y),retain_graph=True)
    hforce,=torch.autograd.grad(v['g'],z,grad_outputs=dz.detach())
    torch.testing.assert_close(gz,-hforce);torch.testing.assert_close(gy,-dy)
    assert ((gz*dz).sum()+(gy*dy/1.3).sum())<=0
    trace=[];settle(model,x,target,mask,trace=trace,tolerance=1e-10)
    assert all((b<=a+1e-11).all() for a,b in zip(trace,trace[1:]))


def test_fp32_ep_independent_reference():
    from experiments.energy_transformer_mnist_checks import gradient_check,device_checks
    assert device_checks()['passed']
    result=gradient_check()
    assert result['passed'],result


def test_no_grad_trajectory_and_nonconvergence_rejected():
    model=EnergyNetwork(3,4,2,heads=1)
    x=torch.ones(2,3,4);target=torch.ones(2,3,2)*.2;mask=torch.ones_like(target)
    with pytest.raises(EquilibriumError):
        ep_gradient(model,x,target,mask,max_steps=1)
    grad,phases=ep_gradient(model,x,target,mask)
    assert all(not p.z.requires_grad and not p.y.requires_grad for p in phases)
    assert all(not g.requires_grad and torch.isfinite(g).all() for g in grad.values())


def test_strict_schema():
    for invalid in ({'precision':'fp16'},{'learning_rate':.01},{'updates':2001},{'batch_size':True},
                    {'wall_seconds':100},{'schema_version':True},{'unexpected':2}):
        with pytest.raises(ConfigError):parse_config({**asdict(Config()),**invalid})


def test_registered_examples_use_the_public_train_contract():
    from pathlib import Path
    from ebl.registry import resolve_experiment_config
    from ebl.schema.base import RunMode
    root=Path(__file__).resolve().parents[1]/'examples/energy_transformer_mnist'
    configs=list(root.glob('*.json'))
    assert len(configs)==8
    for path in configs:
        definition,config=resolve_experiment_config(path,RunMode.TRAIN)
        assert definition.experiment_id=='energy_transformer_mnist.v1'
        assert config.precision=='fp32' and config.batch_size==64


def test_checkpoint_restores_rng_optimizer_and_next_step(tmp_path):
    from ebl.runtime.energy_transformer_mnist import save_checkpoint,restore_checkpoint
    c=Config(arm='digital_bp');m=DigitalNetwork(3,4,2,heads=1)
    opt=torch.optim.Adam(m.parameters(),lr=c.learning_rate)
    a=torch.Generator().manual_seed(1);b=torch.Generator().manual_seed(2)
    x=torch.randn(2,3,4);m(x).square().sum().backward();opt.step();opt.zero_grad()
    path=tmp_path/'resume.pt';save_checkpoint(path,m,opt,c,1,2.,a,b)
    draws=torch.rand(5,generator=a);m(x).square().sum().backward();opt.step()
    expected={n:p.detach().clone() for n,p in m.named_parameters()}
    assert restore_checkpoint(path,m,opt,c,a,b)==(1,2.)
    assert torch.equal(torch.rand(5,generator=a),draws)
    opt.zero_grad();m(x).square().sum().backward();opt.step()
    for name,p in m.named_parameters():torch.testing.assert_close(p,expected[name],rtol=0,atol=0)
    with pytest.raises(ConfigError):restore_checkpoint(path,m,opt,replace(c,case='wider'),a,b)
