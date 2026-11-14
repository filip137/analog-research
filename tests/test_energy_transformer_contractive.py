"""Small mathematical fixtures; task training and replay require GPU FP32."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import pytest
import torch

from ebl.schema.base import ConfigError
from ebl.schema.energy_transformer_mnist import Config, parse_config
from networks.energy_transformer.contractive import bound, initialize_shared, project, recurrent_norms
from networks.energy_transformer.gpu import EnergyNetwork, DigitalNetwork


def model():
    return EnergyNetwork(3,8,2,heads=2,dtype=torch.float64,device_domain='physical',solver_dt=1.)


def test_projection_enforces_bound_and_preserves_drives():
    m=model()
    with torch.no_grad():
        for name in ('query','key','memory','decoder'):getattr(m,name).mul_(4)
    drives={name:getattr(m,name).clone() for name in ('encoder','position')}
    result=project(m)
    assert 0<result['scale']<1
    assert result['after']['normalized_total']<=.7
    assert result['after']['contraction_radius']<.859
    for name,value in drives.items():assert torch.equal(getattr(m,name),value)
    assert project(m)['scale']==pytest.approx(1.,abs=1e-12)


def test_hessian_bound_includes_nonlinear_attention_and_diode_memory():
    m=model();report=project(m);gen=torch.Generator().manual_seed(821)
    certified=report['after']['attention']+report['after']['memory']
    for magnitude in (.2,.5,.9):
        g=torch.randn(3,8,generator=gen,dtype=torch.float64)
        g-=g.mean(-1,keepdim=True)
        g*=math.sqrt(8)*magnitude/g.norm(dim=-1,keepdim=True)
        def coupling(flat):
            v=flat.reshape(3,8)
            q=torch.einsum('nd,had->hna',v,m.query)
            k=torch.einsum('nd,had->hna',v,m.key)
            return (torch.logsumexp(m.kappa*(q@k.transpose(-1,-2)),-1).sum()/m.kappa
                    +.5*torch.relu(v@m.memory.T).square().sum())
        h=torch.autograd.functional.hessian(coupling,g.flatten())
        assert torch.linalg.eigvalsh(h).abs().max()<=certified+1e-12


def test_joint_update_contracts_in_certified_weighted_product_norm():
    m=model();r=project(m)['after'];norms=recurrent_norms(m);d=norms['decoder']
    a=(r['attention']+r['memory'])/math.sqrt(m.epsilon)
    weights=(d,r['contraction_radius']-a)
    gen=torch.Generator().manual_seed(43)
    x=torch.randn(2,3,4,generator=gen,dtype=torch.float64)
    target=torch.randn(2,3,2,generator=gen,dtype=torch.float64);mask=torch.ones_like(target)
    z=torch.randn(2,3,8,generator=gen,dtype=torch.float64)*2
    y=torch.randn(2,3,2,generator=gen,dtype=torch.float64)
    # Same drive for the two trajectories; compare their one-step response.
    x=x[:1].expand_as(x)
    dz,dy=m.force(m.ports(z,y),x,target[:1].expand_as(target),mask,-.005)
    nz,ny=z+dz,y+dy/1.3
    before=max(float((z[0]-z[1]).norm())/weights[0],float((y[0]-y[1]).norm())/weights[1])
    after=max(float((nz[0]-nz[1]).norm())/weights[0],float((ny[0]-ny[1]).norm())/weights[1])
    assert after<=r['contraction_radius']*before+1e-12


@pytest.mark.parametrize('tokens,width,patch',[(16,32,49),(16,64,49),(49,64,16)])
def test_paired_initialization_is_exact_and_digital_has_no_added_parameters(tokens,width,patch):
    e=EnergyNetwork(tokens,width,patch);d=DigitalNetwork(tokens,width,patch)
    digital_ff={n:getattr(d,n).clone() for n in ('value','output','ff1','ff2')}
    er=initialize_shared(e,'contractive_blocks_v1');dr=initialize_shared(d,'contractive_blocks_v1')
    assert er==dr
    for name in ('query','key','encoder','decoder','position'):
        assert torch.equal(getattr(e,name),getattr(d,name))
    for name,value in digital_ff.items():assert torch.equal(getattr(d,name),value)
    assert sum(p.numel() for p in e.parameters())==sum(p.numel() for p in d.parameters())


def test_standard_initialization_and_old_checkpoint_defaults_are_preserved():
    m=model();before={n:p.clone() for n,p in m.named_parameters()}
    assert initialize_shared(m) is None
    for name,p in m.named_parameters():assert torch.equal(p,before[name])
    c=parse_config({})
    assert c.initialization=='standard' and c.recurrent_constraint=='none'


def test_contractive_recipes_are_explicit_and_invalid_combinations_fail():
    root=Path(__file__).resolve().parents[1]/'examples/energy_transformer_mnist/contractive32'
    e=parse_config(json.loads((root/'more_tokens-energy_ep.json').read_text()))
    d=parse_config(json.loads((root/'more_tokens-digital_bp.json').read_text()))
    assert e.recurrent_constraint=='contractive_blocks_v1' and e.max_steps==32
    assert d.recurrent_constraint=='none'
    assert e.initialization==d.initialization=='contractive_blocks_v1'
    for changes in ({'initialization':'unknown'},{'initialization':'standard'},
                    {'device_domain':'declared4'},{'solver_dt':.5},{'recurrent_constraint':'unbounded'}):
        with pytest.raises(ConfigError):parse_config({**asdict(e),**changes})
    with pytest.raises(ConfigError):parse_config({**asdict(d),'recurrent_constraint':'contractive_blocks_v1'})


def test_memory_projection_does_not_erase_other_feasible_maps():
    m=model();project(m)
    before={name:getattr(m,name).clone() for name in ('query','key','decoder','encoder','position')}
    with torch.no_grad():m.memory.mul_(10)
    result=project(m)
    assert result['scales']['memory']<.11
    for name,value in before.items():
        torch.testing.assert_close(getattr(m,name),value,atol=2e-14,rtol=2e-14)
    assert result['after']['normalized_total']<=.7
