"""Physical sparsity, equilibrium sensitivities and strict pilot/data contracts."""
import csv
from pathlib import Path
import time
import pytest
import torch
from torch import nn
from networks.sparse_equilibrium import MaskValues,Port,grid_edges,graph_solve
from methods.sparse_equilibrium import choose_masks,fit
from ebl.schema.sparse_equilibrium import parse_config
from ebl.schema.base import ConfigError
from experiments.sparse_equilibrium.prepare_data import read_pairs


def test_sparse_coordinate_update_changes_only_declared_cells():
    z=torch.linspace(.1,.9,12).view(3,4)
    idx=torch.tensor([1,7,9]); mask=MaskValues(z,idx)
    opt=torch.optim.Adam(mask.parameters(),lr=.01)
    before=z.clone(); target=torch.ones_like(z)
    for _ in range(3):
        opt.zero_grad();(mask(z)-target).square().sum().backward();opt.step()
    after=mask(z);changed=torch.nonzero((after!=before).flatten()).flatten()
    assert torch.equal(changed,idx) and torch.equal(z,before)
    assert sum(p.numel() for p in mask.parameters())==3


def test_linear_sensitivity_matches_implicit_formula_and_finite_difference():
    b=grid_edges(3).double();g=torch.full((len(b),),.4,dtype=torch.double,requires_grad=True)
    x=torch.randn(5,9,dtype=torch.double); edge=4
    def solve(z):return graph_solve(x,z,b,False)
    v=solve(g);h=torch.eye(9,dtype=torch.double)+b.T@(g[:,None]*b)
    analytic=-(v@b[edge])[:,None]*torch.linalg.solve(h,b[edge])[None]
    tangent=torch.zeros_like(g);tangent[edge]=1
    _,actual=torch.autograd.functional.jvp(solve,g,tangent)
    fd=(solve(g+1e-5*tangent)-solve(g-1e-5*tangent))/(2e-5)
    torch.testing.assert_close(actual,analytic)
    torch.testing.assert_close(fd,analytic,atol=1e-8,rtol=1e-6)
    assert torch.linalg.matrix_rank(solve(g+.1*tangent)-v,atol=1e-10)==1


def test_constrained_solution_satisfies_kkt():
    b=grid_edges(3).double();g=torch.full((len(b),),.4,dtype=torch.double)
    torch.manual_seed(9);x=torch.randn(8,9,dtype=torch.double)
    v=graph_solve(x,g,b,True,200)
    h=torch.eye(9,dtype=torch.double)+b.T@(g[:,None]*b);grad=v@h-x
    assert v.min()>=0 and grad.min()>-1e-10
    assert (v*grad).abs().max()<1e-10


def test_zero_initialized_ports_preserve_function_and_lora_budget():
    x=torch.randn(2,768);y=torch.randn(2,768)
    for kind in ('boundary','lora'):
        p=Port(768,kind)
        torch.testing.assert_close(p.input(x),x)
        torch.testing.assert_close(p.output(x,y),y)
    assert sum(p.numel() for p in Port(768,'lora').parameters())*6==46080


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.z=nn.ParameterList([nn.Parameter(torch.tensor([.2,.4,.6,.8])) for _ in range(12)])
    def configure(self,arm):
        for z in self.z:z.requires_grad_(arm=='full')
        return list(self.z)
    def forward(self,x,labels):
        score=sum(z@x.float()[0] for z in self.z)
        return torch.stack((score,-score))[None]
    def clamp(self):pass


def test_fisher_is_mean_individual_square_and_fit_counts_commands():
    model=Tiny();data=[{'tokens':[1,0,0,0],'label':0},{'tokens':[0,0,3,0],'label':1}]
    selected=choose_masks(model,data,[0,1],12,'fisher',2,2,time.monotonic()+20)
    assert all(idx.tolist()==[2] for idx in selected)
    params=model.configure('full');metrics=[]
    r=fit(model,data,[0,1],params,2,2,.001,3,time.monotonic()+20,metrics.append,analog_count=48)
    assert r['ideal_write_commands']==96 and r['trainable_parameters']==48
    assert metrics[-1]['step']==2


@pytest.mark.parametrize('change',[{'unknown':1},{'mask_size':13},{'updates':129},{'wall_seconds':7000},{'task':'test'}])
def test_strict_bounded_config(change):
    with pytest.raises(ConfigError):parse_config({'input_root':'/tmp/inputs',**change})


def test_pair_identity_uses_whitespace_only_and_records_ambiguity(tmp_path):
    def put(name,fields,rows):
        with (tmp_path/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)
    put('train_orig.tsv',['Sentiment','Text'],[{'Sentiment':'Positive','Text':'a  good film'}])
    put('train_paired.tsv',['Sentiment','Text','batch_id'],[
        {'Sentiment':'Positive','Text':'a good film','batch_id':'1'},
        {'Sentiment':'Negative','Text':'a bad film','batch_id':'1'},
        {'Sentiment':'Positive','Text':'another good film','batch_id':'2'},
        {'Sentiment':'Negative','Text':'another bad film','batch_id':'2'}])
    pairs,excluded=read_pairs(tmp_path,'train')
    assert len(pairs)==1 and pairs[0][1]['Sentiment']=='Positive'
    assert excluded==[{'pair_id':'train/2','reason':'ambiguous_original_identity'}]


def test_physical_drn_reconstructs_native_forward_and_all_adapters(monkeypatch):
    from dataclasses import asdict,replace
    import networks.sparse_equilibrium as n
    from networks.opt.decoder import OptConfig,OptDecoder
    from networks.opt.drn import SourceDecoder,OptDrn
    from networks.resistive.differentiable import DrnSettings
    torch.set_num_threads(1);torch.manual_seed(3)
    config=OptConfig(17,4,12,2,6,32);teacher=OptDecoder(config)
    source=SourceDecoder(config);source.load_state_dict(teacher.state_dict())
    source.start=6;source.register_buffer('gains',torch.ones(12))
    settings=DrnSettings(settling_iterations=24,gradient_iterations=8,activation_checkpointing=True)
    native=OptDrn(source,settings).eval()
    payload={'model':native.state_dict(),'contract':{'solver':asdict(settings)}}
    monkeypatch.setattr(n,'OPT',config)
    monkeypatch.setattr(n,'from_hugging_face',lambda *a:teacher)
    monkeypatch.setattr(n,'pinned_load',lambda *a:payload)
    model=n.PhysicalOpt('/unused','drn','cpu');tokens=torch.tensor([[1,2,3,4]])
    with torch.no_grad():
        expected=native(torch.tensor([[1,2,3,4,5]]))[:,3,[0,1]]
        torch.testing.assert_close(model(tokens,[0,1]),expected,atol=1e-5,rtol=1e-5)
    for kind in ('random','fisher','boundary','bias','lora','full'):
        before={k:v.detach().clone() for k,v in model.base.state_dict().items()}
        params=model.configure(kind,[torch.tensor([0]) for _ in model.z] if kind in ('random','fisher') else None)
        before_out=model(tokens,[0,1]).detach()
        loss=model(tokens,[0,1]).square().sum();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        torch.optim.SGD(params,lr=.0001).step();model.clamp()
        if kind!='bias':
            assert all(torch.equal(before[k],v) for k,v in model.base.state_dict().items())
        assert torch.isfinite(model(tokens,[0,1])).all()
        model.zero_grad(set_to_none=True)


def test_native_graph_smoke_publishes_complete_bundle(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import json
    import ebl.runtime.sparse_equilibrium as runtime
    from ebl.schema.sparse_equilibrium import Config
    graph=runtime.graph_pilot
    monkeypatch.setattr(runtime,'graph_pilot',lambda c,s,d,smoke:graph(c,s,d,True))
    config=Config(task='graph',device='cpu',cpu_threads=1,input_root=str(tmp_path))
    request=SimpleNamespace(spec=config,output_dir=tmp_path/'out',command=['readiness-graph-smoke'])
    assert runtime.run_train(request)==0
    receipt=json.loads((tmp_path/'out/completion.json').read_text())
    run=Path(receipt['run_dir']);summary=json.loads((run/'artifacts/summary.json').read_text())
    assert receipt['status']=='complete' and len(summary['records'])==18
    assert json.loads((run/'result.json').read_text())['status']=='complete'
