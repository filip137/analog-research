import torch
from networks.energy_transformer.pseudo_learning import settle, attention as explicit
from networks.energy_transformer.pseudo_large import reduced_settle, attention, Model


def test_eliminated_gs_matches_explicit_free_nudged_and_gradients():
    torch.manual_seed(204)
    a = torch.rand(2, 2, 4, 5, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, 5, 3, dtype=torch.float64, requires_grad=True) * .2
    for cap in [1, 3, 8, 32]:
        state = settle(a, v, cap)
        previous, y = reduced_settle(a, v, cap)
        assert torch.allclose(y, state[1]-1, atol=2e-15, rtol=0)
        u = (v.unsqueeze(-3)+a.unsqueeze(-1)*previous.unsqueeze(-2))/(1+a.unsqueeze(-1))
        assert torch.allclose(u, state[0]-1, atol=2e-15, rtol=0)
        error = torch.randn_like(y)
        for sign in [-1, 1]:
            old = settle(a,v,cap,state,sign*.005*error)[1]-1
            new = reduced_settle(a,v,cap,y,sign*.005*error)[1]
            assert torch.allclose(old,new,atol=2e-15,rtol=0)
        for mode in ['ep','bptt']:
            old = torch.autograd.grad((explicit(a,v,cap,mode)*error).sum(),(a,v),retain_graph=True)
            new = torch.autograd.grad((attention(a,v,mode,cap)*error).sum(),(a,v),retain_graph=True)
            for x,z in zip(old,new):assert torch.allclose(x,z,atol=1e-11,rtol=1e-9)


def test_large_controls_share_initial_tensors_and_optimizer_resume(tmp_path):
    from copy import deepcopy
    m = Model(width=16, depth=2, heads=2, hidden=32)
    control = Model(width=16, depth=2, heads=2, hidden=32, mlp='digital')
    a,b=m.state_dict(),control.state_dict()
    assert all(torch.equal(a[k],b[k]) for k in a.keys() & b.keys())
    opt = torch.optim.AdamW(m.parameters(),lr=.001)
    rng = torch.Generator().manual_seed(19)
    def step(model, optimizer, generator):
        x=torch.randn(2,64,48,generator=generator);t=torch.randint(10,(2,),generator=generator)
        optimizer.zero_grad();torch.nn.functional.cross_entropy(model(x,'ep',32),t).backward();optimizer.step()
    step(m,opt,rng)
    saved={'model':deepcopy(m.state_dict()),'optimizer':deepcopy(opt.state_dict()),'rng':rng.get_state()}
    step(m,opt,rng);expected=deepcopy(m.state_dict())
    m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);rng.set_state(saved['rng'])
    step(m,opt,rng)
    assert all(torch.equal(m.state_dict()[k],v) for k,v in expected.items())
