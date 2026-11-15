import torch
from networks.energy_transformer.pseudo_learning import attention,exact,settle,residual


def test_relaxation_and_ep_against_exact_and_fd():
    torch.manual_seed(81)
    a=torch.softmax(torch.randn(2,2,3,4,dtype=torch.float64),-1).requires_grad_()
    v=(.5*torch.randn(2,2,4,2,dtype=torch.float64).tanh()).requires_grad_()
    e=torch.randn(2,2,3,2,dtype=torch.float64)
    gradients=[]
    for mode in ('ep','bptt','exact'):
        y=attention(a,v,64,mode)
        gradients.append(torch.autograd.grad((y*e).sum(),(a,v)))
        assert torch.allclose(y,exact(a,v),atol=1e-12,rtol=0)
    for group in zip(*gradients):
        for g in group[1:]: assert torch.allclose(group[0],g,atol=1e-10,rtol=1e-9)
    assert residual(a,v,settle(a,v,64))<1e-12
    change=torch.zeros_like(a);change[0,0,0,0]=1e-5
    fd=((exact(a+change,v)-exact(a-change,v))*e).sum()/2e-5
    assert torch.allclose(fd,gradients[0][0][0,0,0,0],atol=1e-8)


def test_finite_cap_is_real_and_shared_value_gradient_sums_queries():
    torch.manual_seed(82)
    a=torch.full((1,1,3,4),.25,dtype=torch.float64,requires_grad=True)
    v=torch.full((1,1,4,2),.6,dtype=torch.float64,requires_grad=True)
    short=attention(a,v,1,'bptt')
    assert (short-exact(a,v)).abs().max()>.01
    ep=attention(a,v,32,'ep')
    gv=torch.autograd.grad(ep.sum(),v)[0]
    assert torch.allclose(gv,torch.full_like(v,3*.2/(.01+.8)),atol=1e-10)
