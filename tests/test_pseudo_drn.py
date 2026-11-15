import torch
from networks.energy_transformer.pseudo_drn import drn_settle,drn_residual,_DRNEP

def test_passive_drn_ep_bptt_and_finite_difference():
    torch.manual_seed(12)
    u=torch.tensor([[.5,-.4,.8,-.6],[-.3,.7,-.8,.9]],dtype=torch.float64,requires_grad=True)
    g1=(.1+torch.rand(4,5,dtype=torch.float64)*.2).requires_grad_()
    g2=(.1+torch.rand(5,4,dtype=torch.float64)*.2).requires_grad_()
    e=torch.randn(2,4,dtype=torch.float64)
    h,o=drn_settle(u,g1,g2,64)
    assert (h==0).any() and (h>0).any()
    assert drn_residual(u,g1,g2,(h,o))<1e-12
    b=torch.autograd.grad((o*e).sum(),(u,g1,g2))
    a=torch.autograd.grad((_DRNEP.apply(u,g1,g2,64)*e).sum(),(u,g1,g2))
    for aa,bb in zip(a,b):assert torch.allclose(aa,bb,rtol=1e-7,atol=1e-9)
    change=torch.zeros_like(g2);change[0,0]=1e-5
    fd=((drn_settle(u,g1,g2+change,64)[1]-drn_settle(u,g1,g2-change,64)[1])*e).sum()/2e-5
    assert torch.allclose(fd,a[2][0,0],rtol=1e-6,atol=1e-8)

def test_l1_score_backward_sign_rule():
    torch.manual_seed(17)
    q=torch.randn(2,3,4,dtype=torch.float64,requires_grad=True);k=torch.randn(2,5,4,dtype=torch.float64,requires_grad=True)
    error=torch.randn(2,3,5,dtype=torch.float64)
    qg,kg=torch.autograd.grad((-torch.cdist(q,k,p=1)*error).sum(),(q,k))
    sign=(q[:,:,None,:]-k[:,None,:,:]).sign()
    assert torch.allclose(qg,-(error[:,:,:,None]*sign).sum(2))
    assert torch.allclose(kg,(error[:,:,:,None]*sign).sum(1))

def test_softmax_current_backward_then_l1_sign_gates():
    torch.manual_seed(19)
    q=torch.randn(2,3,4,dtype=torch.float64,requires_grad=True);k=torch.randn(2,5,4,dtype=torch.float64,requires_grad=True)
    a=torch.softmax(-torch.cdist(q,k,p=1)/2,dim=-1);r=torch.randn_like(a)
    qg,kg=torch.autograd.grad((a*r).sum(),(q,k))
    score_error=a*(r-(a*r).sum(-1,keepdim=True))
    sign=(q[:,:,None,:]-k[:,None,:,:]).sign()
    assert torch.allclose(qg,-(score_error[:,:,:,None]*sign).sum(2)/2)
    assert torch.allclose(kg,(score_error[:,:,:,None]*sign).sum(1)/2)

def test_raw_distance_controls_are_bounded_and_contractive():
    from networks.energy_transformer.pseudo_drn import Model
    from networks.energy_transformer.pseudo_learning import exact,settle
    model=Model(score='l1_raw').double()
    x=torch.randn(1,64,48,dtype=torch.float64)
    _,a,v=model.ports(x)
    assert (a>0).all() and (a<=1).all()
    assert (((a*a/(1+a)).sum(-1)/(.01+a.sum(-1)))<.5).all()
    assert torch.allclose(settle(a,v,64)[1]-1,exact(a,v),atol=1e-12,rtol=0)
