import torch
import pytest
from networks.energy_transformer import fuse
from networks.energy_transformer.fuse_gpu import FuseNetwork, Phase, settle, ep_gradient, Unsettled, NAMES

torch.set_num_threads(1)


def fixture(dtype=torch.float64,kernel='gaussian'):
    m=FuseNetwork(3,3,2,heads=2,dtype=dtype,kernel=kernel)
    gen=torch.Generator().manual_seed(33)
    x=torch.randn(2,3,4,generator=gen,dtype=dtype)*.2
    target=torch.randn(2,3,2,generator=gen,dtype=dtype)*.2
    return m,x,target,torch.ones_like(target)


@pytest.mark.parametrize('kernel',['gaussian','linear'])
def test_explicit_two_rail_circuit_parity(kernel):
    m,x,t,mask=fixture(kernel=kernel)
    net=fuse.Network(fuse.Shape(3,3,heads=2,inputs=4,outputs=2),fuse_kernel=kernel)
    theta=torch.cat([getattr(m,n+'_'+s).flatten() for n in NAMES for s in ('p','n')])
    ph=net.realize(theta)
    bd=net.boundary(x,torch.ones_like(x),t,mask)
    gen=torch.Generator().manual_seed(4)
    v=Phase(*(torch.randn(a.shape,generator=gen,dtype=a.dtype)*.2 for a in m.zero(x).tensors()))
    def expand(v):
        out=x.new_zeros(len(x),net.node_count)
        for name,idx in net.nodes.items():
            kind,i,*rest=name
            a=getattr(v,kind)
            coordinate=rest[0]
            out[:,idx]=a[:,i,coordinate]*(1 if rest[-1]==0 else -1)
        return out
    full=expand(v)
    force=m.force(v,x,t,mask,.005)
    forcefull=expand(Phase(*force))
    for b in range(len(x)):
        args=(bd['source'][b],bd['gate'][b],bd['target'][b],bd['mask'][b],.005)
        torch.testing.assert_close(net.current(ph,full[b],*args),forcefull[b],atol=1e-14,rtol=1e-13)
        torch.testing.assert_close(net.energy(ph,full[b],*args),2*m.energy(v,x,t,mask,.005)[b])
    v=settle(m,x,t,mask,tolerance=1e-12)
    explicit,_=fuse.settle(net,ph,bd)
    torch.testing.assert_close(expand(v),explicit,atol=2e-10,rtol=1e-9)


def implicit(m,x,t,mask,free):
    shapes=[v.shape for v in free.tensors()];sizes=[v.numel() for v in free.tensors()]
    def unpack(s):return Phase(*(a.reshape(sh) for a,sh in zip(s.split(sizes),shapes)))
    s=torch.cat([v.flatten() for v in free.tensors()]).detach().requires_grad_()
    for p in m.parameters():p.requires_grad_(True)
    e=lambda s:m.energy(unpack(s),x,t,mask).sum()
    h=torch.autograd.functional.hessian(e,s)
    loss=.5*((m.gain*unpack(s).o-t).square()*mask).sum()/mask.sum()
    dl,=torch.autograd.grad(loss,s)
    adj=torch.linalg.solve(h,dl).detach()
    de,=torch.autograd.grad(e(s),s,create_graph=True)
    grads=torch.autograd.grad(-(de*adj).sum(),tuple(m.parameters()))
    for p in m.parameters():p.requires_grad_(False)
    return dict(zip(dict(m.named_parameters()),grads)),torch.linalg.eigvalsh(h).min().item()


@pytest.mark.parametrize('kernel',['gaussian','linear'])
def test_energy_descent_and_ep_against_independent_implicit_gradient(kernel):
    m,x,t,mask=fixture(kernel=kernel);trace=[]
    free=settle(m,x,t,mask,tolerance=1e-12,trace=trace)
    assert all(bool((b<=a+1e-13).all()) for a,b in zip(trace,trace[1:]))
    exact,eig=implicit(m,x,t,mask,free)
    assert eig>0
    grad,phases=ep_gradient(m,x,t,mask,beta=.00001,tolerance=1e-13)
    for name in grad:
        torch.testing.assert_close(grad[name],exact[name],atol=2e-7,rtol=5e-5)
    # Independent loss finite difference, including signed and positional maps.
    for name in ('query_p','encoder_n','position_p'):
        p=getattr(m,name);original=p.flatten()[0].item();loss=[]
        for sign in (1,-1):
            p.flatten()[0]=original+sign*1e-5
            v=settle(m,x,t,mask,tolerance=1e-12)
            loss.append((.5*((m.gain*v.o-t).square()*mask).sum()/mask.sum()).item())
        p.flatten()[0]=original
        assert abs((loss[0]-loss[1])/2e-5-exact[name].flatten()[0])<1e-7


def test_fp32_local_statistics_and_fail_closed_settling():
    m,x,t,mask=fixture(torch.float32)
    with pytest.raises(Unsettled):settle(m,x,t,mask,max_steps=1)
    grad,phases=ep_gradient(m,x,t,mask,tolerance=2e-7)
    m64,x64,t64,mask64=fixture()
    m64.load_state_dict(m.state_dict())
    # Compare at identical FP32-rounded data, since RNG sequences vary by dtype.
    free64=settle(m64,x.double(),t.double(),mask.double(),tolerance=1e-12)
    exact,_=implicit(m64,x.double(),t.double(),mask.double(),free64)
    for name in grad:
        a,b=grad[name].double().flatten(),exact[name].flatten()
        assert (a-b).norm()/b.norm()<.05
        assert torch.nn.functional.cosine_similarity(a,b,dim=0)>.99
    assert all(not a.requires_grad for p in phases for a in p.tensors())
