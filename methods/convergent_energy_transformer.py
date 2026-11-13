"""Local phase statistics; digital differentiation is verification-only."""
from dataclasses import replace
import numpy as np
from networks.energy_transformer.convergent import PARAMETERS, ports, simulate, loss, force


def statistics(parameters, state, boundary, measured_ports=None):
    v = ports(parameters,state) if measured_ports is None else measured_ports
    return {
        "query": -np.einsum("hij,hja,id->had",v["p"],v["k"],v["g"]),
        "key": -np.einsum("hij,hia,jd->had",v["p"],v["q"],v["g"]),
        "memory": -v["r"].T@v["g"],
        "encoder": -v["g"].T@boundary.inputs,
        "decoder": -v["y"].T@v["g"],
        "position": -v["g"],
    }


def phase_gradient(parameters, boundary, minus, plus, beta, minus_ports=None, plus_ports=None):
    if not np.isfinite(beta) or beta<=0:
        raise ValueError("Expected positive beta")
    low = statistics(parameters,minus,boundary,minus_ports)
    high = statistics(parameters,plus,boundary,plus_ports)
    return {name:(high[name]-low[name])/(2*beta) for name in PARAMETERS}


def eqprop(parameters, boundary, beta, free=None):
    if not np.isfinite(beta) or beta<=0:
        raise ValueError("Expected positive beta")
    free = simulate(parameters,boundary).states[-1] if free is None else free
    minus = simulate(parameters,boundary,-beta,free).states[-1]
    plus = simulate(parameters,boundary,beta,free).states[-1]
    return phase_gradient(parameters,boundary,minus,plus,beta),minus,plus


def update(parameters, gradient, rate):
    if set(gradient)!=set(PARAMETERS) or not np.isfinite(rate) or rate<=0:
        raise ValueError("Expected all parameter gradients and positive learning rate")
    for name in PARAMETERS:
        if np.shape(gradient[name])!=getattr(parameters,name).shape or not np.isfinite(gradient[name]).all():
            raise ValueError(f"Invalid gradient for {name}")
    return replace(parameters, **{name:getattr(parameters,name)-rate*gradient[name] for name in PARAMETERS})


def implicit_gradient(parameters, state, boundary):
    """Independent activation-coordinate energy removes the normalization gauge."""
    import torch
    n,d = parameters.shape
    # Orthonormal columns spanning the centered token subspace.
    raw = np.eye(d)[:,:d-1]-np.eye(d)[:,[-1]]
    basis = torch.tensor(np.linalg.qr(raw)[0],dtype=torch.float64)
    p = {name:torch.tensor(np.array(getattr(parameters,name)),dtype=torch.float64,requires_grad=True) for name in PARAMETERS}
    v = ports(parameters,state)
    coord = np.concatenate(((v["g"]@basis.numpy()).ravel(),v["y"].ravel()))
    coord = torch.tensor(coord,dtype=torch.float64,requires_grad=True)
    g = coord[:n*(d-1)].reshape(n,d-1)@basis.T
    y = coord[n*(d-1):].reshape(parameters.output_shape)
    x = torch.tensor(boundary.inputs,dtype=torch.float64)
    target = torch.tensor(boundary.target,dtype=torch.float64)
    mask = torch.tensor(boundary.mask,dtype=torch.float64)
    q = torch.einsum("had,nd->hna",p["query"],g)
    k = torch.einsum("had,nd->hna",p["key"],g)
    energy = (-d*np.sqrt(parameters.epsilon)*(1-g.square().sum(-1)/d).sqrt().sum()
              -(g*(x@p["encoder"].T+p["position"])).sum()
              -torch.logsumexp(parameters.kappa*(q@k.transpose(-1,-2)),-1).sum()/parameters.kappa
              -.5*torch.relu(g@p["memory"].T).square().sum()
              +.5*y.square().sum()-(y*(g@p["decoder"].T)).sum())
    stationarity, = torch.autograd.grad(energy,coord,create_graph=True)
    hess = torch.stack([torch.autograd.grad(row,coord,retain_graph=True)[0] for row in stationarity])
    min_eig = float(torch.linalg.eigvalsh(hess.detach()).min())
    if min_eig <= 0 or float(stationarity.detach().abs().max())>1e-7:
        raise ValueError("Reference requires a settled local energy minimum")
    cost = .5*(mask*(y-target).square()).sum()
    dc, = torch.autograd.grad(cost,coord,retain_graph=True)
    adjoint = torch.linalg.solve(hess.detach(),dc.detach())
    gradients = torch.autograd.grad(stationarity,tuple(p.values()),grad_outputs=-adjoint)
    return {name:grad.detach().numpy() for name,grad in zip(PARAMETERS,gradients)},min_eig


def finite_difference_directions(parameters, boundary, gradient, step=1e-5):
    """One frozen direction per group, including every coefficient in that group."""
    from scipy.optimize import root
    rng = np.random.default_rng(20261114)
    errors = {}
    for name in PARAMETERS:
        value = getattr(parameters,name)
        direction = rng.normal(size=value.shape)
        direction /= np.linalg.norm(direction)
        costs = []
        for sign in (-1,1):
            changed = replace(parameters,**{name:value+sign*step*direction})
            equilibrium = simulate(changed,boundary).states[-1]
            # FD subtracts nearly equal losses. Polish only this independent
            # reference so finite settling errors are not amplified by 1/step.
            solved = root(lambda s:force(changed,s,boundary),equilibrium,tol=1e-11)
            if not np.isfinite(solved.x).all() or np.max(np.abs(force(changed,solved.x,boundary)))>1e-12:
                raise RuntimeError("Finite-difference reference did not settle")
            equilibrium = solved.x
            costs.append(loss(changed,equilibrium,boundary))
        finite = (costs[1]-costs[0])/(2*step)
        predicted = float(np.sum(gradient[name]*direction))
        errors[name] = {"finite_difference":finite,"implicit":predicted,
                        "relative_error":abs(finite-predicted)/max(abs(predicted),1e-8)}
    return errors
