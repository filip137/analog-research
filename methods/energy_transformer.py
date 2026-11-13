"""Phase-correlation EqProp with an ideal software programming controller."""
from dataclasses import replace
import numpy as np
from networks.energy_transformer.circuit import (
    Circuit, equilibrium_reference, perturb, simulate,
)

PARAMETERS = ("K", "W", "a", "b", "c")


def statistics(parameters, state, activation_override=None):
    """Local correlations; overrides allow actual SPICE activation readouts."""
    v, prob, r = (parameters.activations(state) if activation_override is None
                  else activation_override)
    return {"K": -np.outer(prob, v), "W": -np.outer(r, v),
            "a": parameters.a-v, "b": -prob, "c": -r}


def phase_gradient(parameters, left, right, denominator, left_activations=None,
                   right_activations=None):
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError("Phase difference needs a nonzero finite denominator")
    low = statistics(parameters, left, left_activations)
    high = statistics(parameters, right, right_activations)
    return {name: (high[name]-low[name])/denominator for name in PARAMETERS}


def eqprop(circuit, target, beta, centered=True):
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("EqProp beta must be positive and finite")
    # All trajectories are physical KCL equilibrations; no trajectory autograd.
    left_nudge = -beta if centered else 0.0
    left = simulate(circuit, target, left_nudge).states[-1]
    right = simulate(circuit, target, beta).states[-1]
    gradient = phase_gradient(circuit.parameters, left, right,
                              2*beta if centered else beta)
    return gradient, left, right


def implicit_gradient(parameters, state, target):
    """Differentiate a scalar reduced energy; solve its adjoint Hessian system."""
    import torch
    p = parameters
    v = torch.tensor(p.split(state)[0], dtype=torch.float64, requires_grad=True)
    tensors = {name: torch.tensor(getattr(p, name), dtype=torch.float64, requires_grad=True)
               for name in PARAMETERS}
    K, W, a, b, c = (tensors[name] for name in PARAMETERS)
    reduced = (0.5*((v-a)**2).sum()
               -torch.logsumexp(p.kappa*(K@v+b), dim=0)/p.kappa
               -0.5*torch.relu(W@v+c).square().sum())
    force, = torch.autograd.grad(reduced, v, create_graph=True)
    H = torch.stack([torch.autograd.grad(row, v, retain_graph=True)[0] for row in force])
    z = torch.linalg.solve(H.detach(), v.detach()-torch.tensor(target, dtype=torch.float64))
    grads = torch.autograd.grad(force, tuple(tensors.values()), grad_outputs=-z)
    return {name: grad.detach().numpy() for name, grad in zip(PARAMETERS, grads)}


def finite_difference_gradient(parameters, target, step=1e-5):
    if step <= 0 or not np.isfinite(step):
        raise ValueError("Expected a positive finite finite-difference step")
    result = {}
    for name in PARAMETERS:
        gradient = np.zeros_like(getattr(parameters, name))
        for index in np.ndindex(gradient.shape):
            losses = []
            for sign in (-1, 1):
                changed = perturb(parameters, name, index, sign*step)
                visible = changed.split(equilibrium_reference(changed, target))[0]
                losses.append(0.5*np.sum((visible-target)**2))
            gradient[index] = (losses[1]-losses[0])/(2*step)
        result[name] = gradient
    return result


def update(circuit, gradient, learning_rate):
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("Expected a positive finite learning rate")
    if set(gradient) != set(PARAMETERS):
        raise ValueError("Expected all declared parameter gradient groups")
    values = {}
    for name in PARAMETERS:
        value = getattr(circuit.parameters, name)
        grad = np.asarray(gradient[name], dtype=float)
        if grad.shape != value.shape or not np.isfinite(grad).all():
            raise ValueError(f"Invalid gradient for {name}")
        values[name] = value-learning_rate*grad
    changed = replace(circuit.parameters, **values)
    if changed.stability_margin() <= 0:
        raise ValueError("Update leaves the declared sufficient stability bound")
    # Rebuild both the four physical devices and their compensation settings.
    return Circuit(changed, circuit.components)


def gradient_errors(estimate, reference):
    result = {}
    for name in PARAMETERS:
        delta = np.asarray(estimate[name])-reference[name]
        norm = float(np.linalg.norm(reference[name]))
        absolute = float(np.linalg.norm(delta))
        result[name] = {"absolute_l2": absolute, "reference_l2": norm,
                        "relative_l2": absolute/norm if norm > 1e-8 else None}
    return result
