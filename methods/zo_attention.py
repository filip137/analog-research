"""Zeroth-order input errors through the digital attention of the analog region.

For a pre-norm block y = x + A(x), A(x) = Attention(LN(x)), backpropagation
returns dx = dy + J_A(x)^T dy. The residual term stays exact; the attention
correction is replaced by the paired activation-perturbation estimate

    1/m sum_i u_i dy.(A(x + eps u_i) - A(x - eps u_i)) / (2 eps),  u_i ~ N(0, I),

over the whole flattened (sequence, token, feature) input of one microbatch.
eps is declared relative to the RMS of that input. Weights and dy are fixed
during the 2m probes. DRN gradients stay the exact simulator (BPTT) gradients,
so only the attention backward is approximated.

Controls: ``exact`` keeps autograd; ``residual`` drops the attention
correction (dx = dy); ``stop`` passes no error across attention (dx = 0), so
only the last analog MLP, which no attention follows, receives a gradient.

``linearized`` freezes the forward softmax pattern P and backpropagates exactly
through LayerNorm, the value and output projections and the fixed P. It drops
only the Q/K (softmax-derivative) path. ``zo`` with ``control='linearized'``
uses that as a control variate and estimates only the remainder,
b + 1/m sum_i u_i [dy.(A(x+eps u_i) - A(x-eps u_i))/(2 eps) - u_i.b], which
stays unbiased with variance proportional to the squared norm of J^T dy - b.
"""

import torch
from torch.nn import functional as F

MODES = ('exact', 'residual', 'stop', 'linearized', 'zo')


class AttentionBackward:
    """Declared replacement for J_A^T dy, with optional exact-VJP diagnostics."""

    def __init__(self, mode, directions=0, epsilon=1e-3, seed=0, device='cpu', probe_chunk=32, control=None):
        if mode not in MODES:
            raise ValueError(f'Expected attention backward mode in {MODES}.')
        if (mode == 'zo') != (directions > 0) or epsilon <= 0:
            raise ValueError('Expected positive directions and epsilon exactly for zo.')
        if control not in (None, 'linearized') or (control and mode != 'zo'):
            raise ValueError('Expected only a linearized control variate, and only for zo.')
        self.mode, self.directions, self.epsilon, self.control = mode, directions, epsilon, control
        self.probe_chunk = probe_chunk
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.diagnose = False
        self.records = []
        self.attention_forwards = 0

    def correction(self, branch, x, grad):
        if self.mode == 'residual':
            estimate = torch.zeros_like(x)
        elif self.mode == 'stop':
            estimate = -grad
        elif self.mode == 'linearized':
            estimate = linearized_vjp(branch.module, x, grad)
        else:
            base = linearized_vjp(branch.module, x, grad) if self.control else None
            eps = self.epsilon * float(x.square().mean().sqrt())
            estimate = torch.zeros_like(x)
            for start in range(0, self.directions, self.probe_chunk):
                m = min(self.probe_chunk, self.directions - start)
                u = torch.randn((m, *x.shape), device=x.device, generator=self.generator)
                probes = torch.cat((x + eps * u, x - eps * u)).flatten(0, 1)
                out = branch(probes).unflatten(0, (2, m, x.shape[0]))
                slope = ((out[0] - out[1]) * grad).flatten(1).sum(1) / (2 * eps)
                if base is not None:
                    slope = slope - (u * base).flatten(1).sum(1)
                estimate += (slope.view(m, *(1,) * x.ndim) * u).sum(0)
                self.attention_forwards += 2 * m * x.shape[0]
            estimate /= self.directions
            if base is not None:
                estimate += base
        if self.diagnose:
            with torch.enable_grad():
                leaf = x.detach().requires_grad_(True)
                exact, = torch.autograd.grad(branch(leaf), leaf, grad)
            self.records.append({'layer': branch.layer, **_compare(grad, estimate, exact)})
        return estimate


def linearized_vjp(layer, x, grad):
    """J^T dy with the softmax pattern held at its forward value."""
    attention = layer.self_attn
    b, t, d = x.shape
    with torch.enable_grad():
        leaf = x.detach().requires_grad_(True)
        h = layer.self_attn_layer_norm(leaf)
        with torch.no_grad():
            q = (attention.q_proj(h) * attention.head_dim**-0.5).view(b, t, attention.heads, -1).transpose(1, 2)
            k = attention.k_proj(h).view(b, t, attention.heads, -1).transpose(1, 2)
            causal = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
            weights = (q @ k.transpose(-1, -2)).masked_fill(causal, float('-inf')).softmax(-1)
        v = attention.v_proj(h).view(b, t, attention.heads, -1).transpose(1, 2)
        out = attention.out_proj((weights @ v).transpose(1, 2).reshape(b, t, d))
        result, = torch.autograd.grad(out, leaf, grad)
    return result


def _cosine(a, b):
    return float(F.cosine_similarity(a.flatten().double(), b.flatten().double(), 0, 1e-30))


def _compare(grad, estimate, exact):
    return {'incoming_norm': float(grad.norm()), 'exact_correction_norm': float(exact.norm()),
            'estimated_correction_norm': float(estimate.norm()),
            'correction_cosine': _cosine(estimate, exact) if estimate.any() else 0.0,
            'input_error_cosine': _cosine(grad + estimate, grad + exact),
            'input_error_relative_error': float((estimate - exact).norm() / (grad + exact).norm())}


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, branch, estimator):
        ctx.branch, ctx.estimator = branch, estimator
        ctx.save_for_backward(x)
        return branch(x)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        with torch.no_grad():
            return ctx.estimator.correction(ctx.branch, x, grad), None, None


def install(model, estimator):
    """Route the analog region's attention backward through the estimator.

    The first analog layer's attention sees only frozen upstream inputs, so its
    backward never runs; every later one sits between two DRN MLPs.
    """
    uninstall(model)
    if estimator.mode == 'exact':
        return model
    for index, layer in enumerate(model.digital.layers[model.start:], model.start):
        def branch(x, layer=layer):
            return layer.self_attn(layer.self_attn_layer_norm(x))
        branch.layer, branch.module = index, layer

        def attention(x, branch=branch):
            return x + _Attention.apply(x, branch, estimator)
        layer.attention = attention
    return model


def uninstall(model):
    for layer in model.digital.layers[model.start:]:
        layer.__dict__.pop('attention', None)
