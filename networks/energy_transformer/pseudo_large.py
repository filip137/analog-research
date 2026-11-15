"""Deep hybrid model, preserving finite-sweep passive-network forwards."""
import math
import torch
from torch import nn
from networks.energy_transformer.pseudo_drn import PassiveDRN
from networks.energy_transformer.pseudo_learning import compare_vectors


def reduced_settle(a, v, steps, y=None, injection=None):
    """Eliminate key updates from block GS; return penultimate/current y=o-1."""
    w = a / (1 + a)
    drive = w @ v
    if injection is not None:
        drive = drive - injection
    denominator = (.01 + a.sum(-1)).unsqueeze(-1)
    feedback = (a * w).sum(-1).unsqueeze(-1)
    y = torch.zeros_like(drive) if y is None else y
    for _ in range(steps):
        previous = y
        y = (drive + feedback * y) / denominator
    return previous, y


class _AttentionEP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, v, steps):
        _, y = reduced_settle(a, v, steps)
        ctx.save_for_backward(a, v, y)
        ctx.steps = steps
        return y

    @staticmethod
    def backward(ctx, error):
        a, v, y = ctx.saved_tensors
        beta = .005 / error.abs().max().clamp_min(1e-20)
        previous_p, yp = reduced_settle(a, v, ctx.steps, y, beta * error)
        previous_m, ym = reduced_settle(a, v, ctx.steps, y, -beta * error)
        g = a.unsqueeze(-1)
        up = (v.unsqueeze(-3) + g * previous_p.unsqueeze(-2)) / (1 + g)
        um = (v.unsqueeze(-3) + g * previous_m.unsqueeze(-2)) / (1 + g)
        dp = up - yp.unsqueeze(-2)
        dm = um - ym.unsqueeze(-2)
        ga = ((dp + dm) * (dp - dm)).sum(-1) / (4 * beta)
        gv = (-(up - um) / (2 * beta)).sum(-3)
        return ga, gv, None


def attention(a, v, mode, steps):
    return _AttentionEP.apply(a, v, steps) if mode == 'ep' else reduced_settle(a, v, steps)[1]


class Block(nn.Module):
    def __init__(self, width, heads, hidden, depth, mlp):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        for name in ['query', 'key', 'value', 'output']:
            setattr(self, name, nn.Linear(width, width, bias=False))
        self.drn = PassiveDRN(width, hidden)
        self.ff1, self.ff2 = nn.Linear(width, hidden, bias=False), nn.Linear(hidden, width, bias=False)
        self.mlp, self.scale = mlp, 1 / math.sqrt(2 * depth)
        for name in ['query', 'key', 'value', 'output', 'ff1', 'ff2']:
            layer = getattr(self, name)
            nn.init.normal_(layer.weight, std=(.2 if name in ['query', 'key'] else 1.) / math.sqrt(layer.in_features))
        if mlp == 'drn':
            del self.ff1, self.ff2
        else:
            del self.drn

    def forward(self, z, mode, steps):
        n = self.norm1(z)
        def split(t):
            return t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2)
        q, k = split(self.query(n)), split(self.key(n))
        a = torch.exp(-torch.cdist(q, k, p=1) / math.sqrt(q.shape[-1]))
        v = .75 * torch.tanh(split(self.value(n)))
        y = attention(a, v, mode, steps).transpose(1, 2).flatten(-2)
        z = z + self.scale * self.output(y)
        n = self.norm2(z)
        # PassiveDRN has gain32; scaling by width/64 gives gain width/2.
        ff = self.drn(n, mode, steps) * (self.width / 64) if self.mlp == 'drn' else self.ff2(torch.relu(self.ff1(n)))
        return z + self.scale * ff


class Model(nn.Module):
    def __init__(self, seed=20261115, width=256, depth=6, heads=8, hidden=1024, mlp='drn'):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Linear(48, width)
        self.position = nn.Parameter(torch.randn(64, width) * .02)
        self.blocks = nn.ModuleList([Block(width, heads, hidden, depth, mlp) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.decoder = nn.Linear(width, 10)
        nn.init.normal_(self.encoder.weight, std=1 / math.sqrt(48)); nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, std=.02); nn.init.zeros_(self.decoder.bias)

    def forward(self, x, mode='ep', steps=32):
        z = self.encoder(x) + self.position
        for block in self.blocks:
            z = block(z, mode, steps)
        return self.decoder(self.norm(z).mean(1))


def gradient_probe(model, x, target, steps=32):
    """Deterministic eval; all groups and global, with an independent longer cap."""
    was_training = model.training
    model.eval()
    parameters = tuple(model.parameters())
    def get(mode, cap):
        y = model(x, mode, cap)
        g = torch.autograd.grad(torch.nn.functional.cross_entropy(y, target), parameters)
        return y.detach(), list(g) + [torch.cat([v.flatten() for v in g])]
    ye, ge = get('ep', steps)
    yb, gb = get('bptt', steps)
    yr, gr = get('bptt', 2 * steps)
    names = [n for n, _ in model.named_parameters()] + ['global']
    groups = {n: {'ep_bptt': compare_vectors(e, b), 'ep_reference': compare_vectors(e, r),
                  'bptt_reference': compare_vectors(b, r)} for n, e, b, r in zip(names, ge, gb, gr)}
    values = [v for row in groups.values() for v in row.values() if v['reference_norm'] > 1e-9]
    result = {'steps': steps, 'groups': groups, 'forward_error': float((ye-yb).abs().max()),
              'reference_forward_error': float((ye-yr).abs().max()),
              'min_cosine': min(v['cosine'] for v in values),
              'max_relative_l2': max(v['relative_l2'] for v in values),
              'pass': all(v['cosine'] >= .99 and v['relative_l2'] <= .05 for v in values)}
    model.train(was_training)
    return result
