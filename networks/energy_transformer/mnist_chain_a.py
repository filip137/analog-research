"""Chain A (bank attention) on MNIST (on-chip-learning #63): equation model for GPU training.

Tokens: 16 patches of 7x7 pixels mapped to +-1, plus a 16-way position one-hot (d_in = 65),
applied to a passive resistor crossbar (conductances 9-88.2 uS, swing s) with d_k key and d_v
value columns. H heads: each has a programmed query voltage per key dimension (bounded to
V_cm +- 0.6 s), a summed anti-bump scorer (fitted cell law), a softmax with total A, and a
pseudoconductance bank (h = 1, h0 = 0.01) whose value anchors are z = 1 + tanh(kappa dV / 2U_T).
Readout: digital linear classifier on the H * d_v bank outputs x_o - 1.
"""
import math
import torch
from torch import nn
from networks.energy_transformer.pseudo_mnist_data import patchify

UT = 8.617333262145e-5 * 300.15
GMIN, GMAX = 9e-6, 88.2e-6
VCM = 1.05
TOKENS, PATCH, D_IN = 16, 7, 65


def inputs(images):
    """uint8 images (B, 28, 28) -> token inputs (B, 16, 65)."""
    px = patchify(images.float() / 255, PATCH) * 2 - 1
    pos = torch.eye(TOKENS, device=images.device).expand(px.shape[0], -1, -1)
    return torch.cat([px, pos], -1)


class ChainA(nn.Module):
    def __init__(self, heads=2, d_k=4, d_v=8, scale=.5, s_eff=75., kappa_cell=.79, kappa=.7911, amp=.57, h0=.01, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: nn.Parameter(torch.randn(*s, generator=g) * .5)
        self.Kp, self.Km, self.Vp, self.Vm = r(d_k, D_IN), r(d_k, D_IN), r(d_v, D_IN), r(d_v, D_IN)
        self.Q = r(heads, d_k)
        self.g = nn.Parameter(torch.zeros(heads))
        self.readout = nn.Linear(heads * d_v, 10)
        self.heads, self.d_k, self.d_v, self.scale = heads, d_k, d_v, scale
        self.s_eff, self.kc, self.kappa, self.amp, self.h0 = s_eff, kappa_cell, kappa, amp, h0

    @staticmethod
    def cond(p):
        return GMIN + (GMAX - GMIN) * torch.sigmoid(p)

    def weights(self, pp, pm):
        gp, gm = self.cond(pp), self.cond(pm)
        return (gp - gm) / (gp + gm).sum(-1, keepdim=True)

    def query(self):
        return VCM + self.scale * .6 * torch.tanh(self.Q)                  # (H, d_k) volts

    def temperature(self):
        return nn.functional.softplus(self.g) * 5

    def forward(self, x, detail=False):
        k = VCM + self.scale * x @ self.weights(self.Kp, self.Km).T           # (B, N, d_k)
        vv = self.scale * x @ self.weights(self.Vp, self.Vm).T                # (B, N, d_v) around V_cm
        z = 1 + torch.tanh(self.kappa * vv / (2 * UT))
        d = self.query()[None, :, None, :] - k[:, None]                        # (B, H, N, d_k)
        u = torch.clamp(self.kc * d / (2 * UT), -30, 30)
        c = 4 / self.s_eff
        ch = torch.cosh(u) ** 2
        f = (c * ch / (1 + c * ch)).sum(-1)                                    # (B, H, N)
        s = -self.temperature()[None, :, None] * f
        p = torch.softmax(s, -1)
        a = self.amp * p
        w = a / (1 + a)
        xo = (self.h0 + torch.einsum('bhn,bne->bhe', w, z)) / (self.h0 + w.sum(-1, keepdim=True))
        out = (xo - 1).reshape(x.shape[0], -1)
        logits = self.readout(out)
        if detail:
            return logits, {'k': k, 'vv': vv, 'z': z, 'f': f, 'p': p, 'a': a, 'xo': xo}
        return logits

    def export(self):
        with torch.no_grad():
            return {'heads': self.heads, 'd_k': self.d_k, 'd_v': self.d_v, 'scale': self.scale,
                    'key_gp_S': self.cond(self.Kp).tolist(), 'key_gm_S': self.cond(self.Km).tolist(),
                    'value_gp_S': self.cond(self.Vp).tolist(), 'value_gm_S': self.cond(self.Vm).tolist(),
                    'query_V': self.query().tolist(), 'temperature_g': self.temperature().tolist(),
                    'readout_W': self.readout.weight.tolist(), 'readout_b': self.readout.bias.tolist(),
                    'law': {'s_eff': self.s_eff, 'kappa_cell': self.kc, 'kappa': self.kappa, 'amp': self.amp, 'h0': self.h0}}


def mos_count(heads, d_k, d_v, tokens=TOKENS):
    """Transistors of the SPICE build: cells, mirrors+switches, softmax, value diff pairs and bank per head, holds."""
    per_head = 5 * tokens * d_k + 2 * tokens + 6 * tokens + (tokens + 2) + 3 * tokens * d_v + 1 + (6 * tokens + 5) * d_v + tokens + 2
    return heads * per_head + 2 * tokens * (d_k + d_v) + 2
