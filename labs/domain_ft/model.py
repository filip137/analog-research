"""OPT-125M with PEFT methods for exp-001 (domain fine-tuning prelims).

The decoder mirrors networks.opt.OptDecoder (same state-dict keys, pre-LN, ReLU MLP,
learned positions with offset 2, tied head) but uses scaled_dot_product_attention.
Adapters act on layers 6-11 (the analog MLP blocks) and are output-neutral at init.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ANALOG = tuple(range(6, 12))
SIGMA = (0.0112185391, 0.01129027418)  # sigma_PV(G) = a + b G  (uS)
G_MID, G_SPAN, G_MIN, G_MAX = 48.6, 39.6, 9.0, 88.2


def sigma_pv(g):
    return SIGMA[0] + SIGMA[1] * g


class Attention(nn.Module):
    def __init__(self, d=768, h=12):
        super().__init__()
        self.h, self.hd = h, d // h
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x):
        b, t, d = x.shape

        def heads(v):
            return v.view(b, t, self.h, self.hd).transpose(1, 2)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(o.transpose(1, 2).reshape(b, t, d))


class Layer(nn.Module):
    def __init__(self, d=768, f=3072, h=12):
        super().__init__()
        self.self_attn = Attention(d, h)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, f), nn.Linear(f, d)
        self.final_layer_norm = nn.LayerNorm(d)
        self.ia3 = None
        self.pa_down = self.pa_up = None
        self.cal1_g = self.cal1_b = self.cal2_g = self.cal2_b = None

    def forward(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        h = self.final_layer_norm(x)
        z = self.fc1(h)
        if self.cal1_g is not None:
            z = z * self.cal1_g + self.cal1_b
        a = F.relu(z)
        if self.ia3 is not None:
            a = a * self.ia3
        y = self.fc2(a)
        if self.cal2_g is not None:
            y = y * self.cal2_g + self.cal2_b
        if self.pa_down is not None:
            y = y + 4.0 * self.pa_up(F.relu(self.pa_down(h)))
        return x + y


class Opt(nn.Module):
    OFFSET = 2

    def __init__(self, vocab=50272, d=768, layers=12, f=3072, h=12, positions=2048):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.embed_positions = nn.Embedding(positions + self.OFFSET, d)
        self.layers = nn.ModuleList(Layer(d, f, h) for _ in range(layers))
        self.final_layer_norm = nn.LayerNorm(d)

    def hidden(self, tokens):
        pos = torch.arange(tokens.shape[1], device=tokens.device) + self.OFFSET
        x = self.embed_tokens(tokens) + self.embed_positions(pos)
        for layer in self.layers:
            x = layer(x)
        return x

    def head(self, x):
        return self.final_layer_norm(x) @ self.embed_tokens.weight.T


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, generator):
        super().__init__()
        self.base = base
        out, inp = base.weight.shape
        self.scale = alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, inp).uniform_(-inp ** -0.5, inp ** -0.5, generator=generator)
                                   .to(base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(out, r, device=base.weight.device))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.T @ self.lora_b.T) * self.scale

    def delta(self):
        return (self.lora_b @ self.lora_a) * self.scale


# --- loading ------------------------------------------------------------------------


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def opt_state(path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    dec = {k.removeprefix("model.decoder."): v.float() for k, v in state.items() if k.startswith("model.decoder.")}
    return dec


def std3_state(opt: dict, source_path, draw_seed: int | None) -> tuple[dict, dict]:
    """Dense OPT state of the std3 crossbar source: clean (exact q) or one CMO programming draw.

    W = output_gain * channel_scale * q for fc1/fc2 of layers 6-11 (per_output_channel_absmax),
    mlp_biases / mlp_norms replace the MLP biases and the MLP LayerNorm, digital.* the rest.
    Deployed: G_a = clamp(48.6 + 39.6 q + sigma_PV(.) e, 9, 88.2), G_r = clamp(48.6 + sigma_PV(48.6) e', 9, 88.2),
    q_hat = (G_a - G_r) / 39.6; one draw per device from a per-matrix generator (seed, matrix name).
    """
    src = torch.load(source_path, map_location="cpu", weights_only=False)
    m = src["model"]
    state = {k.removeprefix("digital."): v.float().clone() for k, v in m.items() if k.startswith("digital.")}
    q = m["q"].float()
    gains, cs = m["output_gains"].float(), m["channel_scales"].float()
    offset, coff, info = 0, 0, {"out_of_range_active": 0, "out_of_range_reference": 0, "devices": 0}
    for k, i in enumerate(ANALOG):
        for j, name in enumerate(("fc1", "fc2")):
            shape = tuple(opt[f"layers.{i}.{name}.weight"].shape)
            n = shape[0] * shape[1]
            qm = q[offset:offset + n].view(shape).double()
            if draw_seed is not None:
                key = f"{draw_seed}:decoder.layers.{i}.{name}"
                g = torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:15], 16))
                ga_star = G_MID + G_SPAN * qm
                ga = ga_star + sigma_pv(ga_star) * torch.randn(shape, generator=g, dtype=torch.float64)
                gr = G_MID + sigma_pv(G_MID) * torch.randn(shape, generator=g, dtype=torch.float64)
                info["out_of_range_active"] += int(((ga < G_MIN) | (ga > G_MAX)).sum())
                info["out_of_range_reference"] += int(((gr < G_MIN) | (gr > G_MAX)).sum())
                info["devices"] += 2 * n
                qm = (ga.clamp(G_MIN, G_MAX) - gr.clamp(G_MIN, G_MAX)) / G_SPAN
            idx = 2 * k + j
            row = cs[coff:coff + shape[0]].double()
            state[f"layers.{i}.{name}.weight"] = (gains[idx].double() * row[:, None] * qm).float()
            state[f"layers.{i}.{name}.bias"] = m[f"mlp_biases.{idx}"].float().clone()
            offset += n
            coff += shape[0]
        state[f"layers.{i}.final_layer_norm.weight"] = m[f"mlp_norms.{k}.weight"].float().clone()
        state[f"layers.{i}.final_layer_norm.bias"] = m[f"mlp_norms.{k}.bias"].float().clone()
    if offset != q.numel():
        raise ValueError("std3 q layout mismatch")
    return state, info


def std3_scales(source_path) -> dict:
    """Per-output-channel weight scale s_j = output_gain * channel_scale of each analog matrix (float64, CPU)."""
    src = torch.load(source_path, map_location="cpu", weights_only=False)
    m = src["model"]
    gains, cs = m["output_gains"].double(), m["channel_scales"].double()
    out, coff = {}, 0
    for k, i in enumerate(ANALOG):
        for j, (name, rows) in enumerate((("fc1", 3072), ("fc2", 768))):
            out[f"layers.{i}.{name}"] = gains[2 * k + j] * cs[coff:coff + rows]
            coff += rows
    if coff != cs.numel():
        raise ValueError("std3 channel_scales layout mismatch")
    return out


@torch.no_grad()
def deploy_targets(w: torch.Tensor, scale: torch.Tensor, tol: float = 1e-6, policy: str = "rescale"):
    """Target q (float64, CPU) and per-channel scale.

    policy "rescale" (P-rescale): channels with max|q| > 1 + tol get s_j' = max_i |W_ji|.
    policy "clip" (P-clip): s_j stays the source scale and q = clamp(W / s_j, -1, 1).
    """
    wd = w.detach().to("cpu", torch.float64)
    s = scale.to(torch.float64).clone()
    amax = wd.abs().amax(1)
    over = amax > s * (1 + tol)
    info = {"channels": int(s.numel()), "max_abs_q_before": float((amax / scale.double()).max()), "policy": policy}
    if policy == "rescale":
        s[over] = amax[over]
        info.update({"channels_rescaled": int(over.sum()), "weights_clipped": 0})
        return wd / s[:, None], s, info
    if policy != "clip":
        raise ValueError(policy)
    q = wd / s[:, None]
    clipped = q.abs() > 1 + tol
    info.update({"channels_rescaled": 0, "channels_with_clipping": int(over.sum()),
                 "weights_clipped": int(clipped.sum()), "weights": int(q.numel())})
    return q.clamp(-1.0, 1.0), s, info


@torch.no_grad()
def program_weights(w: torch.Tensor, scale: torch.Tensor, cp: float, draw_seed: int | None, name: str,
                    tol: float = 1e-6, policy: str = "rescale") -> tuple[torch.Tensor, dict]:
    """Deploy one analog matrix (exp-002): q = W / s_j with the source's per-channel scale s_j.

    A channel with max|q| > 1 + tol is rescaled to s_j' = max_i |W_ji| (counted). Then each weight is an
    active/reference CMO pair: G_a = clamp(48.6 + 39.6 q + cp sigma_PV(G_a*) e, 9, 88.2),
    G_r = clamp(48.6 + cp sigma_PV(48.6) e', 9, 88.2), q_hat = (G_a - G_r)/39.6, W_hat = s_j' q_hat.
    cp = 0 returns s_j' q exactly (the offline weights). e, e' come from the same per-matrix generator
    (key "<seed>:decoder.<name>") as std3_state, so draws are common across cp levels.
    """
    q, s, info = deploy_targets(w, scale, tol, policy)
    info.update({"devices": 2 * int(q.numel()), "out_of_range_active": 0, "out_of_range_reference": 0})
    if cp > 0:
        if draw_seed is None:
            raise ValueError("cp > 0 needs a draw seed")
        g = torch.Generator().manual_seed(int(hashlib.sha256(f"{draw_seed}:decoder.{name}".encode()).hexdigest()[:15], 16))
        ga_star = G_MID + G_SPAN * q
        ga = ga_star + cp * sigma_pv(ga_star) * torch.randn(q.shape, generator=g, dtype=torch.float64)
        gr = G_MID + cp * sigma_pv(G_MID) * torch.randn(q.shape, generator=g, dtype=torch.float64)
        info["out_of_range_active"] = int(((ga < G_MIN) | (ga > G_MAX)).sum())
        info["out_of_range_reference"] = int(((gr < G_MIN) | (gr > G_MAX)).sum())
        q = (ga.clamp(G_MIN, G_MAX) - gr.clamp(G_MIN, G_MAX)) / G_SPAN
    return (s[:, None] * q).to(w.dtype), info


def build_model(opt_path, base="opt", std3_path=None, device="cuda") -> tuple[Opt, dict, dict]:
    """Returns (model, W0 state on device for the analog MLPs, receipt)."""
    opt = opt_state(opt_path)
    receipt = {"opt_weights": str(opt_path), "opt_sha256": sha256_file(opt_path), "base": base}
    if base == "opt":
        state = opt
    else:
        seed = None if base == "std3_clean" else int(base.removeprefix("std3_d"))
        state, info = std3_state(opt, std3_path, seed)
        receipt.update({"std3_source": str(std3_path), "std3_sha256": sha256_file(std3_path),
                        "draw_seed": seed, "programming": info})
    model = Opt()
    model.load_state_dict(state, strict=True)
    return model.to(device), receipt


def teacher(opt_path, device="cuda") -> Opt:
    model = Opt()
    model.load_state_dict(opt_state(opt_path), strict=True)
    return model.to(device).eval().requires_grad_(False)


# --- methods -------------------------------------------------------------------------


def method_size(method: str) -> tuple[str, int]:
    for prefix in ("lora_fc", "lora_attn", "padapter"):
        if method.startswith(prefix):
            return prefix, int(method[len(prefix):])
    return method, 0


def apply_method(model: Opt, method: str, seed: int = 41) -> list[nn.Parameter]:
    """Freeze everything, install the method's parameters, return the trainable list."""
    model.requires_grad_(False)
    kind, size = method_size(method)
    gen = torch.Generator().manual_seed(seed)
    dev = model.embed_tokens.weight.device
    params: list[nn.Parameter] = []
    if kind == "zs":
        return params
    if kind == "full":
        model.requires_grad_(True)
        return list(model.parameters())
    for i in ANALOG:
        L = model.layers[i]
        a = L.self_attn
        if kind == "mlp":
            params += [L.fc1.weight, L.fc2.weight]
        elif kind == "digital":
            for lin in (a.q_proj, a.k_proj, a.v_proj, a.out_proj):
                params += [lin.weight, lin.bias]
            params += [L.self_attn_layer_norm.weight, L.self_attn_layer_norm.bias,
                       L.final_layer_norm.weight, L.final_layer_norm.bias, L.fc1.bias, L.fc2.bias]
        elif kind == "ln":
            params += [L.self_attn_layer_norm.weight, L.self_attn_layer_norm.bias,
                       L.final_layer_norm.weight, L.final_layer_norm.bias]
        elif kind == "bitfit":
            params += [a.q_proj.bias, a.k_proj.bias, a.v_proj.bias, a.out_proj.bias, L.fc1.bias, L.fc2.bias,
                       L.self_attn_layer_norm.bias, L.final_layer_norm.bias]
        elif kind == "ia3":
            L.ia3 = nn.Parameter(torch.ones(L.fc1.out_features, device=dev))
            params.append(L.ia3)
        elif kind in ("calib", "calib_out"):
            # per-output-channel ADC-style gain/offset (fc1 pre-ReLU and fc2 output; calib_out: fc2 only)
            if kind == "calib":
                L.cal1_g = nn.Parameter(torch.ones(L.fc1.out_features, device=dev))
                L.cal1_b = nn.Parameter(torch.zeros(L.fc1.out_features, device=dev))
                params += [L.cal1_g, L.cal1_b]
            L.cal2_g = nn.Parameter(torch.ones(L.fc2.out_features, device=dev))
            L.cal2_b = nn.Parameter(torch.zeros(L.fc2.out_features, device=dev))
            params += [L.cal2_g, L.cal2_b]
        elif kind == "lora_fc":
            L.fc1 = LoRALinear(L.fc1, size, 2 * size, gen)
            L.fc2 = LoRALinear(L.fc2, size, 2 * size, gen)
            params += [L.fc1.lora_a, L.fc1.lora_b, L.fc2.lora_a, L.fc2.lora_b]
        elif kind == "lora_attn":
            a.q_proj = LoRALinear(a.q_proj, size, 2 * size, gen)
            a.v_proj = LoRALinear(a.v_proj, size, 2 * size, gen)
            params += [a.q_proj.lora_a, a.q_proj.lora_b, a.v_proj.lora_a, a.v_proj.lora_b]
        elif kind == "padapter":
            d = L.fc2.out_features
            L.pa_down = nn.Linear(d, size).to(dev)
            L.pa_up = nn.Linear(size, d).to(dev)
            with torch.no_grad():
                L.pa_down.weight.copy_(torch.empty(size, d).uniform_(-d ** -0.5, d ** -0.5, generator=gen))
                L.pa_down.bias.zero_()
                L.pa_up.weight.zero_()
                L.pa_up.bias.zero_()
            params += list(L.pa_down.parameters()) + list(L.pa_up.parameters())
        else:
            raise ValueError(f"unknown method {method}")
    for p in params:
        p.requires_grad_(True)
    return params


def expected_params(method: str) -> int | None:
    kind, s = method_size(method)
    table = {"zs": 0, "mlp": 28_311_552, "digital": 14_215_680, "ln": 18_432, "bitfit": 50_688, "ia3": 18_432,
             "calib": 46_080, "calib_out": 9_216}
    if kind in table:
        return table[kind]
    if kind == "lora_fc":
        return 6 * 2 * s * (768 + 3072)
    if kind == "lora_attn":
        return 6 * 2 * s * (768 + 768)
    if kind == "padapter":
        return 6 * (768 * s + s + s * 768 + 768)
    return None


def peft_state(model: Opt, method: str) -> dict:
    kind, _ = method_size(method)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: p.detach().cpu() for n, p in model.named_parameters() if n in names}


def analog_weights(model: Opt) -> dict:
    """Effective fc1/fc2 weights of layers 6-11 (LoRA merged)."""
    out = {}
    for i in ANALOG:
        for name in ("fc1", "fc2"):
            m = getattr(model.layers[i], name)
            w = m.weight.detach()
            if isinstance(m, LoRALinear):
                w = w + m.delta().detach()
            out[f"layers.{i}.{name}"] = w
    return out


@torch.no_grad()
def deltaw_stats(w0: dict, w1: dict) -> dict:
    """Section 8 statistics per analog matrix."""
    sig_mid = sigma_pv(G_MID) / G_SPAN
    res = {}
    for k in w0:
        a, b = w0[k].double(), w1[k].double()
        d = b - a
        e = {"rel_norm": float(d.norm() / a.norm())}
        if float(d.norm()) == 0.0:
            res[k] = {**e, "zero": True}
            continue
        sv = torch.linalg.svdvals(d.float().cuda() if torch.cuda.is_available() else d.float()).double().cpu()
        cum = (sv ** 2).cumsum(0) / (sv ** 2).sum()
        for p, tag in ((0.5, "r50"), (0.9, "r90"), (0.99, "r99")):
            e[tag] = int((cum < p).sum()) + 1
        # crossbar units: per output channel scale
        s = a.abs().amax(1, keepdim=True).clamp_min(1e-12)
        dq, q0 = d / s, a / s
        sw = sigma_pv(G_MID + G_SPAN * q0) / G_SPAN
        e2 = dq ** 2
        e["xbar"] = {"rms_dq_over_sigma_mid": float(dq.pow(2).mean().sqrt() / sig_mid),
                     "frac_abs_dq_gt_2sigma_mid": float((dq.abs() > 2 * sig_mid).double().mean()),
                     "frac_abs_dq_gt_sigma_w": float((dq.abs() > sw).double().mean()),
                     "energy_frac_gt_2sigma_mid": float(e2[dq.abs() > 2 * sig_mid].sum() / e2.sum()),
                     "sigma_mid": sig_mid}
        # DRN units: one scale per matrix
        S = float(a.abs().max())
        w0n, dn = a / S, d / S
        g0 = 9.0 + 79.2 * w0n.abs()
        sd = sigma_pv(g0) / 79.2
        smed = sigma_pv(float(g0.median())) / 79.2
        e2 = dn ** 2
        e["drn"] = {"rms_dw_over_sigma_med": float(dn.pow(2).mean().sqrt() / smed),
                    "frac_abs_dw_gt_2sigma_med": float((dn.abs() > 2 * smed).double().mean()),
                    "frac_abs_dw_gt_sigma_drn": float((dn.abs() > sd).double().mean()),
                    "energy_frac_gt_2sigma_med": float(e2[dn.abs() > 2 * smed].sum() / e2.sum()),
                    "sigma_med": smed, "median_G": float(g0.median())}
        res[k] = e
    return res
