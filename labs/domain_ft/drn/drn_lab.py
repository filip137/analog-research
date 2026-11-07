"""DRN stretch for exp-001 (main's item 2d): timing of digital PEFT through the DRN, and D1 on the DRN.

Runs inside an archive of the drn repo at 41ad799 (refactor/training-core), experiment
opt_cmo_drn_drift_regimes.v1, with that experiment's model loading, forward policy and
deployment (P&V) code. Usage (cwd = drn code root):

  python drn_lab.py timing --method padapter64|ln --root ~/domain-ft-20261108 [--view deployed|clean]
  python drn_lab.py gap --root ~/domain-ft-20261108 [--blocks 256]

Model: DRN std3 source (dfb9f22f..., opt_cmo_drn.weights.v1) on the pinned OPT-125M teacher.
Deployed draw: the drift-regimes standard deployment of the config's (seed, realization):
every physical device (four per logical weight) programmed by ReramDriftCells + program_verify
(4 attempts, threshold, 60 s verify), read at release (t = 0 after deployment; no further drift).
Programming targets are clamped to [program_min, program_max] (9, 88.2 uS) as in the experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.getcwd())

import ebl.runtime.opt_cmo_drn.runtime as runtime  # noqa: E402
from ebl.schema.opt_cmo_drn_drift_regimes import parse_config  # noqa: E402
from ebl.runtime.opt_cmo_drn.drift_regimes import physical_layout  # noqa: E402
from hardware.devices.cmo import G_MAX_US  # noqa: E402
from hardware.drift.family import ReramDriftCells, ReramFamily, program_verify  # noqa: E402
from methods.drift_regimes import forward  # noqa: E402
from networks.opt.data import pinned_load  # noqa: E402

DRN_SOURCE = Path.home() / ("analog-research/opt-cmo-drn-first-v1/bounded-recovery-v1/inputs/"
                            "dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9.pt")
CONFIG = "campaigns/drift-regimes/configs/exp-003/screen-R0-271001-ln.json"


def load(root: Path, device, config_path=CONFIG):
    cfg = parse_config(json.loads(Path(config_path).read_text()))
    s = cfg.drift_regimes
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher = runtime.teacher_model(root / "model" / "pytorch_model.bin", cfg)
    source = pinned_load(DRN_SOURCE, s.source_sha256)
    model = runtime.from_mapped(teacher, cfg, source).to(device).eval().requires_grad_(False)
    del source
    teacher = teacher.to(device).eval().requires_grad_(False)
    policy = forward.Policy(model, s.forward_policy)
    policy.mode(False)
    receipt = {"config": config_path, "regime": s.regime, "seed": s.seed, "realization": s.realization,
               "source_sha256": s.source_sha256, "teacher_sha256": cfg.teacher_sha256,
               "forward_policy": {"settle_sweeps": s.forward_policy.settle_sweeps,
                                  "gradient_sweeps": s.forward_policy.gradient_sweeps,
                                  "evaluation_sweeps": s.forward_policy.evaluation_sweeps},
               "solver": {k: getattr(cfg.solver, k) for k in cfg.solver.__dataclass_fields__},
               "law": dict(s.law), "deployment_attempts": s.deployment_attempts,
               "deployment_threshold": s.deployment_threshold, "verify_seconds": s.verify_seconds}
    return cfg, s, model, teacher, policy, receipt


@torch.no_grad()
def deploy(model, s, deploy_device):
    """Stage.deploy of the drift-regimes runtime (no RunStore): released physical uS tensors at t = 0."""
    law = ReramFamily(**s.law)
    targets, names, _ = physical_layout(model)
    targets = [t.to(deploy_device).clamp(law.program_min, law.program_max) for t in targets]
    t0 = time.time()
    cells = [ReramDriftCells(t, law=law, seed=f"{s.seed}/{s.realization}", name=n) for t, n in zip(targets, names)]
    stats = {"reset_attempts": 0, "verify_reads": 0, "capped": 0}
    for c in cells:
        n = c.target.numel()
        d = program_verify(c, torch.arange(n, device=c.target.device), c.target.flatten(),
                           attempts=s.deployment_attempts, threshold=s.deployment_threshold,
                           verify_seconds=s.verify_seconds, first_written=True)
        for k in stats:
            stats[k] += int(d[k])
    release = max(c.clock for c in cells)
    physical = [c.value(release).to(model.blocks[0].u1.device) for c in cells]
    err = [float((p.to(t.device) - t).double().square().mean().sqrt()) for p, t in zip(physical, targets)]
    stats.update({"devices": sum(t.numel() for t in targets), "release_seconds": float(release),
                  "deploy_wall_s": time.time() - t0, "error_rms_us_per_matrix": err})
    del cells
    view = forward.bind(physical)
    forward.check_view(model, view)
    return view, stats


def blocks_of(path: Path, split: str, count: int | None, length=256):
    d = torch.load(path, map_location="cpu", weights_only=False)
    s = d["splits"][split]
    n = len(s) // length if count is None else count
    return s[: n * length].view(n, length)


# --- digital PEFT on the DRN suffix --------------------------------------------------


class PAdapter(nn.Module):
    def __init__(self, width, d=768, gen=None):
        super().__init__()
        self.down, self.up = nn.Linear(d, width), nn.Linear(width, d)
        with torch.no_grad():
            self.down.weight.copy_(torch.empty(width, d).uniform_(-d ** -0.5, d ** -0.5, generator=gen))
            self.down.bias.zero_()
            self.up.weight.zero_()
            self.up.bias.zero_()

    def forward(self, h):
        return 4.0 * self.up(F.relu(self.down(h)))


def suffix_logits(model, hidden, view, adapters=None):
    """forward.suffix_logits plus an optional parallel adapter on each DRN MLP (input = MLP input)."""
    for k, i in enumerate(range(model.start, len(model.digital.layers))):
        layer = model.digital.layers[i]
        hidden = layer.attention(hidden)
        x = layer.final_layer_norm(hidden)
        y = model.blocks[k](x, None if view is None else view[k])
        if adapters is not None:
            y = y + adapters[k](x)
        hidden = hidden + y
    return model.digital.logits(hidden[:, :-1])


def timing(a):
    dev = torch.device("cuda")
    if a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    cfg, s, model, teacher, policy, receipt = load(a.root, dev)
    view, dstats = (None, None)
    if a.view == "deployed":
        view, dstats = deploy(model, s, dev if a.deploy_device == "cuda" else torch.device("cpu"))
    gen = torch.Generator().manual_seed(41)
    adapters = None
    if a.method == "padapter64":
        adapters = nn.ModuleList(PAdapter(64, gen=gen) for _ in range(len(model.blocks))).to(dev)
        params = list(adapters.parameters())
        lr = a.lr or 3e-4
    elif a.method == "ln":
        params = []
        for i in range(model.start, len(model.digital.layers)):
            L = model.digital.layers[i]
            params += list(L.self_attn_layer_norm.parameters()) + list(L.final_layer_norm.parameters())
        lr = a.lr or 3e-3
    else:
        raise ValueError(a.method)
    for p in params:
        p.requires_grad_(True)
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    data = blocks_of(a.root / "data/built/pubmed-opt125m-dft2.pt", "train", None)
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(20261108))
    policy.mode(True)
    torch.cuda.reset_peak_memory_stats()
    losses, times = [], []
    for step in range(a.updates):
        torch.cuda.synchronize()
        t0 = time.time()
        idx = order[step * a.batch:(step + 1) * a.batch]
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for part in idx.split(a.micro):
            tok = data[part].to(dev)
            with torch.no_grad():
                h = model.analog_input(tok)
            logits = suffix_logits(model, h, view, adapters)
            loss = F.cross_entropy(logits.flatten(0, 1), tok[:, 1:].flatten(), reduction="sum") / (a.batch * 255)
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(total)
        print(json.dumps({"update": step + 1, "loss": total, "s": times[-1]}), flush=True)
    policy.mode(False)
    steady = times[1:] if len(times) > 1 else times
    out = {"method": a.method, "params": n_params, "view": a.view, "batch": a.batch, "micro": a.micro,
           "seq_len": 256, "lr": lr, "updates": a.updates, "sec_per_update": sum(steady) / len(steady),
           "sec_per_update_incl_first": sum(times) / len(times), "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
           "loss_first": losses[0], "loss_last": losses[-1], "objective": "next-token CE on pubmed (PubMedQA) train blocks",
           "training_policy": "settle_sweeps no-graph + gradient_sweeps differentiated (Policy.mode(True))",
           "deployment": dstats, "receipt": receipt, "mem_fraction": a.mem_fraction}
    write(a, f"timing_{a.method}_{a.view}" + (f"_b{a.batch}" if a.batch != 32 else ""), out)


@torch.no_grad()
def gap(a):
    dev = torch.device("cuda")
    if a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    cfg, s, model, teacher, policy, receipt = load(a.root, dev)
    t0 = time.time()
    view, dstats = deploy(model, s, dev if a.deploy_device == "cuda" else torch.device("cpu"))
    corpora = {"synthtest": "data/built/synthtest-256-dft1.pt", "wikitext": "data/hwa/wikitext-2-raw-v1.pt",
               "pubmed": "data/built/pubmed-opt125m-dft2.pt", "code": "data/built/code-opt125m-dft1.pt"}
    res = {"deployment": dstats, "receipt": receipt, "blocks": a.blocks,
           "evaluation_sweeps": s.forward_policy.evaluation_sweeps, "corpora": {}}
    for name, rel in corpora.items():
        tok = blocks_of(a.root / rel, "test", a.blocks)
        prefix = forward.prefix_states(model, teacher, tok, dev)
        row = {}
        for label, v in (("drn_clean", None), ("drn_deployed", view)):
            t1 = time.time()
            m = forward.evaluate(model, teacher, prefix, v, policy, microbatch=a.micro)
            m["wall_s"] = time.time() - t1
            row[label] = m
        # teacher (OPT-FP32) perplexity on the same blocks
        ce = cnt = 0.0
        for part in prefix.split(a.micro):
            out = forward.teacher_logits(teacher, model.start, part.teacher).flatten(0, 1)
            ce += float(F.cross_entropy(out, part.tokens[:, 1:].flatten(), reduction="sum"))
            cnt += out.shape[0]
        row["opt_fp32"] = {"cross_entropy": ce / cnt, "perplexity": math.exp(ce / cnt), "teacher_kl": 0.0}
        row["deployed_minus_clean_ppl"] = row["drn_deployed"]["perplexity"] - row["drn_clean"]["perplexity"]
        row["deployed_minus_clean_kl"] = row["drn_deployed"]["teacher_kl"] - row["drn_clean"]["teacher_kl"]
        row["clean_minus_opt_ppl"] = row["drn_clean"]["perplexity"] - row["opt_fp32"]["perplexity"]
        res["corpora"][name] = row
        print(json.dumps({name: {k: (v["perplexity"], v["teacher_kl"]) for k, v in row.items() if isinstance(v, dict)}}),
              flush=True)
        del prefix
    res["wall_s"] = time.time() - t0
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    write(a, "gap" + (f"_b{a.blocks}" if a.blocks != 256 else ""), res)


def write(a, name, obj):
    out = a.root / "runs" / ("drn" if not a.smoke else "drn_smoke") / name
    out.mkdir(parents=True, exist_ok=True)
    obj["status"] = "done"
    (out / "final.json").write_text(json.dumps(obj, indent=2))
    print("WROTE", out / "final.json", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("timing", "gap"))
    p.add_argument("--root", type=Path, default=Path.home() / "domain-ft-20261108")
    p.add_argument("--method", default="padapter64")
    p.add_argument("--view", default="deployed", choices=("deployed", "clean"))
    p.add_argument("--deploy-device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--blocks", type=int, default=256)
    p.add_argument("--mem-fraction", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    timing(a) if a.mode == "timing" else gap(a)


if __name__ == "__main__":
    main()
