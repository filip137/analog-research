"""exp-002 on the DRN: offline fine-tuning of the std3 DRN source, then deployment at c_p.

Runs inside an archive of the drn repo at 41ad799 (refactor/training-core), experiment
opt_cmo_drn_drift_regimes.v1 (cwd = drn code root). Model loading and forward policy are that
experiment's (see drn_lab.py); task data, scoring, batching, schedule and data order come from
labs/domain_ft/train.py (imported, so the semantics are identical to the crossbar arms).

  python drn_ft.py train  --target pubmed|hellaswag --arm full|full_hwa|mlp --lr 3e-4 --root R \
         --drn-source dfb9...pt [--updates 1000] [--micro 8] [--deadline UNIX] [--lr-mode weight|literal]
  python drn_ft.py deploy --target pubmed|hellaswag --arm zs|full|full_hwa|mlp [--lr ..] --root R \
         --drn-source ... --cps 0,1,10 --draws 271002,271003,271004 [--test-examples N]
  python drn_ft.py pvcheck --root R --drn-source ...   (gaussian endpoint vs drift-regimes program_verify)

Arms (all start from the DRN std3 source dfb9f22f..., an opt_cmo_drn.weights.v1 payload):
  full      every parameter: digital (embeddings, attention, LNs, final LN, all 12 layers), DRN
            calibration (bias1, bias2, gains, analog-layer final LNs) and analog masters u1/u2;
  full_hwa  full + the DRN std3 HWA noise (methods.hwa.opt.hardware_view, noise_std 0.03 in normalized
            master units = constant 1.188 uS per physical device, fresh every update, held across the
            update's microbatches, straight-through to the masters, noisy G clamped at >= 0);
  mlp       analog masters only (OptDrn.parameter_groups()[0]).
Masters are clamped to [-1, 1] after every step (OptDrn.clamp_masters, the DRN convention) in all arms.

Learning rate (--lr-mode weight, default): the crossbar rate is in effective-weight units. One master
step du moves the logical weight of its 4-device group by S*du, and W_eff = gain*w, so each master
matrix k gets lr / max|W_eff,k| (MASTER_SCALE, from the crossbar std3 source 86522e23...). Digital and
calibration parameters share units with the crossbar and take lr unchanged. --lr-mode literal uses lr
for every parameter.

Deployment: masters -> physical targets G* = 48.6 + 39.6 u uS with drift_regimes.physical_layout
(4 devices per logical weight), each device G = clamp(G* + c_p sigma_PV(G*) xi, 9, 88.199997),
sigma_PV(G) = 0.0112185391 + 0.01129027418 G, xi ~ N(0,1) from a CPU float64 generator per matrix
seeded by sha256("<draw>:decoder.layers.<i>.fc<j>") (the crossbar lab's key). c_p = 0 binds G* itself
and is bit-identical to the clean (master) forward. No faults, drift or read noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def _import_train(lab_dir: Path):
    sys.path.insert(0, str(lab_dir))
    try:
        import train as T  # noqa: E402  (labs/domain_ft/train.py; imports labs/domain_ft/model.py as `model`)
    finally:
        sys.path.remove(str(lab_dir))
    return T


def _lab_dir() -> Path:
    for cand in (os.environ.get("DFT_LAB_DIR"), HERE.parent, Path.home() / "domain-ft-20261108/code/labs/domain_ft"):
        if cand and (Path(cand) / "train.py").exists():
            return Path(cand)
    raise FileNotFoundError("labs/domain_ft/train.py not found (set DFT_LAB_DIR)")


T = _import_train(_lab_dir())
sys.path.insert(0, os.getcwd())

import ebl.runtime.opt_cmo_drn.runtime as runtime  # noqa: E402
from ebl.schema.opt_cmo_drn_drift_regimes import parse_config  # noqa: E402
from ebl.runtime.opt_cmo_drn.drift_regimes import physical_layout  # noqa: E402
from hardware.devices.cmo import G_MAX_US, G_MIN_US  # noqa: E402
from methods.drift_regimes import forward  # noqa: E402
from methods.hwa.opt import hardware_view  # noqa: E402
from networks.opt.data import pinned_load  # noqa: E402

CONFIG = "campaigns/drift-regimes/configs/exp-003/screen-R0-271001-ln.json"
SIGMA = (0.0112185391, 0.01129027418)
HWA_STD = 0.03
# max |W_eff| = max |output_gain * channel_scale * q| per analog matrix of the crossbar std3 source
# (86522e23...), the effective-weight span of one DRN master unit (see module docstring).
MASTER_SCALE = {
    "decoder.layers.6.fc1": 0.86922, "decoder.layers.6.fc2": 0.04703,
    "decoder.layers.7.fc1": 0.87394, "decoder.layers.7.fc2": 0.03933,
    "decoder.layers.8.fc1": 0.83487, "decoder.layers.8.fc2": 0.05260,
    "decoder.layers.9.fc1": 1.35862, "decoder.layers.9.fc2": 0.04227,
    "decoder.layers.10.fc1": 0.92027, "decoder.layers.10.fc2": 0.06050,
    "decoder.layers.11.fc1": 0.92664, "decoder.layers.11.fc2": 0.22062,
}
TARGET_FILES = {"pubmed": ("data/built/pubmed-opt125m-dft2.pt", "lm"), "hellaswag": ("data/hwa/hellaswag-opt125m-v1.pt", "task")}


def sigma_pv(g):
    return SIGMA[0] + SIGMA[1] * g


# --- model --------------------------------------------------------------------------


def load(a, device):
    cfg = parse_config(json.loads(Path(a.config).read_text()))
    s = cfg.drift_regimes
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher = runtime.teacher_model(a.root / "model" / "pytorch_model.bin", cfg)
    source = pinned_load(a.drn_source, s.source_sha256)
    model = runtime.from_mapped(teacher, cfg, source).to(device).eval().requires_grad_(False)
    del source
    teacher = teacher.to(device).eval().requires_grad_(False)
    policy = forward.Policy(model, s.forward_policy)
    policy.mode(False)
    receipt = {"drn_source": str(a.drn_source), "source_sha256": s.source_sha256, "teacher_sha256": cfg.teacher_sha256,
               "config": a.config,
               "forward_policy": {"name": s.forward_policy.name, "settle_sweeps": s.forward_policy.settle_sweeps,
                                  "gradient_sweeps": s.forward_policy.gradient_sweeps,
                                  "evaluation_sweeps": s.forward_policy.evaluation_sweeps},
               "solver": {k: getattr(cfg.solver, k) for k in cfg.solver.__dataclass_fields__}}
    return model, teacher, policy, receipt


def master_names(model):
    return [f"decoder.layers.{model.start + k}.fc{j + 1}" for k in range(len(model.blocks)) for j in range(2)]


class Wrap:
    """train.py's model interface (hidden/head) on the DRN with an optional held view."""

    def __init__(self, model, view=None):
        self.model, self.view = model, view

    def hidden(self, tokens):
        m = self.model
        h = m.digital.embed(tokens)
        for i, layer in enumerate(m.digital.layers):
            if i < m.start:
                h = layer(h)
            else:
                k = i - m.start
                h = layer.attention(h)
                h = h + m.blocks[k](layer.final_layer_norm(h), None if self.view is None else self.view[k])
        return h

    def head(self, x):
        return self.model.digital.logits(x)


class TeacherWrap:
    def __init__(self, teacher):
        self.t = teacher

    def hidden(self, tokens):
        h = self.t.embed(tokens)
        for layer in self.t.layers:
            h = layer(h)
        return h

    def head(self, x):
        return self.t.logits(x)


def run_dir(root: Path, target: str, arm: str, lr: float) -> Path:
    name = "zs" if arm == "zs" else f"{arm}__lr{T.lr_tag(lr)}"
    return root / "runs" / "drn" / "exp002" / target / name


# --- data and evaluation ---------------------------------------------------------------


class Data:
    def __init__(self, root: Path, target: str, test_examples: int | None = None):
        rel, self.kind = TARGET_FILES[target]
        self.path = root / rel
        self.sha256 = T.M.sha256_file(self.path)
        doc = torch.load(self.path, map_location="cpu", weights_only=False)
        if self.kind == "task":
            self.train = T.ChoiceSet(doc, "train")
            self.dev = T.ChoiceSet(doc, "development")
            self.test = T.ChoiceSet(doc, "test", test_examples)
            self.k = self.train.k
            self.n_train = self.train.n
        else:
            self.train_blocks = T.blocks(doc["splits"]["train"])
            self.dev_blocks = T.blocks(doc["splits"]["validation"], 128)
            self.test_blocks = T.blocks(doc["splits"]["test"], 256)
            self.n_train = len(self.train_blocks)


def eval_split(wrap, data, split, device, budget, offline=None):
    if data.kind == "task":
        cs = data.dev if split == "dev" else data.test
        s, c = T.eval_rows(wrap, cs.rows, cs.ctx, device, budget=budget)
        r = T.choice_metrics(s, c, cs.chars, cs.labels, k=cs.k)
        r["metric"] = r["acc_norm"]
        return r
    blk = data.dev_blocks if split == "dev" else data.test_blocks
    r = lm_eval_pair(wrap, blk, device, offline) if offline is not None else T.lm_eval(wrap, blk, device, micro=8)
    r["metric"] = r["ppl"]
    return r


@torch.no_grad()
def lm_eval_pair(wrap, blk, device, offline, micro=8):
    """Perplexity of `wrap` plus KL(offline || wrap) per token."""
    nll = kl = 0.0
    n = 0
    for i in range(0, len(blk), micro):
        t = blk[i: i + micro].to(device)
        lp = wrap.head(wrap.hidden(t)[:, :-1]).float().log_softmax(-1)
        nll += float(-lp.gather(2, t[:, 1:, None]).double().sum())
        lo = offline.head(offline.hidden(t)[:, :-1]).float().log_softmax(-1)
        kl += float((lo.exp() * (lo - lp)).sum(-1).double().sum())
        n += t[:, 1:].numel()
    return {"ppl": math.exp(nll / n), "nll": nll / n, "tokens": n, "blocks": len(blk), "kl_to_offline": kl / n}


# --- training ----------------------------------------------------------------------------


def trainable(model, arm, lr, lr_mode):
    masters, calib, digital = model.parameter_groups()
    names = master_names(model)
    groups = []
    for u, n in zip(masters, names):
        groups.append({"params": [u], "base_lr": lr / MASTER_SCALE[n] if lr_mode == "weight" else lr, "name": n})
    if arm in ("full", "full_hwa"):
        groups.append({"params": calib, "base_lr": lr, "name": "calibration"})
        groups.append({"params": digital, "base_lr": lr, "name": "digital"})
    elif arm != "mlp":
        raise ValueError(arm)
    model.requires_grad_(False)
    for g in groups:
        for p in g["params"]:
            p.requires_grad_(True)
    return groups


def save_atomic(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def cmd_train(a):
    out = run_dir(a.root, a.target, a.arm, a.lr)
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / "final.json"
    if final_path.exists() and (out / "state.pt").exists() and not a.force:
        if json.loads(final_path.read_text()).get("status") == "done":
            print("SKIP done", out, flush=True)
            return 0
    final = {"status": "running", "arch": "drn", "target": a.target, "arm": a.arm, "lr": a.lr, "lr_mode": a.lr_mode,
             "updates_planned": a.updates}
    t_start = time.time()
    code = 0
    try:
        code = train(a, out, final)
    except Exception:
        final["status"] = "failed"
        final["traceback"] = traceback.format_exc()
        print(final["traceback"], file=sys.stderr, flush=True)
        code = 1
    final["wall_s_this_job"] = time.time() - t_start
    final_path.write_text(json.dumps(final, indent=2))
    return code


def train(a, out: Path, final: dict) -> int:
    dev = torch.device("cuda")
    model, teacher, policy, receipt = load(a, dev)
    data = Data(a.root, a.target)
    groups = trainable(model, a.arm, a.lr, a.lr_mode)
    params = [p for g in groups for p in g["params"]]
    n_params = sum(p.numel() for p in params)
    updates = min(a.updates, (4 * data.n_train) // T.BATCH)
    micro = a.micro
    cfg = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "commit": T.git_commit(), "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
           "versions": {"python": sys.version.split()[0], "torch": torch.__version__}, "model": receipt,
           "seed_data": T.SEED, "batch": T.BATCH, "max_len": T.MAX_LEN, "updates": updates, "micro": micro,
           "params": n_params, "groups": [{"name": g["name"], "lr": g["base_lr"], "params": sum(p.numel() for p in g["params"])}
                                          for g in groups],
           "optimizer": "AdamW(0.9,0.999,1e-8,wd0) clip1.0 (global, mixed units) warmup6% cosine->0; fp32, TF32 off; "
                        "clamp_masters after every step",
           "hwa": ({"recipe": "methods.hwa.opt.hardware_view (DRN std3)", "noise_std_master_units": HWA_STD,
                    "noise_std_us": HWA_STD * 39.6, "resample": "every update, held across microbatches",
                    "gradient": "straight-through to masters"} if a.arm == "full_hwa" else None),
           "objective": "choice CE over token_mean" if data.kind == "task" else "next-token CE",
           "data_sha256": {a.target: data.sha256}, "train_policy": "Policy.mode(True): settle no-graph + gradient sweeps",
           "eval_policy": "Policy.mode(False): evaluation sweeps"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    final.update({"params": n_params, "updates_planned": updates})
    opt = torch.optim.AdamW(groups, lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
    order = T.Order(data.n_train)
    gen = torch.Generator(device=dev).manual_seed(41)
    step0, curve, train_time = 0, [], 0.0
    tokens = [0]
    resume = out / "resume.pt"
    if resume.exists():
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        order.epoch, order.pos = ck["order"]
        order.perm = order._perm()
        gen.set_state(ck["noise_rng"])
        step0, curve, train_time = ck["step"], ck["curve"], ck["train_time"]
        tokens[0] = ck.get("tokens", 0)
        final["resumed_from"] = step0
        print(json.dumps({"resumed": step0}), flush=True)
        del ck
    metrics = open(out / "metrics.jsonl", "a")

    def log(rec):
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()

    def checkpoint(step):
        save_atomic({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": opt.state_dict(),
                     "order": (order.epoch, order.pos), "noise_rng": gen.get_state(), "step": step, "curve": curve,
                     "train_time": train_time, "args": cfg["args"], "tokens": tokens[0]}, resume)

    def dev_eval():
        policy.mode(False)
        r = eval_split(Wrap(model), data, "dev", dev, a.eval_budget)
        return r

    eval_at = sorted({x * updates // 1000 for x in T.EVAL_AT} | {updates}) if updates >= 1000 else sorted(
        {max(1, round(updates * f)) for f in (0.25, 0.5, 0.75, 1.0)})
    if a.eval_at:
        eval_at = sorted({int(x) for x in a.eval_at.split(",")} | {updates})
    if step0 == 0 and a.eval0:
        r = dev_eval()
        curve.append({"update": 0, **r})
        log({"dev": r, "update": 0})
    torch.cuda.reset_peak_memory_stats()
    time0 = train_time
    loss_acc, n_acc = 0.0, 0
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(step0 + 1, updates + 1):
        if a.deadline and time.time() > a.deadline:
            torch.cuda.synchronize()
            train_time += time.time() - t0
            checkpoint(step - 1)
            final.update({"status": "resumable", "updates_done": step - 1, "train_s_total": train_time,
                          "train_tokens": tokens[0],
                          "sec_per_update_this_job": (train_time - time0) / max(1, step - 1 - step0)})
            return 3
        lr_f = T.lr_at(step, updates, 1.0)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lr_f
        idx = order.take(T.BATCH)
        policy.mode(True)
        view = hardware_view(model, HWA_STD, gen)[0] if a.arm == "full_hwa" else None
        wrap = Wrap(model, view)
        total = 0.0
        for m0 in range(0, T.BATCH, micro):
            part = idx[m0: m0 + micro]
            if data.kind == "task":
                rows = data.train.rows_of(part)
                tokens[0] += sum(len(data.train.rows[r]) for r in rows)
                s, c = T.row_scores(wrap, [data.train.rows[r] for r in rows], [data.train.ctx[r] for r in rows], dev)
                scores = (s / c.double()).view(-1, data.k)
                loss = F.cross_entropy(scores, data.train.labels[part].to(dev), reduction="sum") / T.BATCH
            else:
                t = data.train_blocks[part].to(dev)
                tokens[0] += t.numel()
                logits = wrap.head(wrap.hidden(t)[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), t[:, 1:].reshape(-1),
                                       reduction="sum") / (T.BATCH * (T.MAX_LEN - 1))
                del logits
            loss.backward()
            total += float(loss.detach())
        del view, wrap
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.clamp_masters()
        if not math.isfinite(total):
            raise FloatingPointError(f"nonfinite loss at update {step}")
        loss_acc += total
        n_acc += 1
        if step % 10 == 0 or step == updates:
            log({"update": step, "train_loss": loss_acc / n_acc, "lr_factor": lr_f, "grad_norm": float(gn)})
            loss_acc, n_acc = 0.0, 0
        if a.verbose:
            print(json.dumps({"update": step, "loss": total}), flush=True)
        if step in eval_at:
            torch.cuda.synchronize()
            train_time += time.time() - t0
            r = dev_eval()
            curve.append({"update": step, **r})
            log({"dev": r, "update": step, "train_s": train_time})
            print(json.dumps({"update": step, "dev_metric": r["metric"], "train_s_total": train_time}), flush=True)
            if step < updates and a.ckpt:
                checkpoint(step)
            torch.cuda.synchronize()
            t0 = time.time()
    done = updates - step0
    final["train_tokens"] = tokens[0]
    final["sec_per_update_this_job"] = (train_time - time0) / max(1, done) if done else None
    final["peak_mem_gb_train"] = torch.cuda.max_memory_allocated() / 2 ** 30
    final["updates_done"] = updates
    final["micro"] = micro
    final["train_s_total"] = train_time
    if curve:
        is_task = data.kind == "task"
        better = (lambda x, y: x > y) if is_task else (lambda x, y: x < y)
        best = curve[0]
        for c in curve[1:]:
            if better(c["metric"], best["metric"]):
                best = c
        final.update({"dev_final": curve[-1]["metric"], "dev_best": best["metric"], "dev_best_update": best["update"],
                      "dev_curve": [(c["update"], c["metric"]) for c in curve],
                      "dev_final_detail": {k: v for k, v in curve[-1].items() if k != "metric"},
                      "metric": "acc_norm" if is_task else "ppl"})
    save_atomic({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "state.pt")
    if resume.exists():
        resume.unlink()
    final["status"] = "done"
    return 0


# --- deployment --------------------------------------------------------------------------


@torch.no_grad()
def program(model, cp: float, draw: int | None):
    """Physical uS tensors of every device of the 12 DRN matrices at programming-error scale cp."""
    targets, names, _ = physical_layout(model)
    info = {"devices_written": 0, "targets_clamped": 0, "out_of_range_low": 0, "out_of_range_high": 0,
            "z_rms_unclamped": None, "z_rms_targets_15_80us": None, "error_rms_us_per_matrix": {}}
    physical, zsq, zn, zisq, zin = [], 0.0, 0, 0.0, 0
    for t, name in zip(targets, names):
        info["devices_written"] += t.numel()
        info["targets_clamped"] += int(((t < G_MIN_US) | (t > G_MAX_US)).sum())
        if cp == 0:
            p = t  # bit-identical to the clean master forward
        else:
            key = f"{draw}:{name}"
            gen = torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:15], 16))
            tt = t.detach().cpu().double().clamp(G_MIN_US, G_MAX_US)
            xi = torch.randn(tt.shape, generator=gen, dtype=torch.float64)
            g = tt + cp * sigma_pv(tt) * xi
            low, high = g < G_MIN_US, g > G_MAX_US
            info["out_of_range_low"] += int(low.sum())
            info["out_of_range_high"] += int(high.sum())
            g = g.clamp(G_MIN_US, G_MAX_US)
            ok = ~(low | high)
            z = (g - tt) / (cp * sigma_pv(tt))
            zsq += float(z[ok].square().sum())
            zn += int(ok.sum())
            inner = (tt >= 15.0) & (tt <= 80.0)
            zisq += float(z[inner].square().sum())
            zin += int(inner.sum())
            info["error_rms_us_per_matrix"][name] = float((g - tt).square().mean().sqrt())
            p = g.float().to(t.device)
            del tt, xi, g, z
        physical.append(p)
    if zn:
        info["z_rms_unclamped"] = math.sqrt(zsq / zn)
    if zin:
        info["z_rms_targets_15_80us"] = math.sqrt(zisq / zin)
        info["devices_targets_15_80us"] = zin
    view = forward.bind(physical)
    forward.check_view(model, view)
    return view, info


def cmd_deploy(a):
    dev = torch.device("cuda")
    model, teacher, policy, receipt = load(a, dev)
    out = run_dir(a.root, a.target, a.arm, a.lr)
    with torch.no_grad():
        base_targets = [t.detach().cpu().double().clone() for t in physical_layout(model)[0]]
        master_ids = {id(p) for p in model.parameter_groups()[0]}
        master_names = {n for n, p in model.named_parameters() if id(p) in master_ids}
        base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k not in master_names}
    run_final = {}
    if a.arm != "zs":
        run_final = json.loads((out / "final.json").read_text())
        if run_final.get("status") != "done":
            raise RuntimeError(f"run not done: {out}")
        state = out / "state.pt"
        model.load_state_dict(torch.load(state, map_location="cpu", weights_only=True), strict=True)
        receipt["finetuned_state"] = str(state)
        receipt["finetuned_state_sha256"] = T.M.sha256_file(state)
    model.requires_grad_(False)
    cost = change_cost(model, base_targets, base_sd, master_names)
    del base_targets, base_sd
    print(json.dumps({"cost": cost["total"]}), flush=True)
    (out / "deploy").mkdir(parents=True, exist_ok=True)
    data = Data(a.root, a.target, a.test_examples)
    wt_path = a.root / "data/hwa/wikitext-2-raw-v1.pt"
    wt = T.blocks(torch.load(wt_path, map_location="cpu", weights_only=False)["splits"]["test"])
    wt_sha = T.M.sha256_file(wt_path)
    policy.mode(False)
    offline = Wrap(model)
    tw = TeacherWrap(teacher)
    cps = [float(x) for x in a.cps.split(",")]
    draws = [int(x) for x in a.draws.split(",")]
    for cp in cps:
        for draw in ([None] if cp == 0 else draws):
            tag = f"cp{cp:g}" + ("" if draw is None else f"_d{draw}")
            path = out / "deploy" / f"{tag}.json"
            if path.exists() and not a.force:
                print("SKIP", path, flush=True)
                continue
            t0 = time.time()
            view, info = program(model, cp, draw)
            t_prog = time.time() - t0
            wrap = Wrap(model, view)
            row = {"arch": "drn", "target": a.target, "arm": a.arm, "lr": a.lr if a.arm != "zs" else None, "c_p": cp,
                   "draw": draw, "deploy_policy": "rescale", "programming": info, "change_cost": cost,
                   "run_final": {k: run_final.get(k) for k in ("lr", "params", "train_tokens", "updates_done",
                                                               "dev_final", "train_s_total")},
                   "program_s": t_prog, "receipt": receipt,
                   "law": {"sigma_pv": list(SIGMA), "clamp_us": [G_MIN_US, G_MAX_US], "devices_per_weight": 4,
                           "seed_key": "sha256('<draw>:decoder.layers.<i>.fc<j>')[:15] -> CPU float64 generator"},
                   "data_sha256": {a.target: data.sha256, "wikitext": wt_sha}, "host": socket.gethostname(),
                   "gpu": torch.cuda.get_device_name(0), "commit": T.git_commit()}
            t1 = time.time()
            row["dev"] = eval_split(wrap, data, "dev", dev, a.eval_budget)
            row["test"] = eval_split(wrap, data, "test", dev, a.eval_budget,
                                     offline=offline if (data.kind == "lm" and cp != 0) else None)
            if data.kind == "task":
                row["test"]["n_examples_evaluated"] = data.test.n
            row["eval_target_s"] = time.time() - t1
            t2 = time.time()
            row["wikitext_test"] = T.lm_eval(wrap, wt, dev, teacher=tw, kl_blocks=128, micro=8)
            row["eval_wikitext_s"] = time.time() - t2
            row["wall_s"] = time.time() - t0
            path.write_text(json.dumps(row, indent=2))
            print(json.dumps({"tag": tag, "dev": row["dev"]["metric"], "test": row["test"]["metric"],
                              "wt_ppl": row["wikitext_test"]["ppl"], "wt_kl": row["wikitext_test"].get("teacher_kl"),
                              "wall_s": row["wall_s"]}), flush=True)
            del view, wrap
    return 0


@torch.no_grad()
def change_cost(model, base_targets, base_sd, master_names) -> dict:
    """exp-004 cost accounting on the DRN (independent of c_p and draw), per physical device.

    Device target G* (uS) from physical_layout of the fine-tuned masters vs the base model's; moved when
    |dG*| > 0, > sigma_PV(G*_new), > 2 sigma_PV(G*_new) (c_p = 1). DRN matrices keep one fixed scale and
    masters are clamped to [-1, 1] in training, so no rescale/clip arises (deploy_policy reported as "rescale"
    with zero channels rescaled). Bytes: dense FP16 per moved device target (2 B), sparse int32 index + FP16
    (6 B) per device moved > 2 sigma, FP16 per changed digital/calibration element (2 B).
    """
    targets, names, _ = physical_layout(model)
    per, tot = {}, {"devices": 0, "moved_any": 0, "moved_gt_1sigma": 0, "moved_gt_2sigma": 0,
                    "channels_rescaled": 0, "weights_clipped": 0, "frac_clipped": 0.0}
    for t, b, name in zip(targets, base_targets, names):
        t = t.detach().cpu().double()
        dg = (t - b).abs()
        sig = sigma_pv(t.clamp(G_MIN_US, G_MAX_US))
        r = {"devices": int(t.numel()), "moved_any": int((dg > 0).sum()), "moved_gt_1sigma": int((dg > sig).sum()),
             "moved_gt_2sigma": int((dg > 2 * sig).sum())}
        per[name] = r
        for k in ("devices", "moved_any", "moved_gt_1sigma", "moved_gt_2sigma"):
            tot[k] += r[k]
    dig_changed = dig_total = 0
    for k, v in model.state_dict().items():
        if k in master_names:
            continue
        b = base_sd[k]
        dig_total += v.numel()
        dig_changed += int((v.detach().cpu() != b).sum())
    tot.update({"policy": "rescale", "digital_elements": dig_total, "digital_changed": dig_changed,
                "bytes_dense_fp16_changed_analog": 2 * tot["moved_any"],
                "bytes_sparse_idx32_fp16_2sigma": 6 * tot["moved_gt_2sigma"],
                "bytes_digital_changed_fp16": 2 * dig_changed, "bytes_scales_changed_fp16": 0})
    return {"total": tot, "per_matrix": per, "definition": change_cost.__doc__.strip()}


# --- checks ------------------------------------------------------------------------------


@torch.no_grad()
def cmd_pvcheck(a):
    """c_p = 1 gaussian endpoint vs the drift-regimes deployment (program_verify, t = release) on the base."""
    sys.path.insert(0, str(HERE))
    import drn_lab  # noqa: E402
    from ebl.schema.opt_cmo_drn_drift_regimes import parse_config as pc  # noqa: E402
    dev = torch.device("cuda")
    model, teacher, policy, receipt = load(a, dev)
    s = pc(json.loads(Path(a.config).read_text())).drift_regimes
    res = {}
    _, info = program(model, 1.0, 271002)
    res["gaussian_cp1_271002"] = info
    view, stats = drn_lab.deploy(model, s, dev)
    targets, names, _ = physical_layout(model)
    res["program_verify_drift_regimes"] = {"error_rms_us_per_matrix": dict(zip(names, stats["error_rms_us_per_matrix"])),
                                           **{k: v for k, v in stats.items() if k != "error_rms_us_per_matrix"}}
    # equality check: c_p = 0 view == clean master forward
    toks = T.blocks(torch.load(a.root / "data/built/pubmed-opt125m-dft2.pt", map_location="cpu", weights_only=False)["splits"]["validation"], 4).to(dev)
    policy.mode(False)
    v0, _ = program(model, 0.0, None)
    l_clean = Wrap(model).head(Wrap(model).hidden(toks)[:, :-1])
    l_cp0 = Wrap(model, v0).head(Wrap(model, v0).hidden(toks)[:, :-1])
    res["cp0_logits_bit_identical"] = bool(torch.equal(l_clean, l_cp0))
    res["cp0_logits_max_abs_diff"] = float((l_clean - l_cp0).abs().max())
    out = a.root / "runs" / "drn" / "exp002" / "pvcheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("train", "deploy", "pvcheck"))
    p.add_argument("--root", type=Path, default=Path.home() / "domain-ft-20261108")
    p.add_argument("--drn-source", type=Path, required=True)
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--target", default="pubmed", choices=tuple(TARGET_FILES))
    p.add_argument("--arm", default="mlp", choices=("zs", "full", "full_hwa", "mlp"))
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--lr-mode", default="weight", choices=("weight", "literal"))
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--eval-budget", type=int, default=8192, help="tokens per evaluation chunk (choice rows)")
    p.add_argument("--deadline", type=float, default=0.0, help="unix time: checkpoint to resume.pt and exit 3")
    p.add_argument("--no-ckpt", dest="ckpt", action="store_false", help="no resume.pt at dev evaluations")
    p.add_argument("--eval0", action="store_true", help="dev evaluation before training")
    p.add_argument("--eval-at", default="", help="comma list of dev-evaluation updates (default 250/500/750/1000)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--cps", default="0,1,10")
    p.add_argument("--draws", default="271002,271003,271004")
    p.add_argument("--test-examples", type=int, default=None, help="first N test examples (choice task only)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.mode == "train":
        if a.arm == "zs":
            raise SystemExit("zs has no training; use deploy")
        return cmd_train(a)
    if a.mode == "deploy":
        return cmd_deploy(a)
    return cmd_pvcheck(a)


if __name__ == "__main__":
    sys.exit(main())
