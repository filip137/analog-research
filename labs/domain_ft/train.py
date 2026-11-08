"""One exp-001 run: fine-tune OPT-125M with one method on one dataset, then evaluate.

python labs/domain_ft/train.py --dataset mnli --method lora_fc16 --lr 1e-3 --root ~/domain-ft-20261108
Writes <root>/runs/<dataset>[@base]/<method>__lr<lr>[__u<U>]/{config,final,test,retention,deltaw}.json,
metrics.jsonl and peft.pt (<= 1M parameters). Selection code reads only dev numbers (final.json).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402

SEED = 20261108
BATCH = 32
MAX_LEN = 256
TASKS = ("mnli", "rct", "hellaswag", "boolq", "casehold", "sst2")
DOMAINS = ("pubmed", "code", "pubmed_lc")
EVAL_AT = (250, 500, 750, 1000)


def lr_tag(lr: float) -> str:
    return f"{lr:g}"


def run_dir(root: Path, a) -> Path:
    ds = a.dataset if a.base == "opt" else f"{a.dataset}@{a.base}"
    method = a.method + (f"_hwa{lr_tag(a.hwa_noise)}" if getattr(a, "hwa_noise", 0.0) else "")
    name = method if a.method == "zs" else f"{method}__lr{lr_tag(a.lr)}"
    if a.updates != 1000:
        name += f"__u{a.updates}"
    if a.seed != 1:
        name += f"__s{a.seed}"
    return root / "runs" / ds / name


# --- data ------------------------------------------------------------------------------


def data_files(root: Path) -> dict:
    b = root / "data" / "built"
    h = root / "data" / "hwa"
    return {
        "mnli": b / "mnli-opt125m-dft1.pt", "rct": b / "rct-opt125m-dft1.pt", "boolq": b / "boolq-opt125m-dft1.pt",
        "casehold": b / "casehold-opt125m-dft1.pt", "hellaswag": h / "hellaswag-opt125m-v1.pt",
        "sst2": h / "sst2-opt125m-v1.pt", "piqa": h / "piqa-opt125m-v1.pt",
        "pubmed": b / "pubmed-opt125m-dft2.pt", "pubmed_lc": b / "pubmed-opt125m-dft1.pt", "code": b / "code-opt125m-dft1.pt",
        "wikitext": h / "wikitext-2-raw-v1.pt", "synthtest": b / "synthtest-256-dft1.pt",
        "arc_easy": b / "arc_easy-test-opt125m-dft1.pt", "lambada": b / "lambada-2000-opt125m-dft1.pt",
    }


_HASHES: dict = {}


def load(path: Path):
    if str(path) not in _HASHES:
        _HASHES[str(path)] = M.sha256_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


class ChoiceSet:
    """Rows of a fixed-K choice split: BOS + left-truncated context + continuation."""

    def __init__(self, doc: dict, split: str, count: int | None = None):
        d = doc["splits"][split]
        self.k = doc["choices"]
        n = len(d["labels"]) if count is None else min(count, len(d["labels"]))
        co, to = d["context_offsets"].tolist(), d["continuation_offsets"].tolist()
        ci, ti = d["context_ids"], d["continuation_ids"]
        self.rows, self.ctx = [], []
        self.truncated = 0
        for e in range(n):
            conts = [ti[to[e * self.k + j]: to[e * self.k + j + 1]] for j in range(self.k)]
            ctx = ci[co[e]: co[e + 1]]
            room = MAX_LEN - 1 - max(len(c) for c in conts)
            if len(ctx) > room:
                ctx = ctx[len(ctx) - room:]
                self.truncated += 1
            for c in conts:
                self.rows.append(torch.cat([torch.tensor([2]), ctx, c]))
                self.ctx.append(1 + len(ctx))
        self.chars = d["continuation_chars"][: n * self.k].double()
        self.labels = d["labels"][:n].clone()
        self.n = n

    def rows_of(self, examples: list[int]) -> list[int]:
        return [e * self.k + j for e in examples for j in range(self.k)]


class VarChoiceSet:
    """ARC-style rows with a variable number of choices."""

    def __init__(self, doc: dict):
        self.rows, self.ctx = [], []
        ex = doc["example"]
        for r, (e, c) in enumerate(zip(ex, doc["continuations"])):
            ctx = torch.tensor(doc["contexts"][e])
            c = torch.tensor(c)
            room = MAX_LEN - 1 - len(c)
            ctx = ctx[-room:] if len(ctx) > room else ctx
            self.rows.append(torch.cat([torch.tensor([2]), ctx, c]))
            self.ctx.append(1 + len(ctx))
        self.example = torch.tensor(ex)
        self.chars = torch.tensor(doc["chars"]).double()
        self.labels = torch.tensor(doc["labels"])
        self.n = len(self.labels)


def blocks(stream: torch.Tensor, count: int | None = None) -> torch.Tensor:
    n = len(stream) // MAX_LEN
    if count is not None:
        if count > n:
            raise ValueError(f"need {count} blocks, have {n}")
        n = count
    return stream[: n * MAX_LEN].view(n, MAX_LEN)


# --- scoring ---------------------------------------------------------------------------


def pad_rows(rows: list[torch.Tensor], ctx: list[int], device):
    lengths = torch.tensor([len(r) for r in rows])
    width = int(lengths.max())
    tok = torch.full((len(rows), width), 1, dtype=torch.int64)
    for i, r in enumerate(rows):
        tok[i, : len(r)] = r
    tok = tok.to(device, non_blocking=True)
    pos = torch.arange(1, width, device=device)[None, :]
    mask = (pos >= torch.tensor(ctx, device=device)[:, None]) & (pos < lengths.to(device)[:, None])
    return tok, mask


def row_scores(model, rows, ctx, device):
    """Per-row float64 sum of continuation log-probs and token counts (head only at scored positions)."""
    tok, mask = pad_rows(rows, ctx, device)
    h = model.hidden(tok)[:, :-1]
    lp = model.head(h[mask]).float().log_softmax(-1)
    tgt = lp.gather(1, tok[:, 1:][mask][:, None]).squeeze(1)
    idx = torch.arange(mask.shape[0], device=device)[:, None].expand_as(mask)[mask]
    sums = torch.zeros(mask.shape[0], dtype=torch.float64, device=device).index_add(0, idx, tgt.double())
    return sums, mask.sum(1)


@torch.no_grad()
def eval_rows(model, rows, ctx, device, budget=24576):
    order = sorted(range(len(rows)), key=lambda r: -len(rows[r]))
    sums = torch.zeros(len(rows), dtype=torch.float64)
    counts = torch.zeros(len(rows), dtype=torch.int64)
    i = 0
    while i < len(order):
        width = len(rows[order[i]])
        take = max(1, budget // width)
        part = order[i: i + take]
        s, c = row_scores(model, [rows[r] for r in part], [ctx[r] for r in part], device)
        sums[part], counts[part] = s.cpu(), c.cpu()
        i += take
    return sums, counts


def choice_metrics(sums, counts, chars, labels, k=None, example=None) -> dict:
    if example is None:
        s, c, ch = sums.view(-1, k), counts.view(-1, k).double(), chars.view(-1, k)
        raw, norm = s.argmax(1), (s / ch).argmax(1)
        ce = F.cross_entropy((s / c), labels)
    else:  # variable choices
        n = len(labels)
        neg = torch.full((n, 8), -math.inf, dtype=torch.float64)
        negn, negm = neg.clone(), neg.clone()
        slot = torch.zeros(len(sums), dtype=torch.int64)
        seen = {}
        for r, e in enumerate(example.tolist()):
            slot[r] = seen.get(e, 0)
            seen[e] = slot[r] + 1
        neg[example, slot] = sums
        negn[example, slot] = sums / chars
        negm[example, slot] = sums / counts.double()
        raw, norm = neg.argmax(1), negn.argmax(1)
        ce = F.cross_entropy(negm, labels)
    return {"acc": 100 * float((raw == labels).double().mean()),
            "acc_norm": 100 * float((norm == labels).double().mean()),
            "choice_ce": float(ce), "n": len(labels)}


@torch.no_grad()
def lm_eval(model, blk: torch.Tensor, device, teacher=None, kl_blocks=0, micro=16) -> dict:
    nll, n, kl, nk = 0.0, 0, 0.0, 0
    for i in range(0, len(blk), micro):
        t = blk[i: i + micro].to(device)
        logits = model.head(model.hidden(t)[:, :-1]).float()
        lp = logits.log_softmax(-1)
        nll += float(-lp.gather(2, t[:, 1:, None]).double().sum())
        n += t[:, 1:].numel()
        if teacher is not None and i < kl_blocks:
            sub = min(micro, kl_blocks - i)
            tl = teacher.head(teacher.hidden(t[:sub])[:, :-1]).float().log_softmax(-1)
            kl += float((tl.exp() * (tl - lp[:sub])).sum(-1).double().sum())
            nk += t[:sub, 1:].numel()
        del logits, lp
    out = {"ppl": math.exp(nll / n), "nll": nll / n, "tokens": n, "blocks": len(blk)}
    if nk:
        out["teacher_kl"] = kl / nk
        out["kl_blocks"] = kl_blocks
    return out


@torch.no_grad()
def lambada_eval(model, doc, device) -> dict:
    rows, ctx = [], []
    for c, t in zip(doc["contexts"], doc["targets"]):
        c, t = torch.tensor(c), torch.tensor(t)
        room = MAX_LEN - 1 - len(t)
        c = c[-room:] if len(c) > room else c
        rows.append(torch.cat([torch.tensor([2]), c, t]))
        ctx.append(1 + len(c))
    hits = 0
    for i in range(0, len(rows), 128):
        tok, mask = pad_rows(rows[i:i + 128], ctx[i:i + 128], device)
        h = model.hidden(tok)[:, :-1]
        pred = model.head(h[mask]).argmax(-1)
        ok = (pred == tok[:, 1:][mask]).long()
        idx = torch.arange(mask.shape[0], device=device)[:, None].expand_as(mask)[mask]
        wrong = torch.zeros(mask.shape[0], dtype=torch.long, device=device).index_add(0, idx, 1 - ok)
        hits += int((wrong == 0).sum())
    return {"acc": 100 * hits / len(rows), "n": len(rows)}


# --- training --------------------------------------------------------------------------


def lr_at(step: int, total: int, peak: float) -> float:
    warm = max(1, round(0.06 * total))
    if step <= warm:
        return peak * step / warm
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


class Order:
    """Seeded epoch permutations; identical across methods for a dataset."""

    def __init__(self, n: int):
        self.n, self.epoch, self.pos = n, 0, 0
        self.perm = self._perm()

    def _perm(self):
        return torch.randperm(self.n, generator=torch.Generator().manual_seed(SEED + self.epoch)).tolist()

    def take(self, k: int) -> list[int]:
        out = []
        while len(out) < k:
            if self.pos >= self.n:
                self.epoch, self.pos = self.epoch + 1, 0
                self.perm = self._perm()
            m = min(k - len(out), self.n - self.pos)
            out += self.perm[self.pos: self.pos + m]
            self.pos += m
        return out


def git_commit() -> str:
    try:
        return (HERE / "COMMIT").read_text().strip()
    except OSError:
        try:
            return subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1, help="replicate: data order 20261108+(s-1), adapter init 41+(s-1)")
    p.add_argument("--base", default="opt", help="opt | std3_clean | std3_d<seed>")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--std3", type=Path, default=Path.home() / "explore-20261101/inputs/crossbar-std3.pt")
    p.add_argument("--micro", type=int, default=0, help="examples per microbatch (0 = auto)")
    p.add_argument("--deadline", type=float, default=0.0, help="unix time; stop and mark killed_deadline")
    p.add_argument("--no-retention", action="store_true")
    p.add_argument("--no-test", action="store_true")
    p.add_argument("--probe", action="store_true", help="timing probe: no evals, no outputs except final.json")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--hwa-noise", type=float, default=0.0,
                   help="exp-002 B2: additive Gaussian noise std x per-output-channel absmax on the analog fc1/fc2 "
                        "weights of layers 6-11, resampled every update, straight-through (hwa docs/reram_cmo_device.md)")
    p.add_argument("--hwa-seed", type=int, default=31)
    p.add_argument("--save-state", action="store_true", help="save the trained parameters to state.pt (exp-002 deploy)")
    a = p.parse_args(argv)
    global SEED
    SEED = 20261108 + (a.seed - 1)
    out = a.out or run_dir(a.root, a)
    out.mkdir(parents=True, exist_ok=True)
    final = {"status": "running", "dataset": a.dataset, "method": a.method, "lr": a.lr, "base": a.base,
             "updates_planned": a.updates, "hwa_noise": a.hwa_noise}
    t_start = time.time()
    try:
        run(a, out, final)
    except Exception:
        final["status"] = "failed"
        final["traceback"] = traceback.format_exc()
        print(final["traceback"], file=sys.stderr, flush=True)
    final["wall_s"] = time.time() - t_start
    (out / "final.json").write_text(json.dumps(final, indent=2))
    return 0 if final["status"] in ("done", "killed_deadline") else 1


def run(a, out: Path, final: dict):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    files = data_files(a.root)
    opt_path = a.root / "model" / "pytorch_model.bin"
    model, receipt = M.build_model(opt_path, a.base, a.std3, dev)
    w0 = {k: v.clone() for k, v in M.analog_weights(model).items()}
    params = M.apply_method(model, a.method, seed=41 + (a.seed - 1))
    n_params = sum(p.numel() for p in params)
    exp = M.expected_params(a.method)
    if exp is not None and n_params != exp:
        raise ValueError(f"param count {n_params} != expected {exp}")
    final["params"] = n_params
    is_task = a.dataset in TASKS
    is_lm = a.dataset in DOMAINS
    doc = load(files[a.dataset]) if (is_task or is_lm) else None
    cfg = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "commit": git_commit(), "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
           "versions": {"python": platform.python_version(), "torch": torch.__version__},
           "model": receipt, "seed_data": SEED, "seed_adapter": 41 + (a.seed - 1), "batch": BATCH, "max_len": MAX_LEN,
           "optimizer": "AdamW(0.9,0.999,1e-8,wd0) clip1.0 warmup6% cosine->0; fp32 + TF32 matmul",
           "objective": "choice CE over token_mean" if is_task else ("next-token CE" if is_lm else "eval"),
           "data_sha256": {}}
    try:
        import transformers
        cfg["versions"]["transformers"] = transformers.__version__
    except Exception:
        pass
    if doc is not None:
        cfg["data_sha256"][a.dataset] = _HASHES[str(files[a.dataset])]
    metrics = open(out / "metrics.jsonl", "a")

    def log(rec):
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()

    # ---- data
    if is_task:
        train = ChoiceSet(doc, "train")
        devset = ChoiceSet(doc, "development")
        k = train.k
        updates = min(a.updates, (4 * train.n) // BATCH)
        cfg["truncated_train_examples"] = train.truncated
        micro = a.micro or max(1, min(BATCH, 12288 // (k * max(len(r) for r in devset.rows[:400]))))
    elif is_lm:
        train_blocks = blocks(doc["splits"]["train"])
        dev_blocks = blocks(doc["splits"]["validation"], 128)
        updates = min(a.updates, (4 * len(train_blocks)) // BATCH)
        micro = a.micro or 16
    else:
        updates, micro = 0, 0
    if a.method == "zs":
        updates = 0
    cfg["updates"] = updates
    cfg["micro"] = micro
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    final["updates_planned"] = updates

    def dev_eval():
        model.eval()
        if is_task:
            s, c = eval_rows(model, devset.rows, devset.ctx, dev)
            r = choice_metrics(s, c, devset.chars, devset.labels, k=k)
            r["metric"] = r["acc_norm"]
        else:
            r = lm_eval(model, dev_blocks, dev)
            r["metric"] = r["ppl"]
        model.train()
        return r

    curve = []
    if updates == 0 and (is_task or is_lm) and not a.probe:
        r = dev_eval()
        curve.append({"update": 0, **r})
        log({"dev": r, "update": 0})

    # ---- train
    opt = None
    if updates:
        fused = all(p.is_cuda for p in params)
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=fused)
        order = Order(train.n if is_task else len(train_blocks))
        noisy = []
        if a.hwa_noise:
            noisy = [getattr(model.layers[i], n).weight for i in M.ANALOG for n in ("fc1", "fc2")]
            hwa_gen = torch.Generator(device=dev).manual_seed(a.hwa_seed)
            cfg["hwa"] = {"noise_model": "additive_gaussian_weight", "noise_std": a.hwa_noise,
                          "noise_scaling": "output_channel_abs_max (current master)", "resample": "every update",
                          "gradient": "straight-through (backward at the noisy weights, applied to the master)",
                          "master_clamp": "none", "seed": a.hwa_seed, "matrices": "fc1/fc2 of layers 6-11"}
            (out / "config.json").write_text(json.dumps(cfg, indent=2))
        model.train()
        torch.cuda.reset_peak_memory_stats()
        train_time, loss_acc, n_acc = 0.0, 0.0, 0
        tokens_seen = 0
        torch.cuda.synchronize()
        t0 = time.time()
        eval_at = sorted({x * updates // 1000 for x in EVAL_AT} | {updates}) if updates >= 1000 else sorted(
            {max(1, round(updates * f)) for f in (0.25, 0.5, 0.75, 1.0)})
        if a.updates > 1000:
            eval_at = sorted(set(range(250, updates + 1, 250)) | {updates})
        for step in range(1, updates + 1):
            if a.deadline and time.time() > a.deadline:
                final["status"] = "killed_deadline"
                final["updates_done"] = step - 1
                return
            for g in opt.param_groups:
                g["lr"] = lr_at(step, updates, a.lr)
            idx = order.take(BATCH)
            total = 0.0
            clean = []
            with torch.no_grad():
                for w in noisy:
                    clean.append(w.detach().clone())
                    amax = w.abs().amax(1, keepdim=True)
                    w.add_(a.hwa_noise * amax * torch.randn(w.shape, generator=hwa_gen, device=w.device))
            for m0 in range(0, BATCH, micro):
                part = idx[m0: m0 + micro]
                if is_task:
                    rows = train.rows_of(part)
                    tokens_seen += sum(len(train.rows[r]) for r in rows)
                    s, c = row_scores(model, [train.rows[r] for r in rows], [train.ctx[r] for r in rows], dev)
                    scores = (s / c.double()).view(-1, k)
                    loss = F.cross_entropy(scores, train.labels[part].to(dev), reduction="sum") / BATCH
                else:
                    t = train_blocks[part].to(dev)
                    tokens_seen += t.numel()
                    logits = model.head(model.hidden(t)[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), t[:, 1:].reshape(-1),
                                           reduction="sum") / (BATCH * (MAX_LEN - 1))
                    del logits
                loss.backward()
                total += float(loss.detach())
            with torch.no_grad():
                for w, c in zip(noisy, clean):
                    w.copy_(c)
            del clean
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            loss_acc += total
            n_acc += 1
            if not math.isfinite(total):
                raise FloatingPointError(f"nonfinite loss at update {step}")
            if step % 10 == 0:
                log({"update": step, "train_loss": loss_acc / n_acc, "lr": lr_at(step, updates, a.lr),
                     "grad_norm": float(gn)})
                loss_acc, n_acc = 0.0, 0
            if a.probe and step == updates:
                torch.cuda.synchronize()
                train_time += time.time() - t0
            if not a.probe and step in eval_at:
                torch.cuda.synchronize()
                train_time += time.time() - t0
                r = dev_eval()
                curve.append({"update": step, **r})
                log({"dev": r, "update": step, "train_s": train_time})
                print(json.dumps({"update": step, "dev_metric": r["metric"], "s_per_update": train_time / step}),
                      flush=True)
                torch.cuda.synchronize()
                t0 = time.time()
        final["sec_per_update"] = train_time / updates
        final["peak_mem_gb_train"] = torch.cuda.max_memory_allocated() / 2 ** 30
        final["updates_done"] = updates
        final["train_tokens"] = tokens_seen
        final["micro"] = micro
    if a.probe:
        final["status"] = "done"
        return

    # ---- dev summary (selection reads only this)
    if curve:
        better = (lambda x, y: x > y) if is_task else (lambda x, y: x < y)
        best = curve[0]
        for c in curve[1:]:
            if better(c["metric"], best["metric"]):
                best = c
        final["dev_final"] = curve[-1]["metric"]
        final["dev_best"] = best["metric"]
        final["dev_best_update"] = best["update"]
        final["dev_curve"] = [(c["update"], c["metric"]) for c in curve]
        final["dev_final_detail"] = {k2: v for k2, v in curve[-1].items() if k2 != "metric"}
        final["metric"] = "acc_norm" if is_task else "ppl"

    model.eval()
    teach = None
    # ---- test (written separately; never read by selection)
    if not a.no_test and (is_task or is_lm):
        test = {}
        if is_task:
            for split in [s for s in ("test", "test_mismatched") if s in doc["splits"]]:
                cs = ChoiceSet(doc, split)
                s, c = eval_rows(model, cs.rows, cs.ctx, dev)
                test[split] = choice_metrics(s, c, cs.chars, cs.labels, k=cs.k)
                test[split]["truncated_examples"] = cs.truncated
        else:
            test["test"] = lm_eval(model, blocks(doc["splits"]["test"], 256), dev)
        (out / "test.json").write_text(json.dumps(test, indent=2))

    # ---- domain gap (Tier D1)
    if a.dataset == "gap":
        teach = M.teacher(opt_path, dev)
        gap = {}
        for name in ("synthtest", "wikitext", "pubmed", "code"):
            d = load(files[name])
            cfg["data_sha256"][name] = _HASHES[str(files[name])]
            gap[name] = lm_eval(model, blocks(d["splits"]["test"], 256), dev, teacher=teach, kl_blocks=256)
        (out / "gap.json").write_text(json.dumps(gap, indent=2))
        (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # ---- retention
    want_ret = not a.no_retention and (a.method != "zs" or a.dataset == "probes")
    if want_ret:
        t_r = time.time()
        if teach is None:
            teach = M.teacher(opt_path, dev)
        ret = {}
        wt = load(files["wikitext"])
        wb = blocks(wt["splits"]["test"])
        ret["wikitext_test"] = lm_eval(model, wb, dev, teacher=teach, kl_blocks=128)
        piqa = ChoiceSet(load(files["piqa"]), "test")
        s, c = eval_rows(model, piqa.rows, piqa.ctx, dev)
        ret["piqa_test"] = choice_metrics(s, c, piqa.chars, piqa.labels, k=piqa.k)
        arc = VarChoiceSet(load(files["arc_easy"]))
        s, c = eval_rows(model, arc.rows, arc.ctx, dev)
        ret["arc_easy_test"] = choice_metrics(s, c, arc.chars, arc.labels, example=arc.example)
        ret["lambada_2000"] = lambada_eval(model, load(files["lambada"]), dev)
        ret["eval_s"] = time.time() - t_r
        for n in ("wikitext", "piqa", "arc_easy", "lambada"):
            cfg["data_sha256"][n] = _HASHES[str(files[n])]
        (out / "retention.json").write_text(json.dumps(ret, indent=2))
        (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # ---- delta W and PEFT parameters
    kind, _ = M.method_size(a.method)
    if updates and kind in ("full", "mlp", "lora_fc"):
        (out / "deltaw.json").write_text(json.dumps(M.deltaw_stats(w0, M.analog_weights(model)), indent=2))
    if updates and a.save_state:
        torch.save(M.peft_state(model, a.method), out / "state.pt")
        final["state_file"] = str(out / "state.pt")
    if updates and 0 < n_params <= 1_000_000:
        torch.save(M.peft_state(model, a.method), out / "peft.pt")
    final["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    final["status"] = "done"


if __name__ == "__main__":
    sys.exit(main())
