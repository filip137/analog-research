"""exp-002 deployment of an (offline fine-tuned) std3 crossbar model across programming error c_p.

python labs/domain_ft/deploy.py --root R --dataset pubmed --run runs/pubmed@std3_clean/full__lr0.0003 \
    --cps 0 1 10 --draws 271002 271003 271004
--run omitted = B0 (the std3 base itself). The run's state.pt (trained parameters) is loaded over the std3 clean
base; every analog fc1/fc2 matrix of layers 6-11 is re-derived as q = W / s_j with the source's per-channel scale
(channels with max|q| > 1 are rescaled and counted) and programmed by M.program_weights at c_p; digital parameters
are loaded unchanged. Writes <out>/cp<c>_d<seed>.json (c_p = 0: cp0.json) with dev, test and WikiText-2 metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
import train as T  # noqa: E402


def deployed_state(offline: dict, scales: dict, cp: float, draw: int | None, base_q: dict | None):
    state = dict(offline)
    info = {"channels_rescaled": 0, "devices": 0, "out_of_range_active": 0, "out_of_range_reference": 0,
            "max_abs_q_before": 0.0, "per_matrix": {}}
    for name, s in scales.items():
        w = offline[f"{name}.weight"]
        wh, i = M.program_weights(w, s, cp, draw, name)
        state[f"{name}.weight"] = wh.to(w.device)
        for k in ("channels_rescaled", "devices", "out_of_range_active", "out_of_range_reference"):
            info[k] += i[k]
        info["max_abs_q_before"] = max(info["max_abs_q_before"], i["max_abs_q_before"])
        info["per_matrix"][name] = {"channels_rescaled": i["channels_rescaled"],
                                    "max_abs_q_before": i["max_abs_q_before"]}
    return state, info


@torch.no_grad()
def change_cost(offline: dict, base_state: dict, scales: dict, base_q: dict) -> dict:
    """exp-004 cost accounting (independent of c_p and draw): which programming targets moved versus the base.

    Active device target G* = 48.6 + 39.6 q (q after any channel rescale); moved by |dG*| = 39.6 |q - q_base|,
    counted when > 0, > 1 sigma_PV(G*_new) and > 2 sigma_PV(G*_new) at c_p = 1. Reference devices never move.
    Bytes: dense FP16 of every changed analog weight (2 B each); sparse int32 index + FP16 value (6 B) at the 2 sigma
    threshold; FP16 of every changed digital element (2 B each; includes biases/LNs/embeddings); FP16 of the
    per-channel scales that changed (rescaled channels).
    """
    per, tot = {}, {"weights": 0, "moved_any": 0, "moved_gt_1sigma": 0, "moved_gt_2sigma": 0, "channels_rescaled": 0}
    for name, sc in scales.items():
        q, s, info = M.deploy_targets(offline[f"{name}.weight"], sc)
        g = M.G_MID + M.G_SPAN * q
        dg = M.G_SPAN * (q - base_q[name]).abs()
        sig = M.sigma_pv(g)
        r = {"weights": int(q.numel()), "moved_any": int((dg > 0).sum()), "moved_gt_1sigma": int((dg > sig).sum()),
             "moved_gt_2sigma": int((dg > 2 * sig).sum()), "channels_rescaled": info["channels_rescaled"]}
        per[name] = r
        for k in tot:
            tot[k] += r[k]
    analog_keys = {f"{n}.weight" for n in scales}
    dig_changed = dig_total = 0
    dig_per = {}
    for k, v in offline.items():
        if k in analog_keys:
            continue
        n = int((v != base_state[k]).sum())
        dig_total += v.numel()
        dig_changed += n
        if n:
            dig_per[k] = n
    tot.update({"digital_elements": dig_total, "digital_changed": dig_changed,
                "bytes_dense_fp16_changed_analog": 2 * tot["moved_any"],
                "bytes_sparse_idx32_fp16_2sigma": 6 * tot["moved_gt_2sigma"],
                "bytes_digital_changed_fp16": 2 * dig_changed,
                "bytes_scales_changed_fp16": 2 * tot["channels_rescaled"]})
    return {"total": tot, "per_matrix": per, "digital_changed_per_tensor": dig_per,
            "definition": change_cost.__doc__.strip()}


@torch.no_grad()
def evaluate(model, a, doc, files, dev, teach, offline_model=None) -> dict:
    model.eval()
    r = {}
    if a.dataset in T.TASKS:
        for split, key in (("development", "dev"), ("test", "test")):
            cs = T.ChoiceSet(doc, split)
            s, c = T.eval_rows(model, cs.rows, cs.ctx, dev)
            r[key] = T.choice_metrics(s, c, cs.chars, cs.labels, k=cs.k)
            r[key]["metric"] = r[key]["acc_norm"]
    else:
        r["dev"] = T.lm_eval(model, T.blocks(doc["splits"]["validation"], 128), dev)
        tb = T.blocks(doc["splits"]["test"], 256)
        r["test"] = T.lm_eval(model, tb, dev)
        r["dev"]["metric"], r["test"]["metric"] = r["dev"]["ppl"], r["test"]["ppl"]
        if offline_model is not None:
            r["test"]["kl_to_offline"] = lm_kl(offline_model, model, tb, dev)
    wt = T.load(files["wikitext"])
    r["wikitext_test"] = T.lm_eval(model, T.blocks(wt["splits"]["test"]), dev, teacher=teach, kl_blocks=128)
    return r


@torch.no_grad()
def lm_kl(p_model, q_model, blk, dev, micro=16) -> float:
    """Per-token KL(p || q) over all predicted positions."""
    kl, n = 0.0, 0
    for i in range(0, len(blk), micro):
        t = blk[i: i + micro].to(dev)
        lp = p_model.head(p_model.hidden(t)[:, :-1]).float().log_softmax(-1)
        lq = q_model.head(q_model.hidden(t)[:, :-1]).float().log_softmax(-1)
        kl += float((lp.exp() * (lp - lq)).sum(-1).double().sum())
        n += t[:, 1:].numel()
    return kl / n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--run", type=Path, default=None, help="fine-tuned run dir (relative to root) with state.pt; none = B0")
    p.add_argument("--arm", required=True)
    p.add_argument("--cps", type=float, nargs="+", default=[0.0, 1.0, 10.0])
    p.add_argument("--draws", type=int, nargs="+", default=[271002, 271003, 271004])
    p.add_argument("--std3", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--deadline", type=float, default=0.0)
    a = p.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    run = (a.root / a.run) if a.run else None
    out = a.out or (a.root / "deploy" / a.dataset / a.arm)
    out.mkdir(parents=True, exist_ok=True)
    files = T.data_files(a.root)
    opt_path = a.root / "model" / "pytorch_model.bin"
    model, receipt = M.build_model(opt_path, "std3_clean", a.std3, dev)
    base_q = None
    scales = M.std3_scales(a.std3)
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    base_q = {n: base_state[f"{n}.weight"].double().cpu() / s.double()[:, None] for n, s in scales.items()}
    trained = {}
    if run is not None:
        fin = json.loads((run / "final.json").read_text())
        if fin.get("status") != "done":
            raise RuntimeError(f"run not done: {run}")
        trained = torch.load(run / "state.pt", map_location=dev, weights_only=True)
        receipt["state_sha256"] = M.sha256_file(run / "state.pt")
        missing = set(trained) - set(base_state)
        if missing:
            raise KeyError(f"unexpected keys in state.pt: {sorted(missing)[:5]}")
    offline = {**base_state, **{k: v.to(dev) for k, v in trained.items()}}
    model.load_state_dict(offline, strict=True)
    model.eval().requires_grad_(False)
    cost = change_cost(offline, base_state, scales, base_q)
    print(json.dumps({"cost": cost["total"]}), flush=True)
    offline_model = None
    if a.dataset not in T.TASKS:
        offline_model, _ = M.build_model(opt_path, "std3_clean", a.std3, dev)
        offline_model.load_state_dict(offline, strict=True)
        offline_model.eval().requires_grad_(False)
    teach = M.teacher(opt_path, dev)
    doc = T.load(T.data_files(a.root)[a.dataset])
    common = {"dataset": a.dataset, "arm": a.arm, "run": str(a.run) if a.run else None,
              "run_final": {k: fin.get(k) for k in ("method", "lr", "hwa_noise", "dev_final", "updates_done", "params",
                                                    "train_tokens", "sec_per_update", "wall_s")}
              if run else None,
              "model": receipt, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
              "commit": T.git_commit(), "torch": torch.__version__,
              "law": {"G_active": "48.6 + 39.6 q", "G_reference": 48.6, "sigma_PV": "c_p (0.0112185391 + 0.01129027418 G)",
                      "clamp": [M.G_MIN, M.G_MAX], "scale": "source per-output-channel s_j = gain*channel_scale; "
                      "rescaled to max|W_j| where max|q|>1", "seed_key": "<draw>:decoder.layers.<i>.<fc>"}}
    for cp in a.cps:
        for draw in (a.draws if cp > 0 else [None]):
            name = "cp0" if cp == 0 else f"cp{T.lr_tag(cp)}_d{draw}"
            if (out / f"{name}.json").exists():
                print("SKIP", name, flush=True)
                continue
            if a.deadline and time.time() > a.deadline:
                print("DEADLINE", name, flush=True)
                return 2
            t0 = time.time()
            state, info = deployed_state(offline, scales, cp, draw, base_q)
            model.load_state_dict(state, strict=True)
            if cp == 0:
                info["max_abs_weight_diff_vs_offline"] = max(
                    float((state[f"{n}.weight"] - offline[f"{n}.weight"]).abs().max()) for n in scales)
            del state
            res = evaluate(model, a, doc, files, dev, teach, offline_model)
            rec = {**common, "c_p": cp, "draw": draw, "programming": info, "change_cost": cost, **res,
                   "eval_s": time.time() - t0}
            (out / f"{name}.json").write_text(json.dumps(rec, indent=2))
            print(json.dumps({"deploy": name, "dev": res["dev"]["metric"], "test": res["test"]["metric"],
                              "wikitext_ppl": res["wikitext_test"]["ppl"], "kl": res["wikitext_test"]["teacher_kl"],
                              "rescaled": info["channels_rescaled"], "s": rec["eval_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
