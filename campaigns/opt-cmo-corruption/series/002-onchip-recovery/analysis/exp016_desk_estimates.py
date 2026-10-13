"""Desk estimates motivating exp-016: how much fault error low-rank or local pair corrections can cancel.

CPU-only and model-based: simulated CMO random-stuck maps on the pinned FP32 OPT-125M (not the HWA
masters or any deployed array), real activations of decoder MLPs 6-11 on validation splits, one
layer at a time. Never dispatches or reads device states. Each weight is the pair q=(Ga-Gr)/HALF with
per-output-channel absmax scaling; each device is stuck independently with probability ``rate``.

- low rank: share of ||dW X||^2 captured by the best rank-r correction of the same input, fitted on
  synthetic text and applied to WikiText (an upper bound for one LoRA branch per matrix);
- local pair: residual share of ||dW X||^2 after exact compensation with known stuck conductances,
  clipped to [GMIN, GMAX]: the active device absorbs a stuck reference ("active only"), and the
  reference may also absorb a stuck active device ("pair"). Both-stuck weights stay as read.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import torch
from transformers import OPTConfig, OPTForCausalLM

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from training.reram_cmo import G_MAX_US, G_MIN_US, HALF_RANGE_US, MID_US, STUCK_US
from workflow.networks.opt_mlp import PINNED

RANKS = (4, 16, 64, 256)
LAYERS = range(6, 12)


def load(snapshot: Path):
    pinned = PINNED["facebook/opt-125m"]
    if hashlib.sha256((snapshot/"pytorch_model.bin").read_bytes()).hexdigest() != pinned["sha256"]:
        raise ValueError("Expected the pinned OPT-125M pytorch_model.bin.")
    model = OPTForCausalLM(OPTConfig.from_pretrained(snapshot)).eval()
    state = torch.load(snapshot/"pytorch_model.bin", weights_only=True, map_location="cpu")
    missing, unexpected = model.load_state_dict(
        {k if k.startswith(("model.", "lm_head")) else "model." + k: v for k, v in state.items()}, strict=False)
    if unexpected or set(missing) - {"lm_head.weight"}:
        raise ValueError(f"Unexpected checkpoint keys: {missing}, {unexpected}.")
    model.tie_weights()
    return model


@torch.no_grad()
def activations(model, corpus: Path, sequences: int) -> dict:
    tokens = torch.load(corpus, weights_only=False)["splits"]["validation"][: sequences * 256].view(sequences, 256)
    inputs, hooks = {}, []
    for layer in LAYERS:
        for name in ("fc1", "fc2"):
            hooks.append(getattr(model.model.decoder.layers[layer], name).register_forward_pre_hook(
                lambda m, a, key=(layer, name): inputs.setdefault(key, []).append(a[0].reshape(-1, a[0].shape[-1]))))
    for begin in range(0, sequences, 8):
        model(tokens[begin:begin + 8])
    for hook in hooks:
        hook.remove()
    return {key: torch.cat(chunks).T for key, chunks in inputs.items()}


def faulty_pair(q, rate, kind, generator):
    stuck = [torch.rand(q.shape, generator=generator) < rate for _ in range(2)]
    values = [G_MIN_US + (G_MAX_US - G_MIN_US) * torch.rand(q.shape, generator=generator) if kind == "random"
              else torch.full_like(q, STUCK_US[kind]) for _ in range(2)]
    active = torch.where(stuck[0], values[0], MID_US + q * HALF_RANGE_US)
    reference = torch.where(stuck[1], values[1], torch.full_like(q, MID_US))
    return active, reference, stuck


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True, help="pinned Hugging Face snapshot directory")
    parser.add_argument("--sequences", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model = load(args.snapshot)
    synthetic = activations(model, ROOT/"data/corpora/opt125m-synthetic-50m.pt", args.sequences)
    wikitext = activations(model, ROOT/"data/corpora/wikitext-2-raw-v1.pt", args.sequences)
    weights = {key: getattr(model.model.decoder.layers[key[0]], key[1]).weight.detach() for key in synthetic}

    print(f"{args.sequences * 256} tokens per corpus; seed {args.seed}; shares averaged over 12 matrices")
    generator = torch.Generator().manual_seed(args.seed)
    fit, transfer = [], []
    for key, W in weights.items():
        absmax = W.abs().amax(1, keepdim=True)
        active, reference, _ = faulty_pair(W/absmax, 0.03, "random", generator)
        dW = absmax * ((active - reference)/HALF_RANGE_US - W/absmax)
        Rs, Rw = dW @ synthetic[key], dW @ wikitext[key]
        U = torch.linalg.svd(Rs, full_matrices=False).U
        fit.append([((U[:, :r].T @ Rs).square().sum()/Rs.square().sum()).item() for r in RANKS])
        transfer.append([((U[:, :r].T @ Rw).square().sum()/Rw.square().sum()).item() for r in RANKS])
    for label, rows in (("synthetic (fit)", fit), ("WikiText (transfer)", transfer)):
        print(f"low rank, random 3%, captured on {label}: " +
              ", ".join(f"r{r} {sum(row[i] for row in rows)/len(rows):.0%}" for i, r in enumerate(RANKS)))

    for kind in ("random", "gmin", "gmax", "open"):
        for rate in (0.01, 0.03, 0.05):
            generator = torch.Generator().manual_seed(args.seed)
            energy = {"none": 0.0, "active": 0.0, "pair": 0.0}
            written = {"active": 0, "pair": 0}
            count = 0
            for key, W in weights.items():
                absmax = W.abs().amax(1, keepdim=True)
                q = W/absmax
                active, reference, (stuck_active, stuck_reference) = faulty_pair(q, rate, kind, generator)
                fix_active = stuck_reference & ~stuck_active
                fix_reference = stuck_active & ~stuck_reference
                active_fixed = torch.where(fix_active, (reference + q*HALF_RANGE_US).clamp(G_MIN_US, G_MAX_US), active)
                reference_fixed = torch.where(fix_reference, (active - q*HALF_RANGE_US).clamp(G_MIN_US, G_MAX_US), reference)
                for name, (a, r) in {"none": (active, reference), "active": (active_fixed, reference),
                                     "pair": (active_fixed, reference_fixed)}.items():
                    energy[name] += ((absmax*((a - r)/HALF_RANGE_US - q)) @ synthetic[key]).square().sum().item()
                written["active"] += int(fix_active.sum())
                written["pair"] += int((fix_active | fix_reference).sum())
                count += q.numel()
            print(f"local {kind:6} {rate:.0%}: residual active only {energy['active']/energy['none']:6.1%}, "
                  f"pair {energy['pair']/energy['none']:5.1%}; weights written active only "
                  f"{written['active']/count:.2%}, pair {written['pair']/count:.2%}")


if __name__ == "__main__":
    main()
