"""Time and compare speed options for one OPT-1.3b whole-model HWA update.

Diagnostic only (not a pipeline stage, not evidence about HWA quality). It
rebuilds the HWA-stage objects through the same public functions the runtime
uses (lifecycle parser, ``load_teacher``, ``build_network``, the declared
sampler, ``batch`` and ``objective``) and repeats the loop body of
``methods.hwa.train_source``:

1. Phase timing of one update in FP32 (CUDA events): teacher pass, noise
   sample, student forward, loss, backward, optimizer step, clamp.
2. Speed variants: TF32 matmuls, bf16 autocast, PyTorch SDPA attention, fused
   Adam, and combinations.
3. Numerics: from the same state, batch and noise draw, loss and gradient
   deviation of each precision variant against FP32; then a short trajectory
   (same batches and noise draws) and a held FP32 development KL afterwards.

Usage: python bench_hwa_step.py LIFECYCLE.json TEACHER.bin OUT.json [--steps N]
Requires EBL_CORPUS_ROOT. Run from the repository root.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import time

import torch
from torch.nn import functional as F

import methods.hwa as hwa
import networks.deployment as deployment
import networks.opt as opt
from ebl.schema.lifecycle import parse_lifecycle

PHASES = ("teacher", "sample", "forward", "loss", "backward", "step", "clamp")


def sdpa_forward(self, x):
    b, t, d = x.shape

    def heads(value):
        return value.view(b, t, self.heads, self.head_dim).transpose(1, 2)

    q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.out_proj(out.transpose(1, 2).reshape(b, t, d))


ORIGINAL_ATTENTION = opt.Attention.forward


def configure(variant):
    torch.backends.cuda.matmul.allow_tf32 = "tf32" in variant
    torch.backends.cudnn.allow_tf32 = "tf32" in variant
    opt.Attention.forward = sdpa_forward if "sdpa" in variant else ORIGINAL_ATTENTION


def autocast(variant):
    if "bf16" in variant:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class Bench:
    def __init__(self, args):
        self.device = torch.device("cuda")
        self.lifecycle = parse_lifecycle(json.load(open(args.lifecycle)))
        self.arm = self.lifecycle.hwa.arm("standard_hwa_model_probe")
        self.teacher, _ = deployment.load_teacher(args.teacher, self.lifecycle.network, self.device)
        corpus, _ = opt.load_corpus(self.lifecycle.data, self.lifecycle.network.model)
        length = self.lifecycle.data.sequence_length
        train, dev = corpus["splits"]["train"], corpus["splits"]["validation"]
        blocks = train[: len(train) // length * length].reshape(-1, length)
        order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(7))
        self.tokens = {"tokens": blocks[order[: 32 * 64]].to(self.device)}
        self.dev = {"tokens": dev[: 32 * length].reshape(32, length)}
        self.batch_size = self.lifecycle.hwa.batch_size

    def build(self, fused=False):
        network = deployment.build_network(self.teacher, self.lifecycle, self.arm.arm_id)
        network.enable_calibration(True)
        network.q.requires_grad_(True)
        groups = [(self.lifecycle.hwa.learning_rate, [network.q] + network.calibration_parameters())]
        network.enable_digital(True)
        groups.append((self.arm.digital_learning_rate, network.digital_parameters()))
        optimizer = torch.optim.Adam([{"params": p, "lr": lr} for lr, p in groups], fused=fused or None)
        sample = hwa.endpoint_sampler(self.lifecycle, self.arm, None, self.device, layout=network.layout)
        network.train()
        return network, optimizer, sample

    def index(self, step):
        begin = (step * self.batch_size) % len(self.tokens["tokens"])
        return torch.arange(begin, begin + self.batch_size, device=self.device)

    def update(self, network, optimizer, sample, step, variant, events=None):
        def mark(name):
            if events is not None:
                events[name].record()

        mark("start")
        features, teacher_logits, labels = opt.batch(network, self.tokens, self.index(step), self.device)
        mark("teacher")
        q = sample(network.q, "open", 0.0)
        mark("sample")
        optimizer.zero_grad(set_to_none=True)
        with autocast(variant):
            logits = network.forward_features(features, q)
            mark("forward")
            loss = opt.objective(logits.float(), teacher_logits, labels, "teacher_kl")
        mark("loss")
        loss.backward()
        mark("backward")
        optimizer.step()
        mark("step")
        with torch.no_grad():
            network.q.clamp_(-1, 1)
        mark("clamp")
        return float(loss.detach())

    def timing(self, variant, steps):
        configure(variant)
        network, optimizer, sample = self.build(fused="fused" in variant)
        for step in range(2):
            self.update(network, optimizer, sample, step, variant)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        totals = dict.fromkeys(PHASES, 0.0)
        started = time.perf_counter()
        for step in range(2, 2 + steps):
            events = {k: torch.cuda.Event(enable_timing=True) for k in ("start",) + PHASES}
            self.update(network, optimizer, sample, step, variant, events)
            torch.cuda.synchronize()
            previous = "start"
            for phase in PHASES:
                totals[phase] += events[previous].elapsed_time(events[phase]) / 1000
                previous = phase
        wall = (time.perf_counter() - started) / steps
        row = {
            "variant": variant, "steps": steps, "seconds_per_update": wall,
            "tokens_per_second": self.batch_size * self.lifecycle.data.sequence_length / wall,
            "phase_seconds": {k: v / steps for k, v in totals.items()},
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        del network, optimizer, sample
        self.release()
        return row

    def release(self):
        gc.collect()
        torch.cuda.empty_cache()

    def gradients(self, variant):
        """Loss and gradients at the initial state, batch 0, first noise draw."""

        configure(variant)
        network, optimizer, sample = self.build()
        features, teacher_logits, labels = opt.batch(network, self.tokens, self.index(0), self.device)
        q = sample(network.q, "open", 0.0)
        with autocast(variant):
            logits = network.forward_features(features, q)
            loss = opt.objective(logits.float(), teacher_logits, labels, "teacher_kl")
        loss.backward()
        grads = {
            "analog_q": network.q.grad.detach().float().cpu(),
            "digital": torch.cat([p.grad.detach().flatten() for p in network.digital_parameters() if p.grad is not None]).float().cpu(),
        }
        value = float(loss.detach())
        del network, optimizer, sample, logits, loss
        self.release()
        return value, grads

    def trajectory(self, variant, steps):
        configure(variant)
        network, optimizer, sample = self.build()
        losses = [self.update(network, optimizer, sample, step, variant) for step in range(steps)]
        configure("fp32")
        metrics = opt.evaluate(network, self.dev, self.device)
        out = {"variant": variant, "losses": losses, "development_kl_fp32_eval": metrics["teacher_kl"],
               "q_sha_prefix": float(network.q.detach().double().sum())}
        del network, optimizer, sample
        self.release()
        return out


def compare(reference, other):
    out = {}
    for key in reference:
        a, b = reference[key], other[key]
        out[key] = {
            "relative_l2": float((a - b).double().norm() / a.double().norm().clamp_min(1e-30)),
            "cosine": float(a.double().dot(b.double()) / (a.double().norm() * b.double().norm()).clamp_min(1e-30)),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lifecycle")
    parser.add_argument("teacher")
    parser.add_argument("output")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--trajectory", type=int, default=16)
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    bench = Bench(args)
    report = {"lifecycle": args.lifecycle, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "timing": [], "gradients": {}, "trajectories": []}

    def save():
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)

    for variant in ("fp32", "fp32+sdpa+fused", "tf32", "tf32+sdpa+fused", "bf16+sdpa+fused"):
        report["timing"].append(bench.timing(variant, args.steps))
        print(json.dumps(report["timing"][-1]), flush=True)
        save()
    if args.timing_only:
        return
    base_loss, base = bench.gradients("fp32")
    report["gradients"]["fp32"] = {"loss": base_loss}
    for variant in ("fp32+sdpa", "tf32", "bf16"):
        loss, grads = bench.gradients(variant)
        report["gradients"][variant] = {"loss": loss, "loss_relative": abs(loss - base_loss) / abs(base_loss),
                                        **compare(base, grads)}
        print(json.dumps({variant: report["gradients"][variant]}), flush=True)
        del grads
        save()
    for variant in ("fp32", "tf32", "bf16"):
        report["trajectories"].append(bench.trajectory(variant, args.trajectory))
        print(json.dumps(report["trajectories"][-1]), flush=True)
        save()


if __name__ == "__main__":
    main()
