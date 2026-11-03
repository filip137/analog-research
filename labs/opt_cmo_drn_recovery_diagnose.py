"""Read-only CPU diagnosis of the declared clean nominal recovery checkpoint.

This is a saved-state sensitivity diagnostic, not a training or selection arm.
Counterfactual conductances exist only in memory. It uses a small fixed prefix
of the development cohort and never writes a checkpoint or changes the queue.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.artifacts import atomic_write_json
from experiments.opt_cmo_drn.config import parse_config
from experiments.opt_cmo_drn.data import pinned_load
from experiments.opt_cmo_drn.runtime import from_mapped, teacher_model
from training.reram_cmo import G_MAX_US, HALF_RANGE_US, MID_US, programming_sigma


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    root = args.study_root.resolve()
    runs = root / "execution/runs"
    settings = json.loads((root / "execution/settings.json").read_text())
    config = parse_config(json.loads((runs / "clean-none-learning/run/config.resolved.json").read_text()))
    config = replace(config, device="cpu")
    declared = {
        "source": ("full-clean", "best_weights"),
        "initial": ("clean-deploy-none", "device_state"),
        "learned": ("clean-none-learning", "recovery_state"),
        "corrupt_initial": ("clean-deploy-random", "device_state"),
        "corrupt_rewrite": ("clean-random-rewrite", "recovery_state"),
    }
    paths, identities = {}, {}
    for name, (run, kind) in declared.items():
        bundle = runs / run / "run"
        result = json.loads((bundle / "result.json").read_text())
        artifact, = [a for a in result["artifacts"] if a["kind"] == kind]
        paths[name] = bundle / artifact["path"]
        identities[name] = {"path": str(paths[name]), "validated_manifest_sha256": artifact["sha256"]}
    source, initial, final = (load(paths[n]) for n in ("source", "initial", "learned"))
    assert initial["source_weights_sha256"] == identities["source"]["validated_manifest_sha256"]
    assert final["initial_device_state_sha256"] == identities["initial"]["validated_manifest_sha256"]
    unchanged = set(source["model"]) == set(final["weights"]["model"]) and all(
        torch.equal(v, final["weights"]["model"][k]) for k, v in source["model"].items())
    threshold_us = config.recovery.threshold_sigma * programming_sigma(MID_US, 2.0)
    threshold_q = threshold_us / HALF_RANGE_US
    report = {
        "role": "read_only_cpu_diagnostic_not_selection",
        "inputs": identities,
        "torch_version": str(torch.__version__),
        "digital_and_master_weights_unchanged": unchanged,
        "threshold_us": threshold_us,
        "threshold_q": threshold_q,
        "arrays": [],
    }
    for a, b, writer in zip(initial["bank"]["cells"], final["bank"]["cells"], final["engine"]["writers"]):
        x = a["held"]
        delta = b["held"] - x
        report["arrays"].append({
            "name": a["name"],
            "sampled_g_quantiles_us": torch.quantile(x.reshape(-1)[::97], torch.tensor([.05, .5, .95])).tolist(),
            "threshold_to_local_sigma_mean": float((threshold_us / programming_sigma(x, 2.0)).mean()),
            "delta_rms_us": float(delta.square().mean().sqrt()),
            "attempt_histogram": [int((writer["attempt_counts"] == i).sum()) for i in range(5)],
            "minimum_held_us": float(b["held"].min()),
            "maximum_held_us": float(b["held"].max()),
        })
    bad_initial, bad_rewrite = load(paths["corrupt_initial"]), load(paths["corrupt_rewrite"])
    compensation = []
    for a, b in zip(bad_initial["bank"]["cells"], bad_rewrite["bank"]["cells"]):
        mask = torch.from_numpy(np.unpackbits(a["rewrite_selected_bits"].numpy(), count=a["held"].numel(),
                                              bitorder="little").reshape(a["held"].shape).astype(bool))
        compensation.append({
            "name": a["name"], "compensating_cells": int(mask.sum()),
            "initial_rms_from_original_target_us": float((a["held"][mask] - a["target"][mask]).square().mean().sqrt()),
            "rewritten_rms_from_original_target_us": float((b["held"][mask] - a["target"][mask]).square().mean().sqrt()),
        })
    report["corrupt_partner_compensation"] = compensation
    del bad_initial, bad_rewrite, final
    print("Saved-state audit complete", flush=True)

    teacher = teacher_model(settings["local_teacher"], config)
    model = from_mapped(teacher, config, source).requires_grad_(False)
    masters = model.parameter_groups()[0]
    for p in masters:
        p.requires_grad_(True)
    corpus = pinned_load(args.corpus_root / f"{config.data.corpus}.pt", config.data.sha256)
    tokens = corpus["splits"]["validation"].reshape(-1)[:2 * config.data.sequence_length]
    tokens = tokens.reshape(2, config.data.sequence_length)[:, :64].contiguous()
    report["cohort"] = {"split": "validation", "sequence_indices": [0, 1], "prefix_tokens": 64,
                        "scored_tokens": 126, "corpus_sha256": config.data.sha256}
    with torch.no_grad():
        reference = teacher(tokens)[:, :-1].flatten(0, 1).softmax(-1)
    held = [a["held"] for a in initial["bank"]["cells"]]

    def view(values):
        normalized = [g / G_MAX_US for g in values]
        return tuple(tuple(normalized[i:i + 2]) for i in range(0, len(normalized), 2))

    def loss(values):
        prediction = model(tokens, view(values)).flatten(0, 1).log_softmax(-1)
        return F.kl_div(prediction, reference, reduction="batchmean")

    model.eval()
    with torch.no_grad():
        baseline = float(loss(held))
    model.train()
    training_loss = loss(held)
    training_loss.backward()
    report["fixed_cohort_eval_kl"] = baseline
    report["fixed_cohort_training_kl"] = float(training_loss.detach())
    # Exact first-step Adam direction in normalized conductance coordinates.
    direction = [-p.grad / (p.grad.abs() + 1e-8) for p in masters]
    derivative = sum(float((p.grad.double() * d.double()).sum()) for p, d in zip(masters, direction))
    report["adam_direction_derivative"] = derivative
    metrics = [json.loads(line) for line in
               (runs / "clean-none-learning/run/metrics.jsonl").read_text().splitlines()]
    first_write = next(row for row in metrics if row.get("event") == "recovery_update"
                       and row["costs"]["write_attempts"] > 0)
    count = first_write["costs"]["write_attempts"]
    strengths = torch.cat([p.grad.abs().reshape(-1) for p in masters])
    indices = torch.topk(strengths, count, sorted=False).indices
    selected = torch.zeros(strengths.numel(), dtype=torch.bool)
    selected[indices] = True
    masks = [mask.reshape(p.shape) for mask, p in zip(selected.split([p.numel() for p in masters]), masters)]
    del strengths, indices
    report["production_first_write"] = {"step": first_write["step"], "cells": count}
    for p in masters:
        p.grad = None
    model.eval()
    rows = []
    with torch.no_grad():
        for step in [-1e-5, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
            changed = [g + HALF_RANGE_US * step * d for g, d in zip(held, direction)]
            row = {"normalized_step": step, "teacher_kl": float(loss(changed))}
            rows.append(row)
            print("continuous", row, flush=True)
        report["central_direction_derivative"] = (rows[1]["teacher_kl"] - rows[0]["teacher_kl"]) / 2e-5
        # Freeze the initial gradient for 10 hypothetical opportunities. This
        # isolates command/noise effects; it does not reproduce training's
        # changing gradients or its effective batch of 32 full sequences.
        commands = []
        for d in direction:
            pending = 10 * config.recovery.learning_rate * d
            commands.append(torch.where(pending.abs() > threshold_q, pending, 0))
        report["frozen_gradient_wave_cells"] = sum(int((d != 0).sum()) for d in commands)
        wave_rows = []
        for scale, noisy in [(0.01, False), (0.1, False), (1.0, False), (1.0, True)]:
            generator = torch.Generator().manual_seed(271002)
            changed = []
            for g, command in zip(held, commands):
                target = g + HALF_RANGE_US * scale * command
                if noisy:
                    target = target + programming_sigma(target, 2.0) * torch.randn(target.shape, generator=generator)
                changed.append(torch.where(command != 0, target, g))
            row = {"command_scale": scale, "noise": noisy, "teacher_kl": float(loss(changed))}
            wave_rows.append(row)
            print("frozen_gradient_wave", row, flush=True)
        # Match only the production first-wave cell count. Select the strongest
        # current diagnostic gradients, not production's unavailable first-wave
        # identities or accumulated commands. This is another explicit proxy.
        commands = [torch.where(mask, d.sign() * threshold_q * (1 + 1e-6), 0)
                    for mask, d in zip(masks, direction)]
        count_matched_rows = []
        for scale, noisy in [(0.01, False), (0.01, True), (0.1, False), (0.1, True), (1.0, False), (1.0, True)]:
            generator = torch.Generator().manual_seed(271002)
            changed = []
            for g, command in zip(held, commands):
                target = g + HALF_RANGE_US * scale * command
                if noisy:
                    target = target + programming_sigma(target, 2.0) * torch.randn(target.shape, generator=generator)
                changed.append(torch.where(command != 0, target, g))
            row = {"command_scale": scale, "noise": noisy, "teacher_kl": float(loss(changed))}
            count_matched_rows.append(row)
            print("count_matched_gradient_proxy", row, flush=True)
    report["continuous_direction_probes"] = rows
    report["frozen_gradient_wave_probes"] = wave_rows
    report["count_matched_gradient_proxy"] = count_matched_rows
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, report)
    print("REPORT=" + str(args.output), flush=True)


if __name__ == "__main__":
    main()
