"""ebl.runtime.teacher_lora — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import copy
import json
import math
import time
from types import SimpleNamespace
import torch
from methods.teacher_config import EXPERIMENT_ID, MERGE_TOLERANCE, TeacherConfig
from methods.teacher_lora import _schedule, adapter_state, add_lora, development_metrics, hugging_face_state, merge_lora, objective, score


# --- the native run ------------------------------------------------------------------


def _inputs(request) -> list[dict]:
    from core.artifacts import sha256_file

    roles = ("teacher_weights", "device_data")
    extra = [role for role in ("weights", "resume", "device_model", "device_state", "selection_receipt")
             if getattr(request, role, None) is not None]
    if extra or any(getattr(request, role, None) is None for role in roles):
        raise ValueError(f"Expected exactly the inputs {list(roles)}. Provided extra value: {extra}.")
    from ebl.runtime.lifecycle import staged_source_inputs

    return [{"role": role, "path": str(getattr(request, role).resolve()), "sha256": sha256_file(getattr(request, role))}
            for role in roles] + staged_source_inputs()


def run_train(request) -> int:
    from ebl.runtime.store import RunStore
    from core.artifacts import atomic_write_json, sha256_file
    from core.tensor_identity import state_hash
    from core.serialization import load
    import networks.tasks as tasks
    import networks.opt as opt_mlp
    from ebl.runtime.lifecycle import ROOT, configure_runtime

    config = request.spec
    if not isinstance(config, TeacherConfig):
        raise TypeError("Expected an opt_task_teacher.v1 config.")
    inputs = _inputs(request)
    store = RunStore.create(output_root=request.output_dir, experiment_id=EXPERIMENT_ID,
                            resolved_config=config.to_dict(), command=request.command, repo_root=ROOT,
                            input_artifacts=inputs, resume_capability="unsupported")
    started = time.monotonic()
    try:
        runtime_math = configure_runtime(config.execution, config.seed)
        device = torch.device(config.execution.device)
        base, base_sha256 = opt_mlp.load_teacher(request.teacher_weights, SimpleNamespace(model=config.model,
                                                                                          teacher=None), device)
        if sha256_file(request.device_data) != config.choices_sha256:
            raise ValueError(f"Expected the task file {config.choices_name!r} with SHA-256 {config.choices_sha256}.")
        document = load(request.device_data)
        tasks.check_task(document, config.model)
        if document["task"] != config.task:
            raise ValueError(f"Expected a {config.task} task file. Provided value: {document['task']!r}.")
        cap = config.max_examples
        positions = base.config.max_positions
        counts = {"train": len(document["splits"]["train"]["labels"]), "development": config.development_examples}
        counts = {name: min(count, cap) if cap else count for name, count in counts.items()}
        train = tasks.cohort(document, "train", counts["train"], positions)
        development = tasks.cohort(document, "development", counts["development"], positions)
        k = document["choices"]
        decision = config.decision
        zero_shot = development_metrics(score(base, development["choices"], device), development["choices"], decision)
        store.append_metric({"stage": "zero_shot", "development": zero_shot})

        model = copy.deepcopy(base)
        parameters = add_lora(model, config)
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        steps_per_epoch = math.ceil(counts["train"] / config.batch_examples)
        total = config.epochs * steps_per_epoch
        scheduler = _schedule(optimizer, round(config.warmup_fraction * total), total)
        history, best, step = [], None, 0
        labels = train["choices"]["labels"]

        def checkpoint(epoch):
            metrics = development_metrics(score(model, development["choices"], device), development["choices"], decision)
            record = {"epoch": epoch, "step": step, **metrics}
            history.append(record)
            store.append_metric({"stage": "development", **record})
            nonlocal best
            if best is None or metrics["decision_accuracy_percent"] > best["metrics"]["decision_accuracy_percent"]:
                best = {"epoch": epoch, "step": step, "metrics": metrics, "state": adapter_state(model)}

        for epoch in range(1, config.epochs + 1):
            model.train()
            order = torch.randperm(counts["train"], generator=torch.Generator().manual_seed(config.seed + epoch))
            for begin in range(0, counts["train"], config.batch_examples):
                examples = order[begin:begin + config.batch_examples]
                optimizer.zero_grad(set_to_none=True)
                loss_value = 0.0
                for offset in range(0, len(examples), config.micro_examples):
                    micro = examples[offset:offset + config.micro_examples]
                    loss = objective(model, train["choices"], micro, labels, k, config.score, device) \
                        * (len(micro) / len(examples))
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite teacher fine-tuning loss.")
                    loss.backward()
                    loss_value += float(loss.detach())
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm))
                optimizer.step()
                scheduler.step()
                step += 1
                if step % 50 == 0 or step == total:
                    store.append_metric({"stage": "train", "epoch": epoch, "step": step, "loss": loss_value,
                                         "grad_norm": norm, "learning_rate": scheduler.get_last_lr()[0]})
                if config.development_every_steps and step % config.development_every_steps == 0 \
                        and begin + config.batch_examples < counts["train"]:
                    checkpoint(epoch)
                    model.train()
            checkpoint(epoch)

        with torch.no_grad():
            for name, value in model.named_parameters():
                if name in best["state"]:
                    value.copy_(best["state"][name].to(value.device))
        selected = development_metrics(score(model, development["choices"], device), development["choices"], decision)
        if selected != best["metrics"]:
            raise RuntimeError("The restored adapters did not reproduce their development selection.")
        merged = merge_lora(model)
        merged_scores = score(merged, development["choices"], device)
        merged_metrics = development_metrics(merged_scores, development["choices"], decision)
        adapter_gap = float((merged_scores - score(model, development["choices"], device)).abs().max())

        # Export and verify the Hugging Face checkpoint.
        path = store.run_dir / "checkpoints/pytorch_model.bin"
        torch.save(hugging_face_state(merged), path)
        reloaded = opt_mlp.from_hugging_face(torch.load(path, map_location="cpu", weights_only=True),
                                             base.config).to(device).eval()
        merged_state, reloaded_state = merged.state_dict(), reloaded.state_dict()
        exact = set(merged_state) == set(reloaded_state) and all(
            torch.equal(reloaded_state[key].cpu(), value.cpu()) for key, value in merged_state.items())
        sample = tasks.padded(development["choices"], torch.arange(min(8, len(development["choices"]["offsets"]) - 1)))
        with torch.no_grad():
            logits_exact = torch.equal(reloaded(sample["tokens"].to(device)), merged(sample["tokens"].to(device)))
        adapted = {f"layers.{i}.{'self_attn.' if n.endswith('_proj') else ''}{n}.weight"
                   for i in range(len(base.layers)) for n in config.targets}
        untouched = {k: v for k, v in base.state_dict().items() if k not in adapted}
        frozen = state_hash(untouched) == state_hash({k: merged.state_dict()[k] for k in untouched})
        changed = sum(not torch.equal(base.state_dict()[k], merged.state_dict()[k]) for k in adapted)
        gates = {"reload_tensors_exact": exact, "reload_logits_exact": logits_exact,
                 "adapter_merged_max_row_logprob_difference": adapter_gap,
                 "adapter_merged_within_tolerance": adapter_gap <= MERGE_TOLERANCE * max(
                     1.0, float(merged_scores.abs().max())),
                 "untouched_parameters_unchanged": frozen, "adapted_weights_changed": changed,
                 "adapted_weights": len(adapted)}
        if not (exact and logits_exact and frozen and gates["adapter_merged_within_tolerance"]):
            raise RuntimeError(f"Merged teacher export gates failed: {gates}.")
        merged_sha256 = sha256_file(path)
        adapters = store.run_dir / "checkpoints/adapter.pt"
        torch.save({"config": config.to_dict()["lora"], "seed": config.seed, "state": best["state"]}, adapters)
        provenance = {
            "schema": "opt_task_teacher.provenance.v1",
            "teacher_id": config.teacher_id,
            "base": {"model": config.model, "revision": opt_mlp.PINNED[config.model]["revision"],
                     "sha256": base_sha256},
            "task": {"name": config.task, "choices": {"name": config.choices_name, "sha256": config.choices_sha256},
                     "decision": decision, "train_examples": counts["train"],
                     "development_examples": counts["development"], "test_split_read": False},
            "lora": {**config.to_dict()["lora"], "layers": len(base.layers),
                     "trainable_parameters": sum(p.numel() for p in parameters)},
            "training": {**config.to_dict()["training"], "optimizer": "adamw", "schedule": "linear_warmup_cosine",
                         "steps": total, "steps_per_epoch": steps_per_epoch},
            "selection": {"rule": "maximum development decision accuracy; ties keep the earlier checkpoint",
                          "epoch": best["epoch"], "step": best["step"], "history": history},
            "development": {"zero_shot": zero_shot, "adapter": best["metrics"], "merged": merged_metrics},
            "merged": {"path": "checkpoints/pytorch_model.bin", "sha256": merged_sha256, "dtype": "float32"},
            "gates": gates,
        }
        atomic_write_json(store.run_dir / "teacher_provenance.json", provenance)
        result = {
            "teacher_id": config.teacher_id,
            "task": config.task,
            "merged_sha256": merged_sha256,
            "development_zero_shot": zero_shot,
            "development_merged": merged_metrics,
            "gain_over_zero_shot_points": merged_metrics["decision_accuracy_percent"]
            - zero_shot["decision_accuracy_percent"],
            "selected": {"epoch": best["epoch"], "step": best["step"]},
            "gates": gates,
            "test_used": False,
            "runtime_math": runtime_math,
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        }
        records = [store.artifact_record(path, kind="teacher_weights"),
                   store.artifact_record(adapters, kind="lora_adapters"),
                   store.artifact_record(store.run_dir / "teacher_provenance.json", kind="teacher_provenance")]
        store.complete(metrics=result, artifacts=records)
        print(json.dumps({"run_dir": str(store.run_dir), "teacher_id": config.teacher_id, "status": "complete",
                          "merged_sha256": merged_sha256}), flush=True)
        return 0
    except BaseException as exc:
        store.fail(exc)
        raise
