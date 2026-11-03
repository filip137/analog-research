"""``opt_task_teacher.v1``: a task-fine-tuned FP32 OPT teacher made with merged LoRA.

LoRA adapters (rank r, scale alpha/r, A uniform +/-1/sqrt(fan_in), B zero, no
dropout) wrap the declared linear layers of every decoder layer of the pinned
OPT-125M. They are trained with multiple-choice cross entropy over the summed
continuation log-likelihoods of a ``choice_task.v1`` file. Selection uses the
development split only; the test split is never read.

The selected adapters are merged into the FP32 weights (W + alpha/r * B @ A)
and exported as a Hugging Face ``pytorch_model.bin``: ``model.decoder.*``
keys and a tied ``lm_head.weight``. ``network.teacher`` then pins it by
SHA-256 in a lifecycle. The export gates are:
- reloading through ``from_hugging_face`` reproduces the merged tensors and
  logits bitwise;
- the merged and adapter forwards agree;
- every parameter outside the adapted weights is unchanged.

LoRA only prepares this teacher; it is never a deployed adapter.

    python -m ebl train --config teacher.json --output-dir RUNS \\
        --teacher-weights <pinned pytorch_model.bin> --device-data <task file .pt>
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

from experiments.schema import RunMode, config_error
from workflow.lifecycle import (
    _LIFECYCLE_ID,
    _SHA256,
    OPT_MODELS,
    TASK_DECISIONS,
    TASK_NAMES,
    Execution,
    _choice,
    _identifier,
    _integer,
    _list,
    _number,
    _object,
    parse_execution,
)

EXPERIMENT_ID = "opt_task_teacher.v1"
TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
# Training objectives: multiple-choice cross entropy over summed (``sum``) or per-token mean
# (``token_mean``) continuation log-likelihoods, or next-token cross entropy on the gold
# continuation only (``gold_lm``, supervised fine-tuning). Selection always uses the declared decision.
SCORES = ("sum", "token_mean", "gold_lm")
# Adapter and merged forwards differ only by FP32 summation order.
MERGE_TOLERANCE = 1e-4


# --- strict config ----------------------------------------------------------------


@dataclass(frozen=True)
class TeacherConfig:
    teacher_id: str
    model: str
    task: str
    choices_name: str
    choices_sha256: str
    decision: str
    development_examples: int
    rank: int
    alpha: float
    targets: tuple[str, ...]
    learning_rate: float
    weight_decay: float
    warmup_fraction: float
    epochs: int
    batch_examples: int
    micro_examples: int
    max_grad_norm: float
    score: str
    seed: int
    development_every_steps: int
    max_examples: int
    execution: Execution

    def to_dict(self) -> dict:
        return {
            "experiment_id": EXPERIMENT_ID,
            "schema_version": 1,
            "teacher_id": self.teacher_id,
            "model": self.model,
            "task": {"name": self.task, "choices": {"name": self.choices_name, "sha256": self.choices_sha256},
                     "decision": self.decision, "development_examples": self.development_examples},
            "lora": {"rank": self.rank, "alpha": self.alpha, "targets": list(self.targets)},
            "training": {
                "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
                "warmup_fraction": self.warmup_fraction, "epochs": self.epochs,
                "batch_examples": self.batch_examples, "micro_examples": self.micro_examples,
                "max_grad_norm": self.max_grad_norm, "score": self.score, "seed": self.seed,
                "development_every_steps": self.development_every_steps, "max_examples": self.max_examples,
            },
            "execution": self.execution.to_dict(),
        }


def parse_config(payload: Any) -> TeacherConfig:
    raw = _object(payload, "config", ("experiment_id", "schema_version", "teacher_id", "model", "task", "lora",
                                       "training", "execution"))
    if raw["experiment_id"] != EXPERIMENT_ID or type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise config_error("experiment/schema", f"to be {EXPERIMENT_ID}/1", [raw["experiment_id"], raw["schema_version"]])
    task = _object(raw["task"], "task", ("name", "choices", "decision", "development_examples"))
    choices = _object(task["choices"], "task.choices", ("name", "sha256"))
    if not isinstance(choices["sha256"], str) or not _SHA256.match(choices["sha256"]):
        raise config_error("task.choices.sha256", "to be a lowercase SHA-256 digest", choices["sha256"])
    lora = _object(raw["lora"], "lora", ("rank", "alpha", "targets"))
    targets = tuple(_choice(v, "lora.targets", TARGETS) for v in _list(lora["targets"], "lora.targets"))
    if not targets or len(set(targets)) != len(targets):
        raise config_error("lora.targets", "to name unique adapted linear layers", targets)
    training = _object(raw["training"], "training", (
        "learning_rate", "weight_decay", "warmup_fraction", "epochs", "batch_examples", "micro_examples",
        "max_grad_norm", "score", "seed", "development_every_steps", "max_examples"))
    batch = _integer(training["batch_examples"], "training.batch_examples", minimum=1)
    micro = _integer(training["micro_examples"], "training.micro_examples", minimum=1)
    if batch % micro:
        raise config_error("training.micro_examples", "to divide training.batch_examples", micro)
    warmup = _number(training["warmup_fraction"], "training.warmup_fraction")
    if not 0 <= warmup < 1:
        raise config_error("training.warmup_fraction", "to be in [0, 1)", warmup)
    return TeacherConfig(
        teacher_id=_identifier(raw["teacher_id"], "teacher_id", _LIFECYCLE_ID),
        model=_choice(raw["model"], "model", tuple(OPT_MODELS)),
        task=_choice(task["name"], "task.name", TASK_NAMES),
        choices_name=_identifier(choices["name"], "task.choices.name", _LIFECYCLE_ID),
        choices_sha256=choices["sha256"],
        decision=_choice(task["decision"], "task.decision", TASK_DECISIONS),
        development_examples=_integer(task["development_examples"], "task.development_examples", minimum=1),
        rank=_integer(lora["rank"], "lora.rank", minimum=1),
        alpha=_number(lora["alpha"], "lora.alpha", positive=True),
        targets=targets,
        learning_rate=_number(training["learning_rate"], "training.learning_rate", positive=True),
        weight_decay=_number(training["weight_decay"], "training.weight_decay"),
        warmup_fraction=warmup,
        epochs=_integer(training["epochs"], "training.epochs", minimum=1),
        batch_examples=batch,
        micro_examples=micro,
        max_grad_norm=_number(training["max_grad_norm"], "training.max_grad_norm", positive=True),
        score=_choice(training["score"], "training.score", SCORES),
        seed=_integer(training["seed"], "training.seed", maximum=2**31 - 1),
        development_every_steps=_integer(training["development_every_steps"], "training.development_every_steps"),
        max_examples=_integer(training["max_examples"], "training.max_examples"),
        execution=parse_execution(raw["execution"]),
    )


def resolve_config(document: TeacherConfig, mode: RunMode) -> TeacherConfig:
    if mode != RunMode.TRAIN:
        raise config_error("the requested run mode", "to be train", mode.value)
    return document


# --- LoRA on the native decoder -----------------------------------------------------

import torch  # noqa: E402  (the config layer above stays importable without numerical work)
from torch import nn  # noqa: E402


class LoraLinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update B @ A scaled by alpha/r."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, generator: torch.Generator):
        super().__init__()
        self.base = base.requires_grad_(False)
        inputs, outputs = base.in_features, base.out_features
        a = torch.empty(rank, inputs).uniform_(-inputs ** -0.5, inputs ** -0.5, generator=generator)
        self.a = nn.Parameter(a.to(base.weight.device))
        self.b = nn.Parameter(torch.zeros(outputs, rank, device=base.weight.device))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale

    def merged(self) -> nn.Linear:
        linear = copy.deepcopy(self.base)
        with torch.no_grad():
            linear.weight += self.scale * (self.b @ self.a)
        return linear


def _owner(layer, name):
    return layer.self_attn if name.endswith("_proj") else layer


def add_lora(model, config: TeacherConfig) -> list[nn.Parameter]:
    """Wrap the target layers of every decoder layer (CPU-seeded, so initialization is portable)."""

    model.requires_grad_(False)
    generator = torch.Generator().manual_seed(config.seed)
    parameters = []
    for layer in model.layers:
        for name in config.targets:
            owner = _owner(layer, name)
            wrapped = LoraLinear(getattr(owner, name), config.rank, config.alpha, generator)
            setattr(owner, name, wrapped)
            parameters += [wrapped.a, wrapped.b]
    return parameters


def merge_lora(model):
    """A plain decoder whose target weights absorb their adapters."""

    merged = copy.deepcopy(model)
    for layer in merged.layers:
        for name in TARGETS:
            owner = _owner(layer, name)
            module = getattr(owner, name)
            if isinstance(module, LoraLinear):
                setattr(owner, name, module.merged())
    return merged.requires_grad_(False)


def adapter_state(model) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()
            if name.endswith((".a", ".b"))}


def hugging_face_state(model) -> dict:
    """``OPTForCausalLM`` keys: the decoder under ``model.decoder.`` and the tied head."""

    state = {f"model.decoder.{k}": v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
    state["lm_head.weight"] = state["model.decoder.embed_tokens.weight"]
    return state


# --- scoring -------------------------------------------------------------------------


def row_sums(model, batch, device):
    """Summed continuation log-probabilities; the head runs only at scored positions."""

    from workflow import tasks

    tokens = batch["tokens"].to(device)
    hidden = model.embed(tokens)
    for layer in model.layers:
        hidden = layer(hidden)
    mask = tasks.scored_mask(batch, device)
    logits = model.logits(hidden[:, :-1][mask])
    targets = tokens[:, 1:][mask]
    log_probs = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    return tasks.row_sums(log_probs, mask)


def objective(model, choices, examples, labels, k, kind, device):
    """The declared fine-tuning loss of some examples."""

    from workflow import tasks

    if kind == "gold_lm":
        batch = tasks.padded(choices, examples * k + labels[examples])
        tokens = (batch["lengths"] - batch["context_lengths"]).to(device)
        return -row_sums(model, batch, device).sum() / tokens.sum()
    batch = tasks.padded(choices, (examples[:, None] * k + torch.arange(k)[None, :]).flatten())
    sums = row_sums(model, batch, device)
    if kind == "token_mean":
        sums = sums / (batch["lengths"] - batch["context_lengths"]).to(device)
    return tasks.choice_loss(sums, labels[examples].to(device), k)


@torch.no_grad()
def score(model, choices, device) -> torch.Tensor:
    from workflow import tasks

    model.eval()
    sums = torch.zeros(len(choices["offsets"]) - 1, dtype=torch.float64)
    for index in tasks.batches(choices):
        sums[index] = row_sums(model, tasks.padded(choices, index), device).cpu()
    return sums


def development_metrics(sums, choices, decision) -> dict:
    from workflow import tasks

    k = choices["k"]
    labels = choices["labels"]
    chars = choices["chars"].view(-1, k).double()
    scores = sums.view(-1, k)
    loss = float(torch.nn.functional.cross_entropy(scores, labels))
    return {
        "examples": len(labels),
        "accuracy_percent": 100 * float((scores.argmax(1) == labels).double().mean()),
        "accuracy_norm_percent": 100 * float(((scores / chars).argmax(1) == labels).double().mean()),
        "choice_loss": loss,
        "decision_accuracy_percent": 100 * float(
            (tasks.decision_scores(scores, chars, decision).argmax(1) == labels).double().mean()),
    }


# --- the native run ------------------------------------------------------------------


def _inputs(request) -> list[dict]:
    from experiments.artifacts import sha256_file

    roles = ("teacher_weights", "device_data")
    extra = [role for role in ("weights", "resume", "device_model", "device_state", "selection_receipt")
             if getattr(request, role, None) is not None]
    if extra or any(getattr(request, role, None) is None for role in roles):
        raise ValueError(f"Expected exactly the inputs {list(roles)}. Provided extra value: {extra}.")
    from workflow.runtime import staged_source_inputs

    return [{"role": role, "path": str(getattr(request, role).resolve()), "sha256": sha256_file(getattr(request, role))}
            for role in roles] + staged_source_inputs()


def _schedule(optimizer, warmup: int, total: int):
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_train(request) -> int:
    from experiments.artifacts import RunStore, atomic_write_json, sha256_file
    from experiments.cifar_crossbar.model import state_hash
    from experiments.cifar_crossbar.runtime import load
    from workflow import tasks
    from workflow.networks import opt_mlp
    from workflow.runtime import ROOT, configure_runtime

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


__all__ = ["EXPERIMENT_ID", "TeacherConfig", "parse_config", "resolve_config", "run_train"]
