"""methods.teacher_config — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from core.contracts import RunMode, config_error
from core.config import _LIFECYCLE_ID, _SHA256, _choice, _identifier, _integer, _list, _number, _object
from networks.config import OPT_MODELS, TASK_DECISIONS, TASK_NAMES
from core.execution import Execution, parse_execution


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
