"""Labelled multiple-choice tasks for task-fine-tuned OPT teachers.

A task file (``crossbar_lifecycle.choice_task.v1``) stores, for each split,
every example's tokenized context and its K tokenized continuations. It also
stores the label and each continuation's character count, which is the
lm-eval ``acc_norm`` denominator. Templates and continuation encoding follow
lm-eval-harness v0.4 (``_encode_pair`` with a ``" "`` target delimiter), with
one difference: every scored row starts with BOS, as OPT documents do.

The official validation split is the task's ``test`` split. ``development``
is a seeded hold-out from the official train split. It is disjoint by group
(HellaSwag source video or article, SST-2 sentence, PIQA goal), and the
remaining train examples are the adaptation (``train``) split. The same task
data also give a ``token_corpus.v1`` stream for HWA and recovery: each
example contributes BOS + context + gold continuation.

The scorer is shared by lifecycle evaluation (held apparent state) and teacher
fine-tuning. It gathers the scored continuation positions before the
vocabulary softmax and sums row log-probabilities in float64.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable

import torch
from torch.nn import functional as F

CHOICE_SCHEMA = "crossbar_lifecycle.choice_task.v1"
CHOICE_SPLITS = ("train", "development", "test")
DECISIONS = ("acc_norm", "acc")
BOS, PAD = 2, 1
# Tokens per scoring batch (rows x padded length); bounds the [rows, length, vocabulary] logits.
TOKEN_BUDGET = 2048


def _hellaswag_text(text: str) -> str:
    text = text.strip()
    # Brackets are artifacts of the WikiHow part of HellaSwag (lm-eval preprocess).
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def _hellaswag(record: dict) -> tuple[str, list[str], int, str]:
    context = record["ctx_a"] + " " + record["ctx_b"].capitalize()
    query = _hellaswag_text(record["activity_label"] + ": " + context)
    return query, [_hellaswag_text(ending) for ending in record["endings"]], int(record["label"]), record["source_id"]


def _sst2(record: dict) -> tuple[str, list[str], int, str]:
    query = f"{record['sentence']}\nQuestion: Is this sentence positive or negative?\nAnswer:"
    return query, ["negative", "positive"], int(record["label"]), record["sentence"].strip()


def _piqa(record: dict) -> tuple[str, list[str], int, str]:
    return f"Question: {record['goal']}\nAnswer:", [record["sol1"], record["sol2"]], int(record["label"]), record["goal"]


# Domain fine-tuning tasks (exp/domain-finetuning, exp-001). BoolQ passages are tail-truncated
# to the 256-token budget before formatting; CaseHOLD contexts end with "(" and take the holding
# without a delimiter.
RCT_SECTIONS = ("background", "objective", "methods", "results", "conclusions")


def _mnli(record: dict) -> tuple[str, list[str], int, str]:
    query = f"{record['premise']}\nQuestion: {record['hypothesis']} True, False or Neither?\nAnswer:"
    return query, ["True", "Neither", "False"], int(record["label"]), record["premise"]


def _rct(record: dict) -> tuple[str, list[str], int, str]:
    query = f"Sentence from a clinical trial abstract: {record['text']}\nSection:"
    return query, list(RCT_SECTIONS), RCT_SECTIONS.index(record["label"]), str(record["abstract_id"])


def _boolq(record: dict) -> tuple[str, list[str], int, str]:
    query = f"{record['passage']}\nQuestion: {record['question']}?\nAnswer:"
    return query, ["no", "yes"], int(bool(record["answer"])), record["passage"]


def _casehold(record: dict) -> tuple[str, list[str], int, str]:
    context = record["context"].split("<HOLDING>")[0]
    return context, [str(e) for e in record["endings"]], int(record["label"]), record["context"]


@dataclass(frozen=True)
class TaskSpec:
    choices: int
    template: str
    group: str
    format: Callable[[dict], tuple[str, list[str], int, str]]
    delimiter: str = " "


TASKS = {
    "hellaswag": TaskSpec(4, "lm_eval_v0.4_hellaswag+bos", "source_id", _hellaswag),
    "sst2": TaskSpec(2, "lm_eval_v0.4_glue_sst2+bos", "sentence", _sst2),
    "piqa": TaskSpec(2, "lm_eval_v0.4_piqa+bos", "goal", _piqa),
    "mnli": TaskSpec(3, "lm_eval_v0.4_mnli+bos", "premise", _mnli),
    "rct": TaskSpec(5, "domain_ft_rct_section+bos", "abstract_id", _rct),
    "boolq": TaskSpec(2, "lm_eval_v0.4_boolq+bos;passage_tail_truncated_256", "passage", _boolq),
    "casehold": TaskSpec(5, "domain_ft_casehold_prefix_holding+bos", "context", _casehold, ""),
}


# --- building a task file (python -m ebl lifecycle task-data) --------------------------


def encode_pair(encode: Callable[[str], list[int]], context: str, continuation: str) -> tuple[list[int], list[int]]:
    """lm-eval ``_encode_pair``: trailing context spaces move into the continuation."""

    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation, context = context[-spaces:] + continuation, context[:-spaces]
    whole, head = encode(context + continuation), encode(context)
    return head, whole[len(head):]


def _encode_split(spec: TaskSpec, encode, examples: list[tuple[int, dict]]) -> dict:
    contexts, continuations, chars, labels, rows = [], [], [], [], []
    for row, record in examples:
        query, choices, label, _ = spec.format(record)
        if len(choices) != spec.choices or not 0 <= label < spec.choices:
            raise ValueError(f"Expected {spec.choices} choices and a valid label in source row {row}.")
        for k, choice in enumerate(choices):
            head, tail = encode_pair(encode, query, spec.delimiter + choice)
            if k == 0:
                contexts.append(head)
            elif head != contexts[-1]:
                raise ValueError(f"Expected one context encoding per example in source row {row}.")
            if not tail:
                raise ValueError(f"Expected a nonempty continuation in source row {row}.")
            continuations.append(tail)
            chars.append(len(choice))
        labels.append(label)
        rows.append(row)

    def ragged(values):
        offsets = torch.tensor([0] + [len(v) for v in values], dtype=torch.int64).cumsum(0)
        flat = torch.tensor([t for v in values for t in v], dtype=torch.int64)
        return flat, offsets

    context_ids, context_offsets = ragged(contexts)
    continuation_ids, continuation_offsets = ragged(continuations)
    return {
        "context_ids": context_ids,
        "context_offsets": context_offsets,
        "continuation_ids": continuation_ids,
        "continuation_offsets": continuation_offsets,
        "continuation_chars": torch.tensor(chars, dtype=torch.int64),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "source_rows": torch.tensor(rows, dtype=torch.int64),
    }


def split_train(spec: TaskSpec, records: list[dict], development_examples: int, seed: int) -> dict:
    """Seeded group-disjoint development hold-out; leftover examples of the last group are excluded."""

    groups: dict[str, list[int]] = {}
    for row, record in enumerate(records):
        groups.setdefault(spec.format(record)[3], []).append(row)
    names = sorted(groups)
    order = torch.randperm(len(names), generator=torch.Generator().manual_seed(seed)).tolist()
    development, excluded, used = [], [], set()
    for i in order:
        if len(development) >= development_examples:
            break
        members = groups[names[i]]
        room = development_examples - len(development)
        development += members[:room]
        excluded += members[room:]
        used.add(names[i])
    if len(development) != development_examples:
        raise ValueError(f"Expected at least {development_examples} train examples for development.")
    train = [row for name in names if name not in used for row in groups[name]]
    shuffle = torch.Generator().manual_seed(seed + 1)
    train = [train[i] for i in torch.randperm(len(train), generator=shuffle).tolist()]
    development = [development[i] for i in torch.randperm(len(development), generator=shuffle).tolist()]
    return {"train": train, "development": development, "excluded": sorted(excluded),
            "development_groups": len(used), "train_groups": len(names) - len(used)}


def build_task(*, task: str, model: str, train: list[dict], validation: list[dict], encode,
               development_examples: int, split_seed: int, sources: dict, tokenizer: dict) -> dict:
    """The choice-task document of one task (official validation = test split)."""

    spec = TASKS[task]
    parts = split_train(spec, train, development_examples, split_seed)
    examples = {
        "train": [(row, train[row]) for row in parts["train"]],
        "development": [(row, train[row]) for row in parts["development"]],
        "test": list(enumerate(validation)),
    }
    splits = {name: _encode_split(spec, encode, items) for name, items in examples.items()}
    keys = {name: {spec.format(record)[3] for _, record in items} for name, items in examples.items()}
    contexts = {name: {spec.format(record)[0] for _, record in items} for name, items in examples.items()}
    return {
        "schema": CHOICE_SCHEMA,
        "model": model,
        "task": task,
        "choices": spec.choices,
        "template": spec.template,
        "bos": BOS,
        "pad": PAD,
        "tokenizer": tokenizer,
        "sources": sources,
        "split": {
            "seed": split_seed,
            "group_key": spec.group,
            "development_examples": development_examples,
            "development_groups": parts["development_groups"],
            "train_groups": parts["train_groups"],
            "excluded_train_examples": len(parts["excluded"]),
        },
        "overlap": {
            "development_train_groups": len(keys["development"] & keys["train"]),
            "test_train_groups": len(keys["test"] & keys["train"]),
            "test_train_contexts": len(contexts["test"] & contexts["train"]),
            "test_development_contexts": len(contexts["test"] & contexts["development"]),
        },
        "splits": splits,
    }


def gold_stream(split: dict) -> torch.Tensor:
    """BOS + context + gold continuation of every example, concatenated."""

    pieces = []
    choices = (len(split["continuation_offsets"]) - 1) // len(split["labels"])
    for e, label in enumerate(split["labels"].tolist()):
        c0, c1 = split["context_offsets"][e : e + 2].tolist()
        r = e * choices + label
        t0, t1 = split["continuation_offsets"][r : r + 2].tolist()
        pieces += [torch.tensor([BOS]), split["context_ids"][c0:c1], split["continuation_ids"][t0:t1]]
    return torch.cat(pieces).to(torch.int64)


def check_task(document: dict, model: str) -> None:
    if document.get("schema") != CHOICE_SCHEMA or document.get("model") != model:
        raise ValueError(f"Expected a {CHOICE_SCHEMA} task file tokenized for {model}.")
    if document.get("task") not in TASKS or document.get("choices") != TASKS[document["task"]].choices:
        raise ValueError("Expected a known task with its declared number of choices.")
    for name in CHOICE_SPLITS:
        split = document["splits"].get(name)
        if not isinstance(split, dict):
            raise ValueError(f"Expected a {name} split in the task file.")
        examples = len(split["labels"])
        if (len(split["context_offsets"]) != examples + 1
                or len(split["continuation_offsets"]) != examples * document["choices"] + 1
                or len(split["continuation_chars"]) != examples * document["choices"]):
            raise ValueError(f"Expected consistent ragged offsets in the {name} split.")


# --- rows and batches ----------------------------------------------------------------


def cohort(document: dict, split: str, count: int, max_positions: int) -> dict:
    """The first ``count`` examples of a split as (example, choice) rows with receipts."""

    from core.tensor_identity import tensor_hash

    data = document["splits"][split]
    choices = document["choices"]
    available = len(data["labels"])
    if count > available:
        raise ValueError(f"Expected at least {count} {split} task examples. Provided value: {available}.")
    tokens, offsets, contexts = [], [0], []
    for e in range(count):
        c0, c1 = data["context_offsets"][e : e + 2].tolist()
        context = data["context_ids"][c0:c1]
        for k in range(choices):
            t0, t1 = data["continuation_offsets"][e * choices + k : e * choices + k + 2].tolist()
            row = torch.cat([torch.tensor([document["bos"]]), context, data["continuation_ids"][t0:t1]])
            if len(row) > max_positions:
                raise ValueError(f"Expected task rows within {max_positions} positions. Provided value: {len(row)}.")
            tokens.append(row)
            offsets.append(offsets[-1] + len(row))
            contexts.append(1 + c1 - c0)
    flat = torch.cat(tokens).to(torch.int64)
    lengths = torch.tensor([len(row) for row in tokens], dtype=torch.int64)
    value = {
        "choices": {
            "tokens": flat,
            "offsets": torch.tensor(offsets, dtype=torch.int64),
            "context_lengths": torch.tensor(contexts, dtype=torch.int64),
            "chars": data["continuation_chars"][: count * choices].clone(),
            "labels": data["labels"][:count].clone(),
            "k": choices,
            "pad": document["pad"],
        },
        "cohort": {
            "split": split,
            "task": document["task"],
            "count": count,
            "rows": count * choices,
            "max_tokens": int(lengths.max()),
            "scored_tokens": int((lengths - torch.tensor(contexts)).sum()),
            "tokens_sha256": tensor_hash(flat),
            "labels_sha256": tensor_hash(data["labels"][:count]),
        },
    }
    return value


def batches(choices: dict, token_budget: int = TOKEN_BUDGET) -> list[torch.Tensor]:
    """Deterministic row batches, longest rows first, each within the padded token budget."""

    lengths = choices["offsets"].diff()
    order = torch.sort(lengths, descending=True, stable=True).indices.tolist()
    result, current, longest = [], [], 0
    for r in order:
        size = int(lengths[r])
        if current and max(longest, size) * (len(current) + 1) > token_budget:
            result.append(torch.tensor(current))
            current, longest = [], 0
        current.append(r)
        longest = max(longest, size)
    if current:
        result.append(torch.tensor(current))
    return result


def padded(choices: dict, rows: torch.Tensor) -> dict:
    """Right-padded tokens of some rows; causal attention keeps padding out of scored positions."""

    offsets = choices["offsets"]
    lengths = offsets.diff()[rows]
    width = int(lengths.max())
    tokens = torch.full((len(rows), width), choices["pad"], dtype=torch.int64)
    for i, r in enumerate(rows.tolist()):
        tokens[i, : int(lengths[i])] = choices["tokens"][offsets[r] : offsets[r + 1]]
    return {"tokens": tokens, "lengths": lengths, "context_lengths": choices["context_lengths"][rows]}


def scored_mask(batch: dict, device) -> torch.Tensor:
    """[rows, width-1] mask of logit positions t-1 that predict a continuation token t."""

    width = batch["tokens"].shape[1]
    positions = torch.arange(1, width, device=device)[None, :]
    return (positions >= batch["context_lengths"].to(device)[:, None]) & (
        positions < batch["lengths"].to(device)[:, None])


def continuation_log_probs(logits, tokens, mask):
    """Vocabulary log-probabilities at the scored positions and of their target tokens."""

    log_probs = logits[mask].log_softmax(-1)
    targets = tokens[:, 1:][mask]
    return log_probs, log_probs.gather(1, targets[:, None]).squeeze(1)


def row_sums(target_log_probs, mask) -> torch.Tensor:
    """Float64 sum of each row's continuation log-probabilities (differentiable)."""

    rows = torch.arange(mask.shape[0], device=mask.device)[:, None].expand_as(mask)[mask]
    total = torch.zeros(mask.shape[0], dtype=torch.float64, device=mask.device)
    return total.index_add(0, rows, target_log_probs.double())


def decision_scores(sums, chars, decision):
    return sums / chars.to(sums) if decision == "acc_norm" else sums


def choice_loss(sums, labels, k):
    """Multiple-choice cross entropy over summed continuation log-likelihoods (teacher fine-tuning)."""

    return F.cross_entropy(sums.view(-1, k), labels)


# --- metrics ------------------------------------------------------------------------


def _kl(teacher_scores, student_scores) -> torch.Tensor:
    target = teacher_scores.log_softmax(1)
    return (target.exp() * (target - student_scores.log_softmax(1))).sum(1)


def finish(student, teacher, choices: dict, *, token_kl: float, scored: int, decision: str) -> dict:
    """Task metrics of the student against the teacher (both [rows] float64 log-likelihood sums)."""

    from core.tensor_identity import tensor_hash

    k, labels = choices["k"], choices["labels"]
    chars = choices["chars"].view(-1, k).double()
    student, teacher = student.view(-1, k).cpu(), teacher.view(-1, k).cpu()
    if not (bool(torch.isfinite(student).all()) and bool(torch.isfinite(teacher).all())):
        raise FloatingPointError("Nonfinite task scores.")
    raw, norm = student.argmax(1), (student / chars).argmax(1)
    teacher_decision = decision_scores(teacher, chars, decision).argmax(1)
    predictions = norm if decision == "acc_norm" else raw
    examples = len(labels)
    return {
        "examples": examples,
        "rows": examples * k,
        "scored_tokens": scored,
        "accuracy_percent": 100 * float((raw == labels).double().mean()),
        "accuracy_norm_percent": 100 * float((norm == labels).double().mean()),
        "choice_kl": float(_kl(teacher, student).mean()),
        "choice_kl_norm": float(_kl(teacher / chars, student / chars).mean()),
        "teacher_agreement_percent": 100 * float((predictions == teacher_decision).double().mean()),
        "teacher_kl": token_kl / scored,
        "predictions": "".join(str(int(v)) for v in predictions),
        "prediction_sha256": tensor_hash(predictions),
    }


def teacher_finish(teacher, choices: dict, *, decision: str) -> dict:
    k, labels = choices["k"], choices["labels"]
    chars = choices["chars"].view(-1, k).double()
    teacher = teacher.view(-1, k).cpu()
    raw, norm = teacher.argmax(1), (teacher / chars).argmax(1)
    predictions = norm if decision == "acc_norm" else raw
    return {
        "examples": len(labels),
        "accuracy_percent": 100 * float((raw == labels).double().mean()),
        "accuracy_norm_percent": 100 * float((norm == labels).double().mean()),
        "predictions": "".join(str(int(v)) for v in predictions),
    }


def chance_percent(k: int) -> float:
    return 100 / k


def paired_standard_error(a: str, b: str, labels: torch.Tensor) -> tuple[float, float]:
    """Paired accuracy difference (a - b, percentage points) and its discordant-pair SE."""

    truth = labels.tolist()
    right_a = [int(p) == t for p, t in zip(a, truth)]
    right_b = [int(p) == t for p, t in zip(b, truth)]
    n = len(truth)
    only_a = sum(x and not y for x, y in zip(right_a, right_b))
    only_b = sum(y and not x for x, y in zip(right_a, right_b))
    delta = 100 * (only_a - only_b) / n
    se = 100 * math.sqrt(max(only_a + only_b - (only_a - only_b) ** 2 / n, 0.0)) / n
    return delta, se
