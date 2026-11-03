"""methods.teacher_lora — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from methods.teacher_config import TARGETS, TeacherConfig


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

    import networks.tasks as tasks

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

    import networks.tasks as tasks

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
    import networks.tasks as tasks

    model.eval()
    sums = torch.zeros(len(choices["offsets"]) - 1, dtype=torch.float64)
    for index in tasks.batches(choices):
        sums[index] = row_sums(model, tasks.padded(choices, index), device).cpu()
    return sums


def development_metrics(sums, choices, decision) -> dict:
    import networks.tasks as tasks

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


def _schedule(optimizer, warmup: int, total: int):
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


__all__ = ["EXPERIMENT_ID", "TeacherConfig", "parse_config", "resolve_config", "run_train"]
