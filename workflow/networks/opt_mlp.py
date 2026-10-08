"""``opt_mlp_suffix``: OPT decoder whose last N decoder MLPs are crossbar matrices.

The pretrained OPT decoder stays complete and FP32. Embeddings, decoder layers
before the suffix, every attention block, the LayerNorms, the MLP biases and
the tied LM head stay digital. In each of the last ``analog_decoder_layers``
decoder layers, ``fc1`` and ``fc2`` become crossbar matrices: each weight is
divided by its matrix's absolute maximum into targets ``q`` in [-1, 1] and
split into square tiles whose input slices are summed digitally, exactly as
for the ResNet suffix. The ReLU between them is digital.

Data are token cohorts of one pinned, pre-tokenized corpus, cut into
non-overlapping sequences. The frozen prefix and the teacher logits are
recomputed for each minibatch instead of cached: the per-token teacher
distribution (50,272 classes) is too large to store. Within a sequence,
positions 0..T-2 are scored against their next token; teacher KL, cross
entropy and perplexity are token-weighted over exactly those positions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.artifacts import sha256_file
from experiments.cifar_crossbar.data import cohort_receipt
from experiments.cifar_crossbar.model import MatrixSpec, state_hash, tensor_hash
from experiments.cifar_crossbar.runtime import load

SPLITS = ("hwa", "training", "development", "test")
SUMMARY_METRICS = ("teacher_kl", "perplexity")
PRESENTATIONS = "sequence_presentations"
CORPUS_SCHEMA = "crossbar_lifecycle.token_corpus.v1"
CORPUS_SPLITS = ("train", "validation", "test")
EVALUATION_BATCH = 4
# A clean digital suffix differs from the teacher only by tiling/scaling rounding.
REPRODUCTION_KL = 1e-6


@dataclass(frozen=True)
class OptConfig:
    vocab_size: int
    hidden_size: int
    decoder_layers: int
    attention_heads: int
    ffn_dim: int
    max_positions: int


# Hugging Face revision and pytorch_model.bin digest of each pinned checkpoint.
PINNED = {
    "facebook/opt-125m": {
        "revision": "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
        "sha256": "2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9",
        "config": OptConfig(50272, 768, 12, 12, 3072, 2048),
    }
}


# --- the digital OPT decoder ---------------------------------------------------


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        b, t, d = x.shape

        def heads(value):
            return value.view(b, t, self.heads, self.head_dim).transpose(1, 2)

        q = heads(self.q_proj(x) * self.head_dim**-0.5)
        k, v = heads(self.k_proj(x)), heads(self.v_proj(x))
        scores = q @ k.transpose(-1, -2)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        weights = scores.masked_fill(causal, float("-inf")).softmax(-1)
        return self.out_proj((weights @ v).transpose(1, 2).reshape(b, t, d))


class DecoderLayer(nn.Module):
    """Pre-LayerNorm OPT decoder layer with a ReLU MLP."""

    def __init__(self, config: OptConfig):
        super().__init__()
        self.self_attn = Attention(config.hidden_size, config.attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def attention(self, x):
        return x + self.self_attn(self.self_attn_layer_norm(x))

    def forward(self, x):
        x = self.attention(x)
        return x + self.fc2(F.relu(self.fc1(self.final_layer_norm(x))))


class OptDecoder(nn.Module):
    """FP32 OPT decoder with learned positions (offset 2) and a tied LM head."""

    POSITION_OFFSET = 2

    def __init__(self, config: OptConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_positions + self.POSITION_OFFSET, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.decoder_layers))
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def embed(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device) + self.POSITION_OFFSET
        return self.embed_tokens(tokens) + self.embed_positions(positions)

    def logits(self, hidden):
        return self.final_layer_norm(hidden) @ self.embed_tokens.weight.T

    def forward(self, tokens):
        hidden = self.embed(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.logits(hidden)


def from_hugging_face(state: dict, config: OptConfig) -> OptDecoder:
    """Build the decoder from an ``OPTForCausalLM`` state dict (any float dtype)."""

    decoder = {k.removeprefix("model.decoder."): v for k, v in state.items() if k.startswith("model.decoder.")}
    head = state.get("lm_head.weight")
    if head is not None and not torch.equal(head, decoder["embed_tokens.weight"]):
        raise ValueError("Expected the LM head to be tied to the token embedding.")
    model = OptDecoder(config)
    model.load_state_dict({k: v.float() for k, v in decoder.items()}, strict=True)
    return model


# --- the crossbar student --------------------------------------------------------


class OptMlpSuffix(nn.Module):
    """Analog fc1/fc2 of the last decoder layers; everything else is the frozen teacher.

    The teacher is held by reference, not registered, so state dicts contain
    only the crossbar targets and the calibration parameters: per-matrix output
    gains, the MLP biases and the LayerNorm in front of each analog MLP.
    """

    def __init__(self, teacher: OptDecoder, analog_layers: int, tile_size: int = 512):
        super().__init__()
        depth = len(teacher.layers)
        if not 1 <= analog_layers <= depth or tile_size < 1:
            raise ValueError("Expected between one and all decoder layers and a positive tile size.")
        self.start, self.tile_size = depth - analog_layers, tile_size
        self.suffix = f"last_{analog_layers}_decoder_mlps"
        self._frozen = (teacher,)
        self.layout: list[MatrixSpec] = []
        values, biases, offset = [], [], 0
        for i in range(self.start, depth):
            for name in ("fc1", "fc2"):
                module = teacher.layers[i].get_submodule(name)
                weight = module.weight.detach()
                scale = max(float(weight.abs().max()), 1e-12)
                self.layout.append(
                    MatrixSpec(f"decoder.layers.{i}.{name}", tuple(weight.shape), offset, weight.numel(), scale)
                )
                values.append(weight.flatten() / scale)
                biases.append(nn.Parameter(module.bias.detach().clone()))
                offset += weight.numel()
        self.q = nn.Parameter(torch.cat(values))
        self.register_buffer("scales", torch.tensor([x.scale for x in self.layout], device=self.q.device))
        self.output_gains = nn.Parameter(torch.ones(len(self.layout), device=self.q.device))
        self.mlp_biases = nn.ParameterList(biases)
        self.mlp_norms = nn.ModuleList(
            copy.deepcopy(teacher.layers[i].final_layer_norm) for i in range(self.start, depth)
        )
        self.read_noise = 0.0
        self.read_generator = torch.Generator(device=self.q.device).manual_seed(0)
        self.enable_calibration(False)

    @property
    def teacher(self) -> OptDecoder:
        return self._frozen[0]

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def enable_calibration(self, enabled: bool):
        self.output_gains.requires_grad_(enabled)
        self.mlp_biases.requires_grad_(enabled)
        self.mlp_norms.requires_grad_(enabled)

    def calibration_parameters(self):
        return [p for p in self.parameters() if p is not self.q and p.requires_grad]

    def prefix_hash(self):
        """Every frozen digital parameter: embeddings, prefix, attention, LayerNorms and head."""

        return state_hash(self.teacher.state_dict())

    @torch.no_grad()
    def features(self, tokens):
        hidden = self.teacher.embed(tokens)
        for i in range(self.start):
            hidden = self.teacher.layers[i](hidden)
        return hidden

    @torch.no_grad()
    def teacher_logits(self, hidden):
        """Digital teacher logits at the scored positions from the shared prefix state."""

        for i in range(self.start, len(self.teacher.layers)):
            hidden = self.teacher.layers[i](hidden)
        return self.teacher.logits(hidden[:, :-1])

    def _matrix(self, x, index, q):
        spec = self.layout[index]
        matrix = q[spec.offset : spec.offset + spec.size].reshape(spec.shape)
        flat = x.reshape(-1, spec.shape[1])
        columns = []
        for output in range(0, spec.shape[0], self.tile_size):
            accum = None
            for row in range(0, spec.shape[1], self.tile_size):
                part = flat[:, row : row + self.tile_size] @ matrix[
                    output : output + self.tile_size, row : row + self.tile_size
                ].T
                if self.read_noise:
                    part = part + self.read_noise * torch.randn(
                        part.shape, device=part.device, generator=self.read_generator
                    )
                accum = part if accum is None else accum + part
            columns.append(accum)
        result = torch.cat(columns, dim=1) * self.scales[index] * self.output_gains[index] + self.mlp_biases[index]
        return result.reshape(*x.shape[:-1], spec.shape[0])

    def forward_features(self, hidden, q=None):
        """Student logits at the scored positions from the frozen prefix state."""

        q = self.q if q is None else q
        if q.shape != self.q.shape:
            raise ValueError("Expected one normalized value per analog MLP weight.")
        for k, i in enumerate(range(self.start, len(self.teacher.layers))):
            hidden = self.teacher.layers[i].attention(hidden)
            y = self._matrix(self.mlp_norms[k](hidden), 2 * k, q)
            hidden = hidden + self._matrix(F.relu(y), 2 * k + 1, q)
        return self.teacher.logits(hidden[:, :-1])

    def forward(self, tokens, q=None):
        return self.forward_features(self.features(tokens), q)

    def mapping_receipt(self):
        tiles = []
        for spec in self.layout:
            for output in range(0, spec.shape[0], self.tile_size):
                for row in range(0, spec.shape[1], self.tile_size):
                    tiles.append(
                        {
                            "module": spec.name,
                            "input_start": row,
                            "output_start": output,
                            "rows": min(self.tile_size, spec.shape[1] - row),
                            "columns": min(self.tile_size, spec.shape[0] - output),
                        }
                    )
        return {
            "suffix": self.suffix,
            "matrices": [
                {"name": s.name, "shape": list(s.shape), "offset": s.offset, "size": s.size, "scale": s.scale}
                for s in self.layout
            ],
            "logical_weights": self.q.numel(),
            "logical_tiles": tiles,
            "tile_capacity": self.tile_size**2,
            "prefix_sha256": self.prefix_hash(),
        }


# --- family interface: teacher and network --------------------------------------


def pinned_digest(network) -> str:
    return PINNED[network.model]["sha256"]


def load_teacher(path, network, device):
    """Load the pinned Hugging Face ``pytorch_model.bin`` as an FP32 decoder."""

    path = Path(path)
    digest = sha256_file(path)
    pinned = PINNED[network.model]
    if digest != pinned["sha256"]:
        raise ValueError(
            f"Expected {network.model} pytorch_model.bin at revision {pinned['revision']} "
            f"(SHA-256 {pinned['sha256']}). Provided value: {path} with SHA-256 {digest}."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    teacher = from_hugging_face(state, pinned["config"]).to(device)
    teacher.eval().requires_grad_(False)
    return teacher, digest


def build_network(teacher, lifecycle) -> OptMlpSuffix:
    return OptMlpSuffix(teacher, lifecycle.network.analog_decoder_layers, lifecycle.deployment.tile_size)


# --- token corpus and cohorts ---------------------------------------------------


def corpus_path(data) -> Path:
    root = os.environ.get("EBL_CORPUS_ROOT")
    if not root:
        raise ValueError("Expected EBL_CORPUS_ROOT to name the directory of pinned token corpora.")
    return Path(root) / f"{data.corpus}.pt"


def load_corpus(data, model: str) -> tuple[dict, Path]:
    path = corpus_path(data)
    digest = sha256_file(path)
    if digest != data.corpus_sha256:
        raise ValueError(
            f"Expected corpus {data.corpus!r} with SHA-256 {data.corpus_sha256}. "
            f"Provided value: {path} with SHA-256 {digest}."
        )
    corpus = load(path)
    if corpus.get("schema") != CORPUS_SCHEMA or corpus.get("model") != model:
        raise ValueError(f"Expected a {CORPUS_SCHEMA} corpus tokenized for {model}.")
    for split in CORPUS_SPLITS:
        stream = corpus["splits"].get(split)
        if not isinstance(stream, torch.Tensor) or stream.dtype != torch.int64 or stream.dim() != 1:
            raise ValueError(f"Expected a one-dimensional int64 {split} token stream.")
    return corpus, path


def _cohort(tokens, split, indices) -> dict:
    return {
        "tokens": tokens.clone(),
        "cohort": {
            "split": split,
            "count": len(tokens),
            "sequence_length": tokens.shape[1],
            "scored_tokens": len(tokens) * (tokens.shape[1] - 1),
            "block_indices_sha256": cohort_receipt(indices)["indices_sha256"],
            "tokens_sha256": tensor_hash(tokens),
        },
    }


def build_cache(teacher, network, lifecycle, device) -> dict:
    """Token cohorts of the pinned corpus; the prefix runs per minibatch."""

    data = lifecycle.data
    corpus, path = load_corpus(data, lifecycle.network.model)
    length = data.sequence_length
    vocabulary = teacher.config.vocab_size

    def blocks(split):
        stream = corpus["splits"][split]
        if int(stream.min()) < 0 or int(stream.max()) >= vocabulary:
            raise ValueError(f"Expected {split} token ids inside the {vocabulary}-token vocabulary.")
        count = len(stream) // length
        return stream[: count * length].reshape(count, length)

    def first(tokens, split, count):
        if count > len(tokens):
            raise ValueError(
                f"Expected at least {count} {split} sequences of {length} tokens. Provided value: {len(tokens)}."
            )
        return _cohort(tokens[:count], split, range(count))

    train = blocks("train")
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(data.data_seed))
    cache = {"corpus": {"name": data.corpus, "sha256": data.corpus_sha256, "path": str(path)}}
    for name in ("hwa", "training"):
        count = data.cohort_size(name)
        if count > len(train):
            raise ValueError(
                f"Expected at least {count} train sequences of {length} tokens. Provided value: {len(train)}."
            )
        cache[name] = _cohort(train[order[:count]], "train", order[:count].tolist())
    cache["development"] = first(blocks("validation"), "validation", data.cohort_size("development"))
    if data.evaluation == "test":
        cache["test"] = first(blocks("test"), "test", data.cohort_size("test"))
    return cache


def check_cache(cache: dict, lifecycle) -> None:
    data = lifecycle.data
    names = ("hwa", "training", "development") + (("test",) if data.evaluation == "test" else ())
    for name in names:
        expected = (data.cohort_size(name), data.sequence_length)
        if name not in cache or tuple(cache[name]["tokens"].shape) != expected:
            raise ValueError(f"Expected the {name} cohort to contain {expected[0]} sequences of {expected[1]} tokens.")
    if cache["corpus"]["sha256"] != data.corpus_sha256:
        raise ValueError("Expected the cache to come from the lifecycle corpus.")


def to_device(cache: dict, device) -> dict:
    moved = dict(cache)
    for name in SPLITS:
        if name in cache:
            moved[name] = {**cache[name], "tokens": cache[name]["tokens"].to(device)}
    return moved


def cohorts(cache: dict) -> dict:
    return {name: cache[name]["cohort"] for name in SPLITS if name in cache}


# --- batches, objective and metrics ---------------------------------------------


def examples(split: dict) -> int:
    return len(split["tokens"])


def batch(network, split: dict, index, device):
    """Frozen prefix state, teacher logits and next-token labels of one minibatch."""

    tokens = split["tokens"][index].to(device)
    hidden = network.features(tokens)
    return hidden, network.teacher_logits(hidden), tokens[:, 1:]


def objective(logits, teacher_logits, labels, name):
    """Token-weighted teacher KL or next-token cross entropy."""

    logits = logits.flatten(0, 1)
    if name == "teacher_kl":
        return F.kl_div(logits.log_softmax(1), teacher_logits.flatten(0, 1).softmax(1), reduction="batchmean")
    return F.cross_entropy(logits, labels.flatten())


def hwa_inputs(cache: dict, device) -> dict:
    return {"tokens": cache["hwa"]["tokens"].to(device)}


def hwa_examples(inputs: dict) -> int:
    return len(inputs["tokens"])


def hwa_batch(inputs: dict, index, generator, network, teacher, lifecycle):
    """No augmentation is defined for text; the epoch generator orders batches only."""

    return batch(network, inputs, index, inputs["tokens"].device)


@torch.no_grad()
def evaluate(network, split: dict, device, *, q=None, batch_size=EVALUATION_BATCH, read_seed=50001) -> dict:
    """Token-weighted metrics of the student (held state ``q``) against the teacher."""

    old_rng = network.read_generator.get_state()
    network.read_generator.manual_seed(read_seed)
    network.eval()
    kl = ce = 0.0
    scored = correct = agreement = 0
    predictions = []
    try:
        for begin in range(0, len(split["tokens"]), batch_size):
            tokens = split["tokens"][begin : begin + batch_size].to(device)
            hidden = network.features(tokens)
            teacher = network.teacher_logits(hidden)
            out = network.forward_features(hidden, q)
            if not bool(torch.isfinite(out).all()):
                raise FloatingPointError("Nonfinite network scores.")
            labels = tokens[:, 1:]
            flat = out.flatten(0, 1)
            kl += float(F.kl_div(flat.log_softmax(1), teacher.flatten(0, 1).softmax(1), reduction="sum"))
            ce += float(F.cross_entropy(flat, labels.flatten(), reduction="sum"))
            p = out.argmax(-1)
            predictions.append(p.cpu())
            correct += int((p == labels).sum())
            agreement += int((p == teacher.argmax(-1)).sum())
            scored += labels.numel()
    finally:
        network.read_generator.set_state(old_rng)
    return {
        "sequences": len(split["tokens"]),
        "scored_tokens": scored,
        "teacher_kl": kl / scored,
        "cross_entropy": ce / scored,
        "perplexity": math.exp(ce / scored),
        "next_token_accuracy_percent": 100 * correct / scored,
        "teacher_agreement_percent": 100 * agreement / scored,
        "prediction_sha256": tensor_hash(torch.cat(predictions)),
    }


@torch.no_grad()
def teacher_metrics(network, split: dict, device, *, batch_size=EVALUATION_BATCH) -> dict:
    ce, scored, correct = 0.0, 0, 0
    for begin in range(0, len(split["tokens"]), batch_size):
        tokens = split["tokens"][begin : begin + batch_size].to(device)
        teacher = network.teacher_logits(network.features(tokens))
        labels = tokens[:, 1:]
        ce += float(F.cross_entropy(teacher.flatten(0, 1), labels.flatten(), reduction="sum"))
        correct += int((teacher.argmax(-1) == labels).sum())
        scored += labels.numel()
    return {
        "scored_tokens": scored,
        "cross_entropy": ce / scored,
        "perplexity": math.exp(ce / scored),
        "next_token_accuracy_percent": 100 * correct / scored,
    }


def prepare_report(teacher, network, cache: dict, lifecycle, device) -> dict:
    """The clean digital suffix must reproduce the teacher on the evaluation split."""

    split = lifecycle.data.evaluation
    digital = evaluate(network, cache[split], device)
    if not digital["teacher_kl"] <= REPRODUCTION_KL:
        raise RuntimeError(
            f"The clean digital suffix does not reproduce the teacher: KL {digital['teacher_kl']:.3g}."
        )
    return {"digital": digital, "teacher": teacher_metrics(network, cache[split], device), "corpus": cache["corpus"]}


def deployment_summary(clean: dict, initial: dict) -> dict:
    return {
        "deployment_teacher_kl": initial["teacher_kl"],
        "deployment_perplexity_ratio": initial["perplexity"] / clean["perplexity"],
    }


# --- corpus preparation (python -m workflow corpus) ------------------------------


def build_corpus(*, model: str, model_dir: Path, texts: dict) -> dict:
    """Tokenize train/validation/test text files with the pinned OPT tokenizer.

    Each split is tokenized as one stream with the tokenizer's default special
    tokens (a single BOS at the start of the split). Requires ``transformers``.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.model_max_length = 1 << 62
    splits, sources = {}, {}
    for split in CORPUS_SPLITS:
        path = Path(texts[split])
        ids = tokenizer(path.read_text(encoding="utf-8"))["input_ids"]
        splits[split] = torch.tensor(ids, dtype=torch.int64)
        sources[split] = {"file": path.name, "sha256": sha256_file(path)}
    files = ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json")
    return {
        "schema": CORPUS_SCHEMA,
        "model": model,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "special_tokens": "tokenizer default: one BOS at the start of each split",
            "files": {name: sha256_file(Path(model_dir) / name) for name in files if (Path(model_dir) / name).exists()},
        },
        "sources": sources,
        "splits": splits,
    }
