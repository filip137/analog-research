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
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from core.artifacts import sha256_file
from core.cohorts import cohort_receipt
from hardware.mapping import MatrixSpec
from core.tensor_identity import state_hash, tensor_hash
from core.tensor_io_native import load
from hardware.config import SCALINGS

SPLITS = ("hwa", "training", "development", "test")
SUMMARY_METRICS = ("teacher_kl", "perplexity")
# Labelled multiple-choice cohorts of a declared ``task`` (networks.tasks).
TASK_SPLITS = ("task_development", "task_test")
TASK_SUMMARY_METRICS = ("accuracy_norm_percent", "accuracy_percent", "choice_kl", "teacher_kl")
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


class StudentInput(NamedTuple):
    """A whole-model student's minibatch: its own tokens and the teacher's prefix state."""

    tokens: torch.Tensor
    teacher_hidden: torch.Tensor


class OptMlpSuffix(nn.Module):
    """Analog fc1/fc2 of the last decoder layers; everything else is the frozen teacher.

    The teacher is held by reference, not registered, so state dicts contain
    only the crossbar targets and the calibration parameters: per-matrix output
    gains, the MLP biases and the LayerNorm in front of each analog MLP.

    ``scaling`` maps each matrix to normalized targets in [-1, 1]: by its own
    abs-max (``per_layer_absmax``, one digital output scale per matrix) or by
    each output channel's abs-max (``per_output_channel_absmax``, a fixed
    digital output scale per channel, the AIHWKit column-wise convention).

    ``scope="model"`` gives the student its own copy of every digital parameter
    (``digital``: embeddings, the earlier layers, all attention and LayerNorms
    and the tied head), trainable for whole-model HWA and carried in the state
    dict. The frozen teacher then only supplies the reference logits. The copy
    omits the analog layers' fc1/fc2 and their LayerNorm, which the crossbar
    targets, ``mlp_biases`` and ``mlp_norms`` replace.
    """

    def __init__(
        self,
        teacher: OptDecoder,
        analog_layers: int,
        tile_size: int = 512,
        scaling: str = "per_layer_absmax",
        scope: str = "analog",
        analog_lora=None,
    ):
        super().__init__()
        depth = len(teacher.layers)
        if not 1 <= analog_layers <= depth or tile_size < 1:
            raise ValueError("Expected between one and all decoder layers and a positive tile size.")
        if scaling not in SCALINGS:
            raise ValueError(f"Expected a scaling in {SCALINGS}. Provided value: {scaling!r}.")
        if scope not in ("analog", "model"):
            raise ValueError(f"Expected an HWA scope of analog or model. Provided value: {scope!r}.")
        self.scaling, self.scope = scaling, scope
        self.start, self.tile_size = depth - analog_layers, tile_size
        self.suffix = f"last_{analog_layers}_decoder_mlps"
        self._frozen = (teacher,)
        self.layout: list[MatrixSpec] = []
        values, biases, rows, offset = [], [], [], 0
        for i in range(self.start, depth):
            for name in ("fc1", "fc2"):
                module = teacher.layers[i].get_submodule(name)
                weight = module.weight.detach()
                scale = max(float(weight.abs().max()), 1e-12)
                self.layout.append(
                    MatrixSpec(f"decoder.layers.{i}.{name}", tuple(weight.shape), offset, weight.numel(), scale)
                )
                if scaling == "per_output_channel_absmax":
                    row = weight.abs().amax(1).clamp_min(1e-12)
                    rows.append(row)
                    values.append((weight / row[:, None]).flatten())
                else:
                    values.append(weight.flatten() / scale)
                biases.append(nn.Parameter(module.bias.detach().clone()))
                offset += weight.numel()
        # Analog LoRA: A (rank x in) and B (out x rank) of every base matrix follow the base matrices
        # in the layout, so programming, faults and rewrites treat them as arrays. They start at zero
        # (an exact no-op) with unit channel scales; adapter training sets both.
        self.base_matrices = len(self.layout)
        self.adapters, self.adapter_scale = {}, 0.0
        if analog_lora is not None:
            self.adapter_scale = analog_lora.alpha / analog_lora.rank
            for k, spec in enumerate(list(self.layout)):
                outputs, inputs = spec.shape
                pair = []
                for suffix, shape in (("lora_a", (analog_lora.rank, inputs)), ("lora_b", (outputs, analog_lora.rank))):
                    self.layout.append(MatrixSpec(f"{spec.name}.{suffix}", shape, offset, shape[0] * shape[1], 1.0))
                    values.append(torch.zeros(shape[0] * shape[1], device=teacher.embed_tokens.weight.device))
                    if scaling == "per_output_channel_absmax":
                        rows.append(torch.ones(shape[0], device=teacher.embed_tokens.weight.device))
                    pair.append(len(self.layout) - 1)
                    offset += shape[0] * shape[1]
                self.adapters[k] = tuple(pair)
        self.q = nn.Parameter(torch.cat(values))
        self.register_buffer("scales", torch.tensor([x.scale for x in self.layout], device=self.q.device))
        if rows:
            # Fixed digital output scale of every output channel, concatenated in layout order.
            self.register_buffer("channel_scales", torch.cat(rows))
            self.channel_offsets = [0]
            for spec in self.layout:
                self.channel_offsets.append(self.channel_offsets[-1] + spec.shape[0])
        self.output_gains = nn.Parameter(torch.ones(self.base_matrices, device=self.q.device))
        self.mlp_biases = nn.ParameterList(biases)
        self.mlp_norms = nn.ModuleList(
            copy.deepcopy(teacher.layers[i].final_layer_norm) for i in range(self.start, depth)
        )
        self.read_noise = 0.0
        self.read_generator = torch.Generator(device=self.q.device).manual_seed(0)
        self.digital = None
        if scope == "model":
            self.digital = copy.deepcopy(teacher)
            for i in range(self.start, depth):
                layer = self.digital.layers[i]
                layer.fc1 = layer.fc2 = layer.final_layer_norm = None
        self.enable_calibration(False)
        self.enable_digital(False)
        self.lora = None
        self.lora_multiplier = 0.0
        self.drift_compensation = None

    def enable_lora(self, config, seed=0):
        """Digital corrections to the analog MLPs, in logical weight units.

        A is uniform +/-1/sqrt(fan_in), B is zero; alpha/r scaling, no
        dropout. CPU initialization makes adapter initialization portable.
        The correction precedes the frozen output gain and remains digital.
        """
        self.lora = None
        self.lora_multiplier = 0.0
        if config is None:
            return
        generator = torch.Generator().manual_seed(seed)
        modules = []
        for spec in self.layout:
            outputs, inputs = spec.shape
            a = torch.empty(config.rank, inputs).uniform_(-inputs ** -0.5, inputs ** -0.5, generator=generator)
            modules.append(nn.ParameterDict({"a": nn.Parameter(a.to(self.q.device)),
                "b": nn.Parameter(torch.zeros(outputs, config.rank, device=self.q.device))}))
        self.lora = nn.ModuleList(modules)
        self.lora_multiplier = config.alpha / config.rank

    def base_state_hash(self):
        return state_hash({k: v for k, v in self.state_dict().items() if not k.startswith("lora.")})

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

    def enable_digital(self, enabled: bool):
        if self.digital is not None:
            self.digital.requires_grad_(enabled)

    def digital_parameters(self):
        return [] if self.digital is None else list(self.digital.parameters())

    def calibration_parameters(self):
        digital = {id(p) for p in self.digital_parameters()}
        return [p for p in self.parameters() if p is not self.q and p.requires_grad and id(p) not in digital]

    def prefix_hash(self):
        """Every frozen digital parameter: embeddings, prefix, attention, LayerNorms and head."""

        return state_hash(self.teacher.state_dict())

    @torch.no_grad()
    def features(self, tokens):
        hidden = self.teacher.embed(tokens)
        for i in range(self.start):
            hidden = self.teacher.layers[i](hidden)
        # A whole-model student runs its own (trainable) prefix from the tokens.
        return hidden if self.digital is None else StudentInput(tokens, hidden)

    @torch.no_grad()
    def teacher_logits(self, hidden):
        """Digital teacher logits at the scored positions from the shared prefix state."""

        if isinstance(hidden, StudentInput):
            hidden = hidden.teacher_hidden
        for i in range(self.start, len(self.teacher.layers)):
            hidden = self.teacher.layers[i](hidden)
        return self.teacher.logits(hidden[:, :-1])

    def _matrix(self, x, index, q):
        spec = self.layout[index]
        flat = x.reshape(-1, spec.shape[1])
        if self.drift_compensation is not None and (self.read_noise or self.lora is not None or self.adapters):
            raise ValueError("Drift calibration requires held readouts without per-MVM noise or LoRA.")
        logical = self._logical(flat, index, q)
        if self.adapters:
            # The analog adapter path x -> A -> B, digitized between the two arrays and scaled by alpha/r.
            a, b = self.adapters[index]
            logical = logical + self._logical(self._logical(flat, a, q), b, q) * self.adapter_scale
        if self.lora is not None:
            adapter = self.lora[index]
            logical = logical + ((flat @ adapter["a"].T) @ adapter["b"].T) * self.lora_multiplier
        result = logical * self.output_gains[index] + self.mlp_biases[index]
        if self.drift_compensation is not None:
            result = self.drift_compensation.after_bias(result, index)
        return result.reshape(*x.shape[:-1], spec.shape[0])

    def _logical(self, flat, index, q):
        """One analog matrix's product in logical weight units (held state ``q``, digital output scale)."""

        spec = self.layout[index]
        matrix = q[spec.offset : spec.offset + spec.size].reshape(spec.shape)
        if self.drift_compensation is not None:
            product = self.drift_compensation.mvm(flat, index, matrix)
        elif self.read_noise:
            columns = []
            for output in range(0, spec.shape[0], self.tile_size):
                accum = None
                for row in range(0, spec.shape[1], self.tile_size):
                    part = flat[:, row : row + self.tile_size] @ matrix[
                        output : output + self.tile_size, row : row + self.tile_size
                    ].T
                    part = part + self.read_noise * torch.randn(
                        part.shape, device=part.device, generator=self.read_generator
                    )
                    accum = part if accum is None else accum + part
                columns.append(accum)
            product = torch.cat(columns, dim=1)
        else:
            # Tiles matter only for per-tile read noise; without it the tiled
            # sum is one matmul (equal up to FP32 summation order).
            product = flat @ matrix.T
        if self.scaling == "per_output_channel_absmax":
            scale = self.channel_scales[self.channel_offsets[index] : self.channel_offsets[index + 1]]
        else:
            scale = self.scales[index]
        return product * scale

    def forward_features(self, hidden, q=None):
        """Student logits at the scored positions from the frozen prefix state.

        A whole-model student instead starts from the tokens of ``hidden``
        (a ``StudentInput``) and uses its own digital parameters throughout.
        """

        q = self.q if q is None else q
        if q.shape != self.q.shape:
            raise ValueError("Expected one normalized value per analog MLP weight.")
        digital = self.teacher
        if self.digital is not None:
            if not isinstance(hidden, StudentInput):
                raise ValueError("Expected the tokens of a whole-model student (features() output).")
            digital = self.digital
            hidden = digital.embed(hidden.tokens)
            for i in range(self.start):
                hidden = digital.layers[i](hidden)
        return self.forward_suffix(hidden, q, digital=digital)

    def forward_suffix(self, hidden, q, *, digital, correction=None, capture=False):
        """Shared suffix traversal for native weights and physical drift adapters.

        Corrections act on the complete MLP output before the residual addition.
        Captures retain the normalized input and uncorrected output for fitting.
        """

        if q.shape != self.q.shape:
            raise ValueError("Expected one normalized value per analog MLP weight.")
        if correction is not None and len(correction) != len(self.mlp_norms):
            raise ValueError("Expected one correction per analog MLP.")
        captures = []
        for k, i in enumerate(range(self.start, len(self.teacher.layers))):
            hidden = digital.layers[i].attention(hidden)
            x = self.mlp_norms[k](hidden)
            y = self._matrix(x, 2 * k, q)
            y = self._matrix(F.relu(y), 2 * k + 1, q)
            if capture:
                captures.append((x, y))
            if correction is not None:
                a, b = correction[k]
                y = y * a + b
            hidden = hidden + y
        logits = digital.logits(hidden[:, :-1])
        return (logits, captures) if capture else logits

    def forward(self, tokens, q=None):
        return self.forward_features(self.features(tokens), q)

    @torch.no_grad()
    def analog_input(self, tokens):
        """Residual stream entering the first analog decoder layer, as in ``forward_features``."""

        hidden = self.features(tokens)
        if self.digital is None:
            return hidden
        hidden = self.digital.embed(hidden.tokens)
        for i in range(self.start):
            hidden = self.digital.layers[i](hidden)
        return hidden

    def mlp_branch(self, k, hidden, q):
        """Analog MLP k split for output calibration, from ``analog_input`` onwards.

        Returns the residual stream after attention, the digitized fc2 output
        before its digital bias, and that bias; ``residual + (z + bias)`` is the
        operation order of ``forward_features``.
        """

        if self.lora is not None or self.adapters or self.read_noise:
            raise ValueError("Output calibration expects the plain held-readout student.")
        if self.drift_compensation is not None:
            raise ValueError("Output calibration and matrix drift compensation are separate methods.")
        digital = self.teacher if self.digital is None else self.digital
        hidden = digital.layers[self.start + k].attention(hidden)
        y = self._matrix(self.mlp_norms[k](hidden), 2 * k, q)
        spec = self.layout[2 * k + 1]
        flat = F.relu(y).reshape(-1, spec.shape[1])
        z = self._logical(flat, 2 * k + 1, q) * self.output_gains[2 * k + 1]
        return hidden, z.reshape(*hidden.shape[:-1], spec.shape[0]), self.mlp_biases[2 * k + 1]

    def head(self, hidden):
        return (self.teacher if self.digital is None else self.digital).logits(hidden[:, :-1])

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
            **(
                {"channel_scales_sha256": tensor_hash(self.channel_scales)}
                if self.scaling == "per_output_channel_absmax"
                else {}
            ),
            "logical_tiles": tiles,
            "tile_capacity": self.tile_size**2,
            "prefix_sha256": self.prefix_hash(),
        }


# --- family interface: teacher and network --------------------------------------


def pinned_digest(network) -> str:
    """The declared fine-tuned teacher's digest, otherwise the pinned checkpoint's."""

    teacher = getattr(network, "teacher", None)
    return PINNED[network.model]["sha256"] if teacher is None else teacher.sha256


def load_teacher(path, network, device):
    """Load the pinned (or declared fine-tuned) ``pytorch_model.bin`` as an FP32 decoder."""

    path = Path(path)
    digest = sha256_file(path)
    pinned = PINNED[network.model]
    if digest != pinned_digest(network):
        source = (f"revision {pinned['revision']}" if getattr(network, "teacher", None) is None
                  else f"fine-tuned teacher {network.teacher.name!r}")
        raise ValueError(
            f"Expected {network.model} pytorch_model.bin of {source} "
            f"(SHA-256 {pinned_digest(network)}). Provided value: {path} with SHA-256 {digest}."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    teacher = from_hugging_face(state, pinned["config"]).to(device)
    teacher.eval().requires_grad_(False)
    return teacher, digest


def build_network(teacher, lifecycle, hwa_arm=None) -> OptMlpSuffix:
    """The crossbar student; a whole-model HWA arm's stages also get a digital copy."""

    deployment = lifecycle.deployment
    scope = "analog" if hwa_arm is None else lifecycle.hwa.arm(hwa_arm).scope
    return OptMlpSuffix(
        teacher, lifecycle.network.analog_decoder_layers, deployment.tile_size, deployment.scaling, scope,
        getattr(lifecycle.network, "analog_lora", None),
    )


# --- token corpus and cohorts ---------------------------------------------------


def corpus_path(data) -> Path:
    root = os.environ.get("EBL_CORPUS_ROOT")
    if not root:
        raise ValueError("Expected EBL_CORPUS_ROOT to name the directory of pinned token corpora.")
    return Path(root) / f"{data.corpus}.pt"


def load_corpus(data, model: str) -> tuple[dict, Path]:
    """The pinned corpus of ``data`` (the lifecycle data or one transfer test)."""

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
    for test in transfer_tests(lifecycle):
        other, other_path = load_corpus(test, lifecycle.network.model)
        stream = other["splits"][test.split]
        if int(stream.min()) < 0 or int(stream.max()) >= vocabulary:
            raise ValueError(f"Expected transfer test {test.label!r} token ids inside the vocabulary.")
        count = min(test.sequences, data.max_examples) if data.max_examples else test.sequences
        available = len(stream) // length
        if count > available:
            raise ValueError(
                f"Expected at least {count} {test.label} sequences of {length} tokens. Provided value: {available}."
            )
        cohort = _cohort(stream[: count * length].reshape(count, length), test.split, range(count))
        cohort["cohort"]["corpus"] = {"name": test.corpus, "sha256": test.corpus_sha256, "path": str(other_path)}
        cache[transfer_split(test)] = cohort
    import networks.context_views as context_views
    context_views.build(cache, corpus, lifecycle)
    if getattr(lifecycle, "task", None) is not None:
        _build_task_cohorts(cache, lifecycle, teacher.config.max_positions)
    return cache


def task_path(task) -> Path:
    """Task files live beside the pinned token corpora (``EBL_CORPUS_ROOT``)."""

    root = os.environ.get("EBL_CORPUS_ROOT")
    if not root:
        raise ValueError("Expected EBL_CORPUS_ROOT to name the directory of pinned token corpora.")
    return Path(root) / f"{task.choices_name}.pt"


def load_task(task, model: str) -> tuple[dict, Path]:
    """The pinned choice-task file of a lifecycle ``task`` section."""

    import networks.tasks as tasks

    path = task_path(task)
    digest = sha256_file(path)
    if digest != task.choices_sha256:
        raise ValueError(
            f"Expected task file {task.choices_name!r} with SHA-256 {task.choices_sha256}. "
            f"Provided value: {path} with SHA-256 {digest}."
        )
    document = load(path)
    tasks.check_task(document, model)
    if document["task"] != task.name:
        raise ValueError(f"Expected a {task.name} task file. Provided value: {document['task']!r}.")
    return document, path


def _task_counts(lifecycle) -> dict:
    task, cap = lifecycle.task, lifecycle.data.max_examples
    counts = {"task_development": task.development_examples, "task_test": task.evaluation_examples}
    if task.training_examples is not None:
        counts["task_train"] = task.training_examples
    return {name: min(count, cap) if cap else count for name, count in counts.items()}


def _build_task_cohorts(cache, lifecycle, max_positions) -> None:
    import networks.tasks as tasks

    task = lifecycle.task
    document, path = load_task(task, lifecycle.network.model)
    for name, count in _task_counts(lifecycle).items():
        cohort = tasks.cohort(document, name.removeprefix("task_"), count, max_positions)
        cohort["choices"]["decision"] = task.decision
        cohort["cohort"]["source"] = {"name": task.choices_name, "sha256": task.choices_sha256, "path": str(path)}
        cache[name] = cohort


def transfer_tests(lifecycle) -> tuple:
    return () if lifecycle.transfer_tests is None else lifecycle.transfer_tests.tests


def transfer_split(test) -> str:
    return test.cache_key


def check_cache(cache: dict, lifecycle) -> None:
    data = lifecycle.data
    names = ("hwa", "training", "development") + (("test",) if data.evaluation == "test" else ())
    for name in names:
        expected = (data.cohort_size(name), data.sequence_length)
        if name not in cache or tuple(cache[name]["tokens"].shape) != expected:
            raise ValueError(f"Expected the {name} cohort to contain {expected[0]} sequences of {expected[1]} tokens.")
    for test in transfer_tests(lifecycle):
        name = transfer_split(test)
        count = min(test.sequences, data.max_examples) if data.max_examples else test.sequences
        if name not in cache or tuple(cache[name]["tokens"].shape) != (count, data.sequence_length):
            raise ValueError(
                f"Expected the {name} cohort to contain {count} sequences of {data.sequence_length} tokens."
            )
        if cache[name]["cohort"]["corpus"]["sha256"] != test.corpus_sha256:
            raise ValueError(f"Expected the {name} cohort to come from its declared corpus.")
    task = getattr(lifecycle, "task", None)
    for name, count in (_task_counts(lifecycle).items() if task is not None else ()):
        value = cache.get(name, {})
        if (value.get("cohort", {}).get("count") != count
                or value["cohort"].get("source", {}).get("sha256") != task.choices_sha256
                or value["choices"].get("decision") != task.decision):
            raise ValueError(f"Expected the {name} cohort to hold {count} examples of the declared task file.")
    if task is None and any(name in cache for name in TASK_SPLITS + ("task_train",)):
        raise ValueError("Unexpected task cohorts in a lifecycle without a task.")
    if cache["corpus"]["sha256"] != data.corpus_sha256:
        raise ValueError("Expected the cache to come from the lifecycle corpus.")
    import networks.context_views as context_views
    context_views.check(cache, lifecycle)


def _splits(cache: dict) -> tuple:
    return SPLITS + tuple(sorted(name for name in cache
                                if name.startswith("transfer_") or (name.startswith("context") and name != "context")
                                or name == "recovery_context" or name in TASK_SPLITS or name == "task_train"))


def to_device(cache: dict, device) -> dict:
    """Move token cohorts once; task rows stay on the CPU and are padded per batch."""

    moved = dict(cache)
    for name in _splits(cache):
        if name in cache and "tokens" in cache[name]:
            moved[name] = {**cache[name], "tokens": cache[name]["tokens"].to(device)}
    return moved


def cohorts(cache: dict) -> dict:
    return {name: cache[name]["cohort"] for name in _splits(cache) if name in cache}


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


def _score_batch(score, out, teacher, labels):
    if not bool(torch.isfinite(out).all()):
        raise FloatingPointError("Nonfinite network scores.")
    flat = out.flatten(0, 1)
    score["kl"] += float(F.kl_div(flat.log_softmax(1), teacher.flatten(0, 1).softmax(1), reduction="sum"))
    score["ce"] += float(F.cross_entropy(flat, labels.flatten(), reduction="sum"))
    p = out.argmax(-1)
    score["predictions"].append(p.cpu())
    score["correct"] += int((p == labels).sum())
    score["agreement"] += int((p == teacher.argmax(-1)).sum())
    score["scored"] += labels.numel()


def _new_score():
    return dict(kl=0., ce=0., correct=0, agreement=0, scored=0, predictions=[])


def _finish_score(score, sequences):
    n = score["scored"]
    return {"sequences": sequences, "scored_tokens": n, "teacher_kl": score["kl"] / n,
            "cross_entropy": score["ce"] / n, "perplexity": math.exp(score["ce"] / n),
            "next_token_accuracy_percent": 100 * score["correct"] / n,
            "teacher_agreement_percent": 100 * score["agreement"] / n,
            "prediction_sha256": tensor_hash(torch.cat(score["predictions"]))}


@torch.no_grad()
def evaluate(network, split: dict, device, *, q=None, batch_size=EVALUATION_BATCH, read_seed=50001) -> dict:
    """Token-weighted metrics of the student (held state ``q``) against the teacher.

    A task cohort (``choices``) instead reports multiple-choice metrics through
    the same prefix, teacher and held-state forward.
    """

    if "choices" in split:
        return _evaluate_choices(network, split, device, q=q, read_seed=read_seed)
    old_rng = network.read_generator.get_state()
    network.read_generator.manual_seed(read_seed)
    network.eval()
    score = _new_score()
    try:
        for begin in range(0, len(split["tokens"]), batch_size):
            tokens = split["tokens"][begin : begin + batch_size].to(device)
            hidden = network.features(tokens)
            teacher = network.teacher_logits(hidden)
            out = network.forward_features(hidden, q)
            _score_batch(score, out, teacher, tokens[:, 1:])
    finally:
        network.read_generator.set_state(old_rng)
    return _finish_score(score, len(split["tokens"]))


@torch.no_grad()
def _evaluate_choices(network, split: dict, device, *, q, read_seed) -> dict:
    import networks.tasks as tasks

    choices = split["choices"]
    rows = len(choices["offsets"]) - 1
    student = torch.zeros(rows, dtype=torch.float64)
    teacher = torch.zeros(rows, dtype=torch.float64)
    token_kl, scored = 0.0, 0
    old_rng = network.read_generator.get_state()
    network.read_generator.manual_seed(read_seed)
    network.eval()
    try:
        for index in tasks.batches(choices):
            batch = tasks.padded(choices, index)
            tokens = batch["tokens"].to(device)
            hidden = network.features(tokens)
            reference = network.teacher_logits(hidden)
            out = network.forward_features(hidden, q)
            if not bool(torch.isfinite(out).all()):
                raise FloatingPointError("Nonfinite network scores.")
            mask = tasks.scored_mask(batch, device)
            student_log_probs, student_targets = tasks.continuation_log_probs(out, tokens, mask)
            teacher_log_probs, teacher_targets = tasks.continuation_log_probs(reference, tokens, mask)
            token_kl += float(F.kl_div(student_log_probs, teacher_log_probs, log_target=True, reduction="sum"))
            scored += int(mask.sum())
            student[index] = tasks.row_sums(student_targets, mask).cpu()
            teacher[index] = tasks.row_sums(teacher_targets, mask).cpu()
    finally:
        network.read_generator.set_state(old_rng)
    return tasks.finish(student, teacher, choices, token_kl=token_kl, scored=scored, decision=choices["decision"])


@torch.no_grad()
def teacher_task_metrics(network, split: dict, device) -> dict:
    """The digital teacher's own task accuracy: the ceiling of every deployed row."""

    import networks.tasks as tasks

    choices = split["choices"]
    teacher = torch.zeros(len(choices["offsets"]) - 1, dtype=torch.float64)
    for index in tasks.batches(choices):
        batch = tasks.padded(choices, index)
        tokens = batch["tokens"].to(device)
        reference = network.teacher_logits(network.features(tokens))
        mask = tasks.scored_mask(batch, device)
        teacher[index] = tasks.row_sums(tasks.continuation_log_probs(reference, tokens, mask)[1], mask).cpu()
    return tasks.teacher_finish(teacher, choices, decision=choices["decision"])


@torch.no_grad()
def evaluate_compensations(network, split, device, *, q, corrections, batch_size=EVALUATION_BATCH):
    """Identical held-state evaluations sharing only teacher computation per batch."""
    if network.read_noise:
        raise ValueError("Shared compensation evaluation requires zero per-MVM noise.")
    old = network.drift_compensation
    network.eval()
    scores = {name: _new_score() for name in corrections}
    try:
        for begin in range(0, len(split["tokens"]), batch_size):
            tokens = split["tokens"][begin:begin+batch_size].to(device)
            hidden = network.features(tokens)
            teacher = network.teacher_logits(hidden)
            for name, correction in corrections.items():
                network.drift_compensation = correction
                out = network.forward_features(hidden, q)
                _score_batch(scores[name], out, teacher, tokens[:, 1:])
    finally:
        network.drift_compensation = old
    return {name: _finish_score(score, len(split["tokens"])) for name, score in scores.items()}


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
    report = {"digital": digital, "teacher": teacher_metrics(network, cache[split], device), "corpus": cache["corpus"]}
    if getattr(lifecycle, "task", None) is not None:
        report["task"] = {}
        for name in TASK_SPLITS:
            clean = evaluate(network, cache[name], device)
            if not (clean["teacher_kl"] <= REPRODUCTION_KL and clean["choice_kl"] <= REPRODUCTION_KL):
                raise RuntimeError(
                    f"The clean digital suffix does not reproduce the teacher on {name}: "
                    f"token KL {clean['teacher_kl']:.3g}, choice KL {clean['choice_kl']:.3g}."
                )
            report["task"][name] = {"digital": clean, "teacher": teacher_task_metrics(network, cache[name], device)}
    return report


def deployment_summary(clean: dict, initial: dict) -> dict:
    return {
        "deployment_teacher_kl": initial["teacher_kl"],
        "deployment_perplexity_ratio": initial["perplexity"] / clean["perplexity"],
    }


# --- corpus preparation (python -m ebl lifecycle corpus) ------------------------------


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
