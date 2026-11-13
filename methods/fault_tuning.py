"""Fault-recovery LR-tuning screen (on-chip-learning fault-recovery-paper series 001).

One CMO deployment of one assignment and one permanent defect case; every arm
starts from the same deployed state and trains on the same recovery data:

- ``hold_naive``: naive Gaussian-endpoint P&V, no learning.
- ``hold_fault_aware``: P&V followed by the one-shot pair rewrite from verify
  reads (``program_verify.fault_rewrite``), no learning.
- ``onchip_noisy``: thresholded noisy CMO writes of digital Adam commands on the
  active devices of every base weight (``ThresholdedWriteAdam``, signed
  accumulation, 2 sigma nominal mid-range gate, no write cap).
- ``onchip_noiseless``: dense exact writes (``IdealizedWriteAdam``, write noise 0).
- ``child_tuning_d``: Child-Tuning_D. A Fisher estimate (mean squared gradient of
  the training loss over the whole recovery set, in fixed batches, at the
  deployed fault-aware state, computed once) selects the top ``reserve_p`` of the
  weights of each matrix; only those receive (noisy, thresholded) writes.
- ``digital_lora``: a digital rank-r adapter (alpha = r) on every analog matrix;
  the physical state is held, zero writes.
- ``analog_lora``: rank-r A and B in extra CMO tiles (active/reference pairs,
  per-output-channel digital scale) with their own permanent defects (own fault
  groups and streams, same rate) and the same fault-aware P&V at initialization.
  A starts from the standard LoRA initialization (that of ``digital_lora`` with
  the same seed), B at zero. The scale of A is each row's initial abs-max; the
  scale of B is fixed beforehand from a pinned file. Trained with the noisy
  thresholded write law on the adapter devices only; the adapter output is added
  digitally (the digital-LoRA path of the network, alpha/r = 1).

Teacher samples are drawn in-job, or (``targets.file``) loaded from
``<EBL_CORPUS_ROOT>/<file>.pt`` -- a stored draw of the same procedure, needed because the CUDA
draw is GPU-specific. Either way the raw-bytes tensor hash must equal ``targets.sha256``.

All arms: Adam (0.9/0.999, 1e-8), ``steps`` updates of ``batch`` sequences drawn
with replacement from the recovery cohort by one fixed generator, cosine decay to
zero without warm-up. Loss: mean negative log-likelihood of ``count`` fixed teacher
samples per position. Selection metric: development teacher KL against the full
FP32 teacher, evaluated online on the held apparent state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import time

import torch
from torch.nn import functional as F

from core.tensor_identity import tensor_hash
from hardware.mapping import MatrixSpec

METHODS = ("hold_naive", "hold_fault_aware", "onchip_noisy", "onchip_noiseless", "child_tuning_d",
           "digital_lora", "analog_lora")
TRAINS = METHODS[2:]
WRITES = ("onchip_noisy", "onchip_noiseless", "child_tuning_d")
LORA = ("digital_lora", "analog_lora")
_HEX = set("0123456789abcdef")


def raw_hash(value: torch.Tensor) -> str:
    """The drift-regimes / recovery-targets cohort and sample hash: SHA-256 of the raw bytes."""
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _digest(value, key):
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"Expected a SHA-256 for {key}.")
    return value


def _exact(raw, keys, key):
    if not isinstance(raw, dict) or set(raw) != set(keys):
        raise ValueError(f"Expected exactly the fields {sorted(keys)} in {key}.")
    return raw


def _int(value, key, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Expected an integer >= {minimum} for {key}.")
    return value


def _float(value, key):
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Expected a positive number for {key}.")
    return float(value)


@dataclass(frozen=True)
class Arm:
    label: str
    method: str
    learning_rate: float | None = None
    rank: int | None = None
    reserve_p: float | None = None

    def to_dict(self):
        return {k: v for k, v in (("label", self.label), ("method", self.method),
                                  ("learning_rate", self.learning_rate), ("rank", self.rank),
                                  ("reserve_p", self.reserve_p)) if v is not None}

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict) or raw.get("method") not in METHODS:
            raise ValueError(f"Expected a fault-tuning arm method in {METHODS}.")
        method = raw["method"]
        keys = {"label", "method"} | ({"learning_rate"} if method in TRAINS else set()) \
            | ({"rank"} if method in LORA else set()) | ({"reserve_p"} if method == "child_tuning_d" else set())
        _exact(raw, keys, f"arm {raw.get('label')!r}")
        label = raw["label"]
        if not isinstance(label, str) or not label or not all(c.isalnum() or c in "-_." for c in label):
            raise ValueError("Expected a plain arm label.")
        p = raw.get("reserve_p")
        if p is not None and not (type(p) in (int, float) and 0 < p < 1):
            raise ValueError("Expected 0 < reserve_p < 1.")
        return cls(label, method,
                   _float(raw["learning_rate"], "learning_rate") if "learning_rate" in raw else None,
                   _int(raw["rank"], "rank", 1) if "rank" in raw else None,
                   None if p is None else float(p))


DATA_KEYS = ("corpus", "corpus_sha256", "split", "start", "sequences", "seed", "cohort_sha256")


@dataclass(frozen=True)
class FaultTuning:
    source_sha256: str
    assignment_seed: int
    fault_rewrite: dict
    recovery: dict
    development: dict
    targets: dict
    schedule: dict
    write_law: dict
    fisher: dict
    lora_seed: int
    analog_lora: dict | None
    deployment: dict | None
    arms: tuple

    def to_dict(self):
        return {"source_sha256": self.source_sha256, "assignment_seed": self.assignment_seed,
                "fault_rewrite": dict(self.fault_rewrite), "recovery": dict(self.recovery),
                "development": dict(self.development), "targets": dict(self.targets),
                "schedule": dict(self.schedule), "write_law": dict(self.write_law), "fisher": dict(self.fisher),
                "lora_seed": self.lora_seed,
                "analog_lora": None if self.analog_lora is None else {
                    "rank": self.analog_lora["rank"],
                    "b_scale": None if self.analog_lora["b_scale"] is None else dict(self.analog_lora["b_scale"])},
                "deployment": None if self.deployment is None else dict(self.deployment),
                "arms": [a.to_dict() for a in self.arms]}

    def rewrite(self):
        from methods.config import FaultRewrite
        r = self.fault_rewrite
        return FaultRewrite(r["detection"], r["roles"], r["seed"], r["threshold_sigma"])

    @classmethod
    def parse(cls, raw):
        keys = ("source_sha256", "assignment_seed", "fault_rewrite", "recovery", "development", "targets",
                "schedule", "write_law", "fisher", "lora_seed", "analog_lora", "deployment", "arms")
        _exact(raw, keys, "fault_tuning")
        rewrite = _exact(raw["fault_rewrite"], ("detection", "roles", "seed", "threshold_sigma"), "fault_rewrite")
        if rewrite["detection"] != "verify" or rewrite["roles"] != "pair":
            raise ValueError("Expected the verify-based pair rewrite.")
        _int(rewrite["seed"], "fault_rewrite.seed", 1)
        _float(rewrite["threshold_sigma"], "fault_rewrite.threshold_sigma")
        data = {}
        for name in ("recovery", "development"):
            d = _exact(raw[name], DATA_KEYS, name)
            _digest(d["corpus_sha256"], f"{name}.corpus_sha256")
            _digest(d["cohort_sha256"], f"{name}.cohort_sha256")
            _int(d["sequences"], f"{name}.sequences", 1)
            _int(d["start"], f"{name}.start")
            if d["seed"] is not None:
                _int(d["seed"], f"{name}.seed")
            if d["split"] not in ("train", "validation", "test"):
                raise ValueError(f"Expected a corpus split for {name}.")
            data[name] = d
        if data["recovery"]["seed"] is None or data["recovery"]["start"] != 0 or data["development"]["seed"] is not None:
            raise ValueError("Expected a seeded recovery selection and a contiguous development slice.")
        t = raw["targets"]
        t = _exact(t, ("kind", "count", "seed", "draw_batch", "sha256") + (("file",) if isinstance(t, dict) and "file" in t else ()),
                   "targets")
        if "file" in t and (not isinstance(t["file"], str) or not t["file"] or "/" in t["file"]):
            raise ValueError("Expected targets.file to name a pinned tensor beside the corpora.")
        if t["kind"] != "teacher_samples":
            raise ValueError("Expected fixed teacher samples as the training target.")
        _int(t["count"], "targets.count", 1), _int(t["seed"], "targets.seed"), _int(t["draw_batch"], "targets.draw_batch", 1)
        _digest(t["sha256"], "targets.sha256")
        s = _exact(raw["schedule"], ("optimizer", "betas", "eps", "steps", "batch", "order_seed", "lr_schedule",
                                     "warmup_steps", "development_every_steps"), "schedule")
        if (s["optimizer"] != "adam" or list(s["betas"]) != [0.9, 0.999] or s["eps"] != 1e-8
                or s["lr_schedule"] != "cosine_to_zero" or s["warmup_steps"] != 0):
            raise ValueError("Expected Adam(0.9, 0.999, 1e-8) with cosine decay to zero and no warm-up.")
        _int(s["steps"], "schedule.steps", 1), _int(s["batch"], "schedule.batch", 1)
        _int(s["order_seed"], "schedule.order_seed"), _int(s["development_every_steps"], "schedule.development_every_steps", 1)
        w = _exact(raw["write_law"], ("threshold_sigma", "accumulate_updates", "max_write_attempts_per_device"),
                   "write_law")
        _float(w["threshold_sigma"], "write_law.threshold_sigma")
        if type(w["accumulate_updates"]) is not bool or w["max_write_attempts_per_device"] is not None:
            raise ValueError("Expected a boolean accumulation flag and no write cap.")
        f = _exact(raw["fisher"], ("estimator", "batch"), "fisher")
        if f["estimator"] != "mean_squared_batch_gradient":
            raise ValueError("Expected the mean squared batch-gradient Fisher estimate.")
        _int(f["batch"], "fisher.batch", 1)
        analog = raw["analog_lora"]
        if analog is not None:
            _exact(analog, ("rank", "b_scale"), "analog_lora")
            _int(analog["rank"], "analog_lora.rank", 1)
            if analog["b_scale"] is not None:
                _exact(analog["b_scale"], ("name", "sha256"), "analog_lora.b_scale")
                _digest(analog["b_scale"]["sha256"], "analog_lora.b_scale.sha256")
        deployment = raw["deployment"]
        if deployment is not None:
            _exact(deployment, ("name", "sha256"), "deployment")
            _digest(deployment["sha256"], "deployment.sha256")
        if not isinstance(raw["arms"], list) or not raw["arms"]:
            raise ValueError("Expected a nonempty list of arms.")
        arms = tuple(Arm.parse(a) for a in raw["arms"])
        if len({a.label for a in arms}) != len(arms):
            raise ValueError("Expected distinct arm labels.")
        for a in arms:
            if a.method == "analog_lora" and (analog is None or analog["b_scale"] is None or a.rank != analog["rank"]):
                raise ValueError("Expected an analog_lora section with a pinned B scale of the arm's rank.")
        return cls(_digest(raw["source_sha256"], "source_sha256"), _int(raw["assignment_seed"], "assignment_seed"),
                   dict(rewrite), data["recovery"], data["development"], dict(t), dict(s), dict(w), dict(f),
                   _int(raw["lora_seed"], "lora_seed"), None if analog is None else
                   {"rank": analog["rank"], "b_scale": None if analog["b_scale"] is None else dict(analog["b_scale"])},
                   None if deployment is None else dict(deployment), arms)


# --- data ---------------------------------------------------------------------------------------

def cohort(lifecycle, settings):
    """A recovery (seeded block selection) or development (contiguous slice) token cohort, hash-pinned."""
    import networks.opt as opt
    data = replace(lifecycle.data, corpus=settings["corpus"], corpus_sha256=settings["corpus_sha256"])
    corpus, _ = opt.load_corpus(data, lifecycle.network.model)
    length = lifecycle.data.sequence_length
    stream = corpus["splits"][settings["split"]]
    blocks = stream[:len(stream) // length * length].reshape(-1, length)
    if settings["seed"] is not None:
        order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(settings["seed"]))
        tokens = blocks[order[:settings["sequences"]]].contiguous()
    else:
        tokens = blocks[settings["start"]:settings["start"] + settings["sequences"]].contiguous()
    if len(tokens) != settings["sequences"] or raw_hash(tokens) != settings["cohort_sha256"]:
        raise ValueError(f"Token-cohort hash mismatch ({settings['split']}).")
    return tokens


@torch.no_grad()
def teacher_samples(network, tokens, count, seed, batch, device):
    """recovery-targets ``teacher_samples``, operation for operation: one CUDA generator, chunks of ``batch``."""
    import networks.opt as opt
    generator = torch.Generator(device=device).manual_seed(seed)
    out = []
    for chunk in tokens.split(batch):
        features = network.features(chunk.to(device))
        hidden = features.teacher_hidden if isinstance(features, opt.StudentInput) else features
        probs = network.teacher_logits(hidden).float().softmax(-1)
        shape = probs.shape[:-1]
        draw = torch.multinomial(probs.flatten(0, 1), count, replacement=True, generator=generator)
        out.append(draw.view(*shape, count))
    return torch.cat(out)


def sample_loss(logits, drawn):
    """Mean NLL of the fixed teacher samples; ``logits`` (B, L-1, V), ``drawn`` (B, L-1, n)."""
    student = logits.flatten(0, 1).log_softmax(-1)
    return -student.gather(-1, drawn.flatten(0, 1)).mean()


def cosine(learning_rate, step, steps):
    """torch CosineAnnealingLR(T_max=steps, eta_min=0) at 0-based ``step``."""
    return learning_rate * 0.5 * (1 + math.cos(math.pi * step / steps))


# --- adapters -----------------------------------------------------------------------------------

def adapter_layout(base_layout, rank):
    """A (rank x in) and B (out x rank) of every base matrix, in the network's analog-LoRA naming."""
    layout, offset = [], 0
    for spec in base_layout:
        outputs, inputs = spec.shape
        for suffix, shape in (("lora_a", (rank, inputs)), ("lora_b", (outputs, rank))):
            layout.append(MatrixSpec(f"{spec.name}.{suffix}", shape, offset, shape[0] * shape[1], 1.0))
            offset += shape[0] * shape[1]
    return layout


def lora_init(base_layout, rank, seed):
    """``OptMlpSuffix.enable_lora``'s A draws (CPU generator, in layout order); B is zero."""
    generator = torch.Generator().manual_seed(seed)
    out = []
    for spec in base_layout:
        outputs, inputs = spec.shape
        out.append(torch.empty(rank, inputs).uniform_(-inputs ** -0.5, inputs ** -0.5, generator=generator))
    return out


def adapter_targets(base_layout, rank, seed):
    """Normalized adapter targets (A / row abs-max, B = 0) and A's per-row scales."""
    values, a_scales = [], []
    for spec, a in zip(base_layout, lora_init(base_layout, rank, seed)):
        scale = a.abs().amax(1).clamp_min(1e-12)
        values += [(a / scale[:, None]).flatten(), torch.zeros(spec.shape[0] * rank)]
        a_scales.append(scale)
    return torch.cat(values), a_scales


def b_scales_from_lora(lora_state, base_layout):
    """Per-output-channel max |B| of a trained digital LoRA (``network.lora`` state dict)."""
    return [lora_state[f"{k}.b"].detach().abs().amax(1).float().cpu() for k in range(len(base_layout))]


# --- Child-Tuning_D -----------------------------------------------------------------------------

def child_mask(fisher, layout, reserve_p):
    """Top ``reserve_p`` of each matrix's weights by Fisher score (ties broken by torch.topk)."""
    mask = torch.zeros_like(fisher, dtype=torch.bool)
    for spec in layout:
        part = fisher[spec.offset:spec.offset + spec.size]
        k = max(1, int(round(reserve_p * spec.size)))
        mask[spec.offset + part.topk(k).indices] = True
    return mask
