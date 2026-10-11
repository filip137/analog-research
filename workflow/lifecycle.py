"""Strict, stdlib-only schema for campaign-owned crossbar lifecycles.

A lifecycle states one experimental loop: the question, the devices and their
defects, then (i) deployment of the digital weights, (ii) hardware-aware
training, (iii) program-and-verify and (iv) on-chip training. The network
section selects a family (``cifar_resnet32_suffix`` or ``opt_mlp_suffix``),
which fixes the data section, metrics and calibration scheme. Every native
stage run embeds the complete normalized lifecycle in its resolved config.

Validation runs before torch, datasets or devices are touched. Cross-section
rules keep matched controls matched by construction: one device technology
fixes the HWA noise model, programming method, relaxation law, on-chip update
law and admissible defect kinds, and every weight-learning arm needs its
no-weight-write control.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional

from experiments.artifacts import content_hash
from experiments.schema import ConfigError, RunMode, config_error

SCHEMA = "crossbar_lifecycle"
SCHEMA_VERSION = 1
EXPERIMENT_ID = "crossbar_lifecycle.v1"
PROVENANCE_SCHEMA = "crossbar_lifecycle.provenance.v1"

STAGE_KINDS = ("prepare", "devices", "hwa", "deploy", "onchip")
CIFAR_FAMILY = "cifar_resnet32_suffix"
OPT_FAMILY = "opt_mlp_suffix"
# Converted layer3 convolutions (two per block) plus the always-analog classifier.
SUFFIX_BY_CONVOLUTIONS = {
    0: "head",
    2: "last_block",
    4: "last_two_blocks",
    8: "last_four_blocks",
}
CLASSES = {"cifar10": 10, "cifar100": 100}
TRAINING_SPLIT = 45000
DEVELOPMENT_SPLIT = 5000
# Pinned OPT checkpoints: decoder layers, maximum sequence length and MLP shape.
OPT_MODELS = {
    "facebook/opt-125m": {"decoder_layers": 12, "max_positions": 2048, "hidden_size": 768, "ffn_dim": 3072}
}
# Each analog layer3 convolution is 64x64x3x3; the analog classifier is 64 x classes.
CIFAR_CONVOLUTION_WEIGHTS = 64 * 64 * 9
CIFAR_FEATURES = 64
FAILURE_KINDS = ("open", "gmax", "random")
# CMO stuck values: 0 uS (open), G_min, G_max, or one uniform value in range per device.
CMO_FAILURE_KINDS = ("open", "gmin", "gmax", "random")
# Hierarchical placement: die and array fault rates, an exact fault total per group.
HIERARCHICAL = "hierarchical_die_array"
# P&V acceptance ranges (percent) of the AIHWKit 1.1.0 CMO programming fit.
CMO_ACCEPTANCE_PERCENT = (0.2, 2.0)
# Weight-to-target scaling: each matrix by its abs-max, or each output channel by
# its own abs-max with a fixed digital output scale per channel (OPT family only).
SCALINGS = ("per_layer_absmax", "per_output_channel_absmax")
# HWA noise model: Gaussian weight noise of ``hwa.noise_std`` relative to the
# matrix full scale (1 after per-layer abs-max scaling) or to each output
# channel's current abs-max (the Analog Foundation Models convention).
ADDITIVE_GAUSSIAN = "additive_gaussian_weight"
NOISE_SCALINGS = ("matrix_full_scale", "output_channel_abs_max")
OBJECTIVES = ("teacher_kl", "cross_entropy")
CALIBRATION = "output_gain_bn_affine_classifier_bias"
OPT_CALIBRATION = "output_gain_mlp_layer_norm_mlp_bias"
# What each network family can report, calibrate and augment.
PRIMARY_METRICS = {CIFAR_FAMILY: ("teacher_kl", "accuracy_percent"), OPT_FAMILY: ("teacher_kl", "perplexity")}
CALIBRATIONS = {CIFAR_FAMILY: CALIBRATION, OPT_FAMILY: OPT_CALIBRATION}
MAX_SEED = 2**31 - 1


@dataclass(frozen=True)
class Technology:
    """The physical regime implied by one device technology."""

    device_model: str
    encoding: str
    defect_placement: str
    failure_kinds: tuple[str, ...]
    corruption_aware_hwa: bool
    hwa_noise_model: str
    program_methods: tuple[str, ...]
    relaxation_models: tuple[str, ...]
    update_laws: tuple[str, ...]


TECHNOLOGIES = {
    # IBM optimized-material ReRAM: literal AIHWKit 1.1.0 cells, one active
    # cell per weight read as q=a-r against its sampled intrinsic reference.
    "om": Technology(
        device_model="aihwkit_1.1.0_reram_array_om",
        encoding="single_cell_intrinsic_reference",
        defect_placement="bernoulli_per_active_cell",
        failure_kinds=FAILURE_KINDS,
        corruption_aware_hwa=True,
        hwa_noise_model="om_endpoint_kernel",
        program_methods=("closed_loop", "open_loop_nominal"),
        relaxation_models=("none",),
        update_laws=("om_closed_loop_pulse_adam", "om_open_loop_pulse_adam"),
    ),
    # Gaussian PCM endpoint programming (Li et al. 2023, AIHWKit PCM
    # programming noise) on a 25 uS differential pair; no pulse model.
    "pcm": Technology(
        device_model="gaussian_endpoint_li2023",
        encoding="differential_pair_25us",
        defect_placement="bernoulli_per_physical_device",
        failure_kinds=FAILURE_KINDS,
        corruption_aware_hwa=True,
        hwa_noise_model="pcm_gaussian_endpoint",
        program_methods=("gaussian_endpoint",),
        relaxation_models=("none", "pcm_drift"),
        update_laws=("pcm_endpoint_reprogramming",),
    ),
    # IBM CMO/HfOx ReRAM (Falcone et al. 2025; AIHWKit 1.1.0 ReRamCMONoiseModel):
    # unclamped Gaussian P&V endpoints in 9-88.2 uS, an active and a mid-range
    # reference device per weight (q = a - r), idealized noisy on-chip writes. HWA
    # injects declared Gaussian weight noise (``hwa.noise_std``) and, for
    # corruption-aware arms, temporary i.i.d. stuck devices on both devices of
    # each pair. Deployed stuck devices follow a die/array hierarchy.
    # Drift and read noise are not modelled yet (docs/reram_cmo_device.md).
    "cmo": Technology(
        device_model="aihwkit_1.1.0_reram_cmo",
        encoding="active_reference_pair",
        defect_placement=HIERARCHICAL,
        failure_kinds=CMO_FAILURE_KINDS,
        corruption_aware_hwa=True,
        hwa_noise_model=ADDITIVE_GAUSSIAN,
        program_methods=("gaussian_endpoint",),
        relaxation_models=("none",),
        update_laws=("cmo_idealized_write_adam",),
    ),
}

_LIFECYCLE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_ARM_ID = re.compile(r"[a-z0-9][a-z0-9_]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


# ---------------------------------------------------------------------------
# Strict primitive parsers. Messages state the expected format first.


def _object(value: Any, path: str, keys) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise config_error(path, "to be an object", value)
    if set(value) != set(keys):
        raise config_error(
            f"{path} fields", f"to be exactly {sorted(keys)}", sorted(value)
        )
    return value


def _choice(value: Any, path: str, choices) -> str:
    if not isinstance(value, str) or value not in choices:
        raise config_error(path, f"to be one of {list(choices)}", value)
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise config_error(path, "to be non-empty text without outer whitespace", value)
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise config_error(path, "to be a boolean", value)
    return value


def _integer(value: Any, path: str, *, minimum: int = 0, maximum: Optional[int] = None) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bound = f"at least {minimum}" + ("" if maximum is None else f" and at most {maximum}")
        raise config_error(path, f"to be an integer {bound}", value)
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
    positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (positive and value <= 0)
        or (maximum is not None and value > maximum)
    ):
        bound = ("positive" if positive else f"at least {minimum}") + (
            "" if maximum is None else f" and at most {maximum}"
        )
        raise config_error(path, f"to be a finite number {bound}", value)
    return float(value)


def _list(value: Any, path: str, *, nonempty: bool = True) -> list:
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise config_error(path, "to be a " + ("non-empty " if nonempty else "") + "list", value)
    return list(value)


def _seeds(value: Any, path: str, *, nonempty: bool = True) -> tuple[int, ...]:
    seeds = tuple(
        _integer(item, f"{path}[{index}]", minimum=1, maximum=MAX_SEED)
        for index, item in enumerate(_list(value, path, nonempty=nonempty))
    )
    if len(set(seeds)) != len(seeds):
        raise config_error(path, "to contain unique seeds", list(seeds))
    return seeds


def _identifier(value: Any, path: str, pattern=_ARM_ID) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise config_error(path, f"to match {pattern.pattern!r}", value)
    return value


# ---------------------------------------------------------------------------
# Sections. ``to_dict`` emits the normalized document that is hashed.


@dataclass(frozen=True)
class Question:
    text: str
    decision: str
    primary_metric: str
    evidence_class: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "decision": self.decision,
            "primary_metric": self.primary_metric,
            "evidence_class": self.evidence_class,
        }


def _question(raw: Any) -> Question:
    raw = _object(raw, "question", ("text", "decision", "primary_metric", "evidence_class"))
    return Question(
        text=_text(raw["text"], "question.text"),
        decision=_text(raw["decision"], "question.decision"),
        # The network family narrows the metric in _cross_validate.
        primary_metric=_choice(
            raw["primary_metric"],
            "question.primary_metric",
            tuple(dict.fromkeys(m for metrics in PRIMARY_METRICS.values() for m in metrics)),
        ),
        evidence_class=_choice(
            raw["evidence_class"], "question.evidence_class", ("exploratory_model_based",)
        ),
    )


@dataclass(frozen=True)
class Network:
    """ResNet-32 with a frozen digital prefix and analog layer3 suffix convolutions."""

    family: str
    dataset: str
    analog_convolutions: int

    @property
    def suffix(self) -> str:
        return SUFFIX_BY_CONVOLUTIONS[self.analog_convolutions]

    @property
    def classes(self) -> int:
        return CLASSES[self.dataset]

    @property
    def analog_weights(self) -> int:
        return self.analog_convolutions * CIFAR_CONVOLUTION_WEIGHTS + CIFAR_FEATURES * self.classes

    @property
    def binding_namespace(self) -> str:
        """Namespace of the per-layer population, fault and programming seeds."""

        return self.dataset

    def describe(self) -> str:
        return f"{self.dataset} ResNet-32, {self.analog_convolutions} analog convolutions + classifier"

    def summary(self) -> dict:
        return {"dataset": self.dataset, "analog_convolutions": self.analog_convolutions, "suffix": self.suffix}

    def mapping_fields(self) -> dict:
        return {"analog_convolutions": self.analog_convolutions}

    def result_fields(self) -> dict:
        return {"dataset": self.dataset, "analog_convolutions": self.analog_convolutions}

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "dataset": self.dataset,
            "analog_convolutions": self.analog_convolutions,
        }


@dataclass(frozen=True)
class OptNetwork:
    """OPT decoder whose last ``analog_decoder_layers`` MLPs (fc1, fc2) are crossbars.

    Embeddings, the earlier decoder layers, every attention block, LayerNorms,
    biases and the tied LM head stay digital.
    """

    family: str
    model: str
    analog_decoder_layers: int

    @property
    def decoder_layers(self) -> int:
        return OPT_MODELS[self.model]["decoder_layers"]

    @property
    def max_positions(self) -> int:
        return OPT_MODELS[self.model]["max_positions"]

    @property
    def analog_weights(self) -> int:
        spec = OPT_MODELS[self.model]
        return self.analog_decoder_layers * 2 * spec["hidden_size"] * spec["ffn_dim"]

    @property
    def binding_namespace(self) -> str:
        return self.model

    def describe(self) -> str:
        return f"{self.model}, analog fc1/fc2 in the last {self.analog_decoder_layers} of {self.decoder_layers} decoder layers"

    def summary(self) -> dict:
        return {
            "model": self.model,
            "analog_decoder_layers": self.analog_decoder_layers,
            "analog_layer_indices": list(range(self.decoder_layers - self.analog_decoder_layers, self.decoder_layers)),
        }

    def mapping_fields(self) -> dict:
        return {"analog_decoder_layers": self.analog_decoder_layers, "analog_modules": ["fc1", "fc2"]}

    def result_fields(self) -> dict:
        return {"model": self.model, "analog_decoder_layers": self.analog_decoder_layers}

    def to_dict(self) -> dict:
        return {"family": self.family, "model": self.model, "analog_decoder_layers": self.analog_decoder_layers}


def _network(raw: Any) -> Network | OptNetwork:
    if not isinstance(raw, Mapping):
        raise config_error("network", "to be an object", raw)
    family = _choice(raw.get("family"), "network.family", (CIFAR_FAMILY, OPT_FAMILY))
    if family == OPT_FAMILY:
        raw = _object(raw, "network", ("family", "model", "analog_decoder_layers"))
        model = _choice(raw["model"], "network.model", tuple(OPT_MODELS))
        return OptNetwork(
            family=family,
            model=model,
            analog_decoder_layers=_integer(
                raw["analog_decoder_layers"],
                "network.analog_decoder_layers",
                minimum=1,
                maximum=OPT_MODELS[model]["decoder_layers"],
            ),
        )
    raw = _object(raw, "network", ("family", "dataset", "analog_convolutions"))
    convolutions = raw["analog_convolutions"]
    if type(convolutions) is not int or convolutions not in SUFFIX_BY_CONVOLUTIONS:
        raise config_error(
            "network.analog_convolutions",
            f"to be one of {sorted(SUFFIX_BY_CONVOLUTIONS)}",
            convolutions,
        )
    return Network(
        family=family,
        dataset=_choice(raw["dataset"], "network.dataset", tuple(CLASSES)),
        analog_convolutions=convolutions,
    )


@dataclass(frozen=True)
class Data:
    data_seed: int
    hwa_images: int
    recovery_images: int
    development_images: int
    evaluation: str
    max_examples: int

    @property
    def smoke(self) -> bool:
        return self.max_examples > 0

    def cohort_size(self, name: str) -> int:
        full = {
            "hwa": self.hwa_images,
            "training": self.recovery_images,
            "development": self.development_images,
            "test": 10000,
        }[name]
        return min(full, self.max_examples) if self.max_examples else full

    def to_dict(self) -> dict:
        return {
            "data_seed": self.data_seed,
            "hwa_images": self.hwa_images,
            "recovery_images": self.recovery_images,
            "development_images": self.development_images,
            "evaluation": self.evaluation,
            "max_examples": self.max_examples,
        }


def _data(raw: Any) -> Data:
    keys = (
        "data_seed",
        "hwa_images",
        "recovery_images",
        "development_images",
        "evaluation",
        "max_examples",
    )
    raw = _object(raw, "data", keys)
    return Data(
        data_seed=_integer(raw["data_seed"], "data.data_seed", maximum=MAX_SEED),
        hwa_images=_integer(raw["hwa_images"], "data.hwa_images", minimum=1, maximum=TRAINING_SPLIT),
        recovery_images=_integer(
            raw["recovery_images"], "data.recovery_images", minimum=1, maximum=TRAINING_SPLIT
        ),
        development_images=_integer(
            raw["development_images"],
            "data.development_images",
            minimum=1,
            maximum=DEVELOPMENT_SPLIT,
        ),
        evaluation=_choice(raw["evaluation"], "data.evaluation", ("test", "development")),
        max_examples=_integer(raw["max_examples"], "data.max_examples"),
    )


@dataclass(frozen=True)
class TextData:
    """Token cohorts of one pinned corpus, cut into non-overlapping sequences.

    HWA and recovery sequences are nested prefixes of the seeded shuffle of
    the corpus train split; development and test sequences are the first
    blocks of the validation and test splits. Test sequences exist exactly
    when the lifecycle evaluates on test.
    """

    corpus: str
    corpus_sha256: str
    sequence_length: int
    data_seed: int
    hwa_sequences: int
    recovery_sequences: int
    development_sequences: int
    test_sequences: int
    evaluation: str
    max_examples: int

    @property
    def smoke(self) -> bool:
        return self.max_examples > 0

    def cohort_size(self, name: str) -> int:
        full = {
            "hwa": self.hwa_sequences,
            "training": self.recovery_sequences,
            "development": self.development_sequences,
            "test": self.test_sequences,
        }[name]
        return min(full, self.max_examples) if self.max_examples else full

    def to_dict(self) -> dict:
        return {
            "corpus": {"name": self.corpus, "sha256": self.corpus_sha256},
            "sequence_length": self.sequence_length,
            "data_seed": self.data_seed,
            "hwa_sequences": self.hwa_sequences,
            "recovery_sequences": self.recovery_sequences,
            "development_sequences": self.development_sequences,
            "test_sequences": self.test_sequences,
            "evaluation": self.evaluation,
            "max_examples": self.max_examples,
        }


def _text_data(raw: Any, network: OptNetwork) -> TextData:
    keys = (
        "corpus",
        "sequence_length",
        "data_seed",
        "hwa_sequences",
        "recovery_sequences",
        "development_sequences",
        "test_sequences",
        "evaluation",
        "max_examples",
    )
    raw = _object(raw, "data", keys)
    corpus = _object(raw["corpus"], "data.corpus", ("name", "sha256"))
    digest = corpus["sha256"]
    if not isinstance(digest, str) or not _SHA256.match(digest):
        raise config_error("data.corpus.sha256", "to be a lowercase SHA-256 digest", digest)
    evaluation = _choice(raw["evaluation"], "data.evaluation", ("test", "development"))
    test = _integer(raw["test_sequences"], "data.test_sequences")
    if (evaluation == "test") != (test > 0):
        raise config_error(
            "data.test_sequences", "to be positive exactly when data.evaluation is test", test
        )
    return TextData(
        corpus=_identifier(corpus["name"], "data.corpus.name", _LIFECYCLE_ID),
        corpus_sha256=digest,
        sequence_length=_integer(
            raw["sequence_length"], "data.sequence_length", minimum=2, maximum=network.max_positions
        ),
        data_seed=_integer(raw["data_seed"], "data.data_seed", maximum=MAX_SEED),
        hwa_sequences=_integer(raw["hwa_sequences"], "data.hwa_sequences", minimum=1),
        recovery_sequences=_integer(raw["recovery_sequences"], "data.recovery_sequences", minimum=1),
        development_sequences=_integer(raw["development_sequences"], "data.development_sequences", minimum=1),
        test_sequences=test,
        evaluation=evaluation,
        max_examples=_integer(raw["max_examples"], "data.max_examples"),
    )


@dataclass(frozen=True)
class TransferTest:
    """A second test cohort from another pinned corpus, scored at every evaluation.

    It only adds measurements: feature caches and deployments depend on it,
    HWA sources do not, so a source trained without it can be reused.
    """

    label: str
    corpus: str
    corpus_sha256: str
    split: str
    sequences: int

    @property
    def cache_key(self) -> str:
        """Cache and metric key of the cohort."""

        return f"transfer_{self.label}"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "corpus": {"name": self.corpus, "sha256": self.corpus_sha256},
            "split": self.split,
            "sequences": self.sequences,
        }


@dataclass(frozen=True)
class TransferTests:
    tests: tuple[TransferTest, ...]

    def to_dict(self) -> list:
        return [test.to_dict() for test in self.tests]


def _transfer_tests(raw: Any) -> TransferTests:
    tests = []
    for i, item in enumerate(_list(raw, "transfer_tests")):
        path = f"transfer_tests[{i}]"
        item = _object(item, path, ("label", "corpus", "split", "sequences"))
        corpus = _object(item["corpus"], f"{path}.corpus", ("name", "sha256"))
        if not isinstance(corpus["sha256"], str) or not _SHA256.match(corpus["sha256"]):
            raise config_error(f"{path}.corpus.sha256", "to be a lowercase SHA-256 digest", corpus["sha256"])
        tests.append(
            TransferTest(
                label=_identifier(item["label"], f"{path}.label"),
                corpus=_identifier(corpus["name"], f"{path}.corpus.name", _LIFECYCLE_ID),
                corpus_sha256=corpus["sha256"],
                # Never a train split: transfer cohorts are evaluation-only tokens.
                split=_choice(item["split"], f"{path}.split", ("validation", "test")),
                sequences=_integer(item["sequences"], f"{path}.sequences", minimum=1),
            )
        )
    if len({test.label for test in tests}) != len(tests):
        raise config_error("transfer_tests", "to have unique labels", [test.label for test in tests])
    return TransferTests(tuple(tests))


@dataclass(frozen=True)
class Characterization:
    """Independent OM populations that measure the P&V endpoint law for HWA."""

    bins: int
    samples: int
    fit_seed: int
    heldout_seed: int

    def to_dict(self) -> dict:
        return {
            "bins": self.bins,
            "samples": self.samples,
            "fit_seed": self.fit_seed,
            "heldout_seed": self.heldout_seed,
        }


@dataclass(frozen=True)
class Devices:
    technology: str
    model: str
    noise_scale: float
    assignment_seeds: tuple[int, ...]
    selection_seeds: tuple[int, ...]
    endpoint_seed_offset: int
    policy: Optional[str] = None
    variation_scale: Optional[float] = None
    characterization: Optional[Characterization] = None

    def endpoint_seed(self, assignment_seed: int) -> int:
        return assignment_seed + self.endpoint_seed_offset

    def to_dict(self) -> dict:
        value = {
            "technology": self.technology,
            "model": self.model,
            "noise_scale": self.noise_scale,
            "assignment_seeds": list(self.assignment_seeds),
            "selection_seeds": list(self.selection_seeds),
            "endpoint_seed_offset": self.endpoint_seed_offset,
        }
        if self.technology == "om":
            value.update(
                policy=self.policy,
                variation_scale=self.variation_scale,
                characterization=self.characterization.to_dict(),
            )
        return value


def _devices(raw: Any) -> Devices:
    if not isinstance(raw, Mapping):
        raise config_error("devices", "to be an object", raw)
    technology = _choice(raw.get("technology"), "devices.technology", tuple(TECHNOLOGIES))
    common = (
        "technology",
        "model",
        "noise_scale",
        "assignment_seeds",
        "selection_seeds",
        "endpoint_seed_offset",
    )
    om_keys = ("policy", "variation_scale", "characterization")
    raw = _object(raw, "devices", common + (om_keys if technology == "om" else ()))
    model = _choice(raw["model"], "devices.model", (TECHNOLOGIES[technology].device_model,))
    extra = {}
    if technology == "om":
        c = _object(
            raw["characterization"],
            "devices.characterization",
            ("bins", "samples", "fit_seed", "heldout_seed"),
        )
        characterization = Characterization(
            bins=_integer(c["bins"], "devices.characterization.bins", minimum=3),
            samples=_integer(c["samples"], "devices.characterization.samples", minimum=1),
            fit_seed=_integer(c["fit_seed"], "devices.characterization.fit_seed", minimum=1, maximum=MAX_SEED),
            heldout_seed=_integer(
                c["heldout_seed"], "devices.characterization.heldout_seed", minimum=1, maximum=MAX_SEED
            ),
        )
        if characterization.fit_seed == characterization.heldout_seed:
            raise config_error(
                "devices.characterization", "to use distinct fit and held-out seeds", dict(c)
            )
        extra = dict(
            # Faults are injected into a counterfactual repaired baseline;
            # the published-corruption overlay is not a lifecycle defect.
            policy=_choice(raw["policy"], "devices.policy", ("repaired",)),
            variation_scale=_number(raw["variation_scale"], "devices.variation_scale", positive=True),
            characterization=characterization,
        )
    return Devices(
        technology=technology,
        model=model,
        noise_scale=_number(raw["noise_scale"], "devices.noise_scale"),
        assignment_seeds=_seeds(raw["assignment_seeds"], "devices.assignment_seeds"),
        selection_seeds=_seeds(raw["selection_seeds"], "devices.selection_seeds", nonempty=False),
        endpoint_seed_offset=_integer(
            raw["endpoint_seed_offset"], "devices.endpoint_seed_offset", minimum=1, maximum=MAX_SEED
        ),
        **extra,
    )


@dataclass(frozen=True)
class DefectCase:
    kind: str
    rate_ppm: int
    # Hierarchical placement only: Beta concentrations of the die and array
    # fault rates; None is infinite (no variation at that level).
    die_concentration: Optional[float] = None
    array_concentration: Optional[float] = None
    hierarchical: bool = False

    @property
    def label(self) -> str:
        if self.kind == "none":
            return "nominal"
        label = f"{self.kind}_{self.rate_ppm}ppm"
        for prefix, value in (("d", self.die_concentration), ("a", self.array_concentration)):
            if value is not None:
                label += f"_{prefix}{value:g}"
        return label

    @property
    def rate(self) -> float:
        return self.rate_ppm / 1e6

    @property
    def sampler_kind(self) -> str:
        # The historical samplers draw a zero-rate open mask for the nominal case.
        return "open" if self.kind == "none" else self.kind

    def to_dict(self) -> dict:
        value = {"kind": self.kind, "rate_ppm": self.rate_ppm}
        if self.hierarchical:
            value.update(die_concentration=self.die_concentration, array_concentration=self.array_concentration)
        return value


def _case(raw: Any, path: str, failure_kinds, hierarchical: bool = False) -> DefectCase:
    concentrations = ("die_concentration", "array_concentration")
    raw = _object(raw, path, ("kind", "rate_ppm") + (concentrations if hierarchical else ()))
    kind = _choice(raw["kind"], f"{path}.kind", ("none",) + tuple(failure_kinds))
    rate = _integer(raw["rate_ppm"], f"{path}.rate_ppm", maximum=1_000_000)
    if (kind == "none") != (rate == 0):
        raise config_error(path, "to use kind 'none' exactly when rate_ppm is 0", dict(raw))
    if not hierarchical:
        return DefectCase(kind, rate)
    values = {
        key: None if raw[key] is None else _number(raw[key], f"{path}.{key}", positive=True)
        for key in concentrations
    }
    if kind == "none" and any(value is not None for value in values.values()):
        raise config_error(path, "to use null concentrations for kind 'none'", dict(raw))
    return DefectCase(kind, rate, hierarchical=True, **values)


def _cases(raw: Any, path: str, failure_kinds, hierarchical: bool = False) -> tuple[DefectCase, ...]:
    cases = tuple(
        _case(item, f"{path}[{i}]", failure_kinds, hierarchical) for i, item in enumerate(_list(raw, path))
    )
    if len(set(cases)) != len(cases):
        raise config_error(path, "to contain unique defect cases", raw)
    return cases


@dataclass(frozen=True)
class Defects:
    placement: str
    persistence: str
    cases: tuple[DefectCase, ...]
    # Hierarchical placement only: consecutive physical arrays (deployment
    # tiles) per die, filled within each group (an OPT decoder layer's MLP).
    arrays_per_die: Optional[int] = None

    def to_dict(self) -> dict:
        value = {"placement": self.placement, "persistence": self.persistence}
        if self.arrays_per_die is not None:
            value["arrays_per_die"] = self.arrays_per_die
        value["cases"] = [case.to_dict() for case in self.cases]
        return value


def _defects(raw: Any, technology: str) -> Defects:
    allowed = TECHNOLOGIES[technology]
    hierarchical = allowed.defect_placement == HIERARCHICAL
    keys = ("placement", "persistence", "cases") + (("arrays_per_die",) if hierarchical else ())
    raw = _object(raw, "defects", keys)
    return Defects(
        placement=_choice(raw["placement"], "defects.placement", (allowed.defect_placement,)),
        persistence=_choice(raw["persistence"], "defects.persistence", ("permanent",)),
        cases=_cases(raw["cases"], "defects.cases", allowed.failure_kinds, hierarchical),
        arrays_per_die=(
            _integer(raw["arrays_per_die"], "defects.arrays_per_die", minimum=1) if hierarchical else None
        ),
    )


@dataclass(frozen=True)
class Deployment:
    scaling: str
    encoding: str
    tile_size: int
    read_noise: float
    calibration: str

    def to_dict(self) -> dict:
        return {
            "scaling": self.scaling,
            "encoding": self.encoding,
            "tile_size": self.tile_size,
            "read_noise": self.read_noise,
            "calibration": self.calibration,
        }


def _deployment(raw: Any, technology: str) -> Deployment:
    raw = _object(raw, "deployment", ("scaling", "encoding", "tile_size", "read_noise", "calibration"))
    return Deployment(
        scaling=_choice(raw["scaling"], "deployment.scaling", SCALINGS),
        encoding=_choice(raw["encoding"], "deployment.encoding", (TECHNOLOGIES[technology].encoding,)),
        tile_size=_integer(raw["tile_size"], "deployment.tile_size", minimum=1),
        read_noise=_number(raw["read_noise"], "deployment.read_noise"),
        # The network family narrows the calibration scheme in _cross_validate.
        calibration=_choice(raw["calibration"], "deployment.calibration", tuple(CALIBRATIONS.values())),
    )


@dataclass(frozen=True)
class Corruption:
    """Temporary training masks; never the deployed fault map."""

    kind: str
    rates_ppm: tuple[int, ...]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "rates_ppm": list(self.rates_ppm)}


# What an HWA arm trains: ``analog`` the analog weights and their calibration
# parameters (the frozen teacher supplies every other digital parameter);
# ``model`` also every digital parameter, as in Rasch et al. 2023 and Büchel et
# al. 2025 (OPT family only).
HWA_SCOPES = ("analog", "model")


@dataclass(frozen=True)
class HwaArm:
    arm_id: str
    method: str
    noise_strength: Optional[float] = None
    corruption: Optional[Corruption] = None
    selection_cases: tuple[DefectCase, ...] = ()
    scope: str = "analog"
    # Adam learning rate of the digital parameters of a ``model``-scope arm.
    digital_learning_rate: Optional[float] = None

    @property
    def trains(self) -> bool:
        return self.method == "hwa"

    def to_dict(self) -> dict:
        if not self.trains:
            return {"id": self.arm_id, "method": self.method}
        # The default analog scope is omitted, so existing arm digests are unchanged.
        scope = (
            {}
            if self.scope == "analog"
            else {"scope": self.scope, "digital_learning_rate": self.digital_learning_rate}
        )
        return {
            "id": self.arm_id,
            "method": self.method,
            "noise_strength": self.noise_strength,
            "corruption": None if self.corruption is None else self.corruption.to_dict(),
            "selection_cases": [case.to_dict() for case in self.selection_cases],
            **scope,
        }


def _hwa_arm(raw: Any, path: str, technology: Technology) -> HwaArm:
    if not isinstance(raw, Mapping):
        raise config_error(path, "to be an object", raw)
    method = _choice(raw.get("method"), f"{path}.method", ("none", "hwa"))
    if method == "none":
        raw = _object(raw, path, ("id", "method"))
        return HwaArm(_identifier(raw["id"], f"{path}.id"), method)
    keys = ("id", "method", "noise_strength", "corruption", "selection_cases")
    if "scope" in raw:
        keys += ("scope", "digital_learning_rate")
    raw = _object(raw, path, keys)
    scope = _choice(raw.get("scope", "analog"), f"{path}.scope", HWA_SCOPES)
    digital_learning_rate = raw.get("digital_learning_rate")
    if scope == "model":
        digital_learning_rate = _number(digital_learning_rate, f"{path}.digital_learning_rate")
    elif digital_learning_rate is not None:
        raise config_error(f"{path}.digital_learning_rate", "to be null for an analog-scope arm", digital_learning_rate)
    corruption = None
    if raw["corruption"] is not None:
        if not technology.corruption_aware_hwa:
            raise config_error(
                f"{path}.corruption", "to be null; the technology has no corruption-aware HWA", raw["corruption"]
            )
        c = _object(raw["corruption"], f"{path}.corruption", ("kind", "rates_ppm"))
        rates = tuple(
            _integer(rate, f"{path}.corruption.rates_ppm[{i}]", minimum=1, maximum=1_000_000)
            for i, rate in enumerate(_list(c["rates_ppm"], f"{path}.corruption.rates_ppm"))
        )
        if tuple(sorted(set(rates))) != rates:
            raise config_error(
                f"{path}.corruption.rates_ppm", "to be unique increasing positive rates", list(rates)
            )
        corruption = Corruption(_choice(c["kind"], f"{path}.corruption.kind", technology.failure_kinds), rates)
    return HwaArm(
        arm_id=_identifier(raw["id"], f"{path}.id"),
        method=method,
        noise_strength=_number(raw["noise_strength"], f"{path}.noise_strength"),
        corruption=corruption,
        selection_cases=_cases(
            raw["selection_cases"],
            f"{path}.selection_cases",
            technology.failure_kinds,
            technology.defect_placement == HIERARCHICAL,
        ),
        scope=scope,
        digital_learning_rate=digital_learning_rate,
    )


@dataclass(frozen=True)
class HwaLrSchedule:
    """A fixed optimizer-update budget; update numbers are one-based."""

    kind: str
    total_steps: int
    warmup_steps: int
    milestone_step: Optional[int] = None
    decay_factor: Optional[float] = None
    min_factor: Optional[float] = None

    def factor(self, step: int) -> float:
        if not 0 <= step <= self.total_steps:
            raise ValueError(f"Expected HWA update in [0, {self.total_steps}]. Provided value: {step!r}.")
        if self.warmup_steps and step <= self.warmup_steps:
            return step / self.warmup_steps
        if self.kind == "linear_warmup_decay":
            return (self.total_steps - step) / (self.total_steps - self.warmup_steps)
        if self.kind == "cosine_warmup_decay":
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_factor + (1 - self.min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0 if step <= self.milestone_step else self.decay_factor

    def boundary_steps(self) -> set[int]:
        steps = {0, 1, self.warmup_steps, self.warmup_steps + 1, self.total_steps}
        if self.milestone_step is not None:
            steps.update((self.milestone_step, self.milestone_step + 1))
        return steps

    def to_dict(self) -> dict:
        value = {"kind": self.kind, "total_steps": self.total_steps, "warmup_steps": self.warmup_steps}
        if self.kind == "warmup_step_decay":
            value.update(milestone_step=self.milestone_step, decay_factor=self.decay_factor)
        elif self.kind == "cosine_warmup_decay":
            value["min_factor"] = self.min_factor
        return value


def _hwa_lr_schedule(raw: Any) -> HwaLrSchedule:
    path = "hwa.lr_schedule"
    if not isinstance(raw, Mapping):
        raise config_error(path, "to be an object", raw)
    kind = _choice(raw.get("kind"), f"{path}.kind",
                   ("linear_warmup_decay", "warmup_step_decay", "cosine_warmup_decay"))
    keys = ("kind", "total_steps", "warmup_steps")
    if kind == "warmup_step_decay":
        keys += ("milestone_step", "decay_factor")
    elif kind == "cosine_warmup_decay":
        keys += ("min_factor",)
    raw = _object(raw, path, keys)
    total = _integer(raw["total_steps"], f"{path}.total_steps", minimum=1)
    warmup = _integer(raw["warmup_steps"], f"{path}.warmup_steps", maximum=total - 1)
    extra = {}
    if kind == "warmup_step_decay":
        extra["milestone_step"] = _integer(
            raw["milestone_step"], f"{path}.milestone_step", minimum=warmup + 1, maximum=total - 1
        )
        extra["decay_factor"] = _number(raw["decay_factor"], f"{path}.decay_factor", positive=True, maximum=1.0)
    elif kind == "cosine_warmup_decay":
        extra["min_factor"] = _number(raw["min_factor"], f"{path}.min_factor", maximum=1.0)
    return HwaLrSchedule(kind, total, warmup, **extra)


@dataclass(frozen=True)
class Hwa:
    noise_model: str
    objective: str
    optimizer: str
    learning_rate: float
    lr_milestones: tuple[int, ...]
    lr_decay: float
    epochs: int
    batch_size: int
    augment: bool
    seed: int
    selection_every: int
    extension_epochs: int
    arms: tuple[HwaArm, ...]
    noise_std: Optional[float] = None
    noise_scaling: Optional[str] = None
    # None: every epoch is one pass over the HWA cohort. Otherwise each epoch is
    # the next ``sequences_per_epoch`` of an endless stream of seeded passes.
    sequences_per_epoch: Optional[int] = None
    # Omitted for historical epoch schedules, preserving their source identities.
    lr_schedule: Optional[HwaLrSchedule] = None

    def arm(self, arm_id: str) -> HwaArm:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise ValueError(f"Expected a declared HWA arm. Provided value: {arm_id!r}.")

    def lr_factor(self, epoch: int) -> float:
        return self.lr_decay ** sum(epoch > milestone for milestone in self.lr_milestones)

    def shared_dict(self) -> dict:
        noise = {"noise_model": self.noise_model}
        if self.noise_std is not None:
            noise.update(noise_std=self.noise_std, noise_scaling=self.noise_scaling)
        stream = {} if self.sequences_per_epoch is None else {"sequences_per_epoch": self.sequences_per_epoch}
        schedule = {} if self.lr_schedule is None else {"lr_schedule": self.lr_schedule.to_dict()}
        return {
            **noise,
            **stream,
            **schedule,
            "objective": self.objective,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "lr_milestones": list(self.lr_milestones),
            "lr_decay": self.lr_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "augment": self.augment,
            "seed": self.seed,
            "selection": {
                "every_epochs": self.selection_every,
                "extension_epochs": self.extension_epochs,
            },
        }

    def to_dict(self) -> dict:
        return {**self.shared_dict(), "arms": [arm.to_dict() for arm in self.arms]}


def _hwa(raw: Any, technology: str) -> Hwa:
    keys = (
        "noise_model",
        "objective",
        "optimizer",
        "learning_rate",
        "lr_milestones",
        "lr_decay",
        "epochs",
        "batch_size",
        "augment",
        "seed",
        "selection",
        "arms",
    )
    noise_model = TECHNOLOGIES[technology].hwa_noise_model
    if noise_model == ADDITIVE_GAUSSIAN:
        keys += ("noise_std", "noise_scaling")
    if isinstance(raw, Mapping) and "sequences_per_epoch" in raw:
        keys += ("sequences_per_epoch",)
    if isinstance(raw, Mapping) and "lr_schedule" in raw:
        keys += ("lr_schedule",)
    raw = _object(raw, "hwa", keys)
    selection = _object(raw["selection"], "hwa.selection", ("every_epochs", "extension_epochs"))
    epochs = _integer(raw["epochs"], "hwa.epochs", minimum=1)
    milestones = tuple(
        _integer(m, f"hwa.lr_milestones[{i}]", minimum=1)
        for i, m in enumerate(_list(raw["lr_milestones"], "hwa.lr_milestones", nonempty=False))
    )
    if tuple(sorted(set(milestones))) != milestones:
        raise config_error("hwa.lr_milestones", "to be unique increasing epochs", list(milestones))
    arms = tuple(
        _hwa_arm(item, f"hwa.arms[{i}]", TECHNOLOGIES[technology])
        for i, item in enumerate(_list(raw["arms"], "hwa.arms"))
    )
    if len({arm.arm_id for arm in arms}) != len(arms):
        raise config_error("hwa.arms", "to have unique ids", [arm.arm_id for arm in arms])
    extra = {}
    if noise_model == ADDITIVE_GAUSSIAN:
        # Standard deviation in normalized weight units (full scale 1); arms scale it.
        extra["noise_std"] = _number(raw["noise_std"], "hwa.noise_std", positive=True)
        extra["noise_scaling"] = _choice(raw["noise_scaling"], "hwa.noise_scaling", NOISE_SCALINGS)
    if "sequences_per_epoch" in raw:
        extra["sequences_per_epoch"] = _integer(raw["sequences_per_epoch"], "hwa.sequences_per_epoch", minimum=1)
    if "lr_schedule" in raw:
        extra["lr_schedule"] = _hwa_lr_schedule(raw["lr_schedule"])
    return Hwa(
        noise_model=_choice(raw["noise_model"], "hwa.noise_model", (noise_model,)),
        objective=_choice(raw["objective"], "hwa.objective", OBJECTIVES),
        optimizer=_choice(raw["optimizer"], "hwa.optimizer", ("adam",)),
        learning_rate=_number(raw["learning_rate"], "hwa.learning_rate", positive=True),
        lr_milestones=milestones,
        lr_decay=_number(raw["lr_decay"], "hwa.lr_decay", positive=True, maximum=1.0),
        epochs=epochs,
        batch_size=_integer(raw["batch_size"], "hwa.batch_size", minimum=1),
        augment=_boolean(raw["augment"], "hwa.augment"),
        seed=_integer(raw["seed"], "hwa.seed", maximum=MAX_SEED),
        selection_every=_integer(selection["every_epochs"], "hwa.selection.every_epochs", minimum=1),
        extension_epochs=_integer(selection["extension_epochs"], "hwa.selection.extension_epochs"),
        arms=arms,
        **extra,
    )


@dataclass(frozen=True)
class Relaxation:
    model: str
    seconds: Optional[float] = None
    compensation: Optional[bool] = None

    def to_dict(self) -> dict:
        if self.model == "none":
            return {"model": "none"}
        return {"model": self.model, "seconds": self.seconds, "compensation": self.compensation}


@dataclass(frozen=True)
class ProgramVerify:
    method: str
    relaxation: Relaxation
    max_cycles: Optional[int] = None
    tolerance_steps: Optional[float] = None
    acceptance_percent: Optional[float] = None

    def to_dict(self) -> dict:
        value = {"method": self.method}
        if self.method == "closed_loop":
            value.update(max_cycles=self.max_cycles, tolerance_steps=self.tolerance_steps)
        if self.acceptance_percent is not None:
            value["acceptance_percent"] = self.acceptance_percent
        value["relaxation"] = self.relaxation.to_dict()
        return value


def _program_verify(raw: Any, technology: str) -> ProgramVerify:
    if not isinstance(raw, Mapping):
        raise config_error("program_verify", "to be an object", raw)
    allowed = TECHNOLOGIES[technology]
    method = _choice(raw.get("method"), "program_verify.method", allowed.program_methods)
    keys = ("method", "relaxation") + (("max_cycles", "tolerance_steps") if method == "closed_loop" else ())
    if technology == "cmo":
        keys += ("acceptance_percent",)
    raw = _object(raw, "program_verify", keys)
    relaxation_raw = raw["relaxation"]
    if not isinstance(relaxation_raw, Mapping):
        raise config_error("program_verify.relaxation", "to be an object", relaxation_raw)
    model = _choice(
        relaxation_raw.get("model"), "program_verify.relaxation.model", allowed.relaxation_models
    )
    if model == "none":
        _object(relaxation_raw, "program_verify.relaxation", ("model",))
        relaxation = Relaxation("none")
    else:
        r = _object(relaxation_raw, "program_verify.relaxation", ("model", "seconds", "compensation"))
        relaxation = Relaxation(
            model,
            _number(r["seconds"], "program_verify.relaxation.seconds", positive=True),
            _boolean(r["compensation"], "program_verify.relaxation.compensation"),
        )
    extra = {}
    if method == "closed_loop":
        extra = dict(
            max_cycles=_integer(raw["max_cycles"], "program_verify.max_cycles", minimum=1),
            tolerance_steps=_number(raw["tolerance_steps"], "program_verify.tolerance_steps", positive=True),
        )
    if technology == "cmo":
        acceptance = _number(raw["acceptance_percent"], "program_verify.acceptance_percent", positive=True)
        if acceptance not in CMO_ACCEPTANCE_PERCENT:
            raise config_error(
                "program_verify.acceptance_percent",
                f"to be a fitted CMO acceptance range in {list(CMO_ACCEPTANCE_PERCENT)}",
                raw["acceptance_percent"],
            )
        extra["acceptance_percent"] = acceptance
    return ProgramVerify(method=method, relaxation=relaxation, **extra)


@dataclass(frozen=True)
class DigitalLora:
    rank: int
    alpha: float
    learning_rate: float

    def to_dict(self) -> dict:
        return {"kind": "lora", "rank": self.rank, "alpha": self.alpha, "learning_rate": self.learning_rate}


@dataclass(frozen=True)
class OnchipArm:
    arm_id: str
    weights: str
    calibration: bool
    digital_adapter: Optional[DigitalLora] = None

    @property
    def trains(self) -> bool:
        return self.weights != "hold" or self.calibration or self.digital_adapter is not None

    def to_dict(self) -> dict:
        value = {"id": self.arm_id, "weights": self.weights, "calibration": self.calibration}
        if self.digital_adapter is not None:
            value["digital_adapter"] = self.digital_adapter.to_dict()
        return value


@dataclass(frozen=True)
class Onchip:
    update_law: str
    objective: str
    learning_rate: float
    calibration_learning_rate: float
    epochs: int
    batch_size: int
    seed: int
    arms: tuple[OnchipArm, ...]
    pulses_per_update: Optional[int] = None
    tolerance_steps: Optional[float] = None
    max_pulses_per_cell: Optional[int] = None
    # Both cohorts contain training data. Selecting an existing cohort leaves
    # the frozen HWA source and feature-cache provenance unchanged.
    training_cohort: Optional[str] = None

    @property
    def cohort_key(self) -> str:
        return "hwa" if self.training_cohort == "hwa" else "training"

    def to_dict(self) -> dict:
        value = {
            "update_law": self.update_law,
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "calibration_learning_rate": self.calibration_learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }
        if self.update_law == "om_closed_loop_pulse_adam":
            value.update(pulses_per_update=self.pulses_per_update, tolerance_steps=self.tolerance_steps)
        if self.update_law.startswith("om_"):
            value["max_pulses_per_cell"] = self.max_pulses_per_cell
        if self.training_cohort is not None:
            value["training_cohort"] = self.training_cohort
        value["arms"] = [arm.to_dict() for arm in self.arms]
        return value


def _onchip(raw: Any, technology: str) -> Onchip:
    if not isinstance(raw, Mapping):
        raise config_error("onchip", "to be an object", raw)
    law = _choice(raw.get("update_law"), "onchip.update_law", TECHNOLOGIES[technology].update_laws)
    keys = (
        "update_law",
        "objective",
        "learning_rate",
        "calibration_learning_rate",
        "epochs",
        "batch_size",
        "seed",
        "arms",
    )
    if law == "om_closed_loop_pulse_adam":
        keys += ("pulses_per_update", "tolerance_steps", "max_pulses_per_cell")
    elif law == "om_open_loop_pulse_adam":
        keys += ("max_pulses_per_cell",)
    if "training_cohort" in raw:
        keys += ("training_cohort",)
    raw = _object(raw, "onchip", keys)
    arms = []
    for index, item in enumerate(_list(raw["arms"], "onchip.arms")):
        path = f"onchip.arms[{index}]"
        keys = ("id", "weights", "calibration") + (("digital_adapter",) if isinstance(item, Mapping) and "digital_adapter" in item else ())
        item = _object(item, path, keys)
        adapter = None
        if "digital_adapter" in item:
            a = _object(item["digital_adapter"], f"{path}.digital_adapter", ("kind", "rank", "alpha", "learning_rate"))
            _choice(a["kind"], f"{path}.digital_adapter.kind", ("lora",))
            adapter = DigitalLora(
                rank=_integer(a["rank"], f"{path}.digital_adapter.rank", minimum=1, maximum=768),
                alpha=_number(a["alpha"], f"{path}.digital_adapter.alpha", positive=True),
                learning_rate=_number(a["learning_rate"], f"{path}.digital_adapter.learning_rate", positive=True),
            )
            if item["weights"] != "hold" or item["calibration"] is not False:
                raise config_error(path, "to hold weights and calibration fixed for digital LoRA", item)
        arms.append(
            OnchipArm(
                arm_id=_identifier(item["id"], f"{path}.id"),
                weights=_choice(item["weights"], f"{path}.weights", ("hold", "rewrite", "learn")),
                calibration=_boolean(item["calibration"], f"{path}.calibration"),
                digital_adapter=adapter,
            )
        )
    arms = tuple(arms)
    if len({arm.arm_id for arm in arms}) != len(arms):
        raise config_error("onchip.arms", "to have unique ids", [arm.arm_id for arm in arms])
    settings = {(arm.weights, arm.calibration, arm.digital_adapter) for arm in arms}
    if len(settings) != len(arms):
        raise config_error("onchip.arms", "to have unique weight/calibration settings", [a.to_dict() for a in arms])
    for arm in arms:
        # A weight-writing arm is interpretable only beside its matched
        # no-weight-write control with the same digital calibration.
        if (arm.weights != "hold" or arm.digital_adapter is not None) and ("hold", arm.calibration, None) not in settings:
            raise config_error(
                "onchip.arms",
                f"to include a hold arm with calibration={arm.calibration} matching {arm.arm_id!r}",
                [a.to_dict() for a in arms],
            )
        if arm.weights == "rewrite" and law == "om_open_loop_pulse_adam":
            raise config_error(
                "onchip.arms", "to omit verified rewrite controls under the open-loop law", arm.to_dict()
            )
    extra = {}
    if "training_cohort" in raw:
        extra["training_cohort"] = _choice(raw["training_cohort"], "onchip.training_cohort", ("recovery", "hwa"))
    if law.startswith("om_"):
        cap = raw["max_pulses_per_cell"]
        extra["max_pulses_per_cell"] = (
            None if cap is None else _integer(cap, "onchip.max_pulses_per_cell", minimum=1)
        )
    if law == "om_closed_loop_pulse_adam":
        extra["pulses_per_update"] = _integer(raw["pulses_per_update"], "onchip.pulses_per_update", minimum=1)
        extra["tolerance_steps"] = _number(raw["tolerance_steps"], "onchip.tolerance_steps", positive=True)
    return Onchip(
        update_law=law,
        objective=_choice(raw["objective"], "onchip.objective", OBJECTIVES),
        learning_rate=_number(raw["learning_rate"], "onchip.learning_rate"),
        calibration_learning_rate=_number(raw["calibration_learning_rate"], "onchip.calibration_learning_rate"),
        epochs=_integer(raw["epochs"], "onchip.epochs", minimum=1),
        batch_size=_integer(raw["batch_size"], "onchip.batch_size", minimum=1),
        seed=_integer(raw["seed"], "onchip.seed", maximum=MAX_SEED),
        arms=arms,
        **extra,
    )


# ---------------------------------------------------------------------------
# The lifecycle and its content-addressed sections.

SECTIONS = (
    "question",
    "network",
    "data",
    "devices",
    "defects",
    "deployment",
    "hwa",
    "program_verify",
    "onchip",
)
# Optional sections are omitted from the document and from every provenance
# record when absent, so lifecycles without them keep their digests.
OPTIONAL_SECTIONS = ("transfer_tests",)
LIFECYCLE_KEYS = ("schema", "schema_version", "lifecycle_id") + SECTIONS

# Upstream artifacts are reusable exactly when the sections they depend on are
# unchanged; the question text, unrelated arms and later stages may change.
ARTIFACT_SECTIONS = {
    "feature_cache": ("network", "data", "transfer_tests"),
    "device_bundle": ("network", "devices", "program_verify"),
    "hwa_source": ("network", "data", "devices", "deployment", "program_verify"),
    "deployment": ("network", "data", "devices", "defects", "deployment", "program_verify", "transfer_tests"),
}


@dataclass(frozen=True)
class Lifecycle:
    lifecycle_id: str
    question: Question
    network: Network | OptNetwork
    data: Data | TextData
    devices: Devices
    defects: Defects
    deployment: Deployment
    hwa: Hwa
    program_verify: ProgramVerify
    # None: a deployment-only loop with no on-chip stage.
    onchip: Optional[Onchip]
    transfer_tests: Optional[TransferTests] = None

    def present_sections(self) -> tuple[str, ...]:
        return SECTIONS + tuple(name for name in OPTIONAL_SECTIONS if getattr(self, name) is not None)

    @property
    def technology(self) -> Technology:
        return TECHNOLOGIES[self.devices.technology]

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "lifecycle_id": self.lifecycle_id,
            **{name: self._section(name) for name in self.present_sections()},
        }

    def digest(self) -> str:
        return content_hash(self.to_dict())

    def section_digest(self, name: str) -> str:
        if name not in SECTIONS + OPTIONAL_SECTIONS:
            raise ValueError(f"Expected a lifecycle section. Provided value: {name!r}.")
        return content_hash(self._section(name))

    def _section(self, name: str):
        value = getattr(self, name)
        return None if value is None else value.to_dict()

    def hwa_arm_digest(self, arm_id: str) -> str:
        return content_hash({"shared": self.hwa.shared_dict(), "arm": self.hwa.arm(arm_id).to_dict()})


def _cross_validate(lifecycle: Lifecycle) -> None:
    devices, data, network = lifecycle.devices, lifecycle.data, lifecycle.network
    family = network.family
    if lifecycle.onchip and any(a.digital_adapter for a in lifecycle.onchip.arms) and family != OPT_FAMILY:
        raise config_error("onchip.arms.digital_adapter", "to be absent outside opt_mlp_suffix", family)
    if lifecycle.question.primary_metric not in PRIMARY_METRICS[family]:
        raise config_error(
            "question.primary_metric",
            f"to be one of {list(PRIMARY_METRICS[family])} for {family}",
            lifecycle.question.primary_metric,
        )
    if lifecycle.deployment.calibration != CALIBRATIONS[family]:
        raise config_error(
            "deployment.calibration", f"to be {CALIBRATIONS[family]!r} for {family}", lifecycle.deployment.calibration
        )
    if family != OPT_FAMILY and lifecycle.deployment.scaling != "per_layer_absmax":
        raise config_error(
            "deployment.scaling", f"to be 'per_layer_absmax' for {family}", lifecycle.deployment.scaling
        )
    if family == OPT_FAMILY and lifecycle.hwa.augment:
        raise config_error("hwa.augment", f"to be false; {family} defines no augmentation", True)
    model_scope = [arm.arm_id for arm in lifecycle.hwa.arms if arm.scope == "model"]
    if model_scope and family != OPT_FAMILY:
        raise config_error("hwa.arms.scope", f"to be analog; {family} defines no whole-model HWA", model_scope)
    if lifecycle.transfer_tests is not None and family != OPT_FAMILY:
        raise config_error("transfer_tests", f"to be absent; {family} defines no transfer cohorts", "present")
    if lifecycle.transfer_tests is not None and any(
        t.corpus_sha256 == data.corpus_sha256 and t.split == "validation" for t in lifecycle.transfer_tests.tests
    ):
        raise config_error("transfer_tests", "not to reuse the lifecycle's own validation split", "validation")
    streamed = lifecycle.hwa.sequences_per_epoch
    if streamed is not None and family != OPT_FAMILY:
        raise config_error("hwa.sequences_per_epoch", f"to be absent; {family} streams no cohort", streamed)
    if streamed is not None and streamed > data.hwa_sequences:
        raise config_error(
            "hwa.sequences_per_epoch", f"to be at most data.hwa_sequences ({data.hwa_sequences})", streamed
        )
    hwa = lifecycle.hwa
    if hwa.lr_schedule is not None:
        if hwa.lr_milestones or hwa.lr_decay != 1.0 or hwa.extension_epochs:
            raise config_error(
                "hwa.lr_schedule",
                "to use neutral legacy settings: lr_milestones [], lr_decay 1 and selection.extension_epochs 0",
                hwa.shared_dict(),
            )
        examples = data.cohort_size("hwa")
        per_epoch = min(streamed, examples) if streamed is not None else examples
        steps = hwa.epochs * ((per_epoch + hwa.batch_size - 1) // hwa.batch_size)
        if hwa.lr_schedule.total_steps != steps:
            raise config_error(
                "hwa.lr_schedule.total_steps",
                f"to match the full training budget ({steps} updates, including any data.max_examples cap); "
                "declare a separate schedule for a diagnostic smoke",
                hwa.lr_schedule.total_steps,
            )
    if family == CIFAR_FAMILY and data.development_images % network.classes:
        raise config_error(
            "data.development_images",
            f"to be a multiple of the {network.classes} {network.dataset} classes",
            data.development_images,
        )
    identities = set(devices.assignment_seeds) | set(devices.selection_seeds)
    if set(devices.assignment_seeds) & set(devices.selection_seeds):
        raise config_error(
            "devices.selection_seeds",
            "to be disjoint from the assignment seeds",
            sorted(set(devices.assignment_seeds) & set(devices.selection_seeds)),
        )
    endpoints = {devices.endpoint_seed(seed) for seed in identities}
    if endpoints & identities or any(seed > MAX_SEED for seed in endpoints):
        raise config_error(
            "devices.endpoint_seed_offset",
            "to map every array to an endpoint seed outside the array identities",
            devices.endpoint_seed_offset,
        )
    if devices.characterization is not None:
        kernel = {devices.characterization.fit_seed, devices.characterization.heldout_seed}
        if kernel & (identities | endpoints):
            raise config_error(
                "devices.characterization",
                "to use population seeds separate from every array",
                sorted(kernel & (identities | endpoints)),
            )
    if any(arm.trains for arm in lifecycle.hwa.arms) and not devices.selection_seeds:
        raise config_error(
            "devices.selection_seeds",
            "to name independent selection arrays for HWA checkpoint selection",
            [],
        )


def parse_lifecycle(payload: Any) -> Lifecycle:
    """Parse one lifecycle document strictly."""

    keys = LIFECYCLE_KEYS + tuple(
        name for name in OPTIONAL_SECTIONS if isinstance(payload, Mapping) and name in payload
    )
    raw = _object(payload, "lifecycle", keys)
    if raw["schema"] != SCHEMA or type(raw["schema_version"]) is not int or raw["schema_version"] != SCHEMA_VERSION:
        raise config_error(
            "lifecycle schema", f"to be {SCHEMA!r} version {SCHEMA_VERSION}", [raw["schema"], raw["schema_version"]]
        )
    devices = _devices(raw["devices"])
    technology = devices.technology
    network = _network(raw["network"])
    lifecycle = Lifecycle(
        lifecycle_id=_identifier(raw["lifecycle_id"], "lifecycle_id", _LIFECYCLE_ID),
        question=_question(raw["question"]),
        network=network,
        data=_text_data(raw["data"], network) if network.family == OPT_FAMILY else _data(raw["data"]),
        devices=devices,
        defects=_defects(raw["defects"], technology),
        deployment=_deployment(raw["deployment"], technology),
        hwa=_hwa(raw["hwa"], technology),
        program_verify=_program_verify(raw["program_verify"], technology),
        onchip=None if raw["onchip"] is None else _onchip(raw["onchip"], technology),
        transfer_tests=_transfer_tests(raw["transfer_tests"]) if "transfer_tests" in raw else None,
    )
    _cross_validate(lifecycle)
    return lifecycle


def load_lifecycle(path: Path | str) -> Lifecycle:
    """Read and parse a lifecycle JSON document."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Expected a readable lifecycle file. Provided value: {str(path)!r}. {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"Expected the lifecycle to contain valid JSON. Provided value: {str(path)!r} "
            f"(line {error.lineno}, column {error.colno}: {error.msg})."
        ) from error
    return parse_lifecycle(payload)


# ---------------------------------------------------------------------------
# Stage selection, execution settings and the registered stage config.


@dataclass(frozen=True)
class StageSelection:
    kind: str
    hwa_arm: Optional[str] = None
    array_seed: Optional[int] = None

    @property
    def stage_id(self) -> str:
        parts = [self.kind] + ([self.hwa_arm] if self.hwa_arm else []) + (
            [str(self.array_seed)] if self.array_seed is not None else []
        )
        return "__".join(parts)

    @property
    def case_id(self) -> str:
        if self.hwa_arm is None:
            return "shared"
        return self.hwa_arm if self.array_seed is None else f"{self.hwa_arm}__{self.array_seed}"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "hwa_arm": self.hwa_arm, "array_seed": self.array_seed}


@dataclass(frozen=True)
class Execution:
    """How a stage executes; it is not a scientific lifecycle setting."""

    device: str
    cpu_threads: int
    disable_cudnn: bool

    def to_dict(self) -> dict:
        return {"device": self.device, "cpu_threads": self.cpu_threads, "disable_cudnn": self.disable_cudnn}


@dataclass(frozen=True)
class StageConfig:
    stage: StageSelection
    execution: Execution
    lifecycle: Lifecycle
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "schema_version": self.schema_version,
            "stage": self.stage.to_dict(),
            "execution": self.execution.to_dict(),
            "lifecycle": self.lifecycle.to_dict(),
        }


def _selection(raw: Any, lifecycle: Lifecycle) -> StageSelection:
    raw = _object(raw, "stage", ("kind", "hwa_arm", "array_seed"))
    kind = _choice(raw["kind"], "stage.kind", STAGE_KINDS)
    arm, seed = raw["hwa_arm"], raw["array_seed"]
    if kind in ("prepare", "devices"):
        if arm is not None or seed is not None:
            raise config_error("stage", f"to select no arm or array for {kind}", dict(raw))
        return StageSelection(kind)
    if not isinstance(arm, str) or arm not in {a.arm_id for a in lifecycle.hwa.arms}:
        raise config_error("stage.hwa_arm", "to name a declared HWA arm", arm)
    if kind == "hwa":
        if seed is not None:
            raise config_error("stage.array_seed", "to be null for an HWA stage", seed)
        return StageSelection(kind, arm)
    if type(seed) is not int or seed not in lifecycle.devices.assignment_seeds:
        raise config_error("stage.array_seed", "to name a declared assignment seed", seed)
    return StageSelection(kind, arm, seed)


def parse_execution(raw: Any) -> Execution:
    raw = _object(raw, "execution", ("device", "cpu_threads", "disable_cudnn"))
    device = raw["device"]
    if not isinstance(device, str) or not (
        device in ("cpu", "cuda") or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise config_error("execution.device", "to be cpu, cuda or cuda:N", device)
    return Execution(
        device=device,
        cpu_threads=_integer(raw["cpu_threads"], "execution.cpu_threads", minimum=1),
        disable_cudnn=_boolean(raw["disable_cudnn"], "execution.disable_cudnn"),
    )


def parse_stage_config(payload: Any) -> StageConfig:
    """Registry parser for one stage of an embedded lifecycle."""

    raw = _object(payload, "config", ("experiment_id", "schema_version", "stage", "execution", "lifecycle"))
    if raw["experiment_id"] != EXPERIMENT_ID or type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise config_error("experiment/schema", f"to be {EXPERIMENT_ID}/1", [raw["experiment_id"], raw["schema_version"]])
    lifecycle = parse_lifecycle(raw["lifecycle"])
    return StageConfig(
        stage=_selection(raw["stage"], lifecycle),
        execution=parse_execution(raw["execution"]),
        lifecycle=lifecycle,
    )


def resolve_stage_config(document: StageConfig, mode: RunMode) -> StageConfig:
    if mode != RunMode.TRAIN:
        raise config_error("the requested run mode", "to be train", mode.value)
    return document


# ---------------------------------------------------------------------------
# Stage DAG and artifact provenance.

STAGE_OUTPUTS = {
    "prepare": ("feature_cache", "mapping"),
    "devices": ("device_bundle",),
    "hwa": ("hwa_source",),
    "deploy": ("deployment",),
    "onchip": ("onchip_states", "measurements"),
}


def stage_selections(lifecycle: Lifecycle) -> list[StageSelection]:
    """All stages of a lifecycle in dependency order."""

    stages = [StageSelection("prepare"), StageSelection("devices")]
    stages += [StageSelection("hwa", arm.arm_id) for arm in lifecycle.hwa.arms]
    for kind in ("deploy", "onchip") if lifecycle.onchip is not None else ("deploy",):
        stages += [
            StageSelection(kind, arm.arm_id, seed)
            for arm in lifecycle.hwa.arms
            for seed in lifecycle.devices.assignment_seeds
        ]
    return stages


def stage_inputs(selection: StageSelection) -> dict[str, Optional[tuple[str, str]]]:
    """CLI input roles; ``None`` marks the external pinned teacher checkpoint."""

    inputs: dict[str, Optional[tuple[str, str]]] = {"teacher_weights": None}
    if selection.kind in ("prepare", "devices"):
        return inputs
    inputs["device_data"] = ("prepare", "feature_cache")
    inputs["device_model"] = ("devices", "device_bundle")
    if selection.kind == "deploy":
        inputs["weights"] = (StageSelection("hwa", selection.hwa_arm).stage_id, "hwa_source")
    elif selection.kind == "onchip":
        upstream = StageSelection("deploy", selection.hwa_arm, selection.array_seed).stage_id
        inputs["device_state"] = (upstream, "deployment")
    return inputs


def provenance(lifecycle: Lifecycle, kind: str, *, hwa_arm: Optional[str] = None) -> dict:
    """Content identity of the lifecycle sections that determined an artifact."""

    if kind not in ARTIFACT_SECTIONS:
        raise ValueError(f"Expected a lifecycle artifact kind. Provided value: {kind!r}.")
    record = {
        "schema": PROVENANCE_SCHEMA,
        "kind": kind,
        "lifecycle_id": lifecycle.lifecycle_id,
        "lifecycle_sha256": lifecycle.digest(),
        "sections": {
            name: lifecycle.section_digest(name)
            for name in ARTIFACT_SECTIONS[kind]
            if name in lifecycle.present_sections()
        },
    }
    if kind in ("hwa_source", "deployment"):
        if hwa_arm is None:
            raise ValueError(f"Expected an HWA arm for a {kind} artifact.")
        record.update(hwa_arm=hwa_arm, hwa_arm_sha256=lifecycle.hwa_arm_digest(hwa_arm))
    return record


def verify_provenance(
    record: Any, lifecycle: Lifecycle, kind: str, *, hwa_arm: Optional[str] = None
) -> None:
    """Accept an upstream artifact only if its determining sections match."""

    expected = provenance(lifecycle, kind, hwa_arm=hwa_arm)
    if (
        not isinstance(record, Mapping)
        or record.get("schema") != PROVENANCE_SCHEMA
        or record.get("kind") != kind
        or not isinstance(record.get("sections"), Mapping)
    ):
        raise ValueError(f"Expected a lifecycle {kind} artifact. Provided value: {record!r}.")
    changed = sorted(
        name for name, digest in expected["sections"].items() if record["sections"].get(name) != digest
    )
    if changed:
        raise ValueError(
            f"Expected the {kind} artifact to share the lifecycle sections it depends on. "
            f"Provided value: changed sections {changed} (artifact lifecycle "
            f"{record.get('lifecycle_id')!r})."
        )
    if hwa_arm is not None and (
        record.get("hwa_arm") != hwa_arm or record.get("hwa_arm_sha256") != expected["hwa_arm_sha256"]
    ):
        raise ValueError(
            f"Expected the {kind} artifact to come from the declared HWA arm settings. "
            f"Provided value: arm {record.get('hwa_arm')!r}."
        )
