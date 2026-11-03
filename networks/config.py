"""networks.config."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from core.contracts import config_error
from core.config import MAX_SEED, _LIFECYCLE_ID, _SHA256, _choice, _identifier, _integer, _list, _number, _object
from typing import Any
from core.config import _choice, _integer, _number, _object
from hardware.config import SCALINGS, TECHNOLOGIES


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


CALIBRATION = "output_gain_bn_affine_classifier_bias"


OPT_CALIBRATION = "output_gain_mlp_layer_norm_mlp_bias"


# What each network family can report, calibrate and augment.
# Task metrics need a declared ``task`` section (labelled multiple-choice evaluation).
TASK_PRIMARY_METRICS = ("task_accuracy_norm_percent", "task_accuracy_percent")


PRIMARY_METRICS = {
    CIFAR_FAMILY: ("teacher_kl", "accuracy_percent"),
    OPT_FAMILY: ("teacher_kl", "perplexity") + TASK_PRIMARY_METRICS,
}


TASK_NAMES = ("hellaswag", "sst2", "piqa")


TASK_DECISIONS = ("acc_norm", "acc")


CALIBRATIONS = {CIFAR_FAMILY: CALIBRATION, OPT_FAMILY: OPT_CALIBRATION}


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
class OptTeacher:
    """A fine-tuned FP32 teacher (LoRA merged into the weights), pinned by SHA-256."""

    name: str
    sha256: str

    def to_dict(self) -> dict:
        return {"name": self.name, "sha256": self.sha256}


@dataclass(frozen=True)
class AnalogLora:
    """Low-rank adapters A (rank x in) and B (out x rank) of every analog MLP matrix, on their own arrays."""

    rank: int
    alpha: float

    def to_dict(self) -> dict:
        return {"rank": self.rank, "alpha": self.alpha}


@dataclass(frozen=True)
class OptNetwork:
    """OPT decoder whose last ``analog_decoder_layers`` MLPs (fc1, fc2) are crossbars.

    Embeddings, the earlier decoder layers, every attention block, LayerNorms,
    biases and the tied LM head stay digital. ``teacher`` names a declared
    fine-tuned checkpoint instead of the pinned pretrained one; it is part of
    the network section, so every artifact identity includes it.
    """

    family: str
    model: str
    analog_decoder_layers: int
    teacher: Optional[OptTeacher] = None
    analog_lora: Optional[AnalogLora] = None

    @property
    def decoder_layers(self) -> int:
        return OPT_MODELS[self.model]["decoder_layers"]

    @property
    def max_positions(self) -> int:
        return OPT_MODELS[self.model]["max_positions"]

    @property
    def adapter_weights(self) -> int:
        spec = OPT_MODELS[self.model]
        rank = 0 if self.analog_lora is None else self.analog_lora.rank
        return self.analog_decoder_layers * 2 * rank * (spec["hidden_size"] + spec["ffn_dim"])

    @property
    def analog_weights(self) -> int:
        spec = OPT_MODELS[self.model]
        return self.analog_decoder_layers * 2 * spec["hidden_size"] * spec["ffn_dim"] + self.adapter_weights

    @property
    def binding_namespace(self) -> str:
        return self.model

    def describe(self) -> str:
        teacher = "" if self.teacher is None else f" (fine-tuned teacher {self.teacher.name})"
        lora = ("" if self.analog_lora is None else
                f", with rank-{self.analog_lora.rank} analog LoRA adapters ({self.adapter_weights:,} weights)")
        return (f"{self.model}{teacher}, analog fc1/fc2 in the last {self.analog_decoder_layers} "
                f"of {self.decoder_layers} decoder layers{lora}")

    def summary(self) -> dict:
        return {
            "model": self.model,
            "analog_decoder_layers": self.analog_decoder_layers,
            "analog_layer_indices": list(range(self.decoder_layers - self.analog_decoder_layers, self.decoder_layers)),
            **({"teacher": self.teacher.to_dict()} if self.teacher is not None else {}),
            **({"analog_lora": self.analog_lora.to_dict()} if self.analog_lora is not None else {}),
        }

    def mapping_fields(self) -> dict:
        return {"analog_decoder_layers": self.analog_decoder_layers, "analog_modules": ["fc1", "fc2"]}

    def result_fields(self) -> dict:
        return {"model": self.model, "analog_decoder_layers": self.analog_decoder_layers,
                **({"fine_tuned_teacher": self.teacher.to_dict()} if self.teacher is not None else {}),
                **({"analog_lora": self.analog_lora.to_dict()} if self.analog_lora is not None else {})}

    def to_dict(self) -> dict:
        value = {"family": self.family, "model": self.model, "analog_decoder_layers": self.analog_decoder_layers}
        if self.teacher is not None:
            value["teacher"] = self.teacher.to_dict()
        if self.analog_lora is not None:
            value["analog_lora"] = self.analog_lora.to_dict()
        return value


def _network(raw: Any) -> Network | OptNetwork:
    if not isinstance(raw, Mapping):
        raise config_error("network", "to be an object", raw)
    family = _choice(raw.get("family"), "network.family", (CIFAR_FAMILY, OPT_FAMILY))
    if family == OPT_FAMILY:
        keys = ("family", "model", "analog_decoder_layers") + tuple(k for k in ("teacher", "analog_lora") if k in raw)
        raw = _object(raw, "network", keys)
        model = _choice(raw["model"], "network.model", tuple(OPT_MODELS))
        teacher = None
        if "teacher" in raw:
            item = _object(raw["teacher"], "network.teacher", ("name", "sha256"))
            if not isinstance(item["sha256"], str) or not _SHA256.match(item["sha256"]):
                raise config_error("network.teacher.sha256", "to be a lowercase SHA-256 digest", item["sha256"])
            teacher = OptTeacher(_identifier(item["name"], "network.teacher.name", _LIFECYCLE_ID), item["sha256"])
        lora = None
        if "analog_lora" in raw:
            item = _object(raw["analog_lora"], "network.analog_lora", ("rank", "alpha"))
            lora = AnalogLora(_integer(item["rank"], "network.analog_lora.rank", minimum=1, maximum=256),
                              _number(item["alpha"], "network.analog_lora.alpha", positive=True))
        return OptNetwork(
            family=family,
            model=model,
            analog_decoder_layers=_integer(
                raw["analog_decoder_layers"],
                "network.analog_decoder_layers",
                minimum=1,
                maximum=OPT_MODELS[model]["decoder_layers"],
            ),
            teacher=teacher,
            analog_lora=lora,
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
class Task:
    """Labelled multiple-choice evaluation of a task-fine-tuned OPT (``networks.tasks``).

    The pinned choice-task file supplies ``task_development`` (the first
    ``development_examples`` held-out train examples) and ``task_test`` (the
    first ``evaluation_examples`` of the official validation split). Both are
    scored at every evaluation. Like transfer tests, they only add
    measurements: feature caches and deployments depend on them, HWA sources do not.
    """

    name: str
    choices_name: str
    choices_sha256: str
    decision: str
    development_examples: int
    evaluation_examples: int
    # Labelled train examples for task-objective HWA (analog LoRA); omitted otherwise.
    training_examples: Optional[int] = None

    def to_dict(self) -> dict:
        value = {
            "name": self.name,
            "choices": {"name": self.choices_name, "sha256": self.choices_sha256},
            "decision": self.decision,
            "development_examples": self.development_examples,
            "evaluation_examples": self.evaluation_examples,
        }
        if self.training_examples is not None:
            value["training_examples"] = self.training_examples
        return value


def _task(raw: Any) -> Task:
    keys = ("name", "choices", "decision", "development_examples", "evaluation_examples")
    raw = _object(raw, "task", keys + (("training_examples",) if isinstance(raw, Mapping) and "training_examples" in raw else ()))
    choices = _object(raw["choices"], "task.choices", ("name", "sha256"))
    if not isinstance(choices["sha256"], str) or not _SHA256.match(choices["sha256"]):
        raise config_error("task.choices.sha256", "to be a lowercase SHA-256 digest", choices["sha256"])
    return Task(
        name=_choice(raw["name"], "task.name", TASK_NAMES),
        choices_name=_identifier(choices["name"], "task.choices.name", _LIFECYCLE_ID),
        choices_sha256=choices["sha256"],
        decision=_choice(raw["decision"], "task.decision", TASK_DECISIONS),
        development_examples=_integer(raw["development_examples"], "task.development_examples", minimum=1),
        evaluation_examples=_integer(raw["evaluation_examples"], "task.evaluation_examples", minimum=1),
        training_examples=(_integer(raw["training_examples"], "task.training_examples", minimum=1)
                           if "training_examples" in raw else None),
    )


@dataclass(frozen=True)
class RecoveryViews:
    """Token-matched recovery contexts independent of frozen HWA provenance."""

    context_length: int
    parent_length: int
    parent_sequences: int
    score_stride: int
    evaluation_lengths: tuple[int, ...]

    @property
    def sequences(self):
        return self.parent_sequences * self.parent_length // self.context_length

    @property
    def input_tokens(self):
        return self.parent_sequences * self.parent_length

    @property
    def scored_tokens(self):
        return self.input_tokens // self.score_stride * (self.score_stride - 1)

    def to_dict(self):
        return {"context_length": self.context_length, "parent_length": self.parent_length,
                "parent_sequences": self.parent_sequences, "score_stride": self.score_stride,
                "evaluation_lengths": list(self.evaluation_lengths)}


def _recovery_views(raw):
    raw = _object(raw, "recovery_views", ("context_length", "parent_length", "parent_sequences",
                                           "score_stride", "evaluation_lengths"))
    sizes = {name: _integer(raw[name], f"recovery_views.{name}", minimum=2, maximum=2048)
             for name in ("context_length", "parent_length", "score_stride")}
    lengths = tuple(_integer(v, "recovery_views.evaluation_lengths", minimum=2, maximum=2048)
                    for v in _list(raw["evaluation_lengths"], "recovery_views.evaluation_lengths"))
    if len(set(lengths)) != len(lengths):
        raise config_error("recovery_views.evaluation_lengths", "to be unique", lengths)
    return RecoveryViews(**sizes, parent_sequences=_integer(raw["parent_sequences"],
                         "recovery_views.parent_sequences", minimum=1), evaluation_lengths=lengths)


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
