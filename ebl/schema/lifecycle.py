"""ebl.schema.lifecycle — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional
from core.artifacts import content_hash
from core.contracts import ConfigError, RunMode, config_error
from core.config import MAX_SEED, _LIFECYCLE_ID, _choice, _identifier, _object, _text
from hardware.config import Defects, Devices, Retention, TECHNOLOGIES, Technology, _defects, _devices, _retention
from networks.config import Deployment, _deployment
from methods.config import DriftCompensation, Hwa, OUTPUT_CALIBRATIONS, Onchip, ProgramVerify, SOURCE_KL, TASK_HWA_OBJECTIVE, _drift_compensation, _hwa, _onchip, _program_verify
from networks.config import CALIBRATIONS, CIFAR_FAMILY, Data, Network, OPT_FAMILY, OptNetwork, PRIMARY_METRICS, RecoveryViews, TASK_PRIMARY_METRICS, Task, TextData, TransferTests, _data, _network, _recovery_views, _task, _text_data, _transfer_tests
from core.execution import Execution, parse_execution


SCHEMA = "crossbar_lifecycle"


SCHEMA_VERSION = 1


EXPERIMENT_ID = "crossbar_lifecycle.v1"


PROVENANCE_SCHEMA = "crossbar_lifecycle.provenance.v1"


STAGE_KINDS = ("prepare", "devices", "hwa", "deploy", "onchip", "retention", "compensate", "drift_recovery", "initial_programming")


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
OPTIONAL_SECTIONS = ("transfer_tests", "recovery_views", "retention", "drift_compensation", "task", "drift_recovery", "initial_programming")


LIFECYCLE_KEYS = ("schema", "schema_version", "lifecycle_id") + SECTIONS


# Upstream artifacts are reusable exactly when the sections they depend on are
# unchanged; the question text, unrelated arms and later stages may change.
ARTIFACT_SECTIONS = {
    "feature_cache": ("network", "data", "transfer_tests", "recovery_views", "task"),
    "device_bundle": ("network", "devices", "program_verify"),
    "hwa_source": ("network", "data", "devices", "deployment", "program_verify"),
    "deployment": ("network", "data", "devices", "defects", "deployment", "program_verify", "transfer_tests", "task"),
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
    recovery_views: Optional[RecoveryViews] = None
    retention: Optional[Retention] = None
    drift_compensation: Optional[DriftCompensation] = None
    task: Optional[Task] = None
    drift_recovery: object = None
    initial_programming: object = None

    @property
    def onchip_cases(self):
        labels = None if self.onchip is None else self.onchip.case_labels
        return tuple(c for c in self.defects.cases if labels is None or c.label in labels)

    @property
    def recovery_examples(self):
        return (self.recovery_views.sequences if self.recovery_views is not None
                else self.data.cohort_size(self.onchip.cohort_key))

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
    initial = lifecycle.initial_programming
    if initial is not None:
        if (family != OPT_FAMILY or devices.technology != 'cmo' or lifecycle.onchip is not None
                or any(getattr(lifecycle, k) is not None for k in
                       ('retention', 'drift_compensation', 'drift_recovery', 'recovery_views', 'task', 'transfer_tests'))
                or network.analog_lora is not None or lifecycle.deployment.read_noise != 0
                or any(a.method == 'lora' for a in lifecycle.hwa.arms)):
            raise config_error('initial_programming', 'to use the frozen OPT pair mapping without other branches/read noise', initial.to_dict())
        if not set(initial.assignment_seeds) <= set(devices.assignment_seeds):
            raise config_error('initial_programming.assignment_seeds', 'to select declared assignments', initial.assignment_seeds)
        if len(lifecycle.defects.cases) != 1 or lifecycle.defects.cases[0].kind != 'none':
            raise config_error('initial_programming', 'to use only the nominal fault-free case', lifecycle.defects.to_dict())
        if initial.validation_sequences > data.development_sequences:
            raise config_error('initial_programming.validation_sequences', 'to select a development prefix', initial.validation_sequences)
    custom = lifecycle.drift_recovery
    if custom is not None:
        if (family != OPT_FAMILY or devices.technology != 'cmo' or lifecycle.onchip is not None
                or lifecycle.retention is not None or lifecycle.drift_compensation is not None
                or lifecycle.recovery_views is not None or lifecycle.task is not None
                or network.analog_lora is not None or lifecycle.deployment.read_noise != 0):
            raise config_error('drift_recovery', 'to use the existing OPT pair encoding without other recovery branches or read noise', custom.to_dict())
        if (custom.law.g_min, custom.law.g_max) != (9., 88.2):
            raise config_error('drift_recovery.law', 'to preserve the source mapping window', custom.to_dict())
        if not set(custom.assignment_seeds) <= set(devices.assignment_seeds):
            raise config_error('drift_recovery.assignment_seeds', 'to select declared assignments', custom.assignment_seeds)
        if any(c.kind != 'none' for c in lifecycle.defects.cases):
            raise config_error('drift_recovery', 'to use a nominal fault-free case', lifecycle.defects.to_dict())
    retention = lifecycle.retention
    if lifecycle.drift_compensation is not None:
        if retention is None or retention.read_noise:
            raise config_error("drift_compensation", "to use held retention snapshots with read_noise false", retention)
        compensation = lifecycle.drift_compensation
        if retention.power_law is None:
            if compensation.seconds is not None or set(compensation.methods) & set(OUTPUT_CALIBRATIONS):
                raise config_error("drift_compensation", "to use age subsets and MLP-output methods only with the scalar power law",
                                   compensation.to_dict())
        elif (compensation.seconds is None or not set(compensation.seconds) <= set(retention.seconds)
              or not set(compensation.sigmas_us) <= set(retention.power_law.sigmas_us)):
            raise config_error("drift_compensation", "to select ages and spreads from the scalar power-law grid",
                               compensation.to_dict())
        if lifecycle.drift_compensation.calibration_sequences > lifecycle.data.cohort_size("training"):
            raise config_error("drift_compensation.calibration_sequences", "to fit within the training cohort", lifecycle.drift_compensation.calibration_sequences)
    if retention is not None:
        if devices.technology != "cmo" or family != OPT_FAMILY or lifecycle.onchip is not None:
            raise config_error("retention", "to remap OPT CMO sources with onchip: null", retention.to_dict())
        if lifecycle.recovery_views is not None or lifecycle.program_verify.relaxation.model != "none":
            raise config_error("retention", "to use unrelaxed sources without recovery views", retention.to_dict())
        if devices.noise_scale != 1 or lifecycle.deployment.read_noise != 0:
            raise config_error("retention", "to use nominal programming noise and zero per-MVM read noise", retention.to_dict())
        if not set(retention.assignment_seeds) <= set(devices.assignment_seeds):
            raise config_error("retention.assignment_seeds", "to select declared assignments", retention.assignment_seeds)
        if not set(retention.case_labels) <= {c.label for c in lifecycle.defects.cases}:
            raise config_error("retention.case_labels", "to select declared defect cases", retention.case_labels)
        used = set(devices.assignment_seeds) | set(devices.selection_seeds) | {lifecycle.hwa.seed}
        used |= {devices.endpoint_seed(s) for s in devices.assignment_seeds + devices.selection_seeds}
        if retention.seed in used:
            raise config_error("retention.seed", "to be disjoint from identity, programming and HWA seeds", retention.seed)
    if lifecycle.onchip and lifecycle.onchip.microbatch_size is not None:
        if family != OPT_FAMILY or lifecycle.onchip.batch_size % lifecycle.onchip.microbatch_size:
            raise config_error("onchip.microbatch_size", "to divide the OPT effective batch", lifecycle.onchip.microbatch_size)
    if lifecycle.onchip and lifecycle.onchip.case_labels is not None:
        labels = lifecycle.onchip.case_labels
        if not labels or not set(labels) <= {c.label for c in lifecycle.defects.cases}:
            raise config_error("onchip.case_labels", "to select declared defect cases", labels)
    views = lifecycle.recovery_views
    if views is not None:
        if family != OPT_FAMILY or lifecycle.onchip is None or lifecycle.onchip.cohort_key != "training":
            raise config_error("recovery_views", "to accompany OPT recovery training", views.to_dict())
        if (views.parent_length % views.context_length or views.context_length % views.score_stride
                or views.parent_length > network.max_positions
                or any(length > network.max_positions for length in views.evaluation_lengths)):
            raise config_error("recovery_views", "to use divisible context/parent/score lengths within the model", views.to_dict())
        if data.max_examples:
            raise config_error("data.max_examples", "to be zero with explicit recovery views (use small declared cohorts for probes)", data.max_examples)
        chunks = views.parent_length // views.context_length
        if lifecycle.onchip.batch_size % chunks or views.sequences % lifecycle.onchip.batch_size:
            raise config_error("onchip.batch_size", "to group whole parents without a partial update", lifecycle.onchip.batch_size)
        counts = [data.development_sequences] + ([data.test_sequences] if data.evaluation == "test" else [])
        if lifecycle.transfer_tests:
            counts += [test.sequences for test in lifecycle.transfer_tests.tests]
        if any(count * data.sequence_length % length for count in counts for length in views.evaluation_lengths):
            raise config_error("recovery_views.evaluation_lengths", "to exactly divide each evaluation token prefix", views.evaluation_lengths)
    if lifecycle.onchip and any(a.digital_adapter for a in lifecycle.onchip.arms) and family != OPT_FAMILY:
        raise config_error("onchip.arms.digital_adapter", "to be absent outside opt_mlp_suffix", family)
    lora_arms = [arm for arm in lifecycle.hwa.arms if arm.method == "lora"]
    lora = getattr(network, "analog_lora", None)
    if lora is not None and (retention is not None or lifecycle.recovery_views is not None or (
            lifecycle.onchip and any(a.digital_adapter for a in lifecycle.onchip.arms))):
        raise config_error("network.analog_lora", "to be absent with retention, recovery views or digital adapters", lora.to_dict())
    if lora is not None and lifecycle.deployment.scaling != "per_output_channel_absmax":
        raise config_error("network.analog_lora", "to use per_output_channel_absmax scaling", lifecycle.deployment.scaling)
    if any(case.target != "all" for case in lifecycle.defects.cases) and lora is None:
        raise config_error("defects.cases.target", "to select base or adapter arrays only with network.analog_lora", "target")
    if lora_arms:
        task_examples = None if lifecycle.task is None else lifecycle.task.training_examples
        if (lora is None or task_examples is None or lifecycle.hwa.objective != TASK_HWA_OBJECTIVE
                or lifecycle.hwa.lr_schedule is not None or lifecycle.hwa.sequences_per_epoch is not None
                or len(lora_arms) != sum(arm.trains for arm in lifecycle.hwa.arms)):
            raise config_error("hwa.arms", "to train LoRA arms only with network.analog_lora, task.training_examples, "
                               f"objective {TASK_HWA_OBJECTIVE!r}, no lr_schedule or streaming, and no other trained arm",
                               [arm.arm_id for arm in lora_arms])
    elif lifecycle.hwa.objective == TASK_HWA_OBJECTIVE:
        raise config_error("hwa.objective", "to be the task objective only for LoRA arms", lifecycle.hwa.objective)
    if lifecycle.onchip is not None:
        if lifecycle.onchip.objective == SOURCE_KL and family != OPT_FAMILY:
            raise config_error("onchip.objective", f"to be {SOURCE_KL!r} only for opt_mlp_suffix", SOURCE_KL)
        if any(arm.write_scope == "adapter" for arm in lifecycle.onchip.arms) and lora is None:
            raise config_error("onchip.arms.write_scope", "to learn adapters only with network.analog_lora", "adapter")
    task = lifecycle.task
    if task is not None:
        if family != OPT_FAMILY:
            raise config_error("task", f"to be absent; {family} defines no labelled task", task.to_dict())
        if retention is not None or lifecycle.recovery_views is not None:
            raise config_error("task", "to be absent with retention, compensation or recovery views", task.to_dict())
    if lifecycle.question.primary_metric in TASK_PRIMARY_METRICS and task is None:
        raise config_error("question.primary_metric", "to name a task metric only with a declared task",
                           lifecycle.question.primary_metric)
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
    rewrites = {arm.fault_rewrite.seed for arm in (lifecycle.onchip.arms if lifecycle.onchip else ())
                if arm.fault_rewrite is not None}
    if rewrites & (identities | endpoints | {lifecycle.onchip.seed if lifecycle.onchip else None}):
        raise config_error(
            "onchip.arms.fault_rewrite.seed",
            "to be separate from every array identity, deployment endpoint and the on-chip seed",
            sorted(rewrites & (identities | endpoints | {lifecycle.onchip.seed})),
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
    custom = None
    initial = None
    if 'initial_programming' in raw:
        from methods.programming.config import InitialProgramming
        try:
            initial = InitialProgramming.parse(raw['initial_programming'])
        except ValueError as error:
            raise config_error('initial_programming', str(error), raw['initial_programming']) from error
    if 'drift_recovery' in raw:
        from methods.maintenance.config import DriftRecovery
        try:
            custom = DriftRecovery.parse(raw['drift_recovery'])
        except ValueError as error:
            raise config_error('drift_recovery', str(error), raw['drift_recovery']) from error
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
        recovery_views=_recovery_views(raw["recovery_views"]) if "recovery_views" in raw else None,
        retention=_retention(raw["retention"]) if "retention" in raw else None,
        drift_compensation=_drift_compensation(raw["drift_compensation"]) if "drift_compensation" in raw else None,
        task=_task(raw["task"]) if "task" in raw else None,
        drift_recovery=custom,
        initial_programming=initial,
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
    if kind in ("retention", "compensate") and (lifecycle.retention is None or seed not in lifecycle.retention.assignment_seeds):
        raise config_error("stage", "to select a declared retention assignment", raw)
    if kind == "compensate" and lifecycle.drift_compensation is None:
        raise config_error("stage", "to declare drift_compensation for a compensate stage", raw)
    if kind == 'drift_recovery' and (lifecycle.drift_recovery is None or seed not in lifecycle.drift_recovery.assignment_seeds):
        raise config_error('stage', 'to select a declared custom-recovery assignment', raw)
    if kind == 'initial_programming' and (lifecycle.initial_programming is None or seed not in lifecycle.initial_programming.assignment_seeds):
        raise config_error('stage', 'to select a declared initial-programming realization', raw)
    return StageSelection(kind, arm, seed)


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
    "retention": ("retention_measurements", "retention_states"),
    "compensate": ("compensation_measurements", "compensation_states"),
    "drift_recovery": ("drift_recovery_measurements", "drift_recovery_trajectories"),
    "initial_programming": ("initial_programming_measurements", "initial_programming_summary", "initial_programming_state", "flowers_model"),
}


def stage_selections(lifecycle: Lifecycle) -> list[StageSelection]:
    """All stages of a lifecycle in dependency order."""

    stages = [StageSelection("prepare"), StageSelection("devices")]
    stages += [StageSelection("hwa", arm.arm_id) for arm in lifecycle.hwa.arms]
    if lifecycle.initial_programming is not None:
        return stages + [StageSelection('initial_programming', arm.arm_id, seed)
                         for arm in lifecycle.hwa.arms for seed in lifecycle.initial_programming.assignment_seeds]
    if lifecycle.drift_recovery is not None:
        return stages + [StageSelection('drift_recovery', arm.arm_id, seed)
                         for arm in lifecycle.hwa.arms for seed in lifecycle.drift_recovery.assignment_seeds]
    if lifecycle.retention is not None:
        kind = "compensate" if lifecycle.drift_compensation is not None else "retention"
        return stages + [StageSelection(kind, arm.arm_id, seed)
                         for arm in lifecycle.hwa.arms for seed in lifecycle.retention.assignment_seeds]
    for kind in ("deploy", "onchip") if lifecycle.onchip is not None else ("deploy",):
        stages += [
            StageSelection(kind, arm.arm_id, seed)
            for arm in lifecycle.hwa.arms
            for seed in lifecycle.devices.assignment_seeds
        ]
    return stages


BASE_SOURCE = ("external", "lora_base_source")


def stage_inputs(selection: StageSelection, lifecycle: Optional["Lifecycle"] = None) -> dict[str, Optional[tuple[str, str]]]:
    """CLI input roles; ``None`` marks the external pinned teacher checkpoint.

    ``BASE_SOURCE`` marks the frozen HWA base of a LoRA arm, an external artifact pinned by the arm.
    """

    inputs: dict[str, Optional[tuple[str, str]]] = {"teacher_weights": None}
    if selection.kind in ("prepare", "devices"):
        return inputs
    inputs["device_data"] = ("prepare", "feature_cache")
    inputs["device_model"] = ("devices", "device_bundle")
    if (selection.kind == "hwa" and lifecycle is not None
            and lifecycle.hwa.arm(selection.hwa_arm).method == "lora"):
        inputs["weights"] = BASE_SOURCE
    if selection.kind in ("deploy", "retention", "compensate", "drift_recovery", "initial_programming"):
        inputs["weights"] = (StageSelection("hwa", selection.hwa_arm).stage_id, "hwa_source")
        if selection.kind == "compensate":
            inputs["device_state"] = (StageSelection("retention", selection.hwa_arm, selection.array_seed).stage_id, "retention_states")
    elif selection.kind == "onchip":
        upstream = StageSelection("deploy", selection.hwa_arm, selection.array_seed).stage_id
        inputs["device_state"] = (upstream, "deployment")
    return inputs


def provenance(lifecycle: Lifecycle, kind: str, *, hwa_arm: Optional[str] = None) -> dict:
    """Content identity of the lifecycle sections that determined an artifact."""

    if kind not in ARTIFACT_SECTIONS:
        raise ValueError(f"Expected a lifecycle artifact kind. Provided value: {kind!r}.")
    sections = ARTIFACT_SECTIONS[kind]
    if kind == "hwa_source" and any(arm.method == "lora" for arm in lifecycle.hwa.arms):
        sections = sections + ("task",)  # LoRA adapters train on the task's labelled examples
    record = {
        "schema": PROVENANCE_SCHEMA,
        "kind": kind,
        "lifecycle_id": lifecycle.lifecycle_id,
        "lifecycle_sha256": lifecycle.digest(),
        "sections": {
            name: lifecycle.section_digest(name)
            for name in sections
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
    if kind == "feature_cache" and record["sections"].get("recovery_views") != expected["sections"].get("recovery_views"):
        changed = sorted(set(changed) | {"recovery_views"})
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
