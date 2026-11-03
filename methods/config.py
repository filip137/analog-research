"""methods.config."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional
from core.contracts import config_error
from core.config import MAX_SEED, _LIFECYCLE_ID, _SHA256, _boolean, _choice, _identifier, _integer, _list, _number, _object
from hardware.config import CMO_ACCEPTANCE_PERCENT, DefectCase, HIERARCHICAL, TECHNOLOGIES, Technology, _cases
from networks.config import OptTeacher
from hardware.config import ADDITIVE_GAUSSIAN


NOISE_SCALINGS = ("matrix_full_scale", "output_channel_abs_max")


OBJECTIVES = ("teacher_kl", "cross_entropy")


# HWA may also train analog LoRA adapters on a labelled task: multiple-choice cross entropy
# over per-token mean continuation log-likelihoods (networks.tasks).
TASK_HWA_OBJECTIVE = "task_choice_token_mean"


HWA_OBJECTIVES = OBJECTIVES + (TASK_HWA_OBJECTIVE,)


# On-chip recovery may also target the clean deployed source itself (KL to the intended weights).
SOURCE_KL = "source_kl"


ONCHIP_OBJECTIVES = OBJECTIVES + (SOURCE_KL,)


@dataclass(frozen=True)
class Corruption:
    """Temporary training masks; never the deployed fault map."""

    kind: str
    rates_ppm: tuple[int, ...]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "rates_ppm": list(self.rates_ppm)}


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
    # ``method: lora``: the frozen HWA base source, and the adapters' Adam rate and warm-up.
    base_source: Optional[OptTeacher] = None
    learning_rate: Optional[float] = None
    warmup_fraction: Optional[float] = None

    @property
    def trains(self) -> bool:
        return self.method in ("hwa", "lora")

    def to_dict(self) -> dict:
        if not self.trains:
            return {"id": self.arm_id, "method": self.method}
        if self.method == "lora":
            return {"id": self.arm_id, "method": self.method, "noise_strength": self.noise_strength,
                    "selection_cases": [case.to_dict() for case in self.selection_cases],
                    "base_source": self.base_source.to_dict(), "learning_rate": self.learning_rate,
                    "warmup_fraction": self.warmup_fraction}
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
    method = _choice(raw.get("method"), f"{path}.method", ("none", "hwa", "lora"))
    if method == "none":
        raw = _object(raw, path, ("id", "method"))
        return HwaArm(_identifier(raw["id"], f"{path}.id"), method)
    if method == "lora":
        raw = _object(raw, path, ("id", "method", "noise_strength", "selection_cases", "base_source", "learning_rate",
                                  "warmup_fraction"))
        base = _object(raw["base_source"], f"{path}.base_source", ("name", "sha256"))
        if not isinstance(base["sha256"], str) or not _SHA256.match(base["sha256"]):
            raise config_error(f"{path}.base_source.sha256", "to be a lowercase SHA-256 digest", base["sha256"])
        warmup = _number(raw["warmup_fraction"], f"{path}.warmup_fraction")
        if not 0 <= warmup < 1:
            raise config_error(f"{path}.warmup_fraction", "to be in [0, 1)", warmup)
        return HwaArm(
            arm_id=_identifier(raw["id"], f"{path}.id"), method=method,
            noise_strength=_number(raw["noise_strength"], f"{path}.noise_strength"),
            selection_cases=_cases(raw["selection_cases"], f"{path}.selection_cases", technology.failure_kinds,
                                   technology.defect_placement == HIERARCHICAL),
            # The frozen base is a whole-model source; only the adapters train.
            scope="model", digital_learning_rate=0.0,
            base_source=OptTeacher(_identifier(base["name"], f"{path}.base_source.name", _LIFECYCLE_ID), base["sha256"]),
            learning_rate=_number(raw["learning_rate"], f"{path}.learning_rate", positive=True),
            warmup_fraction=warmup,
        )
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
        objective=_choice(raw["objective"], "hwa.objective", HWA_OBJECTIVES),
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
class FaultRewrite:
    """One-shot CMO fault-aware rewrite at the start of an arm (docs/reram_cmo_device.md).

    ``detection``: ``verify`` flags devices from controller-visible verify reads
    beyond ``threshold_sigma`` programming sigmas of their targets; ``oracle``
    uses the hidden fault map (non-physical). ``roles``: ``active`` lets a
    healthy active device absorb a stuck reference; ``pair`` also lets a healthy
    reference absorb a stuck active device. ``seed`` names the programming
    stream, separate from every deployment stream.
    """

    detection: str
    roles: str
    seed: int
    threshold_sigma: Optional[float] = None

    def to_dict(self) -> dict:
        value = {"detection": self.detection, "roles": self.roles, "seed": self.seed}
        if self.threshold_sigma is not None:
            value["threshold_sigma"] = self.threshold_sigma
        return value


@dataclass(frozen=True)
class OnchipArm:
    arm_id: str
    weights: str
    calibration: bool
    digital_adapter: Optional[DigitalLora] = None
    fault_rewrite: Optional[FaultRewrite] = None
    write_roles: Optional[str] = None
    # Analog LoRA only: learn only the adapter arrays (``adapter``); the default writes every array.
    write_scope: str = "all"
    # Optional weight learning rate of this arm; the shared ``onchip.learning_rate`` otherwise.
    learning_rate: Optional[float] = None

    @property
    def trains(self) -> bool:
        return self.weights != "hold" or self.calibration or self.digital_adapter is not None

    @property
    def setting(self) -> tuple:
        return (self.weights, self.calibration, self.digital_adapter, self.fault_rewrite, self.write_roles,
                self.write_scope, self.learning_rate)

    def to_dict(self) -> dict:
        value = {"id": self.arm_id, "weights": self.weights, "calibration": self.calibration}
        if self.write_scope != "all":
            value["write_scope"] = self.write_scope
        if self.learning_rate is not None:
            value["learning_rate"] = self.learning_rate
        if self.digital_adapter is not None:
            value["digital_adapter"] = self.digital_adapter.to_dict()
        if self.fault_rewrite is not None:
            value["fault_rewrite"] = self.fault_rewrite.to_dict()
        if self.write_roles is not None:
            value["write_roles"] = self.write_roles
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
    threshold_sigma: Optional[float] = None
    accumulate_updates: Optional[bool] = None
    development_every_steps: Optional[int] = None
    max_write_attempts_per_device: Optional[int] = None
    case_labels: Optional[tuple[str, ...]] = None
    microbatch_size: Optional[int] = None
    write_noise_scale: Optional[float] = None
    # ``development``: intermediate epochs score only the development cohorts; the final epoch is complete.
    epoch_evaluation: Optional[str] = None

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
        if self.update_law == "cmo_thresholded_write_adam":
            value.update(threshold_sigma=self.threshold_sigma, accumulate_updates=self.accumulate_updates)
            if self.max_write_attempts_per_device is not None:
                value["max_write_attempts_per_device"] = self.max_write_attempts_per_device
        if self.development_every_steps is not None:
            value["development_every_steps"] = self.development_every_steps
        if self.case_labels is not None:
            value["case_labels"] = list(self.case_labels)
        if self.microbatch_size is not None:
            value["microbatch_size"] = self.microbatch_size
        if self.write_noise_scale is not None:
            value["write_noise_scale"] = self.write_noise_scale
        if self.epoch_evaluation is not None:
            value["epoch_evaluation"] = self.epoch_evaluation
        value["arms"] = [arm.to_dict() for arm in self.arms]
        return value


def _fault_rewrite(raw: Any, path: str) -> FaultRewrite:
    detection = _choice(raw.get("detection") if isinstance(raw, Mapping) else None, f"{path}.detection",
                        FAULT_REWRITE_DETECTIONS)
    # Only measured detection has a threshold; the oracle reads the fault map itself.
    keys = ("detection", "roles", "seed") + (("threshold_sigma",) if detection == "verify" else ())
    raw = _object(raw, path, keys)
    return FaultRewrite(
        detection=detection,
        roles=_choice(raw["roles"], f"{path}.roles", FAULT_REWRITE_ROLES),
        seed=_integer(raw["seed"], f"{path}.seed", minimum=1, maximum=MAX_SEED),
        threshold_sigma=(_number(raw["threshold_sigma"], f"{path}.threshold_sigma", positive=True)
                         if detection == "verify" else None),
    )


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
    elif law == "cmo_thresholded_write_adam":
        keys += ("threshold_sigma", "accumulate_updates")
        if "max_write_attempts_per_device" in raw:
            keys += ("max_write_attempts_per_device",)
    if "training_cohort" in raw:
        keys += ("training_cohort",)
    if "development_every_steps" in raw:
        keys += ("development_every_steps",)
    if "case_labels" in raw:
        keys += ("case_labels",)
    if "microbatch_size" in raw:
        keys += ("microbatch_size",)
    if "epoch_evaluation" in raw:
        keys += ("epoch_evaluation",)
    if "write_noise_scale" in raw and law.startswith("cmo_"):
        keys += ("write_noise_scale",)
    raw = _object(raw, "onchip", keys)
    arms = []
    for index, item in enumerate(_list(raw["arms"], "onchip.arms")):
        path = f"onchip.arms[{index}]"
        keys = ("id", "weights", "calibration") + tuple(
            name for name in ("digital_adapter", "fault_rewrite", "write_roles", "write_scope", "learning_rate") if isinstance(item, Mapping) and name in item
        )
        item = _object(item, path, keys)
        rewrite = None
        if "fault_rewrite" in item:
            rewrite = _fault_rewrite(item["fault_rewrite"], f"{path}.fault_rewrite")
            if technology != "cmo":
                raise config_error(f"{path}.fault_rewrite", "to be declared only for CMO arrays", technology)
            if item["weights"] not in ("hold", "learn") or "digital_adapter" in item:
                raise config_error(
                    path, "to rewrite faults before holding or learning weights, without a digital adapter", item
                )
        write_roles = None
        if "write_roles" in item:
            write_roles = _choice(item["write_roles"], f"{path}.write_roles", ("verified_partner",))
            if (technology != "cmo" or law != "cmo_thresholded_write_adam" or item["weights"] != "learn"
                    or rewrite is None or rewrite.detection != "verify" or rewrite.roles != "pair"):
                raise config_error(path, "to use verified_partner only for thresholded CMO learning after a verify pair rewrite", item)
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
        scope = _choice(item["write_scope"], f"{path}.write_scope", ("all", "adapter")) if "write_scope" in item else "all"
        rate = _number(item["learning_rate"], f"{path}.learning_rate", positive=True) if "learning_rate" in item else None
        if (scope != "all" or rate is not None) and (item["weights"] != "learn" or law != "cmo_thresholded_write_adam"):
            raise config_error(path, "to set write_scope or learning_rate only for thresholded CMO weight learning", item)
        arms.append(
            OnchipArm(
                arm_id=_identifier(item["id"], f"{path}.id"),
                weights=_choice(item["weights"], f"{path}.weights", ("hold", "rewrite", "learn")),
                calibration=_boolean(item["calibration"], f"{path}.calibration"),
                digital_adapter=adapter,
                fault_rewrite=rewrite,
                write_roles=write_roles,
                write_scope=scope,
                learning_rate=rate,
            )
        )
    arms = tuple(arms)
    if len({arm.arm_id for arm in arms}) != len(arms):
        raise config_error("onchip.arms", "to have unique ids", [arm.arm_id for arm in arms])
    settings = {arm.setting for arm in arms}
    if len(settings) != len(arms):
        raise config_error("onchip.arms", "to have unique weight/calibration settings", [a.to_dict() for a in arms])
    for arm in arms:
        # A weight-writing arm is interpretable only beside its matched
        # no-weight-write control with the same digital calibration.
        writes = arm.weights != "hold" or arm.digital_adapter is not None or arm.fault_rewrite is not None
        if writes and ("hold", arm.calibration, None, None, None, "all", None) not in settings:
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
    if "microbatch_size" in raw:
        extra["microbatch_size"] = _integer(raw["microbatch_size"], "onchip.microbatch_size", minimum=1)
    if "case_labels" in raw:
        labels = tuple(_list(raw["case_labels"], "onchip.case_labels"))
        if any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels):
            raise config_error("onchip.case_labels", "to contain unique case labels", labels)
        extra["case_labels"] = labels
    if law == "cmo_thresholded_write_adam":
        extra.update(
            threshold_sigma=_number(raw["threshold_sigma"], "onchip.threshold_sigma", positive=True),
            accumulate_updates=_boolean(raw["accumulate_updates"], "onchip.accumulate_updates"),
        )
        cap = raw.get("max_write_attempts_per_device")
        if cap is not None:
            extra["max_write_attempts_per_device"] = _integer(
                cap, "onchip.max_write_attempts_per_device", minimum=0, maximum=2**31 - 1
            )
            if any(arm.weights == "rewrite" for arm in arms):
                raise config_error("onchip.arms", "to omit rewrite under a thresholded write-attempt cap", raw["arms"])
            # A one-shot fault rewrite is a programming event before learning,
            # like deployment, so the write-attempt cap does not count it.
    if "write_noise_scale" in raw:
        # Scales on-chip write noise only. CMO rewrite arms reprogram with
        # physical programming noise, so they would not be matched controls.
        extra["write_noise_scale"] = _number(raw["write_noise_scale"], "onchip.write_noise_scale")
        if any(arm.weights == "rewrite" for arm in arms):
            raise config_error("onchip.arms", "to omit rewrite under a declared write_noise_scale", raw["arms"])
        # Fault rewrites program with physical noise too; the exact-write oracle
        # is a separate question, so the two are never mixed in one lifecycle.
        if any(arm.fault_rewrite is not None for arm in arms):
            raise config_error("onchip.arms", "to omit fault_rewrite under a declared write_noise_scale", raw["arms"])
    if "epoch_evaluation" in raw:
        extra["epoch_evaluation"] = _choice(raw["epoch_evaluation"], "onchip.epoch_evaluation", ("development",))
    if "development_every_steps" in raw:
        extra["development_every_steps"] = _integer(
            raw["development_every_steps"], "onchip.development_every_steps", minimum=1
        )
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
        objective=_choice(raw["objective"], "onchip.objective", ONCHIP_OBJECTIVES),
        learning_rate=_number(raw["learning_rate"], "onchip.learning_rate"),
        calibration_learning_rate=_number(raw["calibration_learning_rate"], "onchip.calibration_learning_rate"),
        epochs=_integer(raw["epochs"], "onchip.epochs", minimum=1),
        batch_size=_integer(raw["batch_size"], "onchip.batch_size", minimum=1),
        seed=_integer(raw["seed"], "onchip.seed", maximum=MAX_SEED),
        arms=arms,
        **extra,
    )


MATRIX_COMPENSATIONS = ("global", "tile_global", "row", "column", "row_column", "global_shift", "global_moments")


OUTPUT_CALIBRATIONS = ("mlp_gain", "mlp_shift", "mlp_moments")


@dataclass(frozen=True)
class DriftCompensation:
    methods: tuple[str, ...]
    calibration_sequences: int
    row_column_iterations: int
    gain_bounds: tuple[float, float]
    # Scalar power-law parents only: the calibrated subset of their (age, spread) grid.
    seconds: Optional[tuple[float, ...]] = None
    sigmas_us: Optional[tuple[float, ...]] = None

    def to_dict(self):
        value = {"methods": list(self.methods), "calibration_sequences": self.calibration_sequences,
                 "row_column_iterations": self.row_column_iterations, "gain_bounds": list(self.gain_bounds)}
        if self.seconds is not None:
            value.update(seconds=list(self.seconds), sigmas_us=list(self.sigmas_us))
        return value

    def snapshots(self, retention, model):
        if self.seconds is None:
            return retention.snapshots(model)
        return [(seconds, sigma) for seconds in self.seconds for sigma in self.sigmas_us]


def _drift_compensation(raw):
    keys = ("methods", "calibration_sequences", "row_column_iterations", "gain_bounds")
    subset = "seconds" in raw or "sigmas_us" in raw if isinstance(raw, Mapping) else False
    raw = _object(raw, "drift_compensation", keys + (("seconds", "sigmas_us") if subset else ()))
    methods = tuple(_choice(v, "drift_compensation.methods", MATRIX_COMPENSATIONS + OUTPUT_CALIBRATIONS)
                    for v in _list(raw["methods"], "drift_compensation.methods"))
    bounds = tuple(_number(v, "drift_compensation.gain_bounds", positive=True)
                   for v in _list(raw["gain_bounds"], "drift_compensation.gain_bounds"))
    if len(set(methods)) != len(methods) or len(bounds) != 2 or not bounds[0] <= 1 <= bounds[1]:
        raise config_error("drift_compensation", "to use unique methods and bounds containing one", raw)
    seconds = sigmas = None
    if subset:
        seconds = tuple(_number(v, "drift_compensation.seconds", positive=True)
                        for v in _list(raw["seconds"], "drift_compensation.seconds"))
        sigmas = tuple(_number(v, "drift_compensation.sigmas_us") for v in _list(raw["sigmas_us"], "drift_compensation.sigmas_us"))
        if list(seconds) != sorted(set(seconds)) or list(sigmas) != sorted(set(sigmas)):
            raise config_error("drift_compensation", "to list increasing distinct ages and spreads", raw)
    return DriftCompensation(methods, _integer(raw["calibration_sequences"], "drift_compensation.calibration_sequences", minimum=1),
                             _integer(raw["row_column_iterations"], "drift_compensation.row_column_iterations", minimum=1), bounds,
                             seconds, sigmas)


# What an HWA arm trains: ``analog`` the analog weights and their calibration
# parameters (the frozen teacher supplies every other digital parameter);
# ``model`` also every digital parameter, as in Rasch et al. 2023 and Büchel et
# al. 2025 (OPT family only).
HWA_SCOPES = ("analog", "model")


FAULT_REWRITE_DETECTIONS = ("oracle", "verify")


FAULT_REWRITE_ROLES = ("active", "pair")
