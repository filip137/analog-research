"""hardware.config — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional
from core.contracts import config_error
from core.config import MAX_SEED, _boolean, _choice, _integer, _list, _number, _object, _seeds, _text

ADDITIVE_GAUSSIAN = "additive_gaussian_weight"


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


DEFECT_TARGETS = ("all", "base", "adapter")


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
        update_laws=("cmo_idealized_write_adam", "cmo_thresholded_write_adam"),
    ),
}


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
    # Analog LoRA only: stuck devices in every array (all), only the base MLP arrays or only the adapters.
    target: str = "all"

    @property
    def label(self) -> str:
        if self.kind == "none":
            return "nominal"
        label = f"{self.kind}_{self.rate_ppm}ppm"
        for prefix, value in (("d", self.die_concentration), ("a", self.array_concentration)):
            if value is not None:
                label += f"_{prefix}{value:g}"
        return label if self.target == "all" else f"{label}_{self.target}"

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
        if self.target != "all":
            value["target"] = self.target
        return value


def _case(raw: Any, path: str, failure_kinds, hierarchical: bool = False) -> DefectCase:
    concentrations = ("die_concentration", "array_concentration")
    targeted = hierarchical and isinstance(raw, Mapping) and "target" in raw
    raw = _object(raw, path, ("kind", "rate_ppm") + (concentrations if hierarchical else ()) + (("target",) if targeted else ()))
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
    target = _choice(raw["target"], f"{path}.target", DEFECT_TARGETS) if targeted else "all"
    if kind == "none" and target != "all":
        raise config_error(path, "to target arrays only with stuck devices", dict(raw))
    return DefectCase(kind, rate, hierarchical=True, target=target, **values)


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
class Retention:
    """Inference-only remapping of frozen CMO HWA sources into fitted device models."""

    models: tuple[str, ...]
    seconds: tuple[float, ...]
    assignment_seeds: tuple[int, ...]
    case_labels: tuple[str, ...]
    seed: int
    read_noise: bool
    wan_noise_scales: tuple[float, ...] = (1.0,)
    power_law: object = None

    def to_dict(self):
        value = {"models": list(self.models), "seconds": list(self.seconds),
                 "assignment_seeds": list(self.assignment_seeds), "case_labels": list(self.case_labels),
                 "seed": self.seed, "read_noise": self.read_noise}
        if self.wan_noise_scales != (1.0,):
            value["wan_noise_scales"] = list(self.wan_noise_scales)
        if self.power_law is not None:
            value["power_law"] = self.power_law.to_dict()
        return value

    def snapshots(self, model):
        """Aged (seconds, noise scale) pairs; Wan's 1 s programming fit is never scaled."""
        if self.power_law is not None:
            return [(seconds, sigma) for seconds in self.seconds for sigma in self.power_law.sigmas_us]
        return [(seconds, scale) for seconds in self.seconds
                for scale in (self.wan_noise_scales if model == "reram_wan2022" and seconds != 1 else (1.0,))]


def _retention(raw):
    keys = ("models", "seconds", "assignment_seeds", "case_labels", "seed", "read_noise")
    raw = _object(raw, "retention", keys + tuple(k for k in ("wan_noise_scales", "power_law") if k in raw))
    models = tuple(_choice(v, "retention.models", ("pcm", "reram_cmo", "reram_wan2022"))
                   for v in _list(raw["models"], "retention.models"))
    seconds = tuple(_number(v, "retention.seconds", positive=True)
                    for v in _list(raw["seconds"], "retention.seconds"))
    labels = tuple(_text(v, "retention.case_labels") for v in _list(raw["case_labels"], "retention.case_labels"))
    if len(set(models)) != len(models) or len(set(labels)) != len(labels):
        raise config_error("retention", "to use unique models and case labels", raw)
    if tuple(sorted(set(seconds))) != seconds:
        raise config_error("retention.seconds", "to be unique and increasing", seconds)
    if "reram_wan2022" in models and not set(seconds) <= {1., 86400., 172800.}:
        raise config_error("retention.seconds", "to use only 1, 86400 or 172800 seconds for Wan-2022", seconds)
    if "reram_cmo" in models and min(seconds) < 1:
        raise config_error("retention.seconds", "to be at least one second for CMO", seconds)
    scales = (1.0,)
    if "wan_noise_scales" in raw:
        scales = tuple(float(_number(v, "retention.wan_noise_scales", positive=True))
                       for v in _list(raw["wan_noise_scales"], "retention.wan_noise_scales"))
        if (1.0 not in scales or tuple(sorted(set(scales))) != scales
                or any(not math.isfinite(v) for v in scales)):
            raise config_error("retention.wan_noise_scales", "to be finite, unique, increasing and include 1", scales)
        if "reram_wan2022" not in models or not set(seconds) - {1.}:
            raise config_error("retention.wan_noise_scales", "to scale aged Wan-2022 snapshots only", raw)
    power_law = None
    if "power_law" in raw:
        from hardware.drift.power_law import PowerLaw
        try:
            power_law = PowerLaw.parse(raw["power_law"])
        except ValueError as error:
            raise config_error("retention.power_law", str(error), raw["power_law"]) from error
        if models != ("reram_cmo",) or labels != ("nominal",) or raw["read_noise"] or "wan_noise_scales" in raw:
            raise config_error("retention.power_law", "to use nominal CMO arrays without read noise or Wan scaling", raw)
    return Retention(models, seconds, _seeds(raw["assignment_seeds"], "retention.assignment_seeds"), labels,
                     _integer(raw["seed"], "retention.seed", minimum=1, maximum=MAX_SEED),
                     _boolean(raw["read_noise"], "retention.read_noise"), scales, power_law)


# HWA noise model: Gaussian weight noise of ``hwa.noise_std`` relative to the
# matrix full scale (1 after per-layer abs-max scaling) or to each output
# channel's current abs-max (the Analog Foundation Models convention).

