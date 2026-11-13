"""Strict, lightweight schema for bounded ideal circuit characterization."""
from dataclasses import dataclass, fields
import math
import re
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID = "analog_energy_transformer.v1"


@dataclass(frozen=True)
class Case:
    name: str
    K: tuple
    W: tuple
    a: tuple
    b: tuple
    c: tuple
    target: tuple


@dataclass(frozen=True)
class Config:
    cases: tuple
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    kappa: float = 1.7
    conductance: float = 1e-4
    visible_capacitance: float = 1e-7
    logit_capacitance: float = 1.5e-7
    relu_capacitance: float = 8e-8
    floor: float = 0.02
    betas: tuple = (0.04, 0.02, 0.01, 0.005)
    finite_difference_step: float = 1e-5
    learning_rate: float = 0.1
    wall_seconds: int = 600
    spice_timeout: int = 30
    spice_executable: str = "ngspice"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or any(not _finite(x) for x in value):
        raise ConfigError(f"Expected finite numeric vector for {name}")
    return tuple(float(x) for x in value)


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw)-{f.name for f in fields(Config)} or "cases" not in raw:
        raise ConfigError("Expected only declared analog_energy_transformer fields, including cases")
    data = dict(raw)
    if not isinstance(data["cases"], (list, tuple)) or not 1 <= len(data["cases"]) <= 3:
        raise ConfigError("Expected one to three bounded circuit cases")
    cases = []
    for case in data["cases"]:
        if not isinstance(case, dict) or set(case) != {f.name for f in fields(Case)}:
            raise ConfigError("Expected exactly the declared case fields")
        name = case["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", name):
            raise ConfigError("Expected a safe unique case name")
        vectors = {key: _vector(case[key], key) for key in ("a", "b", "c", "target")}
        d, l, m = (len(vectors[key]) for key in ("a", "b", "c"))
        if not 1 <= d <= 8 or not 2 <= l <= 8 or not 0 <= m <= 8 or len(vectors["target"]) != d:
            raise ConfigError("Invalid or out-of-budget circuit dimensions")
        matrices = {}
        for key, rows in (("K", l), ("W", m)):
            value = case[key]
            if not isinstance(value, (tuple, list)) or len(value) != rows:
                raise ConfigError(f"Wrong row count for {key}")
            matrices[key] = tuple(_vector(row, key) for row in value)
            if any(len(row) != d for row in matrices[key]):
                raise ConfigError(f"Wrong column count for {key}")
        cases.append(Case(name=name, **vectors, **matrices))
    if len({case.name for case in cases}) != len(cases):
        raise ConfigError("Duplicate circuit case name")
    data["cases"] = tuple(cases)
    if "betas" in data:
        data["betas"] = _vector(data["betas"], "betas")
    config = Config(**data)
    if config.experiment_id != EXPERIMENT_ID or type(config.schema_version) is not int or config.schema_version != 1:
        raise ConfigError("Expected analog_energy_transformer.v1 schema 1")
    for name in ("kappa", "conductance", "visible_capacitance", "logit_capacitance", "relu_capacitance", "floor", "finite_difference_step", "learning_rate"):
        value = getattr(config, name)
        if not _finite(value) or value <= 0:
            raise ConfigError(f"Expected positive finite {name}")
    if not 1 <= len(config.betas) <= 4 or any(not 0 < x <= 0.05 for x in config.betas) or len(set(config.betas)) != len(config.betas):
        raise ConfigError("Expected one to four distinct positive nudges <=0.05")
    for name, upper in (("wall_seconds", 600), ("spice_timeout", 30)):
        value = getattr(config, name)
        if type(value) is not int or not 1 <= value <= upper:
            raise ConfigError(f"Expected bounded positive integer {name}")
    if not isinstance(config.spice_executable, str) or not config.spice_executable.strip():
        raise ConfigError("Expected ngspice executable name or path")
    return config


def resolve_config(config, mode):
    if mode != RunMode.CHARACTERIZE:
        raise ConfigError("Analog energy transformer supports characterize only")
    return config
