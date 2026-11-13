"""Strict, bounded schema for the first complete multi-token circuit check."""
from dataclasses import dataclass, fields
import math
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID = "convergent_energy_transformer.v1"


@dataclass(frozen=True)
class Config:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    case: str = "four_token_two_head"
    betas: tuple = (.04,.02,.01,.005)
    learning_rate: float = .05
    wall_seconds: int = 600
    spice_timeout: int = 30
    spice_executable: str = "ngspice"


def parse_config(raw):
    if not isinstance(raw,dict) or set(raw)-{f.name for f in fields(Config)}:
        raise ConfigError("Expected only declared convergent transformer fields")
    data=dict(raw)
    if "betas" in data:
        if not isinstance(data["betas"],(tuple,list)): raise ConfigError("Expected beta sequence")
        data["betas"]=tuple(data["betas"])
    c=Config(**data)
    if c.experiment_id!=EXPERIMENT_ID or type(c.schema_version) is not int or c.schema_version!=1:
        raise ConfigError("Expected convergent_energy_transformer.v1 schema 1")
    if c.case!="four_token_two_head": raise ConfigError("Unknown frozen circuit case")
    if not 1<=len(c.betas)<=4 or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<x<=.05 for x in c.betas):
        raise ConfigError("Expected one to four positive finite nudges <=0.05")
    if len(set(c.betas))!=len(c.betas): raise ConfigError("Duplicate nudges")
    if type(c.learning_rate) not in (int,float) or not math.isfinite(c.learning_rate) or not 0<c.learning_rate<=.1:
        raise ConfigError("Expected positive learning rate <=0.1")
    for name,upper in (("wall_seconds",600),("spice_timeout",30)):
        value=getattr(c,name)
        if type(value) is not int or not 1<=value<=upper: raise ConfigError(f"Out-of-budget {name}")
    if not isinstance(c.spice_executable,str) or not c.spice_executable.strip(): raise ConfigError("Expected SPICE executable")
    return c


def resolve_config(config,mode):
    if mode!=RunMode.CHARACTERIZE: raise ConfigError("This circuit pilot supports characterize only")
    return config
