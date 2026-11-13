"""Bounded native-MOS attention training, through the public train command."""
from dataclasses import dataclass, fields
import math
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID="mos_attention_learning.v1"


@dataclass(frozen=True)
class Config:
    experiment_id: str=EXPERIMENT_ID
    schema_version: int=1
    case: str="three_token_masked_mean"
    epochs: int=200
    beta: float=.005
    learning_rate: float=.003
    train_seed: int=3100
    test_seed: int=3102
    train_count: int=12
    test_count: int=32
    wall_seconds: int=1800
    spice_timeout: int=30
    spice_executable: str="ngspice"


def parse_config(raw):
    if not isinstance(raw,dict) or set(raw)-{f.name for f in fields(Config)}:
        raise ConfigError("Expected only declared MOS attention learning fields")
    c=Config(**raw)
    if c.experiment_id!=EXPERIMENT_ID or type(c.schema_version) is not int or c.schema_version!=1:
        raise ConfigError("Expected mos_attention_learning.v1 schema 1")
    if c.case!="three_token_masked_mean": raise ConfigError("Unknown attention task")
    for name,upper in (("epochs",200),("train_count",12),("test_count",32),
                       ("wall_seconds",1800),("spice_timeout",30)):
        if type(getattr(c,name)) is not int or not 1<=getattr(c,name)<=upper:
            raise ConfigError(f"Out-of-budget {name}")
    for name in ("train_seed","test_seed"):
        if type(getattr(c,name)) is not int or not 0<=getattr(c,name)<2**32:
            raise ConfigError(f"Invalid {name}")
    if c.train_seed==c.test_seed: raise ConfigError("Training and held-out seeds must differ")
    for name,upper in (("beta",.01),("learning_rate",.01)):
        x=getattr(c,name)
        if type(x) not in (int,float) or not math.isfinite(x) or not 0<x<=upper:
            raise ConfigError(f"Invalid {name}")
    if not isinstance(c.spice_executable,str) or not c.spice_executable.strip():
        raise ConfigError("Expected SPICE executable")
    return c


def resolve_config(config,mode):
    if mode!=RunMode.TRAIN: raise ConfigError("This task supports train only")
    return config
