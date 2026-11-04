from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class Solver:
    settling_iterations: int=4
    gradient_iterations: int=4
    voltage_amp: float=5.0
    current_amp: float=0.1
    input_gain: float=100.0
    token_chunk_size: int=256
    activation_checkpointing: bool=True

@dataclass(frozen=True)
class Data:
    corpus: str='opt125m-synthetic-50m'
    sha256: str='ce58085af0ae1add6415372a50bf20ba4238ee8cae5ee55093c83e6a9a36d984'
    sequences: int=195226
    development_sequences: int=256
    sequence_length: int=256
    seed: int=20261008
    mapping_sequences: int=1
