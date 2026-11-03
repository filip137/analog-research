"""core.execution — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from core.contracts import config_error
from core.config import _boolean, _integer, _object


@dataclass(frozen=True)
class Execution:
    """How a stage executes; it is not a scientific lifecycle setting."""

    device: str
    cpu_threads: int
    disable_cudnn: bool

    def to_dict(self) -> dict:
        return {"device": self.device, "cpu_threads": self.cpu_threads, "disable_cudnn": self.disable_cudnn}


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
