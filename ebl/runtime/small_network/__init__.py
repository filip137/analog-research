"""Lazy executable application surface for the small-DRN experiment.

Keeping these imports lazy preserves the lightweight, side-effect-free config
parser: merely describing an experiment must not import torch or datasets.
"""

from typing import Any


def run_train(request: Any) -> int:
    from ebl.runtime.small_network.runtime import run_train as handler

    return handler(request)


def run_linspace(request: Any) -> int:
    from ebl.runtime.small_network.runtime import run_linspace as handler

    return handler(request)


def run_validate(request: Any) -> int:
    from ebl.runtime.small_network.runtime import run_validate as handler

    return handler(request)


def import_legacy_checkpoint(request: Any) -> int:
    from ebl.runtime.small_network.runtime import (
        import_legacy_checkpoint as handler,
    )

    return handler(request)


__all__ = [
    "import_legacy_checkpoint",
    "run_linspace",
    "run_train",
    "run_validate",
]
