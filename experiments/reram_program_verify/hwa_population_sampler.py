"""Pinned-environment sampler for named IBM OM HWA array assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Any

import torch

from experiments.artifacts import atomic_write_json, sha256_file
import training.ibm_om.sampling as population_module
from training.ibm_reram_hwa import (
    _ARRAY_SAMPLING_RECEIPT_SCHEMA,
    _ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION,
    _REQUIRED_AIHWKIT_VERSION,
    _sample_om_array_population_layout,
    save_om_array_population,
)
from training.ibm_reram_program_verify import OM_PRESET


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    return parser


def _request(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Expected --request-json to contain strict JSON.") from error
    expected = {
        "preset",
        "assignment_seed",
        "corruption_policy",
        "binding_keys",
        "binding_shapes",
        "required_aihwkit_version",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Expected exact OM HWA population request fields.")
    keys = value["binding_keys"]
    shapes = value["binding_shapes"]
    seed = value["assignment_seed"]
    if (
        value["preset"] != OM_PRESET
        or value["required_aihwkit_version"] != _REQUIRED_AIHWKIT_VERSION
        or value["corruption_policy"]
        not in {"counterfactual_repaired", "published"}
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**63
        or not isinstance(keys, list)
        or not keys
        or not all(isinstance(key, str) and key for key in keys)
        or len(set(keys)) != len(keys)
        or not isinstance(shapes, list)
        or len(shapes) != len(keys)
        or not all(
            isinstance(shape, list)
            and len(shape) == 2
            and all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and item > 0
                for item in shape
            )
            for shape in shapes
        )
    ):
        raise ValueError("Expected a valid canonical OM HWA population request.")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = _request(args.request_json)
    import aihwkit

    version = str(getattr(aihwkit, "__version__", "unknown"))
    if version != _REQUIRED_AIHWKIT_VERSION:
        raise RuntimeError(
            "Expected AIHWKit version '1.1.0' for OM HWA population sampling. "
            f"Provided value: {version!r}."
        )
    population = _sample_om_array_population_layout(
        request["binding_keys"],
        request["binding_shapes"],
        assignment_seed=request["assignment_seed"],
        corruption_policy=request["corruption_policy"],
    )
    output = args.output.expanduser().resolve()
    receipt = args.receipt.expanduser().resolve()
    save_om_array_population(output, population)
    atomic_write_json(
        receipt,
        {
            "schema": _ARRAY_SAMPLING_RECEIPT_SCHEMA,
            "schema_version": _ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION,
            "backend": "external_pinned_aihwkit_python",
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "aihwkit_version": version,
            "request": request,
            "num_cells": population.size,
            "population_fingerprint": population.fingerprint,
            "population_sha256": sha256_file(output),
            "sampler_source_sha256": sha256_file(Path(__file__).resolve()),
            "population_implementation_sha256": sha256_file(
                Path(population_module.__file__).resolve()
            ),
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
