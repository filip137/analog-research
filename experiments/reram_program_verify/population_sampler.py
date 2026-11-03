"""Pinned-environment helper for sampling IBM AIHWKit ReRAM identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import torch

from core.artifacts import atomic_write_json, sha256_file
from experiments.reram_program_verify.runtime import _save_population
from hardware.devices.om import sample_ibm_reram_population


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True)
    parser.add_argument("--num-devices", required=True, type=int)
    parser.add_argument("--construction-seed", required=True, type=int)
    parser.add_argument("--required-aihwkit-version", required=True)
    parser.add_argument("--enable-published-corruption", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    population = sample_ibm_reram_population(
        preset=args.preset,
        num_devices=args.num_devices,
        construction_seed=args.construction_seed,
        enable_published_corruption=args.enable_published_corruption,
        required_aihwkit_version=args.required_aihwkit_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _save_population(args.output, population)
    import aihwkit

    atomic_write_json(
        args.receipt,
        {
            "schema": "ebl.ibm_reram.population_sampling_receipt",
            "schema_version": 1,
            "backend": "external_pinned_aihwkit_python",
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "aihwkit_version": str(aihwkit.__version__),
            "preset": args.preset,
            "num_devices": args.num_devices,
            "construction_seed": args.construction_seed,
            "enable_published_corruption": args.enable_published_corruption,
            "population_sha256": sha256_file(args.output),
        },
    )
    # Force strict JSON decoding once in the producing environment as a small
    # defense against accidentally writing a partial receipt.
    json.loads(args.receipt.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
