"""Pinned-environment helper for sampling the AIHWKit Wan-2022 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import torch

from core.artifacts import atomic_write_json
from experiments.reram_program_verify.analysis import sample_wan_reference


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-json", required=True)
    parser.add_argument("--samples-per-target", required=True, type=int)
    parser.add_argument("--g-max-us", required=True, type=float)
    parser.add_argument("--noise-scale", required=True, type=float)
    parser.add_argument("--wan-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    targets = json.loads(args.targets_json)
    if not isinstance(targets, list):
        raise ValueError("Expected --targets-json to decode to a list.")
    artifact = sample_wan_reference(
        targets=[float(value) for value in targets],
        samples_per_target=args.samples_per_target,
        g_max_us=args.g_max_us,
        noise_scale=args.noise_scale,
        wan_seed=args.wan_seed,
    )
    artifact["sampling_environment"] = {
        "backend": "external_pinned_aihwkit_python",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "aihwkit_version": artifact["aihwkit_version"],
    }
    atomic_write_json(args.output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
