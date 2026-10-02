"""Load real pre-refactor IBM OM artifacts with the layered implementation.

The artifacts are gitignored study outputs; the test is skipped when they
are absent.  It checks loading and provenance parity only, not numerics.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn.config import _parse_weight_modifier
from experiments.schema import to_plain_data
from model.resistive.builders import ParameterBinding
from model.variable.parameter import DenseWeight
from training.ibm_om.population import load_om_array_population
from training.ibm_om.spec import (
    IbmReramHwaConfig,
    backfill_config_dict,
    backfill_weight_modifier_metadata,
)
from training.ibm_reram_hwa import IbmReramHwaParameterModifier


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs"
    / "train-continuous-mapped-hwa/20260826T190255.837865Z-326fd400-1ab2c53a"
)
DEVICE_MODEL = ROOT / "data" / "ibm_reram_om_pv128_hwa_v1.json"

pytestmark = pytest.mark.skipif(
    not (RUN / "checkpoints" / "resume.pt").is_file() or not DEVICE_MODEL.is_file(),
    reason="Real IBM OM study artifacts are not present.",
)


def _resume() -> dict:
    return torch.load(
        RUN / "checkpoints" / "resume.pt",
        weights_only=True,
        map_location="cpu",
    )


def test_old_checkpoint_metadata_back_fills_to_the_current_parse() -> None:
    metadata = _resume()["metadata"]
    resolved = json.loads((RUN / "config.resolved.json").read_text(encoding="utf-8"))
    for role in ("weight_modifier", "selection_weight_modifier"):
        saved = metadata[role]
        if saved["type"] != "ibm_reram_om_program_verify":
            continue
        source = resolved["settings"][role]
        current = to_plain_data(
            _parse_weight_modifier(source, f"config.modes.train.{role}")
        )
        assert backfill_weight_modifier_metadata(saved) == {
            "type": current["type"],
            "parameters": current["parameters"],
        }


def test_old_version_two_modifier_state_loads() -> None:
    resume = _resume()
    parameters = dict(resume["metadata"]["weight_modifier"]["parameters"])
    config = IbmReramHwaConfig(**parameters)
    population = load_om_array_population(
        RUN / "artifacts" / "ibm_om_population.training.npz"
    )
    bindings = []
    for key, shape in zip(population.binding_keys, population.binding_shapes):
        parameter = DenseWeight(
            (shape[0],),
            (shape[1],),
            1.0,
            "cpu",
            clamp=True,
            clamp_min=0.0,
            clamp_max=1.0,
        )
        bindings.append(ParameterBinding(key, parameter))
    state = resume["modifier_state"]["training"]
    assert state["version"] == 2
    assert state["device_model_sha256"] == sha256_file(DEVICE_MODEL)
    import training.ibm_om.modifier as modifier_module

    original = modifier_module.sample_om_array_population
    modifier_module.sample_om_array_population = lambda *args, **kwargs: population
    try:
        modifier = IbmReramHwaParameterModifier(
            tuple(bindings),
            config,
            device_model_path=DEVICE_MODEL,
            conductance_min=0.0,
            conductance_max=1.0,
        )
    finally:
        modifier_module.sample_om_array_population = original
    modifier.load_state_dict(state)
    assert modifier.state_dict()["version"] == 3


def test_old_deployment_config_rebuilds_with_explicit_fields() -> None:
    deployment = torch.load(
        RUN / "artifacts" / "ibm_om_deployment.pt",
        weights_only=True,
        map_location="cpu",
    )
    saved = deployment["config"]
    rebuilt = asdict(IbmReramHwaConfig(**dict(saved)))
    assert rebuilt == backfill_config_dict(saved)
    assert rebuilt["device_coordinate"] == "raw_active_v1"
    assert rebuilt["initial_state"] == "conditioned_lower_boundary"
