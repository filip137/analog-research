from pathlib import Path
import json
import pytest
from experiments import get_definition, resolve_experiment_config, to_plain_data

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [p for p in sorted((ROOT / "examples").rglob("*.json"))
           if "experiment_id" in json.loads(p.read_text())]

@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_registered_example_resolves_every_supported_mode(path):
    payload = json.loads(path.read_text())
    definition = get_definition(payload["experiment_id"])
    for mode in definition.supported_modes:
        resolved_definition, spec = resolve_experiment_config(path, mode)
        assert resolved_definition == definition
        assert to_plain_data(spec)["experiment_id"] == definition.experiment_id
