"""Ownership, public execution and exact relocation regression contracts."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "architecture_fixtures"


@pytest.mark.parametrize("owner,forbidden", [
    ("core", {"hardware", "networks", "methods", "ebl", "experiments", "campaigns", "labs"}),
    ("hardware", {"networks", "methods", "ebl", "experiments", "campaigns", "labs"}),
    ("networks", {"methods", "ebl", "experiments", "campaigns", "labs"}),
    ("methods", {"ebl", "experiments", "campaigns", "labs"}),
])
def test_imports_follow_ownership(owner, forbidden):
    violations = []
    forbidden |= {"model", "training", "workflow"}
    for path in sorted((ROOT / owner).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                     [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name.split(".")[0] in forbidden:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "\n".join(violations)


def test_cli_discovery_and_collection_do_not_import_numerical_execution():
    code = """
import sys
from ebl.cli import build_parser
from ebl.lifecycle.plan import collect
build_parser()
assert collect([])['complete'] is False
assert not {'torch', 'numpy', 'h5py', 'aihwkit'} & set(sys.modules)
assert not any(n.startswith('ebl.runtime.opt_cmo_drn') for n in sys.modules)
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)


def test_network_construction_is_independent_of_execution():
    code = """
import sys
from networks.opt.drn import OptDrn
from networks.resistive.small import build_model_stack
from networks.resistive.mnist import build_student_network
assert not any(n.split('.')[0] in {'ebl', 'experiments', 'methods', 'labs'} for n in sys.modules)
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)


def test_retired_packages_and_executable_workflow_are_absent():
    for owner in ("model", "training", "workflow"):
        assert not list((ROOT / owner).rglob("*.py"))


def test_shared_sources_remain_byte_identical_to_baseline():
    expected = json.loads((FIXTURES / "shared.json").read_text())
    manifest = json.loads((ROOT / "docs/shared_native_modules.json").read_text())
    assert set(expected) == set(manifest["modules"])
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def test_existing_configuration_digests_are_unchanged():
    from core.artifacts import content_hash
    from core.config import to_plain_data
    from ebl.registry import resolve_experiment_config
    from ebl.schema.base import RunMode
    for key, expected in json.loads((FIXTURES / "configs.json").read_text()).items():
        name, mode = key.rsplit("/", 1)
        _, resolved = resolve_experiment_config(ROOT / name, RunMode(mode))
        assert content_hash(to_plain_data(resolved)) == expected, key


def test_opt_forward_gradient_and_capture_match_pre_migration():
    import torch
    from architecture_cases import opt_traces
    meta = json.loads((FIXTURES / "provenance.json").read_text())
    if torch.__version__ != meta["torch"]:
        pytest.skip(f"Exact migration traces require PyTorch {meta['torch']}")
    assert opt_traces() == json.loads((FIXTURES / "opt.json").read_text())


def test_physical_fault_groups_use_declared_metadata():
    from hardware.faults import MatrixSpec, FaultCase, hierarchical_faults
    layout = [MatrixSpec("arbitrary.lora_a", (8, 8), 0, 64)]
    result = hierarchical_faults(layout, 4, 2, FaultCase("random", .25), 17, "test", 1,
                                 domains={"arbitrary.lora_a": "declared-group"})
    assert result["receipt"]["groups"][0]["group"] == "declared-group"
    assert result["receipt"]["failed_devices"] == 16
    with pytest.raises(ValueError, match="explicit fault domain"):
        hierarchical_faults(layout, 4, 2, FaultCase("random", .25), 17, "test", 1, domains={})


def test_lifecycle_cli_preserves_command_arguments(tmp_path):
    output = tmp_path / "collection.json"
    result = subprocess.run([sys.executable, "-B", "-m", "ebl", "lifecycle", "collect",
                             str(tmp_path / "missing"), "--output", str(output)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert json.loads(output.read_text())["runs"][0]["coverage"] == "missing"
    result = subprocess.run([sys.executable, "-B", "-m", "ebl", "lifecycle", "recovery", "--help"],
                             cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--teacher-weights" in result.stdout
