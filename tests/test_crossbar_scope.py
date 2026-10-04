"""Guard the executable boundary while keeping historical evidence inspectable."""

import ast
import io
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ebl.cli import main
from experiments.definitions import EXPERIMENT_REGISTRY, RETIRED_EXPERIMENT_IDS


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = {
    "cifar10_ibm_om_crossbar.v1", "cifar_crossbar_fault_sweep.v1",
    "cifar_crossbar_full_epochs.v1", "cifar_om_closed_loop_lr.v1",
    "cifar_om_open_loop.v1", "cifar_pcm_fault_recovery.v1",
    "cifar_pcm_hwa_comparison.v1", "cifar_pcm_recovery_epochs.v1",
    "cifar_resnet_suffix_recovery.v1", "mnist_ibm_om_crossbar_relu.v2",
    "mnist_relu.v1", "mnist_relu.v2", "ibm_reram_program_verify.v1",
}


def test_registry_is_crossbar_and_required_controls_only():
    assert set(EXPERIMENT_REGISTRY) == SUPPORTED


@pytest.mark.parametrize("experiment_id", sorted(RETIRED_EXPERIMENT_IDS))
def test_retired_experiment_reports_reproduction_revision(experiment_id):
    error = io.StringIO()
    assert main(["describe", "--experiment", experiment_id], stderr=error) == 2
    assert "retired" in error.getvalue()
    assert "16938bf" in error.getvalue()


@pytest.mark.parametrize("args", [
    ["linspace"], ["checkpoint", "import-legacy"],
    ["train", "--config", "c.json", "--output-dir", "runs", "--base-weights", "w.pt"],
])
def test_retired_cli_surfaces_are_rejected(args):
    error = io.StringIO()
    assert main(args, stderr=error) == 2
    assert "Expected command-line arguments" in error.getvalue()


def test_supported_runtimes_import_without_equilibrium_modules():
    # A fresh process catches both eager and indirect imports and avoids pytest's
    # import cache concealing an accidental dependency on a retired implementation.
    script = '''
import importlib
import importlib.abc
import sys
class BlockRetired(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'model' or fullname.startswith(('model.',
            'experiments.small_network', 'experiments.mnist_relu_drn',
            'experiments.mnist_analog_relu.runtime', 'training.tiki_taka')):
            raise AssertionError('Retired dependency: ' + fullname)
sys.meta_path.insert(0, BlockRetired())
for name in (
    'ebl.cli', 'experiments.mnist_relu.runtime',
    'experiments.mnist_analog_relu.staged_runtime',
    'experiments.reram_program_verify.runtime',
    'experiments.cifar10_crossbar.runtime',
    'experiments.cifar_crossbar.runtime',
    'experiments.cifar_crossbar.fault_runtime',
    'experiments.cifar_crossbar.hwa_fault_runtime',
    'experiments.cifar_crossbar.epoch_runtime',
    'experiments.cifar_crossbar.full_epoch_runtime',
    'experiments.cifar_crossbar.sweep_runtime',
    'experiments.cifar_crossbar.om_open_loop_runtime',
    'experiments.cifar_crossbar.closed_loop_lr_runtime',
):
    importlib.import_module(name)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
        # NumPy/MKL imports elsewhere in the suite can export INTEL while
        # this PyTorch build uses libgomp. Keep the child import check isolated.
        env={**os.environ, "MKL_THREADING_LAYER": "GNU"},
    )
    assert completed.returncode == 0, completed.stderr


def test_active_code_has_no_retired_module_imports():
    forbidden = (
        "model", "experiments.small_network", "experiments.mnist_relu_drn",
        "experiments.mnist_analog_relu.runtime", "experiments.mnist_analog_relu.config",
        "training.tiki_taka", "training.star_drn",
    )
    for directory in ("ebl", "experiments", "training", "campaigns", "labs/tools"):
        for path in (ROOT / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                for name in names:
                    assert not any(name == f or name.startswith(f + ".") for f in forbidden), (path, name)
