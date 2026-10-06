from pathlib import Path
from setuptools import find_namespace_packages
from setuptools.config.pyprojecttoml import read_configuration

def test_package_discovery_keeps_only_crossbar_runtime_and_campaign_modules():
    root = Path(__file__).parents[1]
    config = read_configuration(root / "pyproject.toml", expand=False)
    find = config["tool"]["setuptools"]["packages"]["find"]
    packages = set(find_namespace_packages(where=str(root), include=find["include"], exclude=find["exclude"]))
    assert {"campaigns", "ebl", "experiments", "labs", "training", "workflow", "experiments.cifar_crossbar", "experiments.mnist_analog_relu"} <= packages
    assert not any(p == "model" or p.startswith("model.") for p in packages)
    for name in ("cifar_crossbar/runtime.py", "cifar10_crossbar/runtime.py", "mnist_analog_relu/staged_runtime.py"):
        assert (root / "experiments" / name).is_file()
