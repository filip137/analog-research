from pathlib import Path

from setuptools import Distribution, find_namespace_packages
from setuptools.command.build_py import build_py
from setuptools.config.pyprojecttoml import read_configuration


def test_package_discovery_keeps_runtime_and_campaign_modules() -> None:
    root = Path(__file__).parents[1]
    configuration = read_configuration(
        root / "pyproject.toml",
        expand=False,
    )
    find = configuration["tool"]["setuptools"]["packages"]["find"]
    packages = set(
        find_namespace_packages(
            where=str(root),
            include=find["include"],
            exclude=find["exclude"],
        )
    )
    assert {
        "campaigns",
        "ebl",
        "experiments",
        "labs",
        "core",
        "hardware",
        "networks",
        "methods",
    }.issubset(packages)

    distribution = Distribution()
    distribution.script_name = str(root / "pyproject.toml")
    command = build_py(distribution)
    labs_modules = {
        module
        for _package, module, _path in command.find_package_modules(
            "labs",
            str(root / "labs"),
        )
    }
    assert "datasets" not in labs_modules
    assert {
        "small_network",
        "small_network_config",
        "small_network_core",
    }.isdisjoint(labs_modules)

    experiment_modules = {
        module
        for _package, module, _path in command.find_package_modules(
            "ebl.runtime.small_network",
            str(root / "ebl" / "runtime" / "small_network"),
        )
    }
    assert {"components", "runtime"}.issubset(experiment_modules)
