"""Network families: everything a lifecycle stage needs to know about one network.

The device, program-and-verify and on-chip update code only ever see a flat
target vector ``q`` and its matrix ``layout``. A family module supplies the
rest, with the same functions in each module:

- teacher and network: ``pinned_digest``, ``load_teacher``, ``build_network``;
  the network exposes ``q``, ``layout``, ``read_noise``/``read_generator``,
  ``features``, ``forward_features(features, q)``, ``enable_calibration``,
  ``calibration_parameters``, ``prefix_hash`` and ``mapping_receipt``;
- cohorts: ``build_cache``, ``check_cache``, ``to_device``, ``cohorts``;
- training: ``examples``, ``batch``, ``objective``, ``hwa_inputs``,
  ``hwa_examples``, ``hwa_batch``;
- metrics: ``evaluate`` (always reports ``teacher_kl``), ``SUMMARY_METRICS``,
  ``PRESENTATIONS``, ``prepare_report``, ``deployment_summary``.

Adding a network family means adding one such module, registering it here and
in ``ebl.schema.lifecycle``, and adding its contract tests.
"""

from __future__ import annotations

from types import ModuleType
from importlib import import_module

from networks.config import CIFAR_FAMILY, OPT_FAMILY
FAMILIES = {CIFAR_FAMILY: "networks.cifar", OPT_FAMILY: "networks.opt"}


def family(network) -> ModuleType:
    """The family module of a lifecycle network section."""

    return import_module(FAMILIES[network.family])


__all__ = ["FAMILIES", "family"]
