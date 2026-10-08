"""(i) Deployment: how the digital weights are mapped onto the crossbar.

The network family builds the student: the complete pretrained teacher with a
frozen digital remainder and the declared analog matrices (ResNet-32 suffix
convolutions plus classifier, or the MLPs of the last OPT decoder layers).
Each logical matrix is flattened to [out, in], scaled by its own absolute
maximum into normalized targets q in [-1, 1] and split into square tiles whose
input slices are summed digitally. The technology's encoding then states how
one q is represented physically. Programming those targets is the separate
P&V stage.
"""

from __future__ import annotations

import torch

from experiments.artifacts import content_hash
from workflow.networks import FAMILIES, family

# Physical devices behind one logical weight for each declared encoding.
DEVICES_PER_WEIGHT = {
    "differential_pair_25us": 2,
    "single_cell_intrinsic_reference": 1,
}


def load_teacher(path, network, device):
    """Load and verify the pinned teacher checkpoint of the lifecycle network."""

    return FAMILIES[network.family].load_teacher(path, network, device)


def build_network(teacher, lifecycle):
    """Map the teacher's analog matrices onto normalized crossbar targets."""

    network = family(lifecycle.network).build_network(teacher, lifecycle)
    network.read_noise = lifecycle.deployment.read_noise
    return network


def mapping_receipt(network, lifecycle) -> dict:
    """Deterministic description of the logical-to-physical mapping."""

    receipt = network.mapping_receipt()
    deployment = lifecycle.deployment
    receipt.update(
        **lifecycle.network.mapping_fields(),
        scaling=deployment.scaling,
        encoding=deployment.encoding,
        physical_devices=receipt["logical_weights"] * DEVICES_PER_WEIGHT[deployment.encoding],
        read_noise=deployment.read_noise,
        calibration=deployment.calibration,
        frozen_digital_prefix=True,
    )
    return receipt


def mapping_digest(receipt: dict) -> str:
    return content_hash(receipt)


def targets(network) -> torch.Tensor:
    """Normalized per-layer targets of the network's current digital weights."""

    return network.q.detach().clone()
