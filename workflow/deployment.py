"""(i) Deployment: how the digital weights are mapped onto the crossbar.

The pretrained ResNet-32 stays complete. Its digital prefix is frozen; the
last ``analog_convolutions`` layer3 convolutions and the classifier become
crossbar matrices. Each logical kernel is flattened to [out, in*k*k], scaled
by its own absolute maximum into normalized targets q in [-1, 1] and split
into square tiles whose input slices are summed digitally. The technology's
encoding then states how one q is represented physically. Programming those
targets is the separate P&V stage.
"""

from __future__ import annotations

from pathlib import Path

import torch

from experiments.artifacts import content_hash, sha256_file
from experiments.cifar_crossbar.model import CifarResNet32, CrossbarSuffix
from experiments.cifar_crossbar.prepare import SOURCES
from experiments.cifar_crossbar.runtime import load

# Physical devices behind one logical weight for each declared encoding.
DEVICES_PER_WEIGHT = {
    "differential_pair_25us": 2,
    "single_cell_intrinsic_reference": 1,
}


def load_teacher(path, network, device):
    """Load the pinned public ResNet-32 checkpoint for the lifecycle dataset."""

    path = Path(path)
    digest = sha256_file(path)
    if not digest.startswith(SOURCES[network.dataset][1]):
        raise ValueError(
            "Expected the declared public pretrained ResNet-32 checkpoint digest "
            f"for {network.dataset}. Provided value: {digest}."
        )
    teacher = CifarResNet32(network.classes).to(device)
    teacher.load_state_dict(load(path), strict=True)
    teacher.eval().requires_grad_(False)
    return teacher, digest


def build_network(teacher, lifecycle) -> CrossbarSuffix:
    """Map the teacher's analog suffix onto normalized crossbar targets."""

    network = CrossbarSuffix(
        teacher, lifecycle.network.suffix, lifecycle.deployment.tile_size
    )
    network.read_noise = lifecycle.deployment.read_noise
    return network


def mapping_receipt(network: CrossbarSuffix, lifecycle) -> dict:
    """Deterministic description of the logical-to-physical mapping."""

    receipt = network.mapping_receipt()
    deployment = lifecycle.deployment
    receipt.update(
        analog_convolutions=lifecycle.network.analog_convolutions,
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


def targets(network: CrossbarSuffix) -> torch.Tensor:
    """Normalized per-layer targets of the network's current digital weights."""

    return network.q.detach().clone()
