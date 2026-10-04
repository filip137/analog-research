"""CIFAR ResNet-32 and an explicit, differentiable crossbar suffix.

Checkpoint topology: chenyaofo/pytorch-cifar-models, resnet32, 16/32/64,
five blocks per stage, projection shortcuts. The prefix is never converted.
No modules are replaced using traversal order or an ambiguous layer count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import copy
import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.downsample = (
            None
            if stride == 1 and inputs == outputs
            else nn.Sequential(
                nn.Conv2d(inputs, outputs, 1, stride, bias=False),
                nn.BatchNorm2d(outputs),
            )
        )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(
            self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + identity
        )


class CifarResNet32(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        if classes not in (10, 100):
            raise ValueError("Expected 10 or 100 CIFAR classes.")
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = nn.Sequential(*[BasicBlock(16, 16) for _ in range(5)])
        self.layer2 = nn.Sequential(
            BasicBlock(16, 32, 2), *[BasicBlock(32, 32) for _ in range(4)]
        )
        self.layer3 = nn.Sequential(
            BasicBlock(32, 64, 2), *[BasicBlock(64, 64) for _ in range(4)]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, classes)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x))))))
        return self.fc(self.avgpool(x).flatten(1))


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    h = sha256(str((tuple(value.shape), value.dtype)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_hash(state: dict) -> str:
    h = sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    shape: tuple[int, ...]
    offset: int
    size: int
    scale: float


class CrossbarSuffix(nn.Module):
    """Functional weights keep the pulse plant outside autograd.

    Each logical kernel is flattened [out, in*k*k]. Input-row slices are summed
    digitally; all tile partitions share the original layer's scale. Independent
    read noise is added to each tile's MVM output, never to persistent storage.
    """

    def __init__(self, teacher: CifarResNet32, suffix: str, tile_size: int = 512):
        super().__init__()
        starts = {"head": 5, "last_block": 4, "last_two_blocks": 3, "last_four_blocks": 1}
        if suffix not in starts or tile_size < 1:
            raise ValueError("Expected a supported suffix and positive tile size.")
        self.start, self.suffix, self.tile_size = starts[suffix], suffix, tile_size
        self.model = copy.deepcopy(teacher).eval()
        self.model.requires_grad_(False)
        names = [
            f"layer3.{i}.conv{j}" for i in range(self.start, 5) for j in (1, 2)
        ] + ["fc"]
        self.layout: list[MatrixSpec] = []
        values, offset = [], 0
        for name in names:
            weight = self.model.get_submodule(name).weight.detach()
            scale = max(float(weight.abs().max()), 1e-12)
            self.layout.append(
                MatrixSpec(name, tuple(weight.shape), offset, weight.numel(), scale)
            )
            values.append(weight.flatten() / scale)
            offset += weight.numel()
        self.q = nn.Parameter(torch.cat(values))
        self.register_buffer(
            "scales", torch.tensor([x.scale for x in self.layout], device=self.q.device)
        )
        self.output_gains = nn.Parameter(
            torch.ones(len(self.layout), device=self.q.device)
        )
        self.read_noise = 0.0
        self.read_generator = torch.Generator(device=self.q.device).manual_seed(0)
        self._by_name = {x.name: (i, x) for i, x in enumerate(self.layout)}
        self.enable_calibration(False)

    def train(self, mode: bool = True):
        # Running BN statistics never change incidentally with minibatch order.
        super().train(mode)
        self.model.eval()
        return self

    def enable_calibration(self, enabled: bool):
        self.model.requires_grad_(False)
        self.output_gains.requires_grad_(enabled)
        self.model.fc.bias.requires_grad_(enabled)
        for i in range(self.start, 5):
            for j in (1, 2):
                bn = self.model.get_submodule(f"layer3.{i}.bn{j}")
                bn.weight.requires_grad_(enabled)
                bn.bias.requires_grad_(enabled)

    def calibration_parameters(self):
        return [p for p in self.parameters() if p is not self.q and p.requires_grad]

    def prefix_hash(self):
        prefixes = ("conv1.", "bn1.", "layer1.", "layer2.") + tuple(
            f"layer3.{i}." for i in range(self.start)
        )
        return state_hash(
            {k: v for k, v in self.model.state_dict().items() if k.startswith(prefixes)}
        )

    @torch.no_grad()
    def features(self, x):
        m = self.model
        x = m.layer2(m.layer1(m.relu(m.bn1(m.conv1(x)))))
        for i in range(self.start):
            x = m.layer3[i](x)
        return m.avgpool(x).flatten(1) if self.start == 5 else x

    def _matrix(self, x, name, q):
        index, spec = self._by_name[name]
        matrix = q[spec.offset : spec.offset + spec.size].reshape(spec.shape[0], -1)
        if len(spec.shape) == 4:
            module = self.model.get_submodule(name)
            patches = F.unfold(
                x,
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
        else:
            patches = x.unsqueeze(-1)
        columns = []
        for output in range(0, matrix.shape[0], self.tile_size):
            accum = None
            for row in range(0, matrix.shape[1], self.tile_size):
                part = torch.einsum(
                    "oi,bil->bol",
                    matrix[
                        output : output + self.tile_size, row : row + self.tile_size
                    ],
                    patches[:, row : row + self.tile_size],
                )
                if self.read_noise:
                    part = part + self.read_noise * torch.randn(
                        part.shape, device=part.device, generator=self.read_generator
                    )
                accum = part if accum is None else accum + part
            columns.append(accum)
        result = (
            torch.cat(columns, dim=1) * self.scales[index] * self.output_gains[index]
        )
        if len(spec.shape) == 4:
            h = (
                x.shape[-2]
                + 2 * module.padding[0]
                - module.dilation[0] * (module.kernel_size[0] - 1)
                - 1
            ) // module.stride[0] + 1
            w = (
                x.shape[-1]
                + 2 * module.padding[1]
                - module.dilation[1] * (module.kernel_size[1] - 1)
                - 1
            ) // module.stride[1] + 1
            return result.reshape(x.shape[0], spec.shape[0], h, w)
        return result.squeeze(-1) + self.model.fc.bias

    def forward_features(self, x, q=None):
        q = self.q if q is None else q
        if q.shape != self.q.shape:
            raise ValueError("Expected one normalized value per suffix weight.")
        for i in range(self.start, 5):
            block = self.model.layer3[i]
            identity = x
            x = F.relu(block.bn1(self._matrix(x, f"layer3.{i}.conv1", q)))
            x = F.relu(block.bn2(self._matrix(x, f"layer3.{i}.conv2", q)) + identity)
        if self.start < 5:
            x = self.model.avgpool(x).flatten(1)
        return self._matrix(x, "fc", q)

    def forward(self, x, q=None):
        return self.forward_features(self.features(x), q)

    def mapping_receipt(self):
        tiles = []
        for spec in self.layout:
            rows = spec.size // spec.shape[0]
            for output in range(0, spec.shape[0], self.tile_size):
                for row in range(0, rows, self.tile_size):
                    tiles.append(
                        {
                            "module": spec.name,
                            "input_start": row,
                            "output_start": output,
                            "rows": min(self.tile_size, rows - row),
                            "columns": min(self.tile_size, spec.shape[0] - output),
                        }
                    )
        return {
            "suffix": self.suffix,
            "matrices": [asdict(s) for s in self.layout],
            "logical_weights": self.q.numel(),
            "logical_tiles": tiles,
            "tile_capacity": self.tile_size**2,
            "prefix_sha256": self.prefix_hash(),
        }
