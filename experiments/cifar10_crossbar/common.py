"""Shared deterministic data, metrics and artifact helpers."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.mnist_relu.model import BiasFreeReluTeacher
from training.checkpoint import atomic_torch_save

ROOT = Path(__file__).resolve().parents[2]


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def setup_cuda(seed):
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Expected a working CUDA device; CPU training is prohibited.")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    probe = torch.ones(8, 8, device=device)
    assert float((probe @ probe)[0, 0]) == 8
    torch.cuda.synchronize()
    return device


class Data:
    def __init__(self, config, device):
        self.data_root = Path(os.environ.get("EBL_CIFAR_ROOT", config["data_root"]))
        source = CIFAR10(self.data_root, train=True, download=False)
        test = CIFAR10(self.data_root, train=False, download=False)
        order = torch.randperm(50000, generator=torch.Generator().manual_seed(config["data_seed"]))
        validation_indices, train_indices = order[:5000], order[5000:]
        train_pixels = source.data[train_indices.numpy()].astype(np.float64) / 255
        means = train_pixels.mean(axis=(0, 1, 2))
        stds = train_pixels.std(axis=(0, 1, 2))
        del train_pixels
        mean_gpu = torch.tensor(means, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std_gpu = torch.tensor(stds, device=device, dtype=torch.float32).view(1, 3, 1, 1)

        def encode(images):
            x = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(device=device, dtype=torch.float32) / 255
            return ((x - mean_gpu) / std_gpu).reshape(len(x), 3072)

        full = encode(source.data)
        labels = torch.tensor(source.targets, device=device, dtype=torch.long)
        self.train = full[train_indices.to(device)], labels[train_indices.to(device)]
        self.validation = full[validation_indices.to(device)], labels[validation_indices.to(device)]
        self.test = encode(test.data), torch.tensor(test.targets, device=device, dtype=torch.long)
        self.receipt = dict(dataset="CIFAR10", flatten="CHW", train_examples=45000, validation_examples=5000, test_examples=10000,
                            split_seed=config["data_seed"], train_indices_sha256=tensor_hash(train_indices),
                            validation_indices_sha256=tensor_hash(validation_indices), mean=means.tolist(), std=stds.tolist(),
                            normalization_fit="training_split_only", augmentation="none")
        self.config, self.device = config, device

    def batches(self, split, epoch=0):
        x, y = getattr(self, split)
        if split == "train":
            generator = torch.Generator().manual_seed(self.config["data_seed"] + epoch)
            indices = torch.randperm(len(y), generator=generator)
            self.epoch_order_sha256 = tensor_hash(indices)
            indices = indices.to(self.device)
            batch_size = self.config["batch_size"]
        else:
            count = len(y) if self.config["evaluation_limit"] is None else min(len(y), self.config["evaluation_limit"])
            indices = torch.arange(count, device=self.device)
            batch_size = 512
        for batch_index, start in enumerate(range(0, len(indices), batch_size)):
            if split == "train" and self.config["maximum_batches"] is not None and batch_index >= self.config["maximum_batches"]:
                break
            index = indices[start:start + batch_size]
            yield x[index], y[index]


@torch.no_grad()
def evaluate(logits_fn, data, split, teacher=None):
    n, correct, agreement, ce, kl = 0, 0, 0, 0., 0.
    predictions, reference_predictions = [], []
    for x, y in data.batches(split):
        logits = logits_fn(x)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Expected finite CIFAR-10 logits.")
        pred = logits.argmax(1)
        correct += int((pred == y).sum())
        ce += float(F.cross_entropy(logits, y, reduction="sum"))
        n += len(y)
        predictions.append(pred.cpu())
        if teacher is not None:
            teacher_logits = teacher.logits(x)
            logp = F.log_softmax(teacher_logits, dim=1)
            kl += float((logp.exp() * (logp - F.log_softmax(logits, dim=1))).sum())
            agreement += int((pred == teacher_logits.argmax(1)).sum())
            reference_predictions.append(teacher_logits.argmax(1).cpu())
    return dict(examples=n, accuracy=correct / n, correct=correct, cross_entropy=ce / n,
                kl_teacher_student=kl / n if teacher else None,
                teacher_agreement=agreement / n if teacher else None,
                predictions_sha256=tensor_hash(torch.cat(predictions)),
                teacher_predictions_sha256=tensor_hash(torch.cat(reference_predictions)) if teacher else None)


def heartbeat(store, **fields):
    atomic_write_json(store.run_dir / "heartbeat.json", dict(pid=os.getpid(), time=time.time(), **fields))


def checkpoint(path, **payload):
    atomic_torch_save(cpu_tree(payload), path)


def load_teacher(path, config, data, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") != "cifar10.digital_relu.v1" or tuple(saved["dims"]) != tuple(config["dims"]):
        raise ValueError("Expected a frozen CIFAR-10 digital ReLU checkpoint with matching dimensions.")
    if saved["data_receipt"] != data.receipt:
        raise ValueError("Expected teacher and student to use identical CIFAR-10 preprocessing and splits.")
    if not saved["acceptance_passed"]:
        raise ValueError("Expected the digital teacher to pass its declared validation gate.")
    model = BiasFreeReluTeacher(device=device, dims=tuple(config["dims"]))
    for target, weight in zip(model.parameters(), saved["weights"], strict=True):
        target.data.copy_(weight.to(device))
        target.requires_grad_(False)
    return model, sha256_file(Path(path))
