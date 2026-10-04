"""Digital pretraining, selected strictly on validation cross-entropy."""
import math
import time

import torch
import torch.nn.functional as F

from experiments.mnist_relu.model import BiasFreeReluTeacher
from .common import checkpoint, cpu_tree, evaluate, heartbeat


def run(config, data, device, store):
    model = BiasFreeReluTeacher(device=device, dims=tuple(config["dims"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    initial = evaluate(model.logits, data, "validation")
    selected = dict(epoch=0, validation=initial)
    selected_weights = cpu_tree(model.parameters())
    best = (initial["cross_entropy"], -initial["accuracy"], 0)
    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        factor = config["settings"]["minimum_lr_factor"]
        lr = config["learning_rate"] * (factor + (1 - factor) * (1 + math.cos(math.pi * (epoch - 1) / max(config["epochs"] - 1, 1))) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum, examples = 0., 0
        for batch, (x, y) in enumerate(data.batches("train", epoch), 1):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.logits(x), y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Expected finite digital pretraining loss.")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            examples += len(y)
            if batch % 100 == 0:
                heartbeat(store, stage="pretrain", epoch=epoch, batch=batch)
        validation = evaluate(model.logits, data, "validation")
        rank = validation["cross_entropy"], -validation["accuracy"], epoch
        if rank < best:
            best, selected = rank, dict(epoch=epoch, validation=validation)
            selected_weights = cpu_tree(model.parameters())
        checkpoint(store.run_dir / "checkpoints/latest.pt", schema="cifar10.digital_pretrain_resume.v1", epoch=epoch,
                   weights=model.parameters(), optimizer=optimizer.state_dict(), selected=selected, selected_weights=selected_weights,
                   config=dict(config), data_receipt=data.receipt)
        record = dict(stage="pretrain", epoch=epoch, learning_rate=lr, train_examples=examples, train_cross_entropy=loss_sum / examples,
                      validation=validation, selected_epoch=selected["epoch"], order_sha256=data.epoch_order_sha256, elapsed_seconds=time.time() - start)
        store.append_metric(record)
        heartbeat(store, stage="pretrain", epoch=epoch, validation_accuracy=validation["accuracy"])
        print(f"pretrain epoch={epoch} validation_accuracy={validation['accuracy']:.4f} CE={validation['cross_entropy']:.4f} selected={selected['epoch']}", flush=True)
    for target, source in zip(model.parameters(), selected_weights, strict=True):
        target.data.copy_(source.to(device))
    selected_test = evaluate(model.logits, data, "test")
    passed = selected["validation"]["accuracy"] >= config["settings"]["minimum_validation_accuracy"]
    path = store.run_dir / "checkpoints/teacher.pt"
    checkpoint(path, schema="cifar10.digital_relu.v1", weights=model.parameters(), dims=config["dims"], data_receipt=data.receipt,
               selected=selected, test=selected_test, acceptance_passed=passed)
    return dict(initial_validation=initial, selected=selected, test=selected_test, acceptance_passed=passed,
                checkpoint="checkpoints/teacher.pt", data_receipt=data.receipt), [path, store.run_dir / "checkpoints/latest.pt"]
