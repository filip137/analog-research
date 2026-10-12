"""Matched OPT recovery token views; frozen HWA and deployment sections stay intact."""
from __future__ import annotations

import torch

from experiments.cifar_crossbar.model import tensor_hash


def evaluation_names(lifecycle):
    names = ["development"]
    if lifecycle.data.evaluation == "test":
        names.append("test")
    if lifecycle.transfer_tests:
        names += [test.cache_key for test in lifecycle.transfer_tests.tests]
    return names


def extra_evaluation_names(lifecycle):
    views = lifecycle.recovery_views
    return ([] if views is None else [f"context{length}_{name}"
            for length in views.evaluation_lengths if length != lifecycle.data.sequence_length
            for name in evaluation_names(lifecycle)])


def build(cache, corpus, lifecycle):
    from workflow.networks.opt_mlp import _cohort
    views = lifecycle.recovery_views
    if views is None:
        return
    stream = corpus["splits"]["train"]
    count = len(stream) // views.parent_length
    if count < views.parent_sequences:
        raise ValueError("Not enough complete recovery parent windows.")
    indices = torch.randperm(count, generator=torch.Generator().manual_seed(lifecycle.data.data_seed))[:views.parent_sequences]
    parents = stream[:count * views.parent_length].reshape(count, views.parent_length)[indices]
    tokens = parents.reshape(-1, views.context_length)
    cache["recovery_context"] = _cohort(tokens, "train", indices.tolist())
    cache["recovery_context"]["parent_indices"] = indices
    cache["recovery_context"]["cohort"].update(
        parent_length=views.parent_length, parent_sequences=views.parent_sequences,
        parent_tokens_sha256=tensor_hash(parents), parent_indices_sha256=tensor_hash(indices),
        score_stride=views.score_stride, scored_tokens=views.scored_tokens,
        input_tokens=views.input_tokens,
    )
    for length in views.evaluation_lengths:
        if length == lifecycle.data.sequence_length:
            continue
        for name in evaluation_names(lifecycle):
            old = cache[name]
            tokens = old["tokens"].reshape(-1, length)
            new = _cohort(tokens, old["cohort"]["split"], range(len(tokens)))
            if "corpus" in old["cohort"]:
                new["cohort"]["corpus"] = old["cohort"]["corpus"]
            cache[f"context{length}_{name}"] = new


def check(cache, lifecycle):
    views = lifecycle.recovery_views
    if views is None:
        if "recovery_context" in cache:
            raise ValueError("Unexpected recovery context in a legacy cache.")
        return
    row = cache.get("recovery_context", {})
    tokens = row.get("tokens")
    if tokens is None or tuple(tokens.shape) != (views.sequences, views.context_length):
        raise ValueError("Recovery context cohort shape mismatch.")
    receipt = row["cohort"]
    indices = row["parent_indices"]
    if (indices.numel() != views.parent_sequences or len(indices.unique()) != views.parent_sequences
            or receipt["parent_tokens_sha256"] != tensor_hash(tokens.reshape(-1, views.parent_length))
            or receipt["parent_indices_sha256"] != tensor_hash(indices)
            or receipt["tokens_sha256"] != tensor_hash(tokens)
            or receipt["score_stride"] != views.score_stride
            or receipt["scored_tokens"] != views.scored_tokens):
        raise ValueError("Recovery context cohort receipt mismatch.")
    for length in views.evaluation_lengths:
        if length == lifecycle.data.sequence_length:
            continue
        for name in evaluation_names(lifecycle):
            value = cache[f"context{length}_{name}"]
            if (value["tokens"].shape[1] != length
                    or not torch.equal(value["tokens"].flatten(), cache[name]["tokens"].flatten())
                    or value["cohort"]["tokens_sha256"] != tensor_hash(value["tokens"])):
                raise ValueError("Fixed evaluation context changed its raw token prefix.")


def order(lifecycle, epoch):
    views = lifecycle.recovery_views
    parents = torch.randperm(views.parent_sequences,
                            generator=torch.Generator().manual_seed(lifecycle.onchip.seed + epoch - 1))
    chunks = views.parent_length // views.context_length
    return (parents[:, None] * chunks + torch.arange(chunks)[None, :]).flatten()


def loss_mask(lifecycle, device):
    views = lifecycle.recovery_views
    return torch.arange(views.context_length - 1, device=device) % views.score_stride != views.score_stride - 1


def accounting(lifecycle, epochs):
    views = lifecycle.recovery_views
    return {"input_token_presentations": epochs * views.input_tokens,
            "scored_token_presentations": epochs * views.scored_tokens,
            "optimizer_updates": epochs * views.sequences // lifecycle.onchip.batch_size}
