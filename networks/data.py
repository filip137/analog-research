"""Fixed cohorts of every stage, built and checked by the lifecycle's network family.

ResNet-32: the class-balanced CIFAR 45000/5000 split with cached frozen-prefix
features and teacher logits. OPT: non-overlapping token sequences of one
pinned corpus; the frozen prefix and teacher logits run per minibatch. See
``networks`` for the per-family details.
"""

from __future__ import annotations

from networks import family


def evaluation_split(lifecycle) -> str:
    return lifecycle.data.evaluation


def transfer_splits(lifecycle) -> list[str]:
    """Cache keys of the transfer test and task cohorts, scored with every evaluation."""

    tests = () if lifecycle.transfer_tests is None else lifecycle.transfer_tests.tests
    from networks.context_views import extra_evaluation_names
    tasks = [] if getattr(lifecycle, "task", None) is None else ["task_development", "task_test"]
    return [test.cache_key for test in tests] + extra_evaluation_names(lifecycle) + tasks


def build_cache(teacher, network, lifecycle, device) -> dict:
    """Materialize every cohort a lifecycle reads, truncated for smoke runs."""

    return family(lifecycle.network).build_cache(teacher, network, lifecycle, device)


def check_cache(cache: dict, lifecycle) -> None:
    """Reject caches whose cohorts differ from the lifecycle's data contract."""

    family(lifecycle.network).check_cache(cache, lifecycle)


def to_device(cache: dict, lifecycle, device) -> dict:
    """Move cached tensors once; batch slicing then stays on the device."""

    return family(lifecycle.network).to_device(cache, device)


def cohorts(cache: dict, lifecycle) -> dict:
    return family(lifecycle.network).cohorts(cache)
