from contextlib import nullcontext

import pytest

from training.core.batch import Batch, as_batch
from training.core.modifier import (
    NO_OP_PARAMETER_MODIFIER,
    ParameterModifier,
)


def test_batch_adapter_accepts_pairs_triples_and_existing_batches():
    pair = as_batch((["x0", "x1"], [0, 1]))
    assert pair.inputs == ["x0", "x1"]
    assert pair.targets == [0, 1]
    assert pair.indices is None
    assert pair.example_count == 2

    triple = as_batch([["x2"], [2], [7]])
    assert triple.inputs == ["x2"]
    assert triple.targets == [2]
    assert triple.indices == [7]
    assert triple.example_count == 1

    canonical = Batch(inputs="input", targets="target")
    assert as_batch(canonical) is canonical
    assert Batch.from_raw(canonical) is canonical


@pytest.mark.parametrize("raw_batch", [{"x": 1, "y": 2}, (1,), (1, 2, 3, 4)])
def test_batch_adapter_rejects_unknown_shapes(raw_batch):
    with pytest.raises(
        ValueError,
        match="Expected a batch",
    ):
        as_batch(raw_batch)


def test_batch_count_prefers_indices_over_ambiguous_inputs():
    batch = Batch(
        inputs=("left_input", "right_input"),
        targets=1,
        indices=[4, 9, 12],
    )
    assert batch.example_count == 3


def test_no_op_modifier_implements_the_structural_contract():
    assert isinstance(NO_OP_PARAMETER_MODIFIER, ParameterModifier)

    with NO_OP_PARAMETER_MODIFIER.training_context() as active:
        assert active is NO_OP_PARAMETER_MODIFIER
    with NO_OP_PARAMETER_MODIFIER.evaluation_context() as active:
        assert active is NO_OP_PARAMETER_MODIFIER

    assert NO_OP_PARAMETER_MODIFIER.state_dict() == {}
    NO_OP_PARAMETER_MODIFIER.load_state_dict({})

    with pytest.raises(ValueError, match="Expected the no-op"):
        NO_OP_PARAMETER_MODIFIER.load_state_dict({"unexpected": True})


class _StructurallyCompatibleModifier:
    def training_context(self):
        return nullcontext(self)

    def evaluation_context(self):
        return nullcontext(self)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


def test_modifier_protocol_accepts_future_policy_implementations():
    assert isinstance(_StructurallyCompatibleModifier(), ParameterModifier)
