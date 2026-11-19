"""Opt-in speed settings of the HWA update loop (``hwa.numerics``)."""

import pytest
import torch

from core.contracts import ConfigError
import methods.hwa as hwa
from methods.config import HwaNumerics
import networks.opt as opt
from ebl.schema.lifecycle import parse_lifecycle
from test_workflow_hwa_schedule import document, train

FAST = {"matmul": "tf32", "attention": "sdpa", "fused_adam": True}


def test_numerics_round_trip_and_enter_source_identity():
    plain = parse_lifecycle(document("warmup_step_decay"))
    doc = document("warmup_step_decay")
    doc["hwa"]["numerics"] = FAST
    fast = parse_lifecycle(doc)
    assert plain.hwa.numerics is None and "numerics" not in plain.hwa.to_dict()
    assert fast.hwa.numerics == HwaNumerics("tf32", "sdpa", True)
    assert fast.hwa.to_dict()["numerics"] == FAST
    assert parse_lifecycle({**doc, "hwa": fast.hwa.to_dict()}).hwa == fast.hwa


def test_explicit_fp32_numerics_differ_in_identity_from_omission():
    doc = document("warmup_step_decay")
    doc["hwa"]["numerics"] = {"matmul": "fp32", "attention": "explicit", "fused_adam": False}
    assert parse_lifecycle(doc).hwa.to_dict() != parse_lifecycle(document("warmup_step_decay")).hwa.to_dict()


@pytest.mark.parametrize("field,value", [("matmul", "bf16"), ("attention", "flash"), ("fused_adam", "yes")])
def test_numerics_reject_unknown_values(field, value):
    doc = document("warmup_step_decay")
    doc["hwa"]["numerics"] = {**FAST, field: value}
    with pytest.raises(ConfigError):
        parse_lifecycle(doc)


def test_numerics_require_every_field():
    doc = document("warmup_step_decay")
    doc["hwa"]["numerics"] = {"matmul": "tf32"}
    with pytest.raises(ConfigError):
        parse_lifecycle(doc)


def test_sdpa_attention_matches_the_explicit_reference_and_restores_it():
    torch.manual_seed(0)
    attention = opt.Attention(16, 4)
    x = torch.randn(3, 7, 16)
    reference = attention(x)
    with opt.attention_kernel("sdpa"):
        assert opt.Attention.kernel == "sdpa"
        fast = attention(x)
    assert opt.Attention.kernel == "explicit"
    torch.testing.assert_close(fast, reference, rtol=1e-5, atol=1e-6)


def test_update_scope_restores_fp32_matmuls_even_after_an_error():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError):
        with hwa.update_numerics(HwaNumerics("tf32", "sdpa", False)):
            assert torch.backends.cuda.matmul.allow_tf32 and opt.Attention.kernel == "sdpa"
            raise RuntimeError("stop")
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
    assert opt.Attention.kernel == "explicit"


def test_fast_numerics_train_close_to_the_reference_and_select_in_fp32(monkeypatch, tmp_path):
    seen = []
    original = hwa.selection_score

    def observed(*args, **kwargs):
        seen.append((torch.backends.cuda.matmul.allow_tf32, opt.Attention.kernel))
        return original(*args, **kwargs)

    monkeypatch.setattr(hwa, "selection_score", observed)
    reference, _ = train(document("warmup_step_decay"), tmp_path / "reference")
    doc = document("warmup_step_decay")
    doc["hwa"]["numerics"] = FAST
    fast, _ = train(doc, tmp_path / "fast")
    assert seen and all(state == (False, "explicit") for state in seen)
    for key, value in reference["model"].items():
        torch.testing.assert_close(fast["model"][key], value, rtol=1e-4, atol=1e-5)
