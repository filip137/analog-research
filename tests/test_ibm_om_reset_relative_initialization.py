from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ebl.runtime.store import sha256_file
from ebl.runtime.mnist_relu_drn.ibm_om_reset_relative_initialization import (
    derive_full_span_checkpoint,
)
from methods.checkpoint import atomic_torch_save


TEACHER_SHA = "a" * 64


def _source(path: Path) -> Path:
    lower = 0.1020408197973068
    upper = 1.0
    span = upper - lower
    weights = {
        "base.dense_weight.0": lower
        + span * 0.25 * torch.tensor([[0.0, 0.5], [1.0, 0.25]]),
        "base.dense_weight.1": lower
        + span * 0.125 * torch.tensor([[1.0, 0.0], [0.5, 0.25]]),
    }
    candidates = [
        {
            "scale_fractions": [0.25, 0.125],
            "calibration": {"gain": 56.0, "calibrated_kl": 0.001},
            "mapping": {"layers": []},
        },
        {
            "scale_fractions": [1.0, 1.0],
            "calibration": {"gain": 4.5, "calibrated_kl": 0.002},
            "mapping": {"layers": []},
        },
    ]
    atomic_torch_save(
        {
            "schema": "drn.named-weights",
            "schema_version": 1,
            "catalog": [],
            "weights": weights,
            "metadata": {
                "encoding": "single",
                "conductance_bounds_s": [lower, upper],
                "teacher_sha256": TEACHER_SHA,
                "fixed_logit_gain": 56.0,
                "mapping_scale_fraction_pairs": None,
                "mapping": {
                    "candidates": candidates,
                    "selected_index": 0,
                    "selected": candidates[0],
                    "selection_domain": "nominal",
                },
            },
        },
        path,
    )
    return path


def test_full_span_derivation_is_no_update_and_preserves_logical_fractions(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.pt")
    output = tmp_path / "full-span.pt"
    receipt_path = tmp_path / "full-span.receipt.json"

    receipt = derive_full_span_checkpoint(
        source_path=source,
        output_path=output,
        receipt_path=receipt_path,
        expected_source_sha256=sha256_file(source),
        expected_teacher_sha256=TEACHER_SHA,
    )

    payload = torch.load(output, map_location="cpu", weights_only=True)
    lower, upper = payload["metadata"]["conductance_bounds_s"]
    span = upper - lower
    torch.testing.assert_close(
        (payload["weights"]["base.dense_weight.0"] - lower) / span,
        torch.tensor([[0.0, 0.5], [1.0, 0.25]]),
    )
    torch.testing.assert_close(
        (payload["weights"]["base.dense_weight.1"] - lower) / span,
        torch.tensor([[1.0, 0.0], [0.5, 0.25]]),
    )
    assert payload["metadata"]["fixed_logit_gain"] == 4.5
    assert payload["metadata"]["mapping_scale_fraction_pairs"] == [[1.0, 1.0]]
    assert payload["metadata"]["mapping"]["selected_index"] == 1
    assert receipt["optimizer_updates"] == 0
    assert receipt["output_sha256"] == sha256_file(output)
    assert receipt_path.is_file()


def test_full_span_derivation_fails_closed_on_hash_and_overwrite(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.pt")
    output = tmp_path / "full-span.pt"
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="source checkpoint SHA-256"):
        derive_full_span_checkpoint(
            source_path=source,
            output_path=output,
            receipt_path=receipt,
            expected_source_sha256="0" * 64,
            expected_teacher_sha256=TEACHER_SHA,
        )
    derive_full_span_checkpoint(
        source_path=source,
        output_path=output,
        receipt_path=receipt,
        expected_source_sha256=sha256_file(source),
        expected_teacher_sha256=TEACHER_SHA,
    )
    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        derive_full_span_checkpoint(
            source_path=source,
            output_path=output,
            receipt_path=receipt,
            expected_source_sha256=sha256_file(source),
            expected_teacher_sha256=TEACHER_SHA,
        )


def test_full_span_checkpoint_bytes_are_independent_of_output_filename(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.pt")
    digest = sha256_file(source)
    outputs = []
    for index in range(2):
        output = tmp_path / f"full-span-{index}.pt"
        derive_full_span_checkpoint(
            source_path=source,
            output_path=output,
            receipt_path=tmp_path / f"receipt-{index}.json",
            expected_source_sha256=digest,
            expected_teacher_sha256=TEACHER_SHA,
        )
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]
