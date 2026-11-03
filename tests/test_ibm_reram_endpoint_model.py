from __future__ import annotations

import pytest
import torch

from hardware.devices.endpoints import (
    build_ibm_reram_accepted_endpoint_model,
    sample_ibm_reram_endpoints,
)


CONDITION_KEY = "one_pulse__lower_to_target__tau_step_0.5"


def _quantiles(value: float) -> dict[str, object]:
    return {
        "count": 10,
        "probabilities": [0.0, 0.5, 1.0],
        "values": [value, value, value],
    }


def _artifact(
    *,
    corrupt_fraction: float = 0.0,
    noncorrupt_success_probability: float = 1.0,
    failed_endpoint: float = 0.25,
) -> dict[str, object]:
    records = []
    for target, probabilities in (
        (0.0, [1.0, 0.0]),
        (1.0, [0.0, 1.0]),
    ):
        reachability_classes = {}
        for reachability_class in (
            "target_below_lower_bound",
            "target_inside_bounds",
            "target_above_upper_bound",
        ):
            active = reachability_class == "target_inside_bounds"
            reachability_classes[reachability_class] = {
                "probability": 1.0 if active else 0.0,
                "acceptance_window_reachable_probability": (
                    1.0 if active else None
                ),
                "success_probability": (
                    noncorrupt_success_probability if active else None
                ),
                "accepted_terminal": {
                    "apparent_endpoint": _quantiles(target),
                    "persistent_endpoint": _quantiles(target),
                },
                "failed_terminal": {
                    "apparent_endpoint": _quantiles(failed_endpoint),
                    "persistent_endpoint": _quantiles(failed_endpoint),
                },
            }
        records.append(
            {
                "target": target,
                "tolerance": 0.1,
                "accepted_noncorrupt_residual": {
                    "fit_count": 10,
                    "support": [-0.1, 0.1],
                    "bin_edges": [-0.1, 0.0, 0.1],
                    # The production fitter uses a positive Jeffreys
                    # pseudocount. Keep that contract in this small fixture.
                    "bin_probabilities": [
                        max(probabilities[0], 1e-6),
                        max(probabilities[1], 1e-6),
                    ],
                },
                "outcome_model": {
                    "corrupt_identity_fraction": corrupt_fraction,
                    "noncorrupt_success_probability": (
                        noncorrupt_success_probability
                    ),
                    "noncorrupt_reachability": {
                        "available": True,
                        "classes": reachability_classes,
                    },
                    "failed_noncorrupt_terminal": {
                        "apparent_endpoint": _quantiles(failed_endpoint),
                        "persistent_endpoint": _quantiles(failed_endpoint),
                    },
                    "corrupt_terminal": {
                        "apparent_endpoint": _quantiles(0.55),
                        "persistent_endpoint": _quantiles(0.5),
                    },
                },
            }
        )
    return {
        "schema": "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model",
        "schema_version": 2,
        "conditions": {
            CONDITION_KEY: {
                "fit_status": "fit",
                "reachability_fit_status": "fit",
                "adequate": True,
                "validation": {"per_target": records},
            },
        },
    }


def _select_reachability_class(
    record: dict[str, object],
    *,
    reachability_class: str,
    success_probability: float,
    acceptance_window_reachable_probability: float,
    apparent_endpoint: float,
    persistent_endpoint: float,
) -> None:
    outcome = record["outcome_model"]
    assert isinstance(outcome, dict)
    reachability = outcome["noncorrupt_reachability"]
    assert isinstance(reachability, dict)
    classes = reachability["classes"]
    assert isinstance(classes, dict)
    for name, value in classes.items():
        assert isinstance(value, dict)
        active = name == reachability_class
        value["probability"] = 1.0 if active else 0.0
        value["success_probability"] = (
            success_probability if active else None
        )
        value["acceptance_window_reachable_probability"] = (
            acceptance_window_reachable_probability if active else None
        )
        if active:
            value["accepted_terminal"] = {
                "apparent_endpoint": _quantiles(apparent_endpoint),
                "persistent_endpoint": _quantiles(persistent_endpoint),
            }
            value["failed_terminal"] = {
                "apparent_endpoint": _quantiles(apparent_endpoint),
                "persistent_endpoint": _quantiles(persistent_endpoint),
            }


def test_success_sampler_interpolates_target_histograms_inside_verify_window() -> None:
    targets = torch.full((8_000,), 0.5)
    sample = sample_ibm_reram_endpoints(
        targets,
        _artifact(),
        condition_key=CONDITION_KEY,
        generator=torch.Generator().manual_seed(41),
    )

    assert bool(torch.all(sample.success_noncorrupt))
    assert not bool(torch.any(sample.failed_noncorrupt))
    assert not bool(torch.any(sample.corrupt))
    assert bool(torch.all(sample.accepted))
    assert bool(torch.all(torch.abs(sample.raw_endpoint - targets) <= 0.1))
    # At the midpoint the two endpoint histograms are mixed equally.
    below_midpoint = (sample.raw_endpoint < 0.5).to(torch.float32).mean()
    assert 0.47 < float(below_midpoint) < 0.53


def test_compiled_accepted_endpoint_model_replays_global_hwa_draws() -> None:
    model = build_ibm_reram_accepted_endpoint_model(
        _artifact(corrupt_fraction=1.0),
        condition_key=CONDITION_KEY,
    ).to("cpu", dtype=torch.float32)
    targets = torch.full((8_000,), 0.5, dtype=torch.float32)

    first = model.sample(
        targets,
        generator=torch.Generator(device="cpu").manual_seed(141),
    )
    second = model.sample(
        targets,
        generator=torch.Generator(device="cpu").manual_seed(141),
    )

    assert len(model.fingerprint) == 64
    assert torch.equal(first.residual_x, second.residual_x)
    assert torch.equal(first.apparent_x, second.apparent_x)
    assert torch.equal(first.apparent_x, first.target_x + first.residual_x)
    below_midpoint = (first.apparent_x < 0.5).to(torch.float32).mean()
    assert 0.47 < float(below_midpoint) < 0.53


def test_compiled_accepted_endpoint_model_is_healthy_only_and_fail_closed() -> None:
    artifact = _artifact(
        corrupt_fraction=1.0,
        noncorrupt_success_probability=0.0,
        failed_endpoint=0.9,
    )
    model = build_ibm_reram_accepted_endpoint_model(
        artifact,
        condition_key=CONDITION_KEY,
    ).to("cpu", dtype=torch.float32)

    # The HWA model intentionally compiles only accepted non-corrupt residual
    # rows; deployment corruption and failure branches cannot leak into it.
    sample = model.sample(
        torch.tensor([0.0, 1.0], dtype=torch.float32),
        generator=torch.Generator(device="cpu").manual_seed(142),
    )
    assert bool(torch.all(torch.abs(sample.residual_x) <= 0.1))

    with pytest.raises(ValueError, match="characterized global x range"):
        model.sample(
            torch.tensor([-0.01, 1.01], dtype=torch.float32),
            generator=torch.Generator(device="cpu").manual_seed(143),
        )

    artifact["conditions"][CONDITION_KEY]["adequate"] = False
    with pytest.raises(ValueError, match="held-out adequacy"):
        build_ibm_reram_accepted_endpoint_model(
            artifact,
            condition_key=CONDITION_KEY,
        )


def test_out_of_support_targets_fail_closed_or_are_explicitly_clamped() -> None:
    targets = torch.tensor([-0.2, 0.4, 1.2])
    with pytest.raises(ValueError, match="inside the characterized interval"):
        sample_ibm_reram_endpoints(
            targets,
            _artifact(),
            condition_key=CONDITION_KEY,
            generator=torch.Generator().manual_seed(42),
        )

    sample = sample_ibm_reram_endpoints(
        targets,
        _artifact(),
        condition_key=CONDITION_KEY,
        generator=torch.Generator().manual_seed(43),
        target_out_of_support="clamp",
    )
    assert torch.equal(sample.modeled_target, torch.tensor([0.0, 0.4, 1.0]))
    assert sample.target_was_clamped.tolist() == [True, False, True]


def test_noncorrupt_failures_use_the_separate_terminal_endpoint_branch() -> None:
    sample = sample_ibm_reram_endpoints(
        torch.tensor([0.1, 0.9]),
        _artifact(noncorrupt_success_probability=0.0),
        condition_key=CONDITION_KEY,
        generator=torch.Generator().manual_seed(44),
    )

    assert sample.failed_noncorrupt.tolist() == [True, True]
    assert sample.accepted.tolist() == [False, False]
    assert torch.equal(sample.raw_endpoint, torch.tensor([0.25, 0.25]))


def test_corrupt_mask_is_reused_and_samples_only_the_stuck_outcome_branch() -> None:
    targets = torch.tensor([0.0, 0.55, 1.0])
    corrupt_mask = torch.tensor([True, False, True])
    sample = sample_ibm_reram_endpoints(
        targets,
        _artifact(corrupt_fraction=1.0),
        condition_key=CONDITION_KEY,
        generator=torch.Generator().manual_seed(45),
        corrupt_mask=corrupt_mask,
    )

    assert torch.equal(sample.corrupt, corrupt_mask)
    assert torch.equal(sample.raw_endpoint[corrupt_mask], torch.tensor([0.55, 0.55]))
    assert sample.accepted.tolist() == [False, True, False]


def test_endpoint_clipping_is_explicit_and_reported() -> None:
    sample = sample_ibm_reram_endpoints(
        torch.tensor([0.0]),
        _artifact(
            noncorrupt_success_probability=0.0,
            failed_endpoint=1.25,
        ),
        condition_key=CONDITION_KEY,
        generator=torch.Generator().manual_seed(46),
        endpoint_policy="clip_0_1",
    )
    assert sample.raw_endpoint.tolist() == [1.25]
    assert sample.endpoint.tolist() == [1.0]
    assert sample.endpoint_was_clipped.tolist() == [True]


def test_endpoint_sampler_rejects_incomplete_or_inadequate_fits() -> None:
    artifact = _artifact()
    condition = artifact["conditions"][CONDITION_KEY]
    condition["adequate"] = False

    with pytest.raises(ValueError, match="held-out adequacy"):
        sample_ibm_reram_endpoints(
            torch.tensor([0.5]),
            artifact,
            condition_key=CONDITION_KEY,
            generator=torch.Generator().manual_seed(50),
        )

    sample = sample_ibm_reram_endpoints(
        torch.tensor([0.5]),
        artifact,
        condition_key=CONDITION_KEY,
        generator=torch.Generator().manual_seed(50),
        allow_inadequate=True,
    )
    assert bool(torch.all(torch.isfinite(sample.endpoint)))

    condition["fit_status"] = "incomplete_accepted_target_bins"
    with pytest.raises(ValueError, match="complete accepted-residual"):
        sample_ibm_reram_endpoints(
            torch.tensor([0.5]),
            artifact,
            condition_key=CONDITION_KEY,
            generator=torch.Generator().manual_seed(51),
            allow_inadequate=True,
        )


def test_endpoint_sampler_rejects_pseudocount_only_target_bins() -> None:
    artifact = _artifact()
    records = artifact["conditions"][CONDITION_KEY]["validation"]["per_target"]
    records[0]["accepted_noncorrupt_residual"]["fit_count"] = 0

    with pytest.raises(ValueError, match="fit observations"):
        sample_ibm_reram_endpoints(
            torch.tensor([0.0]),
            artifact,
            condition_key=CONDITION_KEY,
            generator=torch.Generator().manual_seed(52),
        )


def test_bound_unreachable_targets_use_separate_lower_and_upper_branches() -> None:
    artifact = _artifact(noncorrupt_success_probability=0.0)
    records = artifact["conditions"][CONDITION_KEY]["validation"]["per_target"]
    _select_reachability_class(
        records[0],
        reachability_class="target_below_lower_bound",
        success_probability=0.0,
        acceptance_window_reachable_probability=0.0,
        apparent_endpoint=0.3,
        persistent_endpoint=0.2,
    )
    _select_reachability_class(
        records[1],
        reachability_class="target_above_upper_bound",
        success_probability=0.0,
        acceptance_window_reachable_probability=0.0,
        apparent_endpoint=0.7,
        persistent_endpoint=0.8,
    )

    sample = sample_ibm_reram_endpoints(
        torch.tensor([0.0, 1.0]),
        artifact,
        condition_key=CONDITION_KEY,
        generator=torch.Generator().manual_seed(48),
    )

    assert sample.target_below_lower_bound.tolist() == [True, False]
    assert sample.target_inside_bounds.tolist() == [False, False]
    assert sample.target_above_upper_bound.tolist() == [False, True]
    assert sample.acceptance_window_reachable.tolist() == [False, False]
    assert sample.failed_noncorrupt.tolist() == [True, True]
    assert torch.equal(sample.raw_endpoint, torch.tensor([0.3, 0.7]))
    assert torch.equal(sample.persistent_endpoint, torch.tensor([0.2, 0.8]))


def test_apparent_success_does_not_make_an_unreachable_target_persistent() -> None:
    artifact = _artifact()
    records = artifact["conditions"][CONDITION_KEY]["validation"]["per_target"]
    _select_reachability_class(
        records[0],
        reachability_class="target_below_lower_bound",
        success_probability=1.0,
        acceptance_window_reachable_probability=0.0,
        apparent_endpoint=0.0,
        persistent_endpoint=0.25,
    )

    sample = sample_ibm_reram_endpoints(
        torch.zeros(256),
        artifact,
        condition_key=CONDITION_KEY,
        generator=torch.Generator().manual_seed(49),
    )

    assert bool(torch.all(sample.accepted))
    assert bool(torch.all(sample.target_below_lower_bound))
    assert not bool(torch.any(sample.acceptance_window_reachable))
    assert bool(torch.all(torch.abs(sample.raw_endpoint) <= 0.1))
    assert torch.equal(sample.persistent_endpoint, torch.full((256,), 0.25))


def test_fixed_identity_bounds_override_marginal_reachability_sampling() -> None:
    artifact = _artifact()
    records = artifact["conditions"][CONDITION_KEY]["validation"]["per_target"]
    for record in records:
        _select_reachability_class(
            record,
            reachability_class="target_below_lower_bound",
            success_probability=0.0,
            acceptance_window_reachable_probability=0.0,
            apparent_endpoint=0.3,
            persistent_endpoint=0.3,
        )
    sample = sample_ibm_reram_endpoints(
        torch.tensor([0.2]),
        artifact,
        condition_key=CONDITION_KEY,
        generator=torch.Generator().manual_seed(50),
        corrupt_mask=torch.tensor([False]),
        lower_bound=torch.tensor([0.29]),
        upper_bound=torch.tensor([0.8]),
    )

    assert sample.target_below_lower_bound.tolist() == [True]
    assert sample.target_inside_bounds.tolist() == [False]
    assert sample.acceptance_window_reachable.tolist() == [True]
    assert sample.failed_noncorrupt.tolist() == [True]
    assert sample.persistent_endpoint.tolist() == pytest.approx([0.3])


def test_fixed_identity_bounds_limit_the_sampled_persistent_terminal() -> None:
    artifact = _artifact()
    records = artifact["conditions"][CONDITION_KEY]["validation"]["per_target"]
    for record in records:
        _select_reachability_class(
            record,
            reachability_class="target_below_lower_bound",
            success_probability=1.0,
            acceptance_window_reachable_probability=0.0,
            apparent_endpoint=0.6,
            persistent_endpoint=0.2,
        )

    sample = sample_ibm_reram_endpoints(
        torch.tensor([0.6]),
        artifact,
        condition_key=CONDITION_KEY,
        generator=torch.Generator().manual_seed(52),
        corrupt_mask=torch.tensor([False]),
        lower_bound=torch.tensor([0.7]),
        upper_bound=torch.tensor([0.9]),
    )

    assert sample.accepted.tolist() == [True]
    assert sample.target_below_lower_bound.tolist() == [True]
    assert abs(float(sample.raw_endpoint.item()) - 0.6) <= 0.1
    assert sample.persistent_endpoint.tolist() == pytest.approx([0.7])


def test_fixed_corrupt_identity_preserves_its_stuck_endpoint() -> None:
    sample = sample_ibm_reram_endpoints(
        torch.tensor([0.2, 0.8]),
        _artifact(corrupt_fraction=1.0),
        condition_key=CONDITION_KEY,
        generator=torch.Generator().manual_seed(51),
        corrupt_mask=torch.tensor([True, True]),
        lower_bound=torch.tensor([0.4, 0.6]),
        upper_bound=torch.tensor([0.4, 0.6]),
        corrupt_apparent_endpoint=torch.tensor([0.41, 0.59]),
        corrupt_persistent_endpoint=torch.tensor([0.4, 0.6]),
    )

    assert torch.equal(sample.raw_endpoint, torch.tensor([0.41, 0.59]))
    assert torch.equal(sample.persistent_endpoint, torch.tensor([0.4, 0.6]))
    assert sample.accepted.tolist() == [False, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_generator_can_drive_cuda_endpoint_sampling() -> None:
    targets = torch.tensor([0.2, 0.8], device="cuda")
    sample = sample_ibm_reram_endpoints(
        targets,
        _artifact(),
        condition_key=CONDITION_KEY,
        generator=torch.Generator(device="cpu").manual_seed(47),
    )

    assert sample.endpoint.device.type == "cuda"
    assert bool(torch.all(sample.success_noncorrupt))
