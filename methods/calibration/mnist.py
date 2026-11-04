from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from itertools import product
import math
import torch
import torch.nn.functional as F
from networks.teachers.mnist import BiasFreeReluTeacher
from networks.resistive.mnist import paired_scores, mapped_conductances, apply_targets
from methods.core.engine import EvaluationComponents, evaluate
from methods.core.probes import PerBatchProbe
from methods.teacher_batches import teacher_batches


def fit_positive_logit_gain(
    raw_scores: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    gain_min: float,
    gain_max: float,
    steps: int,
) -> dict[str, Any]:
    scores = raw_scores.detach().to(torch.float64)
    teacher = teacher_logits.detach().to(torch.float64)
    teacher_log_prob = F.log_softmax(teacher, dim=1)
    teacher_prob = teacher_log_prob.exp()

    def kl(gain: torch.Tensor) -> torch.Tensor:
        student_log_prob = F.log_softmax(scores * gain, dim=1)
        return (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=1).mean()

    gains = torch.logspace(
        math.log10(gain_min),
        math.log10(gain_max),
        steps,
        dtype=torch.float64,
        device=scores.device,
    )
    values = torch.stack([kl(gain) for gain in gains])
    index = int(values.argmin().item())
    selected = float(gains[index].item())
    return {
        "gain": selected,
        "raw_kl": float(kl(torch.tensor(1.0, device=scores.device)).item()),
        "calibrated_kl": float(values[index].item()),
        "score_rms": float(torch.sqrt(scores.square().mean()).item()),
        "teacher_logit_rms": float(torch.sqrt(teacher.square().mean()).item()),
        "gain_at_grid_boundary": index in {0, steps - 1},
    }

def collect_calibration(
    stack: Any,
    teacher: BiasFreeReluTeacher | BoundedDrnTeacher,
    loader: Iterable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Settled raw output scores and teacher logits, without a cost target."""

    output_layer = stack.bundle.energy.layers()[-1]
    probe = PerBatchProbe(
        "calibration",
        lambda event: (
            paired_scores(output_layer.state).detach(),
            event.batch.targets.logits.detach(),
        ),
    )
    with torch.no_grad():
        result = evaluate(
            EvaluationComponents(stack.network, None, stack.minimizer),
            teacher_batches(loader, teacher, stack.device),
            probes=(probe,),
        )
    pieces = result.probe_value(probe.name)
    return (
        torch.cat([scores for scores, _logits in pieces]),
        torch.cat([logits for _scores, logits in pieces]),
    )

def select_mapping_and_gain(
    stack: Any,
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    spec: Any,
    *,
    measured_candidate_projector: MeasuredCohortAOptimizer | None = None,
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    teacher_weights = tuple(item.detach() for item in teacher.parameters())
    candidates = []
    selected_targets = None
    selected_key = None
    fraction_pairs = (
        spec.mapping.scale_fraction_pairs
        if spec.mapping.scale_fraction_pairs is not None
        else product(spec.mapping.scale_fractions, repeat=2)
    )
    for fractions in fraction_pairs:
        targets, mapping = mapped_conductances(
            teacher_weights,
            encoding=spec.model.encoding,
            scale_fractions=(float(fractions[0]), float(fractions[1])),
            conductance_min=spec.model.conductance_min,
            conductance_max=spec.model.conductance_max,
            range_placement=spec.mapping.range_placement,
        )
        apply_targets(stack.bundle.catalog, targets)
        raw, teacher_logits = collect_calibration(stack, teacher, loader)
        nominal_calibration = fit_positive_logit_gain(
            raw,
            teacher_logits,
            gain_min=spec.mapping.logit_gain_min,
            gain_max=spec.mapping.logit_gain_max,
            steps=spec.mapping.logit_gain_steps,
        )
        calibration = nominal_calibration
        if measured_candidate_projector is not None:
            projected = measured_candidate_projector.preview_reset_targets()
            with torch.no_grad():
                for binding in stack.bundle.catalog.trainable:
                    binding.state.copy_(projected[binding.key])
            raw, teacher_logits = collect_calibration(stack, teacher, loader)
            calibration = fit_positive_logit_gain(
                raw,
                teacher_logits,
                gain_min=spec.mapping.logit_gain_min,
                gain_max=spec.mapping.logit_gain_max,
                steps=spec.mapping.logit_gain_steps,
            )
        record = {
            "scale_fractions": list(fractions),
            "mapping": mapping,
            "calibration": calibration,
        }
        if measured_candidate_projector is not None:
            record.update(
                {
                    "selection_domain": "measured_projected",
                    "nominal_calibration": nominal_calibration,
                }
            )
        candidates.append(record)
        key = (calibration["calibrated_kl"], fractions)
        if selected_key is None or key < selected_key:
            selected_key = key
            selected_targets = tuple(target.detach().clone() for target in targets)
    if selected_targets is None or selected_key is None:
        raise RuntimeError("Expected at least one mapping candidate.")
    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["calibration"]["calibrated_kl"],
            tuple(candidates[index]["scale_fractions"]),
        ),
    )
    apply_targets(stack.bundle.catalog, selected_targets)
    return {
        "search_examples": spec.mapping.calibration_examples,
        "selection_domain": (
            "measured_projected"
            if measured_candidate_projector is not None
            else "nominal"
        ),
        "candidates": candidates,
        "selected_index": selected_index,
        "selected": candidates[selected_index],
    }, selected_targets
