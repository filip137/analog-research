"""ebl.schema.source_fit — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import math


def validate_lora_fit(lifecycle, arm, fit):
    if fit.get("kind") != "lora" or fit.get("base_source", {}).get("sha256") != arm.base_source.sha256:
        raise ValueError("Expected a LoRA source on the declared base.")
    hwa = lifecycle.hwa
    if fit.get("maximum_epochs") != hwa.epochs or fit.get("epochs_run") != hwa.epochs or fit.get("rejection") is not None:
        raise ValueError("Expected the complete declared LoRA epoch budget.")
    history = fit.get("history", [])
    epochs = [e for e in range(hwa.epochs + 1) if e == 0 or e % hwa.selection_every == 0 or e == hwa.epochs]
    if [item.get("epoch") for item in history] != epochs:
        raise ValueError("Expected the complete LoRA checkpoint-selection history.")
    labels = [case.label for case in arm.selection_cases]
    for item in history:
        if [case.get("case") for case in item.get("cases", [])] != labels:
            raise ValueError("LoRA selection cases differ from the declared cases.")
    selected = max(history, key=lambda item: (item["score"], -item["epoch"]))
    if fit.get("selected_epoch") != selected["epoch"] or fit.get("development_score") != selected["score"]:
        raise ValueError("LoRA source does not identify the declared best checkpoint.")


def validate_source_fit(lifecycle, arm, fit):
    """Validate the frozen source's completion and selection before deployment.

    This checks metadata without choosing a different checkpoint or changing an
    optimization decision. It also supports historical epoch-based schedules.
    """
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Expected finite HWA source fit metrics.")
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    finite(fit)
    if not isinstance(fit, dict) or fit.get("status") != "complete" or fit.get("method") != arm.method:
        raise ValueError("Expected a completed HWA source with its declared method.")
    if not arm.trains:
        if fit.get("kind") != "digital_teacher":
            raise ValueError("Expected the unchanged pretrained source.")
        return
    if arm.method == "lora":
        return validate_lora_fit(lifecycle, arm, fit)
    hwa = lifecycle.hwa
    maximum = fit.get("maximum_epochs")
    if (type(maximum) is not int or maximum not in {hwa.epochs, hwa.epochs + hwa.extension_epochs}
            or fit.get("epochs_run") != maximum or fit.get("rejection") is not None):
        raise ValueError("Expected the complete declared HWA epoch budget.")
    history = fit.get("history", [])
    epochs = [epoch for epoch in range(maximum + 1)
              if epoch == 0 or epoch % hwa.selection_every == 0 or epoch in {hwa.epochs, maximum}]
    if not history or [item.get("epoch") for item in history] != epochs:
        raise ValueError("Expected the complete HWA checkpoint-selection history.")
    labels = [case.label for case in arm.selection_cases]
    for item in history:
        cases = item.get("cases", [])
        if [case.get("case") for case in cases] != labels:
            raise ValueError("HWA selection cases differ from the declared cases.")
        score = sum(case["teacher_kl"] for case in cases) / len(labels)
        if not math.isclose(score, item["teacher_kl"], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("HWA selection score differs from its case mean.")
    selected = min(history, key=lambda item: item["teacher_kl"])
    if fit.get("selected_epoch") != selected["epoch"] or fit.get("development_kl") != selected["teacher_kl"]:
        raise ValueError("HWA source does not identify the declared best checkpoint.")
    schedule = hwa.lr_schedule
    if schedule is not None:
        recorded = fit.get("lr_schedule", {})
        bases = [hwa.learning_rate] + ([arm.digital_learning_rate] if arm.scope == "model" else [])
        if (any(recorded.get(key) != value for key, value in schedule.to_dict().items())
                or recorded.get("steps_run") != schedule.total_steps
                or recorded.get("base_learning_rates") != bases
                or recorded.get("final_learning_rates") != [rate * schedule.factor(schedule.total_steps) for rate in bases]):
            raise ValueError("HWA source schedule or completed update budget differs from its declaration.")
        per_epoch = schedule.total_steps // hwa.epochs
        for item in history:
            step = item["epoch"] * per_epoch
            if (item.get("global_step") != step
                    or item.get("learning_rates") != [rate * schedule.factor(step) for rate in bases]):
                raise ValueError("HWA selection checkpoint/update schedule mismatch.")
