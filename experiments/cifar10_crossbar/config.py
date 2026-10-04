"""Strict, CPU-only config parsing for the staged CIFAR-10 comparison."""
from experiments.schema import RunMode, config_error, freeze_json

EXPERIMENT_ID = "cifar10_ibm_om_crossbar.v1"


def parse_config(raw):
    required = {"schema_version", "experiment_id", "stage", "seed", "data_seed", "data_root", "dims", "batch_size", "epochs", "learning_rate", "weight_decay", "maximum_batches", "evaluation_limit", "settings"}
    if set(raw) != required:
        raise config_error("config", f"to contain exactly {sorted(required)}", sorted(raw))
    if raw["experiment_id"] != EXPERIMENT_ID or raw["schema_version"] != 1:
        raise config_error("config", f"to use {EXPERIMENT_ID} schema 1", raw)
    if raw["stage"] not in {"pretrain", "hwa", "deploy", "recover"}:
        raise config_error("config.stage", "to be pretrain, hwa, deploy or recover", raw["stage"])
    if tuple(raw["dims"]) != (3072, 256, 10):
        raise config_error("config.dims", "to be [3072,256,10]", raw["dims"])
    for key in ("seed", "data_seed", "batch_size", "epochs"):
        minimum = 0 if key in {"seed", "data_seed"} else 1
        if type(raw[key]) is not int or raw[key] < minimum:
            raise config_error(key, f"to be an integer >= {minimum}", raw[key])
    import math
    for key in ("learning_rate", "weight_decay"):
        if type(raw[key]) not in (int, float) or not math.isfinite(raw[key]) or raw[key] < 0:
            raise config_error(key, "to be finite and nonnegative", raw[key])
    for key in ("maximum_batches", "evaluation_limit"):
        if raw[key] is not None and (type(raw[key]) is not int or raw[key] < 1):
            raise config_error(key, "to be null or a positive integer", raw[key])
    settings_keys = {
        "pretrain": {"minimum_validation_accuracy", "minimum_lr_factor"},
        "hwa": {"noise_multiplier", "training_assignment", "development_assignment", "development_endpoint"},
        "deploy": {"assignment", "endpoint", "source"},
        "recover": {"condition", "writer", "schedule", "verify_tolerance_q", "maximum_verify_pulses", "total_pulse_cap"},
    }
    if set(raw["settings"]) != settings_keys[raw["stage"]]:
        raise config_error("settings", f"to contain {sorted(settings_keys[raw['stage']])}", raw["settings"])
    s = raw["settings"]
    if raw["stage"] == "pretrain":
        if not 0 <= s["minimum_validation_accuracy"] <= 1 or not 0 < s["minimum_lr_factor"] <= 1:
            raise config_error("settings", "to use valid accuracy and LR factors", s)
    if raw["stage"] == "hwa" and s["noise_multiplier"] not in (0, 0.25, 0.5, 1):
        raise config_error("settings.noise_multiplier", "to be 0, 0.25, 0.5 or 1", s["noise_multiplier"])
    if raw["stage"] == "deploy" and s["source"] not in {"direct", "hwa"}:
        raise config_error("settings.source", "to be direct or hwa", s["source"])
    if raw["stage"] == "recover":
        if s["condition"] not in {"healthy", "faulted"} or s["writer"] not in {"open_loop", "closed_loop_pv"} or s["schedule"] not in {"constant", "exponential"}:
            raise config_error("settings", "to name a supported condition, writer and schedule", s)
        for key in ("maximum_verify_pulses", "total_pulse_cap"):
            if type(s[key]) is not int or s[key] < 1:
                raise config_error(key, "to be a positive integer", s[key])
        if not math.isfinite(s["verify_tolerance_q"]) or s["verify_tolerance_q"] <= 0:
            raise config_error("verify_tolerance_q", "to be finite and positive", s["verify_tolerance_q"])
    for key in ("training_assignment", "development_assignment", "development_endpoint", "assignment", "endpoint"):
        if key in s and (type(s[key]) is not int or s[key] < 0):
            raise config_error(key, "to be a nonnegative integer", s[key])
    return freeze_json(raw)


def resolve_config(document, mode):
    if mode != RunMode.TRAIN:
        raise config_error("mode", "to be train", mode)
    return document
