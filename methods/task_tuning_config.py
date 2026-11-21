"""Native fault-tuning configuration for downstream task adaptation."""
from dataclasses import dataclass

from methods import fault_tuning as ft
from methods.drift_tuning import parse_mapping


@dataclass(frozen=True)
class TaskTuning:
    source_sha256: str
    assignment_seed: int
    mapping: dict
    fault_rewrite: dict
    task_adaptation: dict
    deployment = None
    analog_lora = None
    drift = None
    evaluation_cohorts = ()
    deployment_mapping = ft.FaultTuning.deployment_mapping
    rewrite = ft.FaultTuning.rewrite

    @classmethod
    def parse(cls, raw):
        ft._exact(raw, ("source_sha256", "assignment_seed", "mapping", "fault_rewrite", "task_adaptation"),
                  "fault_tuning")
        mapping = parse_mapping(raw["mapping"])
        if mapping["kind"] != "two_sided":
            raise ValueError("Task adaptation requires a two-sided crossbar.")
        rule = raw["fault_rewrite"]
        if rule != {"seed": 241016, "roles": "pair", "detection": "verify", "threshold_sigma": 5.0}:
            raise ValueError("Expected nominal fault-aware verify and partner rewrite.")
        spec = raw["task_adaptation"]
        ft._exact(spec, ("task", "retention", "phase", "selection", "proxy_seed", "seed", "rates",
                         "steps", "fisher_items", "microbatch", "eval_limit", "routes"), "task_adaptation")
        phase = spec["phase"]
        if phase not in ("canary", "development", "evaluation"):
            raise ValueError("Unknown task adaptation phase.")
        for name in ("task", "retention", "selection"):
            pin = spec[name]
            if pin is None and name == "selection" and phase != "evaluation":
                continue
            ft._exact(pin, ("name", "sha256"), name)
            if not isinstance(pin["name"], str) or not pin["name"] or any(c in pin["name"] for c in "/\\"):
                raise ValueError(f"Expected plain artifact name for {name}.")
            ft._digest(pin["sha256"], name)
        for name in ("steps", "fisher_items", "microbatch", "proxy_seed", "seed"):
            ft._int(spec[name], name, 1)
        if spec["eval_limit"] is not None:
            ft._int(spec["eval_limit"], "eval_limit", 1)
        if phase != "canary" and (spec["steps"] != 128 or spec["fisher_items"] != 1024 or spec["eval_limit"] is not None):
            raise ValueError("A scientific task run requires the complete frozen cohort and schedule.")
        ft._exact(spec["rates"], ("lora", "child", "full"), "rates")
        for name, center in spec["rates"].items():
            ft._float(center, name)
        routes = spec["routes"]
        if not isinstance(routes, list) or not routes or len(set(routes)) != len(routes) or set(routes) - set(ROUTES):
            raise ValueError("Invalid task adaptation routes.")
        if spec["proxy_seed"] == raw["assignment_seed"]:
            raise ValueError("Evaluation and independent proxy assignment must differ.")
        return cls(ft._digest(raw["source_sha256"], "source"), ft._int(raw["assignment_seed"], "seed", 1),
                   mapping, dict(rule), dict(spec))

    def to_dict(self):
        return dict(self.__dict__)


ROUTES = ("offline_lora", "onchip_lora", "offline_child01", "onchip_child01",
          "offline_child10", "onchip_child10", "offline_full")
