"""Generate the strict exact-P0 HWA recovery diagnostic canary.

The canary intentionally reuses two byte-pinned P0 bundles produced by the
clean ``1f4ef1b8`` campaign.  It does not regenerate, remap, program, or fault
an array.  The execution launcher authenticates that cross-commit import
before invoking the public ``python -m ebl train`` surface.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "mnist-ibm-om-crossbar-hwa-recovery-canary-20260905-v1"
CONFIG_ROOT = (
    ROOT
    / "examples"
    / "mnist_analog_relu"
    / "ibm_om_crossbar_hwa_recovery_canary"
)
PLAN_PATH = ROOT / "studies" / f"{STUDY_ID}.json"
REFERENCE_PATH = (
    ROOT
    / "studies"
    / "references"
    / "mnist_ibm_om_crossbar_hwa_p0_1f4ef1b8_20260905.json"
)
SHARD_MANIFEST_PATH = (
    ROOT
    / "campaigns"
    / "manifests"
    / "mnist_ibm_om_crossbar_hwa_recovery_canary_dual_host.json"
)
BASE_CONFIG_PATH = (
    ROOT
    / "examples"
    / "mnist_analog_relu"
    / "ibm_om_crossbar_staged_v2"
    / "tuning"
    / "deploy_hwa_master.json"
)

SOURCE_COMMIT = "1f4ef1b8d05b2b0613bf4e8bec4cce799c034cea"
ASSIGNMENT_SEED = 2090402
ENDPOINT_SEED = 2091402
LEARNING_RATES = (0.0, 3e-6, 1e-5, 3e-5, 1e-4, 2e-4, 3e-4)
OBJECTIVES = ("supervised_cross_entropy", "teacher_kl")
START_STATES = ("hwa_healthy_p0", "hwa_published_fault")
PULSE_CAP_PER_CELL = 640
EPOCHS = 3
CHECKPOINT_POLICY = (
    "best_held_apparent_validation_accuracy_then_objective_then_"
    "earlier_epoch_including_epoch0"
)
SELECTION_METRIC = "validation.apparent_forward.student_accuracy"
SELECTION_OBJECTIVE_METRICS = {
    "supervised_cross_entropy": "validation.apparent_forward.cross_entropy",
    "teacher_kl": "validation.apparent_forward.kl_teacher_student",
}
FRESH_DRAW_SEEDS = {
    "hwa_healthy_p0": 2092501,
    "hwa_published_fault": 2092501,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}.")
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_config(root: Path) -> dict[str, Any]:
    path = root / BASE_CONFIG_PATH.relative_to(ROOT)
    value = _read_json(path)
    expected = {
        "schema_version": 2,
        "experiment_id": "mnist_ibm_om_crossbar_relu.v2",
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"Unexpected exact-P0 base config {key}: {value.get(key)!r}.")
    device = value.get("device")
    runtime = value.get("runtime")
    if (
        not isinstance(device, Mapping)
        or device.get("assignment_seed") != ASSIGNMENT_SEED
        or device.get("endpoint_seed") != ENDPOINT_SEED
        or not isinstance(runtime, Mapping)
        or runtime.get("device") != "cuda"
    ):
        raise ValueError("Expected the exact CUDA tuning assignment/write base config.")
    value["evaluation"] = {"profile": "diagnostic_validation_only"}
    return value


def _rate_token(rate: float) -> str:
    return {
        0.0: "0",
        3e-6: "3em6",
        1e-5: "1em5",
        3e-5: "3em5",
        1e-4: "1em4",
        2e-4: "2em4",
        3e-4: "3em4",
    }[float(rate)]


def _state_token(start_state: str) -> str:
    return {
        "hwa_healthy_p0": "healthy",
        "hwa_published_fault": "faulted",
    }[start_state]


def _objective_token(objective: str) -> str:
    return {
        "supervised_cross_entropy": "ce",
        "teacher_kl": "kl",
    }[objective]


def _adam_config(
    base: Mapping[str, Any],
    *,
    start_state: str,
    objective: str,
    learning_rate: float,
) -> dict[str, Any]:
    value = deepcopy(dict(base))
    value["stage"] = {
        "kind": "on_chip_adam_diagnostic",
        "start_state": start_state,
        "epochs": EPOCHS,
        "optimizer": "pulse_adam",
        "objective": objective,
        "repair_examples": 55000,
        "maximum_batches": 3438,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "layer_scope": "all",
        "forward_state": "held_apparent_q",
        "write_state": "persistent_q",
        "gradient_estimator": (
            "identity_ste_apparent_q_to_persistent_pulse_update"
        ),
        "checkpoint_policy": CHECKPOINT_POLICY,
        "hyperparameters": {
            "source": "literal_diagnostic_grid",
            "learning_rate": learning_rate,
            "pulse_cap_per_cell": PULSE_CAP_PER_CELL,
        },
    }
    return value


def _fresh_config(
    base: Mapping[str, Any], *, start_state: str
) -> dict[str, Any]:
    value = deepcopy(dict(base))
    value["stage"] = {
        "kind": "fresh_apparent_diagnostic",
        "start_state": start_state,
        "intervention": (
            "counterfactual_post_write_apparent_noise_redraw_without_device_write"
        ),
        "draws": 4,
        "relative_scale": 1.0,
        "seed": FRESH_DRAW_SEEDS[start_state],
        "mutation_policy": "do_not_mutate_held_apparent_or_persistent_state",
        "interpretation": "diagnostic_only_not_physical_inference_read_noise",
    }
    return value


def _pair_shard(start_state: str, learning_rate: float) -> str:
    """Keep each CE/KL pair together while balancing both states and rates."""

    state_index = START_STATES.index(start_state)
    rate_index = LEARNING_RATES.index(learning_rate)
    return "local" if (state_index + rate_index) % 2 == 0 else "akib"


def _arm_execution_priority(arm_id: str) -> tuple[int, int, int]:
    """Surface the mechanism control and lowest-rate evidence first."""

    if arm_id.startswith("fresh-apparent-"):
        return (0, 0 if arm_id.endswith("healthy") else 1, 0)
    rate_index = next(
        index
        for index, rate in enumerate(LEARNING_RATES)
        if arm_id.endswith(f"-lr{_rate_token(rate)}")
    )
    state_index = 0 if arm_id.startswith("adam-healthy-") else 1
    objective_index = 0 if "-ce-" in arm_id else 1
    return (rate_index + 1, state_index, objective_index)


def _arm(
    arm_id: str,
    description: str,
    config_path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "description": description,
        "experiment_id": "mnist_ibm_om_crossbar_relu.v2",
        "mode": "train",
        "configs": [f"../{config_path.relative_to(root)}"],
    }


def build_documents(root: Path = ROOT) -> tuple[
    dict[Path, dict[str, Any]], dict[str, Any], dict[str, Any]
]:
    """Return configs, study plan, and dual-host manifest without writing."""

    base = _base_config(root)
    config_root = root / CONFIG_ROOT.relative_to(ROOT)
    configs: dict[Path, dict[str, Any]] = {}
    arms: list[dict[str, Any]] = []
    shard_arms: dict[str, list[str]] = {"local": [], "akib": []}

    for start_state in START_STATES:
        state_token = _state_token(start_state)
        for rate in LEARNING_RATES:
            shard = _pair_shard(start_state, rate)
            for objective in OBJECTIVES:
                objective_token = _objective_token(objective)
                arm_id = f"adam-{state_token}-{objective_token}-lr{_rate_token(rate)}"
                path = config_root / f"{arm_id}.json"
                configs[path] = _adam_config(
                    base,
                    start_state=start_state,
                    objective=objective,
                    learning_rate=rate,
                )
                arms.append(
                    _arm(
                        arm_id,
                        (
                            f"Three-epoch {objective_token.upper()} PulseAdam diagnostic "
                            f"from the exact {state_token} HWA P0 at LR={rate:g}; "
                            "epoch zero is eligible for selection."
                        ),
                        path,
                        root=root,
                    )
                )
                shard_arms[shard].append(arm_id)

    for start_state in START_STATES:
        shard = "local"
        state_token = _state_token(start_state)
        arm_id = f"fresh-apparent-{state_token}"
        path = config_root / f"{arm_id}.json"
        configs[path] = _fresh_config(base, start_state=start_state)
        arms.append(
            _arm(
                arm_id,
                (
                    f"Four authenticated counterfactual fresh-apparent draws from the "
                    f"exact {state_token} P0 persistent state, with no device write or "
                    "mutation of the held origin."
                ),
                path,
                root=root,
            )
        )
        shard_arms[shard].append(arm_id)

    reference_path = root / REFERENCE_PATH.relative_to(ROOT)
    reference = _read_json(reference_path)
    if reference.get("source", {}).get("commit") != SOURCE_COMMIT:
        raise ValueError("Expected the immutable 1f4ef1b8 P0 import reference.")

    plan = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "title": "Exact-P0 low-rate HWA recovery and apparent-refresh diagnostic",
        "hypothesis": (
            "For the exact healthy and published-fault HWA P0 states from commit "
            "1f4ef1b8, sufficiently low-rate teacher-KL PulseAdam can preserve or "
            "improve held-apparent validation performance, while hard-label CE and "
            "a no-write fresh-apparent intervention separate optimizer/objective "
            "effects from verification-conditioned apparent-state fragility."
        ),
        "motivation": (
            "The completed ten-epoch campaign used a grid whose minimum LR was 3e-4, "
            "forced a fixed final checkpoint, and changed the HWA objective from "
            "teacher KL to label CE. This diagnostic reuses the exact byte-identical "
            "P0 bundles rather than programming a new realization."
        ),
        "evidence_class": (
            "exploratory_model_based_aihwkit_om_exact_p0_cross_commit_cuda"
        ),
        "arms": arms,
        "completion_criteria": [
            "Exactly 28 Adam diagnostics cover two exact P0 states by two objectives by seven learning rates, plus two separate no-write fresh-apparent diagnostics.",
            "Every run uses CUDA, the same full ordered 55,000-example epoch stream and full 5,000-example validation split, and never evaluates the official test set.",
            "Every native manifest binds teacher SHA-256 7c1f826c..., healthy P0 SHA-256 5e62de40..., or faulted P0 SHA-256 1530c842... as declared by the tracked import reference.",
            "The launcher authenticates the original clean source commit, source manifests/results, artifact registry, HWA ancestry, exact assignment 2090402, and endpoint 2091402 before allowing cross-commit input reuse.",
            "Each Adam arm records epoch zero and epochs one through three, then selects maximum held-apparent validation accuracy, minimum declared objective as the tie-break, and the earlier epoch as the final tie-break; epoch zero is eligible and persistent-state metrics are diagnostics only.",
            "Both fresh-apparent arms run on local CUDA with the same explicit seed stream and shape; each records four counterfactual draws and proves both origin apparent and persistent state hashes remain unchanged.",
            "The CE/KL pair for one state/LR cell runs on the same host; local and Akib each own 14 Adam directories, while local additionally owns both matched fresh-apparent controls.",
        ],
        "analysis_plan": [
            "Report paired epoch-zero, selected, and training-final held-apparent-primary and persistent-secondary accuracy, cross-entropy, and teacher KL for every state/objective/LR cell, with selected epoch and per-epoch, selected, and training-final pulse/cap telemetry.",
            "For each state/LR cell compare CE versus KL on the same host, and compare every selected checkpoint with its own exact epoch-zero state; do not pool host effects into the objective contrast.",
            "Rank learning rates by maximum selected held-apparent accuracy, then selected-minus-epoch-zero declared-objective delta, then lower learning rate; retain host as a blocking label and do not pool unadjusted absolute metrics across hosts.",
            "Use the four no-write fresh-apparent draws only to quantify verification-conditioned snapshot sensitivity; label them counterfactual and not physical inference read noise.",
            "Treat LR=2e-4 as the explicit q-space learning-rate match to the completed DRN HWA protocol.",
            "Treat LR=3e-4 as a stress anchor bridging the previous grid, not as another newly optimized low rate.",
            "Conclude that an update schedule is helpful only if it beats epoch zero without worsening the held-apparent primary metric; report persistent accuracy and KL separately as secondary diagnostics.",
        ],
    }

    plan_sha256 = _sha256_bytes(_json_bytes(plan))
    reference_sha256 = _sha256_file(reference_path)
    config_hashes = {
        arm["arm_id"]: _sha256_bytes(
            _json_bytes(configs[root / arm["configs"][0][3:]])
        )
        for arm in arms
    }
    shard_manifest = {
        "schema": "ebl.ibm_om_crossbar_hwa_recovery_canary_dual_host",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "study_plan": {
            "path": str(PLAN_PATH.relative_to(ROOT)),
            "sha256": plan_sha256,
        },
        "source_reference": {
            "path": str(REFERENCE_PATH.relative_to(ROOT)),
            "sha256": reference_sha256,
            "source_commit": SOURCE_COMMIT,
        },
        "canonical_collection_shard": "akib",
        "split_policy": (
            "CE/KL objective pair for each state/LR cell remains on one host; "
            "state/rate parity compute-balances 14 Adam arms per host; "
            "rate ranking uses held-apparent accuracy first and within-run objective "
            "change second, with host retained as a blocking label"
        ),
        "shards": {
            name: {
                "arm_ids": sorted(arm_ids, key=_arm_execution_priority),
                "config_sha256_by_arm": {
                    arm_id: config_hashes[arm_id] for arm_id in sorted(arm_ids)
                },
                "expected_arms": len(arm_ids),
                "target": (
                    "integnano-akib" if name == "akib" else "local_cuda_host"
                ),
            }
            for name, arm_ids in shard_arms.items()
        },
        "collection_policy": (
            "Copy a whole authenticated canonical arm directory into an empty target "
            "only; refuse overwrite, duplicate completion, undeclared config hashes, "
            "source-plan mismatch, artifact-hash mismatch, cross-host epoch-zero or "
            "ordered-stream mismatch, or fresh-pair RNG/noise mismatch"
        ),
    }
    return configs, plan, shard_manifest


def generate(root: Path = ROOT) -> dict[str, Path]:
    configs, plan, shard_manifest = build_documents(root)
    for path, value in configs.items():
        _write_json(path, value)
    plan_path = root / PLAN_PATH.relative_to(ROOT)
    manifest_path = root / SHARD_MANIFEST_PATH.relative_to(ROOT)
    _write_json(plan_path, plan)
    _write_json(manifest_path, shard_manifest)
    return {
        "config_root": root / CONFIG_ROOT.relative_to(ROOT),
        "plan": plan_path,
        "shard_manifest": manifest_path,
    }


if __name__ == "__main__":
    for role, path in generate().items():
        print(f"{role}: {path}")


__all__ = [
    "ASSIGNMENT_SEED",
    "CHECKPOINT_POLICY",
    "CONFIG_ROOT",
    "ENDPOINT_SEED",
    "EPOCHS",
    "FRESH_DRAW_SEEDS",
    "LEARNING_RATES",
    "OBJECTIVES",
    "PLAN_PATH",
    "PULSE_CAP_PER_CELL",
    "REFERENCE_PATH",
    "SELECTION_METRIC",
    "SELECTION_OBJECTIVE_METRICS",
    "SHARD_MANIFEST_PATH",
    "SOURCE_COMMIT",
    "START_STATES",
    "STUDY_ID",
    "build_documents",
    "generate",
]
