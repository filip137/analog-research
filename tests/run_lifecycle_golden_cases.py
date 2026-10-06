"""Golden cases for moving every ``run_train`` onto one phase runner.

Each case runs ``python -m ebl train`` (in process, through ``ebl.cli.main``)
on a tiny CPU variant of a checked-in example config and records the complete
run bundle: every ``metrics.jsonl`` record, ``result.json`` metrics and
artifact kinds, and the decoded contents of every checkpoint the run wrote.
A change in phase order, RNG consumption, selection, checkpoint content or
any floating-point expression therefore changes the record.

Wall-clock fields, run identities and temporary paths are normalized; file
hashes are dropped because the decoded checkpoint contents are compared
directly.

Run ``python tests/run_lifecycle_golden_cases.py capture`` on the reference
commit to (re)write ``tests/golden/run_lifecycle``;
``tests/test_run_lifecycle_golden.py`` replays and compares bit-exactly.
Real MNIST (``EBL_MNIST_ROOT``) and the measured ReRAM traces
(``EBL_DEVICE_DATA``) are required; cases skip without them.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ibm_om_golden_cases as om  # noqa: E402

GOLDEN_DIR = ROOT / "tests" / "golden" / "run_lifecycle"
assert_same = om.assert_same
environment = om.environment

DEVICE_DATA = Path(
    os.environ.get(
        "EBL_DEVICE_DATA", "/home/filip/reram_data/march_slope_x3_5k.hdf5"
    )
)
MNIST_ROOT = Path(
    os.environ.get("EBL_MNIST_ROOT", str(Path.home() / "datasets" / "mnist"))
)

# Keys whose values are wall-clock or per-invocation identities.
_VOLATILE_KEYS = frozenset(
    {
        "created_at",
        "finished_at",
        "started_at",
        "duration_seconds",
        "run_id",
        "sha256_file",
    }
)


def inputs_available() -> bool:
    return DEVICE_DATA.is_file() and MNIST_ROOT.is_dir()


# --------------------------------------------------------------------------
# Config and invocation helpers
# --------------------------------------------------------------------------


def _example(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / "examples" / relative).read_text(encoding="utf-8"))


def _cpu(payload: dict[str, Any]) -> dict[str, Any]:
    payload["runtime"]["device"] = "cpu"
    return payload


def _write(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _train(directory: Path, config: Path, *extra: str) -> Path:
    from ebl.cli import main

    output = directory / "runs"
    output.mkdir(exist_ok=True)
    before = set(output.iterdir())
    errors = io.StringIO()
    code = main(
        ["train", "--config", str(config), "--output-dir", str(output), *extra],
        stdout=io.StringIO(),
        stderr=errors,
    )
    created = sorted(set(output.iterdir()) - before)
    if code != 0 or len(created) != 1:
        raise RuntimeError(
            f"train exited {code}; created {created!r}; {errors.getvalue()}"
        )
    return created[0]


# --------------------------------------------------------------------------
# Bundle normalization
# --------------------------------------------------------------------------


def _normalize(value: Any, substitutions: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<volatile>"
                if key in _VOLATILE_KEYS
                else _normalize(item, substitutions)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        normalized = [_normalize(item, substitutions) for item in value]
        return normalized if isinstance(value, list) else tuple(normalized)
    if isinstance(value, str):
        for old, new in substitutions.items():
            value = value.replace(old, new)
        return value
    return value


def substitutions(
    roots: tuple[str, ...], generated: dict[str, Path]
) -> dict[str, str]:
    """Map temporary roots and generated-file hashes to stable names.

    ``torch.save`` embeds a random serialization id, so a checkpoint's file
    hash differs between identical runs; its decoded contents do not.
    """

    from experiments.artifacts import sha256_file

    mapping = {sha256_file(path): f"<sha256:{name}>" for name, path in generated.items()}
    mapping.update({root: f"<root{index}>" for index, root in enumerate(roots)})
    return mapping


# Tensors above this size are stored as an exact digest of their bytes so
# the goldens stay small; smaller tensors stay in full for readable diffs.
DIGEST_NUMEL = 4096


def digest_tensors(value: Any) -> Any:
    """Replace large tensors by ``{dtype, shape, sha256 of raw bytes}``."""

    import hashlib

    if isinstance(value, torch.Tensor):
        if value.numel() <= DIGEST_NUMEL:
            return value
        raw = value.detach().cpu().contiguous().reshape(-1)
        data = raw.view(torch.uint8) if raw.dtype != torch.bool else raw.to(torch.uint8)
        return {
            "tensor_dtype": str(value.dtype),
            "tensor_shape": list(value.shape),
            "tensor_sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {key: digest_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [digest_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(digest_tensors(item) for item in value)
    return value


def _artifact(record: dict[str, Any]) -> dict[str, Any]:
    # Content hashes cover metadata that embeds temporary input paths; the
    # decoded checkpoint is compared instead.
    return {"path": record["path"], "kind": record["kind"]}


def _decode_artifact(path: Path) -> Any:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    return path.read_bytes()


def bundle(run_dir: Path, mapping: dict[str, str]) -> dict[str, Any]:
    """Return the normalized, comparable record of one run bundle."""

    mapping = {str(run_dir): "<run>", **mapping}
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    checkpoints = {
        path.relative_to(run_dir).as_posix(): torch.load(
            path, map_location="cpu", weights_only=False
        )
        for path in sorted((run_dir / "checkpoints").glob("*.pt"))
    }
    artifacts = {
        path.relative_to(run_dir).as_posix(): _decode_artifact(path)
        for path in sorted((run_dir / "artifacts").glob("**/*"))
        if path.is_file()
    }
    return _normalize(
        {
            "status": result["status"],
            "metrics_jsonl": metrics,
            "result_metrics": result["metrics"],
            "result_artifacts": [_artifact(item) for item in result["artifacts"]],
            "artifacts": artifacts,
            "checkpoints": checkpoints,
        },
        mapping,
    )


# --------------------------------------------------------------------------
# Chained runs
# --------------------------------------------------------------------------


class Session:
    """One temporary workspace whose runs may feed one another.

    ``train`` records each run's bundle under its label; ``--resume``,
    ``--weights`` and similar inputs name an earlier run's checkpoint by
    ``(label, file)`` so its path and file hash normalize to that name.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.runs: dict[str, Path] = {}
        self.generated: dict[str, Path] = {}
        self.roots: list[str] = []
        self.record: dict[str, Any] = {}

    def checkpoint(self, label: str, name: str = "weights.pt") -> Path:
        path = self.runs[label] / "checkpoints" / name
        self.generated.setdefault(f"{label}/{name}", path)
        return path

    def external(self, path: Path) -> Path:
        if str(path) not in self.roots:
            self.roots.append(str(path))
        return path

    def train(
        self,
        label: str,
        payload: dict[str, Any],
        inputs: dict[str, Path] | None = None,
    ) -> Path:
        extra: list[str] = []
        for flag, path in (inputs or {}).items():
            extra += [f"--{flag}", str(path)]
        run = _train(self.directory, _write(self.directory, label, payload), *extra)
        self.runs[label] = run
        # A run's artifacts may embed hashes of its own checkpoints.
        for path in sorted((run / "checkpoints").glob("*.pt")):
            self.generated.setdefault(f"{label}/{path.name}", path)
        # Run directories embed wall-clock ids; name them by label.
        mapping = {
            **{str(path): f"<{name}>" for name, path in self.generated.items()},
            **{str(run_dir): f"<run:{name}>" for name, run_dir in self.runs.items()},
            **substitutions(
                (*self.roots, str(self.directory)), self.generated
            ),
        }
        self.record[label] = bundle(run, mapping)
        return run


def _session(fn: Callable[[Session], None]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ebl-golden-") as raw:
        session = Session(Path(raw))
        fn(session)
        return session.record


def _longer(payload: dict[str, Any], epochs: int) -> dict[str, Any]:
    longer = json.loads(json.dumps(payload))
    longer["modes"]["train"]["num_epochs"] = epochs
    return longer


# --------------------------------------------------------------------------
# mnist_relu.v1 teacher
# --------------------------------------------------------------------------


def _teacher_payload() -> dict[str, Any]:
    payload = _cpu(_example("mnist_relu/teacher.json"))
    payload["data"].update(
        {"batch_size": 8, "validation_points": 40, "num_points": 64}
    )
    payload["modes"]["train"].update(
        {"num_epochs": 2, "max_batches": 4, "max_validation_batches": 3}
    )
    return payload


def _with_teacher(session: Session) -> Path:
    session.train("teacher", _teacher_payload())
    return session.checkpoint("teacher")


def run_teacher(resume: bool) -> dict[str, Any]:
    def body(session: Session) -> None:
        payload = _teacher_payload()
        session.train("first", payload)
        if resume:
            session.train(
                "resumed",
                _longer(payload, 3),
                {"resume": session.checkpoint("first", "resume.pt")},
            )

    return _session(body)


# --------------------------------------------------------------------------
# mnist_relu_drn_reset*.v1
# --------------------------------------------------------------------------


def _tiny_reset_selection(payload: dict[str, Any]) -> None:
    selection = payload["modes"]["train"]["learning_rate_selection"]
    selection["parameters"].update(
        {
            "probe_batches": [2],
            "canary_batches": 1,
            "candidate_epochs": 1,
            "max_center_reductions": 0,
            # Two probe batches cannot meet a production stability gate.
            "stability_tolerance": 1.0,
        }
    )
    selection["parameters"]["safety"].update(
        {"warmup_batches": 1, "persistence_batches": 1}
    )


def _reset_payload(relative: str) -> dict[str, Any]:
    payload = _cpu(_example(relative))
    payload["data"].update(
        {"batch_size": 4, "validation_points": 20, "num_points": 32}
    )
    payload["solver"].update({"inference_iterations": 2, "training_iterations": 2})
    payload["modes"]["train"].update(
        {"num_epochs": 2, "max_batches": 2, "max_validation_batches": 2}
    )
    _tiny_reset_selection(payload)
    return payload


# name -> (example config, also resume?)
RESET_CASES: dict[str, tuple[str, bool]] = {
    "kl": ("mnist_relu_drn_reset/cohort_a_kl.json", True),
    "cross_entropy": ("mnist_relu_drn_reset/cohort_a_cross_entropy.json", False),
    "differential": (
        "mnist_relu_drn_reset_differential/cohort_a_paired_squared_error_10ep.json",
        False,
    ),
    "bias_kl": ("mnist_relu_drn_reset_bias/cohort_a_kl.json", False),
    "bias_legacy_kl": ("mnist_relu_drn_reset_bias_legacy/cohort_a_kl.json", False),
    "factorial_bias_logical": (
        "mnist_relu_drn_reset_factorial/bias/logical/teacher_kl.json",
        False,
    ),
    "factorial_legacy": (
        "mnist_relu_drn_reset_factorial/bias_free/legacy_process_global/"
        "paired_squared_error.json",
        True,
    ),
}


def run_reset(name: str) -> dict[str, Any]:
    relative, resume = RESET_CASES[name]

    def body(session: Session) -> None:
        inputs = {
            "teacher-weights": _with_teacher(session),
            "device-data": session.external(DEVICE_DATA),
        }
        payload = _reset_payload(relative)
        session.train("first", payload, inputs)
        if resume:
            session.train(
                "resumed",
                _longer(payload, 3),
                {**inputs, "resume": session.checkpoint("first", "resume.pt")},
            )

    return _session(body)


# --------------------------------------------------------------------------
# small_drn.v1
# --------------------------------------------------------------------------


def _small_payload(relative: str) -> dict[str, Any]:
    payload = _example(f"small_drn/{relative}")
    payload.setdefault("runtime", {})["device"] = "cpu"
    data = payload["data"]
    if data["dataset"] == "mnist":
        data.update({"batch_size": 4, "num_points": 32})
        if "validation_points" in data:
            data["validation_points"] = 20
    else:
        data.update({"num_points": 24, "batch_size": 4})
    payload["solver"].update({"inference_iterations": 2, "training_iterations": 2})
    train = payload["modes"]["train"]
    train.update({"num_epochs": 2, "max_batches": 2, "max_validation_batches": 2})
    selection = train.get("learning_rate_selection")
    if selection is not None and selection["type"] != "none":
        _tiny_reset_selection(payload)
    return payload


def _fake_wan2022() -> None:
    """Replace the AIHWKit Wan 2022 sampler by a deterministic offset.

    The phase runner sits above the device seam; the goldens need only a
    reproducible programmed state and report, not AIHWKit itself.
    """

    from experiments.small_network import deployment

    def program_one(catalog, config):
        binding = catalog.by_key["base.dense_weight.0"]
        with torch.no_grad():
            binding.state.add_(0.001)
        return {"model": config.type, "programming_seed": config.programming_seed}

    def program_many(catalog, layer_configs):
        reports = {}
        for key, config in layer_configs.items():
            with torch.no_grad():
                catalog.by_key[key].state.add_(0.001)
            reports[key] = {
                "model": config.type,
                "programming_seed": config.programming_seed,
            }
        return {"layers": reports}

    deployment.program_wan2022_base_conductance = program_one
    deployment.program_wan2022_base_conductances = program_many


def _base_weights(session: Session, label: str, payload: dict[str, Any]) -> Path:
    """Save the seeded base group of ``payload``'s model as named weights."""

    from experiments.small_network.components import build_model_stack
    from experiments.small_network.config import parse_small_drn_config
    from model.resistive.builders import ParameterCatalog
    from training.checkpoint import save_named_weights

    torch.manual_seed(payload["runtime"]["seed"])
    document = parse_small_drn_config(payload)
    stack = build_model_stack(document.common)
    catalog = ParameterCatalog(
        stack.bundle.catalog.for_group("base", checkpointed_only=True)
    )
    path = session.directory / f"{label}.pt"
    save_named_weights(path, catalog)
    session.generated[label] = path
    return path


def run_small_base(resume: bool) -> dict[str, Any]:
    def body(session: Session) -> None:
        payload = _small_payload("base.json")
        session.train("first", payload)
        if resume:
            session.train(
                "resumed",
                _longer(payload, 3),
                {"resume": session.checkpoint("first", "resume.pt")},
            )

    return _session(body)


def run_small_single(relative: str) -> dict[str, Any]:
    return _session(lambda session: session.train("first", _small_payload(relative)))


def run_small_adapter(relative: str) -> dict[str, Any]:
    """Train an adapter from a seeded base, then resume it."""

    def body(session: Session) -> None:
        payload = _small_payload(relative)
        base = _base_weights(session, "clean_base", payload)
        session.train("first", payload, {"base-weights": base})
        session.train(
            "resumed",
            _longer(payload, 3),
            {"resume": session.checkpoint("first", "resume.pt")},
        )

    return _session(body)


def run_small_measured() -> dict[str, Any]:
    """Cohort-A training, then cohort-B transfer and cohort-B LoRA from it."""

    def body(session: Session) -> None:
        device = {"device-data": session.external(DEVICE_DATA)}
        session.train(
            "cohort_a", _small_payload("measured_cohort_a_raw_mnist.json"), device
        )
        session.train(
            "cohort_b",
            _small_payload("measured_cohort_b_raw_mnist.json"),
            {**device, "weights": session.checkpoint("cohort_a")},
        )
        session.train(
            "cohort_b_lora",
            _small_payload("measured_cohort_b_lora_mnist.json"),
            {**device, "base-weights": session.checkpoint("cohort_a")},
        )

    return _session(body)


def run_small_program_verify() -> dict[str, Any]:
    """FP32 base, shared HWA, then CMO full and LoRA program-verify recovery.

    The chain mirrors ``campaigns/manifests/mnist_wan_cmo_head_to_head_seed17``.
    """

    def body(session: Session) -> None:
        hwa = _small_payload("mnist_wan_cmo_head_to_head/hwa_shared_from_fp32.json")
        fp32 = _small_payload("mnist_perfect_diode.json")
        # The production FP32 source was trained inside the bounded window.
        fp32["model"] = json.loads(json.dumps(hwa["model"]))
        session.train("fp32", fp32)
        session.train("hwa", hwa, {"weights": session.checkpoint("fp32")})
        weights = session.checkpoint("hwa")
        full = _small_payload("mnist_ibm_devices/cmo_full_bptt_recovery.json")
        session.train("full", full, {"weights": weights})
        session.train(
            "full_resumed",
            _longer(full, 3),
            {"resume": session.checkpoint("full", "resume.pt")},
        )
        session.train(
            "lora",
            _small_payload("mnist_ibm_devices/cmo_lora_bptt_recovery.json"),
            {"base-weights": weights},
        )

    return _session(body)


SMALL_SINGLE_CASES: dict[str, str] = {
    "hardware_aware": "hardware_aware.json",
}

_SYNTHETIC_FINGERPRINT = "5" * 64


def run_small_om_bounds() -> dict[str, Any]:
    """IBM OM FP32-bounds training on a synthetic fixed-array population.

    The production population artifact is not in this worktree; the loader
    is replaced by the synthetic IBM OM golden population and the frozen
    digests by those of a placeholder file.
    """

    from training import ibm_om_fp32_bounds

    payload = _small_payload(
        "mnist_om_range_uniform_optimizer_reference_20260826/adam_lr_1e3.json"
    )
    parameters = payload["modes"]["train"]["update_backend"]["parameters"]
    population = om.noisy_population(
        ("base.dense_weight.0", "base.dense_weight.1"),
        ((1568, 100), (100, 20)),
        seed=int(parameters["assignment_seed"]),
        corruption_policy=parameters["corruption_policy"],
        kind="wide",
        fingerprint=_SYNTHETIC_FINGERPRINT,
    )
    ibm_om_fp32_bounds.load_om_array_population = lambda path: population

    def body(session: Session) -> None:
        from experiments.artifacts import sha256_file

        placeholder = session.directory / "population.npz"
        placeholder.write_bytes(b"synthetic IBM OM population placeholder\n")
        parameters["expected_population_sha256"] = sha256_file(placeholder)
        parameters["expected_population_fingerprint"] = _SYNTHETIC_FINGERPRINT
        session.train("first", payload, {"device-data": placeholder})
        session.train(
            "resumed",
            _longer(payload, 3),
            {
                "device-data": placeholder,
                "resume": session.checkpoint("first", "resume.pt"),
            },
        )

    return _session(body)


# --------------------------------------------------------------------------
# mnist_relu_drn_kd.v1
# --------------------------------------------------------------------------


def _kd_payload(relative: str) -> dict[str, Any]:
    payload = _cpu(_example(f"mnist_relu_drn/{relative}"))
    payload["data"].update(
        {"batch_size": 4, "validation_points": 20, "num_points": 32}
    )
    payload["solver"].update({"inference_iterations": 2, "training_iterations": 2})
    train = payload["modes"]["train"]
    for key, value in (
        ("num_epochs", 2),
        ("max_batches", 2),
        ("max_validation_batches", 2),
    ):
        train[key] = value
    return payload


def _kd_chain(steps: tuple[tuple[str, str, str | None, bool], ...]) -> dict[str, Any]:
    """Run ``(label, config, weights-from label, measured)`` in order.

    The last step is also resumed one epoch further.
    """

    def body(session: Session) -> None:
        teacher = _with_teacher(session)
        inputs: dict[str, Path] = {}
        for label, relative, source, measured in steps:
            inputs = {"teacher-weights": teacher}
            if measured:
                inputs["device-data"] = session.external(DEVICE_DATA)
            if source is not None:
                inputs["weights"] = session.checkpoint(source)
            payload = _kd_payload(relative)
            session.train(label, payload, inputs)
        last = steps[-1][0]
        resumed = {key: value for key, value in inputs.items() if key != "weights"}
        session.train(
            f"{last}_resumed",
            _longer(payload, 3),
            {**resumed, "resume": session.checkpoint(last, "resume.pt")},
        )

    return _session(body)


KD_CASES: dict[str, tuple[tuple[str, str, str | None, bool], ...]] = {
    "ideal_single": (("ideal", "ideal_single.json", None, False),),
    "ideal_differential": (("ideal", "ideal_differential.json", None, False),),
    "measured_single": (("measured", "measured_raw_single.json", None, True),),
    "differential_cohort_ab": (
        ("cohort_a", "measured_raw_differential_finetune_10ep.json", None, True),
        (
            "cohort_b_control",
            "measured_raw_differential_cohort_b_literal_deployment_control.json",
            "cohort_a",
            True,
        ),
        (
            "cohort_b",
            "measured_raw_differential_cohort_b_common_window_finetune_10ep.json",
            "cohort_a",
            True,
        ),
    ),
    "quad_cohort_ab": (
        (
            "cohort_a",
            "measured_raw_single_quad_common_window_finetune_10ep.json",
            None,
            True,
        ),
        (
            "cohort_b",
            "measured_raw_single_quad_common_window_cohort_b_finetune_10ep.json",
            "cohort_a",
            True,
        ),
    ),
    "eight_device_one_pulse_down": (
        (
            "threshold",
            "eight_device_cohort_a_one_pulse_down/positive_p90_10ep.json",
            None,
            True,
        ),
    ),
    "sign_sgd": (
        ("sign", "four_device_cohort_a_sign_sgd/balanced_0p21ns.json", None, True),
    ),
}


def _fake_om_sampler(kind: str) -> None:
    """Replace AIHWKit OM population sampling by the synthetic golden one.

    ``kind`` is the bound regime of ``ibm_om_golden_cases.noisy_population``
    that suits the case's target mapping.
    """

    import hashlib

    from training.ibm_om import modifier

    def population(bindings, *, assignment_seed, corruption_policy):
        keys = tuple(binding.key for binding in bindings)
        shapes = tuple(tuple(binding.state.shape) for binding in bindings)
        fingerprint = hashlib.sha256(
            repr((keys, shapes, assignment_seed, corruption_policy, kind)).encode()
        ).hexdigest()
        return om.noisy_population(
            keys,
            shapes,
            seed=assignment_seed,
            corruption_policy=corruption_policy,
            kind=kind,
            fingerprint=fingerprint,
        )

    def external(
        bindings,
        *,
        assignment_seed,
        corruption_policy,
        aihwkit_python,
        population_path,
        receipt_path,
    ):
        sampled = population(
            bindings,
            assignment_seed=assignment_seed,
            corruption_policy=corruption_policy,
        )
        receipt = {"synthetic": True, "fingerprint": sampled.fingerprint}
        population_path.parent.mkdir(parents=True, exist_ok=True)
        population_path.write_bytes(f"synthetic {sampled.fingerprint}\n".encode())
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return sampled, receipt

    modifier.sample_om_array_population = population
    modifier.sample_om_array_population_external = external
    os.environ["EBL_AIHWKIT_PYTHON"] = "/synthetic/aihwkit-python"


# name -> (config, population kind)
KD_OM_CASES: dict[str, tuple[str, str]] = {
    "om_literal": ("ibm_om_hwa_pilot/om_repaired.json", "wide"),
    "om_gaussian": ("ibm_om_hwa_pilot/gaussian_3pct.json", "wide"),
    "om_common_window": (
        "ibm_om_common_window_hwa_pilot/exact_bounds_hwa.json",
        "logical",
    ),
    "om_differential_pair": (
        "ibm_om_differential_pair_hwa_pilot/exact_bounds_hwa.json",
        "logical",
    ),
    "om_reset_relative": (
        "ibm_om_reset_relative_quantized_hwa/continuous_hwa.json",
        "reset",
    ),
    "om_raw_active_qat": ("ibm_om_raw_active_p90_qat/quantized_qat.json", "raw"),
}


def run_kd_om(name: str) -> dict[str, Any]:
    relative, kind = KD_OM_CASES[name]
    _fake_om_sampler(kind)
    device_model = ROOT / "data" / "ibm_reram_om_pv128_hwa_v1.json"

    def body(session: Session) -> None:
        payload = _kd_payload(relative)
        if payload["teacher"]["type"] == "bounded_drn":
            session.train(
                "teacher",
                _small_payload("mnist_bounded_memristor_teacher_10ep.json"),
            )
            teacher = session.checkpoint("teacher")
        else:
            teacher = _with_teacher(session)
        inputs = {
            "teacher-weights": teacher,
            "device-model": session.external(device_model),
        }
        session.train("first", payload, inputs)
        session.train(
            "resumed",
            _longer(payload, 3),
            {**inputs, "resume": session.checkpoint("first", "resume.pt")},
        )

    return _session(body)


# --------------------------------------------------------------------------
# Registry and capture
# --------------------------------------------------------------------------


def all_cases() -> dict[str, Callable[[], Any]]:
    cases: dict[str, Callable[[], Any]] = {
        "teacher/train": lambda: run_teacher(False),
        "teacher/resume": lambda: run_teacher(True),
    }
    for name in RESET_CASES:
        cases[f"reset/{name}"] = lambda name=name: run_reset(name)
    cases["base"] = lambda: run_small_base(False)
    cases["base_resume"] = lambda: run_small_base(True)
    for name, relative in SMALL_SINGLE_CASES.items():
        cases[name] = lambda relative=relative: run_small_single(relative)
    cases["om_fp32_bounds"] = run_small_om_bounds
    cases["lora"] = lambda: run_small_adapter("lora.json")
    cases["digital_lora_wan"] = lambda: (
        _fake_wan2022(),
        run_small_adapter("digital_lora_reram_wan2022_digits.json"),
    )[1]
    cases["passive_layerwise_wan"] = lambda: (
        _fake_wan2022(),
        run_small_adapter("passive_layerwise_lora_reram_wan2022_digits.json"),
    )[1]
    for name, steps in KD_CASES.items():
        cases[f"kd/{name}"] = lambda steps=steps: _kd_chain(steps)
    for name in KD_OM_CASES:
        cases[f"kd/{name}"] = lambda name=name: run_kd_om(name)
    cases["measured"] = run_small_measured
    cases["program_verify"] = run_small_program_verify
    return cases


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.pt"


# Multi-threaded CPU kernels are not bit-reproducible on every path (the
# IBM OM FP32-bounds run differs between identical 4-thread runs), so each
# case runs single-threaded and cases run in parallel instead.
THREADS = os.environ.get("EBL_GOLDEN_THREADS", "1")


def run_isolated(name: str) -> Any:
    """Run one case in a fresh interpreter and return its record.

    Some families name layers from process-global counters, so a case's
    record depends on what already ran in the process; the CLI always runs
    one command per process.
    """

    with tempfile.TemporaryDirectory(prefix="ebl-golden-record-") as raw:
        output = Path(raw) / "record.pt"
        env = {
            **os.environ,
            "OMP_NUM_THREADS": THREADS,
            "MKL_NUM_THREADS": THREADS,
            "EBL_MNIST_ROOT": str(MNIST_ROOT),
            "EBL_DEVICE_DATA": str(DEVICE_DATA),
        }
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "record", name, str(output)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"case {name} failed ({completed.returncode}):\n"
                f"{completed.stderr[-4000:]}"
            )
        return torch.load(output, weights_only=False)


def record(name: str, output: str) -> None:
    import random

    import numpy as np

    # Runtimes seed torch only, but resume.pt stores the Python and NumPy
    # generator states too.
    random.seed(0)
    np.random.seed(0)
    torch.set_num_threads(int(THREADS))
    torch.save(digest_tensors(all_cases()[name]()), output)


def _parallel(fn: Callable[[str], Any], names: list[str]) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    jobs = int(os.environ.get("EBL_GOLDEN_JOBS", "6"))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return dict(zip(names, pool.map(fn, names)))


def capture(selected: list[str] | None = None) -> None:
    cases = all_cases()
    names = selected or list(cases)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    for name, result in _parallel(run_isolated, names).items():
        path = golden_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, path)
        print(f"captured {name}")
    meta_path = GOLDEN_DIR / "meta.json"
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.is_file() and selected
        else {}
    )
    dirty = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split("\n")
    meta.update(
        {
            "commit": commit,
            "dirty_tracked_files": sorted(line[3:] for line in dirty if line),
            **environment(),
            "cases": sorted(cases),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def check(selected: list[str] | None = None) -> None:
    """Run each case twice and compare: proves normalization is complete."""

    names = selected or list(all_cases())
    twice = [f"{name}\0{index}" for name in names for index in (0, 1)]
    def attempt(key: str) -> Any:
        try:
            return run_isolated(key.split("\0")[0])
        except RuntimeError as error:
            return error

    results = _parallel(attempt, twice)
    failed = False
    for name in names:
        first, second = results[f"{name}\0" + "0"], results[f"{name}\0" + "1"]
        if isinstance(first, Exception) or isinstance(second, Exception):
            failed = True
            error = first if isinstance(first, Exception) else second
            print(f"FAILED {name}: {str(error)[-600:]}")
            continue
        try:
            assert_same(first, second, path=name)
        except AssertionError as error:
            failed = True
            print(f"NONDETERMINISTIC {name}: {error}")
            continue
        print(f"deterministic {name}")
    if failed:
        raise SystemExit(1)


def verify(selected: list[str] | None = None) -> None:
    """Replay cases (default: all) against their captured goldens."""

    names = selected or sorted(all_cases())

    def attempt(name: str) -> Any:
        try:
            return run_isolated(name)
        except RuntimeError as error:
            return error

    failed = False
    for name, actual in _parallel(attempt, names).items():
        if isinstance(actual, Exception):
            failed = True
            print(f"FAILED {name}: {str(actual)[-1500:]}")
            continue
        try:
            assert_same(torch.load(golden_path(name), weights_only=False), actual, path=name)
        except AssertionError as error:
            failed = True
            print(f"DIFFERS {name}: {error}")
            continue
        print(f"matches {name}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    command, *names = sys.argv[1:] or ["capture"]
    if command == "record":
        record(*names)
    else:
        {"capture": capture, "check": check, "verify": verify}[command](names or None)
