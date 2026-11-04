#!/usr/bin/env python3
"""Build a paper-ready digits-core reproduction pack.

The pack is intentionally separate from the research tree.  It vendors only the
runtime pieces needed to validate saved digits checkpoints and regenerates the
paper figures from bundled, curated inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "papers" / "paper_core_repro_pack"
TIMING_BUNDLE = PROJECT_ROOT / "labs" / "figures_for_paper_digits" / "timings" / "selected_for_paper"
ERROR_ITER_ROOT = PROJECT_ROOT / "labs" / "figures_for_paper_digits" / "error_vs_iter"
VOLTOL_ROOT = PROJECT_ROOT / "labs" / "figures_for_paper_digits" / "error_vs_vol_tol"
IV_CURVE = PROJECT_ROOT / "labs" / "i_v_npz" / "experimental_curve_voff_0.8_200_points.npz"
ERROR_ITER_HIDDEN3_CORRECT_CONFIG_ROOT = (
    ERROR_ITER_ROOT
    / "exp_clip_165_batch256_20260513"
    / "double_diode_exponential"
    / "hidden_3_correct_config"
    / "configs"
)

REL_TOLS = ("1e-7", "1e-6", "1e-5", "1e-4", "1e-3")
FAMILIES = ("single_diode_exponential", "double_diode_exponential", "experimental")
TEXT_ARTIFACT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt"}

DOUBLE_DIODE_UPDATER_RENAMES = {
    "custom": "float64",
    "float64": "float64",
    "float64_timed": "float64",
    "timedexponentialdoublediodeupdater": "float64",
    "float32": "float32",
    "float64_timed_overrelaxed": "float64_overrelaxed",
    "overrelazedtimedexponentialdoublediodeupdater": "float64_overrelaxed",
    "float64_overrelaxed": "float64_overrelaxed",
    "float32_overrelaxed": "float32_overrelaxed",
    "overrelaxed": "float32_overrelaxed",
    "overrelated": "float32_overrelaxed",
}

SINGLE_DIODE_UPDATER_RENAMES = {
    "custom": "custom",
    "standard": "standard",
    "overrelaxed": "overrelaxed",
    "overrelated": "overrelaxed",
}

EXPERIMENTAL_UPDATER_RENAMES = {
    "custom": "standard",
    "standard": "standard",
    "overrelaxed": "overrelaxed",
    "overrelated": "overrelaxed",
}

LEGACY_EXPERIMENTAL_NEWTON_TOL_FIELDS = (
    "experimental_exponential_newton_tol_progressive",
    "experimental_exponential_newton_tol_start",
    "experimental_exponential_newton_tol_end",
    "experimental_exponential_newton_tol_switch_hi",
    "experimental_exponential_newton_tol_switch_lo",
)

LEGACY_OVERRELAXATION_REJECT_FIELDS = (
    "overrelaxation_reject_steps",
    "overrelaxation_reject_max_tries",
    "overrelaxation_reject_shrink",
    "overrelaxation_reject_eps",
)

LEGACY_TABLE_COLUMNS_TO_DROP = frozenset(LEGACY_OVERRELAXATION_REJECT_FIELDS)

ERROR_ITER_CONFIG_OVERRIDES = {
    (
        "double_diode_exponential",
        "hidden_3",
        "original_exp_clip",
    ): ERROR_ITER_HIDDEN3_CORRECT_CONFIG_ROOT / "correct_config_exp_clip_85.json",
    (
        "double_diode_exponential",
        "hidden_3",
        "exp_clip_165",
    ): ERROR_ITER_HIDDEN3_CORRECT_CONFIG_ROOT / "correct_config_exp_clip_165.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace an existing pack directory.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_file(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Expected {label} to be an existing file. Provided value: {path}")
    return path


def copy_file(src: Path, dst: Path) -> str:
    require_file(src, "source file")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def sanitize_provenance_text(text: str) -> str:
    """Remove machine-specific absolute paths from copied paper artifacts."""
    rewrites = (
        ("/home/filip/paper_simulation_results_", "<paper-source-results>"),
        (str(PROJECT_ROOT / "simulation_results"), "<source-results>"),
        (str(PROJECT_ROOT), "<source-tree>"),
        ("/tmp/", "<scratch>/"),
        ("float64_timed_overrelaxed", "float64_overrelaxed"),
        ("float64_timed", "float64"),
        ("TimedExponentialDOubleDiodeUpdater", "Float64ExponentialDoubleDiodeUpdater"),
        ("OverRelazedTimedExponentialDoubleDiodeUpdater", "OverRelaxedFloat64DoubleDiodeUpdater"),
        ("time_digits_validate_with_residual.log", "validation_runtime.log"),
        ("validation_states_spice_layers.timing.json", "validation_states_spice_layers_metadata.json"),
        (
            "validation_inputs_layer0_spice_layers.timing.json",
            "validation_inputs_layer0_spice_layers_metadata.json",
        ),
    )
    for old, new in rewrites:
        text = text.replace(old, new)
    return text


def sanitize_json_scalars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_json_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_scalars(item) for item in value]
    if isinstance(value, str):
        return sanitize_provenance_text(value)
    return value


def normalize_updater_name(value: Any, rewrites: dict[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    return rewrites.get(key.replace("_", ""), rewrites.get(key, value))


def normalize_config_for_pack(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    non_linearity = data.get("non_linearity")

    if non_linearity == "experimental":
        data["double_diode_updater"] = normalize_updater_name(
            data.get("double_diode_updater"),
            EXPERIMENTAL_UPDATER_RENAMES,
        )
    else:
        data["double_diode_updater"] = normalize_updater_name(
            data.get("double_diode_updater"),
            DOUBLE_DIODE_UPDATER_RENAMES,
        )

    data["single_diode_updater"] = normalize_updater_name(
        data.get("single_diode_updater"),
        SINGLE_DIODE_UPDATER_RENAMES,
    )
    if "experimental_newton_tol" not in data and "experimental_exponential_newton_tol_start" in data:
        data["experimental_newton_tol"] = data["experimental_exponential_newton_tol_start"]
    for field in LEGACY_EXPERIMENTAL_NEWTON_TOL_FIELDS:
        data.pop(field, None)
    for field in LEGACY_OVERRELAXATION_REJECT_FIELDS:
        data.pop(field, None)
    return data


def copy_text_artifact(src: Path, dst: Path) -> str:
    require_file(src, "source text artifact")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".csv":
        return copy_csv_artifact(src, dst)
    text = src.read_text(encoding="utf-8")
    dst.write_text(sanitize_provenance_text(text), encoding="utf-8")
    return str(dst)


def copy_csv_artifact(src: Path, dst: Path) -> str:
    with src.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            dst.write_text("", encoding="utf-8")
            return str(dst)
        fieldnames = [field for field in reader.fieldnames if field not in LEGACY_TABLE_COLUMNS_TO_DROP]
        rows = [
            {field: sanitize_provenance_text(row.get(field, "")) for field in fieldnames}
            for row in reader
        ]

    with dst.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return str(dst)


def copy_curated_artifact(src: Path, dst: Path) -> str:
    if src.suffix.lower() in TEXT_ARTIFACT_SUFFIXES:
        return copy_text_artifact(src, dst)
    return copy_file(src, dst)


def copy_python_tree(src_root: Path, dst_root: Path) -> None:
    for src in src_root.rglob("*.py"):
        rel = src.relative_to(src_root)
        copy_file(src, dst_root / rel)


def patch_vendor_model_for_repro(vendor_root: Path) -> None:
    """Apply small paper-runtime cleanups to copied model code."""
    network_py = vendor_root / "model" / "resistive" / "network.py"
    text = network_py.read_text(encoding="utf-8")
    text = text.replace(
        '        else:\n'
        '            non_linear_interaction = []\n'
        '            print("Nonlinear interaction for this non-linearity not defined yet")\n',
        '        elif non_linearity == "experimental":\n'
        '            # The paper repro pack handles the measured I-V curve in the\n'
        '            # coordinate updater, so there is no analytic energy term here.\n'
        '            non_linear_interaction = []\n'
        '        else:\n'
        '            raise ValueError(\n'
        '                "Expected non_linearity to be one of the bundled resistive nonlinearities. "\n'
        '                f"Provided value: {non_linearity!r}."\n'
        '            )\n',
    )
    network_py.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_hidden_key(value: str) -> int:
    prefix = "hidden_"
    if not value.startswith(prefix):
        raise ValueError(f"Expected hidden key like hidden_2. Provided value: {value!r}.")
    return int(value[len(prefix) :])


def parse_case(value: str) -> tuple[str, int, int]:
    parts = value.split("/")
    if len(parts) != 3 or not parts[1].startswith("hidden_") or not parts[2].startswith("hidden_"):
        raise ValueError(f"Expected case format <family>/hidden_N/hidden_W. Provided value: {value!r}.")
    return parts[0], parse_hidden_key(parts[1]), parse_hidden_key(parts[2])


def config_hidden_size(config_path: Path) -> int | None:
    data = read_json(config_path)
    dims = data.get("dims")
    if isinstance(dims, list) and len(dims) >= 3:
        return int(dims[1])
    return None


def copy_config(src: Path, dst: Path, *, pack_root: Path, updates: dict[str, Any] | None = None) -> str:
    data = normalize_config_for_pack(read_json(require_file(src, "config JSON")))
    if updates:
        data.update(updates)
    if data.get("non_linearity") == "experimental":
        rel_asset = "data/assets/experimental_curve_voff_0.8_200_points.npz"
        data["iv_data_path"] = rel_asset
        data["LABS_IV_CURVE_PATH"] = rel_asset
        if data.get("damping") is None:
            # Some legacy paper configs predate the explicit damping field.
            data["damping"] = 0.5
    data.pop("output_dir", None)
    data.pop("output_root", None)
    data = sanitize_json_scalars(data)
    write_json(dst, data)
    return str(dst)


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def add_timing_jobs(pack_root: Path, jobs: list[dict[str, Any]]) -> None:
    manifest_csv = TIMING_BUNDLE / "tables" / "bundle_manifest.csv"
    for row in read_csv(require_file(manifest_csv, "timing bundle manifest")):
        family, hidden_layers, hidden_size = parse_case(row["case"])
        slug = f"{family}__hidden_{hidden_layers}__hidden_{hidden_size}"
        config_dst = pack_root / "data" / "timing" / "configs" / slug / "config.json"
        weight_dst = pack_root / "data" / "timing" / "weights" / slug / "weights.pt"

        copy_config(TIMING_BUNDLE / row["bundle_config_json"], config_dst, pack_root=pack_root)
        copy_file(TIMING_BUNDLE / row["bundle_weight_checkpoint"], weight_dst)

        reference = None
        if row.get("bundle_spice_npz"):
            ref_dst = pack_root / "data" / "timing" / row["bundle_spice_npz"]
            copy_file(TIMING_BUNDLE / row["bundle_spice_npz"], ref_dst)
            reference = rel(ref_dst, pack_root)

        jobs.append(
            {
                "job_id": f"timing/{family}/hidden_{hidden_layers}/hidden_{hidden_size}",
                "group": "timing",
                "family": family,
                "hidden_layers": hidden_layers,
                "hidden_size": hidden_size,
                "config": rel(config_dst, pack_root),
                "weights": rel(weight_dst, pack_root),
                "reference_npz": reference,
                "num_iterations": int(float(row["num_iterations"])),
                "rel_tol": row.get("rel_tol") or None,
                "overrelaxation_factor": row.get("overrelaxation_factor") or None,
                "source": "selected_for_paper",
            }
        )


def copy_timing_curated_inputs(pack_root: Path) -> None:
    for family in FAMILIES:
        src_dir = TIMING_BUNDLE / "figure_inputs" / family
        if src_dir.exists():
            for src in src_dir.iterdir():
                if src.is_file() and src.suffix.lower() in (".csv", ".json"):
                    copy_curated_artifact(src, pack_root / "data" / "timing" / "figure_inputs" / family / src.name)
    for src in (TIMING_BUNDLE / "tables").iterdir():
        if src.is_file() and src.suffix.lower() in (".csv", ".md", ".tex", ".json"):
            copy_curated_artifact(src, pack_root / "data" / "timing" / "tables" / src.name)


def add_error_iter_jobs(pack_root: Path, jobs: list[dict[str, Any]]) -> None:
    manifest_path = ERROR_ITER_ROOT / "exp_clip_165_batch256_20260513" / "manifest.json"
    manifest = read_json(require_file(manifest_path, "error-vs-iteration manifest"))
    for case in manifest["cases"]:
        family = case["family"]
        hidden_key = case["hidden"]
        hidden_layers = parse_hidden_key(hidden_key)
        hidden_size = config_hidden_size(Path(case["variants"][0]["config"])) or 128
        weight_dst = pack_root / "data" / "error_vs_iter" / "weights" / f"{family}__{hidden_key}" / "weights.pt"
        ref_dst = pack_root / "data" / "error_vs_iter" / "references" / f"{family}__{hidden_key}" / "validation_states_spice_layers.npz"
        copy_file(Path(case["source_weights"]), weight_dst)
        copy_file(Path(case["source_spice"]), ref_dst)

        for variant in case["variants"]:
            variant_name = variant["name"]
            config_src = ERROR_ITER_CONFIG_OVERRIDES.get(
                (family, hidden_key, variant_name),
                Path(variant["config"]),
            )
            config_dst = (
                pack_root
                / "data"
                / "error_vs_iter"
                / "configs"
                / f"{family}__{hidden_key}__{variant_name}.json"
            )
            # Paper provenance: hidden-3 double-diode summaries were rebuilt
            # from corrected amp-4 configs; the older manifest points at a
            # stale voltage_amp=1 config that does not match the SPICE NPZ.
            config_updates = {}
            if family == "double_diode_exponential":
                config_updates["double_diode_updater"] = "float64_overrelaxed"
            copy_config(config_src, config_dst, pack_root=pack_root, updates=config_updates)
            for iteration in case["iterations"]:
                jobs.append(
                    {
                        "job_id": f"error_vs_iter/{family}/{hidden_key}/{variant_name}/iter_{iteration}",
                        "group": "error_vs_iter",
                        "family": family,
                        "hidden_layers": hidden_layers,
                        "hidden_size": hidden_size,
                        "config": rel(config_dst, pack_root),
                        "weights": rel(weight_dst, pack_root),
                        "reference_npz": rel(ref_dst, pack_root),
                        "num_iterations": int(iteration),
                        "rel_tol": None,
                        "overrelaxation_factor": case.get("source_overrelaxation_factor"),
                        "variant": variant_name,
                        "source": "exp_clip_165_batch256_20260513",
                    }
                )


def copy_error_iter_curated_inputs(pack_root: Path) -> None:
    for family in FAMILIES:
        src = ERROR_ITER_ROOT / family / "error_vs_iter_summary_20260302.json"
        if src.exists():
            copy_curated_artifact(src, pack_root / "data" / "error_vs_iter" / "summaries" / family / src.name)


def add_voltol_jobs(pack_root: Path, jobs: list[dict[str, Any]]) -> None:
    manifest_path = VOLTOL_ROOT / "exp_clip_165_batch1_20260513" / "manifest.json"
    manifest = read_json(require_file(manifest_path, "voltage-tolerance manifest"))
    for case in manifest["cases"]:
        family = case["family"]
        hidden_key = case["hidden"]
        hidden_layers = parse_hidden_key(hidden_key)
        hidden_size = config_hidden_size(Path(case["configs"][0]["config"])) or 128
        weight_dst = pack_root / "data" / "vol_tol" / "weights" / f"{family}__{hidden_key}" / "weights.pt"
        ref_dst = pack_root / "data" / "vol_tol" / "references" / f"{family}__{hidden_key}" / "validation_states_spice_layers.npz"
        copy_file(Path(case["weights"]), weight_dst)
        copy_file(Path(case["spice"]), ref_dst)
        for cfg in case["configs"]:
            rel_label = cfg["rel_tol"]
            config_dst = (
                pack_root / "data" / "vol_tol" / "configs" / f"{family}__{hidden_key}__rel_tol_{rel_label}.json"
            )
            copy_config(Path(cfg["config"]), config_dst, pack_root=pack_root)
            jobs.append(
                {
                    "job_id": f"vol_tol/{family}/{hidden_key}/rel_tol_{rel_label}",
                    "group": "vol_tol",
                    "family": family,
                    "hidden_layers": hidden_layers,
                    "hidden_size": hidden_size,
                    "config": rel(config_dst, pack_root),
                    "weights": rel(weight_dst, pack_root),
                    "reference_npz": rel(ref_dst, pack_root),
                    "num_iterations": 512,
                    "rel_tol": rel_label,
                    "overrelaxation_factor": case.get("overrelaxation_factor"),
                    "source": "exp_clip_165_batch1_20260513",
                }
            )

    add_experimental_voltol_jobs(pack_root, jobs)


def add_experimental_voltol_jobs(pack_root: Path, jobs: list[dict[str, Any]]) -> None:
    cases = [
        (
            "hidden_1",
            VOLTOL_ROOT / "experimental" / "hidden_1" / "vol_tol_sweep_omega1.0_batch1_20260513" / "configs",
            PROJECT_ROOT
            / "simulation_results"
            / "digits_medium_network"
            / "hidden_1"
            / "experimental_double_diode_exponential"
            / "hidden_128"
            / "vol_tol_sweep"
            / "model.pt",
            VOLTOL_ROOT / "experimental" / "hidden_1" / "validation_states_spice_layers.npz",
        ),
        (
            "hidden_2",
            VOLTOL_ROOT / "experimental" / "hidden_2" / "vol_tol_sweep_omega1.0_batch1_20260513" / "configs",
            PROJECT_ROOT
            / "simulation_results"
            / "digits_medium_network"
            / "hidden_2"
            / "experimental_double_diode_exponential"
            / "hidden_128"
            / "high_accuracy_rerun_02_3"
            / "model_best.pt",
            PROJECT_ROOT
            / "simulation_results"
            / "digits_medium_network"
            / "hidden_2"
            / "experimental_double_diode_exponential"
            / "hidden_128"
            / "high_accuracy_rerun_02_3"
            / "validation_states_spice_layers.npz",
        ),
        (
            "hidden_3",
            VOLTOL_ROOT / "experimental" / "hidden_3" / "amp_4" / "vol_tol_sweep" / "rel_tol_config",
            PROJECT_ROOT
            / "simulation_results"
            / "digits_medium_network"
            / "hidden_3"
            / "experimental_double_diode_exponential"
            / "hidden_128"
            / "amp_4"
            / "model_best.pt",
            VOLTOL_ROOT / "experimental" / "hidden_3" / "amp_4" / "validation_states_spice_layers.npz",
        ),
    ]
    for hidden_key, cfg_dir, weight_src, ref_src in cases:
        hidden_layers = parse_hidden_key(hidden_key)
        weight_dst = pack_root / "data" / "vol_tol" / "weights" / f"experimental__{hidden_key}" / "weights.pt"
        ref_dst = pack_root / "data" / "vol_tol" / "references" / f"experimental__{hidden_key}" / "validation_states_spice_layers.npz"
        copy_file(weight_src, weight_dst)
        copy_file(ref_src, ref_dst)
        for rel_label in REL_TOLS:
            src_config = cfg_dir / f"config_rel_tol_{rel_label}.json"
            if not src_config.exists():
                src_config = cfg_dir / f"rel_tol_{rel_label}" / "config.json"
            config_dst = (
                pack_root / "data" / "vol_tol" / "configs" / f"experimental__{hidden_key}__rel_tol_{rel_label}.json"
            )
            copy_config(src_config, config_dst, pack_root=pack_root)
            jobs.append(
                {
                    "job_id": f"vol_tol/experimental/{hidden_key}/rel_tol_{rel_label}",
                    "group": "vol_tol",
                    "family": "experimental",
                    "hidden_layers": hidden_layers,
                    "hidden_size": 128,
                    "config": rel(config_dst, pack_root),
                    "weights": rel(weight_dst, pack_root),
                    "reference_npz": rel(ref_dst, pack_root),
                    "num_iterations": 512,
                    "rel_tol": rel_label,
                    "overrelaxation_factor": 1.0,
                    "source": "experimental_vol_tol_selected",
                }
            )


def copy_voltol_curated_inputs(pack_root: Path) -> None:
    summary_paths = {
        "single_diode_exponential": VOLTOL_ROOT
        / "single_diode_exponential"
        / "amp_4_plots_data"
        / "vol_tol_summary_hidden1_hidden2_hidden3_amp4.json",
        "double_diode_exponential": VOLTOL_ROOT
        / "double_diode_exponential"
        / "amp_4_plots_data"
        / "vol_tol_summary_hidden1_hidden2_hidden3_amp4.json",
        "experimental": VOLTOL_ROOT
        / "experimental"
        / "amp_4_plots_data"
        / "vol_tol_summary_hidden1_hidden2_hidden3_amp4.json",
    }
    for family, src in summary_paths.items():
        copy_curated_artifact(src, pack_root / "data" / "vol_tol" / "summaries" / family / src.name)


def write_checksums(pack_root: Path, manifest: dict[str, Any]) -> None:
    checksums = {}
    for path in sorted(pack_root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = rel(path, pack_root)
        if rel_path in {"data/checksums.sha256", "data/manifest.json"}:
            continue
        checksums[rel_path] = sha256(path)
    manifest["checksums"] = checksums
    write_json(pack_root / "data" / "manifest.json", manifest)
    lines = [f"{digest}  {path}" for path, digest in checksums.items()]
    (pack_root / "data" / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def write_pack_code(pack_root: Path) -> None:
    write_text(pack_root / "scripts" / "reproduce.py", REPRODUCE_PY)
    write_text(pack_root / "repro" / "__init__.py", "__all__ = []\n")
    write_text(pack_root / "repro" / "cli.py", CLI_PY)
    write_text(pack_root / "repro" / "manifest.py", MANIFEST_PY)
    write_text(pack_root / "repro" / "config.py", CONFIG_PY)
    write_text(pack_root / "repro" / "digits_validate.py", DIGITS_VALIDATE_PY)
    write_text(pack_root / "repro" / "npz_compare.py", NPZ_COMPARE_PY)
    write_text(pack_root / "repro" / "aggregate.py", AGGREGATE_PY)
    write_text(pack_root / "repro" / "plots.py", PLOTS_PY)
    write_text(pack_root / "README.md", README_MD)
    write_text(pack_root / "requirements.txt", REQUIREMENTS_TXT)

    vendor_root = pack_root / "repro" / "vendor"
    copy_python_tree(PROJECT_ROOT / "model", vendor_root / "model")
    patch_vendor_model_for_repro(vendor_root)
    copy_file(
        PROJECT_ROOT / "labs" / "tools" / "digits_core_repro_custom_minimizer.py",
        vendor_root / "labs" / "custom_minimizer.py",
    )
    write_text(vendor_root / "labs" / "__init__.py", "")


def build_pack(pack_root: Path, *, force: bool) -> None:
    if pack_root.exists():
        if not force:
            raise FileExistsError(
                f"Expected output directory not to exist. Provided value: {pack_root}. Use --force to replace it."
            )
        shutil.rmtree(pack_root)
    pack_root.mkdir(parents=True)
    copy_file(IV_CURVE, pack_root / "data" / "assets" / IV_CURVE.name)

    jobs: list[dict[str, Any]] = []
    add_timing_jobs(pack_root, jobs)
    copy_timing_curated_inputs(pack_root)
    add_error_iter_jobs(pack_root, jobs)
    copy_error_iter_curated_inputs(pack_root)
    add_voltol_jobs(pack_root, jobs)
    copy_voltol_curated_inputs(pack_root)
    write_pack_code(pack_root)

    manifest = {
        "name": "paper_core_repro_pack",
        "created_by": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "description": "Validation-only reproduction pack for the digits-core paper figures.",
        "jobs": sorted(jobs, key=lambda item: item["job_id"]),
        "groups": {
            "timing": sum(1 for job in jobs if job["group"] == "timing"),
            "error_vs_iter": sum(1 for job in jobs if job["group"] == "error_vs_iter"),
            "vol_tol": sum(1 for job in jobs if job["group"] == "vol_tol"),
        },
        "notes": [
            "Training and SPICE regeneration are intentionally out of scope.",
            "Timing figures are regenerated from curated CSVs because validation timing is machine-dependent.",
            "Legacy experimental voltage-tolerance configs with missing damping are normalized to damping=0.5.",
        ],
    }
    write_checksums(pack_root, manifest)


def main() -> int:
    args = parse_args()
    build_pack(args.output_dir.expanduser().resolve(), force=args.force)
    print(f"Wrote {args.output_dir.expanduser().resolve()}")
    return 0


REPRODUCE_PY = r'''
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "repro" / "vendor"))

from repro.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
'''


CLI_PY = r'''
from __future__ import annotations

import argparse
import os
from pathlib import Path

from repro.manifest import PackManifest


PACK_ROOT = Path(__file__).resolve().parents[1]


def configure_environment() -> None:
    # These variables avoid OpenMP shared-memory failures in restricted runners.
    defaults = {
        "KMP_DISABLE_SHM": "1",
        "KMP_SHM_DISABLE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("DRN_B_CLAMP", "1e6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the digits-core paper figures.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List manifest jobs and bundled artifact counts.")
    sub.add_parser("verify", help="Verify manifest checksums.")
    sub.add_parser("figures", help="Regenerate figures and tables from bundled curated inputs.")

    for name in ("validate", "compare", "all"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--group", choices=("timing", "error_vs_iter", "vol_tol", "all"), default="all")
        cmd.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        cmd.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
        cmd.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def selected_jobs(manifest: PackManifest, group: str, limit: int | None):
    jobs = manifest.jobs if group == "all" else manifest.jobs_for_group(group)
    return jobs[:limit] if limit is not None else jobs


def command_list(manifest: PackManifest) -> int:
    print(f"jobs: {len(manifest.jobs)}")
    for group in ("timing", "error_vs_iter", "vol_tol"):
        print(f"{group}: {len(manifest.jobs_for_group(group))}")
    missing_reference = [job.job_id for job in manifest.jobs if job.reference_npz is None]
    print(f"jobs_without_reference_npz: {len(missing_reference)}")
    return 0


def run_many(kind: str, manifest: PackManifest, args: argparse.Namespace) -> int:
    failures = []
    for job in selected_jobs(manifest, args.group, args.limit):
        try:
            if kind == "validate":
                from repro.digits_validate import run_validation

                result = run_validation(PACK_ROOT, job, device=args.device)
                print(f"[validate] {job.job_id} accuracy={result.accuracy:.4f} run_dir={result.run_dir}")
            elif kind == "compare":
                from repro.npz_compare import compare_job

                summary = compare_job(PACK_ROOT, job)
                if summary is None:
                    print(f"[compare] {job.job_id} skipped: no reference NPZ")
                else:
                    print(f"[compare] {job.job_id} p90={summary['node_weighted_rel_l1_percentiles']['p90']:.6g}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((job.job_id, str(exc)))
            print(f"[{kind}] {job.job_id} failed: {exc}")
    if failures:
        print(f"{kind}_failures: {len(failures)}")
        return 1
    return 0


def main() -> int:
    configure_environment()
    args = parse_args()
    manifest = PackManifest.load(PACK_ROOT)
    if args.command == "list":
        return command_list(manifest)
    if args.command == "verify":
        manifest.verify_checksums()
        print("checksums: ok")
        return 0
    if args.command == "figures":
        from repro.plots import regenerate_all_figures

        regenerate_all_figures(PACK_ROOT)
        print(f"wrote figures under {PACK_ROOT / 'outputs'}")
        return 0
    if args.command == "validate":
        return run_many("validate", manifest, args)
    if args.command == "compare":
        return run_many("compare", manifest, args)
    if args.command == "all":
        status = run_many("validate", manifest, args)
        if status:
            return status
        status = run_many("compare", manifest, args)
        if status:
            return status
        from repro.aggregate import aggregate_error_vs_iter, aggregate_vol_tol
        from repro.plots import regenerate_all_figures

        aggregate_error_vs_iter(PACK_ROOT)
        aggregate_vol_tol(PACK_ROOT)
        regenerate_all_figures(PACK_ROOT)
        return 0
    raise AssertionError(args.command)
'''


MANIFEST_PY = r'''
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReproJob:
    job_id: str
    group: str
    family: str
    hidden_layers: int
    hidden_size: int
    config: str
    weights: str
    reference_npz: str | None
    num_iterations: int
    rel_tol: str | None = None
    overrelaxation_factor: str | float | None = None
    variant: str | None = None
    source: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReproJob":
        return cls(**payload)

    def config_path(self, root: Path) -> Path:
        return root / self.config

    def weights_path(self, root: Path) -> Path:
        return root / self.weights

    def reference_path(self, root: Path) -> Path | None:
        return None if self.reference_npz is None else root / self.reference_npz

    def output_dir(self, root: Path) -> Path:
        safe = self.job_id.replace("/", "__")
        return root / "outputs" / "validation" / safe

    def comparison_dir(self, root: Path) -> Path:
        safe = self.job_id.replace("/", "__")
        return root / "outputs" / "comparisons" / safe


@dataclass(frozen=True)
class PackManifest:
    jobs: list[ReproJob]
    checksums: dict[str, str]
    raw: dict[str, Any]

    @classmethod
    def load(cls, root: Path) -> "PackManifest":
        path = root / "data" / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            jobs=[ReproJob.from_dict(item) for item in payload["jobs"]],
            checksums=dict(payload.get("checksums", {})),
            raw=payload,
        )

    def jobs_for_group(self, group: str) -> list[ReproJob]:
        return [job for job in self.jobs if job.group == group]

    def verify_checksums(self, root: Path | None = None) -> None:
        if root is None:
            root = Path(__file__).resolve().parents[1]
        for rel_path, expected in sorted(self.checksums.items()):
            path = root / rel_path
            if not path.exists():
                raise FileNotFoundError(f"Expected checksummed file to exist. Provided value: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    "Expected SHA256 checksum to match manifest. "
                    f"Provided value: path={path}, expected={expected}, actual={actual}."
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
'''


CONFIG_PY = r'''
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    dims: list[int]
    non_linearity: str
    weight_gains: list[float]
    weight_min: float
    weight_max: float
    input_gain: float
    voltage_amp: float
    current_amp: float
    batch_size: int
    num_iterations: int
    seed: int | None
    bias_scale_mode: str
    bias_interaction_type: str
    signed_weights: bool
    quadratic_diode_param: dict[str, Any]
    exponential_diode_param: dict[str, Any]
    hard_sigmoid_param: dict[str, Any]
    double_diode_updater: str | None
    single_diode_updater: str | None
    adaptive_equilibrium: bool
    rel_tol: float
    vn_tol: float
    use_polish: bool
    max_newton_iters: int
    z_thresh: float
    exp_clip: float
    minimizer_impl: str
    damping: float
    overrelaxation_factor: float
    experimental_newton_max_steps: int
    experimental_newton_tol: float
    iv_data_path: str | None


def load_runtime_config(path: Path, *, pack_root: Path, num_iterations: int | None = None) -> RuntimeConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    non_linearity = _required_str(data, "non_linearity")
    quadratic = _required_dict(data, "quadratic_diode_param")
    exponential = _required_dict(data, "exponential_diode_param")
    hard_sigmoid = _required_dict(data, "hard_sigmoid_param")
    if non_linearity == "double_diode_exponential":
        _require_keys(exponential, ("I_s", "V_t", "V_off"), "exponential_diode_param")
    if non_linearity == "single_diode_exponential":
        _require_keys(exponential, ("I_s", "V_t", "V_off"), "exponential_diode_param")
    if non_linearity in ("hard_sigmoid", "double_diode"):
        _require_keys(hard_sigmoid, ("g_on", "g_off", "v_min", "v_max"), "hard_sigmoid_param")

    dims = data.get("dims")
    if not isinstance(dims, list) or len(dims) < 2:
        raise ValueError(f"Expected config 'dims' to be a list with at least two entries. Provided value: {dims!r}.")

    iv_path = data.get("iv_data_path") or data.get("LABS_IV_CURVE_PATH")
    if iv_path is not None:
        candidate = Path(iv_path)
        if not candidate.is_absolute():
            candidate = pack_root / candidate
        iv_path = str(candidate)

    return RuntimeConfig(
        dims=[int(item) for item in dims],
        non_linearity=non_linearity,
        weight_gains=[float(item) for item in _required_list(data, "weight_gains")],
        weight_min=float(_required(data, "weight_min")),
        weight_max=float(_required(data, "weight_max")),
        input_gain=float(_required(data, "input_gain")),
        voltage_amp=float(_required(data, "voltage_amp")),
        current_amp=float(_required(data, "current_amp")),
        batch_size=int(data.get("batch_size", 1)),
        num_iterations=int(num_iterations if num_iterations is not None else _required(data, "num_iterations")),
        seed=int(data["seed"]) if data.get("seed") is not None else None,
        bias_scale_mode=str(data.get("bias_scale_mode", "legacy")),
        bias_interaction_type=str(data.get("bias_interaction_type", "linear")),
        signed_weights=bool(data.get("signed_weights", False)),
        quadratic_diode_param=quadratic,
        exponential_diode_param=exponential,
        hard_sigmoid_param=hard_sigmoid,
        double_diode_updater=data.get("double_diode_updater"),
        single_diode_updater=data.get("single_diode_updater"),
        adaptive_equilibrium=bool(_required(data, "adaptive_equilibrium")),
        rel_tol=float(data.get("rel_tol", 1e-5)),
        vn_tol=float(data.get("vn_tol", 1e-6)),
        use_polish=bool(data.get("use_polish", True)),
        max_newton_iters=int(data.get("max_newton_iters", 32)),
        z_thresh=float(data.get("z_thresh", 1e10)),
        exp_clip=float(_required(data, "exp_clip")) if non_linearity in ("single_diode_exponential", "double_diode_exponential") else float(data.get("exp_clip", 100000.0)),
        minimizer_impl=str(_required(data, "minimizer_impl")),
        damping=float(data.get("damping", 0.5)),
        overrelaxation_factor=float(data.get("overrelaxation_factor", 1.1)),
        experimental_newton_max_steps=int(data.get("experimental_newton_max_steps", 100)),
        experimental_newton_tol=float(data.get("experimental_newton_tol", 1e-5)),
        iv_data_path=iv_path,
    )


def parse_layer_shapes(dims: list[int]) -> tuple[int, list[int], int, list[tuple[int, ...]]]:
    if dims[0] % 2 != 0:
        raise ValueError(f"Expected input layer size to be divisible by 2. Provided value: {dims[0]!r}.")
    shapes = [(int(dim),) for dim in dims]
    return dims[0] // 2, [int(item) for item in dims[1:-1]], int(dims[-1]), shapes


def _required(data: dict[str, Any], name: str) -> Any:
    if name not in data or data[name] is None:
        raise ValueError(f"Expected config field '{name}' to be present. Provided value: {data.get(name)!r}.")
    return data[name]


def _required_str(data: dict[str, Any], name: str) -> str:
    value = _required(data, name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected config field '{name}' to be a non-empty string. Provided value: {value!r}.")
    return value


def _required_dict(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = _required(data, name)
    if not isinstance(value, dict):
        raise ValueError(f"Expected config field '{name}' to be an object. Provided value: {value!r}.")
    return dict(value)


def _required_list(data: dict[str, Any], name: str) -> list[Any]:
    value = _required(data, name)
    if not isinstance(value, list):
        raise ValueError(f"Expected config field '{name}' to be a list. Provided value: {value!r}.")
    return list(value)


def _require_keys(data: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Expected config '{label}' to include keys {keys}. Provided value missing: {missing}.")

'''


DIGITS_VALIDATE_PY = r'''
from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset, random_split

from networks.resistive.solver import CustomQuadraticMinimizer, MinimizerSettings
from networks.function.cost import SquaredError, SquaredErrorPairedOutputs
from networks.function.network import Network
from networks.resistive.network import DeepResistiveEnergy
from repro.config import RuntimeConfig, load_runtime_config, parse_layer_shapes
from repro.manifest import ReproJob


DEFAULT_NUM_POINTS = 2000
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)


@dataclass(frozen=True)
class ValidationResult:
    run_dir: Path
    states_npz: Path
    metadata_json: Path
    accuracy: float


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def run_validation(pack_root: Path, job: ReproJob, *, device: str = "cpu") -> ValidationResult:
    cfg = load_runtime_config(job.config_path(pack_root), pack_root=pack_root, num_iterations=job.num_iterations)
    _set_seed(cfg.seed)
    torch_device = _resolve_device(device)
    run_dir = job.output_dir(pack_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = _digits_loaders(cfg.batch_size, torch_device, cfg.seed)
    del train_loader

    energy_fn, network, free_layers, cost_fn, _, layer_shapes = _build_energy_stack(
        cfg=cfg,
        weights_path=job.weights_path(pack_root),
        device=torch_device,
    )
    minimizer = _build_minimizer(cfg, energy_fn, free_layers)

    inputs_batches = []
    labels_batches = []
    indices_batches = []
    states = {layer.name: [] for layer in network.layers()}
    total = 0
    correct = 0

    for x, y, idx in test_loader:
        network.set_input(x, reset=True)
        minimizer.compute_equilibrium()

        inputs_batches.append(x.detach().cpu())
        labels_batches.append(y.detach().cpu())
        indices_batches.append(idx.detach().cpu())
        for layer in network.layers():
            states[layer.name].append(layer.state.detach().cpu())

        cost_fn.set_target(y)
        errors = cost_fn.error_fn()
        total += int(errors.numel())
        correct += int(errors.numel()) - int(errors.sum().item())

    inputs_npz = run_dir / "validation_inputs.npz"
    states_npz = run_dir / "validation_states.npz"
    np.savez(
        inputs_npz,
        inputs=torch.cat(inputs_batches, dim=0).numpy(),
        labels=torch.cat(labels_batches, dim=0).numpy(),
        indices=torch.cat(indices_batches, dim=0).numpy(),
    )
    np.savez(states_npz, **{name: torch.cat(values, dim=0).numpy() for name, values in states.items()})

    accuracy = correct / total if total else 0.0
    metadata = {
        "job_id": job.job_id,
        "group": job.group,
        "family": job.family,
        "hidden_layers": job.hidden_layers,
        "hidden_size": job.hidden_size,
        "config": job.config,
        "weights": job.weights,
        "reference_npz": job.reference_npz,
        "dims": cfg.dims,
        "layer_shapes": [list(shape) for shape in layer_shapes],
        "device": str(torch_device),
        "batch_size": cfg.batch_size,
        "num_iterations": cfg.num_iterations,
        "rel_tol": cfg.rel_tol,
        "vn_tol": cfg.vn_tol,
        "exp_clip": cfg.exp_clip,
        "max_newton_iters": cfg.max_newton_iters,
        "overrelaxation_factor": cfg.overrelaxation_factor,
        "adaptive_equilibrium": cfg.adaptive_equilibrium,
        "validation_accuracy": accuracy,
        "validation_correct": correct,
        "validation_total": total,
        "validation_inputs": _pack_rel(inputs_npz, pack_root),
        "validation_states": _pack_rel(states_npz, pack_root),
    }
    metadata_json = run_dir / "validation_metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ValidationResult(run_dir=run_dir, states_npz=states_npz, metadata_json=metadata_json, accuracy=accuracy)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _pack_rel(path: Path, pack_root: Path) -> str:
    try:
        return str(path.relative_to(pack_root))
    except ValueError:
        return str(path)


def _resolve_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Expected CUDA to be available for --device cuda. Provided value: cuda.")
    return torch.device(value)


def _digits_loaders(batch_size: int, device: torch.device, seed: int | None):
    digits = load_digits()
    x = digits.data
    y = digits.target
    if DEFAULT_NUM_POINTS < x.shape[0]:
        rng = np.random.RandomState(seed or 0)
        idx = rng.permutation(x.shape[0])[:DEFAULT_NUM_POINTS]
        x = x[idx]
        y = y[idx]
    x_t = torch.tensor(x, dtype=torch.float32, device=device) / 16.0 * 2.0 - 1.0
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    dataset = TensorDataset(x_t, y_t)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed or 0))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(IndexedDataset(test_ds), batch_size=batch_size, shuffle=False),
    )


def _build_energy_stack(*, cfg: RuntimeConfig, weights_path: Path, device: torch.device):
    input_dim, hidden_dims, output_dim, layer_shapes = parse_layer_shapes(cfg.dims)
    energy_fn = DeepResistiveEnergy(
        layer_shapes=layer_shapes,
        weight_gains=cfg.weight_gains,
        input_gain=cfg.input_gain,
        non_linearity=cfg.non_linearity,
        exponential_diode_param=cfg.exponential_diode_param,
        quadratic_diode_param=cfg.quadratic_diode_param,
        hard_sigmoid_param=cfg.hard_sigmoid_param,
        voltage_amp=cfg.voltage_amp,
        current_amp=cfg.current_amp,
        weight_min=cfg.weight_min,
        weight_max=cfg.weight_max,
        bias_scale_mode=cfg.bias_scale_mode,
        bias_interaction_type=cfg.bias_interaction_type,
        signed_weights=cfg.signed_weights,
    )
    energy_fn.set_device(device)
    energy_fn.load(weights_path)
    network = Network(energy_fn)
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape[0] == 10:
        cost_fn = SquaredError(output_layer)
    elif output_layer.shape[0] == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, 10)
    else:
        raise ValueError(f"Expected output layer width 10 or 20 for digits. Provided value: {output_layer.shape!r}.")
    return energy_fn, network, network.free_layers(), cost_fn, output_layer, layer_shapes


def _build_minimizer(cfg: RuntimeConfig, energy_fn, free_layers):
    settings = MinimizerSettings(
        rel_tol=cfg.rel_tol,
        vn_tol=cfg.vn_tol,
        use_polish=cfg.use_polish,
        max_newton_iters=cfg.max_newton_iters,
        z_thresh=cfg.z_thresh,
        exp_clip=cfg.exp_clip,
        experimental_newton_tol=cfg.experimental_newton_tol,
    )
    if cfg.minimizer_impl != "custom":
        raise ValueError(f"Expected minimizer_impl to be 'custom'. Provided value: {cfg.minimizer_impl!r}.")
    return CustomQuadraticMinimizer(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=cfg.num_iterations,
        mode="asynchronous",
        non_linearity=cfg.non_linearity,
        quadratic_diode_param=cfg.quadratic_diode_param,
        exponential_diode_param=cfg.exponential_diode_param,
        voltage_amp=energy_fn.voltage_amp,
        current_amp=energy_fn.current_amp,
        iv_data=None,
        iv_data_path=cfg.iv_data_path,
        double_diode_updater=cfg.double_diode_updater,
        adaptive_equilibrium=cfg.adaptive_equilibrium,
        overrelaxation_factor=cfg.overrelaxation_factor,
        single_diode_updater=cfg.single_diode_updater,
        damping=cfg.damping,
        experimental_newton_max_steps=cfg.experimental_newton_max_steps,
        minimizer_settings=settings,
    )

def _layer_sort_key(name: str):
    digits = "".join(ch for ch in name if ch.isdigit())
    return (0, int(digits)) if digits else (1, name)


'''


NPZ_COMPARE_PY = r'''
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from repro.manifest import ReproJob


PERCENTILES = (50, 60, 70, 80, 90, 95, 99)


def compare_job(pack_root: Path, job: ReproJob) -> dict | None:
    reference = job.reference_path(pack_root)
    if reference is None:
        return None
    cd_npz = job.output_dir(pack_root) / "validation_states.npz"
    if not cd_npz.exists():
        raise FileNotFoundError(f"Expected validation_states.npz before comparison. Provided value: {cd_npz}")
    out_dir = job.comparison_dir(pack_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    return compare_npz(cd_npz, reference, out_dir, pack_root=pack_root)


def compare_npz(cd_npz: Path, reference_npz: Path, output_dir: Path, *, pack_root: Path | None = None) -> dict:
    with np.load(cd_npz, allow_pickle=False) as cd_data, np.load(reference_npz, allow_pickle=False) as ref_data:
        pairs = _comparison_layer_pairs(cd_data.files, ref_data.files, cd_npz, reference_npz)
        total_nodes = 0
        total_mae = None
        total_ref = None
        per_layer = {}
        layer_labels = []
        for cd_layer, ref_layer in pairs:
            label = cd_layer if cd_layer == ref_layer else f"{cd_layer}->{ref_layer}"
            layer_labels.append(label)
            cd = _flatten(cd_data[cd_layer])
            ref = _flatten(ref_data[ref_layer])
            if cd.shape != ref.shape:
                raise ValueError(f"Expected matching shapes for {label}. Provided value: {cd.shape} vs {ref.shape}.")
            node_count = cd.shape[1]
            total_nodes += node_count
            rel_l1 = np.mean(np.abs(cd - ref), axis=1) / (np.mean(np.abs(ref), axis=1) + 1e-12)
            per_layer[label] = {f"p{p}": float(np.percentile(rel_l1, p)) for p in PERCENTILES}
            mae = np.mean(np.abs(cd - ref), axis=1)
            ref_abs = np.mean(np.abs(ref), axis=1)
            total_mae = mae * node_count if total_mae is None else total_mae + mae * node_count
            total_ref = ref_abs * node_count if total_ref is None else total_ref + ref_abs * node_count

    node_weighted = total_mae / (total_ref + 1e-12)
    payload = {
        "cd_npz": _display_path(cd_npz, pack_root),
        "reference_npz": _display_path(reference_npz, pack_root),
        "layers": layer_labels,
        "total_nodes": int(total_nodes),
        "node_weighted_rel_l1_percentiles": {f"p{p}": float(np.percentile(node_weighted, p)) for p in PERCENTILES},
        "per_layer_rel_l1_percentiles": per_layer,
    }
    path = output_dir / "cross_layer_rel_l1_percentiles_node_weighted.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _comparison_layer_pairs(cd_files: list[str], ref_files: list[str], cd_npz: Path, reference_npz: Path) -> list[tuple[str, str]]:
    common = sorted(set(cd_files) & set(ref_files), key=_layer_sort_key)
    if common:
        return [(layer, layer) for layer in common]

    cd_layers = _state_layers(cd_files)
    ref_layers = _state_layers(ref_files)
    if len(cd_layers) == len(ref_layers):
        return list(zip(cd_layers, ref_layers))
    if len(cd_layers) > len(ref_layers):
        # SPICE/reference exports often omit the clamped input layer.  Align the
        # trailing validation layers so free/output states compare by position.
        return list(zip(cd_layers[-len(ref_layers):], ref_layers))
    raise ValueError(f"Expected comparable layer keys in NPZ files. Provided value: {cd_npz}, {reference_npz}.")


def _state_layers(names: list[str]) -> list[str]:
    return sorted(
        [
            name for name in names
            if name.startswith("Layer_") and "Node_Order" not in name
        ],
        key=_layer_sort_key,
    )


def _flatten(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values[:, None]
    return values.reshape(values.shape[0], -1)


def _layer_sort_key(name: str):
    digits = "".join(ch for ch in name if ch.isdigit())
    return (0, int(digits)) if digits else (1, name)


def _display_path(path: Path, pack_root: Path | None) -> str:
    if pack_root is None:
        return str(path)
    try:
        return str(path.relative_to(pack_root))
    except ValueError:
        return str(path)
'''


AGGREGATE_PY = r'''
from __future__ import annotations

import json
from pathlib import Path

from repro.manifest import PackManifest, ReproJob


def aggregate_error_vs_iter(pack_root: Path) -> Path:
    manifest = PackManifest.load(pack_root)
    grouped: dict[str, dict[str, list[dict]]] = {}
    for job in manifest.jobs_for_group("error_vs_iter"):
        summary = _read_comparison(pack_root, job)
        if summary is None:
            continue
        family = job.family
        hidden = f"hidden_{job.hidden_layers}"
        grouped.setdefault(family, {}).setdefault(hidden, []).append(
            {
                "iterations": job.num_iterations,
                **summary["node_weighted_rel_l1_percentiles"],
                "source": _pack_rel(job.comparison_dir(pack_root), pack_root),
            }
        )
    out_dir = pack_root / "outputs" / "summaries" / "error_vs_iter"
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, hidden in grouped.items():
        for rows in hidden.values():
            rows.sort(key=lambda item: item["iterations"])
        (out_dir / f"{family}.json").write_text(
            json.dumps({"non_linearity": family, "hidden": hidden}, indent=2),
            encoding="utf-8",
        )
    return out_dir


def aggregate_vol_tol(pack_root: Path) -> Path:
    manifest = PackManifest.load(pack_root)
    grouped: dict[str, dict[str, list[dict]]] = {}
    for job in manifest.jobs_for_group("vol_tol"):
        summary = _read_comparison(pack_root, job)
        if summary is None:
            continue
        family = job.family
        hidden = f"hidden_{job.hidden_layers}"
        grouped.setdefault(family, {}).setdefault(hidden, []).append(
            {
                "rel_tol": job.rel_tol,
                "rel_tol_value": float(job.rel_tol),
                "p90": summary["node_weighted_rel_l1_percentiles"]["p90"],
                "source": _pack_rel(job.comparison_dir(pack_root), pack_root),
            }
        )
    out_dir = pack_root / "outputs" / "summaries" / "vol_tol"
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, hidden in grouped.items():
        for rows in hidden.values():
            rows.sort(key=lambda item: item["rel_tol_value"])
        (out_dir / f"{family}.json").write_text(
            json.dumps({"non_linearity": family, "hidden": hidden}, indent=2),
            encoding="utf-8",
        )
    return out_dir


def _read_comparison(pack_root: Path, job: ReproJob) -> dict | None:
    path = job.comparison_dir(pack_root) / "cross_layer_rel_l1_percentiles_node_weighted.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pack_rel(path: Path, pack_root: Path) -> str:
    try:
        return str(path.relative_to(pack_root))
    except ValueError:
        return str(path)
'''


PLOTS_PY = r'''
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FAMILY_LABELS = {
    "single_diode_exponential": "Single diode",
    "double_diode_exponential": "Double diode",
    "experimental": "PWL I-V",
}


def regenerate_all_figures(pack_root: Path) -> None:
    out = pack_root / "outputs"
    for family in FAMILY_LABELS:
        timing_csv = pack_root / "data" / "timing" / "figure_inputs" / family / "combined_latest_by_hidden.csv"
        if timing_csv.exists():
            plot_timing(timing_csv, out / "figures" / "timing" / f"{family}_cpu_spice_vs_coordinate_descent_loglog.png")

        err = pack_root / "data" / "error_vs_iter" / "summaries" / family / "error_vs_iter_summary_20260302.json"
        if err.exists():
            plot_error_vs_iter(err, out / "figures" / "error_vs_iter" / f"{family}_p90.png")

        summaries = list((pack_root / "data" / "vol_tol" / "summaries" / family).glob("vol_tol_summary*.json"))
        if summaries:
            plot_vol_tol(summaries[0], out / "figures" / "vol_tol" / f"{family}_p90_vs_rel_tol.png")

    tables_out = out / "tables"
    tables_out.mkdir(parents=True, exist_ok=True)
    for src in (pack_root / "data" / "timing" / "tables").glob("*"):
        if src.suffix.lower() in (".csv", ".md", ".tex", ".json"):
            shutil.copy2(src, tables_out / src.name)


def plot_timing(csv_path: Path, output: Path) -> None:
    rows = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for depth in sorted({int(row["depth"]) for row in rows}):
        sub = sorted([row for row in rows if int(row["depth"]) == depth], key=lambda row: int(row["width"]))
        xs = [int(row["width"]) for row in sub]
        cd = [_float(row.get("coord_user_time_seconds") or row.get("coord_validation_total_seconds")) for row in sub]
        spice = [_float(row.get("spice_total_seconds")) for row in sub]
        ax.plot(xs, cd, marker="o", linewidth=2, label=f"{depth} hidden CD")
        if any(value is not None for value in spice):
            ax.plot(xs, spice, marker="s", linestyle="--", linewidth=2, label=f"{depth} hidden SPICE")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Hidden layer size")
    ax.set_ylabel("Time (s)")
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, output)


def plot_error_vs_iter(summary_path: Path, output: Path) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for hidden, rows in sorted(payload["hidden"].items()):
        xs = [int(row.get("iterations", row.get("iteration"))) for row in rows]
        ys = [float(row["p90"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=hidden.replace("_", " "))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Coordinate-descent iterations")
    ax.set_ylabel("Node-weighted rel. L1 p90")
    ax.set_title(FAMILY_LABELS.get(payload.get("non_linearity"), payload.get("non_linearity", "")))
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output)


def plot_vol_tol(summary_path: Path, output: Path) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for hidden, rows in sorted(payload["hidden"].items()):
        xs = [float(row["rel_tol_value"]) for row in rows]
        ys = [float(row["p90"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=hidden.replace("_", " "))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Relative voltage tolerance")
    ax.set_ylabel("Node-weighted rel. L1 p90")
    ax.set_title(FAMILY_LABELS.get(payload.get("non_linearity"), payload.get("non_linearity", "")))
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
'''


README_MD = r'''
# Digits Core Reproduction Pack

This pack reproduces the digits-core paper figures without training.  It bundles
the selected checkpoint weights, validation configs, SPICE/reference NPZ files,
curated timing CSVs, and paper table inputs.

The pack is self-contained for digits-core validation and figure/table
regeneration once the Python dependencies are installed.  It does not include
training workflows, Optuna/TensorBoard artifacts, or SPICE regeneration code.
The vendored minimizer is a validation-only subset for the three bundled
digits-core nonlinearities.

## Setup

Use Python 3.10+ with the packages in `requirements.txt`.  A CPU-only PyTorch
install is enough for smoke tests; CUDA is useful for full validation reruns.

```bash
python -m pip install -r requirements.txt
```

## Commands

List available jobs:

```bash
python scripts/reproduce.py list
```

Regenerate figures and tables from bundled curated inputs:

```bash
python scripts/reproduce.py figures
```

Run one CPU validation smoke test and compare it to its reference NPZ:

```bash
python scripts/reproduce.py validate --group timing --device cpu --limit 1
python scripts/reproduce.py compare --group timing --limit 1
```

Run all validation jobs on CUDA:

```bash
python scripts/reproduce.py all --device cuda
```

Outputs are written under `outputs/`.  Timing plots use bundled curated timing
CSVs because wall-clock timings are machine-dependent.
'''


REQUIREMENTS_TXT = """\
matplotlib>=3.8
numpy>=1.26
scikit-learn>=1.4
scipy>=1.12
torch>=2.1
"""


if __name__ == "__main__":
    raise SystemExit(main())
