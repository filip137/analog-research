from __future__ import annotations

from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

import pytest

from ebl.cli import main
from experiments.artifacts import RunStore
from experiments.study_workflow import (
    StudyWorkflowError,
    finalize_study,
    load_review,
    load_study_plan,
    prepare_study,
    summarize_study,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _plan(config: Path, *, study_id: str = "workflow-smoke-v1") -> dict:
    return {
        "schema_version": 1,
        "study_id": study_id,
        "title": "Workflow smoke",
        "hypothesis": "The declared treatment improves the selected metric.",
        "motivation": "Exercise the result pipeline without numerical work.",
        "evidence_class": "exploratory",
        "arms": [
            {
                "arm_id": "baseline",
                "description": "One declared baseline run.",
                "experiment_id": "small_drn.v1",
                "mode": "train",
                "configs": [str(config)],
            }
        ],
        "completion_criteria": ["The declared run completes with valid artifacts."],
        "analysis_plan": ["Compare the terminal metric with the hypothesis."],
    }


def _prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "config.json"
    _write_json(config, {"scientific_setting": 1})
    plan = tmp_path / "study-plan.json"
    _write_json(plan, _plan(config))
    root = prepare_study(plan, tmp_path / "results")
    return root, plan, config


def _complete_run(root: Path, config: Path, *, payload: bytes = b"payload") -> Path:
    output_root = root / "runs" / "baseline"
    store = RunStore.create(
        output_root=output_root,
        experiment_id="small_drn.v1",
        resolved_config={"scientific_setting": 1},
        command=(
            "ebl",
            "train",
            f"--config={config}",
            "--output-dir",
            str(output_root),
        ),
        repo_root=tmp_repo(root),
        run_id="run-001",
    )
    store.append_metric({"mode": "train", "metric": 0.5})
    artifact = store.run_dir / "artifacts" / "payload.bin"
    artifact.write_bytes(payload)
    store.complete(
        metrics={"metric": 0.5},
        artifacts=(store.artifact_record(artifact, kind="payload"),),
    )
    return store.run_dir


def tmp_repo(study_root: Path) -> Path:
    # A Git repository is optional for RunStore provenance.  Returning the
    # temporary test root keeps current-simulation refreshes isolated.
    return study_root.parents[1]


def test_plan_is_strict_and_requires_an_initial_hypothesis(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _write_json(config, {})
    payload = _plan(config)
    payload.pop("hypothesis")
    plan = tmp_path / "plan.json"
    _write_json(plan, payload)

    with pytest.raises(StudyWorkflowError, match="keys to be exactly"):
        load_study_plan(plan)

    plan.write_text(
        '{"schema_version": 1, "schema_version": 1}\n',
        encoding="utf-8",
    )
    with pytest.raises(StudyWorkflowError, match="duplicate keys"):
        load_study_plan(plan)

    payload = _plan(config)
    payload["schema_version"] = True
    _write_json(plan, payload)
    with pytest.raises(StudyWorkflowError, match="schema_version to be 1"):
        load_study_plan(plan)


def test_prepare_links_native_run_and_summarizes_metadata(tmp_path: Path) -> None:
    root, plan, config = _prepare(tmp_path)

    assert prepare_study(plan, tmp_path / "results") == root
    study = json.loads((root / "study.json").read_text(encoding="utf-8"))
    assert study["hypothesis"].startswith("The declared treatment")
    assert (root / "runs" / "baseline").is_dir()
    readme = (root / "README.md").read_text()
    assert "historical global" in readme
    assert "experimental_manifest.md" not in readme
    assert "After human" not in readme

    run_dir = _complete_run(root, config)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["study"]["study_id"] == "workflow-smoke-v1"
    assert manifest["study"]["arm_id"] == "baseline"

    summary = summarize_study(root)
    assert summary["state"] == "ready_for_review"
    assert summary["ready_for_review"] is True
    assert summary["validation_mode"] == "metadata_only"
    assert summary["arms"][0]["complete"] == 1
    assert summary["runs"][0]["metrics"] == {"metric": 0.5}
    report = (root / "analysis" / "report.md").read_text(encoding="utf-8")
    assert "## Initial hypothesis" in report
    assert "The declared treatment improves" in report


def test_summary_reports_missing_declared_arm_without_recreating_it(
    tmp_path: Path,
) -> None:
    root, _plan_path, _config = _prepare(tmp_path)
    arm_root = root / "runs" / "baseline"
    arm_root.rmdir()

    summary = summarize_study(root)

    assert summary["state"] == "invalid"
    assert summary["missing_arm_directories"] == ["baseline"]
    assert not arm_root.exists()
    assert "expected a declared arm directory" in (
        root / "analysis" / "report.md"
    ).read_text(encoding="utf-8")


def test_prepare_rejects_a_tampered_materialized_contract(tmp_path: Path) -> None:
    root, plan, _config = _prepare(tmp_path)
    study_path = root / "study.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["hypothesis"] = "A post-prepare replacement hypothesis."
    _write_json(study_path, study)

    with pytest.raises(StudyWorkflowError, match="tracked source plan exactly"):
        prepare_study(plan, tmp_path / "results")


def test_run_rejects_config_not_declared_by_the_study(tmp_path: Path) -> None:
    root, _plan_path, _config = _prepare(tmp_path)
    other = tmp_path / "other.json"
    _write_json(other, {"scientific_setting": 2})

    with pytest.raises(StudyWorkflowError, match="predeclared"):
        RunStore.create(
            output_root=root / "runs" / "baseline",
            experiment_id="small_drn.v1",
            resolved_config={"scientific_setting": 2},
            command=("ebl", "train", "--config", str(other)),
            repo_root=tmp_path,
        )
    assert list((root / "runs" / "baseline").iterdir()) == []


def test_full_artifact_verification_is_opt_in(tmp_path: Path) -> None:
    root, _plan_path, config = _prepare(tmp_path)
    run_dir = _complete_run(root, config, payload=b"first")
    (run_dir / "artifacts" / "payload.bin").write_bytes(b"other")

    metadata_only = summarize_study(root)
    assert metadata_only["ready_for_review"] is True

    fully_verified = summarize_study(root, verify_artifacts=True)
    assert fully_verified["ready_for_review"] is False
    assert fully_verified["state"] == "invalid"
    assert "SHA-256" in " ".join(fully_verified["runs"][0]["errors"])


def test_finalize_records_interpretation_and_is_idempotent(tmp_path: Path) -> None:
    root, _plan_path, config = _prepare(tmp_path)
    _complete_run(root, config)
    review = root / "analysis" / "review.json"
    _write_json(
        review,
        {
            "schema_version": 2,
            "outcome": "supported",
            "manifest_interpretation": (
                "The smoke result supports the declared treatment, subject to "
                "one-run exploratory coverage."
            ),
            "final_interpretation": "The result supports the initial hypothesis.",
            "limitations": "This is a one-run exploratory workflow smoke.",
            "next_steps": ["Repeat with the real scientific configs."],
        },
    )
    manifest = tmp_path / "docs" / "experimental_manifest.md"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(
        "# Finished studies\n\n## Finished simulations\n\n"
        "## Shared validity notes\n\nKeep existing notes.\n",
        encoding="utf-8",
    )

    stdout = StringIO()
    stderr = StringIO()
    code = main(
        [
            "study",
            "finalize",
            "--study-dir",
            str(root),
            "--review",
            str(review),
            "--manifest",
            str(manifest),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    final_path = Path(stdout.getvalue().strip())
    first_document = manifest.read_text(encoding="utf-8")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    report = (root / "analysis" / "report.md").read_text(encoding="utf-8")
    assert final["schema_version"] == 2
    assert final["outcome"] == "supported"
    assert final["manifest_interpretation"].startswith("The smoke result")
    assert first_document == (
        "# Finished studies\n\n## Finished simulations\n\n"
        "## Shared validity notes\n\nKeep existing notes.\n"
    )
    final_before = final_path.read_bytes()
    assert "manifest" not in final
    assert summarize_study(root)["state"] == "reviewed"

    assert (
        finalize_study(root, review_path=review, manifest_path=manifest)
        == final_path
    )
    assert manifest.read_text(encoding="utf-8") == first_document


def test_review_v2_requires_a_short_single_paragraph_manifest_text(
    tmp_path: Path,
) -> None:
    review = tmp_path / "review.json"
    base = {
        "schema_version": 2,
        "outcome": "supported",
        "manifest_interpretation": "A concise supported result with a boundary.",
        "final_interpretation": "The complete interpretation.",
        "limitations": "One exploratory run.",
        "next_steps": ["Repeat it."],
    }
    _write_json(review, base)
    assert load_review(review)["manifest_interpretation"].startswith("A concise")

    multiline = dict(base, manifest_interpretation="First line.\nSecond line.")
    _write_json(review, multiline)
    with pytest.raises(StudyWorkflowError, match="one paragraph"):
        load_review(review)

    overlength = dict(base, manifest_interpretation="x" * 601)
    _write_json(review, overlength)
    with pytest.raises(StudyWorkflowError, match="at most 600 characters"):
        load_review(review)


def test_new_finalization_rejects_a_legacy_review(tmp_path: Path) -> None:
    root, _plan_path, config = _prepare(tmp_path)
    _complete_run(root, config)
    review = root / "analysis" / "review.json"
    _write_json(
        review,
        {
            "schema_version": 1,
            "outcome": "supported",
            "final_interpretation": "The legacy review supports the result.",
            "limitations": "One run.",
            "next_steps": ["Repeat it."],
        },
    )
    manifest = tmp_path / "docs" / "experimental_manifest.md"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text("# Experimental Manifest\n", encoding="utf-8")

    with pytest.raises(StudyWorkflowError, match="review schema version 2"):
        finalize_study(root, review_path=review, manifest_path=manifest)


def test_summary_renders_an_existing_v1_final_record(tmp_path: Path) -> None:
    root, _plan_path, config = _prepare(tmp_path)
    _complete_run(root, config)
    summarize_study(root)
    study_sha = sha256((root / "study.json").read_bytes()).hexdigest()
    _write_json(
        root / "analysis" / "final.json",
        {
            "schema": "ebl.study.final",
            "schema_version": 1,
            "study_id": "workflow-smoke-v1",
            "study_sha256": study_sha,
            "outcome": "mixed",
            "final_interpretation": "The legacy result remains mixed.",
            "limitations": "Legacy single-run evidence.",
            "next_steps": ["Retain the legacy conclusion."],
        },
    )

    summary = summarize_study(root)
    report = (root / "analysis" / "report.md").read_text(encoding="utf-8")
    assert summary["state"] == "reviewed"
    assert "The legacy result remains mixed." in report
    assert "### Manifest interpretation" not in report


def test_refinalizing_v1_preserves_final_and_migrated_manifest_summary(
    tmp_path: Path,
) -> None:
    root, _plan_path, config = _prepare(tmp_path)
    _complete_run(root, config)
    review = root / "analysis" / "review.json"
    _write_json(
        review,
        {
            "schema_version": 1,
            "outcome": "mixed",
            "final_interpretation": "The legacy result remains mixed.",
            "limitations": "Legacy single-run evidence.",
            "next_steps": ["Retain the legacy conclusion."],
        },
    )
    summarize_study(root)
    study_sha = sha256((root / "study.json").read_bytes()).hexdigest()
    final_path = root / "analysis" / "final.json"
    _write_json(
        final_path,
        {
            "schema": "ebl.study.final",
            "schema_version": 1,
            "study_id": "workflow-smoke-v1",
            "finalized_at": "2026-08-01T00:00:00+00:00",
            "study_sha256": study_sha,
            "review_sha256": sha256(review.read_bytes()).hexdigest(),
            "outcome": "mixed",
            "final_interpretation": "The legacy result remains mixed.",
            "limitations": "Legacy single-run evidence.",
            "next_steps": ["Retain the legacy conclusion."],
        },
    )
    final_before = final_path.read_bytes()
    concise_entry = (
        "<!-- BEGIN EBL STUDY SUMMARY workflow-smoke-v1 -->\n"
        "### [workflow-smoke-v1](../results/workflow-smoke-v1/)\n\n"
        "- **Outcome:** mixed\n"
        "- **Interpretation:** A preserved migration-time summary.\n"
        "<!-- END EBL STUDY SUMMARY workflow-smoke-v1 -->"
    )
    manifest = tmp_path / "docs" / "experimental_manifest.md"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(
        "# Experimental Manifest\n\n## Concluded studies\n\n"
        + concise_entry
        + "\n\n<details>\n"
        "<summary>Full study record and provenance</summary>\n\n"
        "<!-- BEGIN EBL STUDY workflow-smoke-v1 -->\n"
        "legacy placeholder\n"
        "<!-- END EBL STUDY workflow-smoke-v1 -->\n\n"
        "</details>\n",
        encoding="utf-8",
    )

    assert (
        finalize_study(root, review_path=review, manifest_path=manifest)
        == final_path
    )
    first_document = manifest.read_text(encoding="utf-8")
    assert final_path.read_bytes() == final_before
    assert concise_entry in first_document
    assert "legacy placeholder" in first_document
    assert "The legacy result remains mixed." not in first_document

    finalize_study(root, review_path=review, manifest_path=manifest)
    assert final_path.read_bytes() == final_before
    assert manifest.read_text(encoding="utf-8") == first_document


def test_study_prepare_is_available_from_public_cli(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _write_json(config, {})
    plan = tmp_path / "plan.json"
    _write_json(plan, _plan(config, study_id="cli-study-v1"))
    stdout = StringIO()
    stderr = StringIO()

    code = main(
        [
            "study",
            "prepare",
            "--plan",
            str(plan),
            "--results-root",
            str(tmp_path / "results"),
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0, stderr.getvalue()
    assert Path(stdout.getvalue().strip()).name == "cli-study-v1"
    study_root = tmp_path / "results" / "cli-study-v1"
    assert (study_root / "study.json").is_file()

    stdout = StringIO()
    stderr = StringIO()
    code = main(
        [
            "study",
            "summarize",
            "--study-dir",
            str(study_root),
            "--json",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    assert json.loads(stdout.getvalue())["state"] == "planned"


def test_inspector_preserves_prepared_study_provenance_checks(tmp_path: Path) -> None:
    from experiments.study_workflow import inspect_run
    root, _, config = _prepare(tmp_path)
    run = _complete_run(root, config)
    assert inspect_run(run, verify_artifacts=True)["valid"]
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["study"]["source_config_sha256"] = "0" * 64
    _write_json(path, manifest)
    assert not inspect_run(run)["valid"]


def test_finalize_without_global_document(tmp_path: Path) -> None:
    root, _, config = _prepare(tmp_path)
    _complete_run(root, config)
    review = root / "analysis/review.json"
    value = {"schema_version": 2, "outcome": "mixed",
             "final_interpretation": "The tiny comparison is inconclusive.",
             "limitations": "One case.", "next_steps": ["Retain evidence."]}
    value["manifest_interpretation"] = "Tiny inconclusive comparison."
    _write_json(review, value)
    out, err = StringIO(), StringIO()
    assert main(["study", "finalize", "--study-dir", str(root), "--review", str(review)], stdout=out, stderr=err) == 0, err.getvalue()
    assert Path(out.getvalue().strip()).is_file()
    assert not (tmp_path / "docs").exists()
