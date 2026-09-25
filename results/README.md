> Workflow migration (2026-09-25): the [question-led policy](../docs/experiment_workflow.md)
> supersedes the legacy planning/global-manifest requirements below. Existing study
> formats remain supported; new pilots use one note and native run bundles.

# Raw LoRA/HWA Results

The repository-root `results/` directory is the canonical home for new raw
LoRA and hardware-aware experiment output in this worktree. Raw contents are
local and ignored by Git; this guide is the only file in the tree intended
for Git.

Use one stable, kebab-case study ID for a scientific comparison. New studies
are prepared from a tracked plan under [`studies/`](../studies/README.md):

```bash
python -m ebl study prepare --plan studies/<study-id>.json
```

The materialized directory has this shape:

```text
results/<study-id>/
  study.json
  README.md
  runs/<arm>/<ebl-run-id>/
  analysis/
  deployment/                 # optional, when the study uses it
```

- `runs/` contains native `python -m ebl` run directories.
- Optional `deployment/` contains device-programming or deployment outputs
  when used.
- `analysis/summary.json` is the machine-readable coverage and terminal-metric
  index.
- `analysis/report.md` is the human-readable study report. It begins as a
  coverage report and gains the reviewed scientific closeout after
  finalization.
- `analysis/review.json` contains the human-authored scientific decision, and
  `analysis/final.json` binds that decision to the finalized study artifacts.
- Other files under `analysis/` contain aggregate JSON, tables, and plots.
- `study.json` and the study-local `README.md` retain the initial hypothesis
  and frozen plan inside the raw result directory.

For prepared workflow studies, `study.json`, `runs/`, and `analysis/` are the
required control structure; `deployment/` remains optional. Older manual
result directories retain their existing layouts.

## Launch example

Pass an arm-specific parent to the public CLI. The runtime creates a new
exclusive run directory below it:

```bash
python -m ebl train \
  --config examples/small_drn/base.json \
  --output-dir results/<study-id>/runs/<arm>
```

Never reuse an existing native run directory. Keep the native files
(`config.resolved.json`, `manifest.json`, `metrics.jsonl`, `result.json`,
`status.json`, artifacts, and checkpoints) together.

Summarize and validate result coverage without loading checkpoints:

```bash
python -m ebl study summarize --study-dir results/<study-id>
```

After scientific review, `python -m ebl study finalize` writes
`analysis/final.json`, completes `analysis/report.md`, and adds a concise
linked interpretation to `docs/experimental_manifest.md`. See the complete
[`experiment workflow`](../docs/experiment_workflow.md).

Everything below `results/` except this README is ignored by Git. Links from
tracked Markdown into a study folder are therefore local links: they resolve
only in a workspace where that ignored study root has been retained or
restored.

## Experiment indexes

- The `Active` block in
  [`docs/current_simulations.md`](../docs/current_simulations.md) is refreshed
  automatically from running native statuses below `results/`. Its paused,
  analyzing, and queued sections remain human-maintained.
- [`docs/experimental_manifest.md`](../docs/experimental_manifest.md) is the
  concise index of concluded studies and links to their full local reports.

The automatic refresh is informational and best-effort. Maintain the
human-owned status and synthesis sections manually; use `study finalize` for
workflow-managed concluded-study entries. None of these documents approves,
validates, or blocks a run.

## Existing raw results

Existing exploratory studies remain at their original locations:

- `labs/cases/passive_layerwise_lora_physical_test/`
- `labs/cases/perfect_diode_hwa_lora_comparison/`
- `labs/cases/mnist_perfect_diode_hwa_lora_comparison/`

They are indexed in the experimental manifest but are not moved into
`results/`.
