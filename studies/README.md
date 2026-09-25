> Workflow migration (2026-09-25): the [question-led policy](../docs/experiment_workflow.md)
> supersedes the legacy planning/global-manifest requirements below. Existing study
> formats remain supported; new pilots use one note and native run bundles.

# Study plans

This directory contains small, tracked scientific plans for new exploratory
studies. Raw runs stay below ignored `results/<study-id>/`; the plan remains in
Git and freezes the initial hypothesis, declared configs, completion criteria,
and analysis plan before execution.

Start with `example-exploratory.json`, give a real study a stable kebab-case
ID, and prepare it with:

```bash
python -m ebl study prepare --plan studies/<study-id>.json
```

One config corresponds to one expected completed run. Put separate seeds or
scientific settings in separate config files. The same config may be retried
after an operational failure, but two successful runs for one declared config
make coverage ambiguous and block finalization.

The review template belongs in `results/<study-id>/analysis/review.json` only
after the declared evidence has been collected and inspected. New closeouts
use review schema version 2: `manifest_interpretation` is a user-authored
one-paragraph summary for the concise manifest, while `final_interpretation`,
limitations, and next steps remain the full scientific closeout rendered in
the study-local report.

The current one-seed deployed-array recovery pilot is declared in
`mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1.json`. It
must create its single source deployment before any of its twelve recovery
arms; all recovery commands then receive that exact sidecar through
`--deployment`.

The remedial one-seed p90/p95/p99 absolute-gradient gate is declared in
`mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1.json`.
Its nine arms reuse the completed pilot's exact source deployment and frozen
unthresholded direct-pulse probability scales; suppressed probabilities are
not redistributed.
