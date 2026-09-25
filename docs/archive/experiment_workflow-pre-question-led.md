# Experiment workflow

This is the lightweight lifecycle for new exploratory studies. The existing
`python -m ebl` runners remain the only scientific execution surface. The
study workflow freezes the question before execution, indexes native results
without loading large tensors, and makes the final human interpretation
durable.

## The lifecycle

```text
tracked study plan
      ↓
prepared results/<study-id>/
      ↓
native ebl runs under runs/<arm-id>/
      ↓
metadata-first study summary
      ↓
human scientific review
      ↓
full closeout in analysis/report.md
      ↓
concise entry in experimental_manifest.md
      ↓
human synthesis in current_state.md when the broad picture changes
```

The workflow has no scheduler, database, experiment catalog, or hidden config
generation. Campaigns remain available when a comparison genuinely needs
cross-worktree dependencies. An ordinary study should use direct `ebl`
commands.

## 1. Declare the hypothesis

Create a tracked strict JSON plan under `studies/`. Start from
[`studies/example-exploratory.json`](../studies/example-exploratory.json).
The plan records:

- one initial hypothesis and its motivation;
- the evidence class;
- every arm, experiment ID, mode, and exact config file;
- completion criteria; and
- the analysis that will be performed after coverage is complete.

Config source files are byte-hashed when the plan is prepared; the native run
also records its usual canonical hash of the fully resolved config. A
workflow-managed run fails before creating a run directory when its
experiment, mode, arm, or source-config hash was not declared.

## 2. Prepare the result directory

```bash
python -m ebl study prepare \
  --plan studies/my-study-v1.json \
  --results-root results
```

This creates:

```text
results/<study-id>/
├── study.json                 # materialized plan and initial hypothesis
├── README.md                  # human-readable plan
├── runs/
│   └── <arm-id>/
└── analysis/
```

Preparation is idempotent only while the tracked source-plan hash is
unchanged. Change the study ID when the scientific contract changes.

## 3. Run the declared arms

Pass the arm directory as the normal `--output-dir`:

```bash
python -m ebl train \
  --config examples/small_drn/base.json \
  --output-dir results/<study-id>/runs/<arm-id>
```

Device-characterization studies use the same lifecycle with the dedicated
mode:

```bash
python -m ebl characterize \
  --config examples/reram_program_verify/production_om_continuous.json \
  --output-dir results/<study-id>/runs/<arm-id>
```

The numerical runtime still creates one exclusive native run directory. When
the output path has the canonical study shape, `manifest.json` automatically
records the study ID, arm ID, evidence class, study hash, plan hash, and source
config hash. Runs elsewhere remain ordinary standalone `ebl` runs.

Failed attempts are retained. Do not reuse a native run directory or delete a
failed attempt merely to make coverage look clean.

## 4. Summarize efficiently

```bash
python -m ebl study summarize \
  --study-dir results/<study-id>
```

The default pass reads only `study.json` and each run's small control files:
`manifest.json`, `status.json`, `config.resolved.json`, and `result.json`. It
copies the compact terminal metrics from `result.json`, but does not parse
`metrics.jsonl`, load checkpoints, or hash tensor artifacts. It checks
artifact paths and sizes and writes:

```text
analysis/summary.json
analysis/report.md
```

`summary.json` is the machine-readable coverage and terminal-metric record.
`report.md` is its human-readable companion: before review it shows the
hypothesis, coverage, native-run evidence, and validation problems. After
finalization it also contains the reviewed outcome, interpretation,
limitations, and next steps.

Use `--verify-artifacts` for a one-time collection or archival audit. That
option hashes every artifact and is intentionally slower.

Coverage is ready for review only when each declared config has exactly one
valid completed run, no run remains active, no undeclared completed config is
present, and no bundle is invalid. Retained well-formed failed attempts are
reported but do not hide a valid replacement.

## 5. Review and finalize

Copy [`studies/review.template.json`](../studies/review.template.json) into the
study's `analysis/` directory and write the scientific conclusion yourself.
Review schema version 2 must state the outcome, a one-paragraph concise
`manifest_interpretation`, the full final interpretation, limitations, and
next steps. The short text is an explicit human scientific judgment; the
workflow never truncates or invents it automatically.

```bash
python -m ebl study finalize \
  --study-dir results/<study-id> \
  --review results/<study-id>/analysis/review.json \
  --manifest docs/experimental_manifest.md
```

Finalization reruns the coverage checks, writes `analysis/final.json`, updates
`analysis/report.md` to the reviewed closeout, and adds an idempotent per-study
presentation to `experimental_manifest.md`. The visible presentation links to
the local study folder and report and contains the short interpretation. The
complete hypothesis, criteria, interpretation, limitations, and next steps
remain in a collapsed record, so negative and inconclusive reasoning is not
lost. Changing a finalized study or review hash fails closed; use a new study
ID for a materially different contract.

Existing schema-version-1 reviews and final records remain readable and
idempotent. They may be retained as historical records, but a new finalization
must use review schema version 2. Re-running `study summarize` on an existing
finalized version-1 study regenerates the full report without modifying its
historical `review.json` or `final.json`.

## 6. Maintain the repository picture

`experimental_manifest.md` is the concluded-study index: one short
human-authored interpretation and links to the study folder and full report
belong in its visible entry. The collapsed record retains the exact conclusion
and provenance.

[`current_state.md`](current_state.md) is the human-readable synthesis. After
a reviewed study changes the direction of the repository, update its broad
picture, current synthesis, and next decisions. Do not turn it into a run log
or copy every metric into it. Operational activity remains in
[`current_simulations.md`](current_simulations.md).

## Study states

The generated summary uses these study-level states:

| State | Meaning |
|---|---|
| `planned` | The hypothesis is frozen but no native run is present. |
| `running` | At least one declared run is active. |
| `incomplete` | Some terminal evidence exists, but declared coverage is missing or duplicated. |
| `invalid` | A bundle, arm, provenance link, or artifact contract is invalid. |
| `ready_for_review` | Exact declared coverage is present and valid. |
| `reviewed` | `analysis/final.json` exists for otherwise review-ready evidence. |
