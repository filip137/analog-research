# Raw LoRA/HWA results

Raw results remain in their existing ignored directories. Scientific questions,
specifications, measurements and interpretations live in the
[question-led campaigns](../campaigns/README.md), following the
[execution policy](../docs/experiment_workflow.md).

## Ownership and new output

Choose one explicit, stable results root for each new comparison and record it in
the owning experiment or pilot note before execution. Keep raw paths stable when
promoting a pilot into a campaign. Related training, deployment, validation and
analysis roots may belong to the same comparison; list each root in its result note.

A native invocation creates its own exclusive run directory, for example:

    python -m ebl train --config examples/small_drn/base.json --output-dir results/<comparison-id>/runs/<arm>

Keep config.resolved.json, manifest.json, status.json, result.json, metrics,
artifacts and checkpoints together. Never select inputs by searching for the newest
checkpoint, overwrite an old run, or count a retry as an independent replicate.

Small pilots can use native run bundles directly. Optional legacy prepared studies
retain study.json, runs/ and analysis/; deployment/ is optional. Historical JSON
plans and existing public CLI contracts remain supported. The CLI can inspect
explicit bundles without changing the science; full artifact verification is a
separate check from metadata inspection and expected-case reconciliation.

## Analysis and interpretation

Keep large generated tables, plots, logs and replay outputs in the owning raw root.
Keep the compact question, measurements, source links, interpretation, limitations
and next decision in the tracked campaign result or pilot note. An existing
analysis/report.md remains historical evidence; new conclusions belong in the
campaign record, without rewriting the raw report.

For historical imports, state which sources exist locally, which metadata was
checked, the date and scope of any original verification, and unresolved coverage.
A recorded complete process does not prove complete study coverage; a stored running
marker does not prove a live job. Keep failed and partial attempts.

The generated campaign ledgers provide navigation. docs/current_simulations.md and
docs/experimental_manifest.md are frozen historical snapshots: do not refresh or
append them. No new manually maintained global run-status index replaces them.

## Existing locations and availability

The migration leaves every raw directory in place, including launcher sidecars and
standalone analysis folders. Each is linked by its scientific owner; workflow-pilot
evidence remains with the existing pilot. Shared results/.launchers contains
cohort-B reproduction, policy-evaluation and LoRA launch receipts, not extra runs.
The two root-level RESET-bias launcher logs belong to the initialization campaign.

Earlier evidence under labs/cases remains there and is linked by the endpoint-model
campaign. Historical references to absent local roots remain labelled unavailable;
a source’s recorded completion is not fresh local validation.

Only this README is tracked inside results/. Campaign evidence links therefore
resolve only in workspaces where the corresponding ignored artifacts are retained
or restored. Do not add raw payloads to Git to make those links work online.
