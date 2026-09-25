# Artifact conventions and copyable templates

## Small pilots

For a small bounded exploratory test, use one Markdown note:
`campaigns/pilots/<name>.md`, optionally linking an existing campaign.
Reuse an already assigned note in place when available. No extra README, idea board,
exploration, hypothesis file, series, result note or ledger is required for a pilot.

The note grows through the roles:

- Explorer: question, expected outcome, matched comparison, measurement/decision rule and budget.
- Codifier/Monitor: config/command links, handles, current handoff, coverage, measurements
  and validation outcome; detailed hashes and logs stay in the run bundles.
- Reviewer: interpretation, limits and next decision in the same note.

Keep the frozen design distinguishable from later observations; record material changes
without adding a chronological account of routine tool calls. Write only the summaries
or plots needed to assess the decision. No pilot frontmatter schema or ledger command is
required. Link all top-level result roots and retain current state and review in this
note. Canonical run bundles and smoke/monitoring rules still apply; no entry in a
central simulation index or manifest is required.

Expand into the full campaign structure only for sustained multi-experiment work or an
explicit request. Preserve/link the pilot rather than copying its narrative into every
new record. The remaining templates and ledger rules apply to full campaigns only.

## Full campaign ownership

The campaign README owns the current research question, scope and direction; experiment/pilot records own decision rules. Named
scientific protocols and explicit user instructions still govern execution. An old plan
remains the record of what its historical runs did; new campaign planning does not
retroactively alter it. All new planning for an adopted question goes into its campaign,
not a second mutable top-level dated plan.

ideas.md maps the whole idea space; it is not a queue or a duplicate outcome ledger.
Explorations can exist without hypotheses. Hypotheses can have multiple experiments,
and an experiment can test multiple hypotheses if the controls identify each claim.

## Minimal metadata

Use YAML-compatible frontmatter with one key per line and JSON values. No multiline
values or YAML extensions. This restricted format needs only Python's standard library.
The ledger consumes only the keys below; scientific configs and run manifests remain
in their existing formats. Dates, sources and detailed decisions belong in the body.

Hypothesis template:

```markdown
---
id: "H-001"
title: "A single falsifiable claim"
---
# H-001 — A single falsifiable claim

## Provenance and motivation
Prospective, or reconstructed after historical observations. Link an exploration.

## Claim and regime
State the intervention, controls, population and expected measurement.

## Support / falsification / inconclusive
State distinguishable outcomes and the scope of any conclusion. Numerical thresholds
must be sourced or labelled proposed and frozen before measuring new results.
```

Experiment template:

```markdown
---
id: "exp-001"
title: "One bounded test"
status: "planned"
hypotheses: ["H-001"]
---
# exp-001 — One bounded test

## Question and dependencies
Link hypotheses and predecessor evidence; name missing inputs.

## Cases and controls
List arms, seeds, held-fixed factors and comparison/evidence regime.

## Execution contract
Explorer: specify the computation and scientific contract; link existing configs when
available. Codifier: implement missing code, then record exact configs/commands and
source identities before running. Do not invent CLI flags or hashes.

## Metrics and decision
Primary outcome, validity guards, mechanism metrics, aggregation, acceptance and
falsification. State what an inconclusive result means.

## Budget, stop and expected outputs
Bound cases, epochs/replays and runtime; define failure and stopping handling.
Name the existing results root before execution and the expected campaign result note.

## Execution and monitoring handoff
Codifier fills this before delegating: exact cases/coverage, source/config identities,
commands, target/handles, logs, remote/local paths, expected progress/cadence, last
observation, next check, monitoring owner, deadline/remaining budget, permitted mechanical
pause/resume, inspect/sync/collect/validate commands and escalation destination.
Monitor updates observations and factual collection/validation evidence here or in the
linked result note. Keep one current handoff; preserve failed/replaced attempts by link.
```

Experiment states: planned, blocked, ready, running, partial, complete, failed,
cancelled, superseded, imported. Imported means historical prose/evidence has been
linked, not that new compute or local validation happened. A ready plan must have its
scientific choices resolved; ready itself does not authorize a new task. Planned and
blocked files are useful specifications, not executable promises. Missing implementation
alone does not block a scientifically ready assignment to the Codifier.

Result template:

```markdown
---
experiment: "exp-001"
evidence: "validated-local"
summary: "A short observation including its scope"
verdicts: {}
---
# Result: exp-001

## Evidence and coverage
Record bundle paths, resolved config/command, commit/environment, checkpoint and
input hashes, validation command/output, included cases, failures and exclusions.

## Measurements
Codifier/Monitor records measured outcomes and validation evidence without deciding
hypothesis verdicts. Operational success need not support the hypothesis.

## Interpretation
Reviewer fills this section and verdicts, for example {"H-001": "contradicts"}.
Separate observations from inference. Explain each verdict in its tested scope.
Record deviations and residual uncertainty, including any correction to an older note.

## Decision
Continue, revise, stop or leave unresolved; link the next experiment when it exists.
```

Evidence classes: imported-summary and validated-local. Verdicts: supports,
contradicts, inconclusive, not-tested. These judgments are written by the reviewer,
not inferred by the generator. Empty verdicts means review is pending, not that the
experiment was inconclusive or not tested. validated-local requires actual collected
evidence validation, including a truthful reconciliation of failures/partial coverage;
it does not imply successful completion. Until validation is possible, keep current
operational observations in the experiment handoff; do not invent an evidence class
or mark an unresolved collection as complete. A result can refer only to hypotheses
declared by its experiment. Multiple result notes are allowed; differing judgments
remain visible.
Complete, partial, failed and imported experiments require at least one result note.

Historical imports remain imported-summary even when their source says its bundles
were validated at the time. To verify them now, create a separate audit experiment and
result; do not relabel an import as newly executed or backdate its hypothesis.

## Campaign / series checklist

Campaign README: stable research question, scope, a short current direction and links
to ideas, the active experiment and the ledger. Keep it roughly one screen. Detailed
evidence, metrics, decision rules, configs, budgets and authorization belong to linked
experiment/result records; do not accumulate a chronological runbook in the README.

Series README: shared instrument/regime, reason for the series and what cannot be
pooled. A historical series may index different regimes, explicitly separated.

Exploration: observation, reasoning, competing explanations, confounds, discriminating
tests and open questions. Prefer useful reasoning over filling a form.

ideas.md: one row per idea, priority, exploration and H-ID when codified. Include all
hypotheses and preserve parked ideas. Do not copy acceptance/rejection status here.

## Generate and check

Pass full campaign directories only; `campaigns/pilots/` does not have a ledger.

From repository root:

```bash
python .agents/skills/experiment-loop/scripts/ledger.py campaigns/YOUR_CAMPAIGN
python .agents/skills/experiment-loop/scripts/ledger.py --check campaigns/YOUR_CAMPAIGN
python -m unittest discover -s .agents/skills/experiment-loop/tests -p 'test_*.py'
```

The checker validates structural requirements, metadata, unique IDs, references in an optional idea board, experiment/result references and generated-ledger freshness. It does not
validate scientific truth, external URLs, exact configs, raw run bundles or hardware.
Use `python -m ebl runs inspect RUN_DIR --verify-artifacts --require-complete` for actual collected runs
and separately reconcile all expected study cases under the reporting contract.

The generator reads only campaign Markdown and writes only ledger.md, deterministically
and atomically. It does not execute Markdown commands or start experiments. No new
runtime dependency or experiment-launch schema is introduced.

Explorations, hypotheses, ideas.md and series are optional until useful. A campaign
can begin with its README and exploration only. Ledger checks validate records present,
not a mandatory planning hierarchy. Experiment `hypotheses: []` is permitted.
