# Question-led experimental workflow

This policy supersedes the former study/manifest lifecycle. Historical plans remain
provenance; scientific contracts in AGENTS.md and explicitly assigned protocols still
govern the computation. See the [experiment-loop skill](../.agents/skills/experiment-loop/SKILL.md).

## Authority and scope

Planning, setup or review alone does not authorize experiments. An assigned execution
task authorizes ordinary implementation, local setup, proportional readiness checks,
launch, monitoring, collection, interpretation and understood operational retries within
its declared scope and budget. Make cases, target, budget, expected duration and output
location visible before substantial execution. Choose lean routine engineering defaults;
escalate material missing scientific choices, unavailable access, effects on unrelated
work or anything outside the assignment. Retries consume the same allowance.

Use local CPU for tiny tests when scientifically appropriate. Existing explicitly
authorized targets remain usable within their assigned scope; a target appearing in an
old manifest is not standing authorization. This migration grants no new remote-host,
GPU-sharing or overnight permission. Verify live resources and preserve unrelated jobs,
priorities, MPS and GPU settings. Record host/environment and measured throughput when
estimating budgets. Remote use requires an explicit target, compatible environment,
source/config/input staging and collection paths; never import code from another
worktree into a controller. Invoke each worktree's CLI as a subprocess.

## Records

Start with what we want to learn and which decision the answer informs. A campaign
organizes sustained work; ideas, explorations, hypotheses and experiments are related
records, not mandatory sequential stages. Explorations need not have hypotheses.

A small pilot uses one evolving `campaigns/pilots/<name>.md` with question, expected
outcome, comparison, decision rule, budget, handoff, evidence and interpretation. No
separate JSON study, result report or ledger is required. Configs remain strict JSON.
For sustained campaigns use [the artifact conventions](../.agents/skills/experiment-loop/references/artifacts.md).
The campaign README is a short overview and links. Experiment notes own specifications,
the current handoff and run locations; result notes own measurements and interpretation.
The generated ledger displays recorded judgments; it neither launches nor concludes.

`campaigns/runner.py`, `schema.py` and `manifests/` remain the optional subprocess
orchestrator. Their executable manifests are distinct from research campaign notes.
Existing `studies/*.json` plans still prepare and enforce their exact config contracts.
New pilots can run directly through `python -m ebl` into named results roots without
preparing a JSON study. Never put a new pilot inside an unrelated legacy study root.

`docs/current_simulations.md` and `docs/experimental_manifest.md` are frozen historical
snapshots: no refresh or append. This overrides older documents and external skills,
including mandatory global-manifest/human-closeout instructions in ebl-study-closeout.
Do not replace them with another manually maintained global list. Protect personal
notes: do not edit `docs/current_state.md` without explicit permission; do not read or
edit `docs/my_notes.md` unless asked. Historical workflow: [archive](archive/experiment_workflow-pre-question-led.md).

## Readiness, integrity and scientific investigation

Use `python -m ebl describe --experiment ID` to discover the actual runtime contract.
Check that the declared computation executes, reads correct inputs/configuration,
produces meaningful artifacts and detects invalid numerical execution. Preserve exact
checkpoint identities, input data/cohorts, held apparent-state forwards and declared
resume capabilities. Full-state restore checks apply when resuming stateful training.

Scientific diagnostics follow the question or its governing contract. Do not add
historical qualification sweeps, calibration or repeated reference comparisons by
default. Skipped checks are untested, not passed. A tiny bounded real run can also be
its readiness smoke; use a separate smoke before substantial or risky computation.
Reuse validation of unchanged evidence unless inputs change, a failure occurs or a
specific concern warrants repetition. Metadata inspection is not full artifact hashing
or numerical/physical validation; state which checks actually ran.

## Launch and status

Use the existing `python -m ebl train`, `validate` or `characterize` command appropriate
to the experiment. Keep commands/configs in the assigned note and actual execution
identity in the native bundle. Every independently executable case gets its own run.
Preserve failed and superseded attempts; retries are not independent replicates.

One owner monitors each running case. The current handoff names cases, exact handles,
output/log/status paths, expected artifact progress, cadence, last observation, next
check, deadline/remaining budget, recovery boundaries and escalation destination.
The executing agent can own short runs directly. For delegation the new owner must
acknowledge handles/deadline and begin observing before the old owner relinquishes it.
Detached processes and future timestamps alone do not establish supervision.

Check compact status, artifact growth and bounded logs against process/scheduler state.
A live process does not prove useful progress; a stale observation does not prove
failure. Use cadence appropriate to expected progress and deadline, not duplicate
polling. Escalate unexpected exits, invalid artifacts, nonfinite values, stalls or
budget risks to the Codifier, while observing unaffected work. The Codifier handles
understood operational retries without changing the scientific contract. Material
scientific changes return to Explorer/user. If supervision must end without a real
supported continuation, report the unmonitored gap and next required action honestly.

## Collection and review

Inspect explicit bundle paths, without preparing a study or updating documents:

```bash
python -m ebl runs inspect RUN_DIR [ANOTHER_RUN_DIR] --json
python -m ebl runs inspect RUN_DIR --verify-artifacts --require-complete
```

The first command reports integrity separately from process state. A structurally
valid failed/running attempt is not completed evidence. `--require-complete` also
fails for unfinished/failed processes. The inspector checks bundle identity, resolved
config digest, terminal records and artifact locations/sizes; `--verify-artifacts`
also checks declared artifact hashes. Independently reconcile assigned case coverage,
correct scientific inputs, selected metrics and domain-specific validity checks.

Review against the predeclared decision rule in the same pilot/result note. State
observations, inference, uncertainty/confounds, exclusions and next decision. Partial
or negative evidence is reviewable; do not label it complete coverage. Label
retrospective hypotheses. An assigned experiment finishes through interpretation,
not merely launch or collection. Further runs need scope/budget authorization.

Legacy `ebl study prepare/summarize` remain available. `study finalize` is an optional
compatibility command for complete legacy studies: it writes only their local
`analysis/final.json` receipt; `--manifest` is a deprecated ignored argument. It does
not write the historical manifest or gate new pilot/result-note reviews. Existing
v1/v2 records remain readable in the branches that supported them.

## Evidence and storage

Keep small configs, source, notes and summaries versioned; keep raw results, datasets,
checkpoints and logs outside Git. Each case retains resolved config, source identity
(including relevant dirty changes), input identities, command/runtime, progress,
metrics, errors and output artifacts. A hash identifies content but does not preserve
it: retain relevant source changes or a snapshot when the checkout may change.

Declare a storage budget and checkpoint retention before launch. Keep final and
scientifically required checkpoints and necessary resume state; per-epoch checkpoints
are not automatically required. Preserve failed/partial attempt evidence. Pruning
requires the declared policy or explicit authorization and an audit of removals;
never rewrite original manifests to pretend removed evidence still exists. These
workflow changes do not silently change scientific runner checkpoint behavior.

## Overhead and acceptance

Read the assigned record first; load contracts only as needed. Pass paths and compact
evidence, update the current handoff in place, and avoid raw dumps, redundant checks
and routine tool-call narration. Create plots/reports only when useful. Centralize
bounded delegation; use available inexpensive monitoring appropriate to the task,
without hard-coded model names or a mandatory four-agent arrangement.

Validate changes with a tiny real question-led pilot through execution and review.
Record execution versus preparation/reading/validation/handoff time and token usage
where available (otherwise unknown). Another agent must recover the question, what
ran, evidence locations, outcome and next decision from records without chat history.
