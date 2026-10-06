---
name: experiment-loop
description: Take bounded DRN and crossbar research questions through specification, execution, monitoring, evidence and interpretation.
---

# Experiment loop

Read root AGENTS.md and the assigned pilot/experiment record first. Enter at the
appropriate role and reuse completed work. The [execution policy](../../../docs/experiment_workflow.md)
owns authorization, resources, checks and storage; [artifact conventions](references/artifacts.md)
own note structure. Use the existing ebl CLI and bundles, not a new launch schema.
Planning alone launches nothing; assigned execution continues through interpretation.

## Explorer

Clarify the question and decision with the user or existing assignment before choosing
runs. State expected outcome, alternatives, comparison, held-fixed factors, measurements,
decision rule, budget and stopping conditions. Use one evolving pilot note for small
tests, without mandatory hypotheses or campaign scaffolding. An accepted specification
can still need implementation. Escalate scientific choices, not routine engineering.

## Codifier

Implement the specification faithfully; retain branch-specific scientific contracts.
Use strict existing configs, explicit input hashes and appropriate runtime; crossbar
loops are campaign lifecycles run by planned stages ([contract](../../../docs/crossbar_lifecycle.md)). Separate
readiness, evidence integrity and additional scientific investigation. No automatic
calibration/qualification sweeps. Make cases, target, budget, duration and output paths
visible. Smoke proportionally, launch once and verify initial semantic progress.
Resolve understood operational failures within the existing budget, preserve attempts,
and rerun affected checks. Do not tune scientific settings as a retry.

Update the current handoff in the owning note: cases, configs/commands, identities,
process/session/scheduler handles, artifact/log paths, owner, expected progress/cadence,
last/next check, deadline/remaining budget, inspect/collect commands, recovery boundaries
and escalation route. Pass only changed handles and compact evidence at transitions.

## Monitor

One owner per case. Short runs stay with the executing agent; delegate only when useful
and permitted. A delegated owner acknowledges handles, deadline and next check and begins
observing before ownership transfers. A detached launcher/future timestamp is insufficient.
Observe artifact progress plus process state and bounded logs; avoid full dumps and
duplicate polling. Escalate incidents to the Codifier and keep unaffected work observed.
Do not improvise repairs, retries, cancellations or scientific changes as Monitor.

Collect and inspect explicit bundles with `python -m ebl runs inspect`; reconcile cases,
failures/exclusions/replacements and measurements. Record factual validity and coverage
separately. Reuse unchanged validation. Do not confuse process success, artifact integrity,
complete coverage and scientific review. A truthful partial bundle can be reviewed.
If continuous supervision is unavailable, explicitly report the gap and next action.

## Reviewer

Interpret collected evidence against the original decision rule. Separate observation
from inference, label uncertainty/confounds and retrospective claims, and give a scoped
verdict plus next decision in the same pilot/result note. Negative science is not an
operational failure. Retries are not replicates. Do not rerun to obtain a preferred answer.
Refresh a full campaign's generated ledger when notes change; pilots need none.
Never append historical global manifests or update human notes as an automatic closeout.

## Context and delegation

Explorer reads question/context; Codifier reads the assigned specification and applicable
execution/scientific contracts; Monitor reads the handoff; Reviewer reads the frozen rule
and compact evidence. Load more only for unresolved facts. The coordinator alone delegates
bounded assignments; workers return without spawning workers. Logical roles need not be
separate agents. Match effort to complexity and available tools, keeping routine monitoring
inexpensive. Do not create separate reports, registries or repeated validations to document
a role transition. Preserve scientific evidence and material decisions, not tool-call diaries.

Adapted from the lean-experiment-workflow experiment-loop on 2026-09-25. Source host,
GPU-sharing, overnight, Conv/BPTT and calibration assumptions are not imported permissions
or scientific requirements. This repository policy overrides stale external-skill advice.
