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
The note lives on the experiment's own branch from its first draft; commit it there
before handing off ([git checkpoints](#git-checkpoints)).

## Codifier

Implement the specification faithfully; retain branch-specific scientific contracts.
Use strict existing configs, explicit input hashes and appropriate runtime. Separate
readiness, evidence integrity and additional scientific investigation. No automatic
calibration/qualification sweeps. Make cases, target, budget, duration and output paths
visible. Smoke proportionally, take the launch [git checkpoint](#git-checkpoints),
launch once from the pinned checkout and verify initial semantic progress.
Resolve understood operational failures within the existing budget, preserve attempts,
and rerun affected checks. Do not tune scientific settings as a retry.

Update the current handoff in the owning note: branch, launch commit and pinned
checkout, cases, configs/commands, identities,
process/session/scheduler handles, artifact/log paths, owner, expected progress/cadence,
last/next check, deadline/remaining budget, inspect/collect commands, recovery boundaries
and escalation route. Commit it on the branch once filled and at each material change.
Pass only changed handles and compact evidence at transitions.

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
The review ends with the closing [git checkpoint](#git-checkpoints).

## Git checkpoints

Each experiment or pilot lives on its own branch, from the first file it changes
until it closes. The user granted standing authority to commit, merge and push
([version control](../../../docs/experiment_workflow.md#version-control)). The DRN
trunk is `refactor/training-core`, checked out at `/home/filip/analog-research/drn`.
The repository is shared with the hwa worktree, so DRN branches are `drn/<id>`,
where `<id>` is the experiment id or the pilot's file name (`20261103-<slug>`).
Reuse the branch and worktree if they already exist.

```bash
TRUNK=/home/filip/analog-research/drn
WT=/home/filip/analog-research/worktrees/drn-<id>
git -C $TRUNK worktree add -b drn/<id> $WT refactor/training-core
```

Edit and commit in `$WT` only, staging the paths you changed by name. Commit at
each checkpoint, and before ending any turn that changed files: uncommitted work
outlives the chat that made it and is what goes stale.

| Checkpoint | Commit on `drn/<id>` | Then |
| --- | --- | --- |
| Explorer hands off | note: question, comparison, decision rule, budget | — |
| Codifier launches | code, tests, configs, note with exact commands | integrate; launch from a pinned checkout of this commit |
| Launch or handoff changes | note handoff: handles, attempts, incidents | — |
| Reviewer closes | result or pilot note, verdict, regenerated ledger | integrate; close |

Integrate from `$WT`:

```bash
git merge refactor/training-core          # resolve here; regenerate ledger.md instead of hand-merging it
python -m pytest <affected tests>         # when code changed
git -C $TRUNK merge --ff-only drn/<id>    # if the trunk moved meanwhile, start again
git -C $TRUNK push origin refactor/training-core
```

If the trunk checkout refuses because of uncommitted changes there, stop and report
them; never stash, discard or commit them. Never force-push, and never rebase,
amend or reset a commit that a run, bundle or note names.

Launch evidence runs from a pinned checkout of the launch commit, not from `$WT`.
Bundles record the source commit and the campaign runner fingerprints it, so a
later note commit in the launch tree would make `--resume` redo finished stages.
Remote hosts stage that same pushed commit (for example with `git archive <sha>`),
never a working tree. Probes and smokes may run in `$WT`. Raw outputs go under
`$TRUNK/results/` or the declared remote root, never inside a worktree, so removing
one never removes evidence.

```bash
git -C $TRUNK worktree add --detach $TRUNK/results/<results_root>/source-<sha> <sha>
```

Close after collection and review:

```bash
git -C $WT status --short --ignored       # caches only; anything else is evidence to move first
git -C $TRUNK worktree remove $WT
git -C $TRUNK worktree remove $TRUNK/results/<results_root>/source-<sha>
git -C $TRUNK branch -d drn/<id>          # refuses an unmerged branch: then it is not closed
```

An experiment with an unmerged branch or a remaining worktree is not closed. List
leftovers with `git -C $TRUNK branch --list 'drn/*' --no-merged refactor/training-core`
and `git -C $TRUNK worktree list`.

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
