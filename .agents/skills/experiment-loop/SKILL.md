---
name: experiment-loop
description: Take crossbar research questions in this worktree (OPT-125M on-chip recovery with corrupt devices; the CIFAR reference) from question to interpretation. Use when designing, specifying, setting up, launching, monitoring or reviewing any experiment or lifecycle.
---

# Experiment loop

Read root AGENTS.md and the assigned pilot/experiment note first, enter at the
appropriate role and reuse completed work. Precedence: the user, then AGENTS.md
(science), the [execution policy](../../../docs/experiment_workflow.md) (authorization,
resources, checks, storage), this skill (procedure) and the note (rules of one
experiment). [Artifact conventions](references/artifacts.md) own note structure.
Planning alone launches nothing; assigned execution continues through interpretation.

Each experiment has three records: the **note** says why (question, frozen
comparison, decision rule, budget, handoff, verdict); the **lifecycle JSON** says
what to compute ([contract](../../../docs/crossbar_lifecycle.md)); the native
**bundles** record what happened.

## Roles

| Role | Starts when | Open | Produces |
| --- | --- | --- | --- |
| Explorer | a question has no frozen comparison | [references/explorer.md](references/explorer.md) | the note's question, frozen comparison, decision rule and budget caps |
| Codifier | a frozen comparison must be implemented or run | [references/codifier.md](references/codifier.md) and the lifecycle contract | lifecycle file, cost probe, plan, launch and handoff |
| Monitor | a run is launched | the note's handoff | observations, collected evidence |
| Reviewer | evidence is collected | the note's frozen rule and evidence | verdict and next decision |

## Who decides each lifecycle field

| Explorer decides (scientific) | Codifier decides (engineering) |
| --- | --- |
| question, decision and primary metric | seed values, keeping their roles disjoint |
| `network`: model and number of analog decoder layers | cohort details within the declared sizes |
| corpus, sequence length, cohort sizes, evaluation split | `deployment.tile_size` (except under CMO hierarchical defects, where it is the physical array size) |
| technology and P&V method | OM characterization bins and samples |
| defect kinds and fractions (`defects.cases`); for CMO the die/array concentrations, `arrays_per_die` and array size | `max_examples` for labelled smokes and probes |
| HWA sources: noise strength (and CMO `hwa.noise_std`), corruption awareness, selection cases | execution device, threads and output paths |
| on-chip arms, update law, learning rates, epochs, write budget | the stage plan and runner commands |
| number of assignment and selection arrays; compute, storage and time caps | |

A field the note leaves open goes back to the Explorer; the Codifier never fills
it. Learning rates come from a separate development lifecycle or a cited earlier
result, frozen before the comparison runs.

## Git checkpoints

Each experiment or pilot lives on its own branch, from the first file it changes
until it closes. The user granted standing authority to commit, merge and push
([version control](../../../docs/experiment_workflow.md#version-control)). The trunk
is `main`, checked out at `/home/filip/analog-research/hwa`. `<id>` is the
experiment id (`exp-023`) or the pilot's file name (`20261103-<slug>`). Reuse the
branch and worktree if they already exist.

```bash
TRUNK=/home/filip/analog-research/hwa
WT=/home/filip/analog-research/worktrees/hwa-<id>
git -C $TRUNK worktree add -b exp/<id> $WT main
```

Edit and commit in `$WT` only, staging the paths you changed by name. Commit at
each checkpoint, and before ending any turn that changed files: uncommitted work
outlives the chat that made it and is what goes stale.

| Checkpoint | Commit on `exp/<id>` | Then |
| --- | --- | --- |
| Explorer hands off | note: question, frozen comparison, budget | — |
| Codifier launches | code, tests, lifecycles, note with describe summary and probe | integrate; launch from a pinned checkout of this commit |
| Launch or handoff changes | note handoff: handles, attempts, incidents | — |
| Reviewer closes | result or pilot note, verdict, regenerated ledger | integrate; close |

Integrate from `$WT`:

```bash
git merge main                          # resolve here; regenerate ledger.md instead of hand-merging it
python -m pytest <affected tests>       # when code changed
git -C $TRUNK merge --ff-only exp/<id>  # if the trunk moved meanwhile, start again
git -C $TRUNK push origin main
```

If the trunk checkout refuses because of uncommitted changes there, stop and report
them; never stash, discard or commit them. Never force-push, and never rebase,
amend or reset a commit that a run, bundle or note names.

Launch campaigns from a pinned checkout, not from `$WT`. The runner fingerprints
the source commit, so a later note commit in the launch tree would make `--resume`
redo finished stages. Probes and smokes may run in `$WT`. Raw outputs go under
`$TRUNK/results/`, never inside a worktree, so removing one never removes evidence.

```bash
git -C $TRUNK worktree add --detach $TRUNK/results/lifecycles/<lifecycle_id>/source-<sha> <sha>
```

Close after collection and review:

```bash
git -C $WT status --short --ignored     # caches only; anything else is evidence to move first
git -C $TRUNK worktree remove $WT
git -C $TRUNK worktree remove $TRUNK/results/lifecycles/<lifecycle_id>/source-<sha>
git -C $TRUNK branch -d exp/<id>        # refuses an unmerged branch: then it is not closed
```

An experiment with an unmerged branch or a remaining worktree is not closed. List
leftovers with `git -C $TRUNK branch --list 'exp/*' --no-merged main`
and `git -C $TRUNK worktree list`.

## Monitor

One owner per case. Short runs stay with the executing agent; delegate only when useful
and permitted. A delegated owner acknowledges handles, deadline and next check and begins
observing before ownership transfers. Observe artifact progress plus process state and
bounded logs; avoid full dumps and duplicate polling. Escalate incidents to the Codifier
and keep unaffected work observed. Do not improvise repairs, retries, cancellations or
scientific changes as Monitor.

Collect and inspect explicit bundles with `python -m ebl runs inspect` and
`python -m workflow collect`; reconcile cases, failures/exclusions/replacements and
measurements. Record factual validity and coverage separately. A truthful partial
bundle can be reviewed. If continuous supervision is unavailable, report the gap and
next action.

## Reviewer

Interpret collected evidence against the original decision rule. Separate observation
from inference, label uncertainty/confounds and retrospective claims, and give a scoped
verdict plus next decision in the same pilot/result note. Negative science is not an
operational failure. Retries are not replicates. Do not rerun to obtain a preferred answer.
Refresh a full campaign's generated ledger when notes change; pilots need none. The
review ends with the closing [git checkpoint](#git-checkpoints).

## Context and delegation

Each role reads only what its row above names, plus the unresolved facts it needs.
The coordinator alone delegates bounded assignments; workers return without spawning
workers. Logical roles need not be separate agents. Match effort to complexity and keep
routine monitoring inexpensive. Do not create separate reports, registries or repeated
validations to document a role transition. Preserve scientific evidence and material
decisions, not tool-call diaries.

Adapted from the lean-experiment-workflow experiment-loop on 2026-09-25. Source host,
GPU-sharing, overnight, Conv/BPTT and calibration assumptions are not imported permissions
or scientific requirements. This repository policy overrides stale external-skill advice.
