# Experiment management for a small research repository

A small repository needs a reliable connection between a question, the computation
that was actually performed, and the conclusion drawn from it. One researcher or
agent can manage that with an experiment note, a runnable configuration or command,
and a separate output directory for each execution attempt.

DRN's workflow addresses the problems below. Its campaign hierarchy, scheduler
integration, strict schemas and worktree conventions support a larger project.
For a smaller repository, introduce each mechanism when it solves an observed
coordination or reproducibility problem. The responsibilities can exist before
there is dedicated software for them.

**Start with the decision the experiment should inform.**

A request such as “try this method” leaves too much open: what counts as an
improvement, which comparison is fair, and when to stop spending compute. Before
substantial execution, record the question, the expected alternatives, the
comparison, the measurements and the decision rule. An exploratory run can instead
state what it is trying to characterize and what would justify a follow-up.

For a comparison, identify what changes and what stays fixed: inputs, initialization,
data split, evaluation procedure, resource allowance and any relevant random
assignments. Choose repetitions and uncertainty reporting to suit the question.
Equal seeds alone do not guarantee a matched comparison if the implementations
consume randomness differently.

For example, comparing two training methods requires a declared starting state,
evaluation set and budget. A result from a different starting checkpoint or a
larger search budget answers a different question. If the comparison changes
during work, record the amendment and distinguish the affected results.

**Distinguish the experiment, its cases and its execution attempts.**

The experiment owns the question and interpretation. A case is one declared
condition, such as a control or an intervention at a particular seed. An attempt
is one invocation that tries to execute that case.

This distinction prevents a crashed run and its retry from becoming two apparent
replicates. Give attempts separate output locations and link retries to their
predecessors. Keep failed and partial attempts visible. A change to scientific
settings creates a revised case; it should not be hidden as an operational retry.

**Give intent, executable settings and evidence clear homes.**

For a small project, three kinds of record are enough:

| Record | What it owns |
| --- | --- |
| One evolving experiment note | Question, comparison, budget, current handoff, evidence links, interpretation and next decision |
| Executable settings | The actual parameters and explicit input references consumed by the program |
| An attempt directory | The settings actually used, source and input identities, logs, metrics, checkpoints and outcome |

The note explains why settings were chosen and links to their exact version. The
run records the effective settings after defaults and overrides. Avoid copying
the same configuration into several manually maintained summaries.

Keep the current handoff concise and update it in place. Preserve substantive
changes of plan and prior attempts by reference, so updating the current view does
not erase how the evidence was produced. Add an index only when finding notes
becomes difficult; an index can link to judgments without becoming another place
where they must be maintained.

**Define the boundary of autonomous execution.**

Planning alone does not authorize experiment execution. An execution assignment should
identify the cases, allowed resources, expected duration, compute and storage
limits, retry allowance and stopping conditions.

Within that boundary, the executor should handle implementation, proportional
checks, monitoring, collection and interpretation without repeatedly asking about
routine engineering choices. A new scientific intervention, increased budget,
different resource target or effect on someone else's work requires a new decision.
The destination repository must establish its own permissions; DRN's permissions
to use hosts or push branches do not transfer with this handoff.

**Check readiness in proportion to the cost of failure.**

Before an expensive run, exercise a small version of the real execution path.
Check that it reads the intended inputs, computes meaningful outputs, writes usable
artifacts and reports failures. For a tiny experiment, the bounded run itself can
provide this check.

Keep readiness separate from scientific investigation. A pipeline smoke establishes
that the computation can execute. It does not establish that the method works, that
the comparison is fair, or that the result generalizes. Additional diagnostics
should answer a concrete question or protect a known invariant. Reuse checks when
their relevant inputs and implementation have not changed.

**Preserve enough identity and state to support the claims you make.**

Record the source revision, effective settings, explicit input identities, random
seeds, relevant environment and command for every attempt. Save the small settings
with the outputs, so later edits cannot silently change their meaning. Use content
hashes or immutable dataset versions where names alone are ambiguous. Avoid
selecting an input by “latest.”

A revision identifies committed code; local modifications need preservation too.
A hash identifies content but cannot reconstruct a missing file. Keep the actual
source and required inputs accessible for the intended reproduction period. For
long-running evidence, execute a fixed source snapshot. A clean commit can be enough
for a small serial project; concurrent edits may justify a separate checkout.

Be precise about continuation. Reloading weights starts a new run from those
weights. Exact continuation may also require optimizer state, random generators,
data position, accumulated updates and simulator state. Claim only the continuation
that has been implemented and checked. Reusing completed work also requires checking
its source, settings, inputs and necessary artifacts; an existing output folder is
insufficient evidence of a valid match.

**Keep one accountable monitoring owner.**

Every active attempt needs someone responsible for noticing failure and taking the
next allowed action. That can be the person who launched it. A detached process,
scheduler acceptance or scheduled reminder does not by itself establish monitoring.

The current handoff should identify the attempt, process or job handle, output
location, expected signs of progress, last observation, next check and remaining
budget. Check both process state and meaningful artifact progress. An alive process
can be stalled; a quiet log can belong to a healthy long operation. Use a cadence
appropriate to the workload.

If ownership changes, the incoming owner should acknowledge the handoff and begin
observing before the outgoing owner stops. If continuous supervision is unavailable,
record that limitation and the next actual inspection. Keep recovery within the
declared protocol and retain the failed attempt's evidence.

**Represent completion as several separate facts.**

One “done” flag cannot answer all of these questions:

| Question | Evidence needed |
| --- | --- |
| Did the process finish? | Exit or terminal status, with any error recorded |
| Are the outputs valid? | Required files, readable contents, identities and relevant numerical checks |
| Is the declared comparison covered? | A mapping from planned cases to valid attempts, including omissions and exclusions |
| What do the results support? | Review against the question and decision rule, with uncertainty and limits |

A successful process can produce unusable evidence. Valid individual runs can
leave the comparison incomplete. Complete coverage can support a negative or
inconclusive conclusion. Partial evidence can still be reviewed, provided the
missing coverage limits the claim. Preserve these distinctions in the note even
if the runner initially exposes only an exit code.

**Finish with interpretation and a next decision.**

Review the result against the declared comparison and decision rule. Separate the
observed measurements from their interpretation. Record uncertainty, failures,
exclusions and plausible confounds. Mark explanations developed after seeing the
result as retrospective. A disappointing result is a valid outcome; rerunning
until it looks favorable changes the evidence.

The review should say what follows: adopt a change, reject it, stop, or propose a
specific further test. A proposed test does not automatically become an authorized
run. An execution assignment reaches its useful endpoint when someone can
understand what was learned and why that next decision follows.

Keep source, small settings and conclusions versioned. Store large outputs outside
Git, with stable references and an explicit retention policy. Retain the evidence
needed for the intended claims and supported continuation. Record intentional
pruning rather than leaving references that imply deleted artifacts still exist.

**Begin with a minimal implementation and expand around demonstrated needs.**

A practical starting point is one note per question, one existing execution entry
point and a unique directory for each attempt. The note can contain these six
prompts:

1. What are we trying to learn, and which decision will it inform?
2. What are the cases, fixed factors, measurements and decision rule?
3. What execution is authorized, with which budget and stopping conditions?
4. What is running, who owns it, and how is progress inspected?
5. Where is the evidence, what was checked, and what coverage is missing?
6. What follows from the results, and what is the next decision?

One person or agent can perform all of these responsibilities. Separate agents,
formal stage transitions and multiple documents are useful only when coordination
requires them. Introduce validated configuration as settings become error-prone;
add an artifact inspector when manual checking becomes repetitive; add scheduling
when independent cases compete for resources. Keep scheduling connected to the
same execution path used for individual runs. Group notes into campaigns when
several experiments support one sustained question.

The acceptance test is a cold handoff: can another researcher, without the original
chat, recover the question, find the actual settings and evidence, determine what
is still active, distinguish a failed attempt from a completed comparison, and
understand the next decision? Use any failure of that test to choose the next
workflow improvement.

These principles are distilled from DRN's [execution policy](../workflow/policy.md),
[experiment-loop responsibilities](../.agents/skills/experiment-loop/SKILL.md)
and [pilot record template](../.agents/skills/experiment-loop/references/pilot.md).
The smaller-repository implementation described here is an adaptation; its file
layout and automation are choices for the receiving project.
