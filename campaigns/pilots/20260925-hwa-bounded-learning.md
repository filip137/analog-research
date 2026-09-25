# Pilot: Does one pulse-Adam recovery minibatch lower held-out teacher KL compared with leaving the same programmed standard crossbar frozen?

State: complete and scientifically reviewed; initial specification recorded 2026-09-25T15:49:21.129440+00:00.
Authorization: user assigned implementation including bounded local workflow pilots.

## Question and decision

Does one pulse-Adam recovery minibatch lower held-out teacher KL compared with leaving the same programmed standard crossbar frozen?
Expected direction: lower final loss after updates, with a negative or inconclusive
outcome equally acceptable. Decide whether the migrated workflow supports a real,
recoverable comparison in this runtime before using it for larger assigned questions.

## Frozen comparison and decision rule

AIHWKit 1.1.0 OM-preset model-based crossbar [784,50,10], float32 CPU,
teacher `data/mnist_relu_teacher_fixed_init_20260816.pt` SHA-256
9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52.
Assignment 87004, four paired endpoint seeds 89402–89405, counterfactual-repaired devices, winsorized
bounds, 128-pulse codebook and P&V cap, held apparent-state forward. No off-chip
HWA or transfer. Treatment pulse Adam: one epoch, one 16-example minibatch,
rates [6e-5,6e-5], pulse cap 64, one minibatch per endpoint; control is frozen. Evaluate the same first
16 validation examples, seed/data seed 42. Primary metric is fixed-final
apparent teacher KL; persistent state remains a secondary diagnostic. Match
initial programmed-state fingerprints and initial apparent-validation metrics.
Four programming realizations of one array cannot establish fresh-array robustness. No official-test
evaluation; the existing loader constructs/loads the test dataset regardless,
so this is not a claim that the test files were never read. The existing runtime requires a pinned historical DRN reference JSON even for this
within-crossbar question. Retain its explicit path/hash from the native config;
any automatically emitted cross-architecture comparison is excluded from this
pilot's inference. The native strict config requires its full four-endpoint cohort,
so the preflight design uses those four paired realizations before any measurement.
No DRN reference simulation or requalification is launched.

Two cases: `frozen` and `updates`. Delta = mean frozen final loss minus mean updates final loss across the four paired
endpoint realizations; retain each endpoint contrast as well. Support improvement if delta > 1e-6; contradict if delta < -1e-6;
otherwise inconclusive. This arbitrary numerical pilot threshold is frozen before
execution, not a statistical significance or practical-effect threshold. A single
seed and tiny cohort do not support general performance conclusions. Invalid matching
or incomplete cases preclude a comparative verdict, but must still be reviewed.

## Budget and execution

Local CPU, one thread. Expected seconds to two minutes per case; hard timeout 180 s
per invocation, 600 s cumulative invocation allowance including at most one understood
operational retry. Stop on numerical/scientific failure; no tuning. Deadline is ten
minutes from launch. No remote access, GPU, dependency installation or unrelated jobs.
Output: `results/workflow-pilot-20260925/`; storage cap 1 GiB. Retain native final/selected/resume artifacts,
metrics, configs, errors and failed attempts; no new per-epoch checkpoint policy.
The tiny bounded computation itself is the smoke; no duplicate scientific invocation.

Configs: `examples/workflow_pilots/hwa_frozen.json` and
`examples/workflow_pilots/hwa_updates.json`. Exact subprocess argument lists,
interpreter/environment, config/source identities and timing receipts will be in
`results/workflow-pilot-20260925/audit/`. Source diff and untracked source files are archived before launch.

## Current execution and monitoring handoff

Owner: implementing coordinator; all cases terminal, collected and reviewed at 2026-09-25T15:57:54.475532+00:00. No running cases or future monitoring promised.
Completed handles: driver PID 2980354; current child PID and exact argv in `results/workflow-pilot-20260925/audit/execution.json`. The coordinator owned the active execution session and native evidence; no other
owner polled these cases. All cases are now terminal.
Evidence roots: `results/workflow-pilot-20260925/{frozen,updates}/`; logs/receipt under `audit/`.
Execution ended 2026-09-25T15:53:35.068849+00:00, before the recorded deadline; no next check needed; stop after both cases or the budget/deadline.
Unexpected incidents return to Codifier (same owner); scientific changes return to
Explorer/user. Never infer coverage from launcher exit alone.
Collection: `python -m ebl runs inspect <exact bundles> --verify-artifacts --require-complete --json`.
Reuse those checks during review; independently verify case matching and primary metric.

## Evidence and coverage

Two of two cases completed with exit 0; both native bundles passed full declared
artifact-hash inspection once. Complete assigned coverage; no failures, retries,
excluded cases or numerical/scientific repairs. No legacy JSON study or global
manifest update was required. The three historical/human documents remained byte-identical.

- **frozen:** [20260925T155324.713749Z-8787916a-9644d5a4](../../results/workflow-pilot-20260925/frozen/20260925T155324.713749Z-8787916a-9644d5a4).
- **updates:** [20260925T155330.686669Z-53c4dd89-86e62645](../../results/workflow-pilot-20260925/updates/20260925T155330.686669Z-53c4dd89-86e62645).

Exact argv/environment/handles and phase times: [execution receipt](../../results/workflow-pilot-20260925/audit/execution.json).
Integrity results: [run inspection](../../results/workflow-pilot-20260925/audit/run-inspection.json).
Matching, measurements and timing: [compact evidence](../../results/workflow-pilot-20260925/audit/evidence.json).
Source identity: receipt commit plus `audit/source.patch` and `audit/untracked-source.tar.gz`,
whose hashes are in the receipt. Collection logic is retained as `audit/collection.py`.
The source snapshot preserves the pre-run specification and configs; these later
review edits do not alter the launched scientific code. Logs are `audit/{frozen,updates}.log`.

| Primary loss | Frozen | Updates | Frozen minus updates |
|---|---:|---:|---:|
| Mean final apparent teacher KL | 0.0454054321162 | 0.0444406229071 | 0.000964809209108 |

All four paired endpoints matched initial apparent and persistent state hashes,
requested deployment, initial validation, source teacher/logical weights and device
population. No test evaluation was performed; the inherited loader still loads test
data. Native `drn_comparison` was null; no cross-architecture conclusion is made.

| Endpoint | Frozen KL | Updated KL | Delta |
|---|---:|---:|---:|
| 89402 | 0.0657681227 | 0.0612845942 | 0.00448352844 |
| 89403 | 0.0296345223 | 0.0306417439 | -0.00100722164 |
| 89404 | 0.0475473739 | 0.0549439788 | -0.00739660487 |
| 89405 | 0.0386717096 | 0.0308921747 | 0.00777953491 |

Recovery applied a mean 22.75 pulses per realization. Apparent
accuracy averaged 95.31% frozen versus 93.75% updated. Persistent-state diagnostic
accuracy averaged 65.625% in both arms; it did not drive the forward or selection.

## Interpretation and next decision

Verdict: **supports**, scoped to the frozen pilot rule. The paired mean KL reduction exceeds the predeclared 1e-6 threshold, supporting
the narrow mean-loss claim. Individual endpoint responses differ, and accuracy
decreased: loss improvement is not an accuracy improvement. These four writes of
one counterfactually repaired OM-preset array are not four independent arrays or
raw measured-device evidence. One minibatch and sixteen validation examples cannot
establish robustness, optimal recovery budgets or a general need for on-chip training.
Held apparent-state refresh includes write noise, so this comparison does not isolate
the gradient contribution from the effect of rewriting cells.


Workflow decision: use these records and native bundle inspection for future assigned
experiments. Preserve this pilot as exploratory workflow evidence; no follow-up run
is queued. Larger scientific claims require a separately bounded relevant question.

## Workflow audit

Final configs frozen 2026-09-25T15:51:47.568638+00:00; reviewed 2026-09-25T15:57:54.475532+00:00.

| Interval | Seconds |
|---|---:|
| Final config → launch preparation | 95.486 |
| Native case invocations, including process startup | 12.010 |
| Process completion → collection start | 143.831 |
| Collection and matching (includes inspection) | 0.187 |
| Artifact inspection subset of collection | 0.185 |
| Collection end → scientific review | 115.388 |
| Final config → review total | 366.907 |

Output at review: 26.557 MiB, below 1 GiB. Zero operational retries,
zero delegated monitoring handoffs, one integrity pass per bundle. Collection reads
were selective but some exploratory output was verbose/truncated; narrow field
extraction is preferable. Exact read counts and token consumption were not instrumented
and remain unknown. Earlier implementation/preflight work and user discussion are outside
this final-config interval. Intervals include work on the other branch and tool latency;
they are not all model inference or avoidable overhead. The inspector is inexpensive;
most elapsed time is preparation/interpretation, so weakening validation is not indicated.

Cases finished between ten-second observations; the active coordinator/driver observed
their exits and collected terminal artifacts. This validates bounded ownership and
completion collection, not live stall detection, remote scheduling or recovery from an
actual failed scientific invocation. Automated tests exercise failed/partial bundles.
No synthetic delays were introduced to manufacture a live-monitoring test.

Recovery acceptance: **passed**. A fresh agent with no chat history recovered both
pilots' questions, rules, exact command locations, cases, evidence, outcomes, limitations,
monitoring state and next decisions from the records. Across both pilots it read
18 distinct files in approximately 45 seconds; all ten Markdown evidence links and
checked config/source/input paths resolved. It performed no experiments, rehashing or
repeated numerical validation. Token usage remains unknown. This review interval is
additional closeout overhead, outside the scientific-review timing table above.

Software validation: 124 focused tests passed, covering CLI, artifacts, legacy studies,
read-only inspection, historical-document preservation, campaign orchestration/schema,
checkpoints and the campaign ledger. Output: `audit/workflow-tests.log`.
After the final generated-README/help correction, all 24 study/CLI checks passed again;
scientific runtime and pilot artifacts were unchanged and were not rerun.
