# Pilot: Does four minibatches of direct equilibrium-propagation training lower held-out loss compared with a matched zero-learning-rate perfect-diode DRN?

State: complete and scientifically reviewed; initial specification recorded 2026-09-25T15:49:21.129440+00:00.
Authorization: user assigned implementation including bounded local workflow pilots.

## Question and decision

Does four minibatches of direct equilibrium-propagation training lower held-out loss compared with a matched zero-learning-rate perfect-diode DRN?
Expected direction: lower final loss after updates, with a negative or inconclusive
outcome equally acceptable. Decide whether the migrated workflow supports a real,
recoverable comparison in this runtime before using it for larger assigned questions.

## Frozen comparison and decision rule

Synthetic moons, perfect-diode DRN [4,8,2], seed 7/data seed 11,
64 generated points, batch size 4, two epochs × two training batches (four updates),
two held-out batches per epoch. Same initialization, order, solver (8 inference and
8 training iterations), direct EP rule and no hardware/noise modifier. Treatment
weight learning rates [0.01,0.02], control [0,0]; bias rates [0]. Primary metric is
the final epoch's held-out cost, not the best checkpoint metric. Matching includes
generated dataset identity and initial parameter tensors reconstructed from the
frozen zero-update control. This is a toy ideal-DRN control, not measured-device,
LoRA, Tiki-Taka or HWA evidence. No MNIST/test dataset is used.

Two cases: `frozen` and `updates`. Delta = frozen final loss minus updates final
loss. Support improvement if delta > 1e-6; contradict if delta < -1e-6;
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

Configs: `examples/workflow_pilots/lora_frozen.json` and
`examples/workflow_pilots/lora_updates.json`. Exact subprocess argument lists,
interpreter/environment, config/source identities and timing receipts will be in
`results/workflow-pilot-20260925/audit/`. Source diff and untracked source files are archived before launch.

## Current execution and monitoring handoff

Owner: implementing coordinator; all cases terminal, collected and reviewed at 2026-09-25T15:57:54.474637+00:00. No running cases or future monitoring promised.
Completed handles: driver PID 2979694; current child PID and exact argv in `results/workflow-pilot-20260925/audit/execution.json`. The coordinator owned the active execution session and native evidence; no other
owner polled these cases. All cases are now terminal.
Evidence roots: `results/workflow-pilot-20260925/{frozen,updates}/`; logs/receipt under `audit/`.
Execution ended 2026-09-25T15:53:21.192837+00:00, before the recorded deadline; no next check needed; stop after both cases or the budget/deadline.
Unexpected incidents return to Codifier (same owner); scientific changes return to
Explorer/user. Never infer coverage from launcher exit alone.
Collection: `python -m ebl runs inspect <exact bundles> --verify-artifacts --require-complete --json`.
Reuse those checks during review; independently verify case matching and primary metric.

## Evidence and coverage

Two of two cases completed with exit 0; both native bundles passed full declared
artifact-hash inspection once. Complete assigned coverage; no failures, retries,
excluded cases or numerical/scientific repairs. No legacy JSON study or global
manifest update was required. The three historical/human documents remained byte-identical.

- **frozen:** [20260925T155311.833998Z-92f8fd54-5c753593](../../results/workflow-pilot-20260925/frozen/20260925T155311.833998Z-92f8fd54-5c753593).
- **updates:** [20260925T155316.976845Z-8e6ee957-de40127b](../../results/workflow-pilot-20260925/updates/20260925T155316.976845Z-8e6ee957-de40127b).

Exact argv/environment/handles and phase times: [execution receipt](../../results/workflow-pilot-20260925/audit/execution.json).
Integrity results: [run inspection](../../results/workflow-pilot-20260925/audit/run-inspection.json).
Matching, measurements and timing: [compact evidence](../../results/workflow-pilot-20260925/audit/evidence.json).
Source identity: receipt commit plus `audit/source.patch` and `audit/untracked-source.tar.gz`,
whose hashes are in the receipt. Collection logic is retained as `audit/collection.py`.
The source snapshot preserves the pre-run specification and configs; these later
review edits do not alter the launched scientific code. Logs are `audit/{frozen,updates}.log`.

| Primary loss | Frozen | Updates | Frozen minus updates |
|---|---:|---:|---:|
| Final held-out mean cost | 1.43520000577 | 0.137297227979 | 1.29790277779 |

Both runs reported two completed epochs and four optimizer steps; each final
evaluation used eight held-out synthetic examples. Accuracy was 100% in both cases.
Reconstructing the deterministic initial weights and data/order from the unchanged
configs gave identical hashes across arms; the frozen saved weights exactly matched
those initial tensors. These are reconstruction checks, not original in-run order
receipts. Native validation labels this synthetic held-out split `held_out_test`;
it is not the MNIST official test split.

## Interpretation and next decision

Verdict: **supports**, scoped to the frozen pilot rule. The result supports lower final cost after four direct EP updates in this
specific ideal perfect-diode toy trajectory. Accuracy did not improve because it
was already 100% on eight examples. It does not test LoRA/Tiki-Taka, programming
noise, HWA, measured devices or the need for on-chip recovery. Initialization/order
matching relies partly on deterministic reconstruction rather than in-run hashes.


Workflow decision: use these records and native bundle inspection for future assigned
experiments. Preserve this pilot as exploratory workflow evidence; no follow-up run
is queued. Larger scientific claims require a separately bounded relevant question.

## Workflow audit

Final configs frozen 2026-09-25T15:49:21.128391+00:00; reviewed 2026-09-25T15:57:54.474637+00:00.

| Interval | Seconds |
|---|---:|
| Final config → launch preparation | 229.055 |
| Native case invocations, including process startup | 11.006 |
| Process completion → collection start | 143.408 |
| Collection and matching (includes inspection) | 3.606 |
| Artifact inspection subset of collection | 0.097 |
| Collection end → scientific review | 126.268 |
| Final config → review total | 513.346 |

Output at review: 0.175 MiB, below 1 GiB. Zero operational retries,
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

Software validation: 128 focused tests passed, covering CLI, artifacts, legacy studies,
read-only inspection, historical-document preservation, campaign orchestration/schema,
checkpoints and the campaign ledger. Output: `audit/workflow-tests.log`.
After the final generated-README/help correction, all 28 study/CLI checks passed again;
scientific runtime and pilot artifacts were unchanged and were not rerun.
