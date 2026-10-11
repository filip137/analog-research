---
id: "exp-008"
title: "Dense on-chip baseline over the full synthetic corpus"
status: "running"
hypotheses: []
---
# exp-008: dense on-chip recovery after standard HWA

## Question and authorization

Does one full synthetic-corpus pass of dense on-chip learning improve a 3%-faulty
array deployed from standard 3% Gaussian HWA, and at what device-write cost?

On October 11 the user requested the 50M-token, one-pass, uncapped-write baseline
at analog LR **5e-3**, and explicitly selected **standard 3% Gaussian HWA first**
after being offered 3% actual chip corruption. Digital transformer parameters
and calibration are frozen. This is a one-array exploratory pilot. It precedes
sparse-update experiments; the earlier multi-source overnight grid was a proposal
and was never launched. It is superseded by this narrower baseline.

The source was trained with Gaussian weight noise of 3%, without temporary stuck
devices, using the frozen whole-model split-LR recipe in
[exp-007](../../001-hwa-robustness/results/exp-007.md). Reuse that checkpoint,
SHA-256 `86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028`,
from loulou; exact original path and result digest are in
[the frozen source catalog](../../../launch/exp1g_four_arm_sources.json).
No new HWA fit or source selection is performed.

## Frozen comparison

Use OPT-125M's final six decoder MLPs (layers 6–11, fc1/fc2): **28,311,552
logical analog weights**. The digital attention, remaining MLPs, embeddings,
normalization, biases and output gains retain the source values. The teacher is
the pinned FP32 OPT-125M checkpoint from `workflow/networks/opt_mlp.py`.

Use the established CMO 2% endpoint model, two physical devices per weight,
active/reference subtraction, held apparent states, no drift or read noise.
Faults are permanent random stuck active or reference devices, exactly 3% per
MLP, uniform placement (no die/array rate variation), 256-device array sides,
48 arrays per die. Recovery assignment is **271001**, endpoint offset **10000**.
This assignment has been examined in earlier HWA work; it is not a new holdout.

The two matched arms start from exactly the same P0 deployment:

| Arm | Analog action | Digital action |
| --- | --- | --- |
| `none` | Hold the deployment | Frozen |
| `onchip_weights` | Dense noisy updates on this array | Frozen |

The source-compatible descriptor retains all five assignment seeds, and the
execution manifest retains their cheap deployment stages. **Only assignment
271001 receives on-chip training.** The four extra deployment rows are not
recovery replicates. The collector explicitly validates this stage subset.

On-chip law: `cmo_idealized_write_adam`, teacher-KL objective, digital Adam moments,
constant normalized-state LR **0.005**, batch **32**, seed **53**, one epoch.
Every nonzero command attempts one active-device write; reference devices stay
fixed, stuck faults persist, and there are no verify reads or retries. There is
no independent write-count cap. Write noise accumulates in the held state; this
is an idealized model-based experiment, not measured hardware or a validated
incremental-pulse law. The HWA-selected LR is an explicit user-chosen starting
point, **not an established on-chip optimum**. No warm-up or decay is introduced.

Adaptation uses all **195,226** training sequences of length **256** from
`opt125m-synthetic-50m`, corpus SHA-256
`ce58085af0ae1add6415372a50bf20ba4238ee8cae5ee55093c83e6a9a36d984`.
That is **49,977,856 input tokens and 6,101 optimizer steps** (final batch 26).
The new strict `onchip.training_cohort: "hwa"` setting selects the existing full
training cohort. The old 8,192-sequence recovery subset and the source's data
provenance remain intact. Adaptation does not touch development or test tokens.

## Measurements and interpretation

Primary: teacher KL on the existing held-out generated-text transfer cohort
(384 sequences). Also report WikiText test KL and perplexity (1,024 sequences),
and the synthetic development cohort (256 sequences). Report initial and final
held apparent-state metrics; persistent-state metrics are diagnostics only.
No test-dependent checkpoint or stopping-time selection: retain the final state
after the declared full pass, plus the untouched matched deployment. Training
loss is logged every 100 updates; held-out evaluation is at the endpoints.

Report total adaptation writes, touched devices, maximum and mean writes per
healthy active device, stuck attempts, out-of-range devices and zero verify
reads. Separate adaptation from initial programming. The maximum possible
attempts per active cell is 6,101; dense writes can approach 172.73 billion total
active-cell attempts. Healthy-cell writes exclude faults and exact zero commands.

Interpret the sign and size of the paired change without selecting a favorable
domain. A gain is evidence for this source/array/rate/budget only. A negative
result remains valid. This pilot cannot establish superiority over the best
no-weight-learning control: stronger Gaussian HWA, corruption-aware HWA,
rewrite and matched digital calibration, with independent arrays, remain needed.

For subsequent sparse experiments the user's **p95** means the largest 5% by
absolute gradient. That is a per-step write fraction, not a maximum writes per
cell. The mask must gate the final Adam command so momentum cannot write masked
cells. Raw-gradient ranking, Adam-command ranking and accumulated-error ranking
are different interventions. Sparse selection and its LR development are not
implemented or dispatched by this baseline manifest.

## Budget and readiness

Target **loulou RTX5090**, one worker, admitted only while idle. Fifi, riri and
trex are occupied by another user's jobs as of 23:41 Paris; leave them alone.
Cap **4 GPU-hours** across readiness, baseline and any authorized operational
retry; **20 GB** new retained storage; hard deadline **October 12, 07:30 Paris**.
The baseline allowance is 12,600 seconds including validation; each readiness
job has 900 seconds. Initial duration estimate is two hours for the baseline,
subject to the measured canary and a conservative margin. No automatic scope,
rate, data or budget changes.

1. One-MLP native smoke: real teacher, 64 training sequences, two dense updates,
   one 3%-faulty array, hold and learn arms. This uses an unadapted teacher source
   and is only a physical/runtime check.
2. Six-MLP canary: exact frozen HWA source, existing 8,192-sequence recovery
   subset, 256 updates, matched hold arm. It validates full-size memory, writes,
   permanent faults, finite metrics, saved-state replay and collection.
3. Full baseline is released only after both native validation/collection gates
   pass and a measured 6,101-step projection fits the original budget and night.
   Canary performance is not an efficacy or LR-selection gate.

Strict lifecycles:
[one-MLP smoke](../../../lifecycles/opt125m-cmo-l6-exp2-dense-l1-smoke.json),
[six-MLP canary](../../../lifecycles/opt125m-cmo-l6-exp2-dense-l6-canary.json),
[full baseline](../../../lifecycles/opt125m-cmo-l6-exp2-dense-50m.json).
The [controller](../../../launch/exp2_dense_baseline.py) invokes public
`workflow plan`, `ebl campaign run`, native artifact inspection and
`workflow collect`, preserving the original source bundle.

## Execution handoff

Source revision `ba8086809c852add58bd51c67e28a3a738505ec0` is staged in
`/home/filip/analog-research/opt-cmo-exp1/hwa-exp2-dense-20261011` on loulou.
Both readiness jobs passed native validation, exact state replay and collection.
The full pass launched on loulou at **00:07 Paris on October 12** after the
measured handoff gate. Its worker PID is **1000812**, attempt token
`94b82b892e63418582faaff9d600c962`; source revision and scientific inputs match
the prepared manifest. Conservative completion estimate: **00:50 Paris**. Canary timing was
0.2041 seconds/update including epoch-end evaluation; the frozen conservative
full-job projection is **2,581 seconds (43 minutes)**, with 35% training margin
and 900 seconds for other stages. Peak reserved GPU memory was 18.2 GiB.
See `readiness/handoff.json` and `readiness/canary-timing.json`. Local evidence root:
`results/lifecycles/opt125m-cmo-l6-exp2-onchip/`. Prepared host plans, exact source
revision, queue specs and gates live under `prepared/` and `readiness/` there.
Large native device states stay in the frozen loulou checkout; compact validated
records are collected under `collected/{l1-smoke,l6-canary,50m}/`.

The existing persistent **run-watch-queue.service** is the sole execution owner,
with campaign ID `opt-cmo-exp2-dense-baseline-20261011`, a durable SQLite ledger,
locked idempotent dispatcher, tmux workers and five-minute semantic checks.
The new live handoff test passed at 00:02 Paris on October 12:
`readiness/live-handoff.json` records restart from PID 752113 to 865779, the same
canary worker PID 998410 and attempt token, no duplicate submissions, preserved
budgets and incident history, and a fresh semantic observation of the active job.
Synthetic incident 7 invoked a bounded read-only Codex investigator; its report
is durable and its input snapshot hash is unchanged. Systemd heartbeat, restart
and user lingering remain enabled. Recheck observation of the full worker after
admission; retain incident and budget history. Missing workers, nonfinite
metrics, failed replay or validation, termination, stalls and budget/storage
risks block downstream dispatch. Retries need an explicitly classified
operational failure; bad scientific performance is not such a failure.

Full baseline session on loulou:
`opt125m-cmo-l6-exp2-dense-50m-94b82b892e63418582faaff9d600c962`.
The completed canary remains visible as
`opt125m-cmo-l6-exp2-dense-l6-canary-1c22d1fe47e141ca996d19eb56d8055e`.
Worker receipts and pane logs are under
`/home/filip/analog-research/opt-cmo-exp1/worker-receipts/ba8086809c852add58bd51c67e28a3a738505ec0/`.
Attach read-only with `tmux attach -r -t SESSION` on loulou.

The completion hook collects the full baseline and writes `collected/review.json`,
with no automatic sparse run or broader sweep. The result note will record the
actual outcome and interpretation after collection.
