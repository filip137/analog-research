---
id: "exp-009"
title: "Digital LoRA beside dense on-chip recovery"
status: "planned"
hypotheses: []
---
# exp-009: digital LoRA on the same faulty MLPs

## Question and authorization

On October 12 the user explicitly requested one synthetic-corpus epoch of LoRA
at several ranks, alongside exp-008, using the 3090s, and confirmed corrections
only to the same faulty MLPs. This authorizes a digital adapter comparison here.
Does digital low-rank correction recover the deployed model without device
writes, and how does recovery change with rank? This is an exploratory,
model-based single-array comparison, not a general on-chip viability claim.

## Frozen comparison

Reuse exp-008's standard 3% Gaussian HWA checkpoint, actual permanent 3% random
faults on array 271001, and final six OPT-125M decoder MLPs (layers 6–11, fc1/fc2;
28,311,552 logical analog weights). Source SHA-256:
`86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028`.
Original deployment SHA-256:
`f80a9447469697cf9e4ff657cb17545d2824e60ae4f12d4694f01f3e4ebaf873`.

Hold every physical conductance and every base-model digital parameter fixed.
For each analog MLP matrix add a digital branch `(alpha/r) B A x` in logical
weight units, before its frozen output gain and bias. Ranks **4, 16, 64**;
alpha equals rank, dropout zero. Initialize A with uniform +/-1/sqrt(fan_in)
and B with zeros; CPU seed 53. The initial function is exactly unchanged.
Adapters stay digital at inference: they cannot be merged into a faulty array
without additional programming. Storage and extra multiply-adds are reported.
This follows the low-rank frozen-base parameterization of
[Hu et al., 2021](https://arxiv.org/abs/2106.09685), with the explicitly restricted
MLP placement and analog-plus-digital inference implementation above.

| Rank | Trainable digital parameters | Extra MACs per token |
| --- | ---: | ---: |
| 4 | 184,320 | 184,320 |
| 16 | 737,280 | 737,280 |
| 64 | 2,949,120 | 2,949,120 |

No change to the teacher, source training recipe, array topology, defect maps,
endpoint realization, objective, data partitions or existing HWA selections.
Copy the exact upstream cache and device bundle. Materialize the original
compact P0 on its original RTX5090 into explicit tensors, verify its held-state
hash, retain both artifacts and bind the new file to the original hash. This
avoids architecture-dependent random-number reconstruction on the 3090s.
The materialization is a representation conversion, not another deployment.

## Learning-rate development and final fits

Separate development lifecycles test constant Adam learning rates **3e-4,
1e-3, 3e-3** at each rank on the existing 8,192-sequence recovery subset
(256 updates at batch 32). Standard Adam beta=(0.9,0.999), epsilon=1e-8,
no weight decay, warm-up or decay. All nine candidates start from P0 with fresh
adapter initialization and optimizer state. Choose the lowest final teacher KL
on the 256-sequence synthetic development cohort separately for each rank;
exact ties choose the smaller LR. Test metrics never enter this rule. These
short screens estimate a useful LR; they do not establish a full-epoch optimum.
The screen budget is additional tuning compute, explicitly reported separately
from the matched 50M-token final fits and the user-selected dense baseline LR.

Restart each rank from P0 with fresh adapters and train **one complete pass** of
195,226 synthetic sequences of length 256: 49,977,856 input tokens, 6,101 updates,
batch 32, final batch 26. Use the identical exp-008 cohort, epoch shuffle and
teacher-KL objective. No repeated exposure from continuing a development fit.
Retain the fixed final endpoint, never the best test checkpoint.

Primary endpoint: generated held-out teacher KL (384 sequences). Also report
WikiText test KL/perplexity (1,024 sequences), development KL, the same-host
unchanged-deployment control, adapter parameter count and extra MACs/token.
Report physical writes, programs, touched cells and verify reads as zero during
adaptation. Report initial programming separately. Dense exp-008 is compared
with its own control; cross-GPU score rounding is recorded, not interpreted as
an independent-array effect. This does not yet cover digital calibration,
frozen-target rewrite, alternate sources or independent array assignments.

## Execution, gates and budgets

Use local nom-cool-2 and nom-cool-1 RTX3090s, one worker per GPU, persistent tmux
workers dispatched by the existing single run-watch service. Do not disturb
other users. A bounded P0 materialization uses loulou's 5090 once exp-008 releases
it. At most two 3090s are occupied by this sweep; full fits use development
measured timing and memory gates. Initial development reservation: 20,000 MiB
plus 2,000 MiB headroom. No automatic batch-size or scientific changes to fit.

Cap the new comparison at **8 GPU-hours**, **40 GB** retained evidence across
workers/controller, (0.1 GPU-hour/2 GB reserved for portable-state preparation, 7.9 GPU-hours/38 GB
for the LoRA queue), and stop admission if the next job cannot finish before
**07:30 Paris on October 12**. Retain final checkpoints, development endpoints,
upstream artifacts, native manifests, metrics and failed-attempt evidence.
No per-minibatch checkpoints. No automatic scientific retries or extensions.

Before final fits: workflow regression tests and tiny native end-to-end replay
must pass; each rank's development run must validate finite metrics, exact
schedule/cohort, immutable base/physical state, zero writes, exact saved-state
replay, and measured peak CUDA memory within reservation. Freeze LR selection
receipts and calibrated progress thresholds before dispatch. Negative efficacy
does not count as an operational failure. Any failed mandatory gate holds that
rank's full fits and creates an incident for the persistent read-only investigator.

The existing service's five-minute semantic polling, self-watchdog, tested
read-only incident invocation, restart recovery, idempotent queue and durable
budgets remain authoritative. This campaign adds jobs and gates, not a second
watchdog. Host preparation, native plan/dry-run, job IDs and completion receipts
will be recorded here before handoff.
