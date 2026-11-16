# Fixed-pool recovery learning curves — plan only

User decision,16November: retain the selected noisy-physical learning rates;
use only the larger2.1M-token recovery pool and observe KL versus training
length. **Do not launch.** This supersedes the preliminary2×/4×/8× pool grid.
Owning issue:filip137/on-chip-learning#33.

Shared protocol:
`campaigns/fault-recovery-paper/series/001-lr-tuning/experiments/exp-005-recovery-token-budget-20261116.md`.
The adjacent `20261116-fault-token-budget/cases.json` contains the two cases
for this backend; `cohorts.json` pins the larger pool and historical reference.
These are planning records, not executable native training configs.

Each run uses8,192×256 input tokens (2,097,152), WikiText-2 train, seed20261104,
from the existing shuffled block order. The original1,024 sequences remain
a prefix. Keep8 teacher samples per prediction position and preserve their
original prefix tensor byte for byte when preparing new targets.

Existing3% faulty deployment, assignment271001, same source weights and
physical law, accumulated nominal2σ/2ρ gate, no write cap. Noisy physical
learning only. Initial LR is1e-3 for XB6/XB12,3e-4 for DRN6/DRN12.
Adam and cosine-to-zero retain their form, with the horizon extended to2,400
updates, batch8. Measure full-teacher development KL every100 updates on
the same64 validation sequences; report0/300/600/1200/2400 explicitly.
This processes up to4,915,200 input tokens, with replacement from the fixed
pool. Report unique sampled blocks separately. Development points are
measurements, not promises of resumable model checkpoints.

The curve measures training length on one fixed larger dataset. Intermediate
points use the long cosine schedule and do not reproduce independently
shorter schedules. Historical300-step/1,024-sequence runs remain contextual
references. A flattening curve near zero LR does not establish an intrinsic
data limit. If KL keeps falling, report no plateau within the tested budget.
No automatic extension, LR/threshold search, or other method is planned.

Provisional shared compute:3–5GPUh expected,8GPUh hard cap including target
preparation,canaries/retries,20GB new output,up to3A100+2V100 on JeanZay.
No execution window is authorized or assigned; the previous deadline expired.
Native target receipts, strict configs, plans, live canaries and monitored
release remain future execution work. No GPU job or data-generation job
was launched for this plan. Prior exp-004 missing controls stay separate.

## Execution authorized — 16 November

User superseded plan-only: launch all four models on Jean Zay V100, with
GPT-6.1 Sol medium monitors and verified root wake-up. Root thread
01a1e42f-9884-72d3-bb0c-dfa0fe071c5b was successfully resumed by one
`codex queue --resume` callback after observed idle, with new-turn and nonce
acknowledgement evidence in the shared launch/wake and monitor-drn records.

Retain8GPUh total and20GB storage;max2V100 concurrent;coordinator latest
finish17Nov09CET. This is a new execution authorization. V100 DRN throughput
must be measured before final full-job wall caps; do not rely on A100 timing.
Native source math is unchanged. New preparation streams one common8k teacher
target draw, guarded by exact historical1024 prefix, then binds strict native
configs to its real hashes. Canary configs retain full8192 data with32 updates;
full runs have2400 updates and development every100. DRN CPUthreads10 and
V100-compatible wrapper replace the A100-only environment. No data/target hash
placeholder may be used in a live run. Root owns retries/cancellations/decisions.
