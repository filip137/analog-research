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
