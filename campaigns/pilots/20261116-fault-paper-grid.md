# Corruption paper grid — native preparation

User authorized scheduling the paper sweep on16November2026; owning issue
filip137/on-chip-learning#33. Shared case inventory and protocol live under
campaigns/fault-recovery-paper/launch/20261116-paper-grid in the research home.
Compute envelope selection is pending; CPU implementation/tests are authorized.
Do not submit GPU jobs until root binds the selected budget and live gates.

Question: across0/1/3/5% faults and independent maps271002–271004, how much
recovery is specific to corruption, relative to the same arm's0% twin?
Keep four HWA sources, physical laws and thresholds; standard recovery remains
1024×256 WikiText train,8sampled targets,300Adam updates,batch8,cosine-to-zero.
Freeze method/model LR using only3% tuning map271001. Paper maps never tune LR.
Use p=.1 Child-D; DRN LoRA initialization seed20261113 stays fixed across maps.
Digital output calibration uses only the same8sampled targets. No drift/readnoise.

48shared deployments support408endpoints:192DRN(8arms percell),216XB(9arms).
Each arm starts from a clone of the same held fault-aware deployment, except
naivehold which uses its saved naive programming state. Keep both0% holds.
AnalogLoRA exists only for XB. Standardize XB6 Bscale using standard sampled
targets and recheck its LR on tuningmap before its paper release; XB12's already
standard Bscale stays fixed. Calibration/r64 boundary checks precede affectedarms.

Required implementation: optional strict fault_tuning.evaluation_cohorts list,
empty by default, each item {name,corpus,corpus_sha256,split,start,sequences,
cohort_sha256}; contiguous256-token sequences, no hidden shuffle. Cohorts are
verified before training. Final endpoint scoring runs before learner/adapter/bank
cleanup, for every arm including holds, using full FP32 teacher and existing
streaming evaluator. Preserve correction modules and held apparent state during
scoring. Save scores/identities perarm in held_out, plus evaluation events, and
assert scoring leaves physical states/cost counters unchanged. No selection or
training uses test cohorts. Required paper cohorts:1024WikiText test,384generated
test; root will provide actual hashes. Tiny tests use small explicitly pinned
cohorts. Old configs without field retain legacy behavior and pass regression.

Future native configs/wrappers must pass parser/tests, pinned source staging,
scheduler test-only and new live canaries before substantive release. Root owns
release/retries/cancellation; monitor agents report evidence only. No source
retraining, changes to numerical learning law, or repeated completed experiments.

## DRN implementation handoff

Optional `fault_tuning.evaluation_cohorts` now accepts an exact list of pinned
contiguous cohorts. The parser rejects unknown fields, duplicate names, invalid
counts/starts and hashes. Legacy configurations omit the new field from their
resolved dataclass representation and retain their original config identity;
access defaults to an empty tuple. Explicit empty lists perform no evaluation.

The runtime verifies all requested cohorts before loading targets or constructing
a learner. Each arm scores its final live apparent bank; LoRA and calibration
hooks remain attached until scoring finishes. Per-arm `held_out` entries and
`fault_tuning_held_out` events include cohort identities and full-teacher scores.
Held hashes, write counters and device RNG must remain unchanged. Evaluation does
not select a checkpoint or affect the declared training step budget. No physical
endpoint replay schema or final-bank retention was added: scoring occurs while
that endpoint is live.

CPU validation uses `/home/filip/miniconda3/envs/py312/bin/python -m pytest` with
`tests/test_fault_paper_evaluation.py`, `tests/test_opt_cmo_drn_fault_tuning.py`,
`tests/test_fault_completion_controls.py` and `tests/test_opt_cmo_drn_recovery.py`.
The new integration test runs real tiny CPU DRN holds, physical learners, LoRA
and calibration and verifies scoring precedes digital-hook cleanup. Additional
tests reject bad cohorts before training and reject held-state, counter and RNG
mutation. No paper configs, GPU launches, source merge or push were performed.
Root must review/integrate the commit and bind actual paper cohort hashes,
resources and live canaries before release. Existing exp-005 monitors continue
unchanged.

Validation result: **51 focused tests passed in 18.41 seconds** on CPU (17 new
paper-evaluation tests plus existing fault-tuning, control and recovery tests).
`git diff --check` passed. Generated-text paper evaluation must use the existing
`wikitext2-train-synthtest` corpus SHA
`8e73b6b43aafb92b8b52e11cdd060d573331fae410350cf2427f3eeba640e224`,
not the synthetic HWA training corpus. Root separately binds the actual cohort
hashes and generates configurations.

## Placement authorized16November17:43UTC

User selected riri/fifi RTX5090s in parallel, with the rest on V100s. Root assigns
DRN to riri/fifi and XB to JeanZayV100. Coordinator operational ceiling48GPUh
and72hours ending19Nov17:43UTC, including prerequisites, tests, canaries and
retries; max2DRN5090+2XBV100 allocations. This supersedes the pending compute
question above. New host-native canaries and measured timing precede full release.
