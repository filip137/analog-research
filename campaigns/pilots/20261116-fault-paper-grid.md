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

## Conditional boundary candidates — CPU preparation

Root requested one immutable source containing the next candidate rates. Prepared
native configurations under `examples/opt_cmo_drn/fault_paper_grid/boundaries/`:

- `drn6-calibration-upper.json`: independent calibration arms at LR .1 and .3.
- `drn12-calibration-upper.json`: independent calibration arms at LR .1 and .3.
- `drn12-lora64-lower.json`: one digital rank-64 LoRA arm at LR 1e-5.

These are **conditional candidates, not released runs or selected rates**. Root
releases a model's calibration pack only if its .03 prerequisite wins under the
predeclared rule. Release the DRN12 lower LoRA point only if 3e-5 wins. Both arms
in a calibration pack restart from the original saved tuning deployment, rather
than continuing one another. Root owns integration, push, remote staging, native
input/source preparation, gates and GPU release; this preparation launches none.

All fields other than the declared arm list are identical to each depth's current
prerequisite template: 3% faults, tuning map 271001, original 1,024-sequence cohort,
eight stored samples, 300 steps, batch 8, cosine to zero, no held-out cohorts,
fixed source/teacher/solver and DRN LoRA initialization seed 20261113. No budget,
source or numerical default was expanded. Native runtime wall guards are inherited;
root's independently bounded launcher sets the actual authorized attempt limit.

DRN6's completed .03 prerequisite was collected to
`/tmp/fault-paper-grid-collected/drn6-prerequisites/run`; native artifact inspection
and exact source/config/input/cohort/sample/deployment/300-step checks passed.
Final dev KL is 0.0895921541779649 with zero physical writes, versus the old .01
endpoint 0.09131397073175393 (1.88560035% improvement, outside the 1% tie band).
The winning upper boundary remains unresolved; no rate is frozen prematurely.
The new run uses the authorized 5090/PyTorch 2.11 environment, while old evidence
used A100/PyTorch 2.5; input identities match, and baseline dev differs only at
small floating-point scale. Preserve this environment distinction in review.

All three candidates pass strict native parsing and native CLI request routing
with a CPU capture handler, which never enters training or opens GPU contexts.
Live `run_local.py --prepare` source/input hashing remains root's staging step;
these CPU checks do not claim that future remote artifacts are already prepared.

Candidate validation: **26 focused CPU tests passed in 2.43 seconds** across
paper launcher acceptance, held-out evaluation and matched controls. Exact
template comparisons confirmed that only the candidate arm lists changed.
Configuration byte SHA-256 values:

- DRN6 calibration upper: `7a83e38be85a123772fd3007c92a76e4dc07220d00de8c84473bf4a46c264902`.
- DRN12 calibration upper: `b35198824da4a2539e8a2f754d3f7bb0abf7730801758f388dc3771b4d37a43d`.
- DRN12 LoRA64 lower: `0450614bf084687cc04c17153592446d64b7d38669629f49095b381ea1107289`.
