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
Use p=.1 Child-D; LoRA initialization seed271001 stays fixed across maps.
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

## Held-out scoring implementation — CPU validation

The HWA-specific `methods/fault_tuning.py` parser now accepts the optional strict
`evaluation_cohorts` list above. Empty/absent lists retain legacy serialization
and lifecycle digests. Names are distinct, shuffle seeds are rejected, and
evaluation cannot overlap recovery or the development slice (including renamed
aliases of the same corpus hash). The drift branch explicitly rejects nonempty
evaluation cohorts; this extension implements the declared corruption grid.

`ebl/runtime/fault_tuning.py` records evaluation corpora as native pinned inputs
and verifies corpus/cohort hashes before target drawing, deployment or training.
Every hold and trained arm scores its final live apparent state through the
existing streaming full-teacher evaluator. Digital/analog adapters and sampled
output calibration remain attached until scoring finishes. Each measurement's
`held_out` list and `fault_tuning_held_out` metric events retain cohort identities,
scores, elapsed time and unchanged-state evidence. The summary continues to name
development KL as the selection metric; held-out values do not select anything.

Scoring fingerprints conductances, write counts, device RNG/counters, held state
and active correction parameters before/after evaluation; mutation or nonfinite
scores fail the run. Physical snapshots hash one CMO matrix at a time rather than
cloning the complete twelve-MLP physical state simultaneously. Live-canary timing
must include this integrity overhead. No shared-manifest module was edited.

CPU validation used `/home/filip/miniconda3/envs/py312/bin/python`, with CUDA hidden:
`pytest tests/test_fault_tuning_evaluation.py tests/test_fault_tuning.py
tests/test_fault_completion_controls.py tests/test_drift_tuning.py
tests/test_workflow_*.py -q` passed **741 tests**, with **5 skipped**, in52.02s.
Tiny OPT/CMO tests cover all arm methods and two digital ranks, live adapter and
calibration scoring before cleanup, pretraining pin failures, contiguous cohorts,
rejection of physical/counter/RNG/correction mutation and nonfinite scores, and
legacy behavior/digest preservation. No GPU work, launch, merge or push occurred.
