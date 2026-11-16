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
