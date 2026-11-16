# Corruption paper grid — native preparation

User authorized scheduling the paper sweep on16November2026; owning issue
filip137/on-chip-learning#33. Shared case inventory and protocol live under
campaigns/fault-recovery-paper/launch/20261116-paper-grid in the research home.
On16November2026 at17:43UTC the user chose DRN on riri/fifi RTX5090 and
crossbar jobs on Jean Zay V100. Root binds a48GPU-hour campaign ceiling over
72hours, ending19November2026 at17:43UTC, including every prerequisite, probe,
held-out evaluation and retry. Concurrency is at most two DRN5090 plus two XB
V100 GPUs. This replaces the earlier pending-envelope choice. Root owns release
after source integration, resource/profile binding and the exact live gates.

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

## XB V100 canary preparation — 16 November, after placement authorization

Native readiness configs are in
`campaigns/fault-recovery-paper/lifecycles/paper-grid-20261116/canary/`:

| Lifecycle/file stem | Arms | Updates | Lifecycle digest prefix |
| --- | ---: | ---: | --- |
| `fault-paper-grid-20261116-xb6-all-arms-canary` | 9 | 4 | `8fa0e3ad937b` |
| `fault-paper-grid-20261116-xb12-all-arms-canary` | 9 | 4 | `d9ff62511c4d` |
| `fault-paper-grid-20261116-xb6-standard-bscale-fit-canary` | 1 | 2 | `23e0065d1ce1` |

Both all-arm configs retain the full1024-sequence recovery pool and both full
held-out cohorts:1024WikiText test and384generated test. They observe development
at2/4updates, use the original3% tuning deployment/map271001, fixed LoRA seed271001,
standard eight sampled targets `df73ce66…`, and all nine declared method arms.
The XB6 legacy scale `494f5c05…` is **readiness only**, never paper evidence or an
LR freeze. XB12 uses its standard frozen scale `e1aa6c11…`. Rebuilt XB6 scale
and the standard-target analog LR recheck still gate XB6 paper release.

The two-update fit canary preserves the prerequisite's exact rank16, LR3e-3,
standard samples and adapter-state label `dlora-r16-3em3`; only its schedule,
readiness question and lifecycle ID differ. It has no held-out scoring, matching
the full prerequisite fit. Its state must not be used to derive the scientific
B scale; that requires the validated300-update fit.

CPU checks: all three native `ebl lifecycle check` and `describe --json` calls
passed; descriptions retain prepare/devices/HWA-import/fault_tuning stages and
28,311,552/56,623,104 logical weights for XB6/XB12. Native filename stems equal
lifecycle IDs. Config assertions verify exact arm coverage, cohorts, sample pins,
matched-control LR and unchanged fit science. The focused evaluation/fault/control/
token-preparation suite passed62tests in2.44s with CUDA hidden. Worker/staging/
manifest Python compilation and `bash -n` on existing wrappers passed. No jobs
were submitted; source staging, scheduler checks and live semantic gates remain.

Root manifest binding must supply the immutable Git-backed source, per-depth HWA
weights, teacher, standard sample file, both evaluation corpora, synthetic HWA
corpus, deployment and scale pins. File hashes and raw tensor hashes are distinct.
`run.slurm` executes a worker copy beside its manifest: stage/hash-compare that
copy against the tracked source worker and bind its identity in the launch gate.
The worker and runtime verify `source_snapshot.json` tracked-file hashes, while
native bundles retain its receipt hash; root must also bind actual source HEAD,
receipt/config/manifest/wrapper hashes and unchanged resource/module/export
profile. A Git-backed snapshot with only its receipt excluded retains clean-source
identity. The existing manifest helper is tuning-specific; it does not generate
these new unit/corpus bindings automatically.

The repaired worker validates native artifacts, arm/method coverage, finite
development curves and complete updates, plus both named held-out records per
all-arm endpoint with exact cohort identities/counts, finite KL and explicit
unchanged-state evidence. Root's live semantic gate still reconciles physical/
correction state, calibration/LoRA invariants, timing and native completion.
`jz_prepare.sh` logs failed preparation units without propagating a
nonzero aggregate exit: inspect each preparation result, or fix this wrapper
before relying on it. These findings were reported to root; no worker or wrapper
changes were included in the canary-config preparation commit. The separately
assigned worker repair below closes its completion-validation gap.

## Launch-worker completion and source guards

`launch/worker.py` now validates every native endpoint before publishing
`complete.json`: exact arm labels/order and methods with no duplicates, finite
final development KL/losses, configured training-loss count and final curve step,
and zero updates/empty curves for holds. Every configured held-out cohort must
appear exactly once with its exact name/corpus/file-and-tensor pins/split/slice,
finite full-teacher KL, correct sequence and next-token-position counts, and
`state_unchanged` explicitly true. Legacy configs without evaluation cohorts
retain their no-held-out path. Test scores are validation evidence, not selection.

The manifest optionally pins `source_receipt_sha256`; new root manifests set it.
Before preparation or execution, the launch-directory worker's own file hash must
equal the tracked worker in the declared source. Receipt file hashes are checked
with resolved paths constrained to that source. Thus a stale external worker,
changed receipt or changed tracked source cannot silently pass the source gate.

Worker regressions include malformed/missing/duplicate/unexpected held-out rows,
every cohort-identity field, nonfinite KL, wrong prediction counts, false state
guards, incomplete/duplicate/wrong arm coverage, legacy behavior and stale source/
receipt/worker rejection. A main-path regression emulates successful native
execution/inspection followed by missing held-out evidence: only a failed receipt
is published, never completion. The combined launch/evaluation/fault/control/token
suite passed99CPUtests in2.53s with CUDA hidden; compilation and diff checks passed.
Only the worker, its tests and this note changed in this bounded repair. No GPU
work, integration, merge or push; root's concurrent prerequisite rename is preserved.

## Independently frozen XB12 column

All XB12 learning rates are frozen from the original 3%/271001 screens and
completion controls. Prepare 12 cells (0/1/3/5%, maps271002–271004), each nine
arms and both full held-out cohorts:108 endpoints. Original1024 sequences,
eight stored teacher targets,300 updates; no test-based selection. Existing
XB12 standard-target B-scale is unchanged. Native configs live in
`campaigns/fault-recovery-paper/lifecycles/paper-grid-20261116/paper/`.

The four-step XB12 paper canary uses3%/271002 and the same native internal
deployment path as the paper cells; it retains all arms and full test scoring.
Run it from this newly integrated source on the identical V100 profile before
releasing any full cell. Full jobs are bounded waves with at most two crossbar
allocations including XB6 prerequisites, within the shared48GPUh ceiling and
19November17:43UTC deadline. Root owns launch and recovery decisions; durable
Sol6.1medium observers report completion/failure and resume the root.
