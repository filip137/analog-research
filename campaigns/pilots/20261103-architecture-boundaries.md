# Architecture boundaries migration

State: reviewed. Engineering acceptance passed; no scientific campaign launched.
Authorization: user-approved architecture plan, 3 November 2026. Implementation
branch: `exp/20261103-architecture-boundaries`; integration follows the skill's
closing git checkpoint. No remote experiment, shared GPU or overnight run.

## Question and frozen comparison

Can network architecture, physical state, algorithms and orchestration have clear
owners while preserving the native scientific/replay contracts? Accept when all
active callers migrate, directional imports and package discovery hold, exact
replay/config comparisons pass, and supported reference execution remains tested.
A physical discrepancy is a separate scientific decision, never an implicit fix.

Production baseline: `435974e24adea2aefbf5f02a4a3d99848f093d66`, including the
CMO compact-state replay fix. Benchmark-only baseline commit: `3a8bd5f`.
Extraction: `014df03`; completed boundary/runner/navigation candidate: `7eac4d9`.
Serialization consolidation and explicit constructor validation: `1d9c620`.
Final tested code: `04039d8` (sampling imports its path from `core`).
DRN coordinated migration: `a238498`; study navigation update: `557fa36`.

## Resulting structure and conflicts

The [architecture map](../../docs/architecture.md) records concrete files and
functions, configuration/state/RNG/serialization/cost ownership, dependency rules,
each conflict's root cause/resolution/check and retained adapters. Its ownership
follows the foundation-to-composition order `core` → `hardware` →
`networks`/`methods` → `ebl`, with the precise allowed
edges in its table. `workflow/` now contains agent guidance and the authoritative
execution policy. Current lifecycle commands are `python -m ebl lifecycle ...`.

Resolved architectural conflicts include reference-owned reusable primitives,
OPT naming inside fault construction, duplicated physical/native OPT traversal,
physics inside stage/method code, collection importing execution, and a campaign
stage runner that reused any completed result. Selected stages and artifact lookup
now use the canonical runner's source/config/input fingerprints and verified native
artifacts. Every maintained launch-kit caller was migrated, including local/Slurm
wrappers, source lookup, inspection and collection. Restricted checkpoint loading
no longer retries unrestricted loading on `TypeError`.

CMO, OM and PCM remain distinct. The migration changes no declared analog subset,
fault persistence, temporary HWA perturbation, held apparent forward/gradient,
drift timing, write reset, stochastic coupling or physical cost law. Existing
fixed-episode drift and exact-endpoint writable behavior remain; no new drift
regime was implemented or adopted from concurrent experiment branches.

## Scientific and engineering checks

- Final pinned HWA CPU suite: **2,965 passed, 41 skipped**, one existing tokenizer
  deprecation warning, 125.78 s. Local tmux integration suite: **15 passed**, 5.83 s;
  temporary Unix sockets required execution outside the filesystem/network sandbox.
  These were synthetic local worker tests with GPUs disabled, not remote jobs.
- AIHWKit 1.1.0 reference checks: **36 passed**, 128 unrelated tests deselected,
  2.39 s. The installed AIHWKit environment lacked pytest; a temporary venv with
  system site packages and pytest 8.4.2 ran the existing tests. It used Python
  3.12.13/Torch 2.11.0+cu130 on CPU; the shared environment was not modified.
- DRN power-law, writable-drift and OPT recovery checks: **24 passed**. All **15**
  declared shared modules are byte-identical between the independent checkouts.
- All **272** maintained lifecycle digests match the frozen baseline exactly.
  The final OPT capture covers **10** native/physical scenarios, both analog-only
  and whole-model students, corrections and capture modes; logits, gradients and
  captures match exactly. The **12** hierarchical-fault scenarios match **96**
  tensors plus scalar metadata, seeds and receipts exactly.
- Every baseline/refactor benchmark trace matches, including gradients, three
  update losses, network/device/writer/RNG state, future drift and cost. The final
  code candidate repeats the captures and a verification-only tiny trace after
  serialization consolidation; its timings are excluded from the paired results.
- Boundary tests check forbidden imports, stdlib-only CLI parsing, explicit fault
  metadata, unsupported physical technology, documentation-only `workflow`, and
  restricted loading without fallback. No top-level import cycles were found in
  the 87 maintained domain modules. Static module imports resolve.
- The wheel builds and imports `core`, `hardware`, `networks`, `methods` and `ebl`
  from an isolated installation outside the checkout; lifecycle CLI discovery
  works there. Three skill reference/link checks and shell syntax checks passed.

No numerical tolerance was used for migration replay. Independent upstream
AIHWKit equation checks retain their existing FP32 `rtol=atol=1e-6` to accommodate
rounding of equivalent expressions; pulse transitions and random coefficient draws
use exact equality. This tolerance is not permission to weaken native replay.

Initial validation exposed environment and migration issues, all retained in logs.
The baseline's child subprocess failures were an MKL/libgomp conflict; setting
`MKL_THREADING_LAYER=GNU` made the affected collector/LoRA subset pass (**30 tests**).
Its tmux failures were sandbox socket restrictions. Extraction checks caught stale
fixture targets and CLI argument assertions, which were migrated. A final collection
attempt caught sampling importing `ROOT` through `hardware.arrays`; the caller now
imports the responsible `core.paths` owner directly. The final suite above supersedes
those failed attempts; none is presented as passing evidence.

## Comparable CPU measurements

Python 3.12.12, Torch 2.5.1+cu121, FP32 CPU, one intra/inter-op thread, deterministic
algorithms, seeds 1729/2718/31415/1618. Three warm-ups and ten repetitions per
operation, **three fresh processes per version/workload**, alternating order. State
restoration is outside timings. Timings compare pinned `3a8bd5f` and `7eac4d9`;
subsequent edits only consolidate unchanged serialization bodies, reject unknown
technology and correct the sampling path import. The final exact replay checks
cover those edits. No tests ran concurrently with timing measurements.

The batch took **313.61 s**, within the 30-minute cap; every process remained below
4 GiB RSS. `tiny` has 2,048 analog weights. `opt_mlp` has one OPT-125M-width MLP,
4,718,592 analog weights, a reduced vocabulary and short synthetic sequences. It
is not a complete OPT-125M training run. Values below are milliseconds: median of
three process medians, with their min/max range. Within-process IQR and all samples
are retained in the raw JSON.

| Workload | Operation | Baseline ms [range] | Refactor ms [range] | After/before |
| --- | --- | --- | --- | --- |
| tiny | forward_backward | 1.064 [1.054, 1.309] | 1.031 [1.027, 1.046] | 0.969 |
| tiny | adaptation_step | 1.208 [1.203, 1.211] | 1.191 [1.190, 1.205] | 0.986 |
| tiny | program | 0.099 [0.098, 0.102] | 0.100 [0.099, 0.101] | 1.004 |
| tiny | drift_update | 0.060 [0.059, 0.060] | 0.060 [0.060, 0.061] | 1.003 |
| tiny | checkpoint_save | 0.521 [0.495, 0.528] | 0.507 [0.486, 0.542] | 0.973 |
| tiny | checkpoint_load | 0.585 [0.579, 0.593] | 0.591 [0.579, 0.594] | 1.010 |
| opt_mlp | forward_backward | 24.461 [23.841, 25.955] | 24.953 [24.494, 26.718] | 1.020 |
| opt_mlp | adaptation_step | 186.371 [178.079, 195.599] | 183.757 [166.167, 287.606] | 0.986 |
| opt_mlp | program | 192.572 [173.567, 193.763] | 171.380 [167.621, 177.750] | 0.890 |
| opt_mlp | drift_update | 159.123 [158.318, 165.649] | 160.281 [160.061, 160.444] | 1.007 |
| opt_mlp | checkpoint_save | 450.015 [443.405, 503.349] | 431.389 [410.580, 490.654] | 0.959 |
| opt_mlp | checkpoint_load | 122.005 [106.589, 129.590] | 109.300 [108.698, 112.918] | 0.896 |

| Workload | Persistent bytes, both versions | Checkpoint bytes, both versions | Baseline peak RSS MiB [range] | Refactor peak RSS MiB [range] |
| --- | --- | --- | --- | --- |
| tiny | 159,744 | 148,834 | 515.6 [509.8, 515.8] | 406.7 [406.3, 408.8] |
| opt_mlp | 368,050,176 | 320,873,890 | 2126.3 [2117.7, 2136.0] | 2012.8 [2003.9, 2022.5] |

Persistent state and checkpoint size are unchanged. Process RSS drops about
110 MiB. A separate three-process import-only check gives baseline 511.5–512.4 MB
versus refactor 401.8–402.0 MB: eager CIFAR/torchvision and its transitive imports
are absent from the OPT path. This supports an import-memory improvement, not
smaller physical state or GPU memory claims.

The baseline profile attributes about 19% of self CPU time to fills, 16% to powers,
15% to multiplication, 11% to addition and 8% to normal draws. Copies account for
about 3%; matrix multiplication is below 1% in this short state-heavy workload.
State restoration is repeated only by the benchmark setup and is excluded from
operation timings. Snapshot copies and held/RNG state remain explicit; no writes,
random draws, precision or physical state were dropped. Per-identity/pulse loops
retain their replay ordering. GPU transfer costs were not measured.

Timing ranges overlap, and adaptation includes a slow refactor process. The roughly
2% higher median forward/backward time and sub-1% drift difference are within the
observed ranges; no established compute regression or speedup is claimed. Lower
programming/checkpoint medians are observations, not an optimization claim: the
kernels are unchanged and allocation/filesystem/host variability matters. Three
processes, a warm filesystem and synthetic sequences cannot predict campaign or
GPU performance. No further kernel optimization was justified by this migration.

## Evidence, adapters and limitations

Ignored raw evidence is retained under `results/architecture-boundaries-20261103/`:
`paired/conditions.json`, per-process `measurements.json`, `paired/summary.json`,
baseline profiles, exact captures, import-memory measurements and validation logs.
The [benchmark guide](../../benchmarks/README.md) gives the reproducible commands.

Permanent supported boundaries are the network-specific physical adapter, native
AIHWKit subprocess sampling, independently mirrored DRN kernels, historical codecs
and family entry points, and the old execution-policy URL as a documentation
redirect. There are no blanket Python import or workflow CLI facades. Historical
commands run in their pinned source checkout; frozen bundles/archives and scientific
notes were not rewritten. The shared study's active experiment checkout is untouched.

Lifecycle mid-stage resume remains unsupported. Completed-stage reuse, historical
epoch checkpoint restore and completed drift-shard continuation keep their own
contracts. GPU execution, remote launch kits and full-scale scientific campaigns
were not exercised; their source callers and local contracts were checked. Skipped
tests remain labelled skipped. No unresolved scientific choice was silently decided.
The outcome is an accepted architecture refactor with exact bounded replay evidence,
not new evidence for any physical model or adaptation method.
