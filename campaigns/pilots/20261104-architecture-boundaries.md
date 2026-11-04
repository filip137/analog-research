# DRN architecture boundaries

## Decision and contract

Implement the accepted HWA-style ownership migration for all maintained DRN families, on one branch with one final cutover. Retire old imports; preserve numerical behavior, configuration digests, artifact schemas and supported continuation. Temporal-correlation and CMO-rewrite physics are separate work. No new scientific campaign or remote/GPU launch.

Baseline: `a238498c7856609ca04448dcfa324c6d79e15660`. Branch: `drn/20261104-architecture-boundaries`. Worktree: `/home/filip/analog-research/worktrees/drn-20261104-architecture-boundaries`.

## Implementation and acceptance

Owners are core (neutral utilities), hardware (physical state/laws), networks (equations/solvers/mapping/data), methods (learning/controllers), ebl (execution), experiments (scientific compositions), campaigns (research), workflow (documentation), labs (reference harnesses). Lower owners must not import higher owners or labs.

Sequence: freeze baseline; extract utilities/devices; extract networks; extract methods; consolidate execution; verify, integrate and close. Held apparent forwards, operation/RNG ordering, controller observation restrictions, and existing warm-start/optimizer/cursor continuation remain invariant.

Acceptance requires existing golden replay, exact bounded before/after traces, import boundaries, config identity, isolated installation, canonical runner reuse and all 15 shared module bytes. Source/timing provenance may truthfully change; scientific payloads may not.

## Validation budget and evidence

Local CPU only, py312 interpreter (`/home/filip/miniconda3/envs/py312/bin/python`, PyTorch 2.5.1+cu121). Existing tests and bounded replay, up to four lifecycle replay workers. No training sweeps. Expected validation duration: tens of minutes, bounded to two hours per full test/replay attempt; stop and diagnose failures rather than repeating unchanged suites. Raw logs and additional baseline traces: `/home/filip/analog-research/drn/results/20261104-architecture-boundaries/`. Existing golden fixtures remain unchanged.

Planning checks: 28 focused tests passed; 119 declared golden fixtures present; 15 shared source files byte-identical. CLI parser currently imports Torch/NumPy. Pre-existing stale `training.output_calibration` import blocks OPT calibration.

## Current handoff

Implementation committed through `d266d38`; explicit CPU replay support is the next checkpoint. Coordinator owns the second lifecycle replay (four single-thread CPU workers; `lifecycle-cpu-replay.log`, records under `lifecycle-cpu-records/` in the evidence root). The first attempt completed in 733.48 s. It exposed ambient CUDA RNG snapshots in the goldens and seven cases launched before the data import fix. Next: compare all saved CPU records, resolve remaining differences, then integrate and push. No research launch or GPU work is authorized.

## Extraction checkpoint

The baseline scoped suite reported 1521 passed, 49 skipped and three pre-existing calibration import failures. An initial unscoped collection incorrectly included frozen results/source trees; its failed log is preserved, and all subsequent suites explicitly target tests/.

Relocated reusable model/training and execution modules. Split pulse plants, endpoint laws, measured cohorts, neutral config/artifact utilities, held CMO arrays and recovery coordination. Hardware receives explicit fault domains. OPT native and physical forwards now share a traversal, and HWA sampling is method-owned. Fixed the stale calibration import.

Validation: 131 initial extraction tests, 111 boundary/recovery tests and 65 path/training-golden tests passed. Eight independent OPT forward/gradient/state traces are bit-exact against baseline; all 573 accepted resolved example/mode configurations are identical. All 15 shared modules remain unchanged. Full post-relocation failures identified missing moved constants, missing ignored device-data staging, stale test instrumentation and relocation paths; those causes have been repaired and checked in affected suites.

Domain settings/builders, external sampler orchestration, physical measured-trace banks and public lifecycle routing are complete. Obsolete Python facades are retired. Remaining: lifecycle replay completion and final integration.


## Acceptance evidence

Interpreter: `/home/filip/miniconda3/envs/py312/bin/python` (PyTorch 2.5.1+cu121).
External CPU sampler: `/home/filip/miniconda3/envs/aihwkit/bin/python` (AIHWKit 1.1.0).
All commands set `CUDA_VISIBLE_DEVICES=''`; ordinary CPU tests use `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`.

- `python -B -m pytest tests -q -p no:cacheprovider`: **1505 passed, 49 skipped**, 78.66 s (`full-cpu-tests-final.log`). Includes the available native CPU characterization replays. The opt-in lifecycle cases run separately; unavailable CUDA/artifact-dependent checks are not claimed passed.
- `python -B -m pytest labs/tests -q -p no:cacheprovider`: **50 passed, 1 skipped**, 5.40 s (`lab-tests.log`).
- `EBL_RUN_LIFECYCLE_GOLDENS=1 EBL_GOLDEN_CPU_ONLY=1 EBL_GOLDEN_JOBS=4 EBL_GOLDEN_RECORD_DIR=<evidence>/lifecycle-cpu-records python -B -m pytest tests/test_run_lifecycle_golden.py -q -p no:cacheprovider`: second attempt running (`lifecycle-cpu-replay.log`). Each case uses one CPU thread. Strict default comparison is unchanged; explicit CPU-only comparison excludes only the absent global CUDA RNG snapshot, asserts CUDA is unavailable, and compares all other saved random, optimizer, device, loader and solver state exactly. GPU RNG continuation is untested. Two focused comparator/registry tests passed; saved base, base-resume, digital-LoRA, HWA and three KD records already match.
- All **573** normalized example/mode configs, **8** OPT forward/gradient/capture/state traces and **15** shared source files match the baseline. Shared bytes also match the current HWA checkout. Examples, historical study plans and existing golden fixture bytes are unchanged.
- A wheel built without dependency downloads installs under `/tmp/drn-wheel-installed`. From `/tmp` with Python isolated mode it passes parser discovery, a tiny perfect-diode training/bundle integrity smoke, resource discovery and real external OM sampling. Every imported project module resolves inside the installation. The wheel contains the new network data and Slurm resource, and no retired Python packages (`wheel-build.log`, `installed-smoke.log`).

The first full post-extraction suite exposed a missing data-builder device import and test instrumentation still patching its old owner; both were repaired and 23 affected tests passed before the full green suite. The raw-active characterization initially differed only in its embedded sampler source hashes and the resulting SQLite file hash. The pinned baseline passed. Direct comparison showed every scientific row unchanged. The harness now checks the live database hash and normalizes only declared source provenance in embedded metadata; a regression test proves seed and trajectory changes still fail. No golden recapture or numerical tolerance change was made. Failed attempt logs are retained.

## Bounded performance measurements

Five separate parser processes on the same CPU/environment: median startup **1.309 s → 0.112 s**, median maximum RSS **420200 → 19892 KiB**. CLI discovery no longer imports the numerical stack (`import-benchmark.json`).

A single-thread tiny physical OPT forward, 2×4 tokens, 20 warmups then five blocks of 100 forwards: median **4.090 → 4.231 ms** (`workload-benchmark.json`). This does not establish a kernel speedup; the modest increase is reported as measured on this host alongside concurrent CPU replay. No GPU performance claim.

Raw evidence totals about 5 MiB plus temporary lifecycle replay outputs. Validation and engineering time are not experiment throughput; token usage is unavailable. No new study result is inferred from this architecture work.


Validation recipes (`drn_baseline.py`, `drn_import_bench.py`, `drn_workload_bench.py`,
`drn_install_smoke.py`) are retained in the raw evidence root. Their recorded paths
refer to checkouts at validation time; reproducing the baseline after integration
requires a separate checkout of the baseline SHA. Existing golden fixture bytes
remain the authoritative numerical references.
