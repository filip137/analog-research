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

Implementation active; coordinator owns validation processes. Full baseline suite running before extraction. No scientific results or performance improvements claimed.

## Extraction checkpoint

The baseline scoped suite reported 1521 passed, 49 skipped and three pre-existing calibration import failures. An initial unscoped collection incorrectly included frozen results/source trees; its failed log is preserved, and all subsequent suites explicitly target tests/.

Relocated reusable model/training and execution modules. Split pulse plants, endpoint laws, measured cohorts, neutral config/artifact utilities, held CMO arrays and recovery coordination. Hardware receives explicit fault domains. OPT native and physical forwards now share a traversal, and HWA sampling is method-owned. Fixed the stale calibration import.

Validation: 131 initial extraction tests, 111 boundary/recovery tests and 65 path/training-golden tests passed. Eight independent OPT forward/gradient/state traces are bit-exact against baseline; all 573 accepted resolved example/mode configurations are identical. All 15 shared modules remain unchanged. Full post-relocation failures identified missing moved constants, missing ignored device-data staging, stale test instrumentation and relocation paths; those causes have been repaired and checked in affected suites.

Remaining: domain configuration/builders, external sampling composition, lifecycle public routing, final boundaries/install/replay checks, documentation and integration.
