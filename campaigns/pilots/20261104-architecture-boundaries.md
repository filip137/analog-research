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
