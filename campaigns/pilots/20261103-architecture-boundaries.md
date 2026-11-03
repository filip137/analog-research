# Architecture boundaries migration

State: implementation. Branch: `exp/20261103-architecture-boundaries`.
Authorization: user-approved architecture plan, 3 November 2026. Engineering work
only; no scientific campaign, remote job, service deployment or shared GPU use.

## Frozen comparison

Production baseline: `435974e24adea2aefbf5f02a4a3d99848f093d66`, including the
CMO compact-state replay fix. Preserve scientific configuration, physical laws,
matrix identities, random coupling, artifact formats and cost accounting exactly.
Separate network architecture, hardware state, methods and execution. Workflow
becomes documentation; its commands move to `ebl lifecycle`.

Tiny and OPT-width synthetic CPU workloads compare baseline and extracted code,
using FP32, one thread, seeds 1729/2718/31415/1618, three warm-ups and ten timed
repetitions in three fresh processes per version. Setup is timed separately from
operations. Report distributions, unique persistent storage, peak RSS and native
checkpoint size/save/load. These measurements make no scientific accuracy claim.

## Budget and evidence

Interpreter: `/home/filip/miniconda3/envs/py312/bin/python` (Python 3.12.12,
Torch 2.5.1+cu121). CPU only, no downloads. Benchmark batch bounded to 30 minutes
and 4 GiB resident memory per process. Raw outputs:
`results/architecture-boundaries-20261103/`. No frozen evidence is rewritten.
Proportional CPU tests accompany each extraction; final suite and baseline replay
comparisons cover the resulting dependency graph. Native AIHWKit checks use the
existing separate interpreter where needed.

## Handoff and decisions

Owner: implementing agent. No experiments launched. DRN and on-chip-learning
changes are limited to shared modules and necessary imports/packaging/maintained
launch instructions. Existing physical regimes remain distinct. Resume capabilities
remain epoch-boundary (historical codecs), completed stage reuse, and completed
drift shard continuation; lifecycle mid-stage resume stays unsupported.

## Evidence and review

Planning preflight: 52 focused replay/drift/checkpoint tests passed. Implementation
checks and comparable measurements will be recorded here before closeout.
