# Crossbar lifecycle workflow

State: implementation and local verification complete, 2026-10-05.
Authorization: user request to restructure the worktree into reusable,
reproducible experimental loops. Proportional checks only: CPU tests and three
labelled 128-image CPU smokes. No GPU, remote or full-cohort training.
Base revision: `37da9dd`.

## Question and decision

Can one campaign-owned definition drive a complete crossbar loop through
reusable functions, one per step? The steps are: question, devices and defects;
then (i) deployment of the digital weights, (ii) HWA, (iii) program-and-verify,
including cycles and relaxation, and (iv) on-chip training. The loop must
reproduce the historical CIFAR laws and run through the public CLI.

Decision: if it can, new crossbar loops are defined as lifecycles, and the nine
historical `cifar_*` families stay frozen for reproduction.

## What changed

- `workflow/` package, with one function per lifecycle step:
  - `lifecycle.py`: strict stdlib-only schema, cross-section rules, section
    digests and stage DAG.
  - `devices.py`: fresh arrays and defects.
  - `deployment.py`: mapping.
  - `hwa.py`: none, noise-aware or corruption-aware HWA with
    selection-array checkpoint choice.
  - `program_verify.py`: closed-loop, open-loop and Gaussian endpoint
    programming, the PCM drift relaxation law, and OM endpoint characterization.
  - `onchip.py`: matched hold, rewrite and learn arms with or without
    calibration, and unified pulse writers.
  - `runtime.py`: the `ebl train` stage entry point.
  - `plan.py` and `__main__.py`: `python -m workflow describe|check|plan|collect`.
- Registered `crossbar_lifecycle.v1` (14 families).
- The campaign runner now accepts the CLI's existing `device_state` and
  `selection_receipt` inputs, so planned lifecycles run under
  `ebl campaign run`.
- Three smoke lifecycles in
  [the CIFAR campaign](../cifar-crossbar-hwa-recovery/lifecycles/README.md).
- Contract in [docs/crossbar_lifecycle.md](../../docs/crossbar_lifecycle.md);
  scope, runtime, README, AGENTS and skill pointers updated.
- No historical module, config, bundle or ledger changed.
  `experiments/cifar_crossbar/` is byte-identical.

## Verification design

- Bit-exact CPU equivalence of each stage with the historical code it
  generalizes, using a random ResNet-32, synthetic cohorts and populations
  (`tests/test_workflow_stages.py`, 26 tests):
  - PCM layered endpoints;
  - OM fresh arrays with closed-loop P&V (V2 `make_array`);
  - nominal open-loop programming;
  - OM kernel tables (`build_kernels`);
  - the PCM drift law (`PcmInferenceNoise`);
  - capped and uncapped closed- and open-loop writers;
  - every training arm for PCM and OM (`full_epoch_runtime.trajectory`);
  - noise-aware HWA (V2 `fit_candidate`), covering selection history, the
    selected source and the trained epoch-1 master.
- Runtime glue on synthetic inputs (`tests/test_workflow_runtime.py`):
  - complete PCM, PCM-drift and OM DAGs;
  - every bundle passes the campaign runner's result contract;
  - section-based reuse is accepted and changed-defect rejection is preserved
    as a failed bundle;
  - exact stage inputs are enforced.
- Schema, planner and manifest checks (`tests/test_workflow_lifecycle.py`, 30 tests).
- Real-data CPU smokes through `python -m workflow plan` and then
  `python -m ebl campaign run --allow-dirty`. They use the pinned CIFAR-10
  teacher, 128-image cohorts, one HWA epoch and one on-chip epoch. They mirror
  the V2 canary's seeds and cohorts (assignment 271001, selection arrays
  211001–211003 for PCM and 211001 for OM), so P0 and epoch-1 results can be
  compared with the archived canary.

## Budget and execution

Host nom-cool-2, Python 3.12.12, PyTorch 2.5.1. Native sampler: AIHWKit 1.1.0
(`/home/filip/miniconda3/envs/aihwkit/bin/python`). Four CPU threads, CUDA
hidden. Source identity in every manifest: `37da9dd` plus uncommitted changes
(dirty hashes recorded).

| Lifecycle | Wall time | Bundles | Storage |
| --- | --- | --- | --- |
| `cifar10-pcm-conv4-smoke` | 35.2 s | 11 | 188 MB |
| `cifar10-pcm-conv4-drift-smoke` | 34.7 s | 11 | 208 MB |
| `cifar10-om-conv4-smoke` | 63.0 s | 11 | 391 MB |

Outputs: `results/lifecycles/<id>/{plan,runs,collected}` (ignored). In the
first PCM runs directory, `attempt-001` holds the preserved dry-run records and
`attempt-002` the execution. Every trajectory keeps its final state: about
4.8 MB (PCM) or 10.3 MB (OM) at four convolutions.

## Current execution and monitoring handoff

Nothing is running. No retention action was taken; the smoke outputs remain
local evidence.

## Evidence and coverage

- Test suite, CPU-only (`CUDA_VISIBLE_DEVICES=""`): **1,891 passed, 12
  skipped**. That is the prior 1,830 plus 61 new tests; the skips need CUDA or
  native AIHWKit.
  - A first full-suite attempt exposed the GPU, which an unrelated job was
    using. Its CUDA tests briefly allocated about 360 MiB; I stopped it after
    about 20 s and reran it CPU-only. The unrelated job was unaffected.
- `python -m ebl runs inspect … --verify-artifacts --require-complete`: all 33
  smoke bundles complete and valid.
- `python -m workflow collect`: 30 rows per lifecycle (3 HWA arms × 2 cases × 5
  on-chip arms), no duplicate coverage, every final state replayed exactly.
- Cross-check against the archived V2 canary
  (`results/cifar-sweep-collected/v2/results/cifar10-*-conv4-fault-sweep-canary-v2`),
  digital source:
  - **OM:** P0 test predictions are identical for both cases (nominal KL
    0.012265; 5 % stuck-high 222.2987 vs 222.2987, accuracy 7.81 %). All 8
    epoch-1 on-chip trajectories match: identical predictions, persistent
    diagnostics and physical pulse counts (509 and 14,499). Remaining KL
    differences are below 2e-7 relative, attributable to CPU versus the
    historical GPU.
  - **PCM:** fault maps match, but P0 does not (nominal KL 0.0040 vs canary
    0.0021). The historical `GaussianEndpointArray` draws programming noise
    from the execution device's generator, and the canary ran on CUDA. The
    behavior is retained and documented as a limitation.
- Smoke sanity, measured but not evidence:
  - The digital mapping reproduces the teacher (KL ≈ 0).
  - 5 % PCM gmax fails 14,843 of 296,192 devices.
  - `standard_hwa` selected epoch 0 in both PCM smokes, so its deployment is
    identical to `digital`.
  - Hold arms performed no writes.
  - Drift relaxation raised the nominal residual from 0.021 (programmed) to
    0.030 (relaxed after 1 h, compensated).

## Interpretation and next decision

The lifecycle path is ready for defining new crossbar loops. It executes all
five stages through the public CLI and campaign runner. On CPU it reproduces
the latest historical PCM and OM laws exactly. On real data the OM path
reproduces archived canary deployments and epoch-1 recovery. The smokes test
the workflow, not the scientific question; their truncated cohorts support no
claim about on-chip learning.

Next decisions for the user:

1. Write the first scientific lifecycle under a new experiment note, with a
   predeclared question, cases, budget and decision rule.
2. Decide whether PCM endpoint draws should become device-independent. This
   would be a new law version, not an edit to the historical class.
3. Optionally port historical families onto `workflow/`, guarded by the same
   equivalence tests.
4. Add missing capabilities when needed:
   - dense CIFAR-10 and MNIST networks;
   - hyperparameter selection lifecycles;
   - stage resume;
   - drift-aware HWA noise;
   - OM relaxation, which needs a sourced model first.

## Workflow audit

Most of the effort went to reading the nine CIFAR runtimes and their shared
modules. Implementation, tests and smokes followed. The smokes took about
2.2 minutes of wall time. Token usage is unknown in this environment. Recovery
check: the lifecycle files, this note and the contract document identify the
question, the commands, the bundle locations, the outcome and the next decisions.
