---
id: "exp-001"
title: "Initial population HWA/programming pilot"
status: "imported"
hypotheses: ["H-001"]
---

# exp-001 — Initial population HWA/programming pilot

## Question and provenance

Initial population HWA/programming pilot. Off-chip BPTT trained against the calibrated IBM OM cap-128 endpoint distribution will reduce the KL and accuracy loss after matched pulse-resolved deployment relative to clean BPTT and a 3% per-output-channel Gaussian HWA control; explicitly retaining the published corrupt/stuck cells may leave a larger irreducible gap than the counterfactual repaired population.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-001-population-hwa.md).

Dependencies and related controls: [Focused cap-128 HWA endpoint prerequisite](../../../../ibm-om-programming/series/001-controller-and-endpoint/results/exp-004-hwa-device-model.md).

## Frozen plans

- [MNIST DRN IBM OM HWA versus pulse-resolved program-and-verify](../../../../../studies/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1.json) — `mnist-ibm-om-hwa-program-verify-pilot-20260822-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.

Historical document locations (now references/forwarding pages): [ibm_om_hwa_program_verify_pilot.md](../../../../../docs/ibm_om_hwa_program_verify_pilot.md).

<a id="source-ibm_om_hwa_program_verify_pilot-ibm-om-hwa-followed-by-pulse-resolved-deployment"></a>

## Historical source: IBM OM HWA followed by pulse-resolved deployment

- Status: cap-128 device artifact and all four training arms are complete;
  the 16 predeclared held-out validation arms are pending
- Device source: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`
- Evidence class: hardware-derived fitted preset, not raw measured traces
- Study: [`mnist-ibm-om-hwa-program-verify-pilot-20260822-v1.json`](../../../../../studies/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1.json)

<a id="source-ibm_om_hwa_program_verify_pilot-question"></a>

## Question

Can off-chip BPTT learn weights that survive an IBM optimized-material (OM)
ReRAM program-and-verify deployment, or does a remaining matched deployment
gap justify a later on-chip SET/RESET recovery experiment?

The pilot deliberately stops before Tiki-Taka, LoRA, or soft SET/RESET
fine-tuning. A later recovery study is justified only if the predeclared
HWA-only arms underperform after the same physical deployment and a matched
recovery arm closes that gap.

<a id="source-ibm_om_hwa_program_verify_pilot-fixed-physical-identities"></a>

## Fixed physical identities

Each named DRN conductance tensor receives a fixed array-shaped AIHWKit OM
assignment. A cell keeps its sampled lower and upper bounds, SET and RESET
step parameters, reference, and corrupt/stuck status for the whole run.
Assignment fingerprints and all tensors are included in exact resume state.
Because the CUDA training interpreter does not import AIHWKit, each named
training, selection, or validation population is sampled once by the pinned
AIHWKit interpreter and transferred through a strict pickle-free NPZ plus JSON
receipt. The runtime verifies the request, tensor schema, seeds, source hashes,
artifact hash, AIHWKit version, binding geometry, and population fingerprint
before using the population.

The two IBM-aware training arms share the literal published-corruption
assignment sampled with seed `83001`:

- `om-published` retains every sampled corrupt/stuck cell;
- `om-repaired` keeps every healthy cell byte-identical and replaces only the
  sampled corrupt sites with independently seeded healthy donor cells.

This construction makes corruption the only physical-identity difference
between those two arms. It is a counterfactual repair control, not IBM's
hardware defect-management procedure and not device reassignment learned
from network performance.

<a id="source-ibm_om_hwa_program_verify_pilot-off-chip-hwa-semantics"></a>

## Off-chip HWA semantics

The optimizer owns a clean FP32 master tensor. For each minibatch, the compact
modifier:

1. maps each clean DRN conductance from `[0.1020408198, 1]` into `x in [0,1]`;
2. reuses that cell's fixed bounds and corrupt/stuck identity;
3. samples fresh accepted, failed, or corrupt terminal endpoints from the
   held-out-validated cap-128 endpoint artifact;
4. exposes the apparent endpoint for the complete forward/backward pass; and
5. restores the clean master before the ideal optimizer step.

There is no persistent physical state between HWA minibatches. This is
off-chip hardware-aware training against repeated hypothetical deployments,
not simulated on-chip incremental training. The endpoint artifact is used
only if its declared held-out adequacy gates pass. A failed gate blocks the
pilot; it does not trigger a Gaussian fallback.

The compact sampler uses deterministic reachability classes from the fixed
cell bounds. A corrupt cell keeps its sampled stuck persistent state while
fresh apparent write variation may be drawn. Non-corrupt terminal endpoint
and acceptance randomness advances once per minibatch.

<a id="source-ibm_om_hwa_program_verify_pilot-common-physical-selection-and-held-out-test"></a>

## Common physical selection and held-out test

All four training arms use the same pulse-resolved full-corrupt OM selection
assignment and endpoint seed `83003`. Selection therefore asks which clean
master checkpoint is best after the same physical deployment, rather than
choosing by clean validation and testing physical behavior afterward.

The four training arms are:

1. clean BPTT;
2. 3% per-output-channel Gaussian HWA;
3. compact OM HWA with corrupt sites repaired; and
4. compact OM HWA with published corruption retained.

The selected checkpoints are evaluated on four held-out full-corrupt
assignments `83101..83104`, with endpoint seeds `83201..83204`. The primary
network comparison is the assignment-level mean, while each assignment-level
result remains visible.

Every pulse-resolved context records programming success, budget exhaustion,
SET, RESET, and total pulses, verify reads, reversals, clipping, and corruption.
It also writes `ibm_om_deployment.pt`, containing the exact apparent and
persistent endpoints, sampled population tensors, pulse counts, RNG
continuation state, binding layout, population fingerprint, device-model
digest, and source checkpoint digest. A later fine-tuning study must consume
this persistent bundle instead of redrawing or reprogramming the base.

<a id="source-ibm_om_hwa_program_verify_pilot-device-model-prerequisite"></a>

## Device-model prerequisite

First run the two OM-only `hwa_production_cap128` characterization arms from
[`ibm-reram-om-hwa-device-model-20260822-v1.json`](../../../../../studies/ibm-reram-om-hwa-device-model-20260822-v1.json).
The focused profile keeps all 4,096 identity/repeat trajectories for the
adaptive controller at every target, but runs the one-pulse calibration only
on the 816 calibration trajectories per target. It uses lower-from-RESET only
and reads each identity's fixed sampled upper bound for analysis without
exposing that bound to the controller. Each arm therefore has 201,392
trajectories (402,784 total), rather than 671,744 per arm and 2,686,976 total
for the broader two-start, two-controller, four-preset cap-128 study. This is a
6.67x reduction in trajectories for the HWA prerequisite; the broader HfO2
and upper-start comparison remains a separately declared deferred control.

Prepare the focused characterization study with:

```bash
python -m ebl study prepare \
  --plan studies/ibm-reram-om-hwa-device-model-20260822-v1.json \
  --results-root results
```

Then launch the two declared arms with the pinned AIHWKit sampler available:

```bash
export EBL_AIHWKIT_PYTHON=/home/filip/miniconda3/envs/aihwkit/bin/python
python -m ebl characterize \
  --config examples/reram_program_verify/hwa_production_cap128_om_continuous.json \
  --output-dir results/ibm-reram-om-hwa-device-model-20260822-v1/runs/om-continuous
python -m ebl characterize \
  --config examples/reram_program_verify/hwa_production_cap128_om_corrupt.json \
  --output-dir results/ibm-reram-om-hwa-device-model-20260822-v1/runs/om-published
```

The persistent two-arm launcher is the equivalent parallel entry point:

```bash
python -m experiments.reram_program_verify.local_short_launcher \
  --study-profile hwa-om-cap128 \
  --aihwkit-python /home/filip/miniconda3/envs/aihwkit/bin/python
```

After both adaptive lower-from-RESET conditions pass their adequacy gates,
combine their endpoint and controller-calibration artifacts without refitting:

```bash
python -m experiments.reram_program_verify.hwa_model \
  --continuous-endpoint <om-continuous>/artifacts/bounded_uniform_model.json \
  --continuous-estimators <om-continuous>/artifacts/step_estimators.json \
  --published-endpoint <om-published>/artifacts/bounded_uniform_model.json \
  --published-estimators <om-published>/artifacts/step_estimators.json \
  --output data/ibm_reram_om_pv128_hwa_v1.json
```

The builder verifies OM, cap 128, continuous versus published corruption,
adaptive lower-from-RESET, half-step tolerance, calibration-only estimators,
and the endpoint adequacy result. It embeds source hashes in the output.

<a id="source-ibm_om_hwa_program_verify_pilot-running-the-pilot"></a>

## Running the pilot

Prepare the tracked study before launching any native run:

```bash
python -m ebl study prepare \
  --plan studies/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1.json \
  --results-root results
```

Each training run receives the exact bounded DRN as
`--teacher-weights` and the calibrated bundle as `--device-model`, for example:

```bash
export EBL_AIHWKIT_PYTHON=/home/filip/miniconda3/envs/aihwkit/bin/python
python -m ebl train \
  --config examples/mnist_relu_drn/ibm_om_hwa_pilot/om_repaired.json \
  --output-dir results/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1/runs/train-om-repaired \
  --teacher-weights results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237/20260820T140222.931115Z-809aec60-8da7bc60/checkpoints/weights.pt \
  --device-model data/ibm_reram_om_pv128_hwa_v1.json
```

The clean and Gaussian arms still receive `--device-model` because their
common selection metric is pulse-resolved OM deployment. Held-out validation
uses `python -m ebl validate` with one declared held-out config, the selected
checkpoint, the same bounded-DRN teacher, and the same device-model bundle.

<a id="source-ibm_om_hwa_program_verify_pilot-claim-boundary"></a>

## Claim boundary

This implementation supports a model-based statement about IBM's published
AIHWKit OM fit under this controller. It is not raw trace replay, a physical
microSiemens calibration, endurance evidence, or proof that a fabricated
array can perform 128 writes with the simulated timing and energy. Network
training must not start if the cap-128 endpoint artifact is missing or
inadequate. On-chip recovery remains a separate, later hypothesis.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
