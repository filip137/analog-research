---
id: "exp-001"
title: "Raw and isotonic cohort-A feasibility"
status: "imported"
hypotheses: []
---

# exp-001 — Raw and isotonic cohort-A feasibility

## Question and provenance

Raw and isotonic cohort-A feasibility. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-001-cohort-a.md).

Historical document locations (now references/forwarding pages): [measured_cohort_a_training.md](../../../../../docs/measured_cohort_a_training.md).

<a id="source-measured_cohort_a_training-measured-cohort-a-reram-training"></a>

## Historical source: Measured cohort-A ReRAM training

This exploratory path asks whether a dense MNIST DRN can train while every
conductance update is constrained by the measured devices in
`march_slope_x3_5k.hdf5`. The follow-on held-out-device protocol is documented
in [`measured_cohort_b_finetuning.md`](exp-002-cohort-b-transfer.md#source-measured_cohort_b_finetuning-measured-cohort-b-deployment-and-fine-tuning).

<a id="source-measured_cohort_a_training-device-protocol"></a>

## Device protocol

- A trace is screened out when its first resistance sample is greater than
  `30 kOhm`.
- Physical `(row, col)` cells are shuffled with NumPy seed 42 and split 50/50;
  bar/non-bar traces from one cell remain in the same cohort.
- Only cohort A is instantiated. With the current file this gives 633 traces
  from 317 cells; cohort B reserves 632 traces from 317 cells.
- Every dense-weight element receives two distinct cohort-A traces and a
  deterministic interpolation coefficient derived from its stable parameter
  key, flat index, and assignment seed 42.
- Resistance is converted to conductance in SI units (`G = 1/R`). The raw arm
  uses those samples directly. The matched control first applies a
  non-increasing PAVA fit to each source trace.
- New runs start at pulse index 0. SGD updates a persistent digital shadow,
  clamps it to `[0, 110 uS]`, and realizes the globally nearest point anywhere
  on the assigned interpolated curve. Ties use the lower pulse index.
- This is a global projection experiment, not sequential pulse programming.
  Cohort B is never read by the numerical update path.

The raw projector stores exact sorted curve values plus pulse identities on
the GPU (about 4.5 GiB for the configured network). The isotonic projector uses
batched binary search and does not materialize per-weight curves. Static curve
caches are reconstructed from the HDF5 digest and deterministic assignment;
resume checkpoints store only shadows, pulse indices, optimizer state, and
diagnostic accumulators.

<a id="source-measured_cohort_a_training-learning-rate-protocol"></a>

## Learning-rate protocol

The production learning rates are never copied from the unbounded model.
Each arm independently:

1. Measures nominal-LR-one proposal units on 32 minibatches, retrying at 64
   and 128 if the two halves differ by more than 10%.
2. Normalizes weight proposal RMS by the corresponding initial measured-weight
   RMS. The hidden-bias Q90 is normalized by the first weight RMS.
3. Starts at relative-update targets `(0.003, 0.01)`. An unsafe center is
   divided by three, up to six times.
4. Evaluates the 3x3 Cartesian grid with multipliers `(1/3, 1, 3)`. Every cell
   receives a restarted 640-minibatch canary; every clean cell receives a
   separate restarted three-epoch candidate.
5. Rejects non-finite values, a loss EMA above four times its prior minimum for
   eight consecutive post-warmup batches, or a dense-weight gradient RMS above
   100 times its first-32-batch median for eight consecutive batches.
6. Selects the minimum final validation loss using an inclusive 2% plateau,
   then higher accuracy and lower update targets as deterministic tie breaks.

Production is a fresh seed-42 restart. Training uses a deterministic
class-stratified 55,000/5,000 MNIST train/validation split. The official test
set is only read by a later explicit `ebl validate --weights ...` command.

<a id="source-measured_cohort_a_training-commands"></a>

## Commands

Raw measured arm:

```bash
python -m ebl train \
  --config examples/small_drn/measured_cohort_a_raw_mnist.json \
  --output-dir results/measured-cohort-a-raw-mnist-exploratory-20260814-v1 \
  --device-data data/march_slope_x3_5k.hdf5
```

Matched isotonic control:

```bash
python -m ebl train \
  --config examples/small_drn/measured_cohort_a_isotonic_mnist.json \
  --output-dir results/measured-cohort-a-isotonic-mnist-exploratory-20260814-v1 \
  --device-data data/march_slope_x3_5k.hdf5
```

After each command, pass that run's explicit
`checkpoints/weights.pt` to `python -m ebl validate` with the matching config.

<a id="source-measured_cohort_a_training-exploratory-feasibility-gate"></a>

## Exploratory feasibility gate

Proceed to cohort B only if raw-A completes without a numerical or LR-safety
failure, reaches at least 90% validation accuracy, remains within five
percentage points of isotonic-A, and has stable late-epoch accuracy with
bounded/interpretable shadow-projection error, clipping, and pulse jumps.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
