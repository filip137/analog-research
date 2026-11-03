---
id: "exp-005"
title: "Teacher-initialized measured DRN distillation"
status: "imported"
hypotheses: []
---

# exp-005 — Teacher-initialized measured DRN distillation

## Question and provenance

Teacher-initialized measured DRN distillation. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-005-teacher-distillation.md).

Historical document locations (now references/forwarding pages): [mnist_relu_drn_kd.md](../../../../../docs/mnist_relu_drn_kd.md).

<a id="source-mnist_relu_drn_kd-teacher-initialized-mnist-drn-distillation"></a>

## Historical source: Teacher-initialized MNIST DRN distillation

This is an exploratory protocol for testing whether measured ReRAM updates can
recover a DRN after deployment, and whether a differential pair of devices is
more useful than one device per physical edge.

<a id="source-mnist_relu_drn_kd-protocol"></a>

## Protocol

1. Train a bias-free `784 -> 50 -> 10` ReLU teacher on the deterministic MNIST
   train/validation split. The selected checkpoint is keyed by
   `teacher.dense_weight.0` and `teacher.dense_weight.1`.
2. Construct a bias-free `1568 -> 100 -> 20` DRN. Inputs and outputs use signed
   dual rails. Normalize each teacher matrix by its absolute maximum, search a
   fixed grid of non-clipping conductance-scale fractions per layer, and select
   the pair with the smallest calibration KL.
3. Fit one positive global readout gain on the calibration subset and freeze it
   for training and evaluation. The training loss is exclusively
   `KL(teacher || DRN)`; labels are used only for accuracy reporting.
4. Use `voltage_amp = 4` and `current_amp = 0.25`. The ideal arms compare one
   bounded conductance with a differential `G+ - G-` pair. In the differential
   interaction, transfer depends on `G+ - G-` and physical loading depends on
   `G+ + G-`.
5. For measured cohort A, assign virtual devices deterministically from the
   first half of formed physical cells. Every assigned device starts at RESET
   pulse zero, then receives exactly one global-nearest write. The single-device
   arm maps each normalized target affinely over that assigned virtual cell's
   calibrated range. The differential arm uses the shared reachable window of
   each independently assigned `G+`/`G-` pair, giving zero weights a common
   baseline before programming. Subsequent SGD targets accumulate in a digital
   shadow and are projected onto each assigned raw pulse-conductance trace.

The canonical configs are:

- `examples/mnist_relu/teacher.json`
- `examples/mnist_relu_drn/ideal_single.json`
- `examples/mnist_relu_drn/ideal_differential.json`
- `examples/mnist_relu_drn/measured_raw_single.json`
- `examples/mnist_relu_drn/measured_raw_differential.json`

The static campaign manifests are
`campaigns/manifests/mnist_relu_drn_kd_ideal.json` and
`campaigns/manifests/mnist_relu_drn_kd_measured.json`.

<a id="source-mnist_relu_drn_kd-learning-rate-calibration"></a>

## Learning-rate calibration

An initial gradient-scale probe targeted a 0.1% conductance-RMS update per
minibatch. With 3,438 minibatches per epoch, that setting produced excessive
cumulative drift: the ideal single-device arm moved from validation KL
`0.010683` to `0.082855` after one epoch, and its selected checkpoint remained
the initialization after 20 epochs.

A matched three-epoch sweep then compared rates 100 and 1,000 times smaller.
The 100-times-smaller vector reduced validation KL to `0.001119`, versus
`0.002002` for the 1,000-times-smaller vector. The canonical rates are therefore
`[2.1e-10, 5.7e-13]` for a single device and the gradient-equivalent doubled
rates `[4.2e-10, 1.14e-12]` for a differential pair.

<a id="source-mnist_relu_drn_kd-comparison-rule"></a>

## Comparison rule

The differential arm is called better only when its selected validation KL is
at least 5% below the matched single-device arm and neither student accuracy
nor teacher agreement is more than 0.2 percentage points worse. Use
`labs/tools/plot_mnist_relu_drn_kd_comparison.py` to create `summary.json` and
the matplotlib validation plot from completed run directories.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
