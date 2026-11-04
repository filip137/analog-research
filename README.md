# Energy-Based Learning Framework

This repository simulates and trains dissipative resistive networks with
coordinate-descent equilibrium solvers. It builds on the original
[energy-based-learning framework](https://github.com/rain-neuromorphics/energy-based-learning).
Network equations and solvers live in `networks/`, physical state and laws in
`hardware/`, reusable learning procedures in `methods/`, and execution in `ebl/`.
See the [architecture and ownership guide](docs/architecture.md).

The supported experiment entrypoint is:

```bash
python -m ebl describe --experiment small_drn.v1
python -m ebl train \
  --config examples/small_drn/base.json \
  --output-dir runs
```

Training, linspace analysis, validation, and legacy checkpoint import are
separate commands. Evaluation commands always require an explicit named
weights artifact; they never search for the “latest” checkpoint.

```bash
python -m ebl linspace \
  --config examples/small_drn/base.json \
  --weights runs/<run-id>/checkpoints/weights.pt \
  --output-dir runs

python -m ebl validate \
  --config examples/small_drn/base.json \
  --weights runs/<run-id>/checkpoints/weights.pt \
  --output-dir runs
```

Every train, linspace, or validate invocation owns a new exclusive run
directory. It starts with a resolved config, source/runtime manifest, and
status, then records command-specific metrics, results, artifacts, logs, or
checkpoints as the work proceeds. See
[`docs/experiment_runtime.md`](docs/experiment_runtime.md) for the extension
boundaries, checkpoint rules, worktree workflow, and campaign protocol.
Research follows the [question-led workflow](workflow/policy.md):
small pilots use one evolving note, while sustained comparisons live in
[scientific campaigns](campaigns/README.md). Native bundles retain exact
execution identities; campaign result notes record observations, interpretation,
limitations and the next decision.
The cohort-A measured ReRAM projection and bounded-learning-rate workflow is
documented in
[cohort-A training record](campaigns/measured-cohort-adaptation/series/001-transfer-and-write-policies/experiments/exp-001-cohort-a.md#source-measured_cohort_a_training-measured-cohort-a-reram-training),
with held-out cohort-B deployment and fine-tuning in
[cohort-B transfer record](campaigns/measured-cohort-adaptation/series/001-transfer-and-write-policies/experiments/exp-002-cohort-b-transfer.md#source-measured_cohort_b_finetuning-measured-cohort-b-deployment-and-fine-tuning),
and physical low-rank recovery in
[measured low-rank recovery record](campaigns/measured-cohort-adaptation/series/002-low-rank-recovery/experiments/exp-005-cohort-b-lora.md#source-measured_cohort_b_lora_recovery-measured-cohort-b-memristor-lora-recovery).
The matched single-pulse and Tiki-Taka continuation of a saved RESET-relative
IBM OM deployment is specified in
[matched pulse-recovery experiment](campaigns/ibm-om-deployment-recovery/series/002-reset-relative/experiments/exp-005-pulse-recovery.md#source-ibm_om_reset_relative_on_chip_recovery-reset-relative-deployed-array-pulse-recovery).
The fixed p90/p95/p99 follow-up for direct deployed-array pulses is specified
in [gradient-threshold experiment](campaigns/ibm-om-deployment-recovery/series/002-reset-relative/experiments/exp-006-gradient-thresholds.md#source-ibm_om_reset_relative_direct_gradient_thresholds-reset-relative-direct-gradient-thresholding).
The IBM OM device integration (assignment, topology, characterization,
mapping, programming and readback layers, each selected from the config) is
described in [`docs/ibm_om_layers.md`](docs/ibm_om_layers.md).

## LoRA/HWA research records

The [five scientific campaigns](campaigns/README.md) organize initialization and
representation, measured-cohort adaptation, endpoint-model HWA/recovery, IBM OM
programming, and IBM OM deployment/recovery. Start there for scientific conclusions
and links to the underlying evidence.

Raw outputs retain their existing ignored [results paths](results/README.md).
The [documentation index](docs/README.md) separates shared contracts and implementation
references from migrated study narratives and frozen historical indexes. The
human-owned [current-state synthesis](docs/current_state.md) is preserved.

Shared drift/calibration/recovery ownership and its byte-identical HWA mirror
contract are documented in [shared native architecture](docs/shared-native-architecture.md).
