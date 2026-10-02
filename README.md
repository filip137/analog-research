# Energy-Based Learning Framework

This repository simulates and trains dissipative resistive networks with
coordinate-descent equilibrium solvers. Its core layout follows the original
[energy-based-learning framework](https://github.com/rain-neuromorphics/energy-based-learning):
model mechanics live in `model/`, reusable learning mechanics live in
`training/`, and executable research compositions live in `experiments/`.

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
New multi-run studies follow the lightweight
[`experiment workflow`](docs/experiment_workflow.md): a tracked plan freezes
the initial hypothesis and exact configs, native runs retain the existing
bundle contract, and reviewed finalization records the interpretation in the
study-local report and experimental manifest.
The cohort-A measured ReRAM projection and bounded-learning-rate workflow is
documented in
[`docs/measured_cohort_a_training.md`](docs/measured_cohort_a_training.md),
with held-out cohort-B deployment and fine-tuning in
[`docs/measured_cohort_b_finetuning.md`](docs/measured_cohort_b_finetuning.md),
and physical low-rank recovery in
[`docs/measured_cohort_b_lora_recovery.md`](docs/measured_cohort_b_lora_recovery.md).
The matched single-pulse and Tiki-Taka continuation of a saved RESET-relative
IBM OM deployment is specified in
[`docs/ibm_om_reset_relative_on_chip_recovery.md`](docs/ibm_om_reset_relative_on_chip_recovery.md).
The fixed p90/p95/p99 follow-up for direct deployed-array pulses is specified
in [`docs/ibm_om_reset_relative_direct_gradient_thresholds.md`](docs/ibm_om_reset_relative_direct_gradient_thresholds.md).
The IBM OM device integration (assignment, topology, characterization,
mapping, programming and readback layers, each selected from the config) is
described in [`docs/ibm_om_layers.md`](docs/ibm_om_layers.md).

## LoRA/HWA result tracking

New raw LoRA/HWA outputs live under [`results/`](results/README.md). The
[current-simulation ledger](docs/current_simulations.md) automatically lists
running native runs while retaining human-maintained paused and queued work.
The [experimental manifest](docs/experimental_manifest.md) gives a concise
linked interpretation for each concluded study. Full closeouts remain in the
study-local `analysis/report.md`, while the broad research synthesis lives in
[`docs/current_state.md`](docs/current_state.md).
