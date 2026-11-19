# Pilot: deployed quality by mapping, fault rate and HWA (crossbar half, holds only)

State: implementing. Branch: `exp/20261119-mapping-faults`. Authorization: user, 19 November 2026,
on-chip-learning issue [#33](https://github.com/filip137/on-chip-learning/issues/33); crossbar share
at most 2.5 V100 GPU-h (of 4 GPU-h for both backends) including canaries and retries, Jean Zay V100
dev QoS (`umg@v100`, `gpu_p13`, `qos_gpu-dev`, `v100-32g`).

Owning (shared) note, which holds every scientific choice: on-chip-learning
`campaigns/fault-recovery-paper/series/003-mapping-deployment/experiments/exp-001-mapping-faults-hwa.md`.
This note transcribes it to lifecycles and records execution; it chooses nothing scientific.

## Question and decision

Which weight-to-conductance mapping gives the best deployed quality under permanent faults, with
and without HWA? Informs the mapping of the #33 rerun under drift-recovery conditions. Decision rule
(owning note): within each model, prefer the mapping with the lowest fault-aware WikiText-test KL at
3 % and 5 % whose 0 % KL is no more than 10 % above base; every cell is reported; the user decides.

## Frozen comparison (transcribed)

32 cells = XB6, XB12 × source {HWA, no-HWA} × mapping {base, two-sided} × rate {0, 1, 3, 5 %}.
Every cell is the paper-grid map-271002 lifecycle of its depth and rate
(`lifecycles/paper-grid-20261116/paper/fault-paper-grid-20261116-<model>-r<rate>-271002.json`)
with only these fields changed (generator `campaigns/fault-recovery-paper/make_mapping_faults.py`):

| Field | Value |
| --- | --- |
| `fault_tuning.arms` | `naive_hold` (`hold_naive`), `fault_aware_hold` (`hold_fault_aware`); no training |
| `fault_tuning.analog_lora` | `null` (no adapter tiles) |
| `fault_tuning.mapping` | base: `reference` (G_act = 48.6 + 39.6 q, G_ref = 48.6 µS); two-sided: `two_sided`, unit gain, margin 0 (G± = 9 + 79.2·max(±q, 0)) |
| `hwa.arms` | HWA: `standard_hwa_model` (sources XB6 std3 `86522e23…`, XB12 `ed53b6cb…`); no-HWA: `{"id": "pretrained", "method": "none"}` with the pinned pretrained-teacher source below |

Unchanged from the paper grid: CMO 2 % Gaussian-endpoint P&V (one draw per device, unclamped),
window 9–88.2 µS, random stuck devices uniform in [9, 88.2] µS, exactly rate × devices per MLP,
assignment/fault map 271002 (endpoint seed 281002), verify-based pair rewrite at 5σ, seed 241016,
partner targets clamped to 9–88.2 µS; development 64 WikiText-2 validation sequences `7411cf50…`,
held-out WikiText-2 test 1,024 `5b51c74a…` and generated 384 `bd00888a…`, all KL against the FP32
teacher. The recovery pool, teacher-sample file, schedule, Fisher and LoRA seed stay pinned but are
unused by holds.

Controls: 0 % twin of every cell; base mapping replays the paper grid (canary below).

## What changed (code, commit `6356522`)

- `methods/fault_tuning.py`: optional `fault_tuning.mapping` (omitted = standard pair, digests of
  every earlier lifecycle unchanged), parsed by `methods.drift_tuning.parse_mapping`;
  `deployment_mapping()` returns it or `drift.mapping`; both at once is rejected; without drift a
  non-reference mapping admits only the two holds and no analog-LoRA section.
- `ebl/runtime/fault_tuning.py`: `Context` takes the mapping from `deployment_mapping()`; new
  `read(ctx, array)` reads through the mapping (base: `array.read()`, bit for bit) and replaces the
  base readout in `run_arm` (initial/held state, writer reads, curve, calibration/LoRA invariance
  checks, final hash) and in `evaluate_held_out` — previously these read (G0 − G1)/39.6, which
  doubles two-sided weights. `device_statistics` / `deployment_records` (written to `summary.json`
  as `deployment_records` and to `metrics.jsonl` when `fault_tuning.mapping` is declared): mapped
  target floor/max fractions; per state (naive, fault-aware) stuck, below-9 µS and above-88.2 µS
  healthy-device fractions, apparent residual RMS/max vs the target; rewrite controller counts
  (flagged, programmed, clipped low/high), analysis-side false positives/negatives and cost; source
  ideal dev KL, clean mapped-readout dev KL.
- `methods/programming/lifecycle.py`: `fault_rewrite` reports `clipped_targets_low/high`
  (numerics unchanged).
- `campaigns/fault-recovery-paper/launch/worker.py`: binds `--hwa-source` to the lifecycle's single
  HWA arm (was hard-coded `standard_hwa_model`) and checks its `source_sha256` against the manifest.
- Tests `tests/test_fault_tuning_mapping.py`: base mapping reproduces the standard pair bit for bit
  (states, hashes, controller report, hold rows); two-sided at 0 % reads the targets within
  programming error (exactly without noise) and never 2q, in development and held-out scoring;
  device statistics; schema accepts mapping + faults hold-only and rejects writers/adapters/drift
  combinations.

## Inputs

- Pretrained (no-HWA) sources: lifecycles `mapping-faults-20261119-{xb6,xb12}-pretrained-source`
  (paper-grid r00 sections; only `prepare`, `devices`, `hwa__pretrained` run, local CPU, nom-cool-2,
  bundles inspected complete/valid): XB6 `3442d7d2b4137c4126b2501a54d27a594b1ccbddc171b39d7329e2e4fd73a1b4`,
  XB12 `480e1323aaed8a12036121d887f1aa38c3316b7cb39354ca14353cfc76c3abc8`; copies in
  `results/mapping-faults-20261119/sources/pinned/` and on Jean Zay under
  `$SCRATCH-root/mapping-faults-20261119/inputs/`. The stage's clean development KL is 1.1e-10.
- All other pins: as the paper grid (std3 `86522e23…`, L12 `ed53b6cb…`, teacher `2d74da66…`, corpora,
  teacher-sample file).

## Budget and execution

Lifecycles: `campaigns/fault-recovery-paper/lifecycles/mapping-faults-20261119/mapping-faults-20261119-<model>-<hwa|nohwa>-<base|two>-r<00|01|03|05>.json`
(32; `describe`: stages prepare, devices, hwa (frozen source), fault_tuning; 28.3M / 56.6M analog weights).
Launch: native Jean Zay path of the paper grid / CMO exp-001 (`stage_source.py`, `jz_setup_source.sh`,
`jz_prepare.sh`, `jz_submit.sh`, `run.slurm` + `worker.py`), one manifest per model × source, several
units per job joined by `+`. Expected per cell (paper-grid stage times on V100): prepare ~30 s,
devices ~25 s, deployment 12 s (XB6) / 23 s (XB12), each hold 34 s / 45 s with both held-out cohorts:
~3 min (XB6), ~4 min (XB12) → ~1.9 GPU-h for 32 cells including job start-up. Cap 2.5 V100 GPU-h.
Canary first (one job): XB6 HWA base 3 % must reproduce paper-grid map 271002 FA hold
(WikiText test 0.05494022420516201, dev 0.05614616473515829; naive WikiText 0.18999810510990667);
XB6 HWA two-sided 0 % FA dev KL should lie near its clean mapped readout (≈ 0.043 per #66).
Raw outputs on Jean Zay `/lustre/fsn1/projects/rech/umg/ucy17uy/mapping-faults-20261119/runs-v1/`;
compact collection in `results/mapping-faults-20261119/`.

## Current execution and monitoring handoff

19 Nov: code committed; pretrained sources being produced locally (CPU). Nothing launched on Jean Zay.
