# Pilot: deployed quality by mapping, fault rate and HWA (crossbar half, holds only)

State: executed and collected; awaiting the user's mapping decision. Branch: `exp/20261119-mapping-faults`. Authorization: user, 19 November 2026,
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

Owner: the executing agent (19 Nov). Launch commit `9a98ce4` (main, pushed); Jean Zay source
`/lustre/fswork/projects/rech/umg/ucy17uy/mapping-faults-20261119/source-9a98ce4/hwa` (receipt
`18367268…`, 2,542 files, 0 mismatches); launch dir `…/mapping-faults-20261119/launch-9a98ce4/`
(manifests `manifest-{xb6,xb12}-{hwa,nohwa}.json`; worker = tracked copy); outputs
`/lustre/fsn1/projects/rech/umg/ucy17uy/mapping-faults-20261119/runs-v1/<model>-<source>/<unit>/`
(`complete.json` / `failed-<job>.json`). All 32 units planned and dry-run on the login node.

| Job | Units | State |
| --- | --- | --- |
| 436970 (canary, 30 min cap) | xb6-hwa-base-r03, xb6-hwa-two-r00 | COMPLETED 6 min 09 s |
| 437159 | xb6-hwa base r00/r01/r05, two r01/r03/r05 | COMPLETED 17 min 32 s |
| 437161 | xb6-nohwa (8) | COMPLETED 21 min 50 s |
| 437164 | xb12-hwa (8) | COMPLETED 29 min 15 s |
| 437167 | xb12-nohwa (8) | COMPLETED 28 min 33 s |

All jobs finished; no retries. GPU time 103 min 19 s = 1.72 V100 GPU-h (cap 2.5). Monitoring ended.

Canary (passed): XB6 HWA base 3 % replays paper-grid map 271002 bit for bit — naive and
fault-aware apparent hashes, controller report (incl. plan hash), dev, WikiText-test and generated
KL identical (FA WikiText 0.05494022420516201, naive 0.18999810510990667). XB6 HWA two-sided 0 %:
FA dev KL 0.04318 vs clean mapped readout 0.04297 (= ideal 0.04297); 32 devices flagged (all false
positives, Gaussian tails), 8 partner targets clipped low.

Collection: `rsync` each `complete.json` run (without checkpoints) to
`results/mapping-faults-20261119/bundles/<unit>/`, `python -m ebl runs inspect <bundle>
--verify-artifacts --require-complete` on Jean Zay, then
`python campaigns/fault-recovery-paper/collect_mapping_faults.py results/mapping-faults-20261119/collected …/table.csv …/table.json`.

## Evidence and coverage

32/32 units `complete.json`; every fault_tuning bundle `python -m ebl runs inspect --verify-artifacts
--require-complete`: complete, valid (Jean Zay, list in `results/mapping-faults-20261119/inspect.txt`);
worker validation passed (holds: zero updates, both held-out cohorts with exact identities and
counts, unchanged state). All on Tesla V100-SXM2-32GB. Bundles without checkpoints:
`results/mapping-faults-20261119/bundles/<unit>/`; table `results/mapping-faults-20261119/table.{csv,json}`.
All 8 base-mapping HWA cells replay the paper-grid map-271002 cells bit for bit (naive and FA
apparent hashes, dev and both held-out KL).

KL against the FP32 teacher (dev 64, WikiText test 1,024, generated 384). Healthy < 9 µS: fraction of
all devices, naive state. Stuck fractions equal the rate exactly. Ideal / clean-mapped dev KL: XB6
HWA 0.0430 / 0.0430, XB12 HWA 0.2041 / 0.2041, no-HWA 0 / 0 (both mappings).

| Model | Source | Mapping | % | Naive dev / WT / gen | FA dev / WT / gen | Healthy < 9 µS | Flagged | Rewritten | Clipped low / high |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| XB6 | HWA | base | 0 | 0.0453 / 0.0439 / 0.0198 | 0.0453 / 0.0439 / 0.0198 | 0.001 | 32 | 32 | 0 / 0 |
| XB6 | HWA | base | 1 | 0.0888 / 0.0875 / 0.0549 | 0.0476 / 0.0460 / 0.0221 | 0.001 | 526,875 | 521,977 | 53,853 / 53,475 |
| XB6 | HWA | base | 3 | 0.1910 / 0.1900 / 0.1332 | 0.0561 / 0.0549 / 0.0287 | 0.001 | 1,580,435 | 1,536,047 | 157,868 / 157,824 |
| XB6 | HWA | base | 5 | 0.3134 / 0.3117 / 0.2285 | 0.0656 / 0.0653 / 0.0372 | 0.001 | 2,633,806 | 2,510,834 | 258,379 / 258,595 |
| XB6 | HWA | two-sided | 0 | 0.0432 / 0.0418 / 0.0183 | 0.0432 / 0.0418 / 0.0183 | 0.251 | 32 | 32 | 8 / 0 |
| XB6 | HWA | two-sided | 1 | 0.0695 / 0.0683 / 0.0405 | 0.0472 / 0.0458 / 0.0223 | 0.248 | 548,951 | 543,643 | 99,652 / 107,762 |
| XB6 | HWA | two-sided | 3 | 0.1272 / 0.1254 / 0.0843 | 0.0581 / 0.0571 / 0.0313 | 0.243 | 1,646,454 | 1,598,428 | 293,583 / 316,905 |
| XB6 | HWA | two-sided | 5 | 0.1956 / 0.1934 / 0.1331 | 0.0685 / 0.0687 / 0.0411 | 0.238 | 2,743,831 | 2,610,543 | 478,538 / 517,876 |
| XB6 | no-HWA | base | 0 | 0.0263 / 0.0264 / 0.0214 | 0.0263 / 0.0264 / 0.0214 | 0.000 | 32 | 32 | 0 / 0 |
| XB6 | no-HWA | base | 1 | 0.8794 / 0.8201 / 0.6577 | 0.0343 / 0.0344 / 0.0284 | 0.000 | 526,213 | 521,345 | 25,633 / 25,448 |
| XB6 | no-HWA | base | 3 | 3.1350 / 2.9619 / 2.7656 | 0.0896 / 0.0875 / 0.0689 | 0.000 | 1,578,742 | 1,534,446 | 75,146 / 74,898 |
| XB6 | no-HWA | base | 5 | 4.2563 / 4.0564 / 4.1254 | 0.1576 / 0.1599 / 0.1264 | 0.000 | 2,631,078 | 2,508,562 | 123,401 / 122,763 |
| XB6 | no-HWA | two-sided | 0 | 0.0007 / 0.0007 / 0.0006 | 0.0007 / 0.0007 / 0.0006 | 0.251 | 32 | 32 | 10 / 0 |
| XB6 | no-HWA | two-sided | 1 | 0.6682 / 0.6401 / 0.4758 | 0.0072 / 0.0076 / 0.0067 | 0.249 | 554,423 | 549,031 | 46,180 / 50,975 |
| XB6 | no-HWA | two-sided | 3 | 2.4852 / 2.3188 / 2.1937 | 0.0296 / 0.0295 / 0.0247 | 0.244 | 1,663,493 | 1,614,371 | 136,244 / 149,814 |
| XB6 | no-HWA | two-sided | 5 | 3.9633 / 3.7046 / 3.7485 | 0.0550 / 0.0570 / 0.0460 | 0.239 | 2,772,099 | 2,635,849 | 222,480 / 245,498 |
| XB12 | HWA | base | 0 | 0.2067 / 0.2002 / 0.0653 | 0.2067 / 0.2002 / 0.0653 | 0.001 | 63 | 63 | 0 / 2 |
| XB12 | HWA | base | 1 | 0.2666 / 0.2613 / 0.1067 | 0.2080 / 0.2015 / 0.0675 | 0.001 | 1,054,378 | 1,044,602 | 105,501 / 105,381 |
| XB12 | HWA | base | 3 | 0.4274 / 0.4226 / 0.2208 | 0.2192 / 0.2126 / 0.0754 | 0.001 | 3,161,860 | 3,073,532 | 310,414 / 311,183 |
| XB12 | HWA | base | 5 | 0.6662 / 0.6539 / 0.3699 | 0.2369 / 0.2313 / 0.0858 | 0.001 | 5,268,813 | 5,022,747 | 506,885 / 506,729 |
| XB12 | HWA | two-sided | 0 | 0.2041 / 0.1973 / 0.0635 | 0.2041 / 0.1973 / 0.0635 | 0.251 | 63 | 63 | 16 / 0 |
| XB12 | HWA | two-sided | 1 | 0.2390 / 0.2331 / 0.0895 | 0.2083 / 0.2017 / 0.0676 | 0.248 | 1,097,776 | 1,087,262 | 195,715 / 211,810 |
| XB12 | HWA | two-sided | 3 | 0.3332 / 0.3267 / 0.1535 | 0.2211 / 0.2138 / 0.0781 | 0.243 | 3,292,458 | 3,196,740 | 575,234 / 624,243 |
| XB12 | HWA | two-sided | 5 | 0.4519 / 0.4420 / 0.2214 | 0.2416 / 0.2350 / 0.0903 | 0.238 | 5,487,590 | 5,220,514 | 938,535 / 1,017,781 |
| XB12 | no-HWA | base | 0 | 0.0601 / 0.0583 / 0.0428 | 0.0600 / 0.0583 / 0.0428 | 0.000 | 63 | 63 | 0 / 0 |
| XB12 | no-HWA | base | 1 | 3.7313 / 3.5508 / 3.7350 | 0.0767 / 0.0745 / 0.0587 | 0.000 | 1,053,097 | 1,043,359 | 51,417 / 51,505 |
| XB12 | no-HWA | base | 3 | 4.7562 / 4.5796 / 5.0150 | 0.3912 / 0.4150 / 0.3634 | 0.000 | 3,158,269 | 3,070,127 | 150,758 / 151,303 |
| XB12 | no-HWA | base | 5 | 4.8397 / 4.6536 / 5.1604 | 4.3683 / 4.1621 / 4.3832 | 0.000 | 5,263,110 | 5,017,248 | 246,927 / 246,863 |
| XB12 | no-HWA | two-sided | 0 | 0.0014 / 0.0013 / 0.0010 | 0.0014 / 0.0013 / 0.0010 | 0.251 | 63 | 63 | 18 / 0 |
| XB12 | no-HWA | two-sided | 1 | 2.4603 / 2.3505 / 2.1556 | 0.0997 / 0.1052 / 0.1562 | 0.249 | 1,108,935 | 1,098,167 | 92,871 / 102,813 |
| XB12 | no-HWA | two-sided | 3 | 5.0261 / 4.8038 / 5.1718 | 0.2468 / 0.2454 / 0.2826 | 0.244 | 3,326,455 | 3,228,843 | 272,943 / 302,285 |
| XB12 | no-HWA | two-sided | 5 | 4.8773 / 4.6808 / 5.1918 | 1.5198 / 1.4933 / 1.5841 | 0.239 | 5,543,778 | 5,270,976 | 445,562 / 493,548 |

## Interpretation and next decision

Against the frozen rule (FA WikiText test at 3 % and 5 %, 0 % within 10 % of base): HWA sources —
base is lower at 3 % and 5 % for both depths (XB6 0.0549/0.0653 vs 0.0571/0.0687; XB12
0.2126/0.2313 vs 0.2138/0.2350), two-sided being equal or better only at 0 %; no-HWA — two-sided is
lower at every rate (XB6 0.0295/0.0570 vs 0.0875/0.1599; XB12 0.2454/1.4933 vs 0.4150/4.1621).
Two-sided cuts naive (pre-rewrite) damage by ~1/3 to 1/2, but with HWA its fault-aware damage is
slightly larger than base, against the frozen expectation (HWA sources were trained for base). The
user decides the rerun's mapping. One fault map, exploratory model-based evidence.

Caveats: the CMO endpoint law is unclamped, so ~25 % of two-sided devices (half the 9 µS
floor targets) are programmed slightly below 9 µS; two-sided roughly doubles the clipped partner
targets. No-HWA XB12 5 % is catastrophic under both mappings (FA 4.16 base, 1.49 two-sided).
