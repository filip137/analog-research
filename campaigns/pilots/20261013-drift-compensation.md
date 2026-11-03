# Pilot: digital compensation of the saved HWA retention arrays

State: complete; execution, collection and interpretation passed. The user assigned this follow-up after the
[retention comparison](20261013-aihwkit-retention.md): measure recovery using
row/column rescaling, normalization shifting and other standard compensation.
This authorizes implementation and the bounded comparison below. The sources,
data, analog subset, fault regime and age grid continue the confirmed pilot.

## Question and decision

How much loss from aging can digital calibration recover with zero device
writes? Compare every predeclared method with its own uncompensated aged array
and original programmed baseline. Teacher KL remains primary; report perplexity
and calibration cost. This is a descriptive single-assignment pilot, not an
on-chip learning or physical-viability claim. Expected: scaling substantially
reduces PCM loss; an alternative is that heterogeneous drift leaves a large
irreducible loss after simple gains.

For each source/model/fault case/age/cohort, define

`aging KL regained (%) = 100 × (aged KL − compensated KL) / (aged KL − programmed KL)`.

Report this only when the denominator exceeds 1e-6 nats/token. Otherwise leave
it undefined and report the absolute KL change. Do not clip negative values or
values above 100%. No held-out method selection, averaging across fault rates,
or best-seed selection. All methods remain in the results, including failures
to improve accuracy. Source clean accuracy is not the recovery target: the
programmed reference retains its original programming error and permanent faults.

## Frozen comparison and methods

Reuse the original native retention bundles and exact source hashes from the
linked pilot: std3, std9 and cdt3; OPT-125M's final six fc1/fc2 MLP pairs;
28,311,552 logical weights and 56,623,104 physical devices. Assignment 271002,
healthy and 3% permanent random faults; PCM, CMO ReRAM and Wan ReRAM; 1 s, 1 day
and 2 days. Read noise remains off. Wan's 1 s realization is its baseline and
therefore all compensators are identity at that point. This gives **378 method
rows**, including 42 identity rows, for **336 nonidentity evaluations**.

Deployment references are simulated observable readouts of the saved programmed
array, reconstructed exactly. This assumes those reference measurements could
have been recorded at deployment; the previous experiment did not actually
perform a calibration protocol. Controllers receive MVM readouts and activation
statistics, never hidden drift coefficients or ideal clean targets.

Physical rows mean **inputs**, and physical columns mean **outputs**. A logical
matrix is stored [output, input] in PyTorch. All correction operators are digital:
input gains precede MVM, output gains follow MVM, per-tile gains precede summation,
and activation shifts follow the frozen MLP bias. Physical conductances remain
unchanged, including permanent faults.

| Method | Fixed rule | Digital parameters (six MLPs) |
| --- | --- | ---: |
| `global` | One ratio of baseline/current mean absolute one-hot readout per MLP matrix | 12 |
| `tile_global` | Same rule per 256×256 logical tile, before tile sums | 432 |
| `row` | Mean-absolute response ratio per input channel of each MLP matrix | 23,040 |
| `column` | Mean-absolute response ratio per output channel of each MLP matrix | 23,040 |
| `row_column` | Ten alternating diagonal least-squares updates against baseline one-hot responses | 46,080 |
| `global_shift` | Global gain, then per-output mean shift to the baseline activation mean | 23,052 |
| `global_moments` | Global gain, then per-output affine correction matching baseline mean and standard deviation | 46,092 |

Global and column readout rules follow the pinned AIHWKit 1.1.0
[drift compensation implementation](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/compensation/drift.py).
The logical-matrix and physical-tile scopes are reported separately. Input-row
and combined row/column fitting are explicit extensions, not claims of built-in
AIHWKit classes. Mean/variance correction follows the activation-statistics idea
in [Joshi et al. 2020](https://www.nature.com/articles/s41467-020-16108-9).
OPT uses LayerNorm and has no BatchNorm running statistics to refresh: these
arms perform affine output correction, and neither replace LayerNorm nor tune
its learned parameters. This is a deliberate architecture-appropriate analogue,
not a reproduction of CNN adaptive BatchNorm.

All gains are bounded to [0.1, 10]; mean-absolute readouts use AIHWKit's 1e-4
floor. Degenerate variance channels use unit gain. These fixed numerical rules
are not tuned on evaluation data. Moment corrections are simultaneous,
one-pass statistics matching after global compensation; there is no iterative
layerwise refinement or gradient optimization. HWA recipes remain frozen;
the follow-up tests digital drift compensation, which the preceding pilot
explicitly excluded relative to the Rasch 2023 evaluation convention.

## Data, calibration cost and controls

Keep sequence length 256 and the same development (256 synthetic sequences),
WikiText test (1,024) and generated-text test (384) cohorts, teacher and tokenizer
hashes. Moment calibration uses only the **first 128 sequences of the existing
training/recovery cohort**, 32,768 input tokens, from the pinned synthetic corpus.
The train split is separate from all evaluation splits. Calibration may overlap
the source's HWA training data, as permitted for this no-weight-write control;
it never uses held-out tokens or labels. Record token hashes and cohort receipts.

Scale-only methods use one-hot probes without text data. Both moment methods
share the same available data and paired baseline/global-aged statistics. Readouts
and statistics are reusable across arms, but each row reports the measurements
required by that method on its own; do not sum shared acquisition across methods
as though independent hardware measurements were made.

At six MLPs a complete one-hot acquisition uses 110,592 logical-tile MVM reads
(221,184 physical-bank reads), once at deployment and once at each age. A moment
acquisition additionally uses 128×256×432 logical-tile MVM reads at deployment
and at each age. These are model-based vector-read counts, not measured hardware
energy/latency or P&V verify reads. Retention writes and P&V verifies stay zero.
Report digital parameter counts and required stored reference bytes. Combined
row/column least squares retains a full baseline readout matrix; simple gains
need only summaries. With per-MVM noise disabled, the one-hot response equals
the held normalized apparent matrix, transposed. The simulator uses this exact
algebraic identity without accessing persistent device state or drift exponents.

The native stage reloads the parent state, checks apparent hashes at every age,
reproduces one uncompensated complete-cohort control per source exactly, and
checks source/model and held-state invariance after compensation. Serialization
replay checks cover all calibration parameters. There is no new device assignment,
programming draw, fault realization or retention noise draw.

## Budget, artifacts and execution handoff

Cap: **one GPU-hour** on one idle authorized RTX 5090, **5 GB new artifacts**,
including readiness and any understood operational retry. Original retention
states are referenced in place and never duplicated or modified. The initial projection was 45–55 minutes, revised to 30–40 after readiness
and shared-teacher evaluation. Measured readiness plus main execution took
31 min 43.16 s; no reduction of scientific coverage was needed.

Output: `results/lifecycles/opt125m-drift-compensation-pilot/`. The strict native
lifecycles are `campaigns/opt-cmo-corruption/lifecycles/opt125m-drift-compensation-{std3,std9,cdt3,probe}.json`.
The readiness lifecycle reuses the prior one-MLP probe arrays and eight-sequence
cohorts, with eight calibration sequences; all seven methods are exercised.
It is excluded from scientific results.

Extension: optional `drift_compensation` lifecycle section, native `compensate`
stage, explicit `workflow plan --retention-source ARM:SEED=retention_states.json`,
and compact digital-state outputs. Existing source and parent-state provenance
gates remain in force. Use public plan/run/inspect/collect commands. One executing
Codex session owned monitoring through collection and interpretation. Readiness completed with exit 0 in tmux `hwa-compensation-probe`: 126 rows,
18.67 s wall time, 7.49 s native compensation-stage time, 2,171,904,000 bytes
peak GPU allocation, 13.31 MB new outputs, and exact original-control replay.
All three readiness bundles passed native artifact inspection. The complete
comparison finished in `hwa-compensation-main` with exit 0 after 1,884.49 s,
within its 3,500-second process-group deadline. Peak GPU allocation was
4,070,872,064 bytes. All nine main native bundles passed artifact inspection.
The three compensation stages took 614.33, 615.74 and 614.94 s. Completion:
2026-10-13 22:19:49 UTC (2026-10-14 00:19:49 Paris). The GPU was released.
No failed run, retry, exclusion or replacement occurred; attempt-002 follows
the launcher's dry-run attempt. Setup/reading/validation time and token use
were not separately measured.

Execution root: `/home/filip/analog-research/opt-cmo-exp1/hwa-compensation-20261013/`.
Frozen revision: `1613de7e4ebf23ecbd3cc961650da33202802fd7`, derived from the
original retention execution snapshot. CUDA environment: PyTorch 2.11.0+cu128
on loulou, `/home/filip/miniconda3/envs/py312/bin/python`. The GPU was idle
before both launches. The output root contains `inputs.json`, the 27-file
`source-receipt.json`, `launch-{probe,main}.json`, phase logs and exit/coverage
receipts. Original retention states remain in their unchanged prior tree.
The monitoring handoff is closed. New native results occupy 1.475 GB; the
execution tree is 1.969 GB. Local collection/plots and both collection archives
bring new storage to approximately 2.25 GB, below the 5 GB cap.

Local evidence: [`results/lifecycles/opt125m-drift-compensation-pilot/compact/`](../../results/lifecycles/opt125m-drift-compensation-pilot/compact/).
All **713 copied files** passed local SHA-256 verification, including the small
calibration checkpoints. The collection archive SHA-256 is
`2cd16b031eff49c884a069394cc3c134fb0929cbcb31355294ecfd35f502a2f7`.
Large feature caches remain remote and are listed in the compact receipt.
All original device states remain in the preceding retention tree.

## Evidence and interpretation

CPU validation passed **523 tests, 28 optional
skips** across the full workflow/retention/compensation suite. The focused
compensation tests passed all **15** with AIHWKit 1.1.0 available. Exact equality
of batched shared-teacher evaluation with separate method evaluation was
checked; only teacher computation is reused. `git diff --check` passed. Tests cover known global/input/output/tile distortions,
moment correction, fixed device states, upstream AIHWKit parity, strict config
and parent validation, native replay, exact controls and collection.


### Coverage and integrity

Coverage is complete: **378 method records** for three sources × three device
models × two fault cases × three ages × seven methods. The 42 Wan baseline/1 s
identity rows reuse their original metrics; 336 nonidentity compensated
snapshots were evaluated. The 126-row readiness sweep is excluded.

The audit matched every apparent-state hash, uncompensated metric and fault
identity to its original native parent. All three complete-cohort control
replays were exactly equal to the earlier evaluation. Frozen source parameters
and apparent states remained unchanged. All device-write and P&V-verify counts
are zero. Calibration probes still incur the separately reported MVM-read cost.
The same 128 training sequences were used for every moment-calibrated array;
the calibration token SHA-256 is
`a47861b33b965ac674e7d7876290386304462e6ec0ec1ab05d804c65b6d0475e`.

### PCM recovery at two days

WikiText aging teacher-KL loss regained, in percent. Healthy and faulty arrays
remain separate; each entry is one source on assignment 271002.

| Method | std3 healthy | std9 healthy | cdt3 healthy | std3 3% faults | std9 3% faults | cdt3 3% faults |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global gain | 99.19 | 99.56 | 99.78 | 78.06 | 90.42 | 96.74 |
| Tile gains | 99.19 | 99.56 | 99.78 | 78.07 | 90.42 | 96.74 |
| Input-row gains | 99.19 | 99.56 | 99.78 | 78.48 | 90.54 | 96.78 |
| Output-column gains | 99.17 | 99.55 | 99.77 | 78.95 | 90.65 | 96.81 |
| Row + column LS | 99.16 | 99.48 | 99.87 | 80.60 | 90.13 | 96.02 |
| Global + mean shift | 99.11 | 99.52 | 99.77 | 80.15 | 90.28 | 96.58 |
| Global + mean/variance | 99.08 | 99.43 | 99.81 | 81.57 | 90.78 | 97.08 |

Corresponding WikiText perplexities (lower is better):

| Source | Fault case | Programmed | Aged, no correction | Global gain | Row + column LS | Global + mean/variance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| std3 | healthy | 49.53 | 84.04 | 49.73 | 49.57 | 49.69 |
| std9 | healthy | 51.18 | 91.48 | 51.30 | 51.19 | 51.27 |
| cdt3 | healthy | 53.86 | 130.67 | 53.92 | 53.69 | 53.84 |
| std3 | 3% random | 53.07 | 91.33 | 60.78 | 58.95 | 58.50 |
| std9 | 3% random | 52.66 | 97.28 | 55.72 | 55.45 | 55.53 |
| cdt3 | 3% random | 54.13 | 140.73 | 55.37 | 55.35 | 55.43 |

Global scaling recovers **99.19–99.78%** of healthy-array aging KL loss on
WikiText. With permanent faults it recovers **78.06%, 90.42% and 96.74%** for
std3, std9 and cdt3 respectively. Global plus moment correction changes those
faulty-array figures to **81.57%, 90.78% and 97.08%**. Combined row/column LS
helps std3's KL relative to global alone, but worsens that metric for std9 and
cdt3 despite slightly lower perplexity. There is no uniform ordering across
metrics. Tile gains offer essentially the same recovery as global gains here.

The generated-text cohort confirms the main effect: global scaling recovers
99.15–99.80% of healthy PCM aging KL loss, and 79.55%, 91.76% and 97.52% on
the three faulty arrays. Moment correction yields 83.78%, 92.45% and 97.88%
on those faulty generated-text cases. These ranges describe the three frozen
sources, not independent array replicates or confidence intervals.

### CMO and Wan results

Their aging gaps are much smaller and calibration is not uniformly beneficial.
For faulty CMO arrays, row/column LS recovers 38.64%, 16.99% and 20.48% of the
WikiText aging KL gap for std3/std9/cdt3, but these correspond to small absolute
changes. For example, std3 goes from 0.19776 uncompensated to 0.19516, against
a programmed 0.19103. Healthy cdt3 CMO worsens under every tested method on
WikiText KL; its original aging gap was only about 0.00072. Its -205.8% result
under moment correction is therefore a small absolute worsening, not a
catastrophic accuracy collapse.

Wan shows source- and metric-dependent effects. Healthy std3 improves from
KL 0.07656 to 0.07476 with global plus moments, while global gain alone
slightly worsens it. Row/column LS worsens faulty cdt3 Wan KL from 0.14292 to
0.14642 while improving perplexity from 54.61 to 54.44. No single tested
compensator consistently removes Wan's stochastic endpoint error.

### Cost and artifacts

Global gain uses **12 digital scalars**, 48 bytes of stored reference summaries,
and no text calibration data. Its complete one-hot calibration requires
110,592 logical-tile MVM reads (221,184 physical-bank reads) per age, plus the
same acquisition at deployment. This run does not optimize that probe schedule.
Row/column LS uses 46,080 digital parameters and a 113.25 MB full baseline
readout reference. Moment correction uses 46,092 parameters, 32,768 unlabeled
training tokens, and 14,266,368 logical-tile MVM reads at each acquisition,
including one-hot probes. These are simulated acquisition requirements, not
measured energy or hardware latency.

- [Full 378-row CSV](../../results/lifecycles/opt125m-drift-compensation-pilot/compact/collected-main/compensation.csv),
  [two-day results](../../results/lifecycles/opt125m-drift-compensation-pilot/analysis/two_day_recovery.csv)
  and [calibration cost](../../results/lifecycles/opt125m-drift-compensation-pilot/analysis/calibration_cost.csv).
- [WikiText recovery fractions](../../results/lifecycles/opt125m-drift-compensation-pilot/analysis/transfer_wikitext_aging_kl_regained_percent.png)
  and [absolute KL](../../results/lifecycles/opt125m-drift-compensation-pilot/analysis/transfer_wikitext_teacher_kl.png).
- [Generated-text recovery fractions](../../results/lifecycles/opt125m-drift-compensation-pilot/analysis/transfer_generated_aging_kl_regained_percent.png)
  and [absolute KL](../../results/lifecycles/opt125m-drift-compensation-pilot/analysis/transfer_generated_teacher_kl.png).
- [PCM perplexity at all ages](../../results/lifecycles/opt125m-drift-compensation-pilot/analysis/pcm_perplexity_over_time.png).
  Matching PDFs are alongside the PNGs. Plots were visually checked.

Regenerate the figures and tables from collected evidence:

```bash
/home/filip/miniconda3/envs/py312/bin/python campaigns/opt-cmo-corruption/launch/compensation_report.py \
  results/lifecycles/opt125m-drift-compensation-pilot/compact/collected-main \
  --output results/lifecycles/opt125m-drift-compensation-pilot/analysis
```

### Scoped verdict and next decision

The requested compensation comparison is complete. Simple global scaling is a
strong zero-device-write control for PCM aging in this pilot. Finer row/column
or activation calibration offers modest, nonuniform additional benefits with
permanent faults. It does not remove the array's original programming or fault
error. CMO/Wan compensation must be justified separately; these results do not
support applying one universal correction across technologies.

This is one synthetic assignment, using CMO-trained sources remapped to the
three fitted models. No claim generalizes to fabricated hardware, other arrays,
longer ages or enabled read noise. OPT LayerNorm parameters stayed frozen;
the normalization arm is affine output-statistics correction, not literal
BatchNorm recalibration. Periphery quantization, ADC/DAC clipping and input-gain
headroom limits are not modelled. The preserved output-gain/bias operators
are digital, and all physical devices retain their original state at each age.

A follow-up on on-chip drift recovery should include global calibration as a
predeclared matched control, then validate any extra benefit across independent
assignments. No further run is needed to close this pilot and none is running.

### User direction: focus on healthy devices (2026-10-14)

The user narrowed the current drift investigation to healthy deployed arrays:
zero permanent faulty active or reference devices. Future drift comparisons
should use the nominal case. The completed healthy/faulty comparison above
remains historical evidence; its frozen lifecycles and bundles retain their
original coverage. This direction concerns deployed devices and does not by
itself change the HWA source checkpoints or their training histories.

The question is now how much aging loss remains on healthy arrays after digital
calibration, and whether that residual warrants further adaptation. Retain each
array's own programmed baseline, the same held-out cohorts and the declared
calibration controls. Assess absolute KL and perplexity changes alongside
recovery percentages; a large unrecovered fraction of a tiny aging gap is not
a large practical loss.

Existing two-day WikiText teacher KL for the standard 3% HWA source, healthy
assignment 271002 (lower is better):

| Model | Programmed baseline | Aged, no correction | Global gain | Row + column LS |
| --- | ---: | ---: | ---: | ---: |
| PCM | 0.04627 | 0.69815 | 0.05154 | 0.05175 |
| CMO | 0.04428 | 0.04987 | 0.04985 | 0.04948 |
| Wan | 0.05184 | 0.07656 | 0.07729 | 0.07867 |

Healthy PCM's large aging penalty is almost entirely removed by global gain
at the tested ages. Healthy CMO has a small absolute residual. Healthy Wan has
a larger residual than CMO in this source, and none of the tested scaling rules
improves its two-day teacher KL; row/column LS does partially improve perplexity
and next-token accuracy. Wan remains a time-dependent endpoint-error model,
not a continuous physical drift trajectory. These observations are specific
to the saved arrays; they do not establish that healthy-array drift requires
weight learning. This scope decision launches no additional experiment.
