# Pilot: healthy HWA arrays after one year

State: complete; execution, collection and review passed. The user requested extending the existing drift/compensation
results to one year after explicitly narrowing the scope to healthy devices.
This is the same exploratory comparison with a longer age, not new training.
The experiment-loop and experiment-run-watchdog procedures apply.

## Question and decision

At 365 days, does aging of healthy deployed arrays leave a meaningful performance
loss after the seven previously declared digital compensators? Compare absolute
teacher KL and perplexity against each array's own programmed baseline, and
report the fraction of aging-induced KL loss recovered. Expected: PCM global
gain remains effective; an alternative is that heterogeneous aging leaves a
larger residual at one year. CMO may retain a small absolute gap even when the
recovered fraction is small. No threshold or method selection on held-out data.

Prior evidence: [retention](20261013-aihwkit-retention.md) and
[compensation](20261013-drift-compensation.md). Healthy PCM global gains recovered
99.19–99.78% of two-day WikiText KL loss; CMO and Wan had smaller absolute aging
gaps and mixed calibration effects. These are three sources on one assignment.

## Frozen comparison

- User-confirmed change: healthy deployed devices only; age 365 days = 31,536,000 s.
- Inherited target: OPT-125M final six MLPs, 28,311,552 logical weights and
  56,623,104 physical devices; assignment 271002; the same std3, std9 and cdt3
  checkpoints, hashes, programming seeds, retention seed 391013 and mappings.
  Healthy deployment does not change the sources' training histories.
- PCM and CMO AIHWKit 1.1.0 laws; read noise off. Wan is unsupported at one year:
  its released polynomial fits exist only at 1 s, 1 day and 2 days. No invented
  interpolation, extrapolation or replacement Wan law.
- PCM's power law and CMO's log-time law can be evaluated at one year; this is
  model extrapolation, not a one-year device measurement. Falcone et al. measure
  representative CMO relaxation up to one week and extrapolate the fit to ten
  years ([section 2.2.2](https://arxiv.org/html/2502.04524v1)).
  [Pinned IBM ReRAM source](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/reram.py),
  [PCM source](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/pcm.py).
- Reconstruct the original healthy arrays from identical frozen inputs and
  streams. Check baseline metrics/hashes and the replayed two-day apparent hash
  against their original native bundles before accepting new evidence. PCM keeps
  its per-device exponents; ages never compound. CMO snapshots remain marginal
  distributions, not a measured joint trajectory.
- Existing cohorts: 256 development sequences, 1,024 WikiText test sequences,
  384 generated test sequences, sequence length 256; teacher KL primary,
  perplexity and next-token accuracy secondary. All hashes are inherited.
- Seven controls: global, tile_global, row, column, row_column, global_shift,
  global_moments; fixed gains [0.1, 10], ten row/column LS iterations, 128
  training calibration sequences. Zero adaptation writes and P&V verifies.
  OPT normalization correction remains output-statistics matching.
- No change to HWA recipes; the consulted Rasch/Buechel choices and deviations
  in the parent notes carry forward. All methods and sources are reported.

The strict lifecycles are
`campaigns/opt-cmo-corruption/lifecycles/opt125m-healthy-year-{retention,compensation}-{std3,std9,cdt3,probe}.json`.
Only nominal is selected by retention.case_labels; source-dependent sections
retain their original provenance. Probe: existing one-MLP digital source, eight
evaluation/calibration sequences, excluded from scientific coverage.

## Budget and execution

Target: one idle RTX 5090 on loulou. The other authorized 5090s were checked
idle before admission. Expected 6–10 minutes from the completed pilots' measured
throughput; cap 20 GPU-minutes including readiness and operational retries,
12 GB new artifacts. Keep the full native states remotely and compact evidence
locally. No duplicated source checkpoints or mutation of previous bundles.

Output root: `results/lifecycles/opt125m-healthy-one-year/`.
Execution checkout: `/home/filip/analog-research/opt-cmo-exp1/hwa-healthy-year-20261014/`.
Runtime starts from the validated compensation snapshot
`1613de7e4ebf23ecbd3cc961650da33202802fd7`; runtime and tests match the prior local
receipt byte-for-byte. Only new lifecycle settings and orchestration are needed.
Use public workflow plan, ebl campaign run, native inspect and workflow collect.
Expected main coverage: 12 retention records (six baselines plus six one-year
snapshots) and 42 compensation records. Readiness adds 4 and 14, separately.

## Current execution and monitoring handoff

Monitoring is closed. The run-watch process (local PID 2439736) used config
`results/lifecycles/opt125m-healthy-one-year/watch.json`; coordinator owns
collection and scientific review. Poll interval 60 seconds, semantic stall
threshold 900 seconds, at most two bounded incident calls; no duplicate launch.
An initial sandbox DNS failure was resolved by running the watcher with normal
network access; subsequent observation verified the intended live process,
GPU and native artifact progress. No experiment restart was involved.

Remote tmux: `hwa-healthy-year`; supervisor PID 1929560, started at about
2026-10-14 09:24:07 UTC. Execution revision
`65c21e4be3a1dfb33d64705f6947382fe8404773`. Under the output root, `worker.pid`,
`status.json`, `probe-status.json`, `main-status.json`, phase logs/receipts,
`exit-code` and terminal `result.json` record progress. The supervisor subtracts
readiness time from the 1,200-second total allowance. Native stage subprocesses
have deadlines and process-group termination on timeout. Preserve partial
evidence; retry only understood operational failures within the same allowance.

## Evidence and coverage

All eight new lifecycles parsed and described successfully. Runtime code and
tests match the previously validated snapshot; no model implementation changed.
Readiness passed in 26.06 seconds: four retention and 14 compensation records,
native artifact validation, exact replay and exact prior baseline reproduction.
Full execution and collection passed. Wan's absent one-year fit is an explicit model
limitation, not missing execution coverage for the two supported models.

## Interpretation and next decision

Single assignment; no claim of fabricated-hardware reliability, universal
compensation failure or on-chip learning benefit. Larger-age results do not
justify multiplying the observed two-day network loss by elapsed time.

### Completed evidence

All 18 main native bundles passed artifact inspection and completion checks.
Main coverage is exactly 12 retention records and 42 compensation records.
Readiness remains separate (4/14 records). All six programmed baselines reproduce
the prior metrics and apparent hashes exactly; all six replayed two-day apparent
hashes match the original arrays. Source hashes, cohorts, calibration partition,
zero defects, zero writes/verifies and held-state invariance passed the audit.

Execution took 500.69 seconds total: 26.06 readiness and 474.58 main. The additional
replay audit took 1.82 seconds. Peak native GPU allocation was 4,180,950,016 bytes.
Native artifacts occupy 7.92 GB; compact collection, archive and plots keep the
total below 8.0 GB, within the 12 GB cap. GPU release and exit code 0 were checked.
No experimental run failed or was repeated. The collection helper initially
failed before imports because its script directory was on sys.path; setting the
known checkout root fixed that read-only packaging step. Failure evidence is at
`results/lifecycles/opt125m-healthy-one-year/staging/audit-first-failure.log`.

All 475 collected file hashes passed locally. The 4,000,208-byte archive SHA-256
is `869ee6f2d1c540439355c82d5fc010a4660db08becff17fc3e2a98f7911f88da`.
The bounded watcher independently confirmed completion and stopped with no
active incident agent. Collection and scientific review were completed by the
coordinator. Setup/analysis time and token usage were not separately measured.

### One-year WikiText results

Teacher KL (nats/token; lower is better). These are all three inherited HWA
sources on healthy assignment 271002, not independent replicates.

| Model | Source | Programmed | Aged | Global | Row + column | Global + moments |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PCM | std3 | 0.04627 | 1.34140 | 0.05988 | 0.06055 | 0.06094 |
| PCM | std9 | 0.07247 | 1.39645 | 0.08071 | 0.08211 | 0.08229 |
| PCM | cdt3 | 0.11373 | 1.74996 | 0.11944 | 0.11818 | 0.11889 |
| CMO | std3 | 0.04428 | 0.05387 | 0.05383 | 0.05299 | 0.05318 |
| CMO | std9 | 0.07094 | 0.07579 | 0.07577 | 0.07547 | 0.07574 |
| CMO | cdt3 | 0.11420 | 0.11528 | 0.11577 | 0.11527 | 0.11693 |

Perplexity (lower is better).

| Model | Source | Programmed | Aged | Global | Row + column | Global + moments |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PCM | std3 | 49.53 | 157.25 | 50.12 | 49.79 | 50.03 |
| PCM | std9 | 51.18 | 169.56 | 51.56 | 51.33 | 51.49 |
| PCM | cdt3 | 53.86 | 234.86 | 54.09 | 53.60 | 53.91 |
| CMO | std3 | 49.59 | 50.20 | 50.17 | 50.04 | 50.20 |
| CMO | std9 | 51.18 | 51.36 | 51.38 | 51.33 | 51.53 |
| CMO | cdt3 | 54.03 | 53.88 | 53.99 | 53.94 | 54.22 |

### Interpretation

Observed: healthy PCM global gain recovers 98.95%, 99.38% and 99.65% of
the one-year aging KL increase for std3, std9 and cdt3, respectively. Absolute
residual KL is larger than at two days, but perplexity and next-token accuracy
remain close to their programmed baselines. For std3, next-token accuracy goes
32.089% -> 20.267% -> 32.044% (programmed, aged, global). Row/column scaling
can improve perplexity more while improving teacher KL less; no single metric
ordering should be substituted for the primary teacher-KL result.

CMO global gains recover only 0.45% and 0.52% for std3/std9, and worsen cdt3
KL (-44.96%). Row/column LS recovers 9.20%, 6.75% and 0.81%, leaving most
of each original KL gap. Those absolute gaps remain small: for std3, KL is
0.04428 at programming, 0.05387 after aging and 0.05299 after row/column LS;
perplexity is 49.59, 50.20 and 50.04. Cdt3 CMO perplexity improves with aging
even though teacher KL increases, so its KL gap is not a prediction-accuracy
collapse. Mean/variance matching is again nonuniform and can worsen results.

The one-year endpoint therefore does not reveal a large healthy-PCM failure
that these scaling methods cannot mostly remove in this setup. Healthy CMO
has a residual that scaling largely leaves intact, but it is modest in absolute
prediction performance. No claim of irreducibility under other calibration or
of a benefit from weight learning follows. Read noise remains disabled, the
periphery is idealized, and this is model extrapolation on one assignment.
Wan has no released one-year fit and is not assigned invented results.

### Artifacts

- [All 42 method rows](../../results/lifecycles/opt125m-healthy-one-year/analysis/one_year_recovery.csv).
- [Six-condition WikiText summary](../../results/lifecycles/opt125m-healthy-one-year/analysis/one_year_wikitext_summary.csv).
- [Teacher KL over time](../../results/lifecycles/opt125m-healthy-one-year/analysis/healthy_teacher_kl_over_time.png) and [perplexity](../../results/lifecycles/opt125m-healthy-one-year/analysis/healthy_perplexity_over_time.png), with matching PDFs.
- [Native scientific audit](../../results/lifecycles/opt125m-healthy-one-year/compact/scientific-audit.json) and [collection receipt](../../results/lifecycles/opt125m-healthy-one-year/compact/compact-receipt.json).

Regenerate from the collected evidence with:

```bash
MPLCONFIGDIR=/tmp/hwa-healthy-year-mpl /home/filip/miniconda3/envs/py312/bin/python campaigns/opt-cmo-corruption/launch/healthy_year_report.py \
  results/lifecycles/opt125m-healthy-one-year/compact/collected-main-compensation \
  --previous results/lifecycles/opt125m-drift-compensation-pilot/compact/collected-main/compensation.json \
  --output results/lifecycles/opt125m-healthy-one-year/analysis
```

The report verifies exact coverage and matched prior baselines, reports every
predeclared method, and combines only the already evaluated ages (1 s, 1 day,
2 days, 365 days). Dashed plot lines are guides, not newly evaluated intermediate
ages. Plots were visually checked. No further job is running.
