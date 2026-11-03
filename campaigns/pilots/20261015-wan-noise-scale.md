# Pilot: how bad can Wan-2022 endpoint error get?

State: complete; execution, collection and review passed. The user assigned execution on 2026-10-15 ("scale up the Wan error,
see how bad it can get") and confirmed the scale grid and controls below. This
authorizes the extension, readiness probe, launch, monitoring, collection and
interpretation within the caps. It is a follow-up to the
[retention](20261013-aihwkit-retention.md), [compensation](20261013-drift-compensation.md)
and [healthy one-year](20261014-healthy-one-year.md) pilots.

## Question and decision

As the Wan-2022 two-day endpoint sigma is multiplied by k = 1–8, how fast does
teacher KL of the frozen six-MLP HWA sources degrade on healthy arrays? At what
k does the damage leave the small-perturbation regime, and when does the network
collapse? Do global gain or row/column least squares recover any of it?

Prior evidence: at k = 1 the healthy Wan two-day loss is +1.55 WikiText perplexity
(std3), and calibration does not remove it. A simulation of the laws on the std3
weights gave ΔKL ≈ 5·σ² for CMO, gain-calibrated PCM and Wan alike, where σ is the
RMS weight error against the target in units of each output channel's largest
weight. Wan's random error (σ ≈ 0.083 at k = 1) does far less damage than PCM's
coherent decay of the same size.

Expected: quadratic growth to about k = 2, then faster. Collapse (perplexity at
least twice the programmed baseline) somewhere around k = 4–6 for std3, later
for std9. Calibration recovers little at small k. At large k, clamping at 0 µS
rectifies noise into a systematic shrink of small weights, which a gain may
partly undo. Alternative: degradation stays graceful up to k = 8. That would mean
random endpoint error of this form is not a destructive regime for these HWA
sources at any plausible magnitude.

Decision informed: whether large random endpoint error is a regime where digital
calibration fails and rewrite or on-chip learning would be worth a matched test,
and how much margin the HWA sources have. No on-chip learning or viability claim.

## Frozen comparison

Inherited unchanged (confirmed in the parent pilots): OPT-125M, final six fc1/fc2
MLPs, 28,311,552 logical weights, 56,623,104 physical devices; sources std3
(`86522e23…`), std9 (`b5837bd8…`) and cdt3 (`8a44f11b…`); healthy assignment
271002 only (user direction of 2026-10-14; no faults, so no fault rewrite applies);
retention seed 391013; read noise off; Wan positive/negative pairs over 0–40 µS.
Cohorts: 256 development, 1,024 WikiText test and 384 generated test sequences
of 256 tokens. Teacher KL is primary; perplexity and next-token accuracy are
secondary. The FP32 teacher scores WikiText perplexity 46.60 and 33.08 % accuracy.

Intervention (confirmed by user): Wan's fitted two-day sigma polynomial is
multiplied by **k ∈ {1, 1.5, 2, 3, 4, 6, 8}**. The 1-second programming baseline
stays at the nominal fit. All k reuse the same standard-normal draw per device:
matched realizations, with k = 1 equal to the original two-day array.
Results are clamped at 0 µS as in AIHWKit. At k = 8 the mid-range sigma is
31.4 µS (79 % of G_max). Every k > 1 is a synthetic stress test beyond the
published fit, not a device model.

Controls (confirmed by user): uncompensated, **global gain** (12 scalars) and
**row/column least squares** (46,080 parameters, ten iterations). They come from
the existing `compensate` stage with the frozen fit rules: gains in [0.1, 10],
references are the programmed array's one-hot readouts, zero writes. The other
five compensators are not run. Calibration recovery is measured against each
array's own programmed baseline, as in the compensation pilot.

Coverage: 3 sources × (1 programmed + 7 scaled) = 24 retention rows, and
3 × 7 × 2 = 42 compensation rows. All rows are reported; nothing is pooled
across sources, and nothing is selected.

Descriptive markers (proposed, fixed before results):

- per-source curvature c(k) = (KL_k − KL_clean) / σ_k², with σ_k the stored
  `residual_rms` (weight error against the target). The quadratic regime ends at
  the first k where c(k) > 1.5 × c(1);
- collapse at the first k where WikiText perplexity ≥ 2 × the programmed baseline;
- calibration helps materially at k if a method regains ≥ 25 % of the KL
  increase over the programmed array for all three sources.

Evidence class: exploratory, model-based, one assignment, one noise realization
per k. Sources are not replicates. Excluded claims: fabricated hardware, other
assignments, faulty arrays, read noise, intermediate or other ages.

## Budget and execution

Extension (Codifier): optional `retention.wan_noise_scales`. It is omitted at
its default `[1]`, so existing lifecycle digests are unchanged.
`RetentionSeries.snapshot(seconds, noise_scale)` reuses the age stream. Scaled
rows carry `noise_scale` and labels `t172800s-x<k>`; the `compensate` stage and
`collect` handle the scale. Tests: scale 1 is identical to unscaled, scaled
errors are exactly k times the same draw, schema validation, and native
retention → compensation → collect with an exact nominal replay. The workflow,
retention and compensation suites passed: 539 tests, 28 optional skips. In the
full `tests/` run, only files that start subprocesses fail, from a leaked
`MKL_THREADING_LAYER` environment setting; those files pass when run alone.

Lifecycles: `campaigns/opt-cmo-corruption/lifecycles/opt125m-wan-scale-{retention,compensation}-{std3,std9,cdt3,probe}.json`.
They are copies of the healthy-year lifecycles with only the id, question,
retention section (Wan, 172,800 s, the scales) and compensation methods changed.
Describe: 8 retention snapshots per source, about 0.74 GB saved state each, and
14 compensation rows per source. The probe uses the earlier one-MLP digital
readiness source with eight-sequence cohorts and is excluded from results.

Target: one idle RTX 5090 on **fifi**. loulou and riri are occupied by another
user's jobs, and trex by another user. fifi has the same GPU model, 170 SMs and
PyTorch 2.11.0+cu128 as loulou, where the original arrays were drawn. The run
gate requires the programmed array and the k = 1 two-day array to reproduce the
original Wan rows (apparent hash and metrics) exactly. If the CUDA RNG does not
replay across hosts, the run stops and escalates; it is not relabelled.
Launcher: `campaigns/opt-cmo-corruption/launch/wan_scale_pilot.py all --inputs <inputs.json>`
(probe, then main; public plan/run/inspect/collect only).

Caps: **45 GPU-minutes** (launcher allowance, including readiness and understood
retries), **8 GB** new artifacts on fifi, under 1 GB compact evidence locally.
Expected 15–25 minutes from the healthy-year throughput (474 s for 54 rows). One
GPU used in the daytime window; no other job of ours is on a 5090.

Execution root on fifi: `/home/filip/analog-research/opt-cmo-exp1/hwa-wan-scale-20261015/`.
Copied inputs: `/home/filip/analog-research/opt-cmo-exp1/inputs/wan-scale-20261015/`.
Output: `results/lifecycles/opt125m-wan-noise-scale/`.

## Current execution and monitoring handoff

Monitoring is closed. Exit code 0 at about 15:06:36 UTC; tmux session ended, GPU
released (no compute processes), no job running. Original record follows.

Launched 2026-10-15 14:56:03 UTC (16:56 Paris) on fifi, tmux `hwa-wan-scale`,
worker PID 2394141. The GPU was checked idle immediately before launch.
Execution commit `0b2a99e` on branch `wan-scale-pilot`; base `65c21e4` is the
validated healthy-year snapshot, shipped as `opt-cmo-exp1/wan-scale-0b2a99e.bundle`.
The checkout was clean.

The inputs file is `inputs/wan-scale-20261015/inputs.json`. All source, teacher
and corpus SHA-256s were verified on fifi before launch. std3 is the fifi copy of
the exp-1g staged source; std9, cdt3 and the probe source were copied from loulou.
The original Wan measurements were copied from the loulou retention bundles,
byte-identical (std3 `3fa4b166…`, std9 `e547f1ed…`, cdt3 `a96f23c3…`,
probe `5ebe69b0…`).

Progress files under the output root: `status.json`, `{probe,main}-status.json`,
`run.log`, `{probe,main}-receipt.json`, `result.json` and `exit-code`. One monitoring
owner, the executing Claude session, polls every 45 s and stops on the exit code
or a dead worker. Recovery: retry only an understood operational failure, within
the 45-minute allowance. A replay-gate failure is escalated, not retried.

Inspect: `python -m ebl runs inspect <bundle> --verify-artifacts --require-complete`.
The launcher runs this for every bundle, then `python -m workflow collect` into
`collected-{probe,main}-{retention,compensation}`. Report:
`campaigns/opt-cmo-corruption/launch/wan_scale_report.py <collected-main-retention> <collected-main-compensation> --output results/lifecycles/opt125m-wan-noise-scale/analysis`.

## Evidence and coverage

**Execution.**
- Total 633.1 s against the 2,700 s allowance: readiness 26.4 s, main 606.6 s.
  No failure, retry, exclusion or replacement occurred.
- Peak native GPU allocation was 3,954,457,600 bytes.
- The launcher ran `ebl runs inspect --verify-artifacts --require-complete` on
  every bundle and `workflow collect` for both kinds; native validation passed.
- Readiness had 8 retention and 14 compensation rows on the one-MLP digital
  source and is excluded from results.

**Coverage.** Exactly as frozen: 24 retention rows (3 sources × programmed + 7 scales)
and 42 compensation rows (3 × 7 × 2 methods). Every row is nominal, Wan, assignment
271002, two days, zero failed devices, zero writes and verifies, exact replay, no
duplicate coverage, and has an unchanged held state for compensation.

**Replay gate.** For std3, std9, cdt3 and the probe, the programmed array and the
k = 1 two-day array matched the original loulou Wan rows exactly, in apparent
SHA-256 and in all metrics. The CUDA RNG therefore replayed across the two 5090
hosts, and k = 1 is the original two-day array.

**Storage.** 4.6 GB on fifi, including 10 files over 50 MB (state and cache
checkpoints, listed in `results/lifecycles/opt125m-wan-noise-scale/remote-large-files.txt`).
The 459 smaller files were copied to `results/lifecycles/opt125m-wan-noise-scale/compact/` (21 MB);
their local and remote SHA-256 lists are identical (`local-sha256.txt`, `remote-sha256.txt`).
Collected `retention.json` `b90525c2…`, `compensation.json` `11a60dcb…`.

### WikiText results

σ is the stored RMS weight error against the target, in normalized weight units.
KL is teacher KL in nats/token. The "regained" columns give the percentage of the
KL increase over the programmed array removed by each method; negative means worse.

| Source | k | σ | KL | Perplexity (× programmed) | Accuracy % | Global KL / ppl / regained | Row + col KL / ppl / regained |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| std3 | prog. | 0.046 | 0.0518 | 49.84 | 32.00 | | |
| std3 | 1 | 0.083 | 0.0766 | 51.39 (1.03) | 31.65 | 0.0773 / 51.59 / −3 % | 0.0787 / 51.02 / −9 % |
| std3 | 2 | 0.159 | 0.1789 | 57.48 (1.15) | 30.55 | 0.1816 / 57.79 / −2 % | 0.1928 / 56.05 / −11 % |
| std3 | 3 | 0.227 | 0.3676 | 70.64 (1.42) | 28.34 | 0.3519 / 69.11 / 5 % | 0.3744 / 65.35 / −2 % |
| std3 | 4 | 0.289 | 0.6974 | 100.95 (2.03) | 24.73 | 0.5902 / 88.70 / 17 % | 0.6021 / 79.93 / 15 % |
| std3 | 6 | 0.406 | 2.0412 | 413.13 (8.29) | 14.17 | 1.2686 / 180.98 / 39 % | 1.0787 / 124.34 / 48 % |
| std3 | 8 | 0.518 | 3.6676 | 2077.39 (41.7) | 7.45 | 2.0757 / 415.21 / 44 % | 1.4555 / 178.24 / 61 % |
| std9 | prog. | 0.045 | 0.0748 | 51.32 | 31.54 | | |
| std9 | 1 | 0.083 | 0.0867 | 52.05 (1.01) | 31.44 | 0.0866 / 52.18 / 1 % | 0.0889 / 51.87 / −18 % |
| std9 | 2 | 0.160 | 0.1371 | 54.90 (1.07) | 30.89 | 0.1382 / 55.15 / −2 % | 0.1565 / 54.56 / −31 % |
| std9 | 3 | 0.229 | 0.2242 | 60.27 (1.17) | 29.93 | 0.2220 / 60.01 / 2 % | 0.2799 / 60.17 / −37 % |
| std9 | 4 | 0.293 | 0.3665 | 70.28 (1.37) | 28.28 | 0.3383 / 67.46 / 10 % | 0.4546 / 69.88 / −30 % |
| std9 | 6 | 0.413 | 0.9671 | 133.99 (2.61) | 21.84 | 0.6712 / 95.06 / 33 % | 0.8825 / 103.65 / 10 % |
| std9 | 8 | 0.527 | 2.2006 | 481.30 (9.38) | 13.08 | 1.1240 / 152.26 / 51 % | 1.2673 / 150.05 / 44 % |
| cdt3 | prog. | 0.044 | 0.1150 | 53.96 | 31.01 | | |
| cdt3 | 1 | 0.082 | 0.1213 | 54.24 (1.01) | 30.99 | 0.1226 / 54.50 / −22 % | 0.1204 / 53.88 / 13 % |
| cdt3 | 2 | 0.159 | 0.1466 | 55.60 (1.03) | 30.73 | 0.1492 / 56.02 / −8 % | 0.1621 / 54.96 / −49 % |
| cdt3 | 3 | 0.230 | 0.1891 | 58.00 (1.07) | 30.31 | 0.1894 / 58.05 / −1 % | 0.2701 / 59.24 / −109 % |
| cdt3 | 4 | 0.295 | 0.2538 | 62.03 (1.15) | 29.58 | 0.2435 / 60.86 / 7 % | 0.4612 / 69.35 / −149 % |
| cdt3 | 6 | 0.418 | 0.5065 | 81.36 (1.51) | 26.48 | 0.3930 / 69.83 / 29 % | 1.0046 / 114.34 / −127 % |
| cdt3 | 8 | 0.533 | 1.0776 | 149.61 (2.77) | 20.86 | 0.5833 / 84.02 / 51 % | 1.4617 / 177.65 / −40 % |

The k = 1.5 rows are in the [summary CSV](../../results/lifecycles/opt125m-wan-noise-scale/analysis/wan_scale_summary.csv).
The generated-text cohort agrees qualitatively: KL rises monotonically in k for
every source, and global gain lowers it at k ≥ 6. For example, std3 at k = 8 goes
from 3.645 to 1.891.

**Predeclared markers** ([markers JSON](../../results/lifecycles/opt125m-wan-noise-scale/analysis/wan_scale_markers.json)):

| Source | Curvature c(1) | Quadratic regime ends | Collapse (perplexity ≥ 2× programmed) |
| --- | ---: | --- | --- |
| std3 | 5.11 | k = 4 (σ 0.29) | k = 4 (100.95 vs 49.84) |
| std9 | 2.50 | k = 6 (σ 0.41) | k = 6 (133.99 vs 51.32) |
| cdt3 | 1.17 | k = 6 (σ 0.42) | k = 8 (149.61 vs 53.96) |

Global gain is material (≥ 25 % regained for all three sources) at k = 6 and 8 only.
Row/column LS is never material: cdt3 is worse than uncalibrated at every k ≥ 1.5.

**Fitted global gains** (from the saved calibration checkpoints, per matrix):
about 1.008 at k = 1 and 1.01 at k = 1.5–2. They fall below one from k = 3:
0.94–0.99 at k = 4 and **0.76–0.87 at k = 8**.

Figures (PNG and PDF):
- [teacher KL](../../results/lifecycles/opt125m-wan-noise-scale/analysis/wan_scale_teacher_kl.png)
- [perplexity](../../results/lifecycles/opt125m-wan-noise-scale/analysis/wan_scale_perplexity.png)
- [KL increase versus σ²](../../results/lifecycles/opt125m-wan-noise-scale/analysis/wan_scale_curvature.png)

All three were visually checked. Regenerate with:

```bash
MPLCONFIGDIR=/tmp/wan-scale-mpl /home/filip/miniconda3/envs/py312/bin/python campaigns/opt-cmo-corruption/launch/wan_scale_report.py \
  results/lifecycles/opt125m-wan-noise-scale/compact/collected-main-retention \
  results/lifecycles/opt125m-wan-noise-scale/compact/collected-main-compensation \
  --output results/lifecycles/opt125m-wan-noise-scale/analysis
```

## Interpretation and next decision

**Observed.**
- **The fitted Wan model has wide margin.** Damage is gradual up to about 2× the
  fitted two-day sigma: at k = 2, WikiText perplexity is +15 % for std3, +7 % for
  std9 and +3 % for cdt3. Collapse needs k = 4 (std3), 6 (std9) or 8 (cdt3).
- **The weight error grows more slowly than k.** σ at k = 8 is 0.52, not
  8 × 0.083 = 0.66, because the clamp at 0 µS cuts off negative excursions.
- **The KL increase is close to quadratic in σ until collapse approaches.** Each
  source has its own curvature: 5.1 for std3, 2.5 for std9 and 1.2 for cdt3. The
  stronger-noise and corruption-aware sources are 2–4 times less sensitive.
- **The robust sources overtake std3 early.** Their programmed arrays start worse,
  but std9 has lower KL than std3 from k = 1.5 and cdt3 from k = 2.
- **Calibration is negligible or harmful through k = 4, and partial after that.**
  Global gain regains 29–51 % of the KL increase at k = 6–8 but leaves every source
  far from its programmed baseline; std3 at k = 8 is still at perplexity 415.
  Row/column LS is erratic: best for std3 at large k, and worse than nothing for
  cdt3 and, below k = 6, for std9.

**Against expectations.**
- Collapse timing and the weak calibration effect at small k match the prediction.
- The quadratic regime lasted longer than predicted: up to k = 3 for std3 and
  k = 4 for std9 and cdt3, not k ≈ 2.
- The predicted mechanism for large-k gain recovery is **contradicted**. The
  prediction was that clamping would shrink small weights and a gain above one
  would restore them. Instead the fitted gains are *below* one. Inference: noise
  inflates the mean absolute readout, so the mean-absolute rule scales the noisy
  matrix down. That reduces the noise power entering the residual stream, a
  shrinkage effect rather than a restored weight scale. The same dilution would
  explain row/column LS fitting gains that hurt the low-noise cdt3 source. Neither
  explanation was tested here.

**Scoped verdict.**
- Random endpoint error of the Wan-2022 form is not a destructive regime for these
  HWA sources within about twice its published two-day fit.
- It becomes destructive only at four times or more, which is outside any measured
  device behaviour.
- At those magnitudes, zero-write digital calibration leaves most of the loss.
- The lever with the largest observed effect is HWA noise strength or corruption
  awareness, not calibration.

**Limits.** One assignment, one healthy array and one noise realization per k
(matched across k). Every k > 1 is beyond the published fit. Read noise is off,
the periphery is idealized, and there are no faults or other ages. The three
sources are not replicates.

**Next decision.** This result does not establish an on-chip-learning benefit. A
matched rewrite or on-chip test against random endpoint error is only relevant
at k ≥ 3–4. Any such test should first justify that magnitude physically, and it
must include global gain and the stronger HWA sources as controls. No further run
is launched by this pilot.
