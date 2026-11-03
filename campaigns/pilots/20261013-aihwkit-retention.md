# Pilot: retention of the frozen HWA-trained OPT arrays

State: complete; the user confirmed the proposed OPT source/remapping comparison
on 2026-10-13. Native integration and the GPU readiness run passed. Authorization: the user requested
finding the three AIHWKit PCM/ReRAM drift models and applying them to the saved
HWA arrays at multiple ages. All declared measurements, collection and review are complete within the pilot cap.

## Question and decision

How does the performance of our frozen HWA sources change with the three
AIHWKit retention models? Report changes from each model's own deployment
baseline, alongside absolute teacher KL and perplexity. This is a descriptive,
exploratory model-based comparison, with no checkpoint selection, retraining,
on-chip adaptation or physical-viability claim.

## Verified model catalogue

Pinned release: AIHWKit **1.1.0**, matching this repository's existing device
contracts. Source: IBM's [PCM implementation](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/pcm.py)
and [ReRAM implementations](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/reram.py).
These are three built-in fitted inference models, not the only customizable
noise classes. `CustomDriftPCMLikeNoiseModel` accepts a user-supplied PCM fit.

| Class | Retention law | Times and caveats |
| --- | --- | --- |
| `PCMLikeNoiseModel` | G(t) = G_programmed × ((t+20)/20)^(-nu); conductance-dependent Gaussian nu sampled once per device | Seconds since its 20 s reference; separate accumulated read noise. Default G_max 25 uS. |
| `ReRamCMONoiseModel` | Mean shift -0.08900206 ln(t) uS; relaxation standard deviation 0.04201137 ln(t)+0.41183342 uS | t=0 returns the programmed state. Positive ages include relaxation, optional read noise, then clamping to 9–88.199997 uS. The prepared sampler supports t=0 or t>=1 s. |
| `ReRamWan2022NoiseModel` | Conductance-dependent fourth-order polynomial standard deviation around the original target | Only 1, 86400 and 172800 s. This is total endpoint error; it ignores the supplied programmed state. No interpolation or separate read-noise fit. |

Important implementation details:

- The CMO 2% fit is selected with `acceptance_range=2`, not `0.02`; the latter
  silently selects the 0.2% fit in this release.
- AIHWKit's Wan converter defaults to 25 uS although the coefficients were
  fitted at 40 uS. The prepared Wan sampler explicitly defaults to 40 uS and
  was checked against upstream instantiated with that same range. Tests also
  cover its 25 uS constructor setting. Do not silently change this between arms.
- A PCM programming-only baseline is the saved G_programmed: calling its drift
  API at t=0 still adds read noise. Wan's programming baseline is its 1 s fit;
  it must not be described as a measured zero-time distribution.
- CMO and Wan give marginal distributions at each age, not a measured joint
  time process. Use separately seeded, held snapshots. A line joining them is
  a comparison of ages, not evidence of continuous motion of one device.
- All time points start from the same programming baseline and target. PCM
  keeps the same per-device exponent. Never compound snapshots or draw new
  noise per example. Permanent faults retain their exact conductance, including
  stuck-open devices excluded from CMO's log10 read-noise calculation.

## Frozen comparison (confirmed 2026-10-13)

The user confirmed using the three current OPT HWA sources, separately remapped
into each technology, at the proposed ages and deployment baselines. The linked
bounded pilot defines the execution cap and fault coverage.

- OPT-125M, final six fc1/fc2 MLP pairs, **28,311,552 logical analog weights**;
  all other components remain digital. Use the existing selected whole-model
  HWA sources, preserving their learned digital parameters and mapping scales.
- Sources: standard 3% Gaussian HWA (`std3`), standard 9% Gaussian HWA (`std9`),
  and 3%-fault-aware HWA (`cdt3`) from [exp-007](../opt-cmo-corruption/series/001-hwa-robustness/results/exp-007.md).
- Remap the same frozen source into CMO active/reference pairs (9–88.199997 uS),
  PCM differential pairs (0–25 uS), and Wan differential pairs (0–40 uS).
  Record two devices per logical weight and the different encoding. This is
  transfer of CMO-trained weights, not technology-specific HWA training or
  aging the same physical CMO array with three valid device laws.
- Common ages: **1 s, 1 day, 2 days**, with separately identified deployment
  baselines. No unsupported Wan interpolation.
- Preserve sequence length 256, the pinned teacher and existing evaluation
  cohorts (1,024 WikiText sequences and 384 generated-text sequences). Primary
  metric teacher KL; report perplexity and weight/conductance residuals.
- Bounded pilot: assignment 271002, healthy and 3% permanent random
  faults, no spatial concentration variation. Report each case separately.
  A single assignment is not general evidence. Across technology mappings,
  use explicit physical fault-mask and stuck-value mapping rules, not an
  implicit reuse of CMO conductance values in another range.
- Report drift/retention without compensation. Preserve the source regime's
  disabled read noise to isolate retention: PCM and CMO read-noise scales are
  zero. Wan's total endpoint error cannot be separated into programming and
  retention components; it has no separate fitted read-noise term.
- No additional training data, writes, verifies or recovery. Initial programming
  cost must remain separate from the zero-write retention evaluation.

The HWA recipes are frozen; this task changes no training decisions. Rasch 2023
uses global digital drift compensation, so an uncompensated sweep would be an
explicit deviation isolating raw degradation. Büchel 2025 Appendix F.2 evaluates
PCM drift and read noise at multiple ages on already trained models, which is
the relevant precedent. Its one-minute-to-one-year grid is not a valid grid for
the Wan model. Both local reference papers were consulted for these distinctions.

Source identities recorded by exp-007:

| Source | Selected artifact SHA-256 | Availability checked here |
| --- | --- | --- |
| std3 | `86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028` | Local staged source exists, 501,121,995 bytes |
| std9 | `b5837bd8256c448289c74695e8d992c9116e9b06c2bc25f202ec150d689c814e` | Original remote checkpoint used; full hash verified |
| cdt3 | `8a44f11ba6ed8a833fa66954c6124847d41a2dd74a7023ae04d775a1c729ac89` | Local staged source exists, 501,122,059 bytes |

Local source root: `results/lifecycles/opt125m-cmo-l6-exp1g-four-arm/staging/sources/`.
Exact source paths and remote receipts are retained in that campaign's
`launch/exp1g_four_arm_sources.json` and collected main bundles. All three source
hashes, the teacher hash and all three corpus hashes were verified before
execution. Exact resolved paths and hashes are retained in `inputs.json`.

## Budget and execution

Execution cap: one idle authorized RTX 5090, up to one GPU-hour and 20 GB of new
artifacts, including a small readiness run and any understood retry. Measured readiness plus main execution took 640.73 seconds wall time. The proposed pilot has 3 sources × 3 model regimes
× 2 fault cases × 1 assignment, with 3 age snapshots plus a baseline for each
regime (72 evaluations before deduplicating Wan's 1 s baseline). The user's confirmation authorizes this bounded pilot, not a larger sweep.

Output root: `results/lifecycles/opt125m-aihwkit-retention-pilot/`.
Retain native input/config/source identities, metrics, baseline device states,
PCM exponents, time-specific seeds and apparent-state hashes. Retain exact
replay evidence and failed attempts. Do not mutate any source/deployment bundle.

The native lifecycle now supports an optional `retention` section and
`workflow plan --hwa-source ARM=PATH`; sources retain their original upstream
provenance. The selected frozen weights are not retrained or reselected.
Lifecycle files are in `campaigns/opt-cmo-corruption/lifecycles/`:

- `opt125m-aihwkit-retention-std3.json`
- `opt125m-aihwkit-retention-std9.json`
- `opt125m-aihwkit-retention-cdt3.json`
- `opt125m-aihwkit-retention-probe.json` (one analog MLP, unchanged teacher,
  eight sequences per evaluation cohort; readiness only).

Each evidence lifecycle declares 24 snapshots and an estimated 4.87 GB of
replay state, for 72 snapshots and 14.61 GB across the three sources. The probe
adds approximately 0.81 GB. A native stage verifies source compatibility,
fault permanence, unchanged model weights and exact serialized replay.

## Completed execution and evidence locations

The executing Codex session owned monitoring through collection and review.
Target: `loulou`, RTX 5090, CUDA environment
`/home/filip/miniconda3/envs/py312/bin/python` (PyTorch 2.11.0+cu128).
The GPU was idle before execution and was released after completion.
Frozen execution revision: `7cbb61a72bbfcdef01c9bd9ae993822fd53bb3f8`
(base `af25aaa109cf6eb2e949836fab2f8d7873eb801b`). Exact source receipt and
all input hashes accompany the native bundles.

- Readiness: 24 rows, 19.17 s wall time, 2.19 GB peak GPU allocation,
  0.812 GB replay states. Readiness results are excluded from scientific evidence.
- Main: **72 rows**, 621.56 s wall time, **4.18 GB** peak GPU allocation.
  Completed 2026-10-13 at 21:29:16 UTC. Retention stages took 189.65, 190.76
  and 190.70 s for std3, std9 and cdt3 respectively.
- Total readiness plus main: **10 min 40.73 s**, within one GPU-hour.
  No failed runs, exclusions, replacements or operational retries.
  `attempt-002` denotes execution after the launcher's dry-run attempt.
- Native result root: **17.21 GB**. The execution tree plus remote and local
  source bundles occupy approximately 18.26 GB; compact collection and plots
  add less than 0.02 GB, keeping new artifacts below 20 GB.
- Setup/reading/validation time and token use were not separately measured.

Remote root:
`/home/filip/analog-research/opt-cmo-exp1/hwa-retention-20261013/results/lifecycles/opt125m-aihwkit-retention-pilot/`.
Full replay states remain there with declared artifact hashes. Compact evidence
was collected locally under
[`results/lifecycles/opt125m-aihwkit-retention-pilot/compact/`](../../results/lifecycles/opt125m-aihwkit-retention-pilot/compact/).
All **217 copied files** passed local SHA-256 verification. The archive hash is
`6aeadc79d7e1558aa9c5e70216db921d37c9a47b4ff8acef81a42de35c3db0e8`.
The collection preserves main/probe launch, exit and validation receipts,
plans, native manifests, metrics and the remote raw-state inventory.

The main and readiness phases used tmux `hwa-retention-main` and
`hwa-retention-probe`, respectively, with process-group deadlines. Both exited
0. The monitoring handoff is closed; no running job remains for this pilot.

## Evidence and coverage

Implemented reusable conductance laws and `RetentionSeries` in
[`training/inference_retention.py`](../../training/inference_retention.py).
The existing PCM programming-event API delegates to the same primitives,
preserving its behavior. New tests cover AIHWKit 1.1.0 numerical agreement,
supported times, unmodified baselines, permanent faults, order-independent
queries and state replay. They are engineering validation, not HWA performance
measurements.

Validation environment: local CPU, Python environment `py312`, PyTorch 2.5.1;
AIHWKit 1.1.0 Python noise modules loaded from the installed `aihwkit` environment.
Final checks passed: **498 tests passed, 26 skipped** in the full workflow and
retention suite (the ordinary environment lacks optional AIHWKit imports).
The focused retention and native retention-workflow suite then passed **all
55 tests** with AIHWKit 1.1.0 available, including direct law and programming
parity. `git diff --check` passed.

```bash
/home/filip/miniconda3/envs/py312/bin/python - <<'PY'
import sys
sys.path.append('/home/filip/miniconda3/envs/aihwkit/lib/python3.12/site-packages')
import pytest
raise SystemExit(pytest.main(['-q', 'tests/test_inference_retention.py', 'tests/test_workflow_retention.py']))
PY
/home/filip/miniconda3/envs/py312/bin/python -m pytest -q tests/test_workflow_*.py tests/test_inference_retention.py
```

## Collected measurements

All nine main native stage bundles and four readiness bundles passed
`ebl runs inspect --verify-artifacts --require-complete`. Coverage is complete:
three sources × three device mappings × two fault cases × four snapshot rows.
The six Wan 1-second rows reuse their baseline state and metrics, so there are
**66 distinct evaluated snapshots**, plus the three clean-source evaluations.

The scientific audit verified source/input hashes, unchanged frozen weights,
exact serialized replay, cohort sizes/token counts and common fault identity.
Each faulty array contains **1,698,696** permanently stuck physical devices out
of **56,623,104**, with identity hash
`06b3159caaa48d4135fa71537b610311e34f1713abebfa2cc01ea43c3c923b26`.
Healthy and faulty cases are reported separately. All retention write and verify
counts are zero; initial programming remains separately recorded.

Primary teacher KL is in nats/token. Below, each cell is deployment baseline
→ two days on **WikiText**. Wan's baseline is the 1-second realization.

| HWA source | Device model | Healthy KL | 3% faults KL | Healthy perplexity | 3% faults perplexity |
| --- | --- | ---: | ---: | ---: | ---: |
| std3 | CMO ReRAM | 0.0443 → 0.0499 | 0.1910 → 0.1978 | 49.59 → 49.78 | 57.52 → 57.82 |
| std3 | PCM | 0.0463 → 0.6982 | 0.1304 → 0.7962 | 49.53 → 84.04 | 53.07 → 91.33 |
| std3 | Wan ReRAM | 0.0518 → 0.0766 | 0.1365 → 0.1647 | 49.84 → 51.39 | 53.40 → 55.46 |
| std9 | CMO ReRAM | 0.0709 → 0.0742 | 0.1482 → 0.1525 | 51.18 → 51.23 | 54.83 → 54.93 |
| std9 | PCM | 0.0725 → 0.7661 | 0.1156 → 0.8391 | 51.18 → 91.48 | 52.66 → 97.28 |
| std9 | Wan ReRAM | 0.0748 → 0.0867 | 0.1179 → 0.1315 | 51.32 → 52.05 | 52.80 → 53.73 |
| cdt3 | CMO ReRAM | 0.1142 → 0.1149 | 0.1520 → 0.1547 | 54.03 → 53.90 | 55.43 → 55.39 |
| cdt3 | PCM | 0.1137 → 1.1507 | 0.1341 → 1.2314 | 53.86 → 130.67 | 54.13 → 140.73 |
| cdt3 | Wan ReRAM | 0.1150 → 0.1213 | 0.1349 → 0.1429 | 53.96 → 54.24 | 54.20 → 54.61 |

The clean frozen sources already differ: WikiText KL/perplexity are
0.04167/49.34 (std3), 0.06966/51.03 (std9) and 0.11338/53.91 (cdt3).
Absolute endpoint ranking therefore includes the source's pre-existing accuracy
tradeoff. Use within-source changes for the retention comparison.

Full 1-second, 1-day and 2-day values for all three cohorts, conductance/weight
residuals, counts and identities are in the
[72-row CSV](../../results/lifecycles/opt125m-aihwkit-retention-pilot/compact/collected-main/retention.csv)
and [two-day summary](../../results/lifecycles/opt125m-aihwkit-retention-pilot/analysis/two_day_summary.csv).
Figures (PNG and matching PDF):

- [WikiText teacher KL](../../results/lifecycles/opt125m-aihwkit-retention-pilot/analysis/transfer_wikitext_teacher_kl.png)
  and [perplexity](../../results/lifecycles/opt125m-aihwkit-retention-pilot/analysis/transfer_wikitext_perplexity.png).
- [Generated-text teacher KL](../../results/lifecycles/opt125m-aihwkit-retention-pilot/analysis/transfer_generated_teacher_kl.png)
  and [perplexity](../../results/lifecycles/opt125m-aihwkit-retention-pilot/analysis/transfer_generated_perplexity.png).

Regenerate figures from collected measurements only:

```bash
/home/filip/miniconda3/envs/py312/bin/python campaigns/opt-cmo-corruption/launch/retention_report.py \
  results/lifecycles/opt125m-aihwkit-retention-pilot/compact/collected-main \
  --output results/lifecycles/opt125m-aihwkit-retention-pilot/analysis
```

## Interpretation and next decision

**Observed:** CMO gives the smallest two-day KL increase in every source/case
on both transfer cohorts. Wan gives a larger increase, and uncompensated PCM
produces a much larger degradation. Healthy-array WikiText PCM perplexity
increases from 49.53 to 84.04 for std3, from 51.18 to 91.48 for std9 and from
53.86 to 130.67 for cdt3. The faulty PCM endpoints are worse still.
The larger Gaussian training noise and corruption-aware training used here do
not eliminate PCM drift degradation.

On the faulty ReRAM arrays, std9 has the lowest two-day WikiText KL; cdt3 has
the lowest generated-text KL. These are corpus-specific endpoint rankings,
not evidence that one HWA recipe dominates. Small CMO perplexity decreases
for cdt3 coexist with increased teacher KL; the two metrics need not rank every
perturbation identically.

**Mechanistic interpretation:** shared CMO mean relaxation largely cancels
between active and reference devices in healthy, unclamped pairs; its random
relaxation and clamping remain. PCM power-law attenuation directly changes
the differential weight without compensation. This explanation follows the
declared laws, but the experiment does not isolate each component causally.

**Limits and verdict:** the requested models have been implemented and applied
successfully, with complete declared coverage. This is an exploratory,
model-based, single-assignment result. Sources were trained under the CMO
regime and remapped; encodings and fitted error scales differ between
technologies. The comparison cannot establish an intrinsic technology ranking.
Wan is a total endpoint-error fit at discrete ages, and CMO/Wan do not provide
a measured joint device trajectory. Separate read noise and drift compensation
were disabled. No on-chip recovery or viability claim follows from this pilot.

The next scientific decision, if PCM deployment is pursued, is whether to
compare explicit digital drift compensation and drift-aware HWA under a new
matched, multi-assignment design. No additional run is required to close this
pilot, and none has been launched.
