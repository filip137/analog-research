---
id: "exp-001"
title: "Experiment 1: standard versus corruption-aware HWA under 0-5 % stuck CMO devices"
status: "running"
hypotheses: []
---
# exp-001: Experiment 1, standard versus corruption-aware HWA

## Question and dependencies

**Question.** OPT-125M runs with CMO fc1/fc2 in its last three decoder layers.
Each HWA source yields one checkpoint, deployed without retraining on
independent arrays. As the fraction of independent, randomly stuck devices
rises from 0 to 5 %, how do these sources degrade in teacher KL?

- the **pretrained** model;
- **standard HWA**, with 3 % per-output-channel Gaussian weight noise;
- **corruption-aware HWA**, with the same noise plus 3 % i.i.d. random stuck
  devices during training.

**Decision informed.**
- Which source or sources to carry into the on-chip recovery experiments.
- Which stuck-device range is informative for recovery, neither negligible
  nor destructive.

**Expected outcome.** Standard HWA beats the pretrained model at every rate.
Corruption-aware HWA beats standard HWA at 2–5 %, at a small cost at 0 %.

**Alternatives that change the decision.**
- Corruption-aware HWA is no better than standard HWA. Then carry only
  standard HWA forward.
- Every source is destroyed at 4–5 %. Then recovery experiments use lower rates.
- HWA does not beat the pretrained model. Then revisit the noise convention
  before any recovery claim.

**Prior evidence.**
[CIFAR PCM HWA/CDT](../../../../../docs/cifar_pcm_hwa_results.md) is a
different network, technology and fault law. There, corruption-aware training
(CDT) helped at matched fault rates but cost fidelity on healthy hardware.
Several fits needed convergence review. No OPT or CMO result exists yet.

**Dependencies.**
- [CMO device contract](../../../../../docs/reram_cmo_device.md) and
  [lifecycle contract](../../../../../docs/crossbar_lifecycle.md).
- The pinned corpus (below).
- Stage 1a (the learning-rate development run) must finish before Experiment 1
  proper is written.

## Cases and controls

`confirmed` means you chose it in conversation on 2026-10-08; `proposed` means
it is open to your change before launch.

| Field | Setting | Status |
| --- | --- | --- |
| Network | `opt_mlp_suffix`, `facebook/opt-125m` at revision `27dcfa74`, at two sizes. **L3**: fc1/fc2 of decoder layers 9–11 on CMO arrays, 14,155,776 weights (28,311,552 devices). **L6**: layers 6–11, 28,311,552 weights (56,623,104 devices). L6 reuses every L3 setting; its HWA rate was tuned on L3 | L3: confirmed; L6 added 2026-10-09 |
| Corpus | `wikitext-2-raw-v1`, SHA-256 `33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01`. Built from Hugging Face `Salesforce/wikitext` at revision `b08601e04326c79dfdd32d625aee71d232d685c3`, the raw split lines concatenated. Tokens: 2,391,885 train, 247,290 validation, 283,288 test | confirmed |
| Metric | `teacher_kl` primary; perplexity, cross-entropy, next-token accuracy and teacher agreement also reported | confirmed |
| Sequences and cohorts | Length 256. HWA: 8,192 train sequences (2.10 M tokens). Development: the first 256 validation sequences. Test: the first 1,024 test sequences (262 K tokens). Adaptation and evaluation tokens are disjoint by split | proposed |
| Device | CMO, 2 % P&V (`gaussian_endpoint`), `active_reference_pair`. 256×256 physical arrays, 48 per die: 144 arrays and 3 dies per MLP. No relaxation, no read noise | confirmed |
| Mapping | `per_output_channel_absmax`: each output channel scaled by its own abs-max, with a fixed digital output scale per channel (AIHWKit column-wise; Analog Foundation Models mapping). See "Why per-channel mapping" below | confirmed 2026-10-09 |
| Evaluation defects | `none` plus `random` at 10,000–50,000 ppm in steps of 10,000, i.e. 0, 1, 2, 3, 4 and 5 %. Independent placement (both concentrations `null`). Exact totals per MLP over active and reference devices; random stuck values in 9–88.2 µS | confirmed |
| HWA sources | `pretrained` (method `none`). `standard_hwa`: noise_std 0.03, `output_channel_abs_max`, no corruption. `cdt_random_3pct`: the same noise plus temporary corruption `random` at 30,000 ppm | confirmed |
| HWA noise convention | Analog Foundation Models eq. 3: W̃[j,:] = W[j,:] + 0.03 · maxᵢ\|W[j,i]\| · ε for output channel j, fresh every minibatch. 3 % is a fixed, literature-motivated choice, not tuned | confirmed |
| Training corruption | Each minibatch, every physical device (active and reference) of every pair is stuck i.i.d. with p = 0.03, at a uniform value in 9–88.2 µS, applied after the noise. Temporary; never a deployed fault map. The 3 % noise level and the 3 % training rate are separate parameters | confirmed |
| HWA optimizer | Adam at **5e-3**, teacher-KL objective, no schedule, for **exactly 50 epochs** over the HWA cohort. Batch 32 (8,192 tokens, 256 steps per epoch), seed 41, data seed 20261008. This is a fixed budget: the 5e-3 run was still improving slightly at epoch 50 (see Stage 1a). Both arms use the same minibatch order and noise draws once per minibatch | confirmed 2026-10-09 |
| Checkpoint selection | Both trained arms use the same rule: the minimum mean development teacher KL over `none` and `random` 30,000 ppm, on selection arrays 211001 and 211002 after real CMO P&V. It runs at epoch 0 and every 2 epochs, with no extension. Test data are never used. This rule also chose 5e-3, so "standard HWA" here is robustness-tuned generic-noise HWA, not naive HWA. Selected on clean hardware alone, the best rate would have been 1e-4. A clean-selected reference source was considered and dropped; add it only if a result depends on the distinction | confirmed 2026-10-09 |
| Hardware realizations | Five independent assignments, 271001–271005. Each has its own fault maps and programming noise; the six rates of one assignment are matched conditions on one array identity | proposed |
| On-chip training | None. This is a deployment-only loop (`onchip: null`); a later recovery lifecycle can reuse these deployments | confirmed |
| Programming model | Identical across all sources and evaluations: AIHWKit 1.1.0 CMO 2 % fit, `noise_scale` 1 | confirmed |

**Stage 1a: learning-rate development.**
- Standard HWA only, at Adam 3e-5, 1e-4 and 3e-4 (the `lrdev-e20-*`
  lifecycles), in three lifecycles.
  Each is Experiment 1 with only the `standard_hwa` arm and its candidate
  rate.
- Every section an HWA source depends on matches Experiment 1 (checked by
  digest). The winning run is therefore reused as Experiment 1's standard
  source and is not retrained.
- Only the `prepare`, `devices` and `hwa__standard_hwa` stages of these
  lifecycles run, and their deploy stages are never run. HWA uses only
  training data; selection uses only the development split and the
  selection arrays. Test data and the Experiment 1 assignment arrays are
  therefore untouched until the rate is frozen.
- Selection is the same as above.
- **Rule (proposed):**
  - Freeze the rate with the lowest selected development KL, for both HWA arms.
  - If two rates lie within 1 % (relative), take the smaller.
  - If the winner is at an edge of the grid, add one rate a factor of 3
    beyond that edge before freezing.
  - If every rate selects epoch 0 (no improvement on development), escalate
    instead of continuing.

**Why per-channel mapping.** This was measured on the pinned teacher before
any run. Under per-matrix scaling, a typical output channel's largest weight
is only 9–25 % of its matrix's outlier maximum. Medians over channels, as a
multiple of each channel's weight RMS:

| Error source | Per-matrix scaling | Per-output-channel scaling |
| --- | --- | --- |
| CMO 2 % P&V (pair) | 0.41–0.99 | 0.07–0.11 |
| 3 % per-channel HWA noise | 0.10–0.16 | 0.10–0.16 |
| 1 % random stuck devices | 1.7–4.1 | 0.31–0.46 |
| 5 % random stuck devices | 3.8–9.1 | 0.70–1.02 |

Per-matrix scaling would leave the HWA noise 3–7× below the device error and
would saturate at 1 % stuck devices.

**Known deviations from the Analog Foundation Models recipe (not tested).**
- Our backward pass goes through the noisy weights, as in the AIHWKit modifier
  and Rasch et al. 2023; that paper uses noise-free weights in the backward
  pass.
- No α·std weight clipping; the master is clamped to [-1, 1].
- No DAC/ADC quantization, since our deployment model has none either.
- Training corruption is Bernoulli per device, while evaluation fixes the
  exact count per MLP. At these sizes the difference is negligible.

## Execution contract

**Lifecycles.** All are in
[`campaigns/opt-cmo-corruption/lifecycles/`](../../../lifecycles/README.md), with
short digests from `python -m workflow check`:

- `opt125m-cmo-l3-exp1-probe` (`c4d6416e2da6`): cost probe, not evidence; 32 sequences, one batch;
- `opt125m-cmo-l3-lrdev-e20-lr3e-5` (`440c4626be9a`), `-lr1e-4`
  (`f03d363da0b8`), `-lr3e-4` (`22ac0ec9d2df`) and `-lr1e-3` (`eaa28558f366`,
  the edge extension): Stage 1a, 20 epochs;
- `opt125m-cmo-l3-lrdev-lr1e-5` (`d8535303dc3d`), `-lr3e-5` (`b672026ad6b7`)
  and `-lr1e-4` (`23509070fc7f`): the superseded first Stage 1a, 4 epochs;
- `opt125m-cmo-l3-lrdev-e50-lr5e-3` (`0877aa975ade`): the 50-epoch 5e-3 run;
  its source is L3's `standard_hwa`, reused with matching digests;
- `opt125m-cmo-l3-exp1` (`60ad6b9c1df6`) and `opt125m-cmo-l6-exp1`
  (`20c5f48423d0`): Experiment 1 proper.

`python -m workflow describe` gives each Experiment 1 lifecycle prepare 1,
devices 1, HWA 3, deploy 15 and on-chip 0: 90 deployed cases over
14,155,776 (L3) or 28,311,552 (L6) analog weights. Of each Stage 1a
lifecycle, only prepare, devices and one HWA stage run.

**Placement.** Each comparison stays on one platform.
- HWA trainings run as single stages
  (`python -m ebl train --config <stage>.json` with explicit inputs) in a clean
  clone of the commit. Both trained sources of one network size share one
  environment:
  - L3 on RTX 5090s with PyTorch 2.11: all Stage 1a runs, the 5e-3 standard
    source (`fifi`) and `cdt_random_3pct` (`riri`).
  - L6 on RTX 3090s with PyTorch 2.5.1 (CUDA 12.1): `standard_hwa` on
    `nom-cool-1` and `cdt_random_3pct` on `nom-cool-2`.
- All deployments of both sizes, and the `pretrained` sources, run on Jean
  Zay V100s (`umg@v100`, `qos_gpu-dev`), with
  `pytorch-gpu/py3/2.5.0`, in `$SCRATCH`. `$WORK` is not used: the umg WORK
  inode quota is 97.5 % full.
- HWA checkpoints travel from the 5090 host to Jean Zay through this
  machine.
- Compact deployments are rebuilt by hash on their own platform, so on-chip
  recovery that reuses them must also run on Jean Zay V100s.

**Teacher and corpus.**

```bash
OPT=~/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6
export EBL_CORPUS_ROOT="$PWD/data/corpora"          # wikitext-2-raw-v1.pt
```

**Commands** for each `<id>` in the order probe, Stage 1a, Experiment 1.
Plans for the probe and Stage 1a are written and dry-run checked.

```bash
python -m workflow plan campaigns/opt-cmo-corruption/lifecycles/<id>.json \
  --teacher-weights $OPT/pytorch_model.bin --output results/lifecycles/<id>/plan --device cuda:0
python -m ebl campaign run --manifest results/lifecycles/<id>/plan/campaign.json \
  --output-dir results/lifecycles/<id>/runs --resume [--allow-dirty: probe only]
python -m workflow collect results/lifecycles/<id>/runs --output results/lifecycles/<id>/collected
```

**What changed in code** for this experiment (contract extensions, each tested
in `tests/test_workflow_cmo.py`):

- `hwa.noise_scaling`: `output_channel_abs_max` scales the Gaussian HWA noise
  by each output channel's current abs-max; `matrix_full_scale` keeps one σ
  per matrix.
- Corruption-aware HWA for CMO: temporary i.i.d. stuck active and reference
  devices of each `kind`, with a straight-through gradient.
- `onchip: null`: deployment-only lifecycles. `workflow collect` also writes
  one row per deployed case to `deployments.json`/`.csv`.
- Compact CMO deployments: a programmed-once array is stored as identities
  and rebuilt from its target, with hash verification. Deployment storage per
  case drops from about 12 to about 0 bytes per weight.
- `deployment.scaling: per_output_channel_absmax` (OPT family): per-channel
  targets with a fixed digital output scale per channel.

## Metrics and decision

**Unit.** One assignment × one stuck-device rate. Never pool across rates.
The primary metric is deployed teacher KL on the 1,024 test sequences.
Perplexity is reported too. Rows come from `deployments.csv` (`deployed_*`,
`clean_*`).

**Decision rule (proposed thresholds, frozen before results).**

- **A beats B at rate p** when both hold:
  - A's KL is lower in at least 4 of the 5 assignments;
  - the mean relative KL reduction (B − A)/B is at least 10 %.

  Otherwise there is no material difference at p. Compare corruption-aware
  vs standard HWA, and each trained source vs pretrained.
- **Fidelity cost on healthy hardware:** corruption-aware vs standard HWA at
  0 %, by the same rule.
- **Destructive rate:** a rate where every source's deployed perplexity exceeds
  twice the digital teacher's perplexity in at least 4 of 5 assignments.
  Recovery experiments then use only the non-destructive rates.
- **Carry forward** every source that is not beaten at any rate in 1–5 % by
  another source.

**Validity guards.**
- The `prepare` gate requires digital KL ≤ 1e-6.
- Report each HWA arm's selected epoch and `convergence_review_required`. A
  selected epoch 0 means HWA did not improve on development; report it as such.
- No nonfinite values.
- Fixed budget (changed 2026-10-09). Both trained arms run exactly 50 epochs
  at 5e-3. A source whose selected epoch is the last is reported as
  "fixed budget, still improving", with its last-10-epoch improvement. That
  does not make the experiment inconclusive.
- Inconclusive: fewer than 5 complete assignments for a source and size.
  Partial coverage stays labelled partial.
- L3 and L6 are reported separately and never pooled.

## Budget, stop and expected outputs

- **Targets:** the [standing compute targets](../../../../../docs/experiment_workflow.md#standing-compute-targets)
  of 2026-10-09.
  - HWA trainings on idle RTX 5090 hosts (`trex`, `riri`, `fifi`,
    `loulou`): at most two from 08:00 to 22:00, any idle one overnight.
  - Deployments on Jean Zay V100s with `umg@v100` and `qos_gpu-dev`.
  - All deployments of one comparison run on one GPU type and environment.
- **Compute cap (proposed):** 12 GPU-hours for the probe, Stage 1a and
  Experiment 1 together. Wall-clock time is unknown until the probe runs.
  Escalate if the probe extrapolation exceeds the cap.
- **Storage cap (proposed):** 5 GB. Expected under 3 GB: about 115 MB per
  deployment bundle and 57 MB per HWA source. `/home` has about 26 GB free.
- **Retention:** HWA sources, deployment bundles, stage results and collected
  rows. No per-epoch checkpoints.
- **Stopping:**
  - a failed `prepare` gate;
  - a nonfinite HWA objective;
  - storage over the cap;
  - extrapolated compute over the cap.

  Retries only for understood operational failures, within the budget.
- **Outputs:** `results/lifecycles/<id>/{plan,runs,collected}`. One result
  note, `results/exp-001.md` in this series.

## Execution and monitoring handoff

**State:** running. Execution was authorized on 2026-10-09 for the targets
above.

**Order:**
1. Stage the commit, teacher and corpus on `riri`, `fifi` and Jean Zay
   `$SCRATCH`, with SHA-256 checks and CMO tests on each.
2. Run the probe on `riri` and on one V100 dev job. Record time, memory and
   storage per stage here, and recheck the placement against them.
3. Stage 1a on 5090s, at most two by day. Freeze the learning rate by the
   rule.
4. Write `opt125m-cmo-l3-exp1`. Train `cdt_random_3pct` on a 5090 while
   the `pretrained` and `standard_hwa` deployments run on V100s.
5. Deploy `cdt_random_3pct` on V100s, copy every bundle back, `collect`,
   and write the result note.

**Monitoring owner:** the executing agent; runs are short.

### Current handoff (2026-10-09)

**Staging.** Source commit `1d4f493`, shipped as git bundle
`hwa-1d4f493.bundle` (SHA-256 `d6f517c3…`) and cloned clean on each target.
The teacher (`2d74da66…`) and corpus (`33e95834…`) were verified by SHA-256
on every target.

| Target | Root | Environment | Tests |
| --- | --- | --- | --- |
| `fifi` (RTX 5090) | `~/analog-research/opt-cmo-exp1/` | `~/miniconda3/envs/py312`, PyTorch 2.11.0+cu128 | workflow suites: 136 passed, 2 skipped |
| `riri` (RTX 5090) | same | same. **The clock is about 2 h slow**, so its run timestamps are 2 h early | 136 passed, 2 skipped |
| Jean Zay V100 | `$SCRATCH/opt-cmo-exp1/` = `/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1/` | `module load pytorch-gpu/py3/2.5.0` (PyTorch 2.5.0) | CMO, OPT, lifecycle: 105 passed, 2 skipped |

**Launchers** live outside the clone, in `<root>/launch/`, so the clone stays clean:
- `run_lifecycle.sh` and `run_lifecycle.slurm` run a whole lifecycle;
- `run_stages.py` (`9061d734…`) runs only named stages through
  `python -m ebl train`, resolving inputs from the plan or from completed
  stages;
- `stage1a.sh` (`9f6bc72d…`).

Logs are in `<root>/logs/`. Runs go to
`<root>/hwa-1d4f493/results/lifecycles/<id>/{plan,runs}`.

**Probe on `fifi`** (complete; not evidence). Four stages in 16 s:
- `prepare`: 1.6 s; gate KL −2.9e-9.
- `devices`: 1.1 s.
- `hwa`: 3.6 s for one step at batch 32 and two selection points of about
  1 s each; peak memory 13.4 GB.
- `deploy`: 2.4 s; 2.4 GB.
- Bundles: HWA source 55 MB, deployment 109 MB.

**Probe on Jean Zay** (complete; not evidence).
- Job 822034 failed after 6 s, before any stage, because `git` is missing
  on compute nodes. Both SLURM wrappers now `module load git/2.53.0`
  (`run_lifecycle.slurm` `107d1ba7…`, `run_stages.slurm` `4a201a4b…`).
- Retry job 822625 completed in 1 min 46 s:
  - `prepare`: 9.3 s;
  - `devices`: 8.3 s;
  - `hwa`: 21.1 s and 13.4 GB, about 6× the 5090;
  - `deploy`: 13.5 s and 2.4 GB, about 6× the 5090.
- Extrapolated, a full deployment (clean plus 6 cases on 1,024 test
  sequences) takes about 10 min on a V100, inside the 2 h dev limit.

**Stage 1a, first attempt (superseded).** Four epochs plus a 2-epoch
extension, at 1e-5, 3e-5 and 1e-4. It ran 13:11–13:19 on `fifi` and `riri`,
about 220 s per rate. All three runs completed. Selected development KL:

| Rate | Epoch 0 | 2 | 4 | 6 | Selected |
| --- | --- | --- | --- | --- | --- |
| 1e-5 | 0.5741 | 0.5166 | 0.4967 | 0.4827 | epoch 6 |
| 3e-5 | 0.5741 | 0.4816 | 0.4559 | 0.4383 | epoch 6 |
| 1e-4 | 0.5741 | 0.4352 | 0.4040 | 0.3856 | epoch 6 |

- Every run was still improving at its final epoch and carries
  `convergence_review_required`, which would make Experiment 1
  inconclusive by its own validity rule.
- The gain is almost all at 3 % stuck devices (KL 1.135 → 0.759 at 1e-4);
  at 0 % it is small (0.0132 → 0.0121).
- On 2026-10-09 the user replaced it with 20 epochs and the grid 3e-5, 1e-4,
  3e-4. The 4-epoch sources are not used by Experiment 1.

**Stage 1a, 20 epochs** (commit `76dc3bf`; launched 13:45 on `fifi` and
`riri`). The 1e-4 and 3e-4 runs completed at 14:02, each using all 30
epochs (20 plus the extension) in about 1,010 s. Selected development KL:

| Rate | Epoch 0 | 10 | 20 | 30 | Selected | KL at 0 % / 3 % stuck |
| --- | --- | --- | --- | --- | --- | --- |
| 1e-4 | 0.5741 | 0.3594 | 0.3215 | 0.2938 | epoch 30 | 0.0108 / 0.577 |
| 3e-4 | 0.5741 | 0.2986 | 0.2505 | 0.2216 | epoch 30 | 0.0114 / 0.432 |

- Both runs were still improving and are flagged.
- No overfitting: the held-out development KL fell in step with the training
  loss (0.0231 → 0.0157 at 3e-4), and the 0 % KL did not degrade.
- 3e-4 won at the edge of the grid, so the rule adds 1e-3 at the same
  budget (`lrdev-e20-lr1e-3`, on `riri`). The user confirmed this on
  2026-10-09.
- If 1e-3 is still improving at its last epoch, the remaining choice is a
  longer budget or explicitly accepting fixed-budget sources; that goes back
  to the user.
- 3e-5 runs on `fifi` after 3e-4.

All four 20-epoch rates completed by about 14:25. Each selected its final
epoch, 30, so all are flagged:

| Rate | Epoch 30 dev KL | KL at 0 % / 3 % stuck |
| --- | --- | --- |
| 3e-5 | 0.3611 | 0.0113 / 0.711 |
| 1e-4 | 0.2938 | 0.0108 / 0.577 |
| 3e-4 | 0.2216 | 0.0114 / 0.432 |
| 1e-3 | 0.1453 | 0.0128 / 0.278 |

- At 1e-3 the improvement slows (KL at 3 % dropped 0.093 from epoch 10 to
  20, then 0.055 from 20 to 30) but has not stagnated. Training loss went
  0.0184 → 0.0164 → 0.0142.
- At 1e-3, clean KL first rose (0.0132 → 0.0167 at epoch 2) and recovered
  by about epoch 22. Higher rates trade a little clean fidelity for
  corruption robustness. The selection score is dominated by the 3 % term.
- The user then chose one run at 5e-3 for exactly 50 epochs
  (`lrdev-e50-lr5e-3`, `0877aa975ade`, on `fifi`). It tests whether HWA
  converges within 50 epochs at a higher rate. Two decisions stay open:
  Experiment 1's final rate and budget, and whether standard HWA is
  selected on 0 % only.
