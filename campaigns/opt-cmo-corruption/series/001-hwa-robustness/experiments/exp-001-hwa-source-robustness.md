---
id: "exp-001"
title: "Experiment 1: standard versus corruption-aware HWA under 0-5 % stuck CMO devices"
status: "planned"
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
| Network | `opt_mlp_suffix`, `facebook/opt-125m` at revision `27dcfa74`. fc1/fc2 of decoder layers 9–11 on CMO arrays: 14,155,776 weights, 28,311,552 devices | confirmed |
| Corpus | `wikitext-2-raw-v1`, SHA-256 `33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01`. Built from Hugging Face `Salesforce/wikitext` at revision `b08601e04326c79dfdd32d625aee71d232d685c3`, the raw split lines concatenated. Tokens: 2,391,885 train, 247,290 validation, 283,288 test | confirmed |
| Metric | `teacher_kl` primary; perplexity, cross-entropy, next-token accuracy and teacher agreement also reported | confirmed |
| Sequences and cohorts | Length 256. HWA: 8,192 train sequences (2.10 M tokens). Development: the first 256 validation sequences. Test: the first 1,024 test sequences (262 K tokens). Adaptation and evaluation tokens are disjoint by split | proposed |
| Device | CMO, 2 % P&V (`gaussian_endpoint`), `active_reference_pair`. 256×256 physical arrays, 48 per die: 144 arrays and 3 dies per MLP. No relaxation, no read noise | confirmed |
| Evaluation defects | `none` plus `random` at 10,000–50,000 ppm in steps of 10,000, i.e. 0, 1, 2, 3, 4 and 5 %. Independent placement (both concentrations `null`). Exact totals per MLP over active and reference devices; random stuck values in 9–88.2 µS | confirmed |
| HWA sources | `pretrained` (method `none`). `standard_hwa`: noise_std 0.03, `output_channel_abs_max`, no corruption. `cdt_random_3pct`: the same noise plus temporary corruption `random` at 30,000 ppm | confirmed |
| HWA noise convention | Analog Foundation Models eq. 3: W̃[j,:] = W[j,:] + 0.03 · maxᵢ\|W[j,i]\| · ε for output channel j, fresh every minibatch. 3 % is a fixed, literature-motivated choice, not tuned | confirmed |
| Training corruption | Each minibatch, every physical device (active and reference) of every pair is stuck i.i.d. with p = 0.03, at a uniform value in 9–88.2 µS, applied after the noise. Temporary; never a deployed fault map. The 3 % noise level and the 3 % training rate are separate parameters | confirmed |
| HWA optimizer | Adam, teacher-KL objective, learning rate frozen by Stage 1a, no schedule. 4 epochs over the HWA cohort, batch 8, seed 41, data seed 20261008. Both arms use the same minibatch order | rate: confirmed approach; epochs, batch: proposed |
| Checkpoint selection | Both trained arms use the same rule: the minimum mean development teacher KL over `none` and `random` 30,000 ppm, on selection arrays 211001 and 211002 after real CMO P&V. It runs at epoch 0 and every epoch, with one 2-epoch extension if the best falls in the final 20 %. Test data are never used | proposed |
| Hardware realizations | Five independent assignments, 271001–271005. Each has its own fault maps and programming noise; the six rates of one assignment are matched conditions on one array identity | proposed |
| On-chip training | None. This is a deployment-only loop (`onchip: null`); a later recovery lifecycle can reuse these deployments | confirmed |
| Programming model | Identical across all sources and evaluations: AIHWKit 1.1.0 CMO 2 % fit, `noise_scale` 1 | confirmed |

**Stage 1a: learning-rate development.**
- Standard HWA only, at Adam 1e-5, 3e-5 and 1e-4, in three otherwise
  identical lifecycles.
- They use the development split only (`evaluation: development`) and their
  own array 281001.
- Selection is the same as above.
- **Rule (proposed):**
  - Freeze the rate with the lowest selected development KL, for both HWA arms.
  - If two rates lie within 1 % (relative), take the smaller.
  - If the winner is at an edge of the grid, add one rate a factor of 3
    beyond that edge before freezing.
  - If every rate selects epoch 0 (no improvement on development), escalate
    instead of continuing.

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

- `opt125m-cmo-l3-exp1-probe` (`63bce8b3f44a`): cost probe, not evidence;
- `opt125m-cmo-l3-lrdev-lr1e-5` (`741d68311e81`);
- `opt125m-cmo-l3-lrdev-lr3e-5` (`7db8226a5d3f`);
- `opt125m-cmo-l3-lrdev-lr1e-4` (`3ba018b76fbb`);
- `opt125m-cmo-l3-exp1`: written after Stage 1a, with every field above and
  the frozen learning rate.

`python -m workflow describe` on the Experiment 1 draft gives these stage
runs: prepare 1, devices 1, HWA 3, deploy 15, on-chip 0. That is 90 deployed
cases over 14,155,776 analog weights. Each Stage 1a lifecycle has prepare 1,
devices 1, HWA 1 and deploy 1.

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
- Inconclusive: a flagged selection, or fewer than 5 complete assignments.
  Partial coverage stays labelled partial.

## Budget, stop and expected outputs

- **Target (proposed, needs your authorization):** the local RTX 3090
  (`cuda:0`). It is currently shared with another job at about 40 %
  utilization, and using it would be GPU sharing.
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

**State:** planned. No stage has run. The probe and the three Stage 1a plans
are written under `results/lifecycles/<id>/plan` and dry-run checked.

**Next actions:**
1. Your confirmation of the proposed fields, and authorization to use the
   shared local GPU.
2. Run the probe and record time, memory and storage per stage here.
3. Run Stage 1a and freeze the learning rate.
4. Write and plan `opt125m-cmo-l3-exp1` and launch it in a persistent
   session.

**Monitoring owner:** the executing agent; runs are short.
