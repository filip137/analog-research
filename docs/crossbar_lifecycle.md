# Crossbar experiment lifecycle

One lifecycle is one reproducible experimental loop. A campaign defines it in a
strict JSON file. That file states the research question, the devices and their
defects, and then four steps: (i) how the digital weights are deployed,
(ii) whether and how HWA training is done, (iii) how program-and-verify (P&V) is
done, and (iv) the on-chip training setup. Each step calls one reusable function
in `methods/`, composed by `ebl/runtime/`. The public `python -m ebl` CLI executes the loop as native run
bundles.

The `network` section selects a **network family**: `opt_mlp_suffix` (OPT-125M
with crossbar MLPs in its last N decoder layers, the current target) or
`cifar_resnet32_suffix` (the ResNet-32 CIFAR reference). The family fixes the
data section, metrics and calibration scheme; the device, P&V and on-chip code
is shared. Changing the number of analog layers, the defect kinds or rates, the
HWA sources or the on-chip arms is a lifecycle setting, never new code.

The historical CIFAR families (`cifar_*`) remain unchanged and reproducible.
New crossbar loops use `crossbar_lifecycle.v1`. The workflow functions reuse the
historical device plants, samplers and update laws, and tests check bit-exact
equivalence with the latest historical loops (see [Equivalence](#equivalence-with-the-historical-families)).

## Where things live

| What | Where |
| --- | --- |
| Lifecycle definition | `campaigns/<campaign>/lifecycles/<lifecycle_id>.json`; the file name equals `lifecycle_id` |
| Decision rule, budget, handoff | the linked experiment or pilot note (unchanged [workflow](../workflow/policy.md)) |
| Measurements and interpretation | the result or pilot note |
| Schema (stdlib only) | `ebl/schema/lifecycle.py` |
| (a) Devices and defects | `hardware/arrays.py`: `fresh_array`, `array_identity`, `CmoArray`; native sampling in `ebl/runtime/device_sampling.py`; CMO device law in `hardware/devices/cmo.py` ([contract](reram_cmo_device.md)) |
| (i) Deployment of digital weights | `networks/deployment.py`: `build_network`, `mapping_receipt`, `targets` |
| (ii) HWA | `methods/hwa.py`: `train_source`, `selection_score` |
| (iii) Program-and-verify | `methods/programming/lifecycle.py`: `fresh`, `program`; characterization in `ebl/runtime/device_sampling.py`, PCM relaxation in `hardware/drift/pcm.py` |
| (iv) On-chip training | `methods/onchip.py`: `train_arm`, `ClosedLoopPulseAdam`, `OpenLoopPulseAdam`, `IdealizedWriteAdam` |
| Network families | `networks/`: `opt.py`, `cifar.py` (one module each, same functions) |
| Cohorts and caches | `networks/data.py`, dispatching to the family |
| One native stage run | `ebl/runtime/lifecycle.py`, reached through `python -m ebl train` |
| Plan, check, collect, corpus | `python -m ebl lifecycle` (`ebl/lifecycle/plan.py`, `ebl/lifecycle/cli.py`) |
| OPT templates | `examples/lifecycles/opt125m-cmo-mlp4-template.json` (CMO, the working device) and `opt125m-om-mlp4-template.json` (OM reference); copy into a campaign; not experiments |

## The definition

Every key is required and unknown keys are rejected. Optional settings are
expressed by method-specific key sets, never by ignored fields. Numbers are
normalized before hashing.

| Section | Fixes |
| --- | --- |
| `question` | `text`, `decision`, `primary_metric` (OPT: `teacher_kl` or `perplexity`, or `task_accuracy_norm_percent`/`task_accuracy_percent` with a declared `task`; CIFAR: `teacher_kl` or `accuracy_percent`), `evidence_class` (`exploratory_model_based`) |
| `network` | OPT: `family`, `model` (`facebook/opt-125m`), `analog_decoder_layers` (1–12, counted from the last layer), an optional `teacher` (`name`, `sha256`) naming a fine-tuned FP32 checkpoint instead of the pinned one, and optional `analog_lora` (`rank`, `alpha`): analog adapter matrices on every analog MLP. CIFAR: `family`, `dataset`, `analog_convolutions` (0, 2, 4 or 8, plus the classifier) |
| `data` | OPT: pinned `corpus` (name and SHA-256), `sequence_length`, split seed, HWA, recovery, development and test sequence counts, `evaluation`, `max_examples`. CIFAR: split seed, HWA, recovery and development image counts, `evaluation`, `max_examples`. `max_examples` is non-zero only for labelled smokes |
| `devices` | technology (`om`, `pcm`, `cmo`) and device model, write-noise scale, array identities and, for OM, population policy, variation and endpoint characterization |
| `defects` | placement, persistence and the defect `cases` |
| `deployment` (i) | scaling, encoding, tile size, MVM read noise, calibration scheme (fixed per family) |
| `hwa` (ii) | noise model (and `noise_std`, `noise_scaling` for CMO), shared optimizer, objective, schedule, epochs, augmentation (CIFAR only), seed and selection rule, plus source `arms` |
| `program_verify` (iii) | method, verify cycles and window (OM), acceptance range (CMO), relaxation |
| `onchip` (iv) | update law, objective, learning rates, epochs, seed, write budget and matched `arms`; or `null` for a deployment-only loop with no on-chip stage |
| `transfer_tests` (optional) | OPT only: extra test cohorts from other pinned corpora, scored at every evaluation. Omitted when absent, so lifecycles without it keep their digests |
| `task` (optional) | OPT only: a labelled multiple-choice task (`name`, pinned `choices` file, `decision` `acc_norm` or `acc`, `development_examples`, `evaluation_examples`) scored at every evaluation; see [Labelled tasks and fine-tuned teachers](#labelled-multiple-choice-tasks-and-fine-tuned-teachers-opt). Omitted when absent |

### (a) Question, devices and defects

`devices.assignment_seeds` name independent array identities. They are deployed
and trained, and results are reported per assignment.
`devices.selection_seeds` name separate arrays, used only to select HWA
checkpoints. The endpoint (programming realization) seed is
`assignment + endpoint_seed_offset`. Seeds must not collide across these roles.
Each analog matrix binds its own population, fault map and programming stream
from `(seed, namespace, matrix)`; the namespace is the OPT model or the CIFAR
dataset, and the matrix is e.g. `decoder.layers.11.fc1`.

A defect case is `{"kind": "none" | "open" | "gmax" | "random", "rate_ppm": n}`.
`none` means `rate_ppm` 0. `rate_ppm` is the corrupted fraction of devices
(20000 ppm = 2 %), applied independently to every analog matrix. A sweep over
corruption fractions is a list of cases in one lifecycle. All cases of one assignment share that array's
identity, so they are matched conditions, not independent arrays. Defects are
`permanent`: the runtime verifies after every epoch that writes did not change
them. Placement depends on the technology.

CMO (IBM CMO/HfOx ReRAM, the working device for OPT; [device contract](reram_cmo_device.md))
uses `hierarchical_die_array` placement:

- **Arrays and dies:** a physical array is `tile_size` × `tile_size`
  devices. A weight's active and reference devices sit on adjacent rows of
  one array, so each deployment tile spans two arrays.
  `defects.arrays_per_die` consecutive arrays of one MLP form a die, and dies
  never straddle MLPs.
- **Case format:** every case also carries `die_concentration` and
  `array_concentration`, either `null` (no variation at that level) or a
  positive Beta concentration:
  `{"kind": "random", "rate_ppm": 50000, "die_concentration": 118, "array_concentration": null}`.
- **Rates:** die rates are drawn from Beta(μκ_d, (1−μ)κ_d), and array rates
  from Beta(p_d κ_a, (1−p_d) κ_a) around their die's rate.
- **Exact totals:** exactly `rate_ppm`/10⁶ of each MLP's physical devices
  fail, active and reference alike. They are allocated to arrays in
  proportion to the array rates.
- **Kinds:** `open` (0 µS), `gmin`, `gmax` or `random` (uniform in range,
  the default).
- **Labels:** a case's label appends `_d<κ_d>` and `_a<κ_a>` for the finite
  levels.

| | OM (`aihwkit_1.1.0_reram_array_om`) | PCM (`gaussian_endpoint_li2023`) | CMO (`aihwkit_1.1.0_reram_cmo`) |
| --- | --- | --- | --- |
| Encoding | one active cell, `q=a-r` with its intrinsic reference | 25 uS differential pair | an active device in 9–88.2 uS and a reference device at 48.6 uS per weight, `q=a-r` (`active_reference_pair`) |
| Defect placement | Bernoulli per active cell; stuck at a fraction of the cell's own bounds | Bernoulli per physical device (either or both banks); stuck at 0, 25 uS or uniform | `hierarchical_die_array`: Beta die and array rates, exact total per MLP; stuck at 0 µS, G_min, G_max or uniform |
| Populations | literal AIHWKit 1.1.0 cells, repaired baseline (`EBL_AIHWKIT_PYTHON`) | analytic | analytic |
| HWA noise model | `om_endpoint_kernel`: residual tables of the declared P&V law on independent characterization cells | `pcm_gaussian_endpoint` | `additive_gaussian_weight`: Gaussian weight noise of `hwa.noise_std` per matrix or per output channel; temporary stuck active/reference devices for corruption-aware arms |
| P&V methods | `closed_loop`, `open_loop_nominal` | `gaussian_endpoint` | `gaussian_endpoint` with `acceptance_percent` 2.0 (or 0.2) |
| Relaxation | `none` (OM retention and drift are not modelled) | `none` or `pcm_drift` | `none` in deployment; the optional frozen-source `retention` branch models CMO relaxation and read noise |
| On-chip laws | `om_closed_loop_pulse_adam`, `om_open_loop_pulse_adam` | `pcm_endpoint_reprogramming` | `cmo_idealized_write_adam` |

These are hardware-derived fitted models and presets. Label results as
exploratory model-based evidence, never as fabricated-hardware measurements
([synapse-data rules](synapse_data.md)).

### (i) Deployment of the digital weights

The pretrained teacher stays complete; everything outside the analog matrices is
frozen and digital. Each analog matrix is flattened to `[out, in]` and scaled
to targets `q` in `[-1, 1]`, in one of two ways:

- `per_layer_absmax` divides the matrix by its own absolute maximum, with one
  digital output scale per matrix.
- `per_output_channel_absmax` (OPT family only) divides each output channel
  by its own absolute maximum, with a fixed digital output scale per channel.
  This is the AIHWKit column-wise convention, and it keeps outlier weights
  from compressing every other channel into a small part of the conductance
  range.

Each matrix is then split into `tile_size` tiles whose input slices are summed
digitally. `read_noise` adds per-tile MVM output noise. Without read noise, the
OPT family computes each matrix as one matmul, which equals the tiled sum up to
FP32 summation order. Each stage writes
`mapping.json`, which records tiles, scales, the frozen-parameter hash and the
physical device count.

| | `opt_mlp_suffix` | `cifar_resnet32_suffix` |
| --- | --- | --- |
| Teacher | `facebook/opt-125m` `pytorch_model.bin` at revision `27dcfa74`, or the declared fine-tuned `network.teacher` (same layout, pinned by its own SHA-256), loaded as FP32 by the native decoder in `networks/opt.py` (tested against Hugging Face) | pinned public ResNet-32 checkpoint |
| Analog matrices | `fc1` (768→3072) and `fc2` (3072→768) of the last `analog_decoder_layers` decoder layers; 4,718,592 weights per layer | last `analog_convolutions` layer3 convolutions plus the classifier |
| Digital and frozen | embeddings, earlier decoder layers, all attention, LayerNorms, tied LM head | prefix convolutions and BN |
| Calibration (`calibration` arms) | per-matrix output gains, the MLP biases and the LayerNorm in front of each analog MLP (`output_gain_mlp_layer_norm_mlp_bias`) | output gains, BN affine of converted blocks, classifier bias |
| Data | non-overlapping sequences of one pre-tokenized corpus (below) | class-balanced 45000/5000 CIFAR split |
| Metrics | token-weighted `teacher_kl`, `cross_entropy`, `perplexity`, next-token accuracy and teacher agreement over positions 0..T-2 of each sequence, with `scored_tokens`; with a `task`, also multiple-choice metrics on `task_development` and `task_test` (below) | `teacher_kl`, `accuracy_percent`, teacher agreement |
| `prepare` gate | the clean digital suffix must reproduce the teacher (KL ≤ 1e-6), and with a `task` also its continuation-token and choice KL | full test split must reproduce the published accuracy |

OPT data: `python -m ebl lifecycle corpus` tokenizes train, validation and test text
files with the pinned tokenizer (one BOS at the start of each split) into
`$EBL_CORPUS_ROOT/<name>.pt`, and prints the SHA-256 the lifecycle must
declare. HWA and recovery sequences are nested prefixes of a seeded shuffle of
the train blocks; development and test sequences are the first blocks of the
validation and test splits. The frozen prefix and teacher logits are recomputed
per minibatch, because per-token teacher distributions (50,272 classes) are too
large to cache. The corpus itself is an Explorer choice, recorded in the note.

#### Labelled multiple-choice tasks and fine-tuned teachers (OPT)

Added for exp-018 (user-approved October 15, 2026).

**Task data.** `python -m ebl lifecycle task-data` (`networks/tasks.py`) reads one
pinned task source: the official train and validation parquet files of
HellaSwag, SST-2 or PIQA.
- It applies lm-eval-harness v0.4 templates and `_encode_pair` encoding. Each
  scored row also starts with BOS.
- It writes two files into the corpus directory:
  - a `crossbar_lifecycle.choice_task.v1` file holding every example's context,
    its K continuations, the label and the continuation character counts;
  - a `token_corpus.v1` stream of BOS + context + gold continuation per example.
- Splits:
  - `test` is the official validation split.
  - `development` is a seeded hold-out from train, disjoint by group (HellaSwag
    source, SST-2 sentence, PIQA goal).
  - The remaining train examples form the stream's train split.
- The command prints both SHA-256 digests, the split counts and an overlap
  receipt.
- The stream is the lifecycle's `data.corpus`; HWA and recovery use its train
  blocks only.

**Task cohorts and metrics.** A `task` section adds `task_development` and
`task_test` cohorts: the first examples of the development and test splits.
Every evaluation (deploy, on-chip epochs, final, replay) scores them on the held
apparent state, through the same prefix, teacher and `forward_features` path:
- Rows are right-padded, which is safe under the causal mask.
- Each continuation's log-probability is summed in float64.
- Metrics:
  - `accuracy_percent`: argmax of the sums.
  - `accuracy_norm_percent`: sums divided by character count, as lm-eval does.
  - `choice_kl` and `choice_kl_norm`: KL against the teacher over the softmax of
    the raw or normalised choice scores.
  - teacher agreement.
  - continuation-token `teacher_kl`.
  - a per-example `predictions` string, for paired tests.
- On-chip development checks also score `task_development`.
- HWA checkpoint selection still uses stream teacher KL.
- `collect` adds `<state>_task_<split>_<metric>` columns.

**Fine-tuned teachers.** `network.teacher` declares a fine-tuned FP32
checkpoint; `load_teacher` and `plan` then require its SHA-256 instead of the
pinned one. The teacher is part of the `network` section, so every artifact
identity includes it, while fault maps and endpoints stay bound to the model.
Such teachers are made by the native experiment `opt_task_teacher.v1`
(`methods/teacher_lora.py`):
- LoRA on the declared linear layers of every decoder layer.
- Loss: multiple-choice cross entropy over the summed continuation
  log-likelihoods.
- Selection uses development data only.
- The selected adapters are merged into FP32 and exported as a Hugging Face
  `pytorch_model.bin`.
- Export gates:
  - the reload reproduces the merged tensors and logits bitwise;
  - the merged and adapter forwards agree;
  - non-adapted parameters are unchanged.
- The run writes `teacher_provenance.json`.

LoRA only prepares the teacher; it is never a deployed digital adapter here.

#### Analog LoRA adapters on a frozen HWA base (OPT)

Added for exp-019 (user-approved October 15, 2026). `network.analog_lora`
(`rank`, `alpha`; per-output-channel scaling only) gives each base matrix W two
further analog matrices: A (rank × in) and B (out × rank), named `<W>.lora_a`
and `<W>.lora_b`.
- **Placement.** They follow the base matrices in the flat target vector `q`,
  so P&V, fault maps, fault rewrites and evaluation treat them as arrays.
- **Computation.** The adapter path is x → A → B, digitised between the arrays
  and scaled by α/r. It is added in logical units before the frozen output gain
  and bias.
- **Initial state.** Adapters start at zero with unit scales, an exact no-op,
  so the `prepare` gate still holds.
- **Fault groups.** Each adapted matrix's A and B form their own fault group, so
  base fault maps are unchanged by adding adapters.
- **Targeted faults.** A defect case may set `target` (`base` or `adapter`; the
  default is all arrays; the label gains `_base` or `_adapter`). Faults then go
  only into that subset's groups, exactly as in the untargeted case.

**LoRA HWA arms.** An HWA arm with `method: lora` trains only the adapters on a
frozen whole-model HWA source. The source comes from another lifecycle and is
pinned by `base_source {name, sha256}`; plan it with
`--lora-base ARM=PATH`.
- The base's analog targets, digital copy and calibration are copied in and
  checked unchanged.
- Adapter masters (logical units; A uniform ±1/√in, B = 0) train with Adam at
  the arm's `learning_rate`, with linear warm-up over `warmup_fraction` and
  cosine decay to zero.
- Each step maps the masters to normalised targets, using their current
  output-channel abs-max as the digital scale, and adds the declared CMO noise
  to every analog weight.
- The objective `hwa.objective: task_choice_token_mean` minimises the task's
  multiple-choice cross entropy over per-token mean continuation
  log-likelihoods, on `task.training_examples` labelled train examples.
- Checkpoints, from epoch 0 (the base without adapters), are selected by
  development decision accuracy on the selection arrays after real P&V.
- A LoRA source's provenance includes the `task` section.

### (ii) HWA: if, and how noise- or corruption-aware

Each arm is a source that later stages deploy:

```json
{"id": "digital", "method": "none"}
{"id": "standard_hwa", "method": "hwa", "noise_strength": 1.0, "corruption": null,
 "selection_cases": [{"kind": "none", "rate_ppm": 0}]}
{"id": "cdt_gmax", "method": "hwa", "noise_strength": 1.0,
 "corruption": {"kind": "gmax", "rates_ppm": [20000, 30000, 50000]},
 "selection_cases": [{"kind": "none", "rate_ppm": 0}, {"kind": "gmax", "rate_ppm": 50000}]}
```

- `method: none` deploys the digital teacher unchanged.
- `method: hwa` trains an FP32 digital master off-chip: the normalized weights
  plus the calibration parameters. It trains on apparent hardware views drawn
  fresh every minibatch, with a straight-through gradient.
- `noise_strength` sets how noise-aware the training is. It scales the
  technology's HWA noise model; 0 trains on clean digital weights.
  - PCM and OM use their endpoint models.
  - CMO uses `additive_gaussian_weight`: q + `noise_std`·`noise_strength`·s·ξ.
    `hwa.noise_scaling` sets s:
    - `matrix_full_scale`: s = 1, the matrix's abs-max after per-layer
      scaling;
    - `output_channel_abs_max`: s is each output channel's current
      abs-max, maxᵢ|q[j,i]|, as in eq. 3 of the Analog Foundation Models
      paper.

    `hwa.noise_std` and `hwa.noise_scaling` are required for CMO and
    rejected otherwise.
- `corruption` sets corruption awareness. It adds temporary failure masks at a
  rate drawn per minibatch from `rates_ppm`, and those masks are never the
  deployed fault map. For CMO, the active and the reference device of every
  pair are stuck i.i.d. at that rate, with the `kind`'s stuck value, after the
  weight noise.
- Checkpoint selection takes the minimum mean development teacher KL over the
  arm's `selection_cases`, measured on the selection arrays after the
  lifecycle's own P&V, including relaxation. Selection runs at epoch 0 and every
  `selection.every_epochs`. If the best epoch falls in the final 20 %, training
  is extended once by `selection.extension_epochs`; if it is still late, the
  source records `convergence_review_required` and later stages propagate the
  flag. Final arrays and test data are never used for selection.
- `lr_schedule` (optional) fixes the learning-rate schedule in optimizer
  updates rather than epochs. Omitting it preserves the historical
  `lr_milestones`/`lr_decay` behavior and all existing source digests. When it is
  present, set `lr_milestones: []`, `lr_decay: 1` and
  `selection.extension_epochs: 0`; the training budget cannot extend.

  ```json
  {"kind": "linear_warmup_decay", "total_steps": 12800, "warmup_steps": 205}
  ```

  The same factor scales every optimizer group's base rate (including
  `digital_learning_rate`) **before** one-based update s. For warm-up length W,
  updates 1 through W use s/W. `linear_warmup_decay` then uses (T−s)/(T−W),
  reaching exactly zero on update T. `warmup_step_decay` additionally requires
  `milestone_step` M and `decay_factor` d: it holds the peak through update M,
  then uses d for updates M+1 through T. Require 0 ≤ W < T, W < M < T and
  0 < d ≤ 1. `cosine_warmup_decay` instead requires `min_factor` m, a finite
  number with 0 ≤ m ≤ 1. After warm-up, its factor is
  m + (1−m) × (1 + cos(π × (s−W)/(T−W))) / 2. The peak is reached at update W;
  the last update T uses exactly m times each group's base rate. For example,
  peak LR 5e-3 and `min_factor: 0.2` give a final LR of 1e-3. The floor applies
  to the decay phase; warm-up starts at 1/W of the base rate. The cosine form
  does not accept step-only `milestone_step` or `decay_factor` fields, and the
  older forms do not accept `min_factor`.

  ```json
  {"kind": "cosine_warmup_decay", "total_steps": 25600, "warmup_steps": 205, "min_factor": 0.2}
  ```

  An explicit W=0 disables warm-up: the logged step-0 factor is one, and the
  first actual update already follows the decay curve (below the peak for
  linear decay and cosine decay with m < 1).

  `total_steps` must equal `epochs` × ceil(effective examples per epoch /
  `batch_size`), including partial minibatches and any `sequences_per_epoch`
  or `data.max_examples` cap. Diagnostic smokes must declare their own shorter
  schedule; the runtime never rescales or truncates a production schedule.
  It also checks the loaded HWA cohort size before training. Metric records
  expose `global_step` and actual group `learning_rates` at ordinary cadence,
  selection and schedule boundaries. The fit records the resolved schedule,
  completed updates, base rates and final rates. Nonfinite rejection retains
  its actual completed-update count and remains an invalid source.
- `scope` (optional, OPT family only) sets which parameters an arm trains.
  - `analog` (the default, and omitted from the arm) trains the analog
    weights and their calibration parameters: the analog layers' MLP biases,
    the LayerNorm in front of each analog MLP, and per-matrix output gains.
    The frozen teacher supplies every other digital parameter.
  - `model` also trains every digital parameter of the student's own copy of
    the network: embeddings, the earlier layers, all attention and LayerNorms,
    and the tied head. Both reference papers fine-tune the whole model this
    way. It requires a nonnegative `digital_learning_rate` (Adam, the same
    `hwa` schedule). Zero is a diagnostic endpoint: it retains the model-scope
    forward/backward path and optimizer group while leaving digital weights
    unchanged. The runtime verifies this and records initial/selected digital
    hashes; analog weights and calibration still train at `hwa.learning_rate`.
    Numerical equality and byte equality are reported separately: zero-rate Adam
    may change the sign bit of a zero without changing its numerical value.
  - Only the analog weights see noise and corruption, and the frozen teacher
    remains the KL target.
  - A `model`-scope source carries its digital weights through deployment and
    on-chip stages. On-chip arms never train them. Deployment bundles of such
    sources are about 4 bytes per digital parameter larger (about 0.4 GB for
    OPT-125M).
- `sequences_per_epoch` (optional, OPT family only) streams a large HWA cohort.
  Without it an epoch is one pass over `data.hwa_sequences`. With it, epoch e
  takes stream positions [(e−1)·S, e·S) of an endless series of seeded passes
  (pass k is a permutation seeded by `data.data_seed + 1,000,000 + k`), so every
  sequence appears once per pass and a pass boundary may fall inside an epoch.
  Selection still runs every `selection.every_epochs`, so a fixed step budget
  keeps its selection cadence whatever the cohort size. The fit records the
  number of passes. Leaving it out keeps a lifecycle's digests unchanged.

### (iii) Program-and-verify: cycles and relaxation

- `closed_loop` runs at most `max_cycles` read–verify–pulse cycles. Each cycle
  reads every apparent state and pulses each cell outside
  `±tolerance_steps × dw_min` once toward its target. The report records
  accepted and exhausted cells, verify reads, pulses and residuals.
- `open_loop_nominal` derives pulse counts from RESET using the nominal
  soft-bounds inverse. It uses no verify reads and no device identities.
- `gaussian_endpoint` makes one endpoint draw per device and request.
  - PCM: permanent failures stay failed.
  - CMO: G = G\* + σ(G\*)·ξ with the fitted σ of the declared
    `acceptance_percent`. The draw is not truncated to the acceptance window
    and not clamped to 9–88.2 µS; the report counts `out_of_range_devices`.
- `relaxation: {"model": "pcm_drift", "seconds": t, "compensation": bool}`
  applies the AIHWKit 1.1.0 PCM drift law with 1/f read noise at time `t`, with
  optional per-tile global drift compensation. The law is the same one used in
  `experiments/cifar_crossbar/pcm_reference.py`.
  - Drift coefficients use each device's programming target.
  - Each programming event, including every on-chip rewrite, produces one held
    relaxed state.
  - Every later forward uses that state: gradients, selection and evaluation.
  - Stuck devices keep their failed conductance.
  - HWA noise models programming error, not drift.

The same P&V function characterizes the OM endpoint tables for HWA. The HWA
noise model therefore matches the declared deployment law. A failed held-out
adequacy gate fails the `devices` stage.

### (iv) On-chip training

An arm combines `weights` (`hold`, `rewrite` or `learn`) with `calibration`
(bool). The parser enforces two rules. Every `rewrite` or `learn` arm needs a
`hold` arm with the same calibration flag; that is its matched no-weight-write
control. Arms must have unique settings. The usual five controls are:

| Arm | weights | calibration | Meaning |
| --- | --- | --- | --- |
| `none` | hold | false | deployment only, no training |
| `calibration` | hold | true | digital calibration, no weight writes |
| `rewrite` | rewrite | true | frozen targets rewritten on the same schedule |
| `onchip_weights` | learn | false | array-specific weight learning |
| `onchip_calibration` | learn | true | joint learning |

Gradients and optimizer state are digital. Every forward uses the held
post-write apparent state. The update laws are:

- `pcm_endpoint_reprogramming` reprograms a clamped digital master after every
  update.
- `om_closed_loop_pulse_adam` accumulates an Adam target and verifies with up to
  `pulses_per_update` pulses inside `tolerance_steps`.
- `om_open_loop_pulse_adam` applies stochastic one-pulse rounding with no
  verify reads, and so admits no verified rewrite control.
- `cmo_idealized_write_adam` writes the digital Adam step directly into the
  held state, once per active device and step: G_a ← G_a + ΔG + σ(G_a+ΔG)·ξ
  on every healthy active device with ΔG ≠ 0. Reference devices are never
  written.
  - There is no master, no verify read and no retry, so write errors persist.
  - Its `rewrite` arm reprograms the active devices to the frozen targets
    every step.
  - Costs: `device_writes`, `device_programs`, `touched_devices`,
    `max_onchip_device_writes`, `verify_reads` (0) and `out_of_range_devices`.

`max_pulses_per_cell` caps OM on-chip writes per cell; `null` means uncapped. Every
arm starts from the same P0, model, minibatch order and seeds. Each final
state is saved and replayed exactly. The persistent OM state is reported only
as a named diagnostic.

Optional `onchip.training_cohort` selects `recovery` (the existing default) or
`hwa` (the already prepared HWA training cohort). Both use the pinned training
split; validation and test cohorts cannot be selected. This permits a full-corpus
adaptation pass with an unchanged frozen HWA source and feature cache. The
`onchip_cohort` metric names the actual cohort and example count, and presentation
counts use that cohort. Omission preserves historical lifecycle identities.

CMO also supports `cmo_thresholded_write_adam`, with required positive
`threshold_sigma` and Boolean `accumulate_updates`. The physical write law is
unchanged. Digital Adam commands, in normalized q units, are issued only when
their magnitude exceeds `threshold_sigma * noise_scale * sigma_PV(48.6) / 39.6`.
This is a fixed **nominal mid-range** noise reference, not an oracle estimate
of each cell's realized error or its conductance-dependent noise. With
`accumulate_updates: true`, signed commands accumulate digitally between writes;
the complete accumulated command is issued at threshold crossing and subtracted
from the accumulator. Opposite commands cancel. Issued attempts are subtracted
even on stuck cells, without querying the hidden fault map. No subthreshold
residual is flushed at the endpoint. Checkpoints retain pending commands and
threshold settings. Physical costs retain the existing device-write and
stuck-attempt definitions. No verification, retry or conductance clamp is added.

Optional `onchip.max_write_attempts_per_device` adds a nonnegative integer cap
to the thresholded law (omitted or `null` means uncapped). Each active device has
a digital int32 count, initially zero at adaptation, incremented for every issued
command including stuck attempts. Initial programming is excluded; reference
devices receive no adaptation writes. A cell at its cap receives no further
commands or pending accumulation, while other cells continue training. No hidden
fault identities or successful-write counters enter this decision. Cap zero is
a no-write control. Capped configurations reject rewrite arms, whose separate
programming schedule is not controlled by this law. Counts are checkpointed;
metrics report total and maximum issued attempts and the number of capped cells.
Physical successful writes are checked against the per-cell attempt counts and
the declared cap. This is a cap on idealized write events, not measured pulses.

Optional `onchip.write_noise_scale` (CMO laws only, a nonnegative number)
multiplies the noise of each on-chip write: G_a ← G_a + ΔG + f·noise_scale·σ(G_a+ΔG)·ξ.
Omitted, the physical law (f = 1) and every existing lifecycle digest and
checkpoint are unchanged. f = 0 is an exact-write oracle for diagnosing write
noise; it is not a physical method. It leaves deployment programming noise,
`devices.noise_scale`, the nominal threshold reference, stuck devices, the held
state's device contract and all cost counters unchanged; the noise stream is
still drawn, so it advances identically. Lifecycles that declare it reject
`rewrite` arms, whose programming noise it does not cover. Writer checkpoints
and thresholded diagnostics record the declared value.

Optional per-arm `fault_rewrite` (CMO only) applies one fault-aware rewrite of
the held deployment at the start of a `hold` or `learn` arm, without calibration
or a digital adapter; see the [CMO device contract](reram_cmo_device.md):

```json
{"id": "pair_rewrite_verify", "weights": "hold", "calibration": false,
 "fault_rewrite": {"detection": "verify", "roles": "pair", "seed": 241016, "threshold_sigma": 5.0}}
```

- `detection: verify` reads every device once (two verify reads per weight). It
  flags a device whose read lies more than `threshold_sigma` programming sigmas
  from its target and uses that read as the stuck value. `detection: oracle`
  takes the hidden fault map instead; it is a non-physical bound and declares no
  threshold.
- `roles: active` lets a healthy active device absorb a stuck reference;
  `roles: pair` also lets a healthy reference absorb a stuck active device.
  Weights with both devices flagged stay as read.
- Partner devices are reprogrammed once with the deployment endpoint law. The
  noise is drawn from per-matrix streams bound to `seed` and the assignment, so
  arms sharing a seed receive the same draw for a device. `seed` must differ
  from every array identity, deployment endpoint and the on-chip seed.
- Arms with a rewrite are distinct settings, and each needs the plain
  `hold` arm with the same calibration flag. A `hold` arm with a rewrite
  does not train. A `learn` arm learns from the rewritten state. Its costs,
  checkpoints and any `max_write_attempts_per_device` exclude the rewrite,
  which is a programming event like deployment.
- Lifecycles that declare `write_noise_scale` reject fault rewrites.
- Each row records the receipt under `fault_rewrite`:
  - controller decisions, flag counts, clipped targets and a plan digest;
  - the rewrite cost: programs, stuck attempts and verify reads;
  - out-of-range devices and the rewritten state's hash;
  - analysis-side counts against the hidden map: stuck devices by role, false
    positives and negatives. These are read only after programming.
- Omitting `fault_rewrite` leaves every existing lifecycle digest and
  checkpoint unchanged.

Optional positive `onchip.development_every_steps` records held-state evaluation
on the fixed development cohort at that optimizer-step interval. It never
selects a checkpoint or changes the fixed final endpoint; test cohorts are not
evaluated at this extra cadence. Omitting it preserves existing configs.

Optional per-arm `write_roles: "verified_partner"` extends thresholded CMO
learning after a `fault_rewrite` with `detection: verify`, `roles: pair`.
Omission preserves the legacy active-only law and serialized configuration.
The original verify scan supplies a frozen int8 role per weight: 0 writes the
active if unflagged, 1 writes its reference if only the active is flagged, and
-1 blocks a pair with both flagged. The map never reads hidden fault identities
or triggers a second scan. A logical command delta-q writes +39.6 delta-q to
an active or -39.6 delta-q to a reference, with the unchanged unclamped noisy
incremental law. One draw per weight preserves the active-only noise stream.
Unselected partners remain unchanged; missed faults consume attempts without
changing state. Both-flagged pairs accumulate no pending commands or attempts.

The cap starts after rewriting and belongs to the one selected physical device
per weight. Physical writes, attempts and touched counts report active/reference
contributions. Checkpoints retain the role map, its digest, optimizer history,
reference-write counters and RNG state. Older checkpoints retain their active
counters and load unchanged. Reference counters count incremental writes only;
P&V program costs stay in their separate receipt. Routing-map bytes are reported
as controller storage, not learned parameters. Native final-state replay remains
mandatory; interrupted-stage resume is still unsupported.

## Stages, artifacts and reuse

### Frozen-source retention sweeps (OPT CMO sources)

An optional `retention` section creates an inference-only branch after HWA:

```json
{"models": ["reram_cmo", "pcm", "reram_wan2022"],
 "seconds": [1, 86400, 172800], "assignment_seeds": [271002],
 "case_labels": ["nominal", "random_30000ppm"], "seed": 391013,
 "read_noise": false}
```

Require `onchip: null`, OPT CMO sources with nominal programming noise, no P&V
relaxation and zero per-MVM read noise. Assignments and cases select subsets
of the unchanged source lifecycle. A `retention__<arm>__<assignment>` stage
replaces deployment and consumes the same teacher, cache, device bundle and
HWA-source inputs. The optional section never changes upstream provenance.
`workflow plan --hwa-source ARM=/absolute/source.pt` substitutes an explicit
frozen source and omits its HWA stage; repeat for multiple sources. The plan
records the file hash and the runtime retains the normal source-provenance,
completion and teacher/mapping gates.

Each frozen normalized weight is remapped into the declared model: CMO uses
active/reference pairs over 9–88.199997 uS (2% or 0.2% programming fit as declared
by the source); PCM uses positive/negative pairs over 0–25 uS; Wan-2022 uses
positive/negative pairs over 0–40 uS. All use two physical devices per weight.
CMO hierarchical fault locations stay matched across the three models. Random
stuck values retain their uniform quantile within each model's range; open,
gmin and gmax map to the corresponding target-model bounds (open is zero).
This is transfer of the frozen source to separately modelled technologies,
not aging one physical CMO array with interchangeable laws.

Programming draws happen once per model, matrix and assignment, with the
existing endpoint seed binding. PCM drift coefficients are sampled once per
device. Every requested age starts from that programming baseline, with
age-specific noise streams and held apparent weights throughout evaluation.
Stuck devices retain their conductance. The `read_noise` flag enables the
separate PCM/CMO read-noise terms only; Wan supplies no separate fit. Drift
compensation and additional writes are absent. CMO/Wan snapshots describe
marginal age distributions, not a measured joint time process.

Wan supports exactly 1, 86400 and 172800 seconds. Its programming baseline is
the 1-second fit, reused exactly for the 1-second row; later snapshots sample
total error about the target without adding programming noise a second time.
An optional `"wan_noise_scales": [1, 1.5, 2, ...]` (finite, increasing, including 1)
multiplies Wan's fitted endpoint sigma for every aged Wan snapshot. The 1-second
programming baseline is never scaled. All scales reuse that age's normal draw,
so they are matched realizations and scale 1 reproduces the unscaled snapshot
exactly. Scaled rows carry `noise_scale` and the snapshot label `t<seconds>s-x<scale>`.
Multiples above one are synthetic stress tests beyond the published fit, not a
device model. The field is omitted from the section at its default `[1]`.
CMO supports ages >=1 second in this sweep. PCM baseline means the programmed
state without invoking its zero-time read-noise function. CMO clamps only the
positive-age snapshots, as in AIHWKit. Saved state includes targets, programmed
conductances, faults, PCM coefficients, seeds, encoding and source identities;
native execution verifies serialized baseline and per-age replay. As with the
existing endpoint arrays, RNG replay requires the same execution-device type
and compatible PyTorch environment.

`ebl lifecycle collect` exports `retention.json` and `.csv`, with each source,
assignment, case, device model and snapshot separate. Rows include baseline,
clean and aged metrics, physical counts, zero retention writes/verifies,
apparent-state hashes and residuals. Readiness smokes remain labelled as such.
This branch does not enable CMO drift during HWA selection or on-chip training.

An additional optional `drift_compensation` section replaces the retention stage
with a native `compensate` stage that consumes a **completed** retention state
index via `workflow plan --retention-source ARM:SEED=/absolute/retention_states.json`.
It requires unchanged parent scientific sections and `retention.read_noise: false`.
Its strict fields are `methods`, `calibration_sequences`, `row_column_iterations`
and `gain_bounds`. Methods are `global`, `tile_global`, `row`, `column`,
`row_column`, `global_shift` and `global_moments`. Calibration sequences are the
first declared count from the training cohort; held-out data is never fitted.
Global/tile/column/input-row gains use baseline-to-aged mean-absolute one-hot
MVM responses. Combined row/column gains use fixed-iteration diagonal least
squares. The last two methods match output means or means/variances after global
gain, using programmed-array reference statistics. OPT has LayerNorm, so these
are affine activation corrections, not BatchNorm-running-statistics updates.

Digital input/output/per-tile operators act around the unchanged held apparent
MVM. Native parent hashes, same-array snapshots, calibration serialization,
one exact uncompensated control and frozen source parameters are checked.
`ebl lifecycle collect` emits `compensation.json` and `.csv`: absolute metrics,
aging KL regained (undefined for an aging gap <=1e-6), required calibration reads,
reference storage and digital parameter counts. No physical writes occur.

### Recovery context views (OPT)

Optional top-level `recovery_views` keeps frozen HWA/deployment data unchanged
while varying on-chip context. Its strict fields are `context_length`,
`parent_length`, `parent_sequences`, `score_stride`, and `evaluation_lengths`.
Complete parent windows from the pinned train stream are selected using
`data.data_seed`; epoch order uses `onchip.seed + epoch - 1`. Children of a
parent remain adjacent and an effective batch must contain whole parents.
The loss excludes the last predictor position of each `score_stride` block at
every context. Context must divide parent length and score stride must divide
context. Explicit view cohorts require `data.max_examples: 0`; probes declare
their own small counts. The view's parent count defines recovery exposure.

Existing evaluation cohorts retain their original context. Each additional
evaluation length reshapes the identical token prefix into a separately named
`context<L>_<split>` cohort; counts must divide exactly. Ordinary unmasked
next-token metrics apply to evaluation. Views bind feature-cache provenance,
but never alter the dependencies of frozen HWA sources or P0. Legacy documents
retain their original digests and reject a view-specific cache.

Optional `onchip.microbatch_size` divides the effective OPT batch. Gradients are
weighted by sequence count (equal scored positions per sequence) and accumulated;
Adam and the physical writer act once per effective batch. Rows record input
tokens, scored tokens and optimizer updates for context views. Optional
`onchip.case_labels` selects a nonempty subset of the already deployed defect
cases, preserving complete P0 provenance and reporting only that subset as
on-chip coverage. Neither field changes frozen source identity.

### Digital LoRA recovery controls

For `opt_mlp_suffix`, an on-chip hold arm may declare
`digital_adapter: {"kind": "lora", "rank": 16, "alpha": 16, "learning_rate": 0.001}`.
It must use `weights: "hold"`, `calibration: false`, and have a matched unadapted
hold arm. Only digital low-rank branches on the existing analog fc1/fc2 matrices
are trained. A is initialized uniformly at +/-1/sqrt(fan_in), B to zero, using
`onchip.seed`; alpha/r scaling, no dropout or weight decay, constant Adam LR.
Corrections use logical weight units before the fixed output gain and bias.
Every base parameter and the held array are verified unchanged. Native states
store adapter tensors separately, replay exactly, and report adapter parameters,
extra MACs/token and zero adaptation writes. These branches remain digital at
inference; merging them into an array would be a separate intervention.

`python -m ebl.runtime.portable_deployment` expands an existing compact CMO P0 on
its original GPU into held tensors, binding the new artifact to its parent hash.
It preserves the original bundle and checks exact apparent-state equality;
this is a serialization conversion, not a new endpoint draw. Portable states
stay explicit when saved again, avoiding cross-GPU RNG reconstruction.

A recovery-only native campaign may contain one onchip stage with four explicit
frozen inputs (`teacher_weights`, `device_data`, `device_model`, `device_state`).
Use `python -m ebl.execution.recovery_collect validate RUN_ROOT --inputs INPUTS.json`
and its `collect` command for this manifest shape. The input contract pins all
four hashes, array identity and apparent-state hashes; the collector checks
native inspection, lineage, trajectory coverage, presentation count, exact replay
and LoRA's immutable physical/base state before exporting native collected rows.

| Stage | Runs | Inputs | Artifact kinds |
| --- | --- | --- | --- |
| `prepare` | 1 | teacher (+ the declared corpus for OPT) | `feature_cache` (CIFAR features or OPT token cohorts), `mapping` |
| `devices` | 1 | teacher | `device_bundle` (OM populations and endpoint tables; PCM is analytic) |
| `hwa__<arm>` | per HWA arm | teacher, `feature_cache`, `device_bundle` | `hwa_source` |
| `deploy__<arm>__<seed>` | per arm × assignment | + `hwa_source` | `deployment`: P0 for every defect case |
| `onchip__<arm>__<seed>` | per arm × assignment | + `deployment` | `measurements`, `onchip_states`, one `onchip_state.<case>.<arm>` per trajectory |

With `onchip: null` the loop stops after `deploy`. A later lifecycle with the
same upstream sections can reuse those deployments as its on-chip inputs.

Each stage is one `ebl train` run with exactly the inputs listed. Artifact
kinds are unique, so the existing campaign runner chains stages by kind.

An upstream artifact is accepted when the lifecycle sections it depends on are
unchanged. The question text, unrelated arms and later stages may differ, so a
new lifecycle can reuse, for example, existing HWA sources with new on-chip
arms:

| Artifact | Depends on |
| --- | --- |
| `feature_cache` | `network`, `data`, and `transfer_tests` and `task` when declared |
| `device_bundle` | `network`, `devices`, `program_verify` |
| `hwa_source` | `network`, `data`, `devices`, `deployment`, `program_verify`, shared `hwa` settings and its own arm |
| `deployment` | all of the above for its arm, plus `defects`, and `transfer_tests` and `task` when declared |

`transfer_tests` never determines a source, so adding a second test domain
reuses existing sources. Each entry names a `label`, a pinned `corpus` (name
and SHA-256), its `split` (`validation` or `test`, never `train`) and a sequence
count. The first sequences of that split are scored next to the lifecycle's own
evaluation split, under the cache and metric key `transfer_<label>`. Collected
rows gain `clean_`, `deployed_`, `initial_` and `final_transfer_<label>_<metric>`
columns. The lifecycle's own validation split cannot be reused as a transfer
cohort. Overlap with another corpus's training text must be checked and recorded
in the experiment note.

Teacher and mapping hashes must also match.

For a frozen-checkpoint deployment campaign, keep the source-dependent lifecycle
sections unchanged, replace each deploy stage's `weights` dependency with the
explicit saved `hwa_source` path, and omit the HWA stage from the execution
manifest. Keep the new prepare and devices stages and every declared assignment;
adding `transfer_tests` requires a new feature cache but no new training.
`python -m ebl.execution.lifecycle_collect validate RUN_ROOT --deployment-only`
checks this manifest shape, the original source bundle and its full artifact
hashes, compatible source provenance, exact input lineage, and complete deployment
coverage. The same explicit mode is available for `export` and `collect`; it does
not relax the default full-lifecycle or source-only collection gates.

Storage: every trajectory keeps its final state for exact replay. For the
CIFAR suffix with four analog convolutions that is about 5 MB (PCM) or 10 MB (OM)
per trajectory. OPT OM states take roughly 70–90 bytes per analog weight
(measured on a tiny run), so about 0.3–0.4 GB per analog decoder layer and
trajectory; the OM template's 30 trajectories at four layers need roughly 45 GB.
CMO states measured 24–32.3 tensor bytes per analog weight on a tiny OPT run,
about 0.15 GB per analog decoder layer and trajectory.
Add one P0 bundle per HWA arm and assignment, and the OM populations of every
assignment and selection array in the device bundle.

A CMO array that was programmed once and never written is stored compactly.
The state holds its fault and apparent-state hashes, and it is rebuilt by
reprogramming the saved target on the same execution-device type, then
verified. CMO P0 bundles therefore hold only the model and target, about
8 bytes per weight, and arms that never write add almost nothing. `python -m ebl lifecycle
describe` prints the analog weight count and this final-state estimate
(trajectories × weights × 90 bytes for OM, 35 for PCM and CMO). Budget it before launch,
and measure memory, throughput and actual sizes on a smoke run with one or two
analog layers before scaling the number of layers.

## Running a lifecycle

Planning and checking launch nothing. Execution follows the [workflow
policy](../workflow/policy.md): make cases, target, budget, duration and output
paths visible before substantial runs.

```bash
export EBL_AIHWKIT_PYTHON=/path/to/aihwkit-1.1.0/bin/python   # OM only
export EBL_CORPUS_ROOT="$PWD/data/corpora"                    # OPT only
export EBL_CIFAR_ROOT="$PWD/artifacts/cifar_inputs/data"      # CIFAR only
OPT=~/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6
# OPT, once per corpus: prints the name and SHA-256 for data.corpus
python -m ebl lifecycle corpus --name <corpus> --model-dir $OPT \
  --train train.txt --validation validation.txt --test test.txt
# OPT labelled task: prints the task file and stream names and SHA-256 for task.choices and data.corpus
python -m ebl lifecycle task-data --task hellaswag --name <task> --model-dir $OPT \
  --train train.parquet --validation validation.parquet --development-examples 2000 --split-seed <seed>
# OPT fine-tuned teacher (opt_task_teacher.v1 config); declare its merged SHA-256 in network.teacher
python -m ebl train --config teacher.json --output-dir results/teachers/<id> \
  --teacher-weights $OPT/pytorch_model.bin --device-data $EBL_CORPUS_ROOT/<task>.pt
python -m ebl lifecycle check
python -m ebl lifecycle describe campaigns/<campaign>/lifecycles/<id>.json
python -m ebl lifecycle plan campaigns/<campaign>/lifecycles/<id>.json \
  --teacher-weights $OPT/pytorch_model.bin \
  --output results/lifecycles/<id>/plan [--device cuda:0] [--cpu-threads 4]
python -m ebl campaign run --manifest results/lifecycles/<id>/plan/campaign.json \
  --output-dir results/lifecycles/<id>/runs [--dry-run] [--resume] [--allow-dirty]
python -m ebl runs inspect RUN_DIR --verify-artifacts --require-complete
python -m ebl lifecycle collect results/lifecycles/<id>/runs --output results/lifecycles/<id>/collected
```

For CIFAR, pass `artifacts/cifar_inputs/cifar10_resnet32-ef93fc4d.pt` as the
teacher. `plan` refuses a teacher whose SHA-256 differs from the pinned one.

`plan` writes one stage config per stage. Each embeds the normalized lifecycle
and its execution settings, so each bundle's `config.resolved.json` holds every
setting. `plan` also writes a runner manifest and a summary. A single stage can
run directly with `python -m ebl train --config <stage>.json` and the listed
inputs. `collect` flattens the on-chip measurements into one row per
assignment, case and arm (`rows.json`/`.csv`). It also writes the deployments,
one row per HWA source, assignment and case, with clean and deployed metrics
(`deployments.json`/`.csv`). It flags duplicated coverage from retries and
never pools it.

## Equivalence with the historical families

On CPU, `tests/test_workflow_stages.py` checks the following against the
historical code, bit for bit:

- PCM fresh arrays and reprogramming against `LayeredPcmArray`;
- OM fresh arrays with closed-loop P&V against the V2 sweep `make_array`;
- open-loop programming against the nominal-reset path;
- OM endpoint tables against `build_kernels`;
- the PCM drift law against `PcmInferenceNoise`;
- the pulse writers against `PulseWriter`, `UncappedClosedLoopAdam` and
  `UncappedOpenLoopAdam`;
- every training arm, PCM and OM, against `full_epoch_runtime.trajectory`;
- a noise-aware HWA arm against the V2 sweep `fit_candidate`, covering selection
  history, the selected source and the trained master.

`tests/test_workflow_runtime.py` runs complete PCM, PCM-drift and OM lifecycles
through the runtime on synthetic inputs. It validates every bundle against the
campaign runner's result contract, and checks section-based reuse and
rejection. These tests do not establish CUDA or cross-version bitwise
equivalence.

The OPT family reuses the same device, P&V, HWA and on-chip code unchanged.
`tests/test_workflow_opt.py` checks the native decoder against Hugging Face
`OPTForCausalLM` on a tiny random model; that the clean crossbar suffix
reproduces the teacher for every number of analog layers; that training never
changes the frozen teacher; the OPT schema rules and cohorts; and a complete OM
lifecycle on a tiny OPT with all five on-chip arms, exact replay and `collect`.
The decoder was also checked once against Hugging Face on the real OPT-125M
weights (maximum logit difference 1e-5, identical top-1 tokens).

CMO has no historical loop to match bit for bit. Instead,
`tests/test_workflow_cmo.py` checks:

- the σ coefficients against AIHWKit 1.1.0 `ReRamCMONoiseModel`;
- the active-minus-reference read;
- unclamped, untruncated and held P&V endpoints;
- the exact idealized write law and the persistence of write errors;
- state replay;
- the constant-σ HWA weight noise;
- the schema rejections;
- a complete tiny-OPT CMO lifecycle with all five arms, exact replay and
  `collect`.

## Not yet supported

- OPT models other than `facebook/opt-125m`; analog attention projections;
  analog MLPs that are not the last contiguous decoder layers.
- Mean defect rates that differ between analog matrices or layers. CMO
  varies rates across dies and arrays around one declared mean.
- Dense CIFAR-10 and MNIST networks.
- Hyperparameter grids inside one arm; select them with a development lifecycle
  and freeze the choice.
- Resume of interrupted stages.
- OM relaxation, and drift-aware HWA noise.
- CMO drift, the CMO read-noise term, hierarchical (die/array) corruption
  during HWA, row- or column-structured faults inside an array, and reassignment of fault
  patterns to other layers; see the [CMO device contract](reram_cmo_device.md).
- The legacy PCM pulse plant.
- Repeated endpoint realizations of one assignment.
- Per-epoch on-chip checkpoints; only final states are kept.
- Labelled tasks other than HellaSwag, SST-2 and PIQA; generative, few-shot or
  variable-K tasks; a `task` together with retention, drift compensation or
  recovery views; masked (answer-only) objectives. Task LoRA as a deployed
  digital adapter is a separate, escalated regime.
- Analog LoRA with per-layer scaling, retention, recovery views or digital
  adapters; LoRA arms beside other trained HWA arms; on-chip learning tested on
  adapter arrays.
- Device-independent PCM and CMO endpoint draws. As in the historical
  `GaussianEndpointArray`, programming noise (and CMO write noise) uses the
  execution device's generator, so CPU and CUDA runs of one lifecycle realize
  different endpoints. Fault maps and OM trajectories (counter-keyed per cell)
  are device-independent.

## Extension points

For a new experiment, check every choice in the experiment note against this
document. If all are supported, the Codifier writes the lifecycle JSON, then
runs `workflow plan` and `ebl campaign run`; no code changes. Otherwise the gap
is one of these points. Extend it there, add a contract or equivalence test,
then write the lifecycle:

| Missing | Where to extend |
| --- | --- |
| network family or model (e.g. another OPT size, analog attention) | a module in `networks/` with the functions listed in its `__init__.py`; `network` parsing and `PRIMARY_METRICS`/`CALIBRATIONS` in `networks/config.py` |
| data or metric | the family module (`build_cache`, `evaluate`) and its `data` parser |
| labelled task or template | `TASKS` in `networks/tasks.py`, `TASK_NAMES` in `networks/config.py` |
| teacher fine-tuning recipe | `methods/teacher_lora.py` (`opt_task_teacher.v1`) |
| analog adapters, targeted faults, adapter training | `OptMlpSuffix` (`analog_lora`), `hardware/arrays.py` (`target`), `methods/hwa.py` (`train_lora`) |
| defect kind or placement | `FAILURE_KINDS` and `_defects` in `hardware/config.py`; fault construction in `hardware/arrays.py` |
| technology, P&V method or relaxation | `TECHNOLOGIES` in `hardware/config.py`; `hardware/arrays.py`, `methods/programming/lifecycle.py` |
| on-chip update law | `TECHNOLOGIES[...].update_laws`; `make_writer` in `methods/onchip.py` |

Never add a per-study runtime or bypass the stages for one experiment.

### Custom drift recovery execution

The optional `drift_recovery` branch uses the shared writable custom drift engine
and a frozen HWA source. `curve_validation_sequences` optionally selects a fixed
prefix for trajectory measurements while the full `validation_sequences` cohort
still selects calibration and learning rate. `heldout_evaluation: "final"` limits
transfer cohorts to each trajectory endpoint; its durability report uses validation.
Omitting these options retains the original evaluation schedule and config digest.

Optional `execution` declares `phase` (`prepare`, `screen`, or `trajectory`), a
deterministic work-unit `key`, the completed `prepared_run` path, and completed
`screen_runs`. All units use native `ebl` bundles. Preparation owns the shared FP32
reference caches, masks, calibration selection and fresh baselines. Consumers bind
the scientific settings, source, assignment, token identities and preparation hash;
trajectory execution requires complete, nonduplicate LR-screen coverage. Whole
trajectories are independently executable, with the same numerical path as a full
run. This supports continuation between completed units, not resumption midway
through an interrupted optimizer event.

### Scalar power-law retention control

For a synthetic, conductance-independent drift control, `retention.power_law`
declares exactly `mean_us`, increasing nonnegative `sigmas_us`, positive
`mean_exponent`, `spread_exponent`, `reference_seconds`, positive
`development_sequences`, and `first_read_policy: "clamp_programmed"`.
This currently requires `models: ["reram_cmo"]`, `case_labels: ["nominal"]`,
read noise off, and no drift compensation. It preserves upstream HWA/cache
provenance and evaluates the first declared number of development sequences.
Its sigma list is crossed with `retention.seconds`, plus one first-read row.

The existing native CMO programming draw is evaluated separately, then clamped
once into the native window to define G_p. This explicitly declared admission
differs from legacy unclamped CMO programming; counts and raw-programming metrics
are reported. Every age evaluates clip(G_p + m u^alpha + s u^beta epsilon),
u=t/t_ref, using the same saved per-device epsilon for all times and amplitudes.
The state stores G_p and epsilon explicitly, and verifies zero-time identity and
replay. There are no defects, calibration or additional writes. This is a
synthetic assumed temporal coupling, not a measured device time process.

FP32 clean post-HWA and digital-teacher logits are cached once on the exact token
cohort, retained as hashed native artifacts, and reused for all rows. Outputs add
clean-HWA KL, spread amplitude, perplexity ratio to first read, and clipping/drift
statistics. `ebl lifecycle collect` retains each time/sigma row separately.

### Initial programming with a fitted Flowers model

The optional `initial_programming` section runs one preparation policy per native
stage and realization, then evaluates passive retention. Its strict settings are
defined by `methods.programming.config.InitialProgramming`. It requires
a pinned frozen OPT/CMO HWA source, a pinned Flowers JSON model and token-cohort
hash, only the nominal defect case, no other recovery branch and no extra read
noise. Upstream source/cache provenance is unchanged when this optional branch
is absent or added. The legacy CMO endpoint law is bypassed explicitly: achieved
first-read anchors equal requested feasible targets.

Policies P0–P6 implement immediate release, waiting, mean prebias, one selective
pair-rewrite pass, their combination, matched random rewrites and high common-mode
mapping. Each full rewrite resets only touched cells and draws fresh randomness.
The Gaussian-copula Markov kernel preserves the supplied cross-age law; the NIG
transform applies to the total residual. A shared transition grid and named
streams preserve matched untouched trajectories across policies. Read snapshots
are held across tokens. Observations are explicitly clipped after sampling;
clipping never modifies latent history. All positive read ages must be 3–6242 s.

The native stage records initial/aged PPL, original-teacher references, clipped
and raw diagnostics by starting-conductance bin, real preparation clocks and
full-program/read counts. Final apparent conductances and selected pair masks
are saved and replayed exactly. Mid-trajectory resume is unsupported. The
`initial_programming_measurements`, `initial_programming_summary`,
`initial_programming_state` and `flowers_model` artifacts are collected through
the standard workflow. `ebl lifecycle collect` exports one row per policy,
realization and age, flagging duplicate coverage. This branch does not assert
measured programming timing, pulse cost or physical equivalence of populations.
