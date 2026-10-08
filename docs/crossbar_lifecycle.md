# Crossbar experiment lifecycle

One lifecycle is one reproducible experimental loop. A campaign defines it in a
strict JSON file. That file states the research question, the devices and their
defects, and then four steps: (i) how the digital weights are deployed,
(ii) whether and how HWA training is done, (iii) how program-and-verify (P&V) is
done, and (iv) the on-chip training setup. Each step calls one reusable function
in `workflow/`. The public `python -m ebl` CLI executes the loop as native run
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
| Decision rule, budget, handoff | the linked experiment or pilot note (unchanged [workflow](experiment_workflow.md)) |
| Measurements and interpretation | the result or pilot note |
| Schema (stdlib only) | `workflow/lifecycle.py` |
| (a) Devices and defects | `workflow/devices.py`: `fresh_array`, `sample_populations`, `array_identity` |
| (i) Deployment of digital weights | `workflow/deployment.py`: `build_network`, `mapping_receipt`, `targets` |
| (ii) HWA | `workflow/hwa.py`: `train_source`, `selection_score` |
| (iii) Program-and-verify | `workflow/program_verify.py`: `fresh`, `program`, `characterize_endpoints`, `PcmDrift` |
| (iv) On-chip training | `workflow/onchip.py`: `train_arm`, `ClosedLoopPulseAdam`, `OpenLoopPulseAdam` |
| Network families | `workflow/networks/`: `opt_mlp.py`, `cifar_resnet32.py` (one module each, same functions) |
| Cohorts and caches | `workflow/data.py`, dispatching to the family |
| One native stage run | `workflow/runtime.py`, reached through `python -m ebl train` |
| Plan, check, collect, corpus | `python -m workflow` (`workflow/plan.py`, `workflow/__main__.py`) |
| OPT template | `examples/lifecycles/opt125m-om-mlp4-template.json` (copy into a campaign; not an experiment) |

## The definition

Every key is required and unknown keys are rejected. Optional settings are
expressed by method-specific key sets, never by ignored fields. Numbers are
normalized before hashing.

| Section | Fixes |
| --- | --- |
| `question` | `text`, `decision`, `primary_metric` (OPT: `teacher_kl` or `perplexity`; CIFAR: `teacher_kl` or `accuracy_percent`), `evidence_class` (`exploratory_model_based`) |
| `network` | OPT: `family`, `model` (`facebook/opt-125m`), `analog_decoder_layers` (1–12, counted from the last layer). CIFAR: `family`, `dataset`, `analog_convolutions` (0, 2, 4 or 8, plus the classifier) |
| `data` | OPT: pinned `corpus` (name and SHA-256), `sequence_length`, split seed, HWA, recovery, development and test sequence counts, `evaluation`, `max_examples`. CIFAR: split seed, HWA, recovery and development image counts, `evaluation`, `max_examples`. `max_examples` is non-zero only for labelled smokes |
| `devices` | technology and device model, write-noise scale, array identities and, for OM, population policy, variation and endpoint characterization |
| `defects` | placement, persistence and the defect `cases` |
| `deployment` (i) | scaling, encoding, tile size, MVM read noise, calibration scheme (fixed per family) |
| `hwa` (ii) | shared optimizer, objective, schedule, epochs, augmentation (CIFAR only), seed and selection rule, plus source `arms` |
| `program_verify` (iii) | method, verify cycles and window, relaxation |
| `onchip` (iv) | update law, objective, learning rates, epochs, seed, write budget and matched `arms` |

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
them. Placement depends on the technology:

| | OM (`aihwkit_1.1.0_reram_array_om`) | PCM (`gaussian_endpoint_li2023`) |
| --- | --- | --- |
| Encoding | one active cell, `q=a-r` with its intrinsic reference | 25 uS differential pair |
| Defect placement | Bernoulli per active cell; stuck at a fraction of the cell's own bounds | Bernoulli per physical device (either or both banks); stuck at 0, 25 uS or uniform |
| Populations | literal AIHWKit 1.1.0 cells, repaired baseline (`EBL_AIHWKIT_PYTHON`) | analytic |
| HWA noise model | `om_endpoint_kernel`: residual tables of the declared P&V law on independent characterization cells | `pcm_gaussian_endpoint` |
| P&V methods | `closed_loop`, `open_loop_nominal` | `gaussian_endpoint` |
| Relaxation | `none` (OM retention and drift are not modelled) | `none` or `pcm_drift` |
| On-chip laws | `om_closed_loop_pulse_adam`, `om_open_loop_pulse_adam` | `pcm_endpoint_reprogramming` |

These are hardware-derived fitted models and presets. Label results as
exploratory model-based evidence, never as fabricated-hardware measurements
([synapse-data rules](synapse_data.md)).

### (i) Deployment of the digital weights

The pretrained teacher stays complete; everything outside the analog matrices is
frozen and digital. Each analog matrix is flattened to `[out, in]` and divided
by its own absolute maximum (`per_layer_absmax`), giving targets `q` in
`[-1, 1]`. It is then split into `tile_size` tiles whose input slices are summed
digitally. `read_noise` adds per-tile MVM output noise. Each stage writes
`mapping.json`, which records tiles, scales, the frozen-parameter hash and the
physical device count.

| | `opt_mlp_suffix` | `cifar_resnet32_suffix` |
| --- | --- | --- |
| Teacher | `facebook/opt-125m` `pytorch_model.bin` at revision `27dcfa74`, loaded as FP32 by the native decoder in `workflow/networks/opt_mlp.py` (tested against Hugging Face) | pinned public ResNet-32 checkpoint |
| Analog matrices | `fc1` (768→3072) and `fc2` (3072→768) of the last `analog_decoder_layers` decoder layers; 4,718,592 weights per layer | last `analog_convolutions` layer3 convolutions plus the classifier |
| Digital and frozen | embeddings, earlier decoder layers, all attention, LayerNorms, tied LM head | prefix convolutions and BN |
| Calibration (`calibration` arms) | per-matrix output gains, the MLP biases and the LayerNorm in front of each analog MLP (`output_gain_mlp_layer_norm_mlp_bias`) | output gains, BN affine of converted blocks, classifier bias |
| Data | non-overlapping sequences of one pre-tokenized corpus (below) | class-balanced 45000/5000 CIFAR split |
| Metrics | token-weighted `teacher_kl`, `cross_entropy`, `perplexity`, next-token accuracy and teacher agreement over positions 0..T-2 of each sequence, with `scored_tokens` | `teacher_kl`, `accuracy_percent`, teacher agreement |
| `prepare` gate | the clean digital suffix must reproduce the teacher (KL ≤ 1e-6) | full test split must reproduce the published accuracy |

OPT data: `python -m workflow corpus` tokenizes train, validation and test text
files with the pinned tokenizer (one BOS at the start of each split) into
`$EBL_CORPUS_ROOT/<name>.pt`, and prints the SHA-256 the lifecycle must
declare. HWA and recovery sequences are nested prefixes of a seeded shuffle of
the train blocks; development and test sequences are the first blocks of the
validation and test splits. The frozen prefix and teacher logits are recomputed
per minibatch, because per-token teacher distributions (50,272 classes) are too
large to cache. The corpus itself is an Explorer choice, recorded in the note.

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
- `noise_strength` sets how noise-aware the training is. It scales the endpoint
  model; 0 trains on clean digital weights.
- `corruption` sets corruption awareness. It adds temporary failure masks at a
  rate drawn per minibatch from `rates_ppm`, and those masks are never the
  deployed fault map.
- Checkpoint selection takes the minimum mean development teacher KL over the
  arm's `selection_cases`, measured on the selection arrays after the
  lifecycle's own P&V, including relaxation. Selection runs at epoch 0 and every
  `selection.every_epochs`. If the best epoch falls in the final 20 %, training
  is extended once by `selection.extension_epochs`; if it is still late, the
  source records `convergence_review_required` and later stages propagate the
  flag. Final arrays and test data are never used for selection.

### (iii) Program-and-verify: cycles and relaxation

- `closed_loop` runs at most `max_cycles` read–verify–pulse cycles. Each cycle
  reads every apparent state and pulses each cell outside
  `±tolerance_steps × dw_min` once toward its target. The report records
  accepted and exhausted cells, verify reads, pulses and residuals.
- `open_loop_nominal` derives pulse counts from RESET using the nominal
  soft-bounds inverse. It uses no verify reads and no device identities.
- `gaussian_endpoint` makes one PCM endpoint draw per request. Permanent
  failures stay failed.
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

`max_pulses_per_cell` caps on-chip writes per cell; `null` means uncapped. Every
arm starts from the same P0, model, minibatch order and seeds. Each final
state is saved and replayed exactly. The persistent OM state is reported only
as a named diagnostic.

## Stages, artifacts and reuse

| Stage | Runs | Inputs | Artifact kinds |
| --- | --- | --- | --- |
| `prepare` | 1 | teacher (+ the declared corpus for OPT) | `feature_cache` (CIFAR features or OPT token cohorts), `mapping` |
| `devices` | 1 | teacher | `device_bundle` (OM populations and endpoint tables; PCM is analytic) |
| `hwa__<arm>` | per HWA arm | teacher, `feature_cache`, `device_bundle` | `hwa_source` |
| `deploy__<arm>__<seed>` | per arm × assignment | + `hwa_source` | `deployment`: P0 for every defect case |
| `onchip__<arm>__<seed>` | per arm × assignment | + `deployment` | `measurements`, `onchip_states`, one `onchip_state.<case>.<arm>` per trajectory |

Each stage is one `ebl train` run with exactly the inputs listed. Artifact
kinds are unique, so the existing campaign runner chains stages by kind.

An upstream artifact is accepted when the lifecycle sections it depends on are
unchanged. The question text, unrelated arms and later stages may differ, so a
new lifecycle can reuse, for example, existing HWA sources with new on-chip
arms:

| Artifact | Depends on |
| --- | --- |
| `feature_cache` | `network`, `data` |
| `device_bundle` | `network`, `devices`, `program_verify` |
| `hwa_source` | `network`, `data`, `devices`, `deployment`, `program_verify`, shared `hwa` settings and its own arm |
| `deployment` | all of the above for its arm, plus `defects` |

Teacher and mapping hashes must also match.

Storage: every trajectory keeps its final state for exact replay. For the
CIFAR suffix with four analog convolutions that is about 5 MB (PCM) or 10 MB (OM)
per trajectory. OPT OM states take roughly 70–90 bytes per analog weight
(measured on a tiny run), so about 0.3–0.4 GB per analog decoder layer and
trajectory; the template's 30 trajectories at four layers need roughly 45 GB.
Add one P0 bundle per HWA arm and assignment, and the OM populations of every
assignment and selection array in the device bundle. `python -m workflow
describe` prints the analog weight count and this final-state estimate
(trajectories × weights × 90 bytes for OM, 35 for PCM). Budget it before launch,
and measure memory, throughput and actual sizes on a smoke run with one or two
analog layers before scaling the number of layers.

## Running a lifecycle

Planning and checking launch nothing. Execution follows the [workflow
policy](experiment_workflow.md): make cases, target, budget, duration and output
paths visible before substantial runs.

```bash
export EBL_AIHWKIT_PYTHON=/path/to/aihwkit-1.1.0/bin/python   # OM only
export EBL_CORPUS_ROOT="$PWD/data/corpora"                    # OPT only
export EBL_CIFAR_ROOT="$PWD/artifacts/cifar_inputs/data"      # CIFAR only
OPT=~/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6
# OPT, once per corpus: prints the name and SHA-256 for data.corpus
python -m workflow corpus --name <corpus> --model-dir $OPT \
  --train train.txt --validation validation.txt --test test.txt
python -m workflow check
python -m workflow describe campaigns/<campaign>/lifecycles/<id>.json
python -m workflow plan campaigns/<campaign>/lifecycles/<id>.json \
  --teacher-weights $OPT/pytorch_model.bin \
  --output results/lifecycles/<id>/plan [--device cuda:0] [--cpu-threads 4]
python -m ebl campaign run --manifest results/lifecycles/<id>/plan/campaign.json \
  --output-dir results/lifecycles/<id>/runs [--dry-run] [--resume] [--allow-dirty]
python -m ebl runs inspect RUN_DIR --verify-artifacts --require-complete
python -m workflow collect results/lifecycles/<id>/runs --output results/lifecycles/<id>/collected
```

For CIFAR, pass `artifacts/cifar_inputs/cifar10_resnet32-ef93fc4d.pt` as the
teacher. `plan` refuses a teacher whose SHA-256 differs from the pinned one.

`plan` writes one stage config per stage. Each embeds the normalized lifecycle
and its execution settings, so each bundle's `config.resolved.json` holds every
setting. `plan` also writes a runner manifest and a summary. A single stage can
run directly with `python -m ebl train --config <stage>.json` and the listed
inputs. `collect` flattens the on-chip measurements into one row per
assignment, case and arm. It flags duplicated coverage from retries and never
pools it.

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

## Not yet supported

- OPT models other than `facebook/opt-125m`; analog attention projections;
  analog MLPs that are not the last contiguous decoder layers.
- Defect rates that differ between analog matrices or layers.
- Dense CIFAR-10 and MNIST networks.
- Hyperparameter grids inside one arm; select them with a development lifecycle
  and freeze the choice.
- Resume of interrupted stages.
- OM relaxation, and drift-aware HWA noise.
- The legacy PCM pulse plant.
- Repeated endpoint realizations of one assignment.
- Per-epoch on-chip checkpoints; only final states are kept.
- Device-independent PCM endpoint draws. As in the historical
  `GaussianEndpointArray`, programming noise uses the execution device's
  generator, so CPU and CUDA runs of one lifecycle realize different PCM
  endpoints. Fault maps and OM trajectories (counter-keyed per cell) are
  device-independent.

## Extension points

For a new experiment, check every choice in the experiment note against this
document. If all are supported, the Codifier writes the lifecycle JSON, then
runs `workflow plan` and `ebl campaign run`; no code changes. Otherwise the gap
is one of these points. Extend it there, add a contract or equivalence test,
then write the lifecycle:

| Missing | Where to extend |
| --- | --- |
| network family or model (e.g. another OPT size, analog attention) | a module in `workflow/networks/` with the functions listed in its `__init__.py`; `network` parsing and `PRIMARY_METRICS`/`CALIBRATIONS` in `workflow/lifecycle.py` |
| data or metric | the family module (`build_cache`, `evaluate`) and its `data` parser |
| defect kind or placement | `FAILURE_KINDS` and `_defects` in `workflow/lifecycle.py`; fault construction in `workflow/devices.py` |
| technology, P&V method or relaxation | `TECHNOLOGIES` in `workflow/lifecycle.py`; `workflow/devices.py`, `workflow/program_verify.py` |
| on-chip update law | `TECHNOLOGIES[...].update_laws`; `make_writer` in `workflow/onchip.py` |

Never add a per-study runtime or bypass the stages for one experiment.
