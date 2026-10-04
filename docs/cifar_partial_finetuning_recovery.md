# Partial fine-tuning for CIFAR hardware recovery

Status: research and experiment proposal, 20 September 2026. Parameter
budgets below were checked against the local model on CPU. No partial
weight-recovery or CIFAR adapter training has been run for this proposal.

The useful question is how much recovery survives when we restrict either
the physical weights that can change or the size of a separate digital
correction. These are different hardware interventions and need separate
cost reporting. I recommend classifier-only recovery first, selected-layer
updates next, and digital low-rank correction as the strongest alternative
that can leave the entire programmed array fixed.

## What is already partial fine-tuning

Existing calibration trains suffix BN scale and offset, one output gain
per mapped matrix, and classifier bias. It updates 1,043 digital parameters
for CIFAR-10 and 1,133 for CIFAR-100 with eight analog convolutions. It does
not update analog weights or BN running means and variances. The four-
convolution counts are 527 and 617.

Bias-only tuning and feature scale/shift tuning therefore overlap with
existing controls. Our convolutions have no biases; a bias-only arm would
mainly train suffix BN offsets and classifier bias. Feature scale/shift at
the same pre-BN locations can duplicate an existing affine correction.
BN-statistics refresh remains a distinct proposed control in the
[next-tests plan](cifar_recovery_next_tests_proposal.md).

## Concrete budgets

All rows below keep **eight convolutions plus classifier deployed on the
same faulty array**. Percentages use the deployed logical weight count as
denominator. They are neither physical device counts nor measured write
savings. Except the calibration row, calibration is frozen after a shared
initial stage; a joint-calibration variant adds the counts above.

| Recovery method | CIFAR-10 trainable quantities | CIFAR-100 trainable quantities | What changes physically? |
|---|---:|---:|---|
| Existing calibration | 1,043 digital parameters | 1,133 digital parameters | No array writes |
| Classifier weights only | 640 weights, 0.217% | 6,400 weights, 2.124% | Classifier only |
| Last residual block + classifier | 74,368 weights, 25.162% | 80,128 weights, 26.593% | Last two convolutions + classifier |
| Last two blocks + classifier | 148,096 weights, 50.108% | 153,856 weights, 51.062% | Last four convolutions + classifier |
| Selected 5% of weights | 14,777 weights | 15,065 weights | Fixed selected subset |
| Full physical recovery | 295,552 weights | 301,312 weights | Full deployed suffix |
| Dense digital classifier residual | 640 added parameters | 6,400 added parameters | No array writes; extra digital classifier computation |
| Digital LoRA, rank 2, every mapped matrix | 10,388 added parameters | 10,568 added parameters | No array writes; extra digital computation |
| Digital LoRA, rank 4, every mapped matrix | 20,776 added parameters | 21,136 added parameters | No array writes; extra digital computation |
| Digital LoRA, rank 8, every mapped matrix | 41,552 added parameters | 42,272 added parameters | No array writes; extra digital computation |

The complete [budget table](../artifacts/cifar_partial_recovery_20260920/parameter_budgets.md)
also covers four-convolution deployments and 1%/10% selection budgets.
The [audit script](../labs/tools/audit_cifar_partial_recovery.py) constructs
the actual suffix and checks the factor parameter counts; its manifest
records the model and script hashes. This audit measures architecture only.

## Candidate methods and what each would answer

**1. Classifier-only physical recovery.** Keep all deployed convolutional
weights fixed and adapt the existing 64-to-class classifier. This tests
whether the corrupted representation remains useful after calibration and
can be decoded with a different decision boundary. It is the cheapest new
physical-weight baseline. It cannot generally undo information already
lost at upstream nonlinearities. Keep a digital dense classifier-residual
arm with the same number of learned weights: that comparison exposes the
effect of putting the correction in a faulty array versus an additional
digital path, while acknowledging the digital path's inference cost.

**2. Layer selection, followed by sparse updates.** Compare the classifier,
last block plus classifier, and full suffix first. On development arrays,
also test each residual block plus classifier individually: the earliest
deployed block might matter more than the last. Then allocate fixed 1%, 5%
and 10% budgets using teacher-KL gradient information from adaptation data,
with equal-budget random masks as controls. Compare unstructured selection
with channel or matrix selection where addressability matters. Count the
gradient-screening data and computation in the budget. Specify the gradient
coordinate and scale normalization before ranking; packed normalized `q`
and physical weight units are not interchangeable sensitivity scores.

This is motivated by data-dependent allocation in
[SPT](https://arxiv.org/abs/2303.08566) and layer/subtensor updates in
[On-Device Training Under 256KB Memory](https://arxiv.org/abs/2206.15472).
Those works support testing selective adaptation; they do not establish
that a particular subset repairs our stuck-device faults. Selection uses
adaptation data and observable reads, without giving the controller the
simulator's hidden fault identities. An oracle mask, if useful diagnostically,
must be labelled separately. Faulty cells remain permanently faulty.

**3. Digital low-rank residuals.** Keep each deployed analog matrix fixed
and add a trainable correction in parallel, before its BN/nonlinearity:

`matrix_output = deployed_analog_output + (alpha / rank) * B(A(input))`.

For a 64-output, 64-input 3x3 convolution, use a 3x3 convolution from 64
channels to rank `r`, followed by a 1x1 convolution from `r` to 64. This is
a rank-r update of the flattened 64-by-576 kernel, with `640*r` parameters.
Use bias-free factors; common calibration already owns the offsets.
Initialize B to zero and A nonzero, so the initial deployed function is
unchanged and gradients can start learning. Do not initialize both to zero.
The classifier has `r*(64 + number_of_classes)` factor parameters.

[LoRA](https://arxiv.org/abs/2106.09685) supplies the factorized update idea.
More directly, [AHWA-LoRA](https://arxiv.org/abs/2411.17367) studies external
adapters around fixed analog transformer weights. Its transformer task and
hardware results motivate this branch; they are not CIFAR stuck-fault
recovery measurements. Here rank 4 adds about 7% of the deployed base weight
count, without base-array reprogramming if the factors stay digital.

A low-rank *learned correction* need not reproduce the physical fault
matrix exactly. Conversely, unstructured faults do not guarantee low-rank
functional correction, so rank must be tested. Merging `B*A` back into the
analog weights generally changes a dense matrix and requires programming
it through the device model. It does not preserve the zero-write advantage.
A separate analog adapter is a further architecture experiment with its
own devices, faults, programming, reads and interconnect; it is not covered
by these ideal digital-adapter counts.

Small bottleneck residual modules are a possible second digital design.
[TinyTL](https://arxiv.org/abs/2007.11622) is relevant because it targets
activation storage as well as parameter count. We should measure peak
training memory and backward computation: training fewer parameters alone
does not promise proportional memory or latency savings. Classifier-only
updates with calibration frozen permit detaching all upstream features;
updating BN or adapters throughout the suffix still needs upstream gradient
propagation. Feature caching would require an explicit matched policy for
augmentation and stochastic reads.

## User-proposed direction: analog FISH mask with one-pulse sampling

The user proposed combining a sparse mask with at most one pulse per cell
per update, encoding gradient magnitude in pulse probability. This is a
particularly useful extension of selective physical recovery. Interpret the
limit per minibatch update; selected cells may receive pulses again on later
updates, without a new cumulative pulse cap.

Two decisions stay distinct: a fixed FISH mask chooses which weights may
ever receive recovery pulses; a Bernoulli draw chooses which eligible
weights receive a pulse now. [FISH Mask](https://arxiv.org/abs/2111.09839)
selects a fixed subset using diagonal Fisher estimates. Probabilistic
pulsing alone is a stochastic update rule; it does not compute a FISH mask.
Stochastic pulse encoding also has established analog-training precedent
in [Gokmen and Vlasov](https://arxiv.org/abs/1603.07341). Those pulse-train
outer-product implementations are not automatically identical to arbitrary
per-cell masked, single-pulse commands.

### Proposed rule in the existing normalized weight coordinate

Let `q_i` be the normalized deployed weight, `g_i = dL/dq_i` the current
minibatch teacher-KL gradient, `m_i` the fixed binary mask, `eta` the SGD
learning rate, and `h_nom` the public nominal pulse step in the same units.

```
p_i = m_i * min(eta * abs(g_i) / h_nom, 1)
z_i ~ Bernoulli(p_i)
direction_i = -sign(g_i) * z_i
```

Issue one physical SET/RESET pulse where `direction_i` is nonzero. There is
no target accumulator or post-write verification loop; forward reads and
gradient computation are still required. Under an ideal symmetric fixed
step `h_nom`, with no probability clipping or weight saturation,
`E[delta_q_i | g_i] = -eta * m_i * g_i`. For an eligible coordinate the
conditional update variance is `h_nom^2 * p_i * (1 - p_i)`. Thus small
gradients generate rare full-size jumps, rather than small deterministic
changes. A lower LR does not make an individual pulse smaller.

For real modeled OM responses, the expectation instead depends on the
direction- and state-dependent mean pulse response. The nominal-step rule
is only an approximate realization of SGD; permanent faults, bounds,
asymmetry and apparent write noise remain active. Log probability clipping,
requested pulse counts, observed changes and the unchanged physical states
outside the mask. If a later arm adjusts probabilities using measured
response estimates, obtain them through declared observable calibration,
count that cost, and never expose hidden device parameters to the controller.

### Relation to the completed OM runs

`UncappedOpenLoopAdam` in
[open_loop_updates.py](../experiments/cifar_crossbar/open_loop_updates.py)
already computes `p = min(abs(Adam_command) / nominal_step, 1)` and sends at
most one pulse per cell per minibatch, without a cumulative cap. Its nominal
OM step is 0.0949 in normalized coordinates. It uses no FISH selection.

The new contributions to our comparison are therefore the fixed mask and
the choice of optimizer command. Raw SGD directly preserves gradient
magnitude in pulse probability and removes Adam's two moment buffers.
Adam scales coordinates using their gradient history, so its pulse
probabilities do not encode raw magnitude. Keep both variants to separate
mask effects from optimizer effects, tuning their LRs independently on
development data. Neither variant by itself implements physical on-chip
gradient computation; our current gradients remain digital.

### Constructing a defensible recovery mask

Estimate scores after the shared calibration prefix on adaptation images,
on the actual deployed model, then keep the selected mask fixed. Declare
normalized `q` coordinates for both mask scores and pulse gradients; using
unscaled software weights would change cross-layer rankings.

For a literal FISH baseline, estimate the diagonal Fisher as the mean of
`(d log p_student(y|x) / dq_i)^2`, with labels sampled from the deployed
student distribution (or a separately declared empirical-label estimator).
For a recovery-specific alternative, score the mean squared **per-example**
teacher-KL gradient. Call the latter a recovery-gradient mask, not a Fisher
estimate: it measures a different quantity and can vanish when teacher and
student match even though the Fisher is nonzero. The square of a mean
minibatch gradient is not the mean of squared per-example gradients.
Count mask-estimation images, backwards passes and storage separately.

High Fisher measures sensitivity, not remaining ability of a failed device
to move. A mask can spend effort on stuck cells. Keep faults hidden from
the controller and report futile pulses diagnostically. A later observable-
response mask would be a separate intervention, with its probe cost counted.
An arbitrary per-cell mask also needs an addressing or gating mechanism;
the simulation's vector pulse API does not demonstrate a hardware-efficient
implementation. Compare channel/tile masks if that constraint matters.

### Small initial comparison

Start with 5% eligible weights, then expand to 1% and 10% only if useful.
Keep the same calibrated array and frozen digital calibration in the first
weight-only comparison; repeat selected methods with joint calibration.

| Eligible weights | SGD probability command | Adam probability command |
|---|---|---|
| All deployed weights | Isolate raw-gradient pulsing | Existing open-loop mechanism |
| Fixed random 5% | Control for merely reducing the writable subset | Matched mask control |
| Fixed FISH 5% | User-proposed primary method | Isolate FISH benefit under existing optimizer |

Include continued digital calibration and the matched closed-loop full
recovery reference. Add the recovery-gradient mask as a diagnostic if FISH
does not allocate updates usefully. Use the dataset/fault panel below,
separating nominal cases to reveal damage to a good deployment.

Compare accuracy and teacher KL at equal data exposure, and also compare
development-selected operating points with matched expected pulse budgets
`sum_i p_i`. A top-Fisher mask and a random mask need not spend the same
number of pulses at the same LR. Record actual pulses, unique cells pulsed,
layer allocation, clipping and pulses into permanent failures. These
statistics distinguish better allocation from simply spending more writes.

Implement this first for OM, which has a declared per-pulse response. Our
current Gaussian PCM programming-endpoint model supplies no single-pulse
learning law and cannot validate this rule without a separate device model.
This remains a proposed comparison; no new recovery run is launched here.

## Where to start, separately by dataset and fault

Historical values below are accuracy (%) / teacher KL in nats at T=1,
three-array means. They use stronger-noise HWA, eight analog convolutions,
5% faults and five recovery epochs. They motivate the new tests, not a
claim about partial recovery. Historical joint recovery starts immediately
after deployment, so these are not calibrated-prefix experiments.

| Dataset / device / fault | Calibration only | Full joint recovery | Role in the new comparison |
|---|---:|---:|---|
| CIFAR-100 / PCM / stuck-high | 48.16 / 1.3738 | 59.65 / 0.8775 | Largest example of useful gain to retain with fewer updates |
| CIFAR-100 / OM / stuck-low | 47.56 / 1.3766 | 54.49 / 1.1075 | Test whether limited physical correction can recover distributed damage |
| CIFAR-100 / OM / random-stuck | 55.30 / 1.0408 | 60.06 / 0.8682 | Test selective updates and digital residuals under dispersed faults |
| CIFAR-100 / OM / stuck-high | 42.18 / 1.5755 | 46.76 / 1.4063 | Difficult case with substantial residual loss |
| CIFAR-10 / PCM / stuck-low | 93.02 / 0.0561 | 93.09 / 0.0513 | Small-gain case where extra recovery may be unnecessary |
| CIFAR-10 / OM / stuck-high | 86.80 / 0.3643 | 86.52 / 0.3485 | Accuracy/KL trade-off; do not assume full recovery is the accuracy winner |

Start with the existing proposal's panel: CIFAR-100, both devices, nominal
plus 5% of each fault; CIFAR-10 PCM stuck-low and OM stuck-high with nominal
references. Use two development arrays per condition. Report every dataset,
device and fault separately. Complete CIFAR-10's small screen before the
CIFAR-100 screen, consistent with the worktree execution order. Follow up
with the other CIFAR-10 faults, four-convolution deployments and 1%, 2%, 3%
and 5% rates. Add 4% explicitly if a complete integer 1–5% sweep is desired;
it was not part of the historical fault sweep.

## Matched staged experiment

1. Start from the same noise-HWA checkpoint and exact saved deployed array
   for every method. Run five common calibration epochs, save all physical,
   digital and RNG states, then branch. Calibration stats remain frozen
   unless a separate BN-refresh comparison is declared for all arms.
2. In the first screen compare: continued calibration; classifier-only
   physical updates; last-block-plus-classifier updates; full physical
   recovery; a digital classifier residual; and rank-4 digital LoRA. Add a
   frozen branch as the zero-adaptation reference. Every trained branch
   receives the same next five adaptation-data passes and teacher-KL
   objective. Freeze calibration in the weight/adapter arms to isolate
   adaptation capacity. The continued-calibration arm tests the practical
   alternative of spending the same data budget on digital calibration.
3. Repeat retained partial/full methods with calibration jointly trainable
   in both arms. Never compare partial plus calibration with full weights
   alone and attribute the difference to partial tuning. Include frozen-
   target rewrite controls using the corresponding physical update masks
   to distinguish learning from reprogramming effects.
4. Explore rank 2/4/8, layer choice and 1%/5%/10% masks only on development
   arrays. Give each method the same declared hyperparameter-search budget;
   a shared LR alone is not evidence that methods are equally optimized.
   Account for mask screening and select epochs on development teacher KL,
   never confirmation arrays or test labels.
5. Confirm selected comparisons on fresh arrays, then repeat on matched
   normal-HWA and corruption-aware-HWA sources. This determines whether
   apparent partial-recovery success depends on the offline source.

A useful proposed goal is to retain at least 90% of the matched full method's
KL reduction while making at most 10% of physical weights eligible for
updates. Use `(KL_frozen - KL_partial) / (KL_frozen - KL_full)` only when the
full method improves KL. Report absolute KL and accuracy regardless; also
report whether accuracy is within 0.5 percentage points of the matched full
method, with paired array differences. These are proposed screening targets,
not established scientific thresholds or claims about existing results.
Digital adapters belong on a separate cost comparison with digital storage,
MACs, training memory, wall time and zero base-array writes recorded.

## Required implementation before numerical runs

- Preserve the declared deployment. Setting `suffix="head"` would move
  faulty convolutions back to clean digital execution; it is not a valid
  implementation of classifier-only recovery on an eight-convolution array.
  Build update masks from the existing `MatrixSpec` offsets instead.
- PCM currently programs the whole Gaussian endpoint tensor each step.
  Add explicitly declared selective endpoint programming, preserving every
  unselected conductance, including both differential banks. Specify RNG
  handling and record addressed cells and reprogram operations. This is a
  new selective-programming abstraction, not a measured incremental pulse
  law or demonstrated physical addressability.
- OM must gate physical pulse eligibility as well as gradient, Adam moments
  and target changes. Zero gradients alone do not prevent closed-loop
  error correction on frozen positions. Keep the underlying OM law fixed;
  report actual pulse counts and verify reads. Do not infer fewer reads
  from a sparse pulse mask when the port still reads the whole array.
- For digital residuals, implement the branch at each existing analog
  matrix output, preserving the analog forward, noise, gains and permanent
  faults. The initial model must match the saved calibrated deployment.
  Existing DRN digital/passive adapter implementations are architecture
  references, not drop-in CIFAR components.
- Check zero-mask no-op behavior, all-mask agreement with the full updater,
  frozen-cell and prefix hashes, fault permanence, zero-B initial function
  equality, and exact resume with masks/factors/optimizer/RNGs restored.
  Measure peak memory and actual costs rather than extrapolating from counts.

Implement numerical studies as a new native EBL contract; keep historical
results and device laws unchanged. This note adds an experiment direction
to the [next-tests proposal](cifar_recovery_next_tests_proposal.md), not a
new running campaign.

Local evidence: [CIFAR-10 analysis](../artifacts/cifar_fault_regime_analysis_20260920/cifar10_analysis.html),
[CIFAR-100 analysis](../artifacts/cifar_fault_regime_analysis_20260920/cifar100_analysis.html),
`results/cifar-sweep-analysis/summary.csv`,
[model](../experiments/cifar_crossbar/model.py),
[PCM endpoints](../experiments/cifar_crossbar/pcm_faults.py),
[OM writer](../experiments/cifar_crossbar/devices.py), and
[recovery runtime](../experiments/cifar_crossbar/full_epoch_runtime.py).
