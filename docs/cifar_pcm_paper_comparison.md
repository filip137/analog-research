# Comparing the fault screen with Li et al. (2023)

Local source: [full paper PDF](../../papers/Impact_of_analog_memory_device_failure_on_in-memory_computing_inference_accuracy.pdf).
The local copy was located and checked on 22 September 2026. Use its
Investigation Method, Results and Methods sections for the comparison below.

## Published methods

The [paper](https://doi.org/10.1063/5.0131797) uses generic off-chip HWA:

- Standard HWA injects PCM weight noise and forward-path non-idealities.
- Noise-strength sweeps test additional robustness to hardware variation.
- Corrupt-device training (CDT) adds temporary failures, resampling their
  locations every minibatch. Updates modify the underlying digital weights.

Deployment uses independently sampled faults at unknown locations. The method
aims to export one model to many chips; it does not evaluate per-chip recovery.
Inference includes programming noise, read noise, drift and drift compensation.
Its CNN experiment is ResNet-32 on CIFAR-10, with five training initializations
and 25 inference realizations per trained model. Table I reports optimized
tolerances of 800 ppm open and 100 ppm random/Gmax faults under its one-day
accuracy criterion. These are not teacher-KL thresholds.

The Methods section distinguishes the temporary training weight corruption
from inference failures on either or both devices of a differential pair.
A reproduction must reconcile those probability conventions explicitly.

## What our completed screen establishes

Our [measured screen](cifar_pcm_fault_results.md) uses original pretrained
digital weights, without generic HWA or CDT. It injects permanent physical-bank
failures and evaluates immediate Gaussian programming endpoints. Its recovery
updates the same array with the same fault map. The final four convolutions
and classifier are analog; the rest of ResNet-32 stays digital.

| Comparison dimension | Paper | Completed screen |
|---|---|---|
| Source weights | HWA, with noise/CDT variants | Original pretrained digital |
| Fault locations during adaptation | Resampled across minibatches | Fixed target array |
| Goal | Robust deployment across chips | Recovery on one deployed chip |
| Evaluation | Accuracy and defect tolerance, including retention | Teacher KL and accuracy immediately after programming |
| CIFAR coverage | CIFAR-10 | CIFAR-10 and CIFAR-100 |

Our observed severity ordering is open < random < Gmax. The recovery gain at
1% faults on CIFAR-100 is measured against calibration and rewriting of a
digital-source deployment. It cannot be used to claim an advantage over the
paper's HWA/CDT method. Numerical tolerance comparisons would additionally
require matched mapping, inference age, peripheral assumptions and criteria.

## Matched experiment needed to answer the original question

Use the existing suffix and Gaussian deployment model consistently across:

1. Original digital targets, retained as the measured reference.
2. Standard HWA, without injected permanent failures during training.
3. HWA plus CDT, with generic resampled faults and training corruption strength
   selected independently of the test arrays.

For each trained source, deploy to fresh fixed arrays. Restore its exact P0
for no recovery, digital calibration, matched frozen-target rewriting,
weight-only recovery and recovery plus calibration. Pair physical identities
and programming draws across sources where possible; each source has its own
programmed P0 because its targets differ. Keep all arms within a source on
that same P0. Measure teacher KL against the original digital teacher and
accuracy before HWA, after HWA, after deployment and after one recovery pass.

The decisive comparison is recovered HWA+CDT against calibrated HWA+CDT and
rewritten HWA+CDT on the same target arrays. HWA must be selected on development
data and trained sufficiently to assess convergence; no selection uses test
labels or the target test-array identities. Report any clean-model cost of
increasing HWA noise or training fault probability. Retention-matched paper
reproduction is a separate comparison from our immediate-endpoint screen.

## Implementation status

The user approved the follow-up, which is now implemented and numerically
complete under the native cifar_pcm_hwa_comparison.v1 experiment. The
[fixed protocol](cifar_pcm_hwa_comparison.md) adds augmented full-data HWA,
noise-strength and CDT sweeps, development-only selection, fresh arrays and
matched recovery controls. The [measured results](cifar_pcm_hwa_results.md)
cover 1,260 final measurements on CIFAR-10/100 and full artifact verification.

The strongest added recovery result is joint weight updates and calibration
on CIFAR-100 with 1% Gmax faults. It improves the selected CDT plus calibration
control on all three arrays, but several source fits selected late checkpoints
and remain unconverged under the predeclared review rule. This supports a
fixed-budget incremental benefit, not a claim beyond fully converged HWA or
superiority to the paper's complete inference/retention model. No old run or
selection was relabelled as new HWA evidence.

## Task comparison checked against the local paper, 22 September 2026

The overlapping prediction task is CIFAR-10 image classification using the
ResNet-32 architecture family. Our recovery does not change the task or
introduce new classes: it tries to restore the same deployed classifier.
The paper additionally evaluates a two-layer LSTM on Penn Treebank and
BERT-base/ALBERT on MRPC; these language tasks are absent from our CIFAR
studies. CIFAR-100 is our additional classification benchmark and is not
evaluated in this paper.

The scientific questions differ. The paper measures tolerance to unknown
device faults after generic HWA/CDT, including inference at different times
after programming. Our latest sweep measures the incremental benefit of
calibration and fixed-array weight recovery after deployment, for each
source, fault kind, device model and analog suffix depth. Our training uses
teacher KL at T=1, with classification accuracy also reported; the paper
does not describe this same teacher-KL recovery task.

Our current models keep the digital prefix frozen and deploy only the last
four or eight convolutions and classifier. The paper does not specify this
same suffix-recovery setup. The prior reference to full-network mapping
should not be taken as a verified layer-by-layer match: its reported
ResNet-32 weight count is approximately 0.36 million, whereas the local
CIFAR-10 model contains 464,432 convolution/classifier weights (466,906 total
parameters including BN affine terms and classifier bias). The paper does
not provide enough architecture detail here to resolve that discrepancy.
Treat the common ResNet-32 name as an architecture-family match, not proof
of an identical implementation, checkpoint or analog layer map.

For inference, the paper includes PCM programming/read noise, drift and
drift compensation, with peripheral nonidealities in HWA. Its tolerance
criterion is evaluated at one day. Our present PCM sweep evaluates immediate
Gaussian programming endpoints without that retention/peripheral pipeline;
OM is a separate additional device model. Our main sweep uses nominal and
1/2/3/5% faults. The paper's reported ResNet-32 tolerance thresholds of 0.08%
open and 0.01% random/Gmax are outputs of its different criterion, not the
complete range of fault rates it tested.

The paper averages five training initializations and 25 inference
realizations per initialization. Our current sweep has one HWA training seed
and three final array identities. Dataset overlap alone therefore does not
make the numerical comparisons a reproduction of the paper's benchmark.

The APL main text does not specify a numerical training-epoch count or a
separate CDT fine-tuning duration. Its Methods section says each setting is
trained from five initial conditions and evaluated after the same number of
epochs, without giving that number. The five initial conditions are not five
epochs. Our 30-epoch fits, conditional extension to 60, and five-epoch recovery
budget are our declared protocol choices, not durations verified from APL.
Reference 9's supplementary Figure 2 describes a ResNet-32 HWA learning-rate
schedule, but it does not establish the APL CDT duration; do not transfer an
epoch count between those studies without additional evidence.
