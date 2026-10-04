# Matched generic HWA/CDT and fixed-array recovery

This exploratory follow-up implements the user-approved comparison described
in [the paper-method audit](cifar_pcm_paper_comparison.md). It tests recovery
after generic HWA has already been trained for the deployment distribution.
It is a suffix-matched comparison inspired by Li et al. (2023), not a numerical
reproduction of the paper's complete peripheral and retention model.

## Fixed scientific contract

- Full pretrained CIFAR-10/100 ResNet-32. Final four convolutions and classifier
  on the existing nine logical crossbar tiles; prefix and BN running statistics
  frozen. Pin the same public teacher checkpoints as the earlier screen.
- Use all 45000 adaptation-training images for HWA, with fresh random padded
  crops and horizontal flips on each pass. A vectorized GPU transform applies
  augmentation before normalization. The same augmentation and minibatch seeds
  are reused across candidates. Teacher targets are computed from the original
  frozen digital network's equivalent suffix on the same augmented features.
- Reserve 5000 training images from adaptation. Use a balanced 1000-image
  prefix of this development split for source and recovery selection. It is
  held out from our adaptation, not from the public model's original pretraining.
- All final metrics use all 10000 official test images. Their labels and
  deployed array identities do not participate in selection.
- Gaussian PCM programming endpoints at 25 µS. No explicit P&V loop, drift,
  extra read noise or ADC/DAC quantization in this matched comparison.
- Per-device independent open/Gmax/random faults; training CDT redraws both
  bank masks and random stuck values every minibatch. Inference and recovery
  keep the final array's masks and stuck values fixed. This physical-bank
  training convention matches our deployment law; the paper's temporary
  logical-weight corruption is not treated as an identical probability law.

## Off-chip training and development selection

One training initialization/data-order seed (41) per dataset; uncertainty in
the final report is conditional on this seed and cannot replace a multi-seed
HWA replication. All fits begin at the original digital weights.

1. Standard HWA: Adam CE/KL at 1e-4, 3e-4 and 1e-3, plus SGD CE at 0.005,
   0.05 and 0.5 (momentum 0.9), with nominal programming noise. Select the
   source by mean development teacher KL across three independent nominal
   array realizations.
2. Increased-noise HWA: fit at 2x and 5x training programming noise with the
   selected optimizer, objective and learning rate. Choose among nominal,
   2x and 5x using equally weighted development KL over nominal and every
   failure/rate combination. Deployment always uses 1x programming noise.
3. CDT: for each failure type, choose the best generic noise candidate for
   nominal plus that type's rates, then fit at corruption probabilities
   0.001, 0.003 and 0.01 per physical bank. Select using equally weighted
   development KL over nominal plus 100, 1000 and 10000 ppm of that type.
   These models use the selected generic optimizer/objective/rate/noise
   settings but begin from the original digital weights, as all other fits do.

This makes 20 candidate fits per dataset. Each runs 30 full training epochs,
batch size 256, evaluating at epoch zero and every five epochs. Epoch zero is
retained as a candidate and explicitly reported if selected. If the best
checkpoint falls in the final 20%, continue that candidate to 60 epochs under
the same settings. This extension is declared before numerical execution.
A best checkpoint still in the last 20% at 60 epochs is explicitly flagged
as requiring convergence review. Such a fit must not support a claim that
recovery exceeds a converged HWA baseline. Fixed-budget numerical results
remain reportable with that qualification.

Operational correction after the first full attempt: SGD at 0.5 produced a
nonfinite objective in both datasets and the original handler stopped the
whole fit stage. The corrected handler records such a candidate as rejected,
retains its earlier finite checkpoints for diagnosis, excludes it from source
selection and continues the unchanged grid. It fails if an entire source
family is rejected. The full campaigns restart from the original digital
weights under a new source snapshot; no unfinished fit is relabelled complete
or reused. Rates, initialization, image budgets and selection criteria remain
unchanged. Reports distinguish attempted, completed and rejected candidates.

The source families are digital, standard HWA, noise-selected HWA, and CDT
for open/Gmax/random faults. Per-source digital-only accuracy and KL are
reported separately from deployed metrics to expose the cost of robustness.

## Recovery development selection

Retain one pass over the original balanced 5000-image recovery cohort, batch
64, teacher KL at T=1. Select global calibration and weight learning rates on
two independent development arrays for each of four tasks: standard HWA at
nominal conditions, and each CDT source at 1% of its associated fault type.
No final array is used.

- Calibration rates: 1e-4, 3e-4, 1e-3.
- Weight-target rates: 1e-5, 1e-4, 1e-3.
- Selection criterion: equally weighted mean final/initial development KL
  across these tasks and arrays. Rate selection is global per dataset, not
  chosen separately on each final array or test outcome.
- Calibration changes only suffix BN affine terms, gains and classifier bias.
  Weight updates use digital gradients with straight-through array reads and
  a fresh Gaussian endpoint reprogramming after each minibatch. Permanent
  failures cannot be repaired and their masks are not optimizer inputs.

## Final paired deployment

Three fresh array seeds 81001–81003 and endpoint seeds 91001–91003 per dataset.
The earlier screen used 51001–51003; HWA validation uses 71001–71003/72001–72003;
recovery selection uses 73001–73002/74001–74002. These roles remain separate.

Digital, standard HWA and noise-selected HWA each cover nominal plus all
three fault types at 100/1000/10000 ppm. Each CDT model covers its nominal
deployment plus the three rates of its own failure type. This gives 42
source/case combinations per final array. Restore each source/case P0 and RNG
for five controls: none, calibration, equally frequent unchanged-target
rewriting plus calibration, weight-only recovery, and recovery plus calibration.
The writing controls make 79 complete reprogramming requests. Pulse counts,
energy and latency are not inferred from these endpoint requests.

Both datasets together have 1260 final measurements (2 × 3 × 42 × 5). Compare
paired KL reductions against calibration and rewriting within each source,
and compare sources on matched physical identities. Report accuracy and
mean/sample SD over arrays, all negative outcomes, selected HWA epochs and
any convergence flags. No significance claim is planned from three arrays.

## Execution

Native experiment: `cifar_pcm_hwa_comparison.v1`. The existing campaign runner
executes six stages per dataset: cache/audit, HWA/CDT fitting, recovery tuning,
and three final screens. Source selection bundles include trained weights;
recovery selection bundles embed their exact source bundle and original hash.
Every native run records input hashes and all final P0/checkpoint artifacts.

The free RTX 5090 on loulou supports the required CUDA architecture and is
substantially faster for this kernel than the available cuDNN-disabled 3090.
Two independent dataset workers share that GPU, with four CPU threads each.
Other GPU jobs are left untouched. A full native small-canary chain must pass
before the two full campaigns start. Source is frozen and staged separately
from the input data. Monitor metrics and receipts at least every 30 minutes;
preserve failed attempts and do not change science when recovering operations.
