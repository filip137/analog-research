# CIFAR fault-rate sweep: verified measurements

The predeclared hypothesis is that array-specific recovery can improve
teacher KL beyond calibrated, corruption-aware HWA for some failure rates
and analog depths. The planned completion gate required every native arm and
artifact to verify, three fresh physical-array draws per condition, five full
recovery epochs, full-test endpoints, and exact saved-state replay. The
experiment is exploratory and model-based.

Both datasets are complete. The eight V2 study roots contain 6,480 control
outcomes (3,240 per dataset), with five 45,000-image recovery epochs and
10,000-image tests at deployment and after each epoch. Each study passed
`python -m ebl study summarize --verify-artifacts` and is
`ready_for_review`; the final scientific review remains pending. The selected
screen-1 checkpoints also passed 192 exact saved-state replays. The original
digital teachers scored 93.53% on CIFAR-10 and 70.16% on CIFAR-100.
The 376 native campaign nodes are complete with none active. V2 study plans
are under `examples/cifar_crossbar/sweep_v2_*/study-plan.json`, with collected
native roots under `results/cifar-sweep-collected/v2/results/`.

The last four or eight ResNet-32 convolutions and classifier were mapped to
PCM or IBM OM simulated arrays. Below, `CDT` means HWA trained with the same
stuck-device kind and mixed 2/3/5% rates; `noise HWA` is the separately
selected generic robustness source. Calibration updates digital gains,
suffix BN affine parameters, and classifier bias without array writes.
Joint recovery updates those parameters and physical array weights. The
accuracy/KL pairs are three-array means after epoch five; KL is
KL(original digital teacher || student) at temperature 1, in nats. The
last column is the *paired* joint-minus-CDT-calibration accuracy gain and
KL reduction on the same three arrays.

## Five-percent stuck-high devices

| Dataset | Device | Convs | Noise HWA + calibration: accuracy / KL | CDT + calibration: accuracy / KL | CDT + joint recovery: accuracy / KL | Paired gain: pp / KL reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | PCM | 4 | 92.19% / 0.0975 | 92.69% / 0.0692 | 92.78% / 0.0600 | +0.09 / 0.0092 |
| CIFAR-10 | PCM | 8 | 87.75% / 0.2698 | 90.42% / 0.1878 | 90.82% / 0.1812 | +0.40 / 0.0066 |
| CIFAR-10 | OM | 4 | 92.45% / 0.0890 | 92.64% / 0.0763 | 92.66% / 0.0751 | +0.02 / 0.0012 |
| CIFAR-10 | OM | 8 | 86.80% / 0.3643 | 84.18% / 0.3683 | 87.76% / 0.2573 | +3.58 / 0.1110 |
| CIFAR-100 | PCM | 4 | 56.65% / 0.9010 | 60.23% / 0.7550 | 63.96% / 0.5747 | +3.73 / 0.1803 |
| CIFAR-100 | PCM | 8 | 48.16% / 1.3738 | 49.55% / 1.2879 | 57.32% / 0.9847 | +7.76 / 0.3031 |
| CIFAR-100 | OM | 4 | 56.62% / 0.9024 | 40.69% / 1.5599 | 60.33% / 0.7478 | +19.64 / 0.8121 |
| CIFAR-100 | OM | 8 | 42.18% / 1.5755 | 21.55% / 2.5996 | 47.98% / 1.3690 | +26.43 / 1.2307 |

The paired accuracy and KL changes in the last column were favorable on
all three arrays except CIFAR-10 OM/four convolutions, where accuracy
improved on two of three (KL improved on all three). The large CIFAR-100 OM
numbers partly reflect a weak *fault-specific* CDT source: generic noise
HWA plus calibration already reached 56.62% and 42.18% for four/eight
convolutions. Generic noise HWA plus **rewrite and calibration** reached
57.39% / 0.8794 KL and 42.18% / 1.5639 KL, respectively. Against those
stronger HWA-only controls, CDT plus joint recovery was ahead by 2.94 and
5.80 accuracy points, and by 0.1316 and 0.1949 KL, on all three arrays.
CIFAR-10 OM/eight convolutions gained 0.96 points and reduced KL by 0.1070
against generic noise HWA plus calibration, much less than its within-CDT
3.58-point gain.

Across 1/2/3/5% stuck-high rates on CIFAR-100, joint recovery after CDT
improved both accuracy and KL over matched CDT plus calibration on all
three arrays in every depth/device condition. The paired accuracy gains
rose with failure rate: PCM four convolutions +1.10/+1.53/+2.79/+3.73
points; PCM eight +4.68/+5.50/+5.85/+7.76; OM four
+8.04/+11.16/+14.00/+19.64; OM eight +13.98/+20.73/+24.19/+26.43.

The effect depends on the failure mode. At 5% stuck-low on CIFAR-10 PCM,
CDT plus calibration scored 93.39% / 0.0062 KL (four convolutions) and
93.22% / 0.0333 (eight), while joint recovery scored 93.34% / 0.0092 and
93.20% / 0.0401. At 5% random-stuck on CIFAR-10 PCM/eight convolutions,
joint recovery raised accuracy from 91.98% to 92.18% but worsened KL from
0.1318 to 0.1505. CIFAR-100 OM/eight-convolution 5% stuck-low and random
cases did show larger within-CDT gains: +6.73 and +6.54 points, with KL
reductions of 0.2538 and 0.2528, each favorable on all three arrays.

## Exploratory diagnosis of the dataset and HWA-source gaps

The task difference is visible before stuck-device injection. With eight
analog convolutions and nominal programming, PCM accuracy was 93.30% on
CIFAR-10 (0.23 points below its teacher) versus 68.41% on CIFAR-100 (1.75
points below its teacher). OM was 92.80% (0.73 below) versus 63.74% (6.42
below). The mapped layer count is identical; CIFAR-100 has only 5,760 more
classifier weights: 301,312 versus 295,552 logical weights at depth eight.
In the cached original-teacher test logits, among correctly classified
images, the top-two class probability gap was below 0.2 for 0.73% of
CIFAR-10 examples and 4.55% of CIFAR-100 examples. These observations are
consistent with greater sensitivity to weight perturbations in the
CIFAR-100 network, but the independently pretrained weights and class
counts prevent a causal attribution to margins alone.

The large within-CDT OM gains also start from a compromised HWA source.
On CIFAR-100 with eight analog convolutions, the selected stuck-high CDT
source scored only 1.36% on the *clean digital network*, compared with
50.62% for the selected generic noise-HWA source and 70.16% for the
original teacher. The corresponding CIFAR-10 clean-source scores were
36.94%, 92.13%, and 93.53%. CDT trains with resampled 2/3/5% stuck-high
devices and is selected on uncalibrated development KL including those
faults; generic noise HWA trains under inflated healthy programming noise
and is selected across all failure kinds. On CIFAR-100 OM/eight
convolutions, the selected CDT candidate's nominal development KL rose
from 0.379 at epoch zero to 4.350 at epoch 60 while its 5% stuck-high
development KL fell from about 808,544 to 4.459. This measured trade-off
explains why a comparison only against *that* CDT calibration exaggerates
the incremental value of array-specific recovery. It does not identify
which part of the HWA objective, surrogate, source selection, or training
budget caused the trade-off.

These data establish finite-budget, simulated gains, but do not establish
that on-chip recovery is necessary against fully optimized HWA. Each HWA
source has one training seed, and 31 of 40 selected HWA fits triggered the
predeclared late-convergence flag under the 60-epoch cap; three array draws
measure array spread, not HWA-seed spread. The most damaged CIFAR-100 OM/
eight-convolution stuck-high case reached only 47.98% after recovery, still
22.18 points below its 70.16% digital teacher. Its recovery curve was still
rising at epoch five. PCM weight-recovery arms performed 3,520 full Gaussian-array
reprograms, whereas calibration performed none. OM arms used pulse updates
with up to 640 recovery pulses per cell. These write costs matter in any
practical comparison; gradients and optimizers in this simulation remain
digital.

The complete [per-case and per-epoch report](../results/cifar-sweep-analysis/report.md)
and [machine-readable results](../results/cifar-sweep-analysis/results.json)
retain all sources, controls, three-array spread, and paired contrasts. The
[protocol](cifar_fault_rate_sweep.md) and
[execution record](cifar_fault_rate_sweep_execution.md) specify the frozen
settings and provenance.
