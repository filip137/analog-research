# Five full recovery epochs on CIFAR: measured results

All four studies completed on 2026-09-18. CIFAR-10 finished at 05:31:33 UTC
and CIFAR-100 at 05:33:33 UTC. Both launchers exited zero and released their
RTX 5090s. All 20 native stages and 480 controls are collected and pass full
artifact-hash verification. The 384 adaptation trajectories produced 1920
post-recovery full-test endpoints. Ninety-six selected first-array endpoints
replay exactly, including predictions, accuracy, KL, original-prefix hashes
and deployment references. The four canaries separately passed 256 replays;
80 implementation/report checks passed.

[Full tables, curves and CSV exports](../results/cifar-full5-analysis/report.md)
and [the frozen scientific contract](cifar_full_epoch_recovery.md) describe the
complete evidence. Formal scientific review and managed-manifest finalization
remain pending the user's interpretation and next experiments.

## What ran

The full pretrained CIFAR ResNet-32 was retained, with a frozen digital prefix
and its final four convolutions plus classifier mapped to crossbars. Digital
test accuracy reproduced at **93.53% / 70.16%** for CIFAR-10/100. Each recovery
epoch used all **45000 adaptation images**; five epochs therefore mean 225000
image presentations and 3520 updates. Evaluate all 10000 test images at
deployment and after every epoch. Original digital-teacher KL is measured in
nats, at temperature one. There is no test-based model or epoch selection.

Each dataset/device has direct digital and standard device-specific HWA
sources, nominal arrays and 1% stuck-low/open, stuck-high/Gmax and random-stuck
cases, three new array identities, and five matched controls: no recovery,
calibration, unchanged-target rewriting plus calibration, weights only, and
joint weight/calibration recovery. Every control restores its exact common
P0, fault identities and RNG. Means below are over three arrays; the full
report includes sample standard deviations and every individual array.

PCM reuses the earlier standard HWA checkpoint and Gaussian endpoint law.
Recovery reprograms targets after each minibatch, without an additional P&V
loop. OM uses literal AIHWKit 1.1.0 preset identities with native corrupt cells
counterfactually repaired before the declared failure injection. It retains
the sampled references and uses apparent q=a-r. It has a separately trained,
30-epoch HWA model selected on development KL (epochs 17 / 8 for CIFAR-10/100).
OM recovery uses the existing closed-loop pulse Adam, one pulse opportunity
per minibatch and at most 640 pulses/cell total. Both use digital gradients.
The PCM procedure is an endpoint-reprogramming abstraction; it is not a PCM
incremental pulse-training result or an analog-backpropagation demonstration.

## CIFAR-100: where recovery improves the tested HWA controls

The following comparisons use standard HWA sources and equally trained
non-weight controls. Each listed recovery improves **both accuracy and KL on
all three paired arrays**. The controls are named explicitly; all other
comparisons are also exported, rather than selecting a model using test data.

| Device / fault | Non-weight control | Control KL / accuracy | Recovery | Recovery KL / accuracy | KL reduction | Accuracy gain |
|---|---|---:|---|---:|---:|---:|
| PCM, Gmax 1% | Calibration | 0.59994 / 61.34% | Joint | 0.35199 / 66.04% | 41.33% | +4.69 pp |
| PCM, random 1% | Calibration | 0.22891 / 67.18% | Weights | 0.17210 / 67.95% | 24.82% | +0.76 pp |
| OM, stuck-low 1% | Rewrite + calibration | 0.44695 / 63.94% | Weights | 0.28197 / 66.80% | 36.91% | +2.86 pp |
| OM, stuck-high 1% | Rewrite + calibration | 0.49041 / 63.23% | Joint | 0.35634 / 66.07% | 27.34% | +2.84 pp |
| OM, random-stuck 1% | Rewrite + calibration | 0.25424 / 66.85% | Joint | 0.19191 / 68.03% | 24.52% | +1.18 pp |

The highest-fault recovery remains partial relative to the original digital
teacher's 70.16%. At 1% high faults, PCM standard HWA has 70.06% accuracy
before programming and 35.02% after deployment; joint recovery reaches 66.04%.
OM standard HWA has 69.57% before programming and 19.66% after deployment;
joint recovery reaches 66.07%. These different drops are not a controlled
ranking of devices: OM native bounds/reference encoding and PCM differential
25-uS failure endpoints differ, as do their HWA training recipes.

Nominal and open-fault PCM favor calibration: nominal KL 0.02597 / accuracy
69.91%, versus joint recovery 0.09562 / 68.75%. At open 1%, calibration gives
KL 0.04890 / 69.44%, versus joint 0.10379 / 68.53%. Extra writes resample
healthy-device noise under this PCM abstraction. Weight-only recovery also
underperforms calibration in these two cases.

Nominal OM rewriting gives KL 0.07020 / accuracy 69.55%, compared with joint
recovery 0.07464 / 69.62%. Thus a small accuracy difference does not establish
a joint fidelity-and-accuracy recovery advantage in that case.

## CIFAR-10: mostly fidelity gains or calibration-driven recovery

For PCM Gmax 1%, weight-only recovery lowers mean KL from calibration's
0.030224 to 0.022957 (24.0%), but accuracy is 93.11% versus 93.16%.
Joint recovery has KL 0.025628 / 93.10%. This is a KL benefit, not a clear
accuracy advantage. Nominal/open PCM favors calibration; random-fault
differences are small.

For OM Gmax 1%, calibration and joint recovery both reach 93.10%; their KL
values are 0.028806 and 0.028558. Rewriting gives the slightly lower KL 0.028222
at 93.09%. Across the OM cases, joint recovery does not beat rewriting in mean
KL. Weight-only recovery sometimes slightly improves accuracy while worsening
KL, so it does not establish a general recovery advantage.

The OM pulse counts show how little joint weight adaptation occurred on
CIFAR-10: from HWA sources, averages were 40 pulses nominal, 766 stuck-low,
310 stuck-high and 82 random-stuck, across 148096 logical weights. Corresponding
CIFAR-100 means were 16467, 70618, 63772 and 42126 pulses across 153856 weights.
These are commanded pulse counts, not calibrated energy estimates. Every
method's costs and cap saturation are exported in `final_costs.csv`.

## Relation to the previous 20 passes

These are five **full epochs**, not five passes over the old 5000-image subset.
For PCM Gmax 1%, joint recovery from standard HWA changes from the old
20-pass result of KL 0.54859 / accuracy 62.84% to 0.35199 / 66.04% on CIFAR-100.
From digital weights it changes from 0.57219 / 62.32% to 0.36564 / 65.93%.
For CIFAR-10 standard HWA, it changes from 0.03192 / 92.98% to
0.02563 / 93.10%.

This is **not** an isolated epoch-count or equal-compute comparison: total
image presentations increased from 100000 to 225000, programming calls from
1580 to 3520, the training cohort and schedules changed, and new arrays were
used. The old noise-enhanced HWA and per-fault corruption-aware HWA/CDT sources
were not repeated in this new study. Its results therefore compare recovery
against the declared **standard HWA** baselines, not the strongest conceivable
fault-aware HWA or a complete reproduction of the cited papers.

## Scope and verification

Results remain exploratory and model-based, conditional on one teacher/HWA
seed and three physical-array assignments. The OM HWA recipe is limited;
neither three-array spread nor a non-late selected epoch proves convergence
or statistical significance. Recovery uses fixed inherited base learning
rates, with no new full-cohort tuning. Some curves are still improving at
epoch five. Original teacher pretraining used the complete original training
set, so development is held out from adaptation, not teacher pretraining.
No read-noise, ADC, retention or full physical-array energy claim is made.

Numerical source v2 identity:
`a6386fdc60ba72707f2b1da31ad7c5b9978d021fb2589aee11b35789a779ee86`.
Remote source: `/home/filip/cifar_full_epoch_sources/v2` on loulou and fifi.
Local mirror: `results/cifar-full5-collected/v2`. The first v1 canary attempts
are preserved separately: numerical runs completed, but native bookkeeping
lacked terminal cache/source metrics. V2 corrected that issue, added a
regression check, and passed all canaries before full launch. No full-run
failures or scientific changes occurred during execution.
