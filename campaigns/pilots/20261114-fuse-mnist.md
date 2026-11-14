# Reciprocal fuse network on masked MNIST

14 November 2026 · [Issue #43](https://github.com/filip137/on-chip-learning/issues/43)
· Backend branch `drn/20261114-fuse-attention` · exploratory, one seed.

**Result:** the 49-token model beats the mean-image baseline by 14.23%,
but nonlinear fuses provide no measured benefit. Independently trained linear
controls give essentially identical results, and disconnecting all fuses leaves
the 256-example probe score unchanged at displayed precision. The 16-token
models learn an average-digit template. The current construction does not
demonstrate useful attention or cross-token computation on this task.

## Scope and frozen protocol

The user requested Claude's reciprocal fuse network on MNIST like the
[linked ET pilot](https://github.com/filip137/analog-research/blob/0e5af26/campaigns/pilots/20261114-energy-transformer-mnist.md).
That pilot is masked image reconstruction. We explicitly asked about the word
“permutated”; no answer arrived, so we used the stated default of matching the
linked task. **No pixel permutation and no digit classification were run.**

Six fresh arms: Gaussian and linear resistors at 16 tokens/D32, 16/D64 and 49/D64.
Four scalar heads retain a two-terminal fuse per query/key pair; this
is not a vector-distance multi-terminal replacement. Encoder, position, query,
key and decoder are signed four-resistor arrays. There is no separate V,
softmax, LayerNorm or diode-memory branch. Historical ET/digital baselines use
the same task, but different architecture, initialization and parameter counts.
The matched mechanism controls are the new identically initialized linear arms.

MNIST raw data at `/home/filip/datasets/mnist`: 55k train, 5k validation, original 10k
test. Split seed 3100; replacement sampling 3101; train masks 3102; fixed validation
and test masks 3103/3104. Hide 8 of 16 image-space 7×7 regions. Pixel voltages [-0.4, 0.4];
missing pixels zero and visibility indicators supplied. All encoder input ports
are clamped to that vector, including zero-valued missing inputs. Train masked
half-MSE; report unclipped MSE/PSNR in [0,1]. No test-driven tuning.

Seed 20261114; name-local signed normal draws standard deviation 0.15/sqrt(fan-in), positions 0.2;
positive/negative conductances = 0.02 + positive/negative part. Adam 0.003, batch 64,
2,000 updates, beta 0.005, conductance floor 0.02. Fuse g = 1, sigma = 0.25; readout gain = 4.
Sources and readout gain are active boundary elements, explicitly. Scalar
Gaussian current is g exp(-dV²/(2sigma²))dV. Physical resistor copies are tied
ideally across rails/tokens; no device mismatch or measured pulse model.

## Results

Every new arm completes 2,000 updates, 10,000 test images, 5,000 validation images and
three 256-example probes. Lower test MSE is better; mean image is 0.0673463.

| Tokens × width | Fuse EP | Linear EP | Prior ET EP | Prior digital BP |
| --- | ---: | ---: | ---: | ---: |
| 16 × 32 | 0.0674530 | 0.0674530 | 0.0638694 | 0.0524059 |
| 16 × 64 | 0.0674381 | 0.0674381 | 0.0640147 | 0.0461110 |
| 49 × 64 | 0.0577657 | 0.0577657 | 0.0544175 | 0.0422274 |

The best fuse result is worse than the prior ET and digital controls. Its normal,
linear-replacement and fully disconnected-fuse validation probe MSEs are all
0.0582221553. Gaussian fuse conductances stay within roughly 0.999998–1 at the
final 49-token training batch; the two 16-token models have fuse conductance 1.
Thus this run does not exercise meaningful nonlinear gating.

The 16-token encoder and query/key signed maps collapse near zero while position
and decoder weights learn a template. The 49-token improvement is consistent
with local patch information: 43.0837% of hidden test pixels share a 4×4 token
with visible pixels; this fraction is zero for 7×7 tokens. Disconnecting all fuse
links removes cross-token paths without changing the measured probe score.
This is evidence about this initialization, conductance floor and 2,000-update
protocol; it is not an impossibility result for reciprocal attention.

[Matched-cohort learning curves](figures/fuse-mnist/learning.png) ·
[16×32 reconstructions](figures/fuse-mnist/reconstructions-small.png) ·
[16×64 reconstructions](figures/fuse-mnist/reconstructions-reference.png) ·
[49×64 reconstructions](figures/fuse-mnist/reconstructions-more_tokens.png).
Reconstructions use the same first four test images, retaining observed pixels.
Figure displays use a fixed [0,1] scale; reported errors never clip predictions.

| Tokens × width | Trainable conductances | Physical resistors | Two-terminal fuses |
| --- | ---: | ---: | ---: |
| 16 × 32 | 10,944 | 319,488 | 1,920 |
| 16 × 64 | 21,888 | 638,976 | 1,920 |
| 49 × 64 | 13,440 | 715,008 | 18,816 |

Logical sharing does not eliminate per-token physical resistor replication.
These counts are ideal element counts, not area or transistor estimates.

## Numerical and evidence verification

The batched antisymmetric-rail reduction matches `fuse.py` explicit currents,
energy and equilibria. A block Gauss-Seidel/lagged-diffusivity numerical solver
solves the unchanged KCL equations, not physical RC time. Free states start at
zero; both nudges start at the same free state. FP32, AMP/TF32 off, maximum 512 sweeps,
KCL infinity residual <= 1e-6; failed phases never update weights. Centered local
squared-drop statistics sum tied physical resistors, using stable algebraic
differences of squares. Training does not backpropagate through trajectories.

All 22 affected tests pass. CPU fixtures verify full-rail parity, energy descent,
implicit derivatives and settled-loss finite differences. Native GPU smoke and
all three 25-update readiness runs pass, including trained-gradient checks.
The full final checks compare each arm's FP32 EP batch gradient on the first
eight saved validation examples with averaged independent FP64 dense-KCL
implicit derivatives; every example has a symmetric Jacobian and positive
definite reference energy Hessian. These are mathematical references, not FP64
training. Final checkpoint hashes are unchanged.

Across six arms, minimum global cosine=0.999999622, maximum global relative
L2 error=0.000908153; worst parameter-group cosine=0.999999339 and relative
L2 error=0.00157697. All fixed gates (cosine > 0.99, relative error < 5%) pass.
Every recorded training and evaluated phase meets the 1e-6 KCL threshold.

Independent parent audit verifies all six native artifact hashes/completeness,
all steps and evaluation counts, identical raw IDX hashes/splits/masks/sampler
RNG to the historical digital controls, equal fuse/linear initial tensors,
clean pinned launch source, FP32 and budget/storage limits. See
`benchmark-001/analysis/parent-verification.json` and the per-arm gradient JSONs.
The parent visually reviewed all reconstructions and the learning curves.

## Execution and retained artifacts

Published source 763e486, pinned in
`/home/filip/analog-research/drn/results/fuse-mnist/source-763e486`.
Command: `python -m experiments.run_fuse_mnist --output-root
/home/filip/analog-research/drn/results/fuse-mnist/benchmark-001
--charged-readiness-seconds 120`. Configs: `examples/fuse_mnist/`.
Local RTX 3090, py312. Supervisor 2552221; durable watcher 2552599 verified process,
GPU and native-artifact progress at 60-second cadence. All six arms have exited.
The bounded completion reviewer independently validated all native bundles and
checkpoint hashes, saved `benchmark-001/watcher-review.json`, and the watcher
exited after marking the run terminal.

Native artifacts, checkpoints, sampler/Adam state, metrics, final checks and
derived plots are in `/home/filip/analog-research/drn/results/fuse-mnist/`.
The composition charges 572.64 s plus 120 s readiness,
692.64 s total, below the 5,400 s allowance.
Per-arm ceilings of 600 s training / 850 s total and 3 GB aggregate output cap were met.
GPU wall times concern this simulator, not analog speed, power or settling time.

Source/results are published on the dedicated feature branch. The prior
shared-trunk publication restriction from the referenced project remains;
neither shared-trunk merge nor worktree removal is claimed. Preserve the
experiment and pinned-source worktrees for review. This bounded performance
evaluation is finished; further initialization/gating changes require a new
experimental comparison, not an unreported retry of these results.
