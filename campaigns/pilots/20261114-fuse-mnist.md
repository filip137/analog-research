# Reciprocal fuse network on masked MNIST

User assignment, 14 November 2026: evaluate Claude's fuse network on MNIST
like the [referenced energy-Transformer pilot](https://github.com/filip137/analog-research/blob/0e5af26/campaigns/pilots/20261114-energy-transformer-mnist.md).
That reference uses masked reconstruction, not permuted digit classification.
A clarification was offered; until corrected, match the referenced task.
Exploratory, ideal-element simulation, one seed. No claim of transistor or
hardware efficiency validation. Issue: https://github.com/filip137/on-chip-learning/issues/43

## Question and frozen comparison

Does this genuinely two-terminal reciprocal network learn image completion,
and does nonlinear fuse gating improve over independently trained linear
resistors? Six arms: gaussian and linear for small (16 tokens, D32), reference
(16,D64), more_tokens (49,D64). Four **scalar** heads preserve the two-terminal
claim; vector-valued similarity would require additional circuitry. The model
has encoder, position, query, key and decoder resistor arrays, no separate V,
no softmax, no LayerNorm and no diode-memory branch. This is an architecture
comparison, not parameter-matched to the prior ET/digital models.

Each signed connection has four positive physical resistors, paired exactly in
this ideal pilot. Conductances initialize as .02+positive/negative signed draws,
with std .15/sqrt(fan-in), position std .2, seed20261114 with name-local offsets.
All input ports are clamped to the reference's observed-pixel and visibility
indicator vector. Fuse g=1, sigma=.25; readout gain4. The output gain and driven
sources are explicit active boundary components. Physical devices are replicated
per token despite shared logical conductance parameters. Counts are reported.

Data, split, order, masks and metric exactly follow the referenced pilot:
55k train/5k validation/original10k test; split3100, sample3101, train mask3102,
validation mask3103, test mask3104. Hide8/16 image-space7x7 regions; pixels map
to[-.4,.4]. Train masked half-MSE, report unclipped [0,1] MSE/PSNR. No test-driven
tuning. Adam .003, batch64, 2000 updates, floor .02 after each update, beta .005.
Existing ET/digital scores are historical task baselines, not equal-parameter
controls. The same-initialization independently trained linear arm is the primary
mechanism control; zero/linear replays on trained fuses are secondary diagnostics.

## Numerical contract

Batched exact antisymmetric-rail reduction of `networks/energy_transformer/fuse.py`.
Block Gauss-Seidel / lagged diffusivity solves the unchanged KCL equations; this
solver is not an RC time or latency model. Every phase starts from zero (free)
or the same free state (positive/negative). Max512 sweeps, KCL infinity residual
<=1e-6 in FP32; failures never update weights. No trajectory backpropagation in
training. Centered local squared-drop EP statistics sum tied physical resistors;
use algebraic difference-of-squares to avoid unnecessary FP32 cancellation.

Readiness: currents, energy, equilibria match the explicit full two-rail network;
energy decreases; EP matches independent implicit gradients and settled-loss
finite differences on a tiny FP64 fixture. GPU FP32 real-data smoke and gradient
checks precede training. Final trained gradients are independently checked on
saved validation examples. Accuracy with invalid gradients is not accepted.
No tolerance relaxation or silent scientific retry. Report terminal failures.

## Budget and handoff

Local RTX3090; py312 interpreter. 90-minute aggregate compute ceiling, including
readiness and retries; each arm max600s training /850s including evaluation.
Expected duration will be measured by readiness. Max3GB outputs,500MB/arm. Keep
initial, every250-update and final checkpoints with Adam and sampler RNG.
Results: `/home/filip/analog-research/drn/results/fuse-mnist/`.
Backend branch `drn/20261114-fuse-attention`, existing worktree of the same name.
Public `python -m ebl train --config examples/fuse_mnist/<case>-<kernel>.json
--output-dir <root>/<case>-<kernel>`; exact CLI verified before launch.

The reference explicitly retains a prior rejection of shared-trunk integration;
do not repeat that rejected action. Publish this assigned feature branch and
launch from a clean pinned source. Retain it for review rather than claiming
trunk closeout. No changes to the user's README or unrelated work.

Current state: implementation/readiness in progress. Five CPU circuit/gradient
checks pass. No full MNIST training launched. Root agent owns short readiness;
one verified run-watch owner will cover the full serial composition.
