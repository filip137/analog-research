# MOS attention circuit and simple task learning

User assignment, 13 November 2026: implement end-to-end analog attention,
modify return paths if necessary for learning-signal propagation, and train
the circuit on a simple task. This extends issue #36 on the explicitly
requested `energy-transformer` branch. The user authorizes implementation
and local task training, not only another single-update characterization.

## Required outcome

1. Explicit MOS multiplier devices for scores and both attention return paths;
   no behavioral multiplication in the evaluated attention circuit.
2. Voltage projections, score summation, softmax, return paths and output
   nudging connected in a complete circuit. Use a native MOS compact model
   for softmax as well, with the previous analytic bank as a labelled reference.
3. All learning phases solved by SPICE for the evaluated training run. The
   controller can read local phase statistics and program tied resistor copies;
   it must not substitute digitally differentiated or ideal-solver gradients.
4. Learn paired-token copying: input [a,0,b,0], targets [a,a,b,b], loss only
   at positions 1 and 3. Nonzero a,b vary independently. Test new continuous
   values and show attention is necessary by removing it at evaluation.
5. Verify settling and gradient direction, retain checkpoints/predictions,
   report held-out loss and attention ablation, and distinguish nominal scalar
   energy from compact-model deviations. No full-silicon or energy-cost claim.

The existing four-token/two-head, width-four circuit is the initial architecture.
Normalization, buffers, current readout and the ReLU macro remain ideal in this
staged implementation. They are explicitly not counted as transistor-validated
hardware. Gain-cell storage is not needed: operands are continuously driven.

## Design and readiness

Start with a four-device differential triode multiplier. Gate differences
encode the first operand, drain differences encode the second; matched current
subtraction cancels common bias and Vds-squared terms in the long-channel law.
Native LEVEL=1 devices establish this circuit identity; compact-model checks
must separately quantify deviations, operating domain and dynamic stability.
The same product cell is required in q*k, p*k and p*q paths.

Use the existing independent energy/EP references. Physical finite differences
must additionally perturb the actual SPICE circuit, since agreement only with
an ideal numerical model is insufficient for the transistor claim.

Readiness is developmental, not a predeclared hypothesis test: freeze the
training initialization/optimizer/config after verifying implementation and
measuring throughput, before the published-source task run. Retain failed
attempts, record design revisions, and never relabel them as independent seeds.
Initial learning candidate: full-batch centered EP with beta=0.005 and Adam
controller, four combinations a,b in {-0.4,+0.4}; independent uniform held-out
values in [-0.4,+0.4]. The initial candidate will be tested in the numerical
reference before committing to the slower circuit training contract.

Acceptance for task learning: held-out masked MSE falls by at least 75% from
initialization, beats the zero-output predictor, and removing attention raises
the trained held-out MSE. Report the full learning curve and terminal result;
do not select the held-out-best epoch. Additional accuracy/energy guarantees
are not implied. Audit all required outcomes above before completing the goal.

## Resources and handoff

Local CPU, one thread. Initial development/readiness allowance: 30 minutes of
computation and 2 GB. First task evidence allowance: 30 minutes and 2 GB,
with exact update count fixed after throughput measurement. No GPU/remote run.
Owner: executing Codex agent, direct supervision with progress/log checks at
most 60 seconds apart while runs are active. Preserve all failed attempts.
Save full waveforms for diagnostic cases; training saves terminal measured
ports, generated netlists/logs and parameter/controller checkpoints, not every
transient sample. This is a declared retention choice before execution.

Development probes: `/tmp/mos-attention-learning/`.
Native output: `/home/filip/analog-research/drn/results/mos-attention-learning/`.
Use existing public `python -m ebl` and RunStore bundles. Formal task runs
start from a clean pinned checkout of a commit merged/pushed to the DRN trunk.
Keep the original ideal/BJT/analytic-MOS evidence unchanged.

Current state: specification recorded; implementing devices and circuit checks.
No task training launched yet.
