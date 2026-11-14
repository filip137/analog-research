# Input-conditioned attention with independent values

Issue: https://github.com/filip137/on-chip-learning/issues/36
Branch: `energy-transformer`, continuing the user's explicitly named branch.
Status: implemented mathematical prototype and prepared comparison; no GPU
task run has launched for this pilot. The improvement goal remains active.

## Question and existing evidence

Can we make attention selective without losing a scalar energy, local EP
gradients or a phase cap of at most128 steps? The completed previous pilot is
[the MNIST comparison](20261114-energy-transformer-mnist.md), published at
`0e5af26`. Its constrained49-token/D64 model has30,784 parameters and test
MSE0.05441748, versus0.04222741 for the digital reference and0.06734631 for the
training mean image. The exact final32-step checkpoint has worst global
EP/BPTT cosine0.999998925 over eight minibatches. This establishes valid
learning, not strong routing: normal/uniform probe MSE is0.05466223/0.05466779.

The limitation is partly imposed by our sufficient contraction constraint,
not by EP or every possible energy transformer. For the existing attention,
normalized tokens satisfy ||g_i||<=sqrt(D). With a_h=||Q_h||2 and b_h=||K_h||2,
each score row has range Delta_h<=2*kappa*D*a_h*b_h. Therefore

    max_j p_ij <= exp(Delta_h)/(N-1+exp(Delta_h)).

For the saved49-token checkpoint the four Delta bounds are
0.3416220,0.3498056,0.3450556,0.3641150; no probability can exceed0.029112.
Uniform attention is1/49=0.020408. Even assigning the whole existing attention
budget0.2*sqrt(0.5) to one head permits Delta<=0.806856, or pmax<=0.044603.
The latter solves the same bound in `contractive.bound`, not an empirical
attention measurement. Merely widening this constrained recurrent model does
not address its routing problem.

Exact old checkpoint:
`/home/filip/analog-research/drn/results/energy-transformer-repair/training-003/more_tokens-energy_ep/attempt-01/20261114T194118.134634Z-02eafd41-aea2c686/checkpoints/final.pt`
SHA256 `afcb5c499450925c1bbb1058054ca1992beefda8b9ad3a114655eaeef2eaa0a8`.

## Proposed energy and circuit equations

Keep the old normalized state g=normalize(z), persistent encoder/position
drive, perfect-diode memory and nudged reconstruction state y. Replace the
old state-derived bilinear attention potential with the following input
conditioned potential. For each head, using fixed input features f_i:

    q_i = Q f_i,       k_j = K f_j,       v_j = V f_j,
    u_j = O v_j,       S_ij = t*kappa*q_i.T*k_j,
    L_ij = S_ij + eta*g_i.T*u_j,
    p_i = softmax(L_i),
    E_att = -(alpha/eta) sum_i [lse(L_i)-lse(S_i)].

The proposed v1 fixes alpha=1, eta=0.1 and score gain t=4. All heads have
independent Q/K/V/O. The subtracted input-only potential keeps energy and
parameter statistics centered; it cancels between EP phases. Crucially,

    -dE_att/dg_i = alpha * sum_j p_ij*u_j.

The reciprocal term eta*g_i.T*u_j in the scores is required. Deleting it while
keeping an independently trained Q/K pathway would invalidate this EP
construction. As eta tends to zero, the forward attention force approaches
ordinary input Q/K/V attention, but the EP response in routing shrinks too;
eta=0 is forbidden. No eta sweep is authorized or implied here.

An equivalent explicit attention-state energy minimizes over each probability
simplex:

    (alpha/eta)*[sum_j p_j log(p_j)-sum_j p_j S_j]
       - alpha*sum_j p_j*g.T*u_j.

Thus the softmax bank remains an algebraic minimum, just as perfect-diode
memory is an algebraic minimum. In the reduced model the complete energy is

    E = sum_i [z_i.T*g_i-D*s_i] - sum_i drive_i.T*g_i
        + E_att - 0.5*||ReLU(U*g)||^2
        + 0.5*||y||^2 - y.T*W*g
        + 0.5*beta*sum(mask*(y-target)^2).

State equations are fz=drive+attention+U.T*ReLU(Ug)+W.T*y-z and
fy=Wg-y-beta*mask*(y-target). With dz/dt=fz and dy/dt=fy/1.3,
grad_z(E)=-J_normalize*fz and grad_y(E)=-fy; J_normalize is positive
semidefinite, so continuous-time energy cannot increase. Numerical Euler
steps retain the existing energy backtracking and physical guards.

Input features have no learned preprocessing outside EP. Concatenate the raw
patch/observation inputs (2P entries) with a fixed D/2-dimensional sinusoidal
token code, center/normalize with epsilon0.5, and multiply by sqrt(D/F), where
F=2P+D/2. Then ||f_i||<=sqrt(D). The existing learned encoder and position
remain separate persistent drives and are trained through their energy
statistics. For N49,D64,P16, F64, this gives38,976 parameters, including an
8D^2 core. The old model has a6D^2 core. Fixed position codes are checkpoint
buffers, not additional trainable weights.

Analog mapping uses signed linear projection macros for Q/K/V/O and their
required returns; matched four-MOS triode multipliers form QK, (O.T*g)V.T
and probability/value products; a fixed-drain weak-inversion MOS bank forms
softmax. The score gain is a linear amplifier after QK multiplication. The
diode, normalization and RC state macros remain as in the prior model.
This is an ideal equation prototype. It does not establish transistor-level
normalization, programming, phase-statistic readout, bandwidth, time or power.

## Local EP statistics and contraction

Let p0=softmax(S), R=p-p0 and A=t*kappa. At fixed dynamical states, per-head
parameter-energy derivatives are

    dE/dQ = -(alpha*A/eta) * (R Kf).T f
    dE/dK = -(alpha*A/eta) * (R.T Qf).T f
    dE/dO = -alpha * g.T (p Vf)
    dE/dV = -alpha * (p.T O.T g).T f.

The implementation resolves batch/head/index transposes explicitly. Encoder,
position, diode-memory and decoder statistics retain their original form.
Centered EP subtracts statistics at the +beta and -beta equilibria and divides
by2*beta*masked-pixel count. All eight groups use these formulas, without
autograd in EP; BPTT exists only as an explicit control. Adam and spectral
projection remain digital programming operations.

The new attention Hessian for query i is

    alpha*eta*Cov_{p_i}(u_j).

If ||u_hj||<=R_h=sqrt(D)*||O_h||2*||V_h||2, its norm is at most
sum_h alpha*eta*R_h^2. This bound does not involve Q/K strength, the input
score gain or token count. Reuse separate normalized budgets0.2 attention,
0.2 memory and0.3 decoder, with the same1e-4 interior margin. Their sum is
below0.7*sqrt(epsilon), including ||W||2^2/(1-beta) for the negative nudge.
The normalized-state energy remains strongly convex on its centered domain,
and the same two-block fixed-point majorant has spectral radius below1.

Q/K/V/O each also have norm at most4/sqrt(D) for forward multiplier range.
Q/K are not reduced by the attention-curvature projection. V/O are jointly
reduced when their product exceeds the attention budget; memory and decoder
retain separate projections. Device checks include the new feedback and
value operands as well as the softmax bank. Ideal linear sums and local
gradient readout circuitry are outside those forward-domain checks.

## Readiness and limitations

The implementation is in `networks/energy_transformer/input_attention.py`.
The shared finite-time solver prepares immutable input ports once per phase
and includes all learned parameter groups in EP and gradient comparisons.
Old recurrent behavior and checkpoint defaults are retained. Public training
adds `energy_bp`, which backpropagates through exactly the same energy model,
solver and projection policy as its EP partner. Attention concentration,
normalized entropy and distance from uniform are recorded during evaluation.

The affected mathematical/runtime suite passes50 tests. The new tests cover
the energy/force identity, every parameter derivative, actual loss finite
differences for Q/K/V/O, EP/BPTT agreement, covariance-Hessian bound, weighted
contraction, added MOS operand guards, exact paired initialization and strict
config rejection. These are tiny CPU mathematical fixtures, mostly FP64,
not GPU FP32 task evidence. Projection idempotence is checked within numerical
roundoff, not bit equality after repeated SVDs.

A synthetic bounded-feature witness can put over98% on one of49 tokens while
satisfying the contraction and forward MOS guards. Its input amplitude is
outside the MNIST voltage range; it demonstrates the mathematical attention
primitive's capacity, not selectivity on this dataset. The gain t=4 is an
explicit architectural choice made before task execution, not a tuned result.

This is one input-attention block. Its output state settles, but input tokens
do not become independent recurrent Q/K/V states. Making upstream features
dynamical or stacking globally coupled blocks introduces mixed Hessian terms
and needs a new proof. Ordinary pretrained transformer weights are not a
drop-in mapping to these fixed features, tied diode memory and normalized
equilibrium output. The two architecture variants also differ in parameter
count and input features, so their comparison is not an isolated causal test
of one added matrix. The within-architecture EP/BPTT pair is the training-rule
control.

## Prepared experiment, pending run authorization

Question: does the added capacity improve learned reconstruction and routing
while retaining the user's trained-state gradient criterion?

1. Run250 EP warm-up updates of the new model at128 steps, then test caps
   32/64/128 on the same eight minibatches of eight held-out examples as before.
   Require global EP/BPTT, EP/reference and BPTT/reference cosine>=0.9 on
   every minibatch, with qualified512/1024 references. Report every group;
   use the smallest passing cap. If none passes, stop this candidate as a
   failed numerical result; do not weaken acceptance or launch full training.
2. From fresh paired initialization, run2,000 updates each for input-conditioned
   EP and input-conditioned BPTT at the selected cap. Run the existing
   constrained recurrent architecture with BPTT at32 steps as a third control.
   Reuse the verified old EP result and old digital reference as context.
3. Evaluate all10,000 test and5,000 validation images, plus the fixed256-image
   normal/uniform/zero probes. Repeat the full gradient check at the final
   new EP checkpoint. No outcome-dependent checkpoint selection.

Hold the existing MNIST files/split/order/masks, seed20261114, batch64,
Adam0.003, beta0.005, epsilon0.5, ideal MOS laws, diode memory and FP32-only
task computation fixed. AMP/TF32 remain off. Record initial tensors, file
hashes, all parameter counts, norm certificates and device margins.

Predeclared decision: promote the new block only if all numerical/domain/
coverage checks pass, EP test MSE is at least5% below the old EP result
(<=0.051697), and uniform attention increases the fixed-probe MSE by at least
1%. Compare new EP/BPTT error and gradient scale as separate training-method
evidence; a relative EP task gap<=5% is the target, not an acceptance threshold
to hide a worse BP control. If new BP improves but EP does not, investigate
training; if neither improves, the architecture hypothesis is unsupported.
One seed and one task remain exploratory. Retrieval and larger models are
follow-ups, not silently included extra experiments.

Prepared configs, all under `examples/energy_transformer_mnist/input_attention/`:

- `warmup-energy_ep.json`:250 updates,128 steps, limited development evaluation.
- `gradient-selection.json`:public `energy_transformer_iterations.v1` validation.
- `input-energy_ep.json`, `input-energy_bp.json`:provisional32-step full pair;
  both must be frozen to the selected cap before their launch commit.
- `recurrent-energy_bp.json`:same old32-step architecture, now trained with BP.

Use `python -m ebl train --config <config> --output-dir <case-root>` and
`python -m ebl validate --config <diagnostic-config> --weights <exact-checkpoint>
--output-dir <diagnostic-root>`. Preserve source/config/checkpoint identities
in native bundles. Proposed output root:
`/home/filip/analog-research/drn/results/energy-transformer-input-attention/`.

Proposed additional allowance: at most60 minutes on the local RTX3090,
including smokes, warm-up, diagnostics, full cases and operational retries;
charge it against the remaining prior six-hour campaign allowance, do not
reset the old debit (<3.14 hours). Expected duration is20–45 minutes, an
estimate pending measurement for this architecture. Storage<=2GB, keep
initial,250-update and final/checkpoint artifacts according to the existing
runner; never prune failed attempts. No remote GPU or concurrent sharing of
an occupied GPU. Use a common absolute deadline through the existing runtime
deadline interface; no per-case budget reset.

Before substantial execution: obtain authorization for this changed scientific
comparison; inspect live GPU occupancy; run proportional GPU FP32 smokes;
commit/push the feature source and stage a clean pinned launch checkout. Shared
trunk publication remains separately pending after the earlier approval-review
rejection; do not retry it under this pilot. One owner must monitor actual
native metrics and checkpoints and inspect final bundles with artifact hashes.
No GPU handles, launch commit or live monitoring assignment exist yet.

Current owner: root. Next action: review/authorize the prepared comparison,
then execute through evidence collection and interpretation. This preparation
does not establish a performance improvement and does not complete the active
improvement goal. The new pilot remains open; the old repair stays complete.
