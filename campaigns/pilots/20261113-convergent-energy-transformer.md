# Convergent Energy Transformer: analog circuit and EP

13 November 2026. User objective: demonstrate a small/mid-scale transformer
implemented as an analog circuit and trainable with equilibrium propagation.

## Decision and scientific scope

Advance from the fixed-memory retrieval primitive to a recurrent multi-token,
multi-head Energy Transformer. Start with masked-image completion: the corrupted
input stays clamped throughout every relaxation, and the output has an explicit
squared-error nudge port. Pretrained conventional transformer preservation is
not a requirement. Train from a declared random initialization first; the
published finite-step ET checkpoint is only a later transfer candidate.

The reference is [Høier, Kerjan and Scellier, Convergent Energy Transformer](https://openreview.net/forum?id=Qrfml76eWJ).
The primary indexed abstract establishes fixed-point EP as the goal; direct
access to the full PDF is blocked by OpenReview's browser verification. The
following is an independently derived circuit-oriented convergent ET variant,
not a claimed reproduction of the paper's complete architecture or results.
The attention and tied-memory terms follow the original
[Energy Transformer](https://arxiv.org/abs/2302.07253). An exact paper comparison
remains open until its full technical specification can be inspected.

## Explicit mathematical contract

Token state z has shape N x D. Output y and the fixed input x have shape N x P.
For each token, u=z-mean(z), s=sqrt(mean(u^2)+epsilon), g=u/s. Epsilon is
strictly positive; normalization has scalar unit gain. Define Phi(z)=D*s.
Then g=grad Phi and H=Hessian Phi is positive semidefinite, with the common
shift as its null direction and positive curvature in the centered subspace.

Independent projections Q_i^h=A_h g_i and K_j^h=B_h g_j produce
scores Q_i^h dot K_j^h and p_ij^h=softmax_j(kappa Q_i^h dot K_j^h).
All pairs, including self pairs, are allowed in the first version. The memory
activation is r_i=ReLU(U g_i). There is no independent value projection.

With a_i=L x_i+b_i and decoder R, the scalar energy is

    E(z,y;x) = sum_i [z_i dot g_i - Phi(z_i) - a_i dot g_i]
             - (1/kappa) sum_hi logsumexp_j(kappa Q_i^h dot K_j^h)
             - (1/2) sum_i ||ReLU(U g_i)||^2
             + (1/2) sum_i ||y_i||^2 - sum_i y_i dot R g_i.

The nudged energy is F_lambda=E+lambda*C, with
C=(1/2) sum_ip mask_ip*(y_ip-target_ip)^2. The target never enters the free
phase. Input, mask and weights are held fixed across free and nudged phases.

The token drive includes BOTH attention derivatives:

    A_h^T sum_j p_ij^h K_j^h + B_h^T sum_j p_ji^h Q_j^h.

The dynamics are

    tau_z zdot_i = -z_i + L x_i + b_i + attention_i
                   + U^T ReLU(U g_i) + R^T y_i,
    tau_y ydot_i = -y_i + R g_i - lambda mask_i*(y_i-target_i).

Writing r_z=-tau_z*zdot and r_y=-tau_y*ydot, independent differentiation gives

    grad_z F = H r_z,   grad_y F = r_y,
    dF/dt = -sum_i r_z_i^T H_i r_z_i/tau_z - ||r_y||^2/tau_y <= 0.

All coordinates within a token use the same time constant. Algebraic
normalization, products, projections and softmax are quasistatic in this
first circuit contract. Adding bandwidth or unmatched time constants requires
new analysis; it is not covered by the ideal Lyapunov proof.

Bounded g makes attention, memory and fixed-input forces bounded. For
1+lambda*mask>0 the leaky output remains bounded, hence so does z. The
Lyapunov equality set gives stationary centered tokens/output; the raw token
mean is a decoupled stable leak toward the mean drive. This supports approach
to the equilibrium set, not uniqueness or a global optimum for arbitrary
weights. EP requires a locally stable differentiable equilibrium branch.
The pilot checks the minimum activation-coordinate Hessian eigenvalue and
settling from multiple starts. Small residual alone is not a global proof.

## Circuit boundary

Use capacitors and leak resistors for z/y state nodes. Each signed linear
projection uses differential positive resistors into ideal virtual-ground
current-sensing ports, with identical logical coefficients in its return
projection. Attention score and p*K/p*Q products use explicit two-input ideal
analog multiplier macros. Each attention row has a matched-BJT softmax pool.
Normalization uses shared mean, squared-amplitude, square-root and division
macros; ReLU uses a precision-rectifier macro. Gain/bias and nudge currents are
controlled sources. Do not hide the whole vector field inside one behavioral
current source or call these macro models transistor implementations.

First-stage EP reads the free/+beta/-beta circuit states; phase statistics and
weight programming may be computed by a digital controller. No trajectory
autograd supplies its update. The parameter statistics are -g*x for L,
-g for b, -y*g for R, -ReLU(Ug)*g for U, and the corresponding
-p*K*g / -p*Q*g correlations for A/B, summed over tokens sharing a weight.
All physical copies of a logical parameter must be programmed together.

## Staged targets and success criteria

| Stage | Tokens / width / heads / head width / memory | Logical core weights | Evidence |
| --- | --- | ---: | --- |
| Circuit check | 4 / 4 / 2 / 2 / 6 | 56 | Explicit SPICE, KCL/energy and EP parity |
| Small task model | 16 / 64 / 4 / 16 / 256 | 24,576 | Held-out masked-image learning with EP |
| Medium task model | 49 / 256 / 8 / 32 / 1024 | 393,216 | Same circuit equations, calibrated scaling |

Core count is 2*H*head_width*D + M*D. Encoder, decoder and positional biases
are extra. These are recurrent models, not stacks of independently trained
blocks. Fully parallel replicated resistor mappings can have far more devices
than logical weights; time sharing changes execution and needs a matching
energy-descent schedule. Medium scale is a target, not validated feasibility.

1. Circuit: branch-current KCL matches the compact equations; numerical energy
   derivatives match the analytic dissipation; SPICE free/nudged equilibria
   match the independent ODE. Include a missing-key-feedback negative control.
2. EP: centered phase gradients match implicit equilibrium derivatives and
   selected finite differences; tightening settling and reducing beta give
   the expected convergence before numerical cancellation dominates.
3. Training: repeated EP updates improve held-out completion loss; compare
   the same architecture/initialization/data with exact equilibrium gradients,
   a frozen model, a no-attention ablation and a trivial reconstruction baseline.
   A single-sample loss reduction is readiness only.
4. Physical: replace new ideal macros by transistor-level blocks, recheck
   finite-bandwidth settling and EP under measured or explicitly modeled
   nonidealities. Only then make circuit power/latency estimates with the
   controller, state storage, weight copies and programming included.

## First implementation check: frozen specification

One deterministic four-token case, D=4, H=2, head width=2, M=6, P=1,
epsilon=0.5, kappa=1/sqrt(2), independent parameter seed 20261113.
Input x=[0.7,0,-0.4,0], target=[0.7,-0.2,-0.4,0.5], mask=[0,1,0,1].
Random normal A/B/U/L/R have standard deviation 0.08; b standard deviation
0.2. This initialization tests equations/learning direction, not useful
trained performance. Three starts: zeros and deterministic +/- perturbations.
Nudges: beta=0.04,0.02,0.01,0.005; both signs start from the same free state.
One learning step eta=0.05 updates all six parameter groups.

Acceptance: normalized numerical KCL residual <=1e-9; independent forces and
energy derivative agree to 1e-10; positive local activation Hessian; SPICE
terminal state error <=1e-5 V and residual <=1e-6; EP relative L2 error <=2e-3
per non-negligible group at beta=0.005 (absolute <=1e-8 for tiny groups);
finite-difference relative error <=2e-4; the stated single update lowers loss.
Preserve a failing scientific case; do not retune it into a pass.

Implementation/readiness allowance: local CPU, one thread, <=10 minutes total
numerical/SPICE execution, <=100 MB artifacts, no remote host/GPU/dataset run.
Probes live under /tmp/convergent-energy-transformer/. Native evidence will
use the public EBL CLI and a pinned pushed source under
/home/filip/analog-research/drn/results/convergent-energy-transformer/.
No substantial training launch is authorized merely by the scale targets.

## Handoff

Branch: drn/20261113-convergent-energy-transformer.
Worktree: /home/filip/analog-research/worktrees/drn-20261113-convergent-energy-transformer.
Base: e1b8db36beedb7809037b275d933ce07981116cb (existing ideal retrieval circuit).
Owner: executing Codex agent. No active experiment processes at specification.
The previous branch's merge/push approval is unresolved and is not inferred
from selecting this new scientific direction. Prepare concrete changes and
readiness evidence before any publication decision.


## Implementation handoff — 13 November

The user explicitly selected staged analog blocks first, then transistor
circuits. The complete 4-token/2-head model and its explicit SPICE export are
implemented. The mathematical model has 80 logical parameters and 20 RC state
nodes. Normalization and multiplication are individual analog macros; matrix
products are differential resistor-current sums and each softmax row uses BJTs.

Eight initial model/circuit tests passed, including four SPICE transients and a
SPICE-derived programming step. After CLI integration, 33 model/CLI/architecture
tests passed; one pre-existing version-dependent test skipped, and the unchanged
SPICE test was excluded from that second invocation. There was no GPU work.

Two understood implementation/reference defects were preserved in the readiness
history. The first netlist lacked a positive-rail input buffer, so a projection
loaded the output capacitor; adding the specified buffer corrected the circuit.
The ODE residual was sufficiently small for inference but could contaminate a
finite-difference subtraction; only the independent FD reference now polishes
its stationary root to 1e-12. No scientific case, EP beta, learning rate or
acceptance threshold was changed. Failed attempt artifacts remain under
/tmp/convergent-energy-transformer/spice-check-001/.

Exact native readiness command, from this worktree with one CPU thread:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ebl characterize --config examples/convergent_energy_transformer_circuit.json --output-dir /home/filip/analog-research/drn/results/convergent-energy-transformer/readiness
```

The strict config fixes the declared case and budget. The native bundle records
all parameters/boundaries, source/tool identities, three free starts, all four
+/- beta pairs, six SPICE transients (three free, two nudged, one after update),
all six parameter groups, independent implicit/FD references and the omitted-key
negative control. It is local readiness, not a published-source evidence run or
a held-out task-training study. The executing agent monitors it directly.
