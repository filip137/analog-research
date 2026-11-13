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


## Local readiness result — complete, 13 November

Issue: https://github.com/filip137/on-chip-learning/issues/36.
Source: d65e30d36df253f61268c2058b52cebc6c1b823e, clean in the manifest.
Native bundle:
`/home/filip/analog-research/drn/results/convergent-energy-transformer/readiness/20261113T143844.756988Z-82444872-4ca77d4b`.
`python -m ebl runs inspect BUNDLE --verify-artifacts --require-complete` passed.
All 12 frozen acceptance checks pass. All declared cases/phases are present;
none was excluded. Native duration: 9.19 s; bundle size: 53,698,568 bytes.
The two preserved pytest SPICE attempts plus this bundle total 98,867,803 bytes,
within the declared 100 MB readiness allowance. Preparation/reading time and
per-task token usage were not separately instrumented.

| Measurement | Result |
| --- | ---: |
| Branch-current KCL max error | 1.37e-16 |
| Independent energy/dissipation identity error | 2.22e-16 |
| Maximum sampled energy increase (roundoff) | 1.07e-14 |
| Numerical terminal residual | 3.54e-13 |
| Free-state disagreement across three starts | 3.25e-13 |
| SPICE maximum transient error | 1.37e-6 V |
| SPICE maximum terminal-state error | 3.17e-12 V |
| SPICE port-function error | 6.39e-11 V |
| SPICE measured KCL residual | 3.67e-15 |
| Worst centered EP relative L2 error, beta=0.005 | 2.83e-5 |
| Worst SPICE-derived EP relative L2 error | 2.86e-5 |
| Worst independent directional FD relative error | 2.07e-9 |
| Minimum activation-coordinate Hessian eigenvalue | 0.5933 |
| Minimum Hessian eigenvalue after programming | 0.5885 |
| Single-example loss before / after EP update | 0.12377497 / 0.11397463 |

The centered EP error decreases by approximately four whenever beta halves.
All six parameter groups have nonzero reference gradients, including encoder,
query and key. The omitted-key-feedback control breaks the prescribed force by
2.95e-4 and the energy/dissipation identity by 9.36e-5. It is a contract-failure
control, not a claim that this particular broken trajectory has positive energy
slope everywhere. ReLU switching margin at the free equilibrium is 1.14e-4.

The explicit prototype has 20 RC state nodes, 992 weight resistor instances,
32 softmax BJTs, 192 multiplier macros, 16 normalized outputs and 24 rectifier
macros. Positive and negative signal rails use 80 ideal buffers each. There
are 80 logical trainable parameters (56 in attention/memory); physical copies
are programmed together by the software controller. The very small equilibrium
errors are properties of ideal matched simulation, not fabricated precision.

Verdict: the complete multi-token, multi-head ideal circuit has the intended
energy-decreasing dynamics, stable local equilibria and correct EP learning
signals on this frozen readiness case. One measured-circuit update decreases
its loss. This does not yet demonstrate dataset training, generalization,
transistor-level realizability, useful attention at scale, or efficiency.

## Scaling decision and next gated work

For the direct fully parallel differential-resistor layout, let
Wcore=2*H*head_width*D+M*D. The physical resistor count is
4*N*Wcore+6*N*P*D, because forward and return projections and different token
instances have distinct physical components. This count excludes summation,
readout and leak resistors, amplifiers, wiring and the programming controller.
Multiplier count is 3*H*N*N*head_width; softmax BJT count is H*N*N.

| Candidate | Logical total parameters | State capacitors | Weight resistor instances | Multiplier macros |
| --- | ---: | ---: | ---: | ---: |
| Current N=4,D=4,H=2,a=2,M=6,P=1 | 80 | 20 | 992 | 192 |
| Small N=16,D=64,H=4,a=16,M=256,P=4 | 26,112 | 1,088 | 1,597,440 | 49,152 |
| Medium N=49,D=256,H=8,a=32,M=1024,P=16 | 413,952 | 13,328 | 78,274,560 | 1,843,968 |

These counts make physical reuse/interconnect a research decision before
claiming a medium-scale chip. A circuit-equivalent numerical run alone is not
a full SPICE simulation at that size. Time multiplexing must come with a
specified state-storage/update schedule and renewed energy/EP verification.

Next work, in order:

1. Publish the completed circuit implementation and run its pinned-source
   evidence checkpoint. The trunk is clean and this branch is a fast-forward
   descendant; no conflict resolution or user-file changes are needed.
2. Prepare a strict small masked-image training pilot, initially the 26,112
   parameter configuration above. Fix dataset/split/masks, initialization,
   update count and wall/storage budgets before any substantial run. Measure
   settling failures and EP bias as well as held-out loss, and include matched
   exact-equilibrium-gradient, frozen, no-attention and trivial-fill controls.
3. Use the small trained model to select and validate the physical array reuse
   schedule and new transistor blocks. Scale to the medium configuration only
   after task learning and the mapping/settling gates pass.

Publication status: local branch and results are retained; no jobs are active.
The prior automatic approval review requires explicit conversation approval
before merging/pushing the shared DRN trunk. Selecting the convergent scientific
variant and staged circuit detail was not treated as that publication approval.
The requested small/mid-scale demonstration remains open. No dataset, GPU,
remote-host or transistor-level run has been launched.


## Publication authorization and branch — 13 November

The user approved the merge/push and pinned CPU verification, and requested a
new branch named energy transformer. The Git-compatible branch is
`energy-transformer`, created from `afdd832` in the existing isolated worktree.
It is the current development branch; the earlier `drn/` branch names remain
historical provenance. The explicit branch-name request overrides the normal
`drn/<id>` naming convention for this direction.

Publish `energy-transformer` and fast-forward `refactor/training-core` to the
same reviewed source. Run the unchanged strict circuit config from a detached
checkout of the published launch commit, with one CPU thread, <=600 seconds
and <=100 MB for this evidence bundle. The prior readiness bundles stay
unchanged. The output root is
`/home/filip/analog-research/drn/results/convergent-energy-transformer/evidence`.
The executing agent directly monitors the bounded run and verifies completion,
artifact hashes, case coverage and all acceptance criteria.


## Published BJT reference — 13 November

The approved publication and pinned check completed. Both `energy-transformer`
and `refactor/training-core` were pushed at
`99143e7531612ec20780458e7d3c15837d0c9a00`. The source was clean in the native
manifest. Bundle:
`/home/filip/analog-research/drn/results/convergent-energy-transformer/evidence/20261113T150322.467298Z-82444872-00a83316`.
All 12 acceptance checks and artifact-hash/completion inspection pass. Duration
8.73 s; maximum transient discrepancy 1.3653e-6 V; maximum SPICE-derived EP
relative error 2.8524e-5; loss 0.1237749663 -> 0.1139746266. All declared
phases and parameter groups are present. This is the published BJT baseline.

The user then requested replacing the BJT softmax with MOSFETs, referring to
Abbott and Machta (arXiv:2610.12571). Use the ideal weak-inversion MOSFET bank
in their equations (25)/(31) as the first staged replacement; preserve this
BJT baseline and the energy architecture. The full paper's dyadic saddle-point
learning architecture is not being substituted for our convergent model.
