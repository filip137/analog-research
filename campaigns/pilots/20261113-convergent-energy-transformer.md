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


## MOSFET replacement: predeclared contract

User-requested device change, 13 November. Primary source:
https://arxiv.org/pdf/2610.12571, section 2.5 equation (25), section 3.2
figure 6/equation (31), and appendix E's explicit ideal-device limitation.
Only the softmax device implementation changes. The frozen token case, weights,
energy, both attention return paths, nudges, learning rate and scientific
thresholds remain unchanged.

Use an ideal three-terminal weak-inversion MOSFET device law:

    Ids = I0 exp((Vg-Vth)/(n VT)) [exp(-Vs/VT)-exp(-Vd/VT)].

Devices in one bank have a common source, equal clamped drain voltages and a
fixed current sink. Thus the bracket is common and Ids_i/Itail is exactly
softmax(Vg_i/(n VT)) in this ideal model. Set
Vg_i=Vbias+kappa*n*VT*score_i to preserve our original softmax(kappa*score).
The derived activation, energy and EP statistics are unchanged. The full
paper's dyadic saddle-point architecture is not imported.

Declared constants: VT=0.026 V, n=1.5, Vth=0.7 V, I0=1e-8 A (the paper's
ideal-law parameters); Vbias=0.7 V, common drain=3 V, tail=1e-9 A. Tail<I0
keeps even a winner-take-all branch below the paper's specified current bound.
Check Vg<Vth+n*min(Vs,Vd) on every saved transient sample. Ideal current
readout amplification converts these nA currents back to the existing
normalized voltage ports; this is not a power estimate.

Implementation uses a named three-terminal SPICE subcircuit with a behavioral
channel-current law. It is an ideal MOSFET device model, not a foundry BSIM/EKV
card or a silicon measurement. The old explicit BJT option stays selectable
as `bjt`; the new default is `mosfet_weak_inversion`. The matched BJT config
is `examples/convergent_energy_transformer_bjt_reference.json`.

Verification: standalone equal/dominant logits, common-mode shift and a
centered finite-difference Jacobian versus diag(p)-p*p^T; then the existing
complete circuit/gradient/learning checks. Require weak-inversion margin>0,
maximum branch current/I0<1, current-law and tail-fraction error<1e-6.
Retain failures. No tuning of initial weights, nudges, learning rate or
scientific tolerances to obtain a pass. Absolute current solver tolerance
becomes 1e-18 A for the nA device law (the BJT path retains its old settings).

Budget for this replacement: one CPU thread, <=10 minutes combined checks,
<=200 MB including proportional test artifacts and one <=100 MB native bundle.
No dataset/GPU/remote run. Owner: executing Codex agent, direct monitoring.
Native output: /home/filip/analog-research/drn/results/convergent-energy-transformer/mosfet/evidence/.
The published BJT bundle remains the matched reference; do not rerun or
relabel it. Commit/push the passing implementation before the pinned native
MOSFET characterization, under the user's publication approval.

### MOSFET readiness and numerical startup

The standalone bank passes equal/dominant-logit, common-mode and Jacobian
checks. The first full-circuit attempt failed during Newton startup; the
ideal exponential behavioral device has no compact-model voltage limiting.
Seeding only the source/gate and normalization nodes still failed because
the clamped drain/supply nodes had inconsistent initial guesses. Seeding all
of those algebraic nodes consistently with the prescribed RC initial state
resolved the failure. The RC initial states and all component laws are
unchanged; the simulator still computes the transient and equilibria.
Failures remain in `/tmp/convergent-energy-transformer/mosfet-test-001` and
`mosfet-test-002`; the successful waveforms remain in `mosfet-test-003` and
`mosfet-test-004`. The third attempt's terminal pytest output was lost at a
context boundary, so the full-circuit test was repeated once with a saved log.

Readiness verification: ten mathematical/schema/netlist/standalone-bank tests
passed before the startup fix. The complete MOSFET circuit/gradient/update
test, ownership import checks, lazy CLI imports and public CLI tests then
passed together: 20 passed in 8.18 s. Log:
`/tmp/convergent-energy-transformer/mosfet-test-004-pytest.log`.
CPU only; the PyTorch CUDA-driver warning does not affect this CPU execution.
The readiness artifacts occupy about 104 MiB, within the combined 200 MB
allowance with room for the single bounded native evidence bundle.

Native launch, from a clean detached checkout of the next published commit:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/filip/miniconda3/bin/python -m ebl characterize --config examples/convergent_energy_transformer_circuit.json --output-dir /home/filip/analog-research/drn/results/convergent-energy-transformer/mosfet/evidence
```

Expected runtime: tens of seconds, hard limit 600 seconds. The executing
agent owns direct monitoring through artifact-hash validation and scientific
review. No task-training or finite-bandwidth hardware claim follows from this
characterization.

### Published MOSFET evidence and current handoff

Implementation source `457ff5ca6ea65e941362f62d1e110876f9317e7e` was pushed
on `energy-transformer` and `refactor/training-core` before execution. The
native manifest records that exact source with `dirty=false`. Evidence:
`/home/filip/analog-research/drn/results/convergent-energy-transformer/mosfet/evidence/20261113T152306.656808Z-3aa61c4a-e3fccb6d`.
Process exit 0; all 14 acceptance checks pass. Native completion and full
artifact-hash inspection pass. Coverage is complete for the frozen prototype:
three starts, eight numerical nudges, six SPICE transients, all six parameter
groups and one programming update. Native execution took 10.97 s and indexed
75,131,783 bytes; readiness plus evidence files total 182,673,998 bytes.
Preparation/reading/handoff time and token attribution are not separately
instrumented. No jobs remain active.

| Measurement | MOSFET result |
| --- | ---: |
| Maximum transient discrepancy | 1.3639e-6 V |
| Maximum equilibrium discrepancy | 1.8583e-12 V |
| Maximum per-group SPICE EP relative error | 2.8524e-5 |
| Loss before / after update | 0.123774966281 / 0.113974626633 |
| Minimum weak-inversion margin, saved samples | 0.1437567 V |
| Maximum branch current / I0 | 0.0250704 |
| Maximum current-law residual / tail | 4.0636e-14 |
| Maximum tail-fraction sum error | 5.9953e-15 |

The ideal MOSFET bank preserves the intended softmax, both attention return
paths, energy descent and EP statistics on this complete frozen case. The
matched BJT evidence above gives the same practical loss and gradient results.
These are exact-law circuit checks with deliberately ideal active blocks;
their tiny discrepancies do not establish fabricated-device accuracy, speed
or power. The standalone dominant-logit test exercises a wider softmax range
than this four-token case, whose attention remains close to uniform.

Current implementation: 20 RC state nodes, 992 differential weight resistor
instances, 32 ideal weak-inversion MOSFET device instances, 192 ideal product
macros, 16 normalized activation outputs and 24 ideal rectifiers. Mean,
variance, square root/division, buffers, current readouts and nudging sources
remain ideal. Phase statistics and tied-weight programming use software.
The BJT reference remains selectable; no foundry model is claimed.

The completed pinned source checkouts were removed after validation; source
commits and immutable bundles remain. The active development worktree is
`/home/filip/analog-research/worktrees/drn-20261113-convergent-energy-transformer`
on `energy-transformer`. The original primitive milestone (#32) is complete
and its merged worktree has been removed. The broader small/mid-scale
demonstration (#36) remains open. Next is to specify the small task-training
pilot and physical reuse mapping, followed by realistic transistor/bandwidth
checks under a declared protocol; no further run is launched by this result.
