# Input-conditioned attention with independent values

Issue: https://github.com/filip137/on-chip-learning/issues/36
Branch: `energy-transformer`, continuing the user's explicitly named branch.
Status: the original and singular-clipping comparisons are complete. Singular
clipping fixes rank collapse and improves test error2.1%, but falls short of
promotion and digital quality. The user now explicitly requests an Explorer
round listing alternatives and testing them; preparation is in progress.

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

The affected mathematical/runtime suite passes55 tests. The new tests cover
the energy/force identity, every parameter derivative, actual loss finite
differences for Q/K/V/O, EP/BPTT agreement, covariance-Hessian bound, weighted
contraction, added MOS operand guards, exact paired initialization and strict
config rejection. These are tiny CPU mathematical fixtures, mostly FP64,
not GPU FP32 task evidence. Projection idempotence is checked within numerical
roundoff, not bit equality after repeated SVDs.

A subsequent FP32 toy check exposed cancellation that a global cosine hid:
direct phase-statistic subtraction gave query/key cosines0.7867/0.8172,
despite global0.99999999. The fix evaluates the exact finite softmax contrast
using expm1 of the state-induced logit difference before adding common input
scores. For p=softmax(l), e=expm1(delta_l), and a=sum(p*e):

    softmax(l+delta_l)-p = p*(e-a)/(1+a).

Product statistics are likewise expanded as differences of factors. This is
an algebraic identity for a finite contrast, not a linear response
approximation or an FP64 fallback. Its FP64 equivalence is tested against
ordinary phase subtraction, including large state differences. The repaired
FP32 toy has global cosine0.99999999976, query/key cosines0.999999990/
0.999999978, and worst group relative error0.000324. These remain CPU
mathematical fixtures; GPU/task qualification is still pending. Physical
measurement of such small phase-current differences would need its own
precision/noise validation and is not established by stable simulator math.

Consequently the prepared trained-state diagnostic requires query, key, value
and output group cosines>=0.9 in addition to the existing global gate; those
groups also must pass512/1024 reference stability. Historical diagnostics
retain their old default global-only rule. This stricter requirement was
declared before any new task run.

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

## Authorized experiment

Question: does the added capacity improve learned reconstruction and routing
while retaining the user's trained-state gradient criterion?

1. Run250 EP warm-up updates of the new model at128 steps, then test caps
   32/64/128 on the same eight minibatches of eight held-out examples as before.
   Require global and Q/K/V/output EP/BPTT, EP/reference and BPTT/reference
   cosine>=0.9 on every minibatch, with qualified512/1024 references. Report every group;
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

Frozen candidate configs, all under `examples/energy_transformer_mnist/input_attention/`:

- `warmup-energy_ep.json`:250 updates,128 steps, limited development evaluation.
- `gradient-selection.json`:public `energy_transformer_iterations.v1` validation.
- `input-energy_ep.json`, `input-energy_bp.json`:provisional32-step full pair;
  the launcher selects the already committed `caps/32`, `caps/64` or `caps/128`
  pair after qualification. Each has a matching final diagnostic config.
- `recurrent-energy_bp.json`:same old32-step architecture, now trained with BP.

Use `python -m ebl train --config <config> --output-dir <case-root>` and
`python -m ebl validate --config <diagnostic-config> --weights <exact-checkpoint>
--output-dir <diagnostic-root>`. Preserve source/config/checkpoint identities
in native bundles. Proposed output root:
`/home/filip/analog-research/drn/results/energy-transformer-input-attention/`.

User-authorized additional allowance: at most60 minutes on the local RTX3090,
including smokes, warm-up, diagnostics, full cases and operational retries;
charge it against the remaining prior six-hour campaign allowance, do not
reset the old debit (<3.14 hours). Expected duration is20–45 minutes, an
estimate pending measurement for this architecture. Storage<=2GB, keep
initial,250-update and final/checkpoint artifacts according to the existing
runner; never prune failed attempts. No remote GPU or concurrent sharing of
an occupied GPU. Use a common absolute deadline through the existing runtime
deadline interface; no per-case budget reset.

Before substantial execution: inspect live GPU occupancy; run proportional GPU FP32 smokes;
commit/push the feature source and stage a clean pinned launch checkout. Shared
trunk publication remains separately pending after the earlier approval-review
rejection; do not retry it under this pilot. One owner must monitor actual
native metrics and checkpoints and inspect final bundles with artifact hashes.
The actual launch source, handles and completed execution are recorded below.

Current owner: root. The user replied "yes" to the explicit one-hour
comparison request; that comparison has now been collected and reviewed.
The next proposed projection control awaits its own authorization. This preparation
does not establish a performance improvement and does not complete the active
improvement goal. The new pilot remains open; the old repair stays complete.


## Launch preparation and bounded-input capacity check

A stronger constructed witness now uses only the actual input voltage range:
24 visible tokens with pixel values in[-0.4,0.4] and indicator1,25 hidden
tokens with pixel/indicator0. Explicit norm-capped Q/K maps give probability
0.4707368 from a hidden query to a selected visible token, versus uniform
0.0204082. The normalized curvature certificate is0.4557171, contraction
radius0.5397969; triode/drain/weak-inversion margins are0.51959/0.18772/0.11919V.
This is an untrained mathematical witness, not a sampled MNIST image or the
benchmark's exact coarse-region mask distribution. It removes the earlier
out-of-range-amplitude caveat for representability, not the need for learning.

`experiments.run_energy_transformer_input_attention` serially invokes the
public CLI for two GPU smokes,250-update warm-up, trained gradient selection,
three full arms and the final EP gradient check. It retains a tiny CUDA context
between children, writes PID/status/native metric growth/coverage, enforces one
absolute deadline, verifies native artifacts and permits only the existing
zero-update CUDA initialization retry. Other operational failures stop for
diagnosis; failed scientific qualification prevents full training. The warm-up
has600s training/1200s total to preserve the runner's600s evaluation reserve;
this does not extend the common3600s ceiling. All possible caps and full-pair
configs are committed before launch. The root monitor owns scientific review;
run-watch owns deterministic process/artifact observations and bounded incidents.

## Active execution —14November2026

Clean published launch source `f27b8c8225ab0f727d90bb6280e62e5eb73b96ab`:
`/home/filip/analog-research/drn/results/energy-transformer-input-attention/source-f27b8c8`.
Command: `python -u -m experiments.run_energy_transformer_input_attention
--output-root /home/filip/analog-research/drn/results/energy-transformer-input-attention/benchmark-001
--seconds 3600`.
Started20:43:14.786UTC; absolute deadline21:43:14.786UTC. Supervisor2573088;
current child is recorded in `child.pid`. No other GPU compute process was
present at admission.

Both two-update GPU smokes passed full declared coverage and native artifact
hash checks on their first attempts. At20:45:23UTC, warm-up child2573434 had
reached156/250 updates in112.6training seconds, residual1.79e-7 and normalized
bound0.699928. This is launch progress, not scientific acceptance.

Persistent run-watch2573932 is armed with dispatch, verified process/GPU and
semantic-artifact observations, no probe errors, and60-second cadence. Its
config, state, incident budgets and explicit recovery scope are under the run
root (`watch.json`, `watch-state/`, `handoff.md`). Last successful observation
20:45UTC; next due20:46UTC. Parent root owns scientific review and repairs;
watch owns routine monitoring and bounded incident diagnosis. All eight
expected native cases are named by the launcher; full arms are conditional
on warm-up qualification. Read `status.json`, `coverage.json`, `selection.json`,
native `metrics.jsonl` and final `result.json`. Never replace an existing
attempt or reset the absolute deadline during recovery.

At20:47UTC, the250-update checkpoint qualified32 steps (the smallest tested
cap). All eight reference batches passed; minimum global EP/BPTT cosine was
0.99999985093 and minimum Q/K/V/output cosine0.99999998146. Fixed-probe MSE
was0.05684640; uniform attention gave0.05752294 (+1.19%), zero attention
0.11502143. This warm-up is development evidence, not the final promotion
decision. The full32-step EP run was at1346/2000 updates at20:52:16UTC,
with maximum recorded phase residual9.54e-7; subsequent matched BPTT runs
and final same-cohort gradient qualification remain pending.

Independent collection uses `python -m experiments.review_energy_transformer_input_attention
<benchmark-root> --old-root /home/filip/analog-research/drn/results/energy-transformer-repair/training-003`.
It rereads native summaries/checkpoints/manifests, compares initial maps,
cohorts and final sampling/masking RNGs, audits all training bounds/margins,
verifies checkpoint identity and selected-group gradient gates, and produces
the predeclared decision plus identical-cohort validation/reconstruction figures.
The script is prepared; no final result or performance claim exists yet.

## Candidate result and proposed projection follow-up

The input-attention EP arm completed2,000updates and full held-out coverage:
test MSE0.0544708864, versus old EP0.0544174788. Its uniform-attention penalty
is0.191259%, below the1% target. Matched BPTT also completed and reached
0.0544720605, effectively the same result. The current candidate does not
meet the predeclared promotion rule. The old-model BPTT control, final gradient diagnostic and independent audit
subsequently passed; the complete comparison is recorded below.

Read-only final-weight analysis identifies another capacity limitation. Define
stable rank as ||M||F^2/||M||2^2 (a continuous effective dimension, not exact
algebraic rank). Old EP's query/key heads have stable rank1.0001–1.0002 and
memory1.00004. New EP's query/key heads are1.0001–1.0028, values1.0001–1.0002,
output heads1.011–1.021 and memory1.00035. The digital reference retains
query/key ranks2.49–4.92 and FF ranks6.28/3.95. Thus parameter count substantially
overstates the number of strong learned directions in either constrained EP model.

The existing programming limiter rescales a whole matrix when its top singular
value exceeds its bound. Each such update also shrinks all weaker directions,
even those already well below the bound. This mechanism can erase useful modes
over repeated updates. It is a hypothesis for the observed collapse, not a
causal conclusion from weights alone; task-driven concentration or other
architectural effects remain possible.

Prepared intervention: clip only singular values exceeding the selected norm
radius. Memory and decoder then use the Euclidean projection onto their
spectral-norm balls. Q/K retain the same physical operand norm cap. For coupled
V/O constraints, use the existing common reduction to choose target radii and
clip each matrix to its radius; this is not an exact Euclidean projection onto
the full joint nonconvex constraint set. The same scalar energy, feedback0.1,
score gain4, independent Q/K/V/O, separate0.2/0.2/0.3 budgets, local EP rule and
physical laws remain unchanged. Initialization deliberately keeps the original
radial projection, so the new controls start from identical tensors.

Code: `networks/energy_transformer/singular_projection.py`; explicit policy
`input_attention_singular_v1`. The60-test suite verifies preservation of
unsaturated singular values, prevention of erasure under repeated dominant-mode
updates, identical initial maps, unchanged contraction/physical norm limits,
and EP/BPTT agreement after projection. This is prepared mathematical evidence;
no singular-clipping task run has launched.

Proposed follow-up, pending explicit scientific authorization: reuse the current
radial EP/BPTT results and run one fresh singular-clipping EP/BPTT pair,2,000
updates each, with the same250-update warm-up,32/64/128 selection and final
eight-batch gradient gate. Exact configs are under
`examples/energy_transformer_mnist/input_attention/singular_clip/`, including
both GPU smokes and all frozen cap alternatives. Invoke the same public
`python -m ebl train/validate` commands. No seed, LR, data, task, architecture,
norm budget, beta or physical-domain change is included. Report stable-rank
trajectories at the already saved250-update checkpoints as well as quality,
routing ablations and gradient agreement. Retain the original5% quality and1%
routing promotion thresholds; do not lower them after seeing this result.

Use only the remainder of the already authorized one-hour window, ending
21:43:14.786UTC, and the existing2GB storage ceiling. Expected extra execution
is15–20minutes; exact throughput must be checked in the smokes/warm-up. No
extension, independent seed, extra hyperparameter setting or silent scientific
retry is included. The current pipeline continues unchanged while the user
reviews this proposal. Further task runs require the user's explicit answer;
automatic goal continuation is not run authorization.


## Verified first comparison —21:04UTC

All eight native cases completed on their first attempts in1241.37seconds
(20.69minutes), with artifact-hash inspection. All three full arms completed
2,000updates,10,000test images,5,000validation images and three256-image
attention probes. Reused old EP/digital controls retain their original native
identities. The independent audit passes all23 checks, including exact paired
initial maps, identical datasets/cohorts/final sampling and masking RNGs,
FP32 finite weights, every training constraint and physical margin, final
checkpoint identity and gradient qualification. No gate was relaxed.

| Model | Parameters | Test MSE | Uniform penalty | Training seconds |
| --- | ---: | ---: | ---: | ---: |
| Input attention, EP |38,976|0.0544708864|0.191259%|429.95|
| Input attention, matched BPTT |38,976|0.0544720605|0.191232%|249.02|
| Old recurrent attention, EP |30,784|0.0544174788|0.010169%|321.39|
| Old recurrent attention, matched BPTT |30,784|0.0544183724|0.010106%|196.29|
| Digital reference |30,784|0.0422274134|38.099749%|7.02|

The new EP result is0.0981% worse than old EP, and differs from its own BPTT
control by only0.00216%. Zero attention worsens its probe MSE25.99%, whereas
uniform attention worsens it only0.191%: the value path supplies useful global
information, but selective routing adds little. Mean maximum attention is
0.039778 and normalized entropy0.987847. These are one-seed results; the
different architectures/parameter counts are not an isolated causal comparison.
GPU timing/memory remains simulator cost, not an analog hardware claim.

Final32-step EP/BPTT minimum cosine is0.99999979047 globally and0.99999969054
over Q/K/V/output; all-group minimum is0.99998794380. EP/reference minima are
0.99999979061/0.99999972087. Every512/1024 reference qualifies. Final EP
checkpoint SHA256 is `50b00328d14ffdc1c87ce838c568532d020e73c15876728421a5538b998d4e8a`.
New EP's maximum training residual is9.54e-7; minimum triode/drain/weak margins
are0.57457/0.21333/0.15285V. Maximum normalized bound0.699945 and contraction
radius0.803534 stay within the declared limits. The old BPTT control has a
maximum residual1.19e-6, slightly above the historical stopping tolerance;
fixed-cap execution does not claim every phase reached that tolerance.

Evidence root:
`/home/filip/analog-research/drn/results/energy-transformer-input-attention/benchmark-001/analysis/`.
`review.json` contains all checks/identities; `comparison.md` the table;
`weight-ranks.json` records hashed checkpoint spectra at initialization and
every250updates. Validation, detailed validation, weight-rank and reconstruction
figures were visually inspected: EP/BPTT curves/reconstructions overlap, both
constrained architectures plateau above the digital reference, and memory/query
stable ranks collapse near1 early in training. This supports, but does not
causally prove, the projection mechanism proposed above.

Verdict: do not promote this architecture under the frozen rule. Numerical EP
quality is not the present bottleneck; fixing how the programming constraint
preserves multiple directions is the next controlled question. The existing
watcher reached terminal verification and was stopped after collection; no
task workers remain. Runtime storage was approximately124MB before final
additional figures, below2GB. The original21:43:14.786UTC deadline remains
unchanged for any separately approved follow-up. No new task has launched.


## Follow-up orchestration prepared —21:19UTC

The existing serial runner now accepts `--variant singular_clip --prior-root`.
It requires the original comparison's complete numerical review, reuses the
old recurrent BPTT control, and inherits the original absolute deadline rather
than resetting an hour. The2GB allowance covers both output roots together.
The collector explicitly checks the different projection policies, identical
initial tensors/data/RNGs, otherwise matching scientific settings and unchanged
energy/dynamics source files. The cap-selection rule is unchanged; if the
selected cap differs, that difference is reported rather than hidden.

Prepared launch command, **not executed**:

```bash
python -u -m experiments.run_energy_transformer_input_attention \
  --variant singular_clip \
  --prior-root /home/filip/analog-research/drn/results/energy-transformer-input-attention/benchmark-001 \
  --output-root /home/filip/analog-research/drn/results/energy-transformer-input-attention/singular-001
```

Run only from a clean pinned source after explicit approval, with a fresh
watch identity and the existing21:43:14.786UTC deadline. The matching collector
uses the new output root, `--radial-root` pointing to `benchmark-001` and
`--old-root` pointing to `energy-transformer-repair/training-003`. Neither a
new scientific run nor an extension is authorized by automatic continuation.

Verification: all61 relevant mathematical/schema/gradient/runner tests pass.
A read-only replay of the existing bundles through the updated collector
reproduces all23 prior checks, the comparison values and the verdict exactly;
its temporary outputs are outside the native evidence tree. This verifies
backward-compatible collection, not the unexecuted follow-up's task quality.


## Approved follow-up running —21:24UTC

The user approved the singular-clipping comparison and explicitly requested
standing authorization for follow-ups within approved goals and budgets. The
shared research AGENTS.md now records that rule (commit e7daecf); this branch's
AGENTS.md mirrors it so backend agents do not reinstate per-variant approval.
Existing deadlines and budgets remain binding.

The serial pipeline launched at21:24:11UTC from clean pinned source
`a76aaa24d913745408034204091d2bc4a375db15`, at
`/home/filip/analog-research/drn/results/energy-transformer-input-attention/source-a76aaa2`.
Output root is
`/home/filip/analog-research/drn/results/energy-transformer-input-attention/singular-001`.
Supervisor2599030; watcher2599199; current native child is in `child.pid`.
Both FP32 GPU smokes passed on their first attempts with complete native
artifact inspection; the250-update warm-up is producing training metrics.
Local RTX3090 was idle with ample memory at launch. No unrelated process changed.

The command is the prepared `--variant singular_clip --prior-root benchmark-001`
composition above. It retains the21:43:14.786UTC hard deadline (about19minutes
at launch),2GB combined storage allowance, original initialization and scientific
settings. Expected execution15–20minutes; actual throughput is monitored before
claiming the complete pair fits. Failed/partial results remain evidence, and no
budget extension or acceptance-threshold change is implied.

The verified persistent watch owns routine process/GPU/artifact/log observations
at60-second cadence and bounded incident diagnosis. Root owns repairs and
scientific interpretation. Watch/config/current handoff are under the output
root; `status.json`, `coverage.json`, `metrics.jsonl`, `supervisor.log` and native
metrics/checkpoints are progress evidence. Final collection uses the prepared
reviewer with the original radial and digital/control roots. Next check:
warm-up completion and32/64/128 qualification, then full-arm progress.


## Partial result and operational budgeting defect —22:24UTC review

The follow-up terminated at21:34UTC after588.64seconds. EP completed1,341 of
2,000updates; BPTT completed0. Both arms still completed10,000test and
5,000validation images, but this is incomplete training coverage and the
final gradient gate did not run. Preserve `result.json` as written: its
`scientific_checks_failed` label conflates this operational budget stop with
scientific failure. No numerical divergence was reported.

Cause: the runtime always reserved600seconds for final evaluation whenever
configured wall time exceeded600seconds, even after the shared absolute deadline
shortened a case. EP stopped when that reserve began; BPTT then had no permitted
training interval. EP evaluation plus setup actually took23.89seconds. The whole
pipeline exited with554.74seconds left in the original window. The watcher
correctly detected incomplete coverage at21:34:19, but its overly restricted
handoff returned a blocked decision to root instead of repairing the operational
fault. Root did not act before expiry. This is an execution/supervision failure,
not evidence that the proposed model failed numerically. The stale watcher has
now been stopped after reviewing terminal evidence.

Verified partial evidence (`singular-001/analysis/partial-review.json`): all seven
checks pass for matching initial maps, configurations except projection,
cohorts, sampling/masking RNGs at the common1,250-update checkpoint, native
inspections, all1,341 physical/contraction records and finite FP32 final weights.
At1,250updates the fixed256-image validation MSE is0.053901407 with singular
clipping versus0.054899889 with radial scaling:1.8187% lower error at equal
updates. The partial final checkpoint has test MSE0.053574972, with full test
coverage, and uniform-attention penalty0.7534%. No full matched EP/BPTT quality
claim or promotion is supported. The250-update32-step gradient gate did pass:
minimum global EP/BPTT cosine0.999995953, Q/K/V/output minimum0.999997398;
512/1024 references qualify. These are not final-checkpoint gradient results.

The projection intervention prevents the observed rank collapse: partial final
memory stable rank54.83, Q/K roughly8.8–10.5 and values5.2–7.2, compared with
approximately1 in the radial baseline. The saved trajectories show the same
difference at matched update counts. The effective-rank increase yields a modest
quality gain so far; it does not establish digital-baseline competitiveness.
`partial-progress.png/svg` was visually inspected; `partial-weight-ranks.json`
retains hashed checkpoint spectra. Final partial EP checkpoint SHA256:
`e0f84c1488caefa0b745011995e4071414f15305475fd672570a41ea71708a2c`.

Operational repair: reserve final evaluation using the already-required initial
probe's measured throughput, scaled to the actual requested evaluation work
(including both trained and initial test forwards), with3x headroom and a10second
floor. Log the estimate and available time in native `artifacts/budget.json`.
Keep both the absolute wall deadline and remaining cumulative training allowance
binding. Incomplete probes retain the conservative fallback. Classify time-limit
incompletion as `budget_incomplete`, separately from failed science. All64
relevant tests pass, including the reproduced580second-remaining failure,
slow-evaluation safety, charged prior training time and explicit-only extension.
No repaired GPU execution has launched after the expired window.

Prepared recovery: one fresh identical serial comparison under `singular-002`,
including proportional smokes, the250-update qualification, both2,000-update
arms and final gradient gate. Fresh paired training avoids ambiguous replay of
a final checkpoint saved after a timed-out phase may have consumed its next
sampling/masking draws. Preserve and charge `singular-001`; do not call it a
replicate or overwrite it. The public composition now supports an explicitly
recorded `--approved-deadline-unix` extension and `--charge-root` for retained
attempts. Without an explicit extension, the expired original deadline remains
binding. Proposed new allowance:20minutes on the same local RTX3090, within the
original aggregate6h/2GB ceilings; this time extension awaits user authorization.
The next watcher handoff must permit understood operational repairs within that
allowance rather than referring every fault back to an inactive root.


## Approved recovery running —22:35UTC

The user explicitly approved the20-minute extension. `singular-002` launched
at22:35:15.365UTC and is limited to22:55:15.365UTC (Unix1794696915.3653052),
including all smokes, diagnostics, training/evaluation and operational retries.
The prior elapsed1241.37s comparison and588.64s partial attempt stay charged;
aggregate6h and combined2GB limits remain. No new scientific setting changed.

Clean source is `c0df4ab99ed19c954acb8d36104edec6055dec07`, pinned at
`/home/filip/analog-research/drn/results/energy-transformer-input-attention/source-c0df4ab`.
Output root is
`/home/filip/analog-research/drn/results/energy-transformer-input-attention/singular-002`.
Supervisor2654842; persistent watcher2655250; current native child is in
`child.pid`. Local RTX3090 was idle and had sufficient memory at admission.

Both FP32 two-update smokes passed on their first attempts with complete native
artifact inspection. The250-update warm-up is producing metrics. Its native
`artifacts/budget.json` confirms a13.17s measured evaluation reserve instead of
the previous600s constant, leaving its full600s training allowance available.
This is readiness/progress evidence, not final quality or gradient validation.

The command uses the prepared serial composition with `--variant singular_clip`,
`--prior-root .../benchmark-001`, `--charge-root .../singular-001`,
`--seconds 1200`, `--approved-deadline-unix 1794696915.3653052` and
`--output-root .../singular-002`. Exactly one pipeline owns the cases.

Monitoring ownership is corrected: the verified persistent watch observes
process/GPU/artifact/log evidence every60seconds. Its bounded incident agent
is explicitly delegated understood operational repairs within the SAME deadline
and terminal collection/scientific review, including the existing collector,
figures and the current pilot note. It must preserve attempts, test and pin
repairs, and may not retune science or extend time. It must not simply refer an
already-authorized operational repair to an inactive root. Terminal review
separates full coverage and numerical checks from promotion; partial data cannot
be called complete. The current runtime handoff contains exact commands and
artifact paths. Next milestone: warm-up/gradient qualification, then full EP
and BPTT coverage and final gradient gate.


## Corrected comparison terminal review —14 November,22:55UTC

`singular-002` completed all seven first-attempt cases in1033.43seconds
(17.22minutes), before the unchanged22:55:15.365UTC hard deadline. Both
smokes,250-update warm-up,8x8 cap qualification at32/64/128 with512/1024
references, both2,000-update full arms and final8x8 gradient diagnostic are
terminal. The selected cap is32. Each full arm covers10,000test images,
5,000validation images and all three256-image probes. All seven native
full-artifact-hash inspection receipts are valid with no errors; the
assigned independent CPU collector passes all31 checks with no failed checks.
No task worker remains. No GPU work, retries or scientific changes were added
during terminal collection.

Pinned launch source remains `c0df4ab99ed19c954acb8d36104edec6055dec07`.
Final EP checkpoint SHA256:
`fa7df3d08716eb812297e5f6bf9537e211ec77b0e5cfa4065d5868f72f9ec23a`.
Warm-up checkpoint SHA256:
`03dd7044dc7b60c7d1c6b4f86c4b34c8c02f17116a7808161102cd6569b8c32c`.
Paired initial tensors, dataset/cohorts and final sampling/masking RNGs match;
projection is the declared intervention relative to the retained radial pair.
Prior1241.37s benchmark and588.64s partial attempt remain charged, giving
2863.44s across these three roots; earlier campaign debits are not reset.
Their combined outputs occupy approximately322MB, below the aggregate2GB
ceiling. The launcher receipt, rather than inner training time, determines
this attempt's1033.43s debit. No allowance correction or extension was needed.

| Model | Test MSE | Uniform probe penalty | Training seconds |
| --- | ---: | ---: | ---: |
| Singular clip, EP |0.0533348291|0.787130%|432.27|
| Singular clip, matched BPTT |0.0533306156|0.787358%|258.68|
| Radial limit, EP |0.0544708864|0.191259%|429.95|
| Radial limit, matched BPTT |0.0544720605|0.191232%|249.02|
| Old recurrent, EP |0.0544174788|0.010169%|321.39|
| Old recurrent, matched BPTT |0.0544183724|0.010106%|196.29|
| Digital reference |0.0422274134|38.099749%|7.02|

Final EP/BPTT cosine minima are0.9999956037 globally and0.9999960055 across
the required Q/K/V/output groups; all-group minimum is0.9998441123.
EP/reference minima are0.9999955978 globally and0.9999959985 for required
groups. Qualified512/1024 references and the same gradient cohort are retained.
The EP/BPTT test gap is0.007901%, supporting numerical agreement.

All required figures were directly inspected with the image tool:
`analysis/validation-detail.png`, `weight-ranks.png` and `reconstructions.png`.
The first shows nearly coincident EP/BPTT update curves, improving over the
radial/recurrent controls while remaining above digital. The second shows
final memory stable rank57.09 and query-head ranks11.08–11.53 for singular
EP, compared with approximately1 under radial projection; matched BPTT tracks
these trajectories closely. Reconstructions across all six shown examples
are similar for EP/BPTT, with substantial blur/ambiguity remaining in missing
regions. `analysis/figure-review.json` records exact inspected image hashes
and observations. `analysis/review.json`, `comparison.md` and
`weight-ranks.json` hold the full evidence under the explicit `singular-002`
root above.

Verdict: complete coverage and numerical validity, but **do not promote**.
Singular clipping improves EP test MSE1.9895% over old EP (2.0856% over radial
EP), below the frozen5% requirement. Its0.7871% uniform-attention penalty is
below the required1%. Broader learned directions and stronger attention
(mean maximum0.074395, normalized entropy0.944194) support a real effect of
projection, but it is insufficient under the original decision rule. One seed
and ideal circuit simulation limit generalization; GPU costs are not analog
hardware measurements. No retuning, new experiments or budget extensions are
part of this review. The watcher may stop this run after this complete,
scientifically negative result; the broader improvement objective remains
unresolved.


## Explorer round —15 November2026 local date, explicit user request

Question: which part of attention limits useful routing: score contrast,
returned-value influence, key eligibility, or information available to queries?
The user said "act as an explorer list the different possibilities and then
test them". This is a new execution assignment under the updated standing
AGENTS authority, not an automatic continuation of the expired20-minute window.
Allocate at most90minutes from actual GPU admission, including readiness,
qualification, screens, confirmations and retries, on the local RTX3090 only.
Keep the original6h aggregate ceiling: prior debit<3.14h plus2863.44s leaves
>2.06h, so the proposed1.5h round remains within it. New outputs<=1GB and
combined input-attention roots<=2GB. Do not overlap an occupied GPU; the observed
sparse-attention job is unrelated and must be left alone. No remote host or
unrelated workload changes. One evolving note and issue36 continue to own this
work; no new standalone research issue is needed for this user-assigned follow-up.

Frozen alternatives (all38,976parameters,49tokens,D64,perfect-diode memory):

| Case | Change from singular-clipping baseline | What it tests |
| --- | --- | --- |
| sharp8 | score gain4→8 | Greater input-score contrast |
| sharp16 | score gain4→16 | Stronger contrast, including saturation risk |
| drive2 | return gain1→2, reciprocal feedback0.1→0.05 | Stronger values with the same curvature bound |
| drive4 | return gain1→4, reciprocal feedback0.1→0.025 | Larger return influence with unchanged gain×feedback |
| visible_keys | disconnect keys of completely unobserved patches | Whether position-only keys dilute useful routing |
| context_query | add the mean feature of observed tokens to each query feature, then preserve its original norm | Whether hidden queries lack image information |
| sharp8_drive2 | score gain8, return gain2, feedback0.05 | A predeclared contrast/influence interaction |

All seven are tested; they are not a Cartesian hyperparameter sweep. The scalar
attention energy and its local derivatives remain explicit. Fixed query features
and key masks depend only on observed inputs/indicators and position, never hidden
targets. Query context preserves each original feature norm, hence the same
operand bounds. The masked softmax is over eligible branches: negative infinity
is software notation for disconnecting a bank branch, not a transistor voltage.
Its uniform ablation averages over the SAME eligible set, and entropy uses that
set's size. Gain×feedback remains0.1, keeping the same attention Hessian bound;
all separate0.2/0.2/0.3 budgets and initial weight tensors remain unchanged.
The input amplifier, branch switch and query-summary/normalization are ideal
analog macros. Device-complete circuits and power/time benefits remain untested.

Common science: FP32GPU only, seed20261114, batch64, Adam0.003, beta0.005,
epsilon0.5, identical MNIST data/cohorts/masks and no digital learned preprocessing.
Every case first has two2-update GPU smokes, then a250-update128-step warm-up.
The existing8×8 trained-checkpoint diagnostic selects32/64/128 using global and
Q/K/V/output cosine>=0.9 versus BPTT and qualified512/1024 references. Reject a
variant failing numerical/device/gradient gates; retain it as evidence without
retuning. Passing variants receive a fresh750-update EP screen at the selected
cap. This reset keeps equal-update screens matched to the baseline, without
changing a live checkpoint's integration rule.

Rank all valid screens by the fixed256-image validation probe, never test error.
The reference is the explicit750-update probe/checkpoint of `singular-002`
EP (native run20261114T223928.237031Z-1367c83b-54121c73). Reuse its exact
initial/cohort/RNG and completed evidence; do not rerun a control solely for new
metadata. Promote at most the best two screens improving reference probe MSE by
>=0.5% to fresh matched2,000-update EP/BPTT comparisons and final8×8 gradient
gates. Tie-break by case name. If none qualifies, finish with that negative
result. If time is insufficient, label affected coverage partial rather than
quietly excluding slow cases or extending the allowance.

Full comparisons use10,000test,5,000validation and all three256-image ablations.
Keep the existing promotion threshold:>=5% lower test error than old recurrent
EP and>=1% uniform-attention penalty, with valid gradients and physical bounds.
Also report improvement over the stronger singular-clipping baseline. These are
one-seed validation-selected exploratory results; no statistical generalization
or hardware advantage is established by a winner. Record attention entropy/max,
uniform/zero ablations, effective ranks, and RMS attention/input/memory/decoder
forces to distinguish sharpness from actual contribution. Runtime throughput
and resource use are simulator measurements.

Exact recipes are under `examples/energy_transformer_mnist/attention_explorer/`.
The composition will use only public `python -m ebl train/validate` commands.
The same absolute deadline propagates into every child; measured evaluation
reservation and explicit retained-attempt accounting remain enabled. Recovery
and terminal collection must have an active persistent owner with authority to
repair understood operational faults within this scope. Scientific failures
remain results, not an excuse to relax gates or change settings in flight.


One additional reference measurement is declared before launch: replay the
explicit completed singular-002 EP checkpoint through the public train command
with `--resume`, identical config and an already-reached2,000-update target.
It performs zero optimizer steps and supplies the new force measurements.
Require exact equality of weights, optimizer, sampling/masking RNGs and step,
and test-MSE agreement within1e-8 with the existing baseline. Charge its wall
time. This is a read-only reference measurement, not a new training replicate.
All model/schema/energy/local-statistic/gradient checks currently pass; freeze
final readiness and actual handles below at launch.


Explorer readiness:98 relevant tests pass. The suite covers energy/force and
all eight weight-statistic derivatives for every variant and intervention,
EP/BPTT agreement, masked-branch behavior, query-feature norm preservation,
identical initial tensors/certificates, legacy-checkpoint defaults and all89
strict public recipes. Both public compositions import and expose their CLI.
A read-only collector test with existing control artifacts and deliberately
missing screens/confirmation correctly reports incomplete coverage; it does
not turn an empty or partial plan into a successful comparison.

Prepared run composition:

```bash
python -u -m experiments.run_energy_transformer_explorer \
  --output-root /home/filip/analog-research/drn/results/energy-transformer-input-attention/explorer-001 \
  --baseline-root /home/filip/analog-research/drn/results/energy-transformer-input-attention/singular-002 \
  --charge-root /home/filip/analog-research/drn/results/energy-transformer-input-attention/benchmark-001 \
  --charge-root /home/filip/analog-research/drn/results/energy-transformer-input-attention/singular-001 \
  --charge-root /home/filip/analog-research/drn/results/energy-transformer-input-attention/singular-002 \
  --seconds 5400 --deadline-unix <actual-admission-plus-5400>
```

The independent collector is
`python -m experiments.review_energy_transformer_explorer <explicit-output-root>`.
It checks actual selected confirmations, not just runner exit status, and
records exclusions alongside completed cases. Force-reference measurements
are at2,000updates; screen force ratios are at750, clearly labelled as different
checkpoints. Full-arm ratios provide the equal-update comparison. The launcher,
source and watcher handles below establish execution; preparation alone is
not evidence that any candidate helped.
