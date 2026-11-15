# Biased pseudoconductance attention

2026-11-15. User assigned design and validation; follow-up to on-chip-learning #49.

## Question and scope

Can a fully biased one-query, three-key, two-feature MOS bank select and mix
fresh values under supplied attention controls, and return its value/control
derivatives with a nudged pass? Score computation is an external input.
The current-driven anchored operator is w_j=g_j/(1+g_j),
y=sum(w_j v_j)/(h0+sum(w_j)); h=1 and h0=0.01 in normalized units.

Use an ideal weak-inversion reference and a generic BSIM3 sensitivity. Explicit
MOS couplings, anchor bias controllers, input mirrors and read mirrors are
required. Sources supplying reference currents/voltages and output load
compliance are declared external ports. This is exploratory circuit evidence,
not a PDK, layout, silicon or full analog-learning-system demonstration.

## Cases and decision rule

Seed 20261115, 36 fresh signed 3x2 value arrays (12 per selected token), a
uniform-control ablation using equal total control current, an 11-point
two-value mixture, a joint value/control permutation, all controls off,
two transient starts and finite differences of six values and three controls.
Compare the same actual SPICE circuit in every derivative calculation.

Selection: RMSE <0.05 and MSE <uniform MSE/5. Mixture: RMSE <0.05 against
the specified saturating-weight/null-entry operator. Gradient: relative L2
error <0.05, or absolute L2 <1e-4 for near-zero reference groups. Settling:
two-start output disagreement <1e-3. Forward and backward verdicts are separate.
Do not fit held-out values or silently substitute the clamped-key operator.

## Allowance and implementation

Local CPU, up to three recorded circuit design revisions, <=1800 s cumulative
SPICE runtime and <=500 MB raw results. No training, GPU or remote execution.
Expected duration seconds to minutes. The executing agent owns monitoring;
native metrics/logs advance after each case. A 10 s per-process timeout detects
an abnormal simulator. Keep failed attempts. Operational retries preserve the
science and use the same allowance.

Branch/worktree: drn/20261115-pseudoconductance-attention,
/tmp/drn-20261115-pseudoconductance-attention. Native results:
/home/filip/analog-research/drn/results/pseudoconductance-attention/.
Simulator environment: /tmp/pseudoconductance-spice-env.
Code, exact CLI, launch commit and outcomes will be recorded below.

## Handoff

Execution and review complete; all three declared 108-case bundles are retained
and artifact-verified. Forward aggregation is supported in the tested nominal
model; a complete hardware control-gradient readout is still unresolved.
No jobs remain active and no follow-up is launched. Final checkpoint below owns
the runnable example, figure generator and review.

Readiness revision 1 used 1 nA reference currents and threshold +/-0.5 V.
One selective input produced ideal read-mirror ratio error 10.49%, because the
10 pA PMOS reference diode did not satisfy the saturated-current approximation;
generic BSIM output also exceeded the ideal prediction. Artifacts retained at
/tmp/pseudoconductance-readiness-r1/. Revision 2 uses 100 pA reference current,
generic thresholds +/-0.7 V and PMOS mirror W/L=1u/10u (NMOS 10u/10u) to retain
read-mirror compliance while lowering core inversion level. Same topology and
acceptance criteria. Two of three circuit design revisions now specified.
These generic parameters are an illustrative design choice, not a process fit.

Revision-2 DC smoke: ideal selected output [0.689639,0.098533], analytic
reference [0.689592,0.098463], BSIM3 [0.712835,0.102156]. Ideal read-copy
error fell to 4.84e-5; BSIM3 error 0.00131. This is readiness, not the declared
held-out test. The full circuit now has 51 MOS devices including input mirrors.

The first native UIC transient aborted at 100 ns despite simulator exit 0.
Retained at /tmp/pseudoconductance-readiness-r2/transient/. Repair: use the
simulator's consistent DC bias under .ic state constraints before release,
instead of leaving other compact-model charges at zero; require both absence
of an aborted analysis and the declared 20 ms terminal time. A regression test
covers the zero-exit/aborted case. The repaired transient reached 20 ms.

Executable: /tmp/pseudoconductance-spice-env/bin/ngspice (41).
Python/test interpreter: /home/filip/miniconda3/envs/py312/bin/python.
Commands after the launch checkpoint, from a pinned source checkout:

```bash
/home/filip/miniconda3/envs/py312/bin/python -m ebl characterize --config examples/pseudoconductance_ideal.json --output-dir /tmp/pseudoconductance-native-results/ideal
/home/filip/miniconda3/envs/py312/bin/python -m ebl characterize --config examples/pseudoconductance_bsim3.json --output-dir /tmp/pseudoconductance-native-results/bsim3
```

Each model uses 108 SPICE cases (DC plus two transient starts), no training.
Parent owns both cases; metrics advance every simulation and stdout every
12 cases. Operational repairs share the declared 1800 s total allowance.

## Initial evidence and declared follow-up

Published launch e2a2101; pinned source /tmp/pseudoconductance-source-e2a2101.
Both 108-case runs completed their full coverage:
- ideal: /tmp/pseudoconductance-native-results/ideal/20261115T204938.973576Z-a6462845-d6577e8b
- BSIM3: /tmp/pseudoconductance-native-results/bsim3/20261115T204940.237559Z-15a58626-9131a503

Selection RMSE 0.00587 ideal / 0.00572 BSIM3 versus uniform 0.34702 / 0.34746.
All ideal checks passed. BSIM3 passed selection, mixing, permutation, value
gradient and compliance. Its nominal control statistic failed (10.47% error),
and 20 ms transients missed the DC output by up to 0.04889. Original results
remain unchanged. Total SPICE CPU-wall usage reported by these runs: 6.60 s.

Inspection found that the 1 pA output-anchor bias reference was still displaced
after 20 ms. Also, the nominal control statistic assumes effective coupling
equals the reference current. Using the recorded actual branch current in
the already-derived homogeneous-control formula reduced control-gradient error
to 0.56% on the SAME saved free/nudged phases. This is a diagnostic based on
ideal SPICE current probes, not an implemented branch-current sensor.

Revision 3 (readout/time only; hardware unchanged): retain the same inputs,
models, criteria and both gradient statistics, select the measured-branch
statistic explicitly, and extend the two transient observations to 500 ms.
Repeat matched coverage as a separate native bundle with
examples/pseudoconductance_bsim3_branch_readout.json. The original 20 ms and
nominal-readout failures remain reported. Expected additional SPICE runtime
<30 s; <=1793 s remains before smoke/repair overhead. No additional bias tuning.

Home filesystem has insufficient free space for raw bundles. Native temporary
results will use /tmp/pseudoconductance-native-results; compact final evidence
will be copied into the owning backend if space permits. No existing data deleted.

## Final evidence and review

Follow-up published launch: baf39f49621d277064c34b588d10986af30c099f,
clean pinned source /tmp/pseudoconductance-source-baf39f4. Command:

```bash
/home/filip/miniconda3/envs/py312/bin/python -m ebl characterize --config examples/pseudoconductance_bsim3_branch_readout.json --output-dir /tmp/pseudoconductance-native-results/bsim3-branch-readout
```

Final bundle:
/tmp/pseudoconductance-native-results/bsim3-branch-readout/20261115T205514.091549Z-11e73014-a31808e8.
It completed all 108 cases. `ebl runs inspect --verify-artifacts --require-complete`
passed for all three explicit bundles above. Scientific failures in the first
native bundle are retained; process completion is distinct from acceptance.

| Metric | Ideal, initial | BSIM3, initial | BSIM3, follow-up |
| --- | ---: | ---: | ---: |
| Selected-value RMSE | 0.005873 | 0.005723 | 0.005723 |
| Matched uniform RMSE | 0.347019 | 0.347464 | 0.347464 |
| Uniform / selective MSE | 3491.30 | 3685.86 | 3685.86 |
| Mixture RMSE vs specified operator | 0.0000758 | 0.005725 | 0.005725 |
| Value-gradient relative L2 error | 0.00283% | 2.9154% | 2.9154% |
| Nominal-drop control-gradient error | 0.00251% | 10.4655%, fail | 10.4655%, fail |
| Measured-branch control-gradient error | Not separately declared | Diagnostic 0.5588% | 0.5588% |
| Terminal output error vs DC | 2.2e-13 at 20 ms | 0.04889 at 20 ms, fail | 5.4e-13 at 500 ms |

The follow-up passes the original numerical thresholds with its explicitly
selected measured-branch statistic and extended observation. Hardware, input
cases, FD epsilon (1e-4), nudge beta (1e-3) and criteria are unchanged.
Two initial state voltages 0.2/0.4 V reach and remain within 1e-3 of the DC
output at 53.99/58.72 ms. Output-anchor bias reference settling is slow at
1 pA; these data establish neither efficient latency nor energy.

Actual example: v=[[-.6,.4],[.2,-.5],[.7,.1]], a=[.2,.6,1.5], output
[.34285837,-.05294461] versus ideal [.34298119,-.05282200]. Joint permutation
error is zero. The all-off diagnostic has residual output [.04850923,.04850923]
and is NOT an exact suppression result. Off behavior was not a pass/fail gate.
Settled state voltages span .28528–.33680 V; minimum anchor VDS .72985 V.
Follow-up maximum read-copy ratio error at final endpoints is .00131975.

The full netlist is [examples/pseudoconductance_nominal.cir](../../examples/pseudoconductance_nominal.cir),
a byte-identical copy of the tested gradient-free case. Device count is
6 couplings + 8 anchors + 5 controller references + 16 input mirrors +
16 state-readout mirrors = 51 MOS, with 8 explicit 1 pF state capacitors.
VDD=1.8 V, VREF=.3 V, external read-load compliance=1.2 V, Iref=100 pA;
key anchors use 100 pA and output anchors 1 pA. Values [-.75,.75] enter as
25–175 pA reference currents. NMOS W/L=10u/10u, PMOS 1u/10u. Five shared
diode-connected NMOS references turn supplied currents into gate voltages.
Generic BSIM3 cards are not foundry-fitted; no layout, noise or mismatch test.

The backward test uses FREE/+beta/-beta phases and software current differences,
not an implemented two-phase learning controller. All input derivatives and
three control derivatives are finite-differenced at one mixed-control point
with e=[.7,-.4]. Value gradients use the explicit state-current mirrors;
the improved control statistic uses ideal zero-volt branch-current probes:
sum_d (Iedge_jd/Iref)/a_j * (lambda_jd-lambda_out,d).
No physical branch-current sensor, product/reduction, phase storage or score
generator is implemented. The successful relative L2 result does not imply an
exact BSIM3 adjoint identity or a per-component uniform error guarantee.

Interpretation: a biased MOS star can perform attention-like value selection
and mixing with one coupling transistor per edge-feature. The natural ideal
operator has weights a/(1+a), a zero-valued anchor entry and no softmax. Its
nominal forward behavior survives explicit input/read loading. The attractive
control-gradient reading still depends on an unimplemented sensing path;
nominal-drop readout alone fails the declared derivative criterion. This
supports the aggregation primitive, not complete analog attention, training,
recurrence benefit, PDK viability or an area/energy advantage.

The next hardware question is a shared low-overhead branch sensor/product
path, followed by bias settling and all-off offset checks. These are proposed
next decisions only; no additional experiments are launched.

Three affected implementation tests passed, including abort detection.
The figure generator was executed and both circuit/validation figures visually
inspected. Reproduce figures with explicit bundle selection:

```bash
/home/filip/miniconda3/envs/py312/bin/python -m experiments.energy_transformer.pseudoconductance_figures --bundle /tmp/pseudoconductance-native-results/bsim3-branch-readout/20261115T205514.091549Z-11e73014-a31808e8 --output /tmp/pseudoconductance-figures
```

Shared explanation, connection diagram, result plots and small provenance JSON:
/home/filip/analog-research/on-chip-learning/explorations/pseudoconductance-attention-20261115/.
Owning issue: https://github.com/filip137/on-chip-learning/issues/50.
Three evidence bundles consumed 324 SPICE calls and 8.7357 s simulator wall
time, plus small readiness/test calls; about 27 MB raw bundles. Original
1800 s/500 MB limits were respected. Preparation, analysis and token usage
were not instrumented. No GPU, remote compute or training was used.

Durable raw evidence including failed readiness:
results/pseudoconductance-attention/20261115/native-and-readiness.tar.gz
(8,200,709 bytes, SHA256
dd0c0d507759f22e8c551020dc146aa85c7eb9c78a4f10d8f0ff9dd9186ce0c3).
Extract to /tmp to recreate exact original bundle directories and rerun the
inspector. Archive includes native netlists/logs/traces/configs/manifests/results
and readiness revisions 1/2. Source is retained in published launch commits.
Small backend evidence.json also retains all three summaries and provenance.
Only this pilot's worktrees/merged branch are removed at integration closeout;
raw results and original failed evidence remain.
