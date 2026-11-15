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

Design/readiness in progress. No evidence run launched yet.

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
