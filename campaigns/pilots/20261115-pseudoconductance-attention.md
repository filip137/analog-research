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
