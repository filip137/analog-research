# Summed anti-bump attention scorer (scalable scorer canary)

2026-11-16. Assigned by the user as stage 3 of on-chip-learning
[#57](https://github.com/filip137/on-chip-learning/issues/57): evaluate in SPICE the
scorer options that passed the stage-2 equation study. It follows
[20261116-bump-scorer-attention](20261116-bump-scorer-attention.md) (#55) and
[20261116-crossbar-qk-attention](20261116-crossbar-qk-attention.md) (#56).

## Question and scope

Does a "sum then exponentiate" scorer, whose cost grows linearly with key
dimension, retrieve fresh bindings at transistor level as its equations predict?
The candidate is the one the stage-2 rule selected: C1, summed Delbrück anti-bump
currents.

**Per key j and dimension d, a 5-MOS Delbrück bump–antibump cell:**
- an NMOS outer pair with gates q_d and k_jd;
- a series NMOS middle leg, each device S = 16 times wider;
- an NMOS tail copying a 1 nA reference.

Ideally the outer legs carry I_b·c·cosh²u/(1 + c·cosh²u), with c = 4/S and
u = κ(q_d − k_jd)/2U_T.

**Summation.** The outer drains of all dimensions of a key sum on one wire into a
PMOS diode, and a PMOS copy delivers the sum.

**Two variants:**
- **Star** (selected but at the margin in stage 2; it fails at N = 8). An NMOS sink
  mirror pulls the summed current through an ideal resistor R from a buffered
  reference V_A. V_A is a controller diode at 0.57·100 pA, buffered by an ideal VCVS.
  The resulting voltage G_j drives coupler gate g_a,j directly, replacing that key's
  attention controller. Ideally a_j = 0.57·exp(−κR·I_sum,j/U_T).
- **Softmax** (Elfadel–Wyatt). The summed current flows through R to an ideal
  V_low = 0.6 V. The voltage drives the gate of one PMOS per key; the PMOS sources
  share a tail current of 57 pA (a PMOS mirror) and their drains feed the existing
  attention controllers. Ideally a_j = 0.57·softmax_j(−κR·I_sum,j/U_T).

**Resistor.** R = g·U_T/(κ·I_b), with g the trained temperature and κ = 0.7911 for
BSIM3 (2/3 for the ideal model). This is about 120 MΩ. It is an ideal element; its
transistor realization is a separate question.

**Trained designs.** Stage-2 C1D law models at N = 3, seed 0, amplitude 0.57.
Source: on-chip-learning `orthogonal-ideas/energy-transformer/checks/results/`:
- star: `antibump_star_n3_seed0.json`, sha256 8ced24648ff10672d85d34d0cf5bfb6528562ba16a04d0f7f7bf07938cc1f1d0, g = 3.4358;
- softmax: `antibump_softmax_n3_seed0.json`, sha256 fb1c3d41f0c3e7e02ba082459e5c7150d25058cabafa9e0b6611c0a07ae15c6b, g = 3.6241.

The projection K (2×3) and value weights are copied into the configs.
Query/key voltages are V_cm + s·K·x (1.05 V, 0.75 V per unit) from ideal sources;
#56 already showed the crossbar producing them. Values are ideal currents, and the
51-MOS bank is unchanged.

## Models, cases and decision rule

Ideal weak-inversion (κ = 2/3; wiring check) and generic BSIM3. Seed 20261116.

**Cases, per variant and model:**
1. **Retrieval:** 36 fresh examples from the toy distribution at N = 3. Feature 0
   carries v and feature 1 carries −v.
2. **Uniform control:** ideal attention sources at the mean of the equation-law
   currents evaluated at the measured summed scores.
3. **Cell-law sweep:** key 0 fixed, query dimension 0 swept over Δ in [−0.3, 0.3] V
   (13 points), measuring the summed current. Fit the effective S and κ. Reported.
4. **D = 8 summation:** one key with eight cells into one summing diode, 6 random
   Δ vectors. Compare the summed current with the sum of single-cell measurements.
   Reported.
5. **Settling:** two 0.5 s transients from 0.2 V and 0.4 V state starts, with the
   transient-only abstol of 1e-18 A from #55.

**Gates:**
- retrieval MSE ≤ 1e-3, and ≥ 5× below the context-average MSE on the same examples;
- node voltages inside the rails, and anchor V_DS > 4U_T;
- settling < 1e-3.

**Reported:** SPICE vs equation RMSE, uniform MSE, cell-law fit and residual, D = 8
additivity error, and supply power. No retuning after launch.

## Allowance

Local CPU, ≤ 1800 s cumulative SPICE across both variants, ≤ 3 recorded revisions
per variant, ≤ 500 MB. Simulator `/tmp/pseudoconductance-spice-env/bin/ngspice`.
Branch `drn/20261116-antibump-scorer`; worktree
`/home/filip/analog-research/worktrees/drn-20261116-antibump-scorer`. Native
results: `/home/filip/analog-research/drn/results/antibump-scorer/`.

## Handoff

Specification recorded; implementation in progress. Nothing launched. The
pre-specification wiring smoke (untrained keys, both variants and models) ran in the
scratchpad. In it the BSIM3 summed current at a match was 0.10 nA against the law's
0.40 nA, which suggests an effective S larger than drawn.
