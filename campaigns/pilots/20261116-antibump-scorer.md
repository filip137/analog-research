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

## Pre-launch readiness corrections (dirty-tree smokes, scratchpad)

1. **Trained designs replaced.** The first readiness smoke failed retrieval in both
   models (MSE 3–9e-3 against about 1e-4 predicted). The stage-2 models used an
   unbounded linear projection, which put query/key voltages at −0.43 to 2.5 V,
   outside the rails.
   - **Fix:** retrained with the physical passive crossbar projection of #56
     (conductances 9–88.2 µS; on-chip-learning `scorer_scaling.py` arms `C1D*/xbar`).
     Its effective projection keeps q and k within 0.71–1.39 V.
   - **Equations:** softmax N = 3 MSE 2.2e-5, weight 0.961 (N = 8: 2.0e-5, 0.948).
     Star N = 3 MSE 1.9e-5, weight 0.878: it misses the stage-2 weight criterion, but
     keeps its stage-2 selection.
   - **New sources:** `checks/results/antibump_xbar_star_n3_seed0.json`
     (sha256 93fb1d4f1a28c26f98e1df5237e5b765152862af5fccf95654cc9f50a514f4ff, g = 4.4270) and `antibump_xbar_softmax_n3_seed0.json`
     (sha256 24d4ebcaf7a6643d8156863d1ade5c6a0e4a59f950c46e754fb1e2ab12b0f410, g = 4.7783). The configs use their effective
     projections.
   - **Specification correction:** the projection is the crossbar's effective
     (G⁺ − G⁻)/ΣG matrix, not an unconstrained K.
2. **Convergence hints.** One ideal single-cell netlist in the D = 8 summation test
   converged to a wrong operating point (case 4: additivity error 0.26, the only
   outlier). I added `.nodeset` hints for the summation-test nodes.
3. **Transient tolerance.** The BSIM3 star transient from 0.2 V aborted at
   abstol = 1e-18 A, on the probe of a nearly-off coupler at femtoamps. With 1e-16
   and 1e-14 A it completes, and the endpoint equals the DC solution to 1e-13. This
   pilot therefore uses a transient-only abstol of 1e-16 A; DC stays at 1e-20 A.

**Second readiness** (before the tolerance change): ideal star and softmax and BSIM3
softmax passed all gates. BSIM3 softmax gave MSE 5.4e-5, context/retrieval 527×, a
cell fit of S_eff ≈ 75 against 16 drawn, and κ ≈ 0.79.

## Handoff

The launch checkpoint is this commit, merged and pushed. From a pinned checkout:

```bash
PY=/home/filip/miniconda3/envs/py312/bin/python
R=/home/filip/analog-research/drn/results/antibump-scorer
cd $R/source-<launch sha>
for v in star softmax; do for m in ideal bsim3; do
  $PY -m ebl characterize --config examples/antibump_${v}_$m.json --output-dir $R/${v}_$m
done; done
$PY -m ebl runs inspect $R/*/2026* --verify-artifacts --require-complete
```
