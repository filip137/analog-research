# Bump-chain attention scorer feeding pseudoconductance aggregation

2026-11-16. Assigned by the user as Phase 2 of on-chip-learning
[#55](https://github.com/filip137/on-chip-learning/issues/55). It follows
[20261115-pseudoconductance-attention](20261115-pseudoconductance-attention.md) (#50).

## Question and scope

Can a transistor-level two-dimensional bump-chain scorer turn a query voltage and
three key voltages into attention currents that drive the existing 51-MOS
aggregation bank? Concretely: does it select the value whose key matches the query,
and are its derivatives with respect to the query and keys given by its ideal law?

Phase 1 (ideal equations, on-chip-learning
`orthogonal-ideas/energy-transformer/scorer-stage.md`) selected the bump chain
among four analog scorer families on fresh-binding keyed retrieval.

**Circuit, per key j and score dimension d:**
- an NMOS differential pair with gates q_d and k_jd, whose tail is the previous
  stage's output;
- two diode-connected PMOS loads carrying I1 and I2;
- a series PMOS current correlator giving I1·I2/(I1+I2) = I_tail/(4cosh²(κΔ/2U_T)).

An NMOS mirror passes the dimension-0 correlator current to the dimension-1 tail.
The dimension-1 correlator drains directly into the key's attention controller,
g_a,j, which replaces the ideal attention reference current of #50. The first
tails are copies of one 800 pA reference.

**Ideal law:**

```
a_j = (I_b / 100 pA) · Π_d sech²(κ (q_d − k_jd) / (2 U_T)),   peak a = 0.5
```

The 51-MOS bank, its bias and input/read mirrors, VDD, V_REF and the read
receivers are unchanged from #50. Query and key voltages are ideal sources.
Generating them from a projection is out of scope.

## Models, cases and decision rule

Two models, as in #50:
- an ideal weak-inversion subcircuit with κ = 2/3, the wiring reference;
- generic nominal BSIM3, the sensitivity case.

Seed 20261116. Values are 36 fresh signed 3×2 arrays, uniform on ±0.75. Keys are
fresh per case in [0.8, 1.3] V per dimension, with pairwise L∞ separation of at
least 0.25 V.

**Cases:**
1. **Selective:** the query equals key (i mod 3), 12 cases per key.
2. **Matched uniform control:** the same values, with the #50 ideal attention
   sources set to the mean of that case's three measured scorer currents.
3. **Law sweep:** 11 queries along the segment from key 0 to key 1 of a fixed key
   set. Measure every scorer current (zero-volt probe between correlator and
   controller) and fit one κ by least squares of ln a against the law. These cases
   are used only to fit κ.
4. **Far query:** a query 0.3 V outside the key box. Reported only.
5. **Derivatives:** at q = k_0 + (40, −25) mV, central differences of every a_j with
   respect to q_d and k_jd (ε = 1 mV), compared with the law's derivative using the
   fitted κ (for the ideal model, κ = 2/3). End to end, dL/dq_d is the block's
   measured-branch readout dL/da_j (two ±β nudges, β = 0.001, e = (0.7, −0.4))
   times the law derivative, compared with central differences of e·y with respect
   to q_d.
6. **Settling:** transients of 0.5 s from 0.2 V and 0.4 V state starts.

**Gates:**
- Selection RMSE < 0.05 and MSE at least 5× below uniform.
- Scorer-derivative relative L2 error < 5% (absolute 1e-4 near zero).
- Two-start settled output difference < 1e-3.
- Node voltages inside the rails, and anchor V_DS > 4U_T.

The end-to-end query gradient and the law-fit residual are reported, not gated.
Forward and backward verdicts are separate. Values, keys and gates are not
tuned after launch.

## Allowance

Local CPU only, at most 3 recorded circuit revisions, ≤ 1800 s cumulative SPICE
time across all attempts, and ≤ 500 MB raw results. No training, GPU or remote
compute. Expected duration: seconds to minutes per model. The executing agent owns
monitoring through review. The simulator is the #50 environment,
`/tmp/pseudoconductance-spice-env/bin/ngspice` (ngspice 41).

Branch and worktree: `drn/20261116-bump-scorer-attention`,
`/home/filip/analog-research/worktrees/drn-20261116-bump-scorer-attention`.
Native results: `/home/filip/analog-research/drn/results/bump-scorer-attention/`.

## Implementation

**Code:**
- `networks/energy_transformer/bump_scorer.py`: the scorer netlist, its law and
  the law's derivatives.
- An optional `scorer=` hook in `pseudoconductance.export`. The default 51-MOS
  netlist is unchanged; a test asserts this.
- `ebl/schema/bump_scorer_attention.py` and `ebl/runtime/bump_scorer_attention.py`.
- Configs `examples/bump_scorer_ideal.json` and `examples/bump_scorer_bsim3.json`.
- `tests/test_bump_scorer_attention.py`.

The scorer has 46 MOS: 15 per key plus one shared tail reference. With the bank,
the total is 97 MOS.

**Fixed details** (set before launch, from the specification):
- The law sweep and derivative point use keys (0.85, 0.90), (1.20, 0.95) and
  (1.00, 1.25) V, with the #50 base values.
- The κ fit grid-searches κ in [0.4, 1.2] with a free log-amplitude, over sweep
  currents of at least 1e-3.
- The far query is (1.6, 1.6) V.
- The law derivative is the measured a_j times the law's d ln a/dq, at the fitted κ
  (2/3 for the ideal model).

**Readiness:** a dirty-tree smoke of the ideal model, at the scratchpad path
`bump-readiness/ideal`. It passed all gates in 105 SPICE calls and 7.2 s. The fit
recovered κ = 0.6664, against the true 2/3. The ideal-SPICE unit test tolerates a
numerical floor of about 5 fA at starved branches.

## Attempts

**Attempt 1** (launch `ea426d5`, pinned at `results/bump-scorer-attention/source-ea426d5`):
- **Ideal:** completed and passed every gate. Bundle `/home/filip/analog-research/drn/results/bump-scorer-attention/ideal/20261116T170120.648113Z-2c15b415-bf5181a8`.
- **BSIM3:** all 103 DC cases completed. Then ngspice aborted `transient-0`, the
  0.2 V start, at t = 2.5 µs with "timestep too small" on the probe of coupler 2,
  feature 1. The runtime caught the zero-exit abort, as designed. Failed bundle
  retained: `/home/filip/analog-research/drn/results/bump-scorer-attention/bsim3/20261116T170130.466080Z-4cf6ea47-c86d2aa0`.

**Diagnosis** (scratchpad probes, recorded here):
- At the derivative point a₂ ≈ 3.6e-5, so coupler 2 carries femtoamp currents.
- The scorer-free #50 circuit starts from 0.2 V without trouble, even with
  controls of 1e-4.
- Neither more transient iterations (`itl4` = 100 or 1000) nor 1–10 fF of
  scorer-node capacitance helps.
- Replacing the absolute current tolerance abstol = 1e-20 A by any value from
  1e-18 to 1e-15 A lets the transient complete. The settled output is identical
  across that range, and within 9.4e-5 of the DC solution.

**Revision 1, a solver setting:** transient analyses use abstol = 1e-18 A, through
a new `transient_abstol` argument. DC analyses keep 1e-20. The default #50
netlists, DC and transient, are byte-identical to `ea426d5`. No circuit, case,
value or gate changed. Both models are relaunched from the revision commit. The
attempt-1 bundles stay as evidence. This counts as revision 1 of the 3 allowed.

**Attempt 2** (revision `e569ecb`, pinned at `results/bump-scorer-attention/source-e569ecb`):

| Model | Bundle | Integrity |
| --- | --- | --- |
| ideal | `results/bump-scorer-attention/ideal/20261116T170443.915555Z-2c15b415-8c39dab9` | complete, valid |
| bsim3 | `results/bump-scorer-attention/bsim3/20261116T170454.219899Z-4cf6ea47-8ab955ac` | complete, valid |

Both pass `ebl runs inspect --verify-artifacts --require-complete`, with 105 SPICE
calls each.

## Results

| Check | Ideal (κ = 2/3) | Generic BSIM3 | Gate |
| --- | ---: | ---: | --- |
| Selection RMSE, 36 fresh key/value sets | 0.0142 | **0.0083** | < 0.05 |
| Uniform / selective MSE | 653× | **1909×** | ≥ 5× |
| Output vs ideal operator at measured currents | 1.1e-4 | 0.0113 | reported |
| Selected scorer current (law peak 0.5) | 0.500 | 0.594 (0.590–0.599) | reported |
| Largest unselected current | 0.0026 | 0.0010 | reported |
| κ fit from sweep (amplitude ratio, rms ln residual) | 0.6664 (0.999, 0.001) | 0.7911 (1.150, 0.055) | reported |
| Scorer derivative, law vs FD, relative L2 | 0.02% | **1.65%** | < 5% |
| End-to-end dL/dq, branch readout × law | 1.0% | 13.5% | reported |
| End-to-end dL/dk | 0.8% | 11.0% | reported |
| Far-query output | (0.0023, 0.0018) | (0.0485, 0.0485) | reported |
| Transient vs DC, two starts | 5e-14 | 9.4e-5 | < 1e-3 |
| Node voltages / min anchor V_DS / min scorer tail | 0.286–0.336 / 1.06 / 0.27 V | 0.285–0.337 / 0.73 / 0.19 V | rails, > 4U_T |
| Supply power (largest case) | 11.6 nW | 11.7 nW | reported |

## Review

**Forward: passed.** The explicit scorer and the unchanged bank select the value
whose key matches the query. They do so for fresh keys and values in every case,
with BSIM3 RMSE 0.0083. That is comparable to #50 with ideal supplied controls
(0.0057). The BSIM3 scorer delivers 19% more current at the peak than the law, and
a 1.15× amplitude remains after the κ fit. Selection is insensitive to this gain.
The fitted κ = 0.79 agrees with the controller-slope estimate of about 0.78 from #50.

**Far query.** The BSIM3 output is 0.0485, the same value as the #50 all-off
offset. That offset belongs to the bank at near-zero coupling, not to the scorer.

**Scorer derivative: passed.** With κ fitted on the separate sweep, the law
predicts ∂a/∂q and ∂a/∂k within 1.65%.

**End-to-end gradient: not gated, and the error is larger than the derivative
error.** The branch readout times the law derivative gives 11–13% error against
full-circuit finite differences. The cause:
- The two off-target keys carry a current floor of about 1e-4 (about 10 fA). The
  law derivative scaled by that measured current overstates their sensitivity:
  0.009 against a finite difference of 0.0018 or less.
- Their dL/da is large (about 2.2), because adding weight to another token moves
  the output a lot.
- Retrospectively, and not part of the declared test: keeping only the matching
  key's term gives 4.2% (q) and 3.0% (k).

An explicit scorer backward pass therefore has to model the floor, or treat
sub-floor keys as having zero gradient.

**Not shown:**
- mismatch or noise;
- a projection circuit producing q and k;
- an analog backward pass through the scorer;
- training;
- PDK or silicon behaviour;
- energy or latency advantage.

**Cost:**
- Instrumented SPICE time 27 s: readiness 7.2 s, attempt 1 9.2 s, attempt 2 10.4 s.
- 18 diagnostic probe runs, not instrumented, of a few seconds each.
- Raw results 34 MB.
- 1 of 3 revisions used.
- All within the 1800 s and 500 MB allowance.

## Handoff

Closed. Execution, collection and review are complete, and no jobs remain. The
attempt-1 bundles (ideal passed; BSIM3 failed at its transient) are retained beside
the attempt-2 evidence. Next decisions belong to on-chip-learning #55 and its
follow-ups: mismatch, the floor-aware scorer backward pass, and a projection that
produces q and k.
