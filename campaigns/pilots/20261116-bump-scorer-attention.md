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

## Handoff

The launch checkpoint is this commit, once merged into `refactor/training-core`
and pushed. Run from a pinned checkout of that commit:

```bash
PY=/home/filip/miniconda3/envs/py312/bin/python
R=/home/filip/analog-research/drn/results/bump-scorer-attention
cd $R/source-<launch sha>
$PY -m ebl characterize --config examples/bump_scorer_ideal.json --output-dir $R/ideal
$PY -m ebl characterize --config examples/bump_scorer_bsim3.json --output-dir $R/bsim3
$PY -m ebl runs inspect $R/ideal/<run> $R/bsim3/<run> --verify-artifacts --require-complete
```

The executing agent owns both short runs, which are expected to take under a
minute each.
