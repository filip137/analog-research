# Crossbar query/key projection feeding the bump scorer

2026-11-16. Assigned by the user as Phase B of on-chip-learning
[#56](https://github.com/filip137/on-chip-learning/issues/56). It follows
[20261116-bump-scorer-attention](20261116-bump-scorer-attention.md) (#55) and
[20261115-pseudoconductance-attention](20261115-pseudoconductance-attention.md) (#50).

## Question and scope

Does a passive resistive crossbar holding a trained shared query/key projection,
driving the transistor-level bump scorer and the 51-MOS pseudoconductance bank,
perform fresh-binding keyed retrieval in SPICE as its ideal equations predict?

**Crossbar.** One copy per token: three keys and one query.
- Each token x = [value, kx, ky] (the query token has value 0) is applied
  differentially, V_cm ± s·x_i, with V_cm = 1.05 V and s = 0.75 V per unit.
- Each of the 2 output rows connects through conductances G⁺_di to V_cm + s·x_i
  and G⁻_di to V_cm − s·x_i. That makes 12 resistors per token and 48 in total.
- The rows are unloaded: they drive only the bump-scorer differential-pair gates.
  So q_d = V_cm + s·Σ_i (G⁺ − G⁻)·x_i / Σ_i (G⁺ + G⁻).
- The row nodes replace the ideal query/key voltage sources of #55.

**Trained design.** Phase A of #56 trained the conductances in ideal equations.
The selected model is the passive crossbar with shared W and the tail current
fixed at 800 pA (N = 3, seed 0), held-out MSE 7.7e-6. Exported as on-chip-learning
`orthogonal-ideas/energy-transformer/checks/results/crossbar_qk_n3_seed0.json`,
sha256 `a9a4149ae714fc3bcfbfe776cc47d9f390b1f9c11f47498cf421194a693de878`.
The conductances and value weights are copied into the strict configs, with that
hash as provenance. All conductances lie in 12.9–78.5 µS, inside the CMO range of
9–88.2 µS.

**Values.** They come from the learned value projection, v = V·x, as ideal input
currents 100 pA·(1 + v). Feature 0 carries v and feature 1 carries −v. The
scorer, its 800 pA tail and the bank are unchanged from #55 and #50.

**Not modelled.** Device programming error, drift, read noise, mismatch, and wire
resistance are absent. Token input voltages are ideal sources.

## Models, cases and decision rule

Two models, as before:
- **Ideal weak-inversion devices (κ = 2/3):** a wiring check whose equation
  reference uses κ = 2/3.
- **Generic BSIM3:** the evidence. Its equation reference uses the #55 fit,
  κ = 0.7911 and amplitude ratio 1.1495, which Phase A also trained with.

Seed 20261116. **Cases:**
1. **Retrieval:** 36 fresh examples from the toy distribution. Three keys at random
   distinct slots of a randomly rotated 8-slot unit circle, values uniform on
   ±0.4, and the query equal to key (i mod 3).
2. **Uniform control:** ideal attention sources at the mean of each case's three
   measured scorer currents.
3. **Settling:** two 0.5 s transients from 0.2 V and 0.4 V state starts, on
   example 0.

**Gates:**
- Retrieval MSE over both features ≤ 1e-3, and ≥ 5× below the context-average MSE
  on the same examples.
- Crossbar row voltages within 1 mV of the linear formula, in every case.
- Node voltages inside the rails, and anchor V_DS > 4U_T.
- Two-start settled output difference < 1e-3.

**Reported:**
- SPICE vs equation RMSE;
- uniform MSE;
- row-voltage range against the bump's tested 0.8–1.3 V;
- crossbar dissipation, and bank plus scorer supply power.

No retuning after launch.

## Allowance

Local CPU only, ≤ 1800 s cumulative SPICE, ≤ 3 recorded revisions, ≤ 500 MB raw
results. The simulator is `/tmp/pseudoconductance-spice-env/bin/ngspice`. The
executing agent owns monitoring. Branch `drn/20261116-crossbar-qk-attention`;
worktree `/home/filip/analog-research/worktrees/drn-20261116-crossbar-qk-attention`.
Native results: `/home/filip/analog-research/drn/results/crossbar-qk-attention/`.

## Implementation

**Code:**
- `networks/energy_transformer/crossbar_qk.py`: netlist, formula and dissipation.
- A crossbar drive option in `bump_scorer.add_scorer`. The ideal query/key sources
  of #55 stay the default.
- `ebl/schema/crossbar_qk_attention.py` and `ebl/runtime/crossbar_qk_attention.py`.
- Configs `examples/crossbar_qk_{ideal,bsim3}.json`, which carry the trained
  conductances and value weights plus the source sha256.
- `tests/test_crossbar_qk_attention.py`.

Transients use the #55 solver setting, abstol = 1e-18 A, as transient-only;
DC analyses keep 1e-20 A.

**Readiness:** dirty-tree smokes of both models, at the scratchpad path
`xqk-readiness`, passed all gates in 74 SPICE calls each.

## Attempts

**Attempt 1** (launch `00c16d3`, pinned at `results/crossbar-qk-attention/source-00c16d3`):

| Model | Bundle | Integrity |
| --- | --- | --- |
| ideal | `results/crossbar-qk-attention/ideal/20261116T172322.257534Z-3575ed40-8dcf979f` | complete, valid |
| bsim3 | `results/crossbar-qk-attention/bsim3/20261116T172332.117870Z-7a661d54-c17d2e7e` | complete, valid |

Both pass `ebl runs inspect --verify-artifacts --require-complete`, with 74 SPICE
calls each. No revision was needed.

## Results

| Check | Ideal (κ = 2/3) | Generic BSIM3 | Gate |
| --- | ---: | ---: | --- |
| Retrieval MSE, 36 fresh examples, both features | 3.1e-5 | **4.2e-5** | ≤ 1e-3 |
| Context-average MSE / retrieval MSE | 903× | **677×** | ≥ 5× |
| Equation prediction MSE (same weights and examples) | 3.1e-5 | 3.8e-6 | reported |
| SPICE vs equation RMSE | 1.1e-4 | 0.0059 | reported |
| Uniform-control MSE | 0.0281 | 0.0280 | reported |
| Weight on target from measured scorer currents | 0.957 | 0.969 | reported |
| Selected scorer current (equation) | 0.499 (0.499) | 0.593 (0.574) | reported |
| Crossbar row voltage vs formula, max error | 9e-16 V | 9e-16 V | < 1 mV |
| Row voltage range (fraction outside 0.8–1.3 V) | 0.795–1.305 V (4.2%) | same | reported |
| Settling, two starts | 1.1e-6 | 2.0e-5 | < 1e-3 |
| Node voltages / min anchor V_DS / min scorer tail | 0.291–0.314 / 1.06 / 0.28 V | 0.291–0.315 / 0.73 / 0.20 V | rails, > 4U_T |
| Crossbar dissipation (mean) | 331 µW | 331 µW | reported |
| Bank and scorer supply power (largest case) | 10.8 nW | 10.9 nW | reported |

## Review

**All declared gates pass.** The trained shared projection, programmed as
resistors, drives the transistor-level scorer and bank. They retrieve the
query-matched value for fresh bindings, with BSIM3 MSE 4.2e-5, 677× below the
context average.

**SPICE vs equations.** SPICE output departs from the equations by RMSE 0.0059.
That is the same size as the bank's own BSIM3 deviation in #50 (0.0057). The
crossbar is exact, the scorer gain is about 3% above the fitted law, and uniform
attention gives 0.028. The worst single error is 0.0125.

**The crossbar dominates power.** At CMO conductances the crossbar dissipates
about 30,000× the power of the bank and scorer. Because the rows are unloaded,
only conductance ratios matter. Devices about 1000× more resistive, with the same
9.8:1 range, would bring it to about 0.3 µW without changing the computation.
That is a device-selection question this pilot does not test.

**Not shown:**
- device programming error, drift, read noise, mismatch, wire resistance;
- on-chip training of the crossbar;
- an analog value projection;
- larger N;
- PDK or silicon behaviour;
- energy or latency advantage.

**Cost:**
- Instrumented SPICE 18.4 s: readiness 9.2 s, attempt 1 9.2 s.
- Raw results 20 MB.
- No revision used.

## Handoff

Closed. Execution, collection and review are complete, and no jobs remain. Next
decisions belong to on-chip-learning #56 follow-ups: device nonidealities and
crossbar conductance scale, larger N and on-chip training of W.
