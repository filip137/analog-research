# End-to-end analog attention step (token sequence to layer output)

2026-11-17. Assigned by the user as stage 3 of on-chip-learning
[#58](https://github.com/filip137/on-chip-learning/issues/58): an end-to-end analog
design from the token sequence x to the attention scores and the end of the
attention layer, with the compatibility of each stage and KV caching. It follows
[20261116-antibump-scorer](20261116-antibump-scorer.md) (#57) and
[20261116-crossbar-qk-attention](20261116-crossbar-qk-attention.md) (#56).

## Question and scope

Can one transistor-level circuit take a token sequence, write its keys and values
into an analog KV cache, score a fresh query, normalize and weight the values, and
produce the attention output that the equations predict? Two chains selected by the
stage-2 equation study (on-chip-learning `checks/end_to_end.py`) are compared:

- **Chain A (EP-trainable bank):** softmax currents drive the controllers of the
  51-MOS pseudoconductance bank. Each stored value drives a PMOS diff pair whose
  branch current Iref·(1 ± tanh(κ(V_v − V_cm)/2U_T)) is the bank's value input.
- **Chain B (feed-forward translinear):** each softmax current is mirrored into the
  tails of NMOS diff pairs whose gates hold the value. The differential output current
  is amp·Iref·Σ_j p_j tanh(κ(V_v,j − V_cm)/2U_T). No bank. This is the value-mixing
  block of Abbott & Machta (arXiv:2610.12571, October 2026), here at BSIM3 level.

**Common front end (one transient per attention step):**
1. **Token DAC and crossbar.** Ideal PWL sources apply tokens x = [value, kx, ky] in
   sequence as V_cm ± s·x_i (1.05 V, 0.75 V per unit) to an explicit resistor crossbar
   with three unloaded columns: two key columns and one value column. Conductances are
   9–88.2 µS. The query token comes fourth.
2. **KV cache.** During token j's window (20 µs), transmission gates copy the three
   columns onto 5 pF hold capacitors k_j0, k_j1, v_j. The gates are BSIM3 NMOS+PMOS
   with W/L = 10/1 µm; the ideal model uses ideal switches.
3. **Scorer.** 5-MOS anti-bump cells, S = 16, I_b = 1 nA (#57). The query gates are
   driven live by the key columns, the key gates by the hold capacitors. Each key's
   summed current is mirrored and **integrated on a 1 pF capacitor** for
   T = g·C·U_T/(κ·I_b) after a reset to V_low = 0.6 V. This replaces #57's ideal
   150 MΩ resistor; switches steer the mirror into a dump node outside the window.
4. **Normalization.** A PMOS Elfadel–Wyatt softmax with a 57 pA tail reads the held
   score voltages.

**Trained designs** (passive crossbar projections, fixed total attention 0.57, learnable
affine readout standing for the next-stage crossbar). Source: on-chip-learning
`orthogonal-ideas/energy-transformer/checks/results/`, N = 3, seed 0. The value column
uses the same 0.75 V/unit swing (arms `A-tanh/xs`, `B-tanh/xs`):
- A: `e2e_A_tanhxs_n3_seed0.json`, sha256 6c9b78ff49ea7e0ef29a56d200442a9edfa24306950ac0a5c0b3eff02b41d10b, g = 5.0646, readout 1.0838·y.
  Equations: test MSE 2.7e-5 (N = 3) and 2.4e-5 (N = 8).
- B: `e2e_B_tanhxs_n3_seed0.json`, sha256 cfa282b78c7d94de9aa27f921b8b723a6fce2f08acd2c19f52910bd6f77dd961, g = 4.7409, readout 1.0632·y.
  Equations: test MSE 2.4e-5 and 2.0e-5.

## Models, cases and decision rule

Ideal weak-inversion law (κ = 2/3, wiring check) and generic BSIM3 (κ_design = 0.7911).
Seed 20261117.

**Cases, per chain and model:**
1. **Retrieval:** 36 fresh N = 3 examples, every slot used as the query in turn. Each
   example is one full transient covering write, query, integration and hold:
   - chain A holds 0.3 s, which also lets the bank settle;
   - chain B holds 0.02 s.
2. **Hold sweep (BSIM3):** example 0 repeated with 1 s and 3 s holds. Reported: output
   change and stored-node drift.

**Gates:**
- **BSIM3:** retrieval MSE ≤ 1e-3 and ≥ 5× below the context-average MSE, using the
  trained readout on feature 0.
- **Ideal model:** SPICE vs its own law ≤ 2e-3 RMS. This is a wiring check, because
  the ideal law's κ = 2/3 differs from the trained κ.
- **Chain A also:** late drift (last 10% of the hold) < 1e-3, node voltages inside the
  rails, and anchor V_DS > 4U_T.

**Reported:**
- differential readout (y₀ − y₁)/2;
- SPICE vs law, and softmax vs law;
- summed score vs law;
- KV pedestal (held − column at switch-off), droop rate, and column vs formula;
- supply power and MOS counts.

No retuning after launch.

## Allowance

Local CPU, ≤ 120 CPU-min cumulative SPICE (stage-3 allowance of #58), ≤ 3 recorded
revisions per configuration, ≤ 2 GB. Simulator `/tmp/pseudoconductance-spice-env/bin/ngspice`.
Interpreter `/home/filip/miniconda3/envs/py312/bin/python`. Branch
`drn/20261117-end-to-end-attention`; worktree
`/home/filip/analog-research/worktrees/drn-20261117-end-to-end-attention`. Native
results: `/home/filip/analog-research/drn/results/end-to-end-attention/`.

## Pre-launch readiness corrections (dirty-tree smokes, scratchpad)

1. **Value diff-pair headroom.** With W = 1 µm PMOS, the tail fell into triode at
   V_cm = 1.05 V: about 43 pA reached the input instead of 100 pA, and both features
   read −0.56. Widening the diff-pair and tail PMOS to 20 µm fixed it.
2. **Clock edges.** With 10 ns switch edges, BSIM3 chain A stopped at the slot-1
   switch-off ("timestep too small" on an anchor probe). 100 ns edges pass.
3. **Convergence hints.** The ideal chain A operating point failed for some examples.
   I added `.nodeset` hints for the scorer, softmax and bank nodes, as in #57.
4. **Ideal gate.** The ideal law has κ = 2/3, so with κ = 0.7911 weights its value gain
   is about 15% low. The ideal model is gated on SPICE vs its own law, not on retrieval.

## Attempts

**Attempt 1** (launch `4b4e82a`, pinned at `results/end-to-end-attention/source-4b4e82a`).
All four bundles are complete, valid and artifact-verified (`ebl runs inspect
--verify-artifacts --require-complete`). Cumulative SPICE was 63 s; storage 343 MB.
- `A_ideal/20261117T090930.090189Z-ff4d4c39-cb3c18fc`
- `A_bsim3/20261117T090930.094071Z-e3af9313-9792cd9c`
- `B_ideal/20261117T090930.090221Z-03c16244-96f77df4`
- `B_bsim3/20261117T090930.093335Z-ea81a3c7-f0c0b113`

## Results

| Check | A, ideal | A, BSIM3 | B, ideal | B, BSIM3 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Retrieval MSE, 36 fresh steps | 1.4e-3 | **3.8e-4** | 1.3e-3 | **1.9e-4** | BSIM3 ≤ 1e-3 |
| Context average / retrieval | 20× | **74×** | 22× | **151×** | BSIM3 ≥ 5× |
| SPICE vs own law, RMS (decoded) | **1e-4** | 0.014 | **<1e-4** | 0.010 | ideal ≤ 2e-3 |
| Law MSE (same weights and κ) | 1.4e-3 | 5.8e-5 | 1.3e-3 | 2.9e-5 | reported |
| Worst absolute error | 0.095 | 0.079 | 0.080 | 0.055 | reported |
| Differential readout MSE | 1.4e-3 | 3.8e-4 | 1.3e-3 | 1.9e-4 | reported |
| Softmax on target, SPICE / law | 0.989 / 0.989 | 0.978 / 0.992 | 0.985 / 0.985 | 0.973 / 0.990 | reported |
| Summed score vs law, RMS / I_b | 0.000 | 0.260 | 0.000 | 0.287 | reported |
| Integration window T (C = 1 pF) | 196 µs | 166 µs | 184 µs | 155 µs | reported |
| Late drift (last 10% of hold) | 9e-7 | 7e-5 | — | — | A < 1e-3 |
| Min anchor V_DS | 1.06 V | 0.73 V | — | — | A > 4U_T |
| KV pedestal, keys (mean / max) | 0 | 0.36 / 1.22 mV | 0 | 0.34 / 1.11 mV | reported |
| KV pedestal, values (mean / max) | 0 | 0.29 / 0.37 mV | 0 | 0.30 / 0.39 mV | reported |
| Drift after switch-off, 1 s and 3 s holds | — | keys ≤ 0.93 mV, values ≤ 8 µV; output change 1.5e-5 | — | keys ≤ 0.74 mV, values ≤ 0.11 mV; output change 5.9e-4 | reported |
| Column vs crossbar formula | < 1e-10 V | ≤ 14 µV | < 1e-10 V | ≤ 12 µV | reported |
| Supply power (end of step) | 26 nW | 25 nW | 21 nW | 20 nW | reported |
| MOS count (with switches) | 112 | 148 | 63 | 99 | reported |

**Interpretation.**
1. **All gates pass.** One transistor-level circuit takes a token sequence through a
   resistor crossbar into a sampled analog KV cache, scores a fresh query, normalizes,
   weights the values and produces the attention output. Both chains retrieve, and
   chain B, without a bank, is the more accurate and the smaller (99 against 148 MOS).
2. **The ideal model is exact in wiring:** SPICE equals its own law to 1e-4. Its
   retrieval MSE of 1.3–1.4e-3 is the expected cost of running κ = 0.7911 weights
   under κ = 2/3 and is not a gate.
3. **The BSIM3 gap to the law** (3.8e-4 vs 5.8e-5 for A; 1.9e-4 vs 2.9e-5 for B)
   comes mainly from the anti-bump cell. As in #57, its summed current departs from the
   drawn law by about 0.26 I_b RMS, which lowers the softmax weight on the target from
   0.99 to 0.97–0.98.
4. **Capacitor integration replaces the 150 MΩ resistor** without any loss:
   1 pF, about 160 µs, three switches per key.
5. **KV cache in generic BSIM3:**
   - Charge injection leaves a 0.3–1.2 mV pedestal on the 5 pF hold nodes.
   - A further drift of at most 0.93 mV follows switch-off and has stopped by 1 s:
     the 1 s and 3 s holds are identical.
   - Value pedestals of about 0.3 mV are within the 1 mV tolerance found for the diff
     pair in stage 2.
   - The generic cards have no junction-leakage area, so real retention is **not**
     modelled.
6. **Not included:** the readout affine is the trained next-stage stand-in, applied
   in software. Mismatch, noise, the output projection to the next layer, and on-chip
   training are not included.
