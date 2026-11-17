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

## Results
