# On-chip learning of the bank attention chain (SPICE in the loop)

2026-11-17. Assigned by the user (continue chain A toward on-chip learning; mismatch
deferred), tracked in on-chip-learning [#62](https://github.com/filip137/on-chip-learning/issues/62).
Circuit: chain A, L5 of [20261117-attention-energy](20261117-attention-energy.md).

## Question and scope

Can chain A be trained through its own circuit? The gradient signals are measured in
the BSIM3 circuit, and only the update arithmetic is digital. Does training reach the
#58 retrieval gate, both from random conductances and when fine-tuning an offline
design?

**Signals and rule.** The equation version is on-chip-learning
`checks/onchip_learning.py`, which matches exact backprop to cosine 1.00000 at β = 0.01
and trains chain A from random conductances to MSE about 5e-6.
- **Bank (EP).** One transient per training example: write, query, integrate, a 150 ms
  free settle, then a loss nudge at the output node for 50 ms at +β and 50 ms at −β,
  β = 0.1. The nudge is a behavioural sink, I_REF·β(t)·(x_t − x_o), with x_o read from
  the output read mirror. The read mirrors give the node coordinates x_j and x_o at the
  end of each phase:
  - dL/da_j = α²/(2β)·[½(x_j − x_o)²]₊₋;
  - dL/dz_j = −α²/(2β)·(x_j⁺ − x_j⁻).
- **Local backward rule:**
  - **softmax:** measured currents p_j;
  - **scorer:** the summed score currents (temperature gradient) and the held q/k
    voltages, with the derivative of the fitted cell law (S = 75, κ = 0.79);
  - **value pairs:** measured value currents z_j, giving dz/dV = κ/2U_T·(1 − (z − 1)²);
  - **crossbar:** the voltage across each conductance divided by its column total.
- **Digital:** Adam on the bounded conductance parameters (9–88.2 µS), on the
  integration window (temperature) and on the readout. Training traces are not kept;
  per-step metrics and evaluations are.

**Circuit change, a summing-mirror bias.** A 1 nA bias current sits at each scorer
summing mirror, switched on with the cell tails. It is common to every key, so it
cancels in the softmax. Without it, the mirror node (about 1 pF) could not slew to its
operating point within the 200 µs warm-up when a key's current was about 0.1 nA, so
well-matched keys did not integrate at all (readiness note 1).

**Runs:**
- **(a) Fine-tune.** Starts from `e2e_A_tanhxs_s0.5_n3_seed0.json` (offline, drawn cell
  law S = 16; sha256 d806cd62…). Batch 12, 60 steps, lr 0.01, evaluations every 20
  steps.
- **(b) From scratch.** Random conductances (seed 0), batch 12, 300 steps, lr 0.03,
  evaluations every 50 steps.
- **Both:** fresh training examples each step (seed 777), and a fixed test set of 96
  examples (seed 20261117) evaluated in normal operation (no nudge, 40 ms hold).

## Decision rule

**Gate:** final SPICE test MSE ≤ 1e-3 and ≥ 5× below the context average.
**Reported:** the test-MSE trajectory, gradient alignment with exact equation
gradients, and failed simulations.

## Allowance

Local CPU, ≤ 240 CPU-min of SPICE (#62), including readiness. Interpreter
`/home/filip/miniconda3/envs/py312/bin/python`. Branch `drn/20261117-onchip-attention`;
native results `/home/filip/analog-research/drn/results/onchip-attention/`.

## Pre-launch readiness (scratch)

1. **The summing mirror did not integrate small currents.**
   - Probing showed the PMOS diode at about 0.3 pA while the cells drew 95 pA from the
     node's capacitance: the node starts near 1.5 V after gating and slews at 0.1 V/ms.
   - Small-current keys therefore stayed at V_low. In the #59 design this hid behind
     the matching key, which also had the smallest current.
   - A 1 nA gated bias fixes it: score rises match T·ΔI/C exactly at 0.1 nA, and are
     about 5% compressed above 1 V.
2. **Nudge size and timing.**
   - With β = 0.01 and a 40 ms free hold, bank drift (about 6e-4 per 40 ms) swamped the
     nudge response (about 1e-4).
   - With β = 0.1, a 150 ms free settle and 50 ms phases, the ±β responses bracket the
     free state, with about 12% second-order asymmetry.
3. **Gradient alignment** with exact equation gradients on the same 24-example batch:
   - overall cosine 0.997 at random initialization;
   - per group 0.97–1.00 at the fine-tune start.
   - The temperature gradient at random initialization has the opposite sign to the
     model law. All keys then have nearly equal currents, so the measured ordering of
     few-mV differences decides it; the rule uses the measured currents.
4. **Short trainings** (36 test examples):
   - fine-tune, 20 steps: 2.7e-4 → 1.6e-4 (lr 0.003), 1.3e-4 (lr 0.01);
   - from scratch, 60 steps: 8.4e-2 → 2.7e-2.

## Attempts

## Results
