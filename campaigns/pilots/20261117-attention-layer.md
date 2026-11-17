# Analog attention layer end-to-end: attention, simple layer norm, DRN MLP

2026-11-17. Requested by the user, tracked in on-chip-learning
[#61](https://github.com/filip137/on-chip-learning/issues/61). It builds on chain B of
[20261117-end-to-end-attention](20261117-end-to-end-attention.md) (#58), with the
energy options of [20261117-attention-energy](20261117-attention-energy.md) (#59) and
the sizes of [20261117-attention-size](20261117-attention-size.md) (#60).

## Question and scope

Can one transient carry a token sequence through the analog attention head, a simple
analog layer norm and a DRN MLP to a class decision, with the accuracy the equations
predict?

**Task.** The token is x = [v1, v2, kx, ky]. Each value is 2-D, with random length 0.2–1
and random direction. The query matches one of 3 keys, and the target is which of 6
60° sectors the retrieved value's direction falls in.

**Circuit, in one transient per example:**
1. **Crossbar:** 4 token inputs, 2 key columns and 2 value columns, conductances
   9–88.2 µS, s = 0.5 V/unit, 2 µs duty-cycled windows.
2. **Holds:** transmission-gate holds (300 fF) for the 3 keys, the 3 values, the query
   and a V_cm reference.
3. **Scorer:** 1 µm anti-bump cells (S = 4), 0.5× scorer mirrors, tail gating with a
   200 µs warm-up, 300 fF integration.
4. **Softmax:** PMOS Elfadel–Wyatt.
5. **Value weighting:** NMOS diff pairs, two features, giving
   I_p,e − I_m,e = I_tail·o_e.
6. **Simple layer norm:** n = o / (‖o‖₁ + ε), ε = 0.05 (L1, no centring), in two parts:
   - **Rectifier, per feature:** PMOS copies place +I_p and −I_m on a node. An NMOS diode
     (source on a 0.25 V pedestal) carries D⁺. A PMOS diode from a 1.2 V low rail (bulk
     at V_DD) carries D⁻, which is level-shifted by two mirrors.
   - **Translinear normalizer:** 2 µm × 2 µm devices, with the four log diodes plus an ε
     diode (2.85 pA) driving equal transistors on a shared source with a 10 pA tail.
7. **Interface:** 30 GΩ transimpedance resistors, so I_T gives V_U = 0.3 V, then ideal
   unity buffers (differential VCVS) drive u = V_ref + V_U·[n, −n, +1, −1].
8. **DRN** (as `pseudo_drn.py`): 6 inputs → 8 hidden → 6 outputs, positive resistors in
   9–88.2 µS, perfect diodes as behavioural softplus clamps (1 mV corner), 9 µS shunts.
   The class is the argmax of the output voltages, taken in software.

**Weights.** on-chip-learning `checks/results/layer_B_norm_drn8_seed1.json`, sha256
0eb75130685cecbb86afb3cb2c05dcbb8b25408a815a82142feab798021513ad, from
`attention_layer.py`, arm analog/norm/drn-h8, seed 1. Equation test accuracy 99.5%,
trained on the fitted 1 µm cell law. The integration window uses g / 0.942, the
fitted cell amplitude.

## Cases and decision rule

**BSIM3:**
- 96 fresh examples, seed 20261117.
- **Gate:** accuracy ≥ 85% and within 5 points of the equation model on the same
  examples; DRN output within 3 mV of its law given the measured DRN inputs.
- **Reported:** per-block errors (attention o vs law, normalizer n vs its law at the
  measured o, interface, DRN), class agreement, energy, and DRN power.

**Ideal law (wiring check):**
- 36 examples, with ideal-law parameters: S = 4, κ = 2/3, a temperature of 3 (so the
  scores stay in range), and the rectifier PMOS bulk on the rail, since the ideal law has
  no junctions.
- **Gate:** attention o RMS ≤ 0.02, normalizer RMS ≤ 0.03, DRN ≤ 3 mV.

## Allowance

Local CPU, ≤ 150 CPU-min of SPICE (#61). Interpreter
`/home/filip/miniconda3/envs/py312/bin/python`. Branch `drn/20261117-attention-layer`;
native results `/home/filip/analog-research/drn/results/attention-layer/`.

## Pre-launch readiness corrections (scratch)

1. **Normalizer pedestal.** With the log diodes at ground, the normalizer's shared source
   would have to sit below ground. Moving the diodes onto a 0.25 V pedestal keeps it at
   0.21–0.31 V.
2. **Negative half of the rectifier.**
   - The D⁻ copy from the low rail could not feed a pedestaled diode, so it now goes
     through two level-shift mirrors.
   - With the PMOS bulk tied to a 0.6 V rail, the source–bulk junction forward-biased
     whenever the node rose above the rail. That PMOS then stole about 95% of D⁺.
   - Fix: bulk at V_DD and the rail raised to 1.2 V. In an isolated BSIM3 test the
     normalizer then matches o/(‖o‖₁ + ε) to within about 0.02, including for 1 pA inputs.
   - A 1.0 V rail left a dead band that 10 ms could not cross for |D| of a few pA.
3. **Score range in the ideal model.** The ideal S = 4 cell has no short-channel
   sharpening (pedestal 0.5 vs 0.07 for BSIM3), so with g = 20.7 its scores saturate the
   softmax. Hence the ideal wiring check runs at its own temperature against its own law.
4. **Interface.** A behavioural source with a 3·10¹⁰ V/A current gain aborted one example
   in 96. It was replaced by 30 GΩ resistors and unity buffers.

## Attempts

## Results
