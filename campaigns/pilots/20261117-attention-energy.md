# Energy ladder of the end-to-end analog attention step

2026-11-17. Assigned by the user ("optimize the power consumption"), tracked in
on-chip-learning [#59](https://github.com/filip137/on-chip-learning/issues/59). Circuit
and gates from [20261117-end-to-end-attention](20261117-end-to-end-attention.md) (#58).

## Question and scope

How far can the energy of one N = 3 attention step be reduced without failing the #58
accuracy gates, and where does the remaining energy go?

**Metric.** Energy per step: block supplies, token drivers and clock drivers, integrated
over the step. Average power at that step period is also reported.
- **Token drivers:** charge sourced from a 1.8 V rail, a class-B bound. The energy
  delivered into the crossbar, a lower bound, is reported alongside.
- **Clock drivers:** charge drawn from VDD while driving gates high, i.e. CV² per
  rising edge.
- **Assumed:** everything is power-gated between steps.
- **Outside the count:** bias generators, DACs, the readout and the next layer.

**Cumulative rungs** (both chains, generic BSIM3, the same 36 fresh examples as #58,
seed 20261117):

| Rung | Change | Settings |
|---|---|---|
| L0 | #58 design, with power accounting | s = 0.75 V/unit, 20 µs windows, 5 pF holds; hold 0.3 s (A) / 0.02 s (B) |
| L1 | crossbar duty cycle: inputs return to V_cm after the query window; the query is sampled onto its own holds | windows 6 µs |
| L2 | scorer cell tails switched on only for warm-up + integration | 200 µs warm-up (t_reset = t_warm = 200 µs) |
| L3 | settling: bank's explicit 1 pF node capacitors removed (A); shorter hold | 30 ms (A) / 10 ms (B) |
| L4 | 1 pF holds; the diff pairs' V_cm reference sampled through an identical gate (matched pedestal) | windows 3 µs |
| L5 | swing 0.5 V/unit, weights retrained on the fitted BSIM3 cell law (S_eff = 75, #57) | windows 2 µs; hold 40 ms (A) / 10 ms (B) |

**Weights** (on-chip-learning `orthogonal-ideas/energy-transformer/checks/results/`):
- **L0–L4:** the #58 designs `e2e_{A,B}_tanhxs_n3_seed0.json`.
- **L5:** `e2e_{A,B}_tanhxs_fit_s0.5_n3_seed0.json`, from `end_to_end.py --scale 0.5
  --cell-s 75`, arms A-tanh/xs and B-tanh/xs:
  - sha256 ae284879d975… (A) and 04631527af03… (B), full hashes in the configs;
  - equations at N = 3: 1.1e-5 (A) and 1.1e-5 (B).

**Ideal-model wiring checks:** L0 and L5, SPICE vs its own law ≤ 2e-3.

## Decision rule (gates per BSIM3 rung)

- **Retrieval:** MSE ≤ 1e-3, and ≥ 5× below the context-average MSE.
- **Key sampling:** held key vs crossbar formula ≤ 5 mV.
- **Value sampling:** held value vs formula, relative to the sampled reference when
  one exists, ≤ 1 mV. Complete sampling keeps the cached state independent of what the
  capacitor held before; these single-step runs always start from V_cm.
- **Chain A also keeps the #58 gates:** late drift < 1e-3, rails, anchor V_DS > 4U_T.
- **Reported, not gated:**
  - settling relative to a long hold (0.3 s for A, 0.1 s for B), on examples 0–2;
  - query sampling error, which includes the common-mode gate-charge shift when the
    cells switch on and is shared with the matching key;
  - pedestals.

## Allowance

Local CPU, ≤ 180 CPU-min cumulative SPICE (#59), ≤ 2 recorded attempts per
configuration, ≤ 4 GB. Interpreter `/home/filip/miniconda3/envs/py312/bin/python`,
simulator `/tmp/pseudoconductance-spice-env/bin/ngspice`. Branch
`drn/20261117-attention-energy`; native results
`/home/filip/analog-research/drn/results/attention-energy/`.

## Pre-launch readiness (dirty-tree scratch ladders, about 25 CPU-min)

The rung settings above were chosen from scratch ladders.

1. **Scorer gating by reference current failed.** Switching the 1 nA reference off left
   the 2 pF tail-gate line taking ~1 ms to recharge. I gated the cell tails with
   series switches instead, keeping the bias. The cells then need a ~200 µs warm-up:
   20 µs fails, 100 µs gives MSE 4.9e-4, 200 µs gives 9e-5.
2. **Uniform device scaling (0.3×, 0.1×) speeds up settling but changes the BSIM3 cell
   law,** so retrieval fails. Not used.
3. **Write windows.** 0.5–1 µs windows pass retrieval but leave low-voltage keys up to
   80 mV short of their target. Transmission gates are weak near mid-rail at 1.8 V.
   Hence the sampling gates and the 6 / 3 / 2 µs windows. Wider (40 µm) switches inject
   7.5 mV on values and degrade MSE to 1.2e-3; not used.
4. **The swing could not be reduced with the drawn-law weights:** s = 0.5 gives MSE
   2.2e-3 (A) and 6.3e-3 (B). Retraining on the fitted cell law (S_eff = 75) makes
   s = 0.5 pass; s = 0.375 still fails, at 3.3e-3.
5. **A settling gate against a long hold was dropped before launch.** At 1e-3 it fails
   even for the #58 chain B baseline (1.2e-3 at 20 ms vs 0.1 s). It is now reported. Chain
   A keeps its late-drift gate, which sets its L5 hold at 40 ms (late drift 1.03e-3 at
   30 ms).
6. **Transient tolerance 3e-15 A.** At 1e-15, one chain A L5 example aborted at a switch
   edge. At 3e-15 and 1e-14, outputs agree with 1e-15 to 1e-6 on three examples.

## Attempts

## Results
