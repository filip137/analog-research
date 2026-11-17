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

**Attempt 1** (launch `fd865dd`, pinned at `results/attention-energy/source-fd865dd`).
All 16 bundles are complete, valid and artifact-verified (`ebl runs inspect
--verify-artifacts --require-complete`). Cumulative SPICE was 288 s. Bundles are under
`results/attention-energy/<chain>_<rung>_<model>/`.

## Results

Energy per step, mean of 36 steps. Every BSIM3 rung passes every gate; the ideal L0/L5
wiring checks pass with SPICE matching its law to ≤ 1e-4.

| Chain A | L0 | L1 | L2 | L3 | L4 | L5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Energy per step** | **56.5 µJ** | 11.8 nJ | 6.79 nJ | 4.94 nJ | 2.58 nJ | **1.19 nJ** |
| Crossbar drivers (rail) / delivered | 56.5 µJ / 35.4 µJ | 4.72 / 2.89 nJ | 4.72 / 2.89 nJ | 4.72 / 2.89 nJ | 2.36 / 1.44 nJ | 0.90 / 0.37 nJ |
| Core | 7.06 nJ | 7.05 nJ | 2.06 nJ | 0.21 nJ | 0.21 nJ | 0.28 nJ |
| Clocks | 2.0 pJ | 2.3 pJ | 2.9 pJ | 2.9 pJ | 3.0 pJ | 2.9 pJ |
| Step time / average power | 0.30 s / 188 µW | 0.30 s / 39 nW | 0.30 s / 23 nW | 30 ms / 163 nW | 30 ms / 85 nW | 40 ms / 29 nW |
| Retrieval MSE (×context) | 3.8e-4 (74×) | 4.6e-4 (61×) | 8.7e-5 (321×) | 1.0e-4 (269×) | 1.1e-4 (260×) | **2.5e-5 (1128×)** |
| Key / value sampling error | 1.2 / 0.37 mV | 1.8 / 0.37 | 1.9 / 0.37 | 1.9 / 0.37 | 3.3 / 0.75 | 4.2 / 0.70 |
| Late drift (gate < 1e-3) | 7e-5 | 7e-5 | 7e-5 | 7.3e-4 | 7.4e-4 | 7.1e-4 |
| vs 0.3 s hold (reported) | — | — | — | 3.0e-3 | 3.0e-3 | 2.1e-3 |
| MOS | 148 | 152 | 158 | 158 | 160 | 160 |

| Chain B | L0 | L1 | L2 | L3 | L4 | L5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Energy per step** | **3.86 µJ** | 5.20 nJ | 4.88 nJ | 4.86 nJ | 2.44 nJ | **0.94 nJ** |
| Crossbar drivers (rail) / delivered | 3.85 µJ / 2.44 µJ | 4.83 / 2.96 nJ | 4.83 / 2.96 nJ | 4.83 / 2.96 nJ | 2.41 / 1.48 nJ | 0.91 / 0.37 nJ |
| Core | 0.37 nJ | 0.37 nJ | 46 pJ | 26 pJ | 26 pJ | 26 pJ |
| Step time / average power | 20 ms / 191 µW | 20 ms / 258 nW | 20 ms / 239 nW | 10 ms / 468 nW | 10 ms / 236 nW | 10 ms / 90 nW |
| Retrieval MSE (×context) | 1.8e-4 (151×) | 2.0e-4 (142×) | 8.6e-5 (327×) | 1.3e-4 (210×) | 1.8e-4 (151×) | 1.8e-4 (155×) |
| Key / value sampling error | 1.1 / 0.39 mV | 1.8 / 0.39 | 1.9 / 0.39 | 1.9 / 0.39 | 3.0 / 0.62 | 4.1 / 0.57 |
| vs 0.1 s hold (reported) | 1.2e-3 | 1.2e-3 | 2.0e-3 | 5.4e-3 | 8.8e-3 | 9.6e-3 |
| MOS | 99 | 103 | 109 | 109 | 111 | 111 |

**Final core breakdown.**
- **A (L5):** bank 94 pJ, value conversion 101 pJ, scorer 79 pJ (mostly the 1 nA bias
  reference held for 40 ms), softmax 8 pJ.
- **B (L5):** scorer 24 pJ, softmax 2 pJ.
- **Not counted:** chain B's diff-pair drain current is supplied by the output
  compliance sources, which stand for the next stage's input; about 1.4 pJ.

**Interpretation.**
1. **Energy falls 47,700× (A) and 4,100× (B), and accuracy is no worse.** Chain A's MSE
   improves 15×, because holding the query too makes its switch pedestal common to the
   matching key.
2. **The baseline was dominated by driving the crossbar for the whole hold.** Duty
   cycling (L1) alone removes about 99.98% of the energy.
3. **After L5 the crossbar drivers are 76% (A) and 97% (B) of the step.**
   - What sets them: 4 windows × 2 µs at s = 0.5 V/unit, with µS conductances.
   - Only 41% of the rail energy reaches the crossbar; the rest is driver headroom.
4. **The windows are now set by the transmission gates, which are weak near mid-rail
   at 1.8 V,** not by the crossbar's RC. Bootstrapped switches are the next lever, along
   with drivers fed from rails near V_cm ± s, or higher-resistance devices (outside the
   CMO range).
5. **Chain B, the smaller circuit, wins on energy per step;** chain A wins on average
   power at its longer, slower step.
6. **Per generated token with a KV cache (derived, not simulated),** only the new token's
   window is written each step. The query comes from the same token. So crossbar energy
   per token is about one window, roughly a quarter of the 4-window step measured here.
7. **Not modelled:** history dependence of read-before-settle (reported long-hold
   differences up to 1e-2 for B), mismatch, leakage, noise, and bias-generator and DAC
   overheads.
