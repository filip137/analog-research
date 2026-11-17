# Size ladder of the end-to-end analog attention step

2026-11-17. Assigned by the user ("minimize the size of the transistors and
capacitors"), tracked in on-chip-learning
[#60](https://github.com/filip137/on-chip-learning/issues/60). The starting point is
the final energy design (L5) of [20261117-attention-energy](20261117-attention-energy.md) (#59).

## Question and scope

How small can the transistors and capacitors of one N = 3 attention step be while the
#59 accuracy gates still pass and the circuit survives device mismatch?

**Inventory at L5:**
- **MOS gate area:** 29,690 µm² (A) and 24,330 µm² (B). 65% is the anti-bump middle
  legs (S = 16 × 10 µm, 10 µm long).
- **Capacitance:** 15 pF.

**Size metric:** total MOS gate area plus total explicit capacitance, also reported as
area at 2 fF/µm². Energy per step is reported alongside.

**Mismatch model** (the generic BSIM3 cards carry none):
- **Pelgrom offsets:** σ(V_T) = A_VT/√(WL) and σ(β)/β = A_β/√(WL), with A_VT = 7 mV·µm
  and A_β = 1 %·µm. These are declared assumptions typical of a 10 nm oxide, not PDK
  data. They are applied as BSIM3 `delvto`/`mulu0` on every MOS.
- **kT/C:** an offset of σ = √(kT/C) is added after each hold and score capacitor is
  released. It is applied as a series source to the gates, which is conservative,
  because it ignores the gate capacitance.

**Readout calibration.** Each Monte Carlo instance gets a two-parameter readout
calibration (gain, offset) fitted on 12 separate examples. This stands for the trained
next-stage crossbar.

## Scratch findings that set the ladder (dirty-tree, about 60 CPU-min)

1. **Block sensitivity** (one family mismatched at a time, 8 instances × 12 examples, L5):
   - The scorer cells, mid legs, mirrors, softmax, switches and kT/C at 1 pF are
     insensitive.
   - **Chain A's bank fails mismatch at L5 sizes:** 12% raw yield, 62% calibrated.
   - Within the bank, read-mirror errors are absorbed by the calibration (100%). The
     value input mirrors are not (62%): their ~3% per-slot error is large against values
     of about ±0.3 around the bank's 1 + v encoding.
   - **Chain B is limited by its value-weighting diff pairs** (translinear) and the
     softmax: 62% raw yield, 100% calibrated.
2. **The cell law vs geometry.** I made SPICE fits of a single cell's law, at W = L from
   10 down to 0.5 µm and S from 16 down to 2. Short-channel effects raise S_eff as the
   cell shrinks. 1 µm cells with S = 4 give S_eff ≈ 55, κ 0.78, amplitude 0.94, and a
   1.8% residual. The weights retrained on that law (`end_to_end.py --scale 0.5
   --cell-s 54.6 --cell-kappa 0.78`) lose no SPICE accuracy. At 0.5 µm the law deviates
   by 5%.
3. **Capacitors.** 300 fF holds and score capacitors keep nominal accuracy. 100 fF
   degrades it: B 4.8e-4, A 9.7e-4.
4. **Blocks that cannot shrink:**
   - **Value-weighting diff pairs (B):** at 0.5× in W and L the calibrated yield falls
     to 38–75%.
   - **Switches:** narrower switches raise the injection mismatch, and yield falls.
   - **Scorer PMOS mirrors:** 0.3× costs chain A nominal accuracy (2.8–4.1e-4).
   - **A larger value swing** (readout gain fixed at 0.5) improves A's yield but raises
     the tanh nonlinearity floor to 2.4e-4. Not used.
5. **Chain A's value input mirrors at 2× W and L** reach about 92% calibrated yield.
6. **The key-sampling gate was redefined.** At 300 fF the key switch pedestal is about
   10 mV. That pedestal is deterministic and shared with the held query, so it is not
   the history dependence the gate is meant to catch. Keys are now gated on charging
   completeness: node vs column just before the switch opens, which is 0.4 µV at
   300 fF / 2 µs. Values keep the #59 gate (vs formula, relative to the sampled
   reference).

## Ladder (cumulative, generic BSIM3, the 36 #58 examples)

| Rung | Change | Weights |
|---|---|---|
| Z0 | L5 as is | `e2e_{A,B}_tanhxs_fit_s0.5_n3_seed0.json` (#59) |
| Z1 | 300 fF holds and score capacitors (integration window scales with C) | same |
| Z2 | anti-bump cells at W = L = 1 µm with S = 4 (tails and their reference included) | `e2e_{A,B}_tanhxs_cell_f0.1_S4_n3_seed0.json` (sha256 e8a9b830… A, 76a15ef4… B) |
| Z3 | A: value input mirrors 2× W and L. B: scorer PMOS mirrors 0.5× | same as Z2 |

## Decision rule (per rung)

**Gates:**
- **Nominal circuit:** retrieval MSE ≤ 1e-3 and ≥ 5× below the context average.
- **Key settling:** ≤ 5 mV.
- **Value sampling:** ≤ 1 mV.
- **Chain A also:** late drift < 1e-3 and anchor V_DS > 4U_T.
- **Monte Carlo:** 20 instances, each run on the 36 test examples plus the 12
  calibration examples. Yield, the share of instances whose calibrated MSE passes the
  retrieval gate, must be ≥ 90%.

**Reported:** raw and differential-readout yield, median and worst MSE, the 2× A_VT case
(10 instances), area, capacitance and energy.

## Allowance

Local CPU, ≤ 300 CPU-min SPICE in total (#60), including the scratch work above.
≤ 2 attempts per configuration. Interpreter
`/home/filip/miniconda3/envs/py312/bin/python`, simulator
`/tmp/pseudoconductance-spice-env/bin/ngspice`. Branch `drn/20261117-attention-size`.
Native results: `/home/filip/analog-research/drn/results/attention-size/`. Monte Carlo
traces are deleted after summarizing; per-instance results stay in `summary.json`.

## Attempts

**Attempt 1** (launch `922c959`, pinned at `results/attention-size/source-922c959`).
All 8 bundles are complete, valid and artifact-verified. Recorded SPICE was 182 CPU-min;
with about 60 CPU-min of scratch work, the total is about 240 of the 300 allowed.

## Results

| | Gate area | Capacitance | Area (2 fF/µm²) | Nominal MSE | Yield, calibrated (gate ≥ 0.9) | Yield, raw | Calibrated MSE median / worst | Yield at 2× A_VT | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B Z0** | 24,330 µm² | 15.0 pF | 31,830 | 1.8e-4 | **1.00** | 0.60 | 3.2e-4 / 8.3e-4 | 0.50 | 0.94 nJ |
| **B Z1** | 24,330 | 4.5 pF | 26,580 | 1.9e-4 | **1.00** | 0.60 | 3.4e-4 / 8.0e-4 | 0.50 | 0.94 nJ |
| **B Z2** | 3,297 | 4.5 pF | 5,547 | 2.0e-4 | **0.95** | 0.55 | 3.7e-4 / 1.0e-3 | 0.30 | 0.95 nJ |
| **B Z3** | **2,847** | 4.5 pF | **5,097** | 1.9e-4 | **0.90** | 0.50 | 4.5e-4 / 1.3e-3 | 0.30 | 0.95 nJ |
| A Z0 | 29,690 | 15.0 pF | 37,190 | 2.5e-5 | 0.45 ✗ | 0.05 | 1.1e-3 / 4.7e-3 | 0.00 | 1.19 nJ |
| A Z1 | 29,690 | 4.5 pF | 31,940 | 2.4e-4 | 0.40 ✗ (value sampling 1.6 mV ✗) | 0.05 | 1.1e-3 / 4.7e-3 | 0.00 | 1.18 nJ |
| A Z2 | 8,657 | 4.5 pF | 10,907 | 2.4e-4 | 0.30 ✗ (value sampling ✗) | 0.05 | 1.4e-3 / 4.7e-3 | 0.00 | 1.19 nJ |
| A Z3 | 13,457 | 4.5 pF | 15,707 | 3.2e-4 | 0.75 ✗ (value sampling ✗) | 0.05 | 7.2e-4 / 1.8e-3 | 0.00 | 1.20 nJ |

Chain B passes every gate at every rung. Chain A fails the mismatch-yield gate at every
rung, including the unmodified #59 design, and fails value sampling at 300 fF.

**Area breakdown, B Z3 (µm²):** value-weighting diff pairs 2,100, switches 480,
scorer mirrors 150, softmax 50, anti-bump cells 67 (19 + 48), plus 4.5 pF of
capacitance (≈2,250 µm²).

**Interpretation.**
1. **Chain B shrinks 6.2× in total area** (8.5× in gate area, 3.3× in capacitance),
   holding 90% calibrated yield and the nominal accuracy. The scorer goes from 21,100 to
   67 µm². The SPICE-fitted law of the 1 µm, S = 4 cell plus retrained weights makes
   this free.
2. **What is left is precision-limited, not law-limited:**
   - The value diff pairs (74% of B's gate area) compare values of about ±20 mV
     against offsets of about 1 mV. Shrinking them fails yield.
   - The capacitors (now about 45% of total area) are bounded by value sampling
     accuracy.
   - The switches are bounded by injection mismatch.
3. **Chain A is not mismatch-robust at any size**, including the #58/#59 design: 45%
   yield. Its value input mirrors add a per-slot gain error to the bank's 1 + v
   encoding, which the two-parameter calibration cannot remove. 2× W and L on those
   mirrors reaches 75%, not 90%.
4. **The per-chip readout calibration is essential.** Raw yields are 50–60% (B) and 5%
   (A). The 2× A_VT case passes only 30–50% (B), so the margin depends strongly on the
   assumed Pelgrom coefficient.
5. **Not modelled:** PDK mismatch data, layout parasitics (they add capacitance to the
   small nodes), flicker and thermal noise beyond kT/C, and temperature.
