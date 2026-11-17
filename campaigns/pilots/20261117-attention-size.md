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

## Results
