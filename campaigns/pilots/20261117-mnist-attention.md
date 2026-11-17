# Chain A (bank attention) on MNIST: GPU training and SPICE inference

2026-11-17. Requested by the user, tracked in on-chip-learning
[#63](https://github.com/filip137/analog-research/issues/63): scale the EP-capable bank
attention chain (#58–#62) to MNIST. Train offline on the local GPU, then run a few
inference samples through the transistor-level circuit.

## Model (`networks/energy_transformer/mnist_chain_a.py`)

- **Tokens:** 16 patches of 7×7 pixels (±1) plus a 16-way position one-hot, so d_in = 65.
- **Crossbar:** passive (9–88.2 µS, s = 0.5 V/unit), with d_k key and d_v value columns.
- **Heads:** H heads, each with:
  - a programmed query voltage per key dimension (V_cm ± 0.3 V);
  - a summed anti-bump scorer (fitted cell law S_eff = 75, κ = 0.79);
  - a softmax (A = 0.57);
  - a pseudoconductance bank (h0 = 0.01) with diff-pair value anchors z = 1 + tanh(κ ΔV / 2U_T).
- **Readout:** digital linear, on the H·d_v bank outputs.
- **Training:** exact backprop through the closed-form bank, Adam, float64, local
  RTX 3090. Train/validation/test as `pseudo_mnist_data.py` (55,000 / 5,000 / 10,000).

## Stage 1: GPU sweep and selection (declared in #63 before the runs)

- **Grid:** H ∈ {2, 4, 8} × d_k ∈ {2, 4} × d_v ∈ {8, 16}; 20 epochs, seed 0, batch 256,
  lr 0.01. Configs: `examples/analog_attention_mnist_sweep_h*_k*_v*.json`.
- **Rule:** the most accurate configuration (validation) with ≤ 13,000 MOS in the SPICE
  build. It is then retrained for 60 epochs with cosine decay, and its test accuracy is
  reported.
- **Allowance:** ≤ 60 GPU-min. Native results:
  `/home/filip/analog-research/drn/results/mnist-attention/`.

Scratch probes before the declaration: H4/dk4/dv16 reached 91.9% validation at 20 epochs,
H8/dk4/dv16 93.9%.

## Stage 1 result (recorded, see Attempts)

Selected: H = 4, d_k = 4, d_v = 16 (12,118 MOS), 91.9% validation at 20 epochs. After the
60-epoch cosine retrain: **test 93.19%**, validation 92.56%.

## Stage 2: SPICE inference

**Netlist** (`networks/energy_transformer/mnist_spice.py`, 11,339 MOS, 2,600 crossbar
resistors, 321 hold capacitors):
- 16 tokens are written in sequence (2 µs windows) through the 65-input crossbar into
  1 pF transmission-gate holds, plus a V_cm reference hold.
- **Per head:** programmed query sources; 64 anti-bump cells (10 µm, S = 16) with tail
  gating and the #62 summing-mirror bias (1 nA, on a 10 pA standing floor); 1 pF
  integration; a PMOS softmax into the attention controllers; and a 16-feature
  pseudoconductance bank with PMOS diff-pair value inputs.
- Readout digital.

**Readiness findings (scratch, about 95 CPU-min):**
1. **Convergence.** The #58–#62 tolerance (reltol 1e-6) aborts at the first sampling edge
   once there are 16 keys. reltol 1e-4 and 1e-5 both converge and give identical outputs
   to all printed digits on two images. reltol 1e-4 is used, with abstol 3e-15 A.
   Without the standing 10 pA floor on the mirror bias, an unbiased summing diode leaves
   its node floating while the cells are gated off.
2. **Score headroom at this scale.** The trained temperatures (13–31) with 1 nA mirror
   bias made the score integrators rise by up to 1 V of common offset. In two heads every
   score sat at the rail and the softmax currents collapsed (attention error up to 0.70).
   - **Fix:** a programmed sink per integrator during integration, equal to the bias plus
     a per-head offset, so that only key differences integrate; the softmax is invariant
     to a common offset.
   - The offset is calibrated digitally from 1,000 training images: amp·I_b × the 10%
     quantile of the per-image minimum key score.
3. **Cell law at four summed dimensions.** Measured key-sum currents are 0.6–0.75 of the
   S = 75 law. A fit to 384 key sums from 6 training images (SPICE) gives S_eff = 303,
   κ = 0.95, amplitude 0.695 and a common offset of 0.215 I_b. Its RMS residual is
   0.028 I_b, against 0.061 for the S = 75 law at its best scale.
4. **Ten test images (0–9), two variants:**
   - **C1:** the original network with the integration window scaled by 1/0.66 →
     agreement 8/10, accuracy 9/10 (equations 9/10), attention error 0.07–0.18, output
     RMS 0.04–0.10.
   - **C2:** the network retrained on the fitted law (93.22% test), window / 0.695 →
     agreement 9/10, accuracy 10/10 (equations 9/10), attention error 0.18–0.33, output
     RMS 0.06–0.15.

**Recorded cases** (declared before launch):
- **C2, primary:** fitted-law network, from the native retraining run.
- **C1, reference:** original final network.
- **Both:** 20 fresh test images (10–29) that were not used in readiness, 10 parallel
  workers.

**Gate:** class agreement with the equations ≥ 90% on the simulated images, all of them
simulated. **Reported:** accuracy (SPICE, equations), per-head attention and output
errors, late drift, runtime.

**Allowance:** SPICE ≤ 240 CPU-min in total (about 95 used in readiness, about 120
planned).

## Attempts

## Results
