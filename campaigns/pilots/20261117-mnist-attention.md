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

## Stage 2: SPICE inference (to be specified after selection)

## Attempts

## Results
